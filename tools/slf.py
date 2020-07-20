"""
Storey-loss-function (SLF) Generator adopted from the work of Sebastiano Zampieri by Davit Shahnazaryan

The tool allows the automatic production of EDP-DV SLFs based on input fragility, consequence and quantity data.

Considerations for double counting should be done at the input level and the consequence function should mirror it.

SLF estimation procedure:       Ramirez and Miranda 2009, CH.3 Storey-based building-specific loss estimation (p. 17)
FEMA P-58 for fragilities:      https://femap58.atcouncil.org/reports
For consequence functions:      https://femap58.atcouncil.org/reports
Python tools for reference:     https://blog.resellerclub.com/the-6-best-python-gui-frameworks-for-developers/
                                https://www.youtube.com/watch?v=627VBkAhKTc

EDP:    Engineering Demand Parameter
DV:     Decision Variable
DS:     Damage State
"""
from pathlib import Path
import pandas as pd
import numpy as np
import math
import pickle
import json
from scipy import stats
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')


class SLF:
    def __init__(self, project_name, component_data, correlation_tree, edp_bin=None, correlation="Correlated",
                 regression="Weibull", n_realizations=20, conversion=1.0, do_grouping=True, sflag=True):
        """
        Initialization of the Master Generator
        TODO, add option for mutually exclusive damage states
        TODO, include possibility of including quantity uncertainties along with the mean values
        :param project_name: str                Name of the project to save in the database
        :param component_data: str              Component data file name, e.g. "*.csv"
        :param correlation_tree: str            Correlation tree file name, e.g. "*.csv"
        :param edp_bin: float                   EDP sampling unit, % for IDR, g for PFA (non-negative)
        :param correlation: str                 Whether the elements are "Independent" or "Correlated"
        :param regression: str                  Whether the regression function is based on "Weibull" or "Papadopoulos"
        :param n_realizations: int              Number of realizations
        :param conversion: float                Conversion factor from usd to euro, e.g. if provided in euro, use 1.0;
                                                if 1 usd = 0.88euro, use 0.88
                                                or use 1.0 if ratios are used directly
        :param do_grouping: bool                Whether to do performance grouping or use the entire component database
        :param sflag: bool                      Save data
        """
        if edp_bin is None:
            edp_bin = [0.1, 0.05]
        self.dir = Path.cwd()
        self.database_dir = self.dir / "Database"
        self.project_name = project_name
        self.component_data = component_data
        self.correlation_tree = correlation_tree
        self.edp_bin = edp_bin
        self.correlation = correlation
        self.regression = regression
        self.n_realizations = n_realizations
        self.conversion = conversion
        self.do_grouping = do_grouping
        self.sflag = sflag
        self.edp_range = None

    def get_edp_range(self, edp):
        """
        Identifies the EDP range based on edp type and input edp bin
        :param edp: str                         EDP type (IDR or PFA)
        :return: None
        """
        # EDP range
        if edp == "IDR" or edp == "PSD":
            # stop calculation at 10% drift
            edp_bin = self.edp_bin[0] / 100.
            self.edp_range = np.arange(0, 0.2 + edp_bin, edp_bin)
        elif edp == "PFA":
            # stop calculation at 2.5g of acceleration
            edp_bin = self.edp_bin[1]
            self.edp_range = np.arange(0, 10.0 + edp_bin, edp_bin)
        else:
            raise ValueError("[EXCEPTION] Wrong EDP type, must be 'IDR'/'PSD' or 'PFA'")

    def store_results(self, filepath, data, filetype):
        """
        Store results in the database depending on file type selected
        :param filepath: str                            Filepath, e.g. "directory/name"
        :param data:                                    Data to be stored
        :param filetype: str                            Filetype, e.g. npy, json, pkl, csv
        :return: None
        """
        if filetype == "npy":
            np.save(f"{filepath}.npy", data)
        elif filetype == "pkl" or filetype == "pickle":
            with open(f"{filepath}.pkl", 'wb') as handle:
                pickle.dump(data, handle)
        elif filetype == "json":
            with open(f"{filepath}.json", "w") as json_file:
                json.dump(data, json_file)
        elif filetype == "csv":
            data.to_csv(f"{filepath}.csv")

    def get_component_data(self):
        """
        Gets component information from the user provided .csv file
        TODO: for UI update, the user will have the following options to manipulate the data and create the .csv file through the UI
                    1. Add new component
                    2. View existing components (e.g., FEMA P-58 PACT or added previously by the user)
                    3. Edit an existing component
                    4. Delete an existing component
                    5. EDP and EDP bin should be visible in the tools, currently it is not an input argument
        Current version: Direct manipulation within the .csv file, add new entries with empty IDs (the tool with assign
        the IDs automatically) or select ID from the drop down list, which will select the already existing one.
        New created entries will not be saved within the database, and will be deleted if the .csv file is modified.

        Notes on creation of a correlation tree: Make sure that the MIN DS assigned does not exceed the possible Damage
        States defined for the element. Essentially, if a component is dependent on another component and has only one
        DS, i.e. DS1, which occurs only if the causation element is for example at its DS3, then the following would be
        true. And Item3 would be damaged earlier and has more DSs. The software will not detec any errors, so it depends
        on the user.
        Item ID: 	Dependant on item 	MIN DS|DS0 	MIN DS|DS1 	MIN DS|DS2 	MIN DS|DS3 	MIN DS|DS4 	MIN DS|DS5
        Item 1: 	Independent 		Independent	Independent Independent Independent Independent Independent
		Item 2:		1					Undamaged 	Undamaged  	Undamaged 	DS1 		DS1 		DS1
		Item 3: 	1 					Undamaged 	Undamaged 	DS1 		DS2 		DS3 		DS3

        :return: DataFrame                  DataFrame containing the component data
        """
        component_data = self.component_data
        n_items = len(self.component_data)

        # Check for any missing values
        for key in component_data.keys():
            # 'best fit' features
            if key.endswith('best fit'):
                component_data[key].fillna('normal', inplace=True)
            if key == "ITEM":
                component_data[key].fillna(pd.Series(np.arange(1, n_items+1, 1), dtype='int'), inplace=True)
            if key == "ID":
                component_data[key].fillna("B", inplace=True)

        # Replace all nan with 0.0 for the rest of the DataFrame, except for Group and Component
        component_data[component_data.columns.difference(["Group", "Component"])] = \
            component_data[component_data.columns.difference(["Group", "Component"])].fillna(0, inplace=False)

        return component_data

    def group_components(self, component_data):
        """
        Component performance grouping
        :param component_data: DataFrame        DataFrame containing the component data
        :return: dict                           Dictionary containing DataFrames for each performance group
        """
        groups = np.array(component_data["Group"])
        components = np.array(component_data["Component"])

        # Perform grouping
        if self.do_grouping:
            # Check if grouping was assigned
            # Group not assigned
            if np.isnan(groups).any():
                # Populate with a placeholder
                component_data["Group"].fillna(-1, inplace=True)

                # Only EDP was assigned (Component and Group unassigned)
                if components.dtype != "O":
                    # Populate with a placeholder
                    component_data["Component"].fillna("-1", inplace=True)
                    # Select unique EDPs
                    unique_edps = component_data.EDP.unique()
                    component_groups = {}
                    for group in unique_edps:
                        component_groups[group] = component_data[(component_data["EDP"] == group)]

                # EDP and Component assigned
                else:
                    idr_s = component_data[(component_data["EDP"] == "IDR") & (component_data["Component"] == "S")]
                    idr_ns = component_data[(component_data["EDP"] == "IDR") & (component_data["Component"] == "NS")]
                    pfa_ns = component_data[(component_data["EDP"] == "PFA") & (component_data["Component"] == "NS")]

                    component_groups = {"IDR S": idr_s, "IDR NS": idr_ns, "PFA NS": pfa_ns}

            # Group is assigned
            else:
                if components.dtype != "O":
                    # Populate with a placeholder
                    component_data["Component"].fillna("-1", inplace=True)

                unique_groups = np.unique(groups)
                component_groups = {}
                for group in unique_groups:
                    component_groups[group] = component_data[(component_data["Group"] == group)]

        # No grouping
        else:
            if components.dtype != "O":
                # Populate with a placeholder
                component_data["Component"].fillna("-1", inplace=True)
            if np.isnan(groups).any():
                # Populate with a placeholder
                component_data["Group"].fillna(-1, inplace=True)

            # If no performance grouping is done, automatic group tag for the single group is assigned the EDP value
            key = component_data["EDP"].iloc[0]
            component_groups = {key: component_data}

        return component_groups

    def get_correlation_tree(self, component_data):
        """
        Gets the correlation tree and generates the correlation tree matrix
        :param component_data: DataFrame        DataFrame containing the component data
        :return: ndarray                        Correlation tree matrix
        """
        # Get possible max DS
        max_DS = []
        ds_limit = []
        for item in component_data.index:
            ds = component_data.loc[item]["Damage States"]
            max_DS.append(f"DS{ds+1}")
            ds_limit.append(ds)

        # Get correlation tree
        correlation_tree = self.correlation_tree
        # Select the items within the component performance group
        correlation_tree = correlation_tree.loc[component_data.index]

        # Check integrity of the provided input correlation table
        if len(correlation_tree.keys()) < max(ds_limit) + 3:
            raise ValueError("[EXCEPTION] Unexpected (fewer) number of features in the correlations DataFrame")

        idx = 0
        for item in component_data.index:
            for feature in correlation_tree.keys():
                ds = str(correlation_tree.loc[item][feature])
                if ds == max_DS[idx]:
                    raise ValueError("[EXCEPTION] MIN DS assigned in the correlation tree must not exceed the possible "
                                     "DS defined for the element")
            idx += 1

        # Check that dimensions of the correlation tree and the component data match
        if len(component_data) != len(correlation_tree):
            raise ValueError("[EXCEPTION] Number of items in the correlation tree and component data should match")

        # Create the correlation matrix
        items = correlation_tree.values[:, 0]
        c_tree = np.delete(correlation_tree.values, 0, 1)
        matrix = np.zeros((c_tree.shape[0], c_tree.shape[1]))
        for j in range(c_tree.shape[1]):
            for i in range(c_tree.shape[0]):
                if j == 0:
                    if c_tree[i][j] == "Independent" or c_tree[i][j] == "INDEPENDENT":
                        matrix[i][j] = items[i]
                    elif items[i] == "" or math.isnan(items[i]):
                        matrix[i][j] = np.nan
                    else:
                        matrix[i][j] = c_tree[i][j]
                else:
                    if math.isnan(matrix[i][j - 1]):
                        matrix[i][j] = np.nan
                    elif c_tree[i][j] == "Independent" or c_tree[i][j] == "INDEPENDENT":
                        matrix[i][j] = 0
                    elif c_tree[i][j] == "Undamaged" or c_tree[i][j] == "UNDAMAGED":
                        matrix[i][j] = 0
                    elif c_tree[i][j] == "DS1":
                        matrix[i][j] = 1
                    elif c_tree[i][j] == "DS2":
                        matrix[i][j] = 2
                    elif c_tree[i][j] == "DS3":
                        matrix[i][j] = 3
                    elif c_tree[i][j] == "DS4":
                        matrix[i][j] = 4
                    else:
                        matrix[i][j] = 5

        return matrix

    def derive_fragility_functions(self, component_data):
        """
        Derives fragility functions
        :param component_data: DataFrame        DataFrame containing the component data
        :return: dict                           Fragility functions of all components at all DSs
        """
        # Get all DS columns
        max_ds_selected = 0
        for column in component_data.columns:
            if column.endswith("Median Demand"):
                max_ds_selected += 1

        # Select only numeric features
        component_data = component_data.select_dtypes(exclude=["object"])

        # Fragility parameters
        means_fr = np.zeros((len(component_data), max_ds_selected))
        covs_fr = np.zeros((len(component_data), max_ds_selected))

        # Consequence parameters
        means_cost = np.zeros((len(component_data), max_ds_selected))
        covs_cost = np.zeros((len(component_data), max_ds_selected))

        # Deriving fragility functions
        # TODO, modify 4, make indentation smarter
        data = component_data.values[:, 4:]

        for item in range(len(data)):
            for ds in range(5):
                means_fr[item][ds] = data[item][ds]
                covs_fr[item][ds] = data[item][ds + 5]
                means_cost[item][ds] = data[item][ds + 10] * self.conversion
                covs_cost[item][ds] = data[item][ds + 15]

        # Deriving the ordinates of the fragility functions
        fragilities = {"EDP": self.edp_range, "ITEMs": {}}
        n_items = 0
        for item in range(len(data)):
            fragilities["ITEMs"][item + 1 + n_items] = {}
            for ds in range(5):
                mean = np.exp(np.log(means_fr[item][ds]) - 0.5 * np.log(covs_fr[item][ds] ** 2 + 1))
                std = np.log(covs_fr[item][ds] ** 2 + 1) ** 0.5
                if mean == 0 and std == 0:
                    fragilities["ITEMs"][item + 1 + n_items][f"DS{ds + 1}"] = np.zeros(len(self.edp_range))
                else:
                    frag = stats.norm.cdf(np.log(self.edp_range / mean) / std, loc=0, scale=1)
                    frag[np.isnan(frag)] = 0
                    fragilities["ITEMs"][item + 1 + n_items][f"DS{ds + 1}"] = frag

        return fragilities, means_cost, covs_cost

    def get_DS_probs(self, component_data, fragilities):
        """
        Evaluates probabilities of having each damage state for every EDP
        :param component_data: DataFrame        DataFrame containing the component data
        :param fragilities: dict                Fragility functions of all components at all DSs
        :return: dict                           Probabilities of being in a given DS
        """
        # Number of components in the building inventory
        num_items = len(component_data)

        if self.correlation == "Independent":
            # Evaluating probability of having each damage state for every EDP
            # Items
            damage_probs = {}
            for item in fragilities["ITEMs"]:
                damage_probs[item] = {}
                # DS
                for ds in range(6):
                    y = fragilities["ITEMs"][item]
                    if ds == 0:
                        damage_probs[item][f"DS{ds}"] = 1 - y[f"DS{ds + 1}"]
                    elif ds == 5:
                        damage_probs[item][f"DS{ds}"] = y[f"DS{ds}"]
                    else:
                        damage_probs[item][f"DS{ds}"] = y[f"DS{ds}"] - y[f"DS{ds + 1}"]
            return damage_probs

    def perform_Monte_Carlo(self, fragilities, corr_tree=None):
        """
        Performs Monte Carlo simulations and simulates DS for each EDP step
        :param fragilities: dict                Fragility functions of all components at all DSs
        :param corr_tree: ndarray               Correlation tree matrix
        :return: dict                           Damage states of each component for each simulation
        """
        num_items = len(fragilities["ITEMs"])
        ds_range = np.arange(0, 6, 1)
        num_edp = len(fragilities["EDP"])
        damage_state = {}
        if self.correlation == "Independent":
            # Evaluate the DS on the i-th component for EDPs at the n-th simulation
            # Items
            for item in fragilities["ITEMs"]:
                damage_state[item] = {}
                # Simulations
                for n in range(self.n_realizations):
                    # Generate random data between (0, 1)
                    random_array = np.random.rand(num_edp)
                    damage = np.zeros(num_edp)
                    # DS
                    for ds in range(5, 0, -1):
                        y1 = fragilities["ITEMs"][item][f"DS{ds}"]
                        if ds == 5:
                            damage = np.where(random_array <= y1, ds_range[ds], damage)
                        else:
                            y = fragilities["ITEMs"][item][f"DS{ds + 1}"]
                            damage = np.where((random_array >= y) & (random_array < y1), ds_range[ds], damage)
                    damage_state[item][n] = damage
            return damage_state

        elif self.correlation == "Correlated":
            if corr_tree is None:
                raise ValueError("[EXCEPTION] Correlation matrix is missing")

            idx = 0
            for item in fragilities["ITEMs"]:
                damage_state[item] = {}
                for n in range(self.n_realizations):
                    if corr_tree[idx][0] == item:
                        random_array = np.random.rand(num_edp)
                        damage = np.zeros(num_edp)
                        # DS
                        for ds in range(5, 0, -1):
                            y1 = fragilities["ITEMs"][item][f"DS{ds}"]
                            if ds == 5:
                                damage = np.where(random_array <= y1, ds_range[ds], damage)
                            else:
                                y = fragilities["ITEMs"][item][f"DS{ds + 1}"]
                                damage = np.where((random_array >= y) & (random_array < y1), ds_range[ds], damage)
                        damage_state[item][n] = damage
                    else:
                        # -1 to indicate no assignment to a final DS to sub correlated elements
                        damage_state[item][n] = np.zeros(num_edp) - 1
                idx += 1

            return damage_state

        else:
            raise ValueError("[EXCEPTION] Wrong type of correlation, must be 'Independent' or 'Correlated'")

    def test_correlated_data(self, damage_state, matrix, fragilities):
        """
        Tests if any non-assigned DS exist (i.e. -1) and makes correction if necessary
        :param damage_state: dict               Damage states of each component for each simulation
        :param matrix: ndarray                  Correlation tree matrix
        :param fragilities: dict                Fragility functions of all components at all DSs
        :return: dict                           Damage states of each component for each simulation
        """
        num_items = len(damage_state)
        ds_range = np.arange(0, 6, 1)
        iteration = 1
        test = 0
        for i in damage_state:
            for j in damage_state[i]:
                test += sum(damage_state[i][j] == -1)

        # Start the iterations
        min_ds = {}
        y_new = {}
        while test != 0:
            iteration += 1

            # Items
            for item in range(num_items):
                min_ds[item + 1] = {}
                y_new[item + 1] = {}

                # TODO, remove iteration on EDP, not elegant
                # EDP values
                for edp in range(len(self.edp_range)):
                    min_ds[item + 1][edp] = {}
                    y_new[item + 1][edp] = {}

                    # Simulations
                    for n in range(self.n_realizations):
                        y_new[item + 1][edp][n] = {}
                        # Find the DS on the causation element
                        if damage_state[item + 1][n][edp] == -1:
                            # if the causation element still has not been assigned a DS
                            if damage_state[int(matrix[item][0])][n][edp] == -1:
                                # The marker will make the engine simulate the DS at the successive iteration
                                damage_state[item + 1][n][edp] = -1
                            else:
                                # Finds the minimum DS for the i-th element
                                min_ds[item + 1][edp][n] = matrix[item][1 + int(damage_state[matrix[item][0]][n][edp])]
                                # Recalculates the probability of having each DS in the condition of having a min DS
                                # Damage states
                                for ds in range(5):
                                    # All DS smaller than min_DS have a probability of 1 of being observed
                                    if ds + 1 <= min_ds[item + 1][edp][n]:
                                        # probability of having DS >= min_k
                                        y_new[item + 1][edp][n][ds+1] = 1
                                        # if min_DS is zero then the probabilities are unchanged
                                    elif min_ds[item + 1][edp][n] == 0:
                                        y = fragilities["ITEMs"][item + 1][f"DS{ds + 1}"]
                                        y_new[item + 1][edp][n][ds+1] = y[edp]
                                    else:
                                        # conditional probability of having DS >= DS_k given min_DS
                                        y = fragilities["ITEMs"][item + 1][f"DS{ds + 1}"]
                                        y1 = fragilities["ITEMs"][item + 1][f"DS{int(min_ds[item + 1][edp][n]) + 1}"]
                                        y_new[item + 1][edp][n][ds+1] = y[edp] / y1[edp]
                                        if math.isnan(y_new[item + 1][edp][n][ds+1]):
                                            y_new[item + 1][edp][n][ds+1] = 0

                                # Simulates the DS at the given EDP, for the new set of probabilities
                                rand_value = np.random.rand(1)[0]
                                for ds in range(5, 0, -1):
                                    if ds == 5:
                                        if rand_value <= y_new[item+1][edp][n][ds-1]:
                                            damage = ds_range[ds]
                                        else:
                                            damage = 0
                                    else:
                                        if y_new[item+1][edp][n][ds+1] <= rand_value < y_new[item+1][edp][n][ds]:
                                            damage = ds_range[ds]
                                        else:
                                            damage = damage
                                    damage_state[item+1][n][edp] = damage

            test = 0
            for i in damage_state:
                for j in damage_state[i]:
                    test += sum(damage_state[i][j] == -1)

        print(f"[ITERATIONS] {iteration} iterations to reach solution")
        return damage_state

    def calculate_loss(self, component_data, damage_state, means_cost, covs_cost):
        """
        Evaluates the damage cost on the individual i-th component at each EDP level for each n-th simulation
        :param component_data: DataFrame                DataFrame containing the component data
        :param damage_state: dict                       Damage states of each component for each simulation
        :param means_cost: ndarray                      Means of repair costs
        :param covs_cost: ndarray                       Covariances of repair costs
        :return: dict                                   Repair costs
        """
        repair_cost = {}
        idx = 0
        for item in damage_state.keys():
            repair_cost[item] = {}
            for n in range(self.n_realizations):
                for ds in range(6):
                    if ds == 0:
                        repair_cost[item][n] = np.where(damage_state[item][n] == ds, ds, -1)

                    else:
                        best_fit = component_data.iloc[item - 1][f"DS{ds}, best fit"]
                        idx_list = np.where(damage_state[item][n] == ds)[0]
                        for idx_repair in idx_list:
                            if best_fit == 'normal truncated':
                                # TODO, Add options to truncate the distribution, add option to do multi-modal
                                #  distributions
                                pass
                            elif best_fit == 'lognormal':
                                a = np.random.normal(means_cost[idx][ds - 1],
                                                     covs_cost[idx][ds - 1] * means_cost[idx][ds - 1])
                                while a < 0:
                                    std = covs_cost[idx][ds - 1] * means_cost[idx][ds - 1]
                                    m = np.log(
                                        means_cost[idx][ds - 1] ** 2 / np.sqrt(means_cost[idx][ds - 1] ** 2 + std ** 2))
                                    std_log = np.sqrt(np.log(
                                        (means_cost[idx][ds - 1] ** 2 + std ** 2) / means_cost[idx][ds - 1] ** 2))
                                    a = np.random.lognormal(m, std_log)
                            else:
                                a = np.random.normal(means_cost[idx][ds - 1],
                                                     covs_cost[idx][ds - 1] * means_cost[idx][ds - 1])
                                while a < 0:
                                    a = np.random.normal(means_cost[idx][ds - 1], covs_cost[idx][ds - 1] *
                                                         means_cost[idx][ds - 1])

                            repair_cost[item][n][idx_repair] = a
            idx += 1

        # Evaluate the total damage cost multiplying the individual cost by each element quantity
        quantities = component_data["Quantity"]
        total_repair_cost = {}  # Total repair costs
        replacement_cost = {}  # Replacement costs
        loss_ratios = {}  # Loss ratios
        idx = 0
        for item in damage_state.keys():
            total_repair_cost[item] = {}
            replacement_cost[item] = max(means_cost[idx])
            loss_ratios[item] = {}
            for n in range(self.n_realizations):
                total_repair_cost[item][n] = repair_cost[item][n] * quantities.iloc[item - 1]
                loss_ratios[item][n] = repair_cost[item][n] / replacement_cost[item]
            idx += 1

        # Evaluate total loss for the floor segment
        total_loss_storey = {}
        for n in range(self.n_realizations):
            total_loss_storey[n] = np.zeros(len(self.edp_range))
            for item in damage_state.keys():
                total_loss_storey[n] += total_repair_cost[item][n]

        # Evaluate total loss ratio for the floor segment
        replacement_cost = np.array([replacement_cost[i] for i in replacement_cost])
        total_replacement_cost = sum(np.array(quantities) * replacement_cost)

        total_loss_storey_ratio = {}
        for n in range(self.n_realizations):
            total_loss_storey_ratio[n] = total_loss_storey[n] / total_replacement_cost

        return loss_ratios, total_loss_storey, total_loss_storey_ratio, total_replacement_cost, repair_cost

    def perform_regression(self, total_loss_storey, total_loss_ratio, edp, percentiles=None):
        """
        Performs regression and outputs final fitted results for the EDP-DV functions
        :param total_loss_storey: dict                      Total loss for the floor segment
        :param total_loss_ratio: dict                       Total loss ratio for the floor segment
        :param edp: str                                     EDP
        :param percentiles: list                            Percentiles to estimate
        :return: dict                                       Fitted EDP-DV functions
        """
        if percentiles is None:
            percentiles = [0.16, 0.50, 0.84]

        # Into a DataFrame for easy access for manipulation
        total_loss_storey = pd.DataFrame.from_dict(total_loss_storey)
        total_loss_ratio = pd.DataFrame.from_dict(total_loss_ratio)

        losses = {"loss_curve": total_loss_storey.quantile(percentiles, axis=1),
                  "loss_ratio_curve": total_loss_ratio.quantile(percentiles, axis=1)}

        mean_loss = np.mean(total_loss_storey, axis=1)
        mean_loss_ratio = np.mean(total_loss_ratio, axis=1)
        losses["loss_curve"].loc['mean'] = mean_loss
        losses["loss_ratio_curve"].loc['mean'] = mean_loss_ratio

        # Setting the edp range
        if edp == "IDR" or edp == 'PSD':
            edp_range = self.edp_range * 100
        else:
            edp_range = self.edp_range

        ''' Fitting the curve, EDP-DV functions '''
        if self.regression == "Weibull":
            def fitting_function(x, a, b, c):
                return a * (1 - np.exp(-(x / b) ** c))

        elif self.regression == "Papadopoulos":
            def fitting_function(x, a, b, c, d, e):
                return e*x**a/(b**a + x**a) + (1-e)*x**c/(d**c + x**c)

        else:
            raise ValueError("[EXCEPTION] Wrong type of regression function")

        # Fitted loss functions at specified quantiles
        losses_fitted = {}
        fitting_parameters = {}
        for q in percentiles:
            popt, pcov = curve_fit(fitting_function, edp_range, losses["loss_ratio_curve"].loc[q], maxfev=10**6)
            losses_fitted[q] = fitting_function(edp_range, *popt)
            fitting_parameters[q] = {"popt": popt, "pcov": pcov}

        # Fitting the mean
        popt, pcov = curve_fit(fitting_function, edp_range, losses["loss_ratio_curve"].loc['mean'], maxfev=10**6)
        losses_fitted['mean'] = fitting_function(edp_range, *popt)

        fitting_parameters['mean'] = {"popt": popt, "pcov": pcov}

        return losses, losses_fitted, fitting_parameters

    def get_in_euros(self, losses, total_replacement_cost):
        """
        Transforms losses to Euros
        :param losses: dict                         Fitted EDP-DV functions
        :param total_replacement_cost: float        Total replacement cost
        :return: dict                               Fitted EDP-DV functions in Euros
        """
        edp_dv = losses.copy()
        for key in losses.keys():
            edp_dv[key] = losses[key] * total_replacement_cost
        return edp_dv

    def get_slfs(self, outputs):
        """
        Normalizes EDP-DV functions with respect to the total story replacement cost
        :param outputs: dict                        Outputs
        :return: DataFrame                          Normalized mean SLFs of all performance groups
        """

        total_story_cost_actual = sum([outputs[key]['total_replacement_cost'] for key in outputs])
        total_story_cost = sum(outputs[key]['edp_dv_euro']['mean'][-1] for key in outputs)

        print(f"[WARNING] Actual total story cost is {total_story_cost_actual:.0f} with respect to "
              f"{total_story_cost:.0f} used for normalization")

        slfs = {}
        for key in outputs:
            slfs[key] = outputs[key]['edp_dv_euro']['mean'] / total_story_cost

        return slfs

    def accuracy(self, y, yhat):
        """
        Estimates prediction accuracy
        :param y: ndarray                           Actual values
        :param yhat: ndarray                        Estimations
        :return: float                              Maximum error in %
        """
        error_max = max(abs(y - yhat)/max(y))*100
        return error_max

    def master(self):
        """
        Runs the whole framework
        :return: dict                               Fitted EDP-DV functions
        """
        # Read component data
        component_data = self.get_component_data()

        # Group components
        component_groups = self.group_components(component_data)

        # Number of components per each performance group
        items_per_group = []
        for key in component_groups.keys():
            items_per_group.append(len(component_groups[key]))

        # Start toolbox for each component performance group
        cnt = 0
        # For visualization and storing relevant information for easy access
        cache = {}
        # For storing into a .xlsx file
        outputs = {}
        for group in component_groups:

            if not component_groups[group].empty:

                # Select component inventory to analyze
                component_data = component_groups[group]

                # Obtain correlation matrix
                matrix = self.get_correlation_tree(component_data)

                # EDP name and range
                edp = component_data["EDP"].iloc[0]
                self.get_edp_range(edp)

                # Obtain component fragility and consequence functions
                fragilities, means_cost, covs_cost = self.derive_fragility_functions(component_data)

                # Damage probabilities, placeholder
                damage_probs = self.get_DS_probs(component_data, fragilities)

                # Perform Monte Carlo simulations for damage state sampling
                damage_state = self.perform_Monte_Carlo(fragilities, matrix)

                # Populate the damage state matrix for correlated components
                damage_state = self.test_correlated_data(damage_state, matrix, fragilities)

                # Perform loss assessment
                loss_ratios, total_loss_storey, total_loss_ratio, total_replacement_cost, repair_cost = \
                    self.calculate_loss(component_data, damage_state, means_cost, covs_cost)

                # Perform regression
                losses, losses_fitted, fitting_pars = self.perform_regression(total_loss_storey, total_loss_ratio, edp)

                # Transform the EDP-DV functions into Euros
                edp_dv_functions = self.get_in_euros(losses_fitted, total_replacement_cost)

                # Quantifying the error on the mean curve
                error_median_max = self.accuracy(losses["loss_ratio_curve"].loc['mean'], losses_fitted['mean'])

                # Store outputs into a dictionary for saving as a .pickle file
                """
                component:                  Component inventory
                fragilities:                Component fragility functions
                edp_dv_euro:                Fitted EDP-DV functions in terms of EDP range vs Euro/USD etc.
                losses:                     Simulated fractile EDP-DV functions in terms of EDP range vs Euro/USD etc.
                edp_dv_fitted:              Normalized fitted functions 
                total_loss_storey:          Sampled story losses in Euros
                total_replacement_cost:     Total cost of the story
                SLFs:                       Normalized EDP-DV functions
                fit_pars:                   Regression parameters
                Accuracy:                   Accuracy of the regression function, e.g. error, accuracy
                """
                cache[f"{group}"] = {'component':                   component_data,
                                     'fragilities':                 fragilities,
                                     'edp_dv_euro':                 edp_dv_functions,
                                     'losses':                      losses,
                                     'edp_dv_fitted':               losses_fitted,
                                     'total_loss_storey':           total_loss_storey,
                                     'total_replacement_cost':      total_replacement_cost,
                                     'fit_pars':                    fitting_pars,
                                     'accuracy':                    error_median_max,
                                     "regression":                  self.regression}

                outputs[f"{group}"] = {'fit_pars':                  fitting_pars,
                                       'accuracy':                  error_median_max,
                                       'edp_dv':                    edp_dv_functions}

                cnt += 1

        # Get the normalized SLFs
        slfs = self.get_slfs(cache)

        cache['SLFs'] = slfs
        outputs['SLFs'] = slfs

        # Storing the outputs if asked
        if self.sflag:
            self.store_results(self.database_dir / f"{self.project_name}_{self.correlation}", cache, "pkl")
            print("[SUCCESS] Successful completion!")
        else:
            print("[SUCCESS] Successful completion!")

        return outputs, cache
