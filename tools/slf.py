"""
Storey-loss-function (SLF) Generator adopted from the work of Sebastiano Zampieri by Davit Shahnazaryan

The tool allows the automatic production of SLFs based on input fragility, consequence and quantity data.

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
from scipy import stats
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')


class SLF:
    def __init__(self, project_name, component_data, correlation_tree=None, edp_bins=None, correlation="Correlated",
                 regression="Weibull", n_realizations=20, conversion=1.0, replCost=0.0, do_grouping=True):
        """
        Initialization of the Master Generator
        TODO, add option for mutually exclusive damage states
        TODO, include possibility of including quantity uncertainties along with the mean values
        :param project_name: str                Name of the project to save in the database
        :param component_data: dict             Component data inventory
        :param correlation_tree: dict           Correlation tree
        :param edp_bins: float                  EDP sampling unit, % for PSD, g for PFA (non-negative)
        :param correlation: str                 Whether the elements are "Independent" or "Correlated"
        :param regression: str                  Whether the regression function is based on "Weibull" or "Papadopoulos"
        :param n_realizations: int              Number of realizations
        :param conversion: float                Conversion factor from usd to euro, e.g. if provided in euro, use 1.0;
                                                if 1 usd = 0.88euro, use 0.88
                                                or use 1.0 if ratios are used directly
        :param replCost: float                  Replacement cost of the building (used when normalizing the SLFs)
        :param do_grouping: bool                Whether to do performance grouping or use the entire component database
        """
        if edp_bins is None:
            edp_bins = [0.1, 0.05]
        self.dir = Path.cwd()
        self.database_dir = self.dir / "cache"
        self.project_name = project_name
        self.component_data = component_data
        self.correlation_tree = correlation_tree
        self.edp_bins = edp_bins
        self.correlation = correlation
        self.regression = regression
        self.n_realizations = n_realizations
        self.conversion = conversion
        self.replCost = replCost
        self.do_grouping = do_grouping
        self.edp_range = None
        self.edp_bin = None

    def get_edp_range(self, edp):
        """
        Identifies the EDP range based on edp type and input edp bin
        :param edp: str                         EDP type (PSD or PFA)
        :return: None
        """
        # EDP range
        if edp == "IDR" or edp == "PSD":
            # stop calculation at 10% drift
            self.edp_bin = self.edp_bins[0] / 100.
            self.edp_range = np.arange(0, 0.2 + self.edp_bin, self.edp_bin)
        elif edp == "PFA":
            # stop calculation at 2.5g of acceleration
            self.edp_bin = self.edp_bins[1]
            self.edp_range = np.arange(0, 10.0 + self.edp_bin, self.edp_bin)
        else:
            raise ValueError("[EXCEPTION] Wrong EDP type, must be 'IDR'/'PSD' or 'PFA'")

    def get_component_data(self):
        """
        Gets component information from the user provided .csv file
        TODO: for UI update, the user will have the following options to manipulate the data and create the .csv file through the UI
                    1. Add new component
                    2. View existing components (e.g., FEMA P-58 PACT or added previously by the user)
                    3. Edit an existing component
                    4. Delete an existing component
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
                    psd_s = component_data[(component_data["EDP"] == "PSD") & (component_data["Component"] == "S")]
                    psd_ns = component_data[(component_data["EDP"] == "PSD") & (component_data["Component"] == "NS")]
                    pfa_ns = component_data[(component_data["EDP"] == "PFA") & (component_data["Component"] == "NS")]

                    component_groups = {"PSD S": psd_s, "PSD NS": psd_ns, "PFA NS": pfa_ns}

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

        # Verify integrity of the provided correlation tree
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
        matrix = np.zeros((c_tree.shape[0], c_tree.shape[1]), dtype=int)
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
                    else:
                        matrix[i][j] = int(c_tree[i][j][-1])

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
        # TODO, modify 4, make index reading smarter
        data = component_data.values[:, 4:]

        # Get parameters of the fragility and consequence functions
        for item in range(len(data)):
            for ds in range(max_ds_selected):
                means_fr[item][ds] = data[item][ds]
                covs_fr[item][ds] = data[item][ds + 5]
                means_cost[item][ds] = data[item][ds + 10] * self.conversion
                covs_cost[item][ds] = data[item][ds + 15]

        # Deriving the ordinates of the fragility functions
        fragilities = {"EDP": self.edp_range, "ITEMs": {}}
        for item in range(len(data)):
            fragilities["ITEMs"][item + 1] = {}
            for ds in range(max_ds_selected):
                mean = np.exp(np.log(means_fr[item][ds]) - 0.5 * np.log(covs_fr[item][ds] ** 2 + 1))
                std = np.log(covs_fr[item][ds] ** 2 + 1) ** 0.5
                if mean == 0 and std == 0:
                    fragilities["ITEMs"][item + 1][f"DS{ds + 1}"] = np.zeros(len(self.edp_range))
                else:
                    frag = stats.norm.cdf(np.log(self.edp_range / mean) / std, loc=0, scale=1)
                    frag[np.isnan(frag)] = 0
                    fragilities["ITEMs"][item + 1][f"DS{ds + 1}"] = frag

        return fragilities, means_cost, covs_cost

    def perform_Monte_Carlo(self, fragilities):
        """
        Performs Monte Carlo simulations and simulates DS for each EDP step
        :param fragilities: dict                Fragility functions of all components at all DSs
        :return: dict                           Damage states of each component for each simulation
        """
        num_ds = len(fragilities["ITEMs"][1])
        ds_range = np.arange(0, num_ds + 1, 1)
        num_edp = len(fragilities["EDP"])
        damage_state = {}

        # Evaluate the DS on the i-th component for EDPs at the n-th simulation
        # Items
        for item in fragilities["ITEMs"]:
            damage_state[item] = {}
            # Simulations
            for n in range(self.n_realizations):
                # Generate random data between (0, 1)
                random_array = np.random.rand(num_edp)
                damage = np.zeros(num_edp, dtype=int)
                # DS
                for ds in range(num_ds, 0, -1):
                    y1 = fragilities["ITEMs"][item][f"DS{ds}"]
                    if ds == num_ds:
                        damage = np.where(random_array <= y1, ds_range[ds], damage)
                    else:
                        y = fragilities["ITEMs"][item][f"DS{ds + 1}"]
                        damage = np.where((random_array >= y) & (random_array < y1), ds_range[ds], damage)
                damage_state[item][n] = damage
        return damage_state

    def assign_ds_to_dependent(self, damage_state, matrix):
        """
        Corrects the assignment of DSs for dependent components
        :param damage_state: dict               Damage states of each component for each simulation
        :param matrix: ndarray                  Correlation tree matrix
        :return: dict                           Damage states of each component for each simulation
        """
        # Loop over each component
        for i in range(matrix.shape[0]):
            # Check if component is dependent or independent
            if i + 1 != matrix[i][0]:
                # -- Component is dependent
                # Causation component ID
                m = matrix[i][0]
                # Dependent component ID
                j = i + 1
                # Loop for each simulation
                for n in range(self.n_realizations):
                    causation_ds = damage_state[m][n]
                    correlated_ds = damage_state[j][n]

                    # Get dependent components DS conditioned on causation component
                    temp = np.zeros(causation_ds.shape)
                    # Loop over each DS
                    for ds in range(1, matrix.shape[1]):
                        temp[causation_ds == ds-1] = matrix[j-1][ds]

                    # Modify DS if correlated component is conditioned on causation component's DS, otherwise skip
                    damage_state[j][n] = np.maximum(correlated_ds, temp)

        return damage_state

    def calculate_costs(self, component_data, damage_state, means_cost, covs_cost):
        """
        Evaluates the damage cost on the individual i-th component at each EDP level for each n-th simulation
        :param component_data: DataFrame                DataFrame containing the component data
        :param damage_state: dict                       Damage states of each component for each simulation
        :param means_cost: ndarray                      Means of repair costs
        :param covs_cost: ndarray                       Covariances of repair costs
        :return: dict                                   Repair costs
        """
        # Number of damage states
        num_ds = means_cost.shape[1]

        repair_cost = {}
        idx = 0
        for item in damage_state.keys():
            repair_cost[item] = {}
            for n in range(self.n_realizations):
                for ds in range(num_ds + 1):
                    if ds == 0:
                        repair_cost[item][n] = np.where(damage_state[item][n] == ds, ds, -1)

                    else:
                        # Best fit function
                        best_fit = component_data.iloc[item - 1][f"DS{ds}, best fit"]
                        # EDP ID where ds is observed
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
        quantities = component_data["Quantity"]     # Component quantities
        total_repair_cost = {}                      # Total repair costs
        idx = 0
        for item in damage_state.keys():
            total_repair_cost[item] = {}
            for n in range(self.n_realizations):
                total_repair_cost[item][n] = repair_cost[item][n] * quantities.iloc[item - 1]
            idx += 1

        # Evaluate total loss for the storey segment
        total_loss_storey = {}
        for n in range(self.n_realizations):
            total_loss_storey[n] = np.zeros(len(self.edp_range))
            for item in damage_state.keys():
                total_loss_storey[n] += total_repair_cost[item][n]

        # Calculate if replCost was set to 0, otherwise use the provided value
        if self.replCost == 0.0 or self.replCost is None:
            raise ValueError("[EXCEPTION] Replacement cost should be a non-negative non-zero value.")
        else:
            total_replacement_cost = self.replCost

        total_loss_storey_ratio = {}
        for n in range(self.n_realizations):
            total_loss_storey_ratio[n] = total_loss_storey[n] / total_replacement_cost

        return total_loss_storey, total_loss_storey_ratio, repair_cost

    def perform_regression(self, total_loss_storey, total_loss_ratio, edp, percentiles=None):
        """
        Performs regression and outputs final fitted results for the SLFs
        :param total_loss_storey: dict                      Total loss for the floor segment
        :param total_loss_ratio: dict                       Total loss ratio for the floor segment
        :param edp: str                                     EDP
        :param percentiles: list                            Percentiles to estimate
        :return: dict                                       Fitted SLFs
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

        ''' Fitting the curve, SLFs '''
        if self.regression == "Weibull":
            def fitting_function(x, a, b, c):
                return a * (1 - np.exp(-(x / b) ** c))

        elif self.regression == "Papadopoulos":
            def fitting_function(x, a, b, c, d, e):
                return e*x**a/(b**a + x**a) + (1-e)*x**c/(d**c + x**c)

        else:
            raise ValueError("[EXCEPTION] Wrong type of regression function")

        # Fitted loss functions at specified quantiles normalised by the Replacement Cost
        losses_fitted = {}
        fitting_parameters = {}
        for q in percentiles:
            maxVal = max(losses["loss_ratio_curve"].loc[q])
            popt, pcov = curve_fit(fitting_function, edp_range, losses["loss_ratio_curve"].loc[q]/maxVal, maxfev=10**6)
            losses_fitted[q] = fitting_function(edp_range, *popt)*maxVal
            # Truncating at zero to prevent negative values
            losses_fitted[q][losses_fitted[q] <= 0] = 0.0
            fitting_parameters[q] = {"popt": popt, "pcov": pcov}

        # Fitting the mean
        maxVal = max(losses["loss_ratio_curve"].loc['mean'])
        popt, pcov = curve_fit(fitting_function, edp_range, losses["loss_ratio_curve"].loc['mean']/maxVal, maxfev=10**6)
        losses_fitted['mean'] = fitting_function(edp_range, *popt)*maxVal

        fitting_parameters['mean'] = {"popt": popt, "pcov": pcov}

        return losses, losses_fitted, fitting_parameters

    def compute_accuracy(self, y, yhat):
        """
        Estimates prediction accuracy
        :param y: ndarray                           Actual values
        :param yhat: ndarray                        Estimations
        :return: float                              Maximum error in %, Cumulative error in %
        """
        error_max = max(abs(y - yhat)/max(y)) * 100
        error_cum = self.edp_bin * sum(abs(y - yhat) / max(y)) * 100
        return error_max, error_cum

    def master(self):
        """
        Runs the whole framework
        :return: dict                               Fitted SLFs
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
        slf_output = {}
        # For storing into a .xlsx file
        outputs = {}

        for group in component_groups:

            if not component_groups[group].empty:

                # Select component inventory to analyze
                component_data = component_groups[group]

                # Obtain correlation matrix
                if self.correlation == "Correlated":
                    matrix = self.get_correlation_tree(component_data)
                else:
                    matrix = None

                # EDP name and range
                edp = component_data["EDP"].iloc[0]
                self.get_edp_range(edp)

                # Obtain component fragility and consequence functions
                fragilities, means_cost, covs_cost = self.derive_fragility_functions(component_data)

                # Perform Monte Carlo simulations for damage state sampling
                damage_state = self.perform_Monte_Carlo(fragilities)

                # Populate the damage state matrix for correlated components
                if self.correlation == "Correlated":
                    damage_state = self.assign_ds_to_dependent(damage_state, matrix)

                # Perform loss assessment
                total_loss_storey, total_loss_storey_ratio, repair_cost = \
                    self.calculate_costs(component_data, damage_state, means_cost, covs_cost)

                # Perform regression
                losses, slfs, fitting_pars = self.perform_regression(total_loss_storey, total_loss_storey_ratio, edp)

                # Quantifying the error on the mean curve
                error_max, error_cum = self.compute_accuracy(losses["loss_ratio_curve"].loc['mean'], slfs['mean'])

                # Store outputs into a dictionary for saving as a .pickle file
                cache[f"{group}"] = {'component':                   component_data,
                                     'fragilities':                 fragilities,
                                     'total_loss_storey':           total_loss_storey,
                                     'total_loss_storey_ratio':     total_loss_storey_ratio,
                                     'repair_cost':                 repair_cost,
                                     'damage_states':               damage_state,
                                     'losses':                      losses,
                                     'slfs':                        slfs,
                                     'fit_pars':                    fitting_pars,
                                     'accuracy':                    [error_max, error_cum],
                                     "regression":                  self.regression}

                slf_output[f"{group}"] = {"slfs":                   slfs}

                outputs[f"{group}"] = {'fit_pars':                  fitting_pars,
                                       'accuracy':                  [error_max, error_cum],
                                       'slfs':                      slfs}

                cnt += 1

        # Completion
        print("[SUCCESS] Successful completion!")

        return outputs, slf_output, cache
