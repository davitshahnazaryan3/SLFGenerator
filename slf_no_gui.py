"""
Storey-loss-function (SLF) Generator adopted from the work of Sebastiano Zampieri by Davit Shahnazaryan

The tool allows the automatic production of EDP-DV SLFs based on input fragility, consequence and quantity data.

Considerations for double counting should be done at the input level and the consequence function should mirror it.

SLF estimation procedure:       Ramirez and Miranda 2009, CH.3 Storey-based building-specific loss estimation (p. 17)
FEMA P-58 for fragilities:      https://femap58.atcouncil.org/reports
For consequence functions:      https://femap58.atcouncil.org/reports
Python tools for reference:     https://blog.resellerclub.com/the-6-best-python-gui-frameworks-for-developers/
                                https://www.youtube.com/watch?v=627VBkAhKTc
For n Monte Carlo simulations:  https://www.vosesoftware.com/riskwiki/Howmanyiterationstorun.php

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
    def __init__(self, project_name, component_data_filename, correlation_tree_filename, edp_bin=None,
                 correlation="Correlated", regression="Weibull", n_realizations=20, conversion=1.0, do_grouping=True,
                 sflag=True):
        """
        Initialization of the Master Generator
        :param project_name: str                Name of the project to save in the database
        :param component_data_filename: str     Component data file name, e.g. "*.csv"
        :param correlation_tree_filename: str   Correlation tree file name, e.g. "*.csv"
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
        self.database_dir = self.dir/"Database"
        self.project_name = project_name
        self.component_data_filename = component_data_filename
        self.correlation_tree_filename = correlation_tree_filename
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
        Store results in the database
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

        Notes on creation of a correlation tree: Make sure that the MIN DS assigned does not exceed the possible Damage States
        defined for the element. Essentially, if a component is dependent on another component and has only one DS, i.e. DS1, which
        occurs only if the causation element is for example at its DS3, then the following would be true. And Item3 would be damaged
        earlier and has more DSs. The software will not detec any errors, so it depends on the user.
        Item ID: 	Dependant on item 	MIN DS|DS0 	MIN DS|DS1 	MIN DS|DS2 	MIN DS|DS3 	MIN DS|DS4 	MIN DS|DS5
        Item 1: 	Independent 		Independent	Independent Independent Independent Independent Independent
		Item 2:		1					Undamaged 	Undamaged  	Undamaged 	DS1 		DS1 		DS1
		Item 3: 	1 					Undamaged 	Undamaged 	DS1 		DS2 		DS3 		DS3
		
        :return: DataFrame                  DataFrame containing the component data
        """
        component_data = pd.read_csv(self.dir/"client"/self.component_data_filename)
        n_items = len(component_data)
            
        # Check for any missing values in the best_fit features
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
        Component grouping by performance
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
            
        # Check for errors in the input
        correlation_tree = pd.read_csv(self.dir/"client"/self.correlation_tree_filename)
        
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
                    if math.isnan(matrix[i][j-1]):
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
        :return: dict                           Fragilities of all components at all DSs
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
        data = component_data.values[:, 4:]

        # Get parameters of the fragility and consequence functions
        for item in range(len(data)):
            for ds in range(max_ds_selected):
                means_fr[item][ds] = data[item][ds]
                covs_fr[item][ds] = data[item][ds+5]
                means_cost[item][ds] = data[item][ds+10]*self.conversion
                covs_cost[item][ds] = data[item][ds+15]
                
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
        :param fragilities: dict                Fragilities of all components at all DSs
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
                        y = fragilities["ITEMs"][item][f"DS{ds+1}"]
                        damage = np.where((random_array >= y) & (random_array < y1), ds_range[ds], damage)
                damage_state[item][n] = damage
        return damage_state
    
    def test_correlated_data(self, damage_state, matrix):
        """
        Tests if any non-assigned DS exist (i.e. -1) and makes correction if necessary
        :param damage_state: dict               Damage states of each component for each simulation
        :param matrix: ndarray                  Correlation tree matrix
        :return: dict                           Damage states of each component for each simulation
        """
        # Loop over each component
        for i in range(matrix.shape[0]):
            # Check if component is correlated or independent
            if i + 1 != matrix[i][0]:
                # -- Component is correlated 
                # Causation component ID
                m = matrix[i][0]
                # Correlated component ID
                j = i + 1
                # Loop for each simulation
                for n in range(self.n_realizations):
                    causation_ds = damage_state[m][n]
                    correlated_ds = damage_state[j][n]
                   
                    # Get correlated components DS conditioned on causation component
                    temp = np.zeros((causation_ds.shape))
                    # Loop over each DS
                    for ds in range(1, matrix.shape[1]):
                        if ds == 1:
                            temp = np.where(causation_ds==ds-1, matrix[j-1][ds], causation_ds)
                        else: 
                            temp = np.where(temp==ds-1, matrix[j-1][ds], temp)
                            temp = np.where(temp==ds-1, matrix[j-1][ds], temp)
                            temp = np.where(temp==ds-1, matrix[j-1][ds], temp)
                            temp = np.where(temp==ds-1, matrix[j-1][ds], temp)
                  
                    # Modify DS if correlated component is conditioned on causation component's DS, otherwise skip                
                    damage_state[j][n] = np.maximum(correlated_ds, temp)

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
                        best_fit = component_data.loc[item-1][f"DS{ds}, best fit"]
                        idx_list = np.where(damage_state[item][n] == ds)[0]
                        for idx_repair in idx_list:
                            if best_fit == 'normal truncated':
                                pass
                            elif best_fit == 'lognormal':
                                a = np.random.normal(means_cost[idx][ds-1], covs_cost[idx][ds-1]*means_cost[idx][ds-1])
                                while a < 0:
                                    std = covs_cost[idx][ds-1]*means_cost[idx][ds-1]
                                    m = np.log(means_cost[idx][ds-1]**2/np.sqrt(means_cost[idx][ds-1]**2+std**2))
                                    std_log = np.sqrt(np.log((means_cost[idx][ds-1]**2+std**2)/means_cost[idx][ds-1]**2))
                                    a = np.random.lognormal(m, std_log)
                            else:
                                a = np.random.normal(means_cost[idx][ds-1], covs_cost[idx][ds-1]*means_cost[idx][ds-1])
                                while a < 0:
                                    a = np.random.normal(means_cost[idx][ds-1], covs_cost[idx][ds-1] *
                                                         means_cost[idx][ds-1])
                            
                            repair_cost[item][n][idx_repair] = a
            idx += 1
        
        # Evaluate the total damage cost multiplying the individual cost by each element quantity
        quantities = component_data["Quantity"]             # Component quantities
        total_repair_cost = {}                              # Total repair costs
        replacement_cost = {}                               # Replacement costs
        loss_ratios = {}                                    # Loss ratios
        idx = 0
        for item in damage_state.keys():
            total_repair_cost[item] = {}
            replacement_cost[item] = max(means_cost[idx])
            loss_ratios[item] = {}
            for n in range(self.n_realizations):
                total_repair_cost[item][n] = repair_cost[item][n]*quantities[item-1]
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
        total_replacement_cost = sum(np.array(quantities)*replacement_cost)
        
        total_loss_storey_ratio = {}
        for n in range(self.n_realizations):
            total_loss_storey_ratio[n] = total_loss_storey[n]/total_replacement_cost
        
        return loss_ratios, total_loss_storey, total_loss_storey_ratio, total_replacement_cost, repair_cost
    
    def perform_regression(self, total_loss_storey, total_loss_ratio, edp, percentiles=None):
        """
        Performs regression and outputs final fitted results for the EDP-DV functions
        :param total_loss_storey: dict                      Total loss for the floor segment
        :param total_loss_ratio: dict                       Total loss ratio for the floor segment
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
            edp_range = self.edp_range*100
        else:
            edp_range = self.edp_range

        ''' Fitting the curve, EDP-DV functions, assuming Weibull Distribution '''
        if self.regression == "Weibull":
            def fitting_function(x, a, b, c):
                return a*(1 - np.exp(-(x/b)**c))

        elif self.regression == "Papadopoulos":
            def fitting_function(x, a, b, c, d, e):
                return e*x**a/(b**a + x**a) + (1-e)*x**c/(d**c + x**c)

        else:
            raise ValueError("[EXCEPTION] Wrong type of regression function")
        
        losses_fitted = {}
        fitting_parameters = {}
        for q in percentiles:
            popt, pcov = curve_fit(fitting_function, edp_range, losses["loss_ratio_curve"].loc[q], maxfev=10**6)
            losses_fitted[q] = fitting_function(edp_range, *popt)
            fitting_parameters[q] = {"popt": popt, "pcov": pcov}
            
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
            edp_dv[key] = losses[key]*total_replacement_cost
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
        
    def master(self, sensitivityflag=False):
        """
        Runs the whole framework
        :param sensitivityflag: bool                Whether to run both 'independent' and 'correlated' options
        :return: dict                               Fitted EDP-DV functions
        """
        # Perform sensitivity, i.e. run both independent and correlated options
        if sensitivityflag:
            correlation = ["Independent", "Correlated"]
            for c in correlation:
                self.correlation = c
                # Read component data
                component_data = self.get_component_data()
                component_groups = self.group_components(component_data)
                items_per_group = []
                for key in component_groups.keys():
                    items_per_group.append(len(component_groups[key]))
                    
                cnt = 0
                outputs = {}
                for group in component_groups:
                    if not component_groups[group].empty:
                        component_data = component_groups[group]
                        matrix = self.get_correlation_tree(component_data)
                        edp = group[0:3]
                        self.get_edp_range(edp)
                        
                        # Get number of items of the previous performance group
                        if cnt == 0:
                            n_items = 0
                        else:
                            n_items = sum(items_per_group[:cnt])
                        
                        fragilities, means_cost, covs_cost = self.derive_fragility_functions(component_data, n_items)
                        damage_probs = self.get_DS_probs(component_data, fragilities)
                        damage_state = self.perform_Monte_Carlo(fragilities, matrix)
                        damage_state = self.test_correlated_data(damage_state, matrix, fragilities)
                        loss_ratios, total_loss_storey, total_loss_ratio, total_replacement_cost, repair_cost = \
                            self.calculate_loss(component_data, damage_state, means_cost, covs_cost)
                        losses, losses_fitted = self.perform_regression(total_loss_storey, total_loss_ratio, edp)
                        edp_dv_functions = self.get_in_euros(losses_fitted, total_replacement_cost)
                        outputs[group] = {'component': component_data, 'correlation_tree': matrix,
                                          'fragilities': fragilities, 'damage_states': damage_state, 'losses': losses,
                                          'edp_dv_fitted': losses_fitted, 'edp_dv_euro': edp_dv_functions,
                                          'total_replacement_cost': total_replacement_cost, 'total_loss_storey': total_loss_storey}
                        cnt += 1
                    
                slfs = self.get_slfs(outputs)
                outputs['SLFs'] = slfs
                    
                if self.sflag:
                    self.store_results(self.database_dir/f"{self.project_name}_{self.correlation}", outputs, "pkl")
                    print("[SUCCESS] Successful completion, outputs have been stored!")
                else:
                    print("[SUCCESS] Successful completion, outputs have not been stored!")
                    
        else:
            print(f"[INITIATE] Running assuming {self.correlation} components")
            # Read component data
            component_data = self.get_component_data()
            component_groups = self.group_components(component_data)
            items_per_group = []
            for key in component_groups.keys():
                items_per_group.append(len(component_groups[key]))
                
            cnt = 0
            outputs = {}
            for group in component_groups:
                if not component_groups[group].empty:
                    component_data = component_groups[group]
                    matrix = self.get_correlation_tree(component_data)
                    edp = component_data["EDP"].iloc[0]
                    self.get_edp_range(edp)

                    fragilities, means_cost, covs_cost = self.derive_fragility_functions(component_data)
                    damage_state = self.perform_Monte_Carlo(fragilities, matrix)
                    damage_state = self.test_correlated_data(damage_state, matrix, fragilities)
                    loss_ratios, total_loss_storey, total_loss_ratio, total_replacement_cost, repair_cost = \
                        self.calculate_loss(component_data, damage_state, means_cost, covs_cost)
                    losses, losses_fitted, fitting_pars = self.perform_regression(total_loss_storey, total_loss_ratio, edp)
                    edp_dv_functions = self.get_in_euros(losses_fitted, total_replacement_cost)
                    outputs[group] = {'component': component_data, 'correlation_tree': matrix, 'fragilities': fragilities,
                                       'damage_states': damage_state, 'losses': losses, 'edp_dv_fitted': losses_fitted,
                                       'edp_dv_euro': edp_dv_functions, 'total_replacement_cost': total_replacement_cost, 
                                       'total_loss_storey': total_loss_storey, "fit_pars": fitting_pars}
                    
                    cnt += 1
            
            slfs = self.get_slfs(outputs)
            outputs['SLFs'] = slfs
            
            if self.sflag:
                self.store_results(self.database_dir/f"{self.project_name}_{self.correlation}", outputs, "pkl")
                print("[SUCCESS] Successful completion, outputs have been stored!")
            else:
                print("[SUCCESS] Successful completion, outputs have not been stored!")
        
        return outputs


if __name__ == "__main__":
    """
    Takes as input:
    Component data              .csv file including quantity and fragility information for each component
    Correlation tree            .csv file containing the correlation tree
    edp_bin                     Step of edp
    n_realizations              number of simulations per edp
    """
    slf = SLF("test1", "test/idr_inv.csv", "test/idr_corr.csv", correlation="Correlated", n_realizations=100,
              sflag=False, do_grouping=False, regression="Weibull")
    run_master = False
    if run_master:
        outputs = slf.master(sensitivityflag=False)
    else:
        component_data = slf.get_component_data()
        component_groups = slf.group_components(component_data)
        items_per_group = []
        for key in component_groups.keys():
            items_per_group.append(len(component_groups[key]))
            
        ouputs = {}
        for group in component_groups:
            if not component_groups[group].empty:
                component_data = component_groups[group]
                matrix = slf.get_correlation_tree(component_data)
                edp = group[0:3]
                slf.get_edp_range(edp)
                
                fragilities, means_cost, covs_cost = slf.derive_fragility_functions(component_data)
                damage_state = slf.perform_Monte_Carlo(fragilities)
                damage_state = slf.test_correlated_data(damage_state, matrix, fragilities)
                # loss_ratios, total_loss_storey, total_loss_ratio, total_replacement_cost, repair_cost = \
                #     slf.calculate_loss(component_data, damage_state, means_cost, covs_cost)
                # losses, losses_fitted, fitting_pars = slf.perform_regression(total_loss_storey, total_loss_ratio, edp)
                # edp_dv_functions = slf.get_in_euros(losses_fitted, total_replacement_cost)
                

    # import matplotlib.pyplot as plt
    # fig1, ax = plt.subplots(figsize=(4, 3), dpi=100)
    # edp = fragilities["EDP"]
    # for i in range(1, 4):
    #     prob1 = fragilities["ITEMs"][1][f"DS{i}"]
    #     prob2 = fragilities["ITEMs"][2][f"DS{i}"]
    #     if i == 3:
    #         label1 = "Causation component"
    #         label2 = "Dependent component"
    #     else:
    #         label1 = label2 = None
    #     plt.plot(edp*100, prob1, color="b", marker="*", markevery=5, label=label1)
    #     plt.plot(edp*100, prob2, color="r", ls="--", marker="o", markevery=5, label=label2)
    # plt.grid(True, which="major", axis="both", ls="--", lw=1.0)
    # plt.ylim(0, 1)
    # plt.xlim(0, 5)
    # plt.xlabel("IDR [%]")
    # plt.ylabel("P[D>DS| IDR]")
    # ax.legend(loc='best', frameon=False)
    
#    loss = edp_dv_functions["mean"]/1000
#    fig2, ax = plt.subplots(figsize=(4, 3), dpi=100)
#    plt.plot(edp*100, loss, label="Independent", marker="o", markevery=5)
#    plt.grid(True, which="major", axis="both", ls="--", lw=1.0)
#    plt.ylim(0, 120)
#    plt.xlim(0, 5)
#    print(loss[20])

















