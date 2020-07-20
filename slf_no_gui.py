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
    def __init__(self, project_name, component_data_filename, correlation_tree_filename, edp_bin=0.1,
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
        
    def get_edp_range(self, edp, edp_bin=None):
        """
        Identifies the EDP range based on edp type and input edp bin
        :param edp: str                         EDP type (IDR or PFA)
        :param edp_bin: float                   EDP sampling unit, % for IDR, g for PFA (non-negative) 
        :return: None
        """
        if edp_bin is not None:
            edp_bin = edp_bin
        
        # EDP range
        if edp == "IDR" or edp == "PSD":
            # stop calculation at 10% drift
            edp_bin = self.edp_bin/100.
            self.edp_range = np.arange(0, 0.2 + edp_bin, edp_bin)
        elif edp == "PFA":
            # stop calculation at 2.5g of acceleration
            edp_bin = 0.25
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
        if len(component_data.keys()) != 31:
            raise ValueError("[EXCEPTION] Unexpected number of features in the components DataFrame")
            
        # Check for any missing values in the best_fit features
        for key in component_data.keys():
            if key.endswith('best fit'):
                component_data[key].fillna('normal', inplace=True)
                
        # Replace all nan with 0.0 for the rest of the DataFrame
        component_data.fillna(0.0, inplace=True)

        return component_data
    
    def group_components(self, component_data):
        """
        Component grouping by performance
        :param component_data: DataFrame        DataFrame containing the component data
        :return: dict                           Dictionary containing DataFrames for each performance group
        """
        if self.do_grouping:
            idr_s = component_data[(component_data["EDP"] == "IDR") & (component_data["Component"] == "S")]
            idr_ns = component_data[(component_data["EDP"] == "IDR") & (component_data["Component"] == "NS")]
            pfa_ns = component_data[(component_data["EDP"] == "PFA") & (component_data["Component"] == "NS")]
            
            component_groups = {"IDR S": idr_s, "IDR NS": idr_ns, "PFA NS": pfa_ns}
        else:
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
        for item in component_data.index:
            for feature in component_data.keys():
                ds = component_data.loc[item][feature]
                if ds == 0:
                    f = feature[0:3]
                    break
            max_DS.append(f)
            
        # Check for errors in the input
        correlation_tree = pd.read_csv(self.dir/"client"/self.correlation_tree_filename)
        
        # Select the items within the component performance group
        correlation_tree = correlation_tree.loc[component_data.index]
        
        if len(correlation_tree.keys()) != 8:
            raise ValueError("[EXCEPTION] Unexpected number of features in the correlations DataFrame")
            
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

    def derive_fragility_functions(self, component_data, n_items=None):
        """
        Derives fragility functions
        :param component_data: DataFrame        DataFrame containing the component data
        :param n_items: int                     Number of items from the previous group
        :return: dict                           Fragilities of all components at all DSs
        """
        if n_items == None:
            n_items = 0
            
        # Select only numeric features
        component_data = component_data.select_dtypes(exclude=["object"])
        
        # Fragility parameters
        means_fr = np.zeros((len(component_data), 5))
        covs_fr = np.zeros((len(component_data), 5))
        
        # Consequence parameters
        means_cost = np.zeros((len(component_data), 5))
        covs_cost = np.zeros((len(component_data), 5))
        
        # Deriving fragility functions
        data = component_data.values[:, 3:]
        for item in range(len(data)):
            for ds in range(5):
                means_fr[item][ds] = data[item][ds]
                covs_fr[item][ds] = data[item][ds+5]
                means_cost[item][ds] = data[item][ds+10]*self.conversion
                covs_cost[item][ds] = data[item][ds+15]
                
        # Deriving the ordinates of the fragility functions
        fragilities = {"EDP": self.edp_range, "ITEMs": {}}
        for item in range(len(data)):
            fragilities["ITEMs"][item+1+n_items] = {}
            for ds in range(5):
                mean = np.exp(np.log(means_fr[item][ds])-0.5*np.log(covs_fr[item][ds]**2+1))
                std = np.log(covs_fr[item][ds]**2+1)**0.5
                if mean == 0 and std == 0:
                    fragilities["ITEMs"][item+1+n_items][f"DS{ds+1}"] = np.zeros(len(self.edp_range))
                else:
                    frag = stats.norm.cdf(np.log(self.edp_range/mean)/std, loc=0, scale=1)
                    frag[np.isnan(frag)] = 0
                    fragilities["ITEMs"][item+1+n_items][f"DS{ds+1}"] = frag
        
        return fragilities, means_cost, covs_cost
    
    def get_DS_probs(self, component_data, fragilities):
        """
        Evaluates probabilities of having each damage state for every EDP
        :param component_data: DataFrame        DataFrame containing the component data
        :param fragilities: dict                Fragilities of all components at all DSs
        :return: dict                           Probabilities of beining in a given DS
        """
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
                        damage_probs[item][f"DS{ds}"] = 1 - y[f"DS{ds+1}"]
                    elif ds == 5:
                        damage_probs[item][f"DS{ds}"] = y[f"DS{ds}"]
                    else:
                        damage_probs[item][f"DS{ds}"] = y[f"DS{ds}"] - y[f"DS{ds+1}"] 
            return damage_probs
    
    def perform_Monte_Carlo(self, fragilities, corr_tree=None):
        """ 
        Performs Monte Carlo simulations and simulates DS for each EDP step
        :param fragilities: dict                Fragilities of all components at all DSs
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
                            y = fragilities["ITEMs"][item][f"DS{ds+1}"]
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
        :param fragilities: dict                Fragilities of all components at all DSs
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
                min_ds[item+1] = {}
                y_new[item+1] = {}

                # EDP values
                for edp in range(len(self.edp_range)):
                    min_ds[item+1][edp] = {}
                    y_new[item+1][edp] = {}
                    
                    # Simulations
                    for n in range(self.n_realizations):
                        y_new[item+1][edp][n] = {}
                        # Find the DS on the causation element
                        if damage_state[item+1][n][edp] == -1:
                            # if the causation element still has not been assigned a DS
                            if damage_state[int(matrix[item][0])][n][edp] == -1:
                                # The marker will make the engine simulate the DS at the successive iteration
                                damage_state[item+1][n][edp] = -1
                            else:
                                # Finds the minimum DS for the i-th element
                                min_ds[item+1][edp][n] = matrix[item][1+int(damage_state[matrix[item][0]][n][edp])]
                                # Recalculates the probability of having each DS in the condition of having a min DS
                                # Damage states.
                                for ds in range(5):
                                    # All DS smaller than min_DS have a probability of 1 of being observed
                                    if min_ds[item+1][edp][n] >= ds + 1:
                                        # probability of having DS >= min_k
                                        y_new[item+1][edp][n][ds+1] = 1
                                        # if min_DS is zero then the probabilities are unchanged
                                    elif min_ds[item+1][edp][n] == 0:
                                        y = fragilities["ITEMs"][item+1][f"DS{ds+1}"]
                                        y_new[item+1][edp][n][ds+1] = y[edp]
                                    else:
                                        # conditional probability of having DS >= DS_ds given min_DS
                                        y = fragilities["ITEMs"][item+1][f"DS{ds+1}"]
                                        y1 = fragilities["ITEMs"][item+1][f"DS{int(min_ds[item+1][edp][n])+1}"]
                                        y_new[item+1][edp][n][ds+1] = y[edp]/y1[edp]
                                        if math.isnan(y_new[item+1][edp][n][ds+1]):
                                            y_new[item+1][edp][n][ds+1] = 0
                                            
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
                                    a = np.random.normal(means_cost[idx][ds-1], covs_cost[idx][ds-1]*
                                                         means_cost[idx][ds-1])
                            
                            repair_cost[item][n][idx_repair] = a
            idx += 1
        
        # Evaluate the total damage cost multiplying the individual cost by each element quantity
        quantities = component_data["Quantity"]
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
        
        return loss_ratios, total_loss_storey, total_loss_storey_ratio, total_replacement_cost, repair_cost, total_repair_cost
    
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
                return (e*x**a/(b**a + x**a) + (1-e)*x**c/(d**c + x**c))
        else:
            raise ValueError("[EXCEPTION] Wrong type of regression function")
        
        losses_fitted = {}
        for q in percentiles:
            popt, pcov = curve_fit(fitting_function, edp_range, losses["loss_ratio_curve"].loc[q], maxfev=10**6)
            losses_fitted[q] = fitting_function(edp_range, *popt)
            
        popt, pcov = curve_fit(fitting_function, edp_range, losses["loss_ratio_curve"].loc['mean'], maxfev=10**6)
        losses_fitted['mean'] = fitting_function(edp_range, *popt)
        
        return losses, losses_fitted
    
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
            slfs[key] = outputs[key]['edp_dv_fitted']['mean'] / total_story_cost
            
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
                    loss_ratios, total_loss_storey, total_loss_ratio, total_replacement_cost, repair_cost, total_repair_cost = \
                        self.calculate_loss(component_data, damage_state, means_cost, covs_cost)
                    losses, losses_fitted = self.perform_regression(total_loss_storey, total_loss_ratio, edp)
                    edp_dv_functions = self.get_in_euros(losses_fitted, total_replacement_cost)
                    outputs[group] = {'component': component_data, 'correlation_tree': matrix, 'fragilities': fragilities,
                                       'damage_states': damage_state, 'losses': losses, 'edp_dv_fitted': losses_fitted,
                                       'edp_dv_euro': edp_dv_functions, 'total_replacement_cost': total_replacement_cost, 
                                       'total_loss_storey': total_loss_storey, "total_repair_cost": total_repair_cost}
                    
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
    slf = SLF("nspfa_p", "nspfa_inv.csv", "nspfa_corr.csv", correlation="Correlated", n_realizations=3200,
              sflag=True, do_grouping=False, regression="Papadopoulos")
    run_master = True
    if run_master:
        outputs = slf.master(sensitivityflag=False)
    else:
        component_data = slf.get_component_data()
        component_groups = slf.group_components(component_data)
        items_per_group = []
        for key in component_groups.keys():
            items_per_group.append(len(component_groups[key]))
            
        cnt = 0
        ouputs = {}
        for group in component_groups:
            if not component_groups[group].empty:
                component_data = component_groups[group]
                matrix = slf.get_correlation_tree(component_data)
                edp = group[0:3]
                slf.get_edp_range(edp)
                
                if cnt == 0:
                    n_items = 0
                else:
                    n_items = sum(items_per_group[:cnt])
                
                fragilities, means_cost, covs_cost = slf.derive_fragility_functions(component_data, n_items)
                damage_probs = slf.get_DS_probs(component_data, fragilities)
                damage_state = slf.perform_Monte_Carlo(fragilities, matrix)
                damage_state = slf.test_correlated_data(damage_state, matrix, fragilities)
                loss_ratios, total_loss_storey, total_loss_ratio, total_loss, repair_cost = slf.calculate_loss(component_data,
                                                                                                 damage_state,
                                                                                                 means_cost, covs_cost)
                losses, losses_fitted = slf.perform_regression(total_loss_storey, total_loss_ratio, edp)
#                edp_dv_functions = slf.get_in_euros(losses_fitted, total_loss)

                cnt += 1
    
                if cnt == 1:
                    break
