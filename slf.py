"""
Storey-loss-function (SLF) Generator adopted from the work of Sebastiano Zampieri

The tool allows the automatic production of EDP-DV SLFs based on input fragility, consequence and quantity data.

FEMA P-58 for fragilities:      https://femap58.atcouncil.org/reports
For consequence functions:      https://femap58.atcouncil.org/reports
Python GUI for reference:       https://blog.resellerclub.com/the-6-best-python-gui-frameworks-for-developers/
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
    def __init__(self, project_name, component_data_filename, correlation_tree_filename, edp="IDR", edp_bin=0.1,
                 correlation="Correlated", n_realizations=20, sflag=False):
        """
        Initialization of the Master Generator
        :param project_name: str                Name of the project to save in the database
        :param component_data_filename: str     Component data file name, e.g. "*.csv"
        :param correlation_tree_filename: str   Correlation tree file name, e.g. "*.csv"
        :param edp: str                         Engineering demand parameter, "IDR" in % or "PFA" in g
        :param edp_bin: float                   EDP sampling unit, % for IDR, g for PFA (non-negative)
        :param correlation: str                 Whether the elements are "Independent" or "Correlated"
        :param n_realizations: int              Number of realizations
        :param sflag: bool                      Save data
        """
        self.dir = Path.cwd()
        self.database_dir = self.dir/"Database"
        self.project_name = project_name
        self.component_data_filename = component_data_filename
        self.correlation_tree_filename = correlation_tree_filename
        self.edp = edp
        self.edp_bin = edp_bin
        self.correlation = correlation
        self.n_realizations = n_realizations
        self.sflag = sflag
        
        # EDP range
        if self.edp == "IDR" or self.edp == "PSD":
            # stop calculation at 10% drift
            self.edp_bin = self.edp_bin/100
            self.edp_range = np.arange(0, 0.1 + self.edp_bin, self.edp_bin)
        elif self.edp == "PFA":
            # stop calculation at 2.5g of acceleration
            self.edp_range = np.arange(0, 2.5 + self.edp_bin, self.edp_bin)
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
        :return: DataFrame                  DataFrame containing the component data
        """
        component_data = pd.read_csv(self.dir/"client"/self.component_data_filename)
        if len(component_data.keys()) != 23:
            raise ValueError("[EXCEPTION] Unexpected number of features in the components DataFrame")
            
        return component_data

    def get_correlation_tree(self, component_data):
        """
        Gets the correlation tree and generates the correlation tree matrix
        TODO: for UI update to be updated similar to the component data
        :param component_data: DataFrame    DataFrame containing the component data
        :return: ndarray                    Correlation tree matrix
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
        if len(correlation_tree.keys()) != 8:
            raise ValueError("[EXCEPTION] Unexpected number of features in the correlations DataFrame")
            
        for item in correlation_tree.index:
            for feature in correlation_tree.keys():
                ds = str(correlation_tree.loc[item][feature])
                if ds == max_DS[item]:
                    raise ValueError("[EXCEPTION] MIN DS assigned in the correlation tree must not exceed the possible "
                                     "DS defined for the element")
        
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

    def derive_fragility_functions(self, component_data):
        """
        Derives fragility functions
        :param component_data: DataFrame        DataFrame containing the component data
        :return: dict                           Fragilities of all components at all DSs
        """
        # Fragility parameters
        means_fr = np.zeros((len(component_data), 5))
        covs_fr = np.zeros((len(component_data), 5))
        
        # Consequence parameters
        means_cost = np.zeros((len(component_data), 5))
        covs_cost = np.zeros((len(component_data), 5))
        
        # Deriving fragility functions
        data = component_data.values[:, 3:]
        for item in component_data.index:
            for ds in range(5):
                means_fr[item][ds] = data[item][ds]
                covs_fr[item][ds] = data[item][ds+5]
                means_cost[item][ds] = data[item][ds+10]
                covs_cost[item][ds] = data[item][ds+15]
                
        # Deriving the ordinates of the fragility functions
        fragilities = {"EDP": self.edp_range, "ITEMs": {}}
        for item in component_data.index:
            fragilities["ITEMs"][item+1] = {}
            for ds in range(5):
                mean = np.exp(np.log(means_fr[item][ds])-0.5*np.log(covs_fr[item][ds]**2+1))
                std = np.log(covs_fr[item][ds]**2+1)**0.5
                if mean == 0 and std == 0:
                    fragilities["ITEMs"][item+1][f"DS{ds+1}"] = np.zeros(len(self.edp_range))
                else:
                    frag = stats.norm.cdf(np.log(self.edp_range/mean)/std, loc=0, scale=1)
                    frag[np.isnan(frag)] = 0
                    fragilities["ITEMs"][item+1][f"DS{ds+1}"] = frag
        
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
            for item in range(num_items):
                damage_probs[item+1] = {}
                # DS
                for ds in range(6):
                    y = fragilities["ITEMs"][item+1]
                    if ds == 0:
                        damage_probs[item+1][f"DS{ds}"] = 1 - y[f"DS{ds+1}"]
                    elif ds == 5:
                        damage_probs[item+1][f"DS{ds}"] = y[f"DS{ds}"]
                    else:
                        damage_probs[item+1][f"DS{ds}"] = y[f"DS{ds}"] - y[f"DS{ds+1}"] 
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
            for item in range(num_items):
                damage_state[item+1] = {}
                # Simulations
                for n in range(self.n_realizations):
                    # Generate random data between (0, 1)
                    random_array = np.random.rand(num_edp)
                    damage = np.zeros(num_edp)
                    # DS
                    for ds in range(4, -1, -1):
                        y = fragilities["ITEMs"][item+1][f"DS{ds+1}"]
                        damage = np.where(random_array >= y, ds_range[ds], damage)
                    damage_state[item+1][n] = damage
            return damage_state
        
        elif self.correlation == "Correlated":
            if corr_tree is None:
                raise ValueError("[EXCEPTION] Correlation matrix is missing")
                
            for item in range(num_items):
                damage_state[item+1] = {}
                for n in range(self.n_realizations):
                    if matrix[item][0] == item + 1:
                        random_array = np.random.rand(num_edp)
                        damage = np.zeros(num_edp)
                        for ds in range(4, -1, -1):
                            y = fragilities["ITEMs"][item+1][f"DS{ds+1}"]
                            damage = np.where(random_array >= y, ds_range[ds], damage)
                        damage_state[item+1][n] = damage
                    else:
                        # -1 to indicate no assignment to a final DS to sub correlated elements
                        damage_state[item+1][n] = np.zeros(num_edp) - 1
                        
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
        
        # Star the iterations
        min_ds = {}
        y_new = {}
        while test != 0:
            iteration += 1
            
            # Items
            for item in range(num_items):
                min_ds[item+1] = {}
                y_new[item+1] = {}
                
                # TODO, remove iteration on EDP, not elegant
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
                            if damage_state[matrix[item][0]][n][edp] == -1:
                                # The marker will make the engine simulate the DS at the successive iteration
                                damage_state[item+1][n][edp] = -1
                            else:
                                # Finds the minimum DS for the i-th element
                                min_ds[item+1][edp][n] = matrix[item][1+int(damage_state[matrix[item][0]][n][edp])]
                                # Recalculates the probability of having each DS in the condition of having a min DS
                                # Damage states
                                for ds in range(5):
                                    # All DS smaller than min_DS have a probability of 1 of being observed
                                    # TODO, check if it is ds+1 or ds
                                    if ds + 1 <= min_ds[item+1][edp][n]:
                                        # probability of having DS >= min_k
                                        y_new[item+1][edp][n][ds] = 1
                                        # if min_DS is zero then the probabilities are unchanged
                                    elif min_ds[item+1][edp][n] == 0:
                                        y = fragilities["ITEMs"][item+1][f"DS{ds+1}"]
                                        y_new[item+1][edp][n][ds] = y[edp]
                                    else:
                                        # conditional probability of having DS >= DS_k given min_DS
                                        y = fragilities["ITEMs"][item+1][f"DS{ds+1}"]
                                        y1 = fragilities["ITEMs"][item+1][f"DS{int(min_ds[item+1][edp][n])+1}"]
                                        y_new[item+1][edp][n][ds] = y[edp]/y1[edp]
                                        if math.isnan(y_new[item+1][edp][n][ds]):
                                            y_new[item+1][edp][n][ds] = 0
                                            
                                # Simulates the DS at the given EDP, for the new set of probabilities
                                rand_value = np.random.rand(1)[0]
                                for ds in range(4, -1, -1):
                                    if rand_value >= y_new[item+1][edp][n][ds]:
                                        damage_state[item+1][n][edp] = ds_range[ds]
            
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
        num_items = len(damage_state)
        repair_cost = {}
        for item in range(num_items):
            repair_cost[item+1] = {}
            for n in range(self.n_realizations):
                for ds in range(6):
                    if ds == 0:
                        repair_cost[item+1][n] = np.where(damage_state[item+1][n] == ds, ds, -1)
                    
                    else:
                        idx_list = np.where(damage_state[item+1][n] == ds)[0]
                        for idx in idx_list:
                            a = np.random.normal(means_cost[item][ds-1], covs_cost[item][ds-1]*means_cost[item][ds-1])
                            while a < 0:
                                a = np.random.normal(means_cost[item][ds-1], covs_cost[item][ds-1]*means_cost[item][ds-1])
                            repair_cost[item+1][n][idx] = a
        
        # Evaluate the total damage cost multiplying the individual cost by each element quantity
        quantities = component_data["Quantity"]
        total_repair_cost = {}                              # Total repair costs
        replacement_cost = {}                               # Replacement costs
        loss_ratios = {}                                    # Loss ratios
        for item in range(num_items):
            total_repair_cost[item+1] = {}
            replacement_cost[item+1] = max(means_cost[item])
            loss_ratios[item+1] = {}
            for n in range(self.n_realizations):
                total_repair_cost[item+1][n] = repair_cost[item+1][n]*quantities[item]
                loss_ratios[item+1][n] = repair_cost[item+1][n] / replacement_cost[item+1]
        
        # Evaluate total loss for the floor segment
        total_loss_storey = {}
        for n in range(self.n_realizations):
            total_loss_storey[n] = np.zeros(len(self.edp_range))
            for item in range(num_items):
                total_loss_storey[n] += total_repair_cost[item+1][n]
        
        # Evaluate total loss ratio for the floor segment
        replacement_cost = np.array([replacement_cost[i] for i in replacement_cost])
        total_replacement_cost = sum(np.array(quantities)*replacement_cost)
        
        total_loss_ratio = {}
        for n in range(self.n_realizations):
            total_loss_ratio[n] = total_loss_storey[n]/total_replacement_cost
        
        return loss_ratios, total_loss_storey, total_loss_ratio
    
    def perform_regression(self, total_loss_storey, total_loss_ratio, percentiles=None):
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
        
        # Setting the edp range
        if self.edp == "IDR" or self.edp == 'PSD':
            edp_range = self.edp_range*100
        else:
            edp_range = self.edp_range
                    
        ''' Fitting the curve, EDP-DV functions, assuming Weibull Distribution '''
        def fitting_function(x, a, b, c):
            return a*(1 - np.exp(-(x/b)**c))
        
        losses_fitted = {}
        for q in percentiles:
            popt, pcov = curve_fit(fitting_function, edp_range, losses["loss_ratio_curve"].loc[q])
            losses_fitted[q] = fitting_function(edp_range, *popt)
        
        return losses, losses_fitted
    
    
if __name__ == "__main__":
    """
    Takes as input:
    Component data              .csv file including quantity and fragility information for each component
    Correlation tree            .csv file containing the correlation tree
    edp_bin                     Step of edp
    nrealizations               number of simulations per edp
    """

    slf = SLF("Case1", "component_data.csv", "correlation_tree.csv", correlation="Correlated")
    component_data = slf.get_component_data()
    matrix = slf.get_correlation_tree(component_data)
    fragilities, means_cost, covs_cost = slf.derive_fragility_functions(component_data)
    damage_probs = slf.get_DS_probs(component_data, fragilities)
    damage_state = slf.perform_Monte_Carlo(fragilities, matrix)
    damage_state = slf.test_correlated_data(damage_state, matrix, fragilities)
    loss_ratios, total_loss_storey, total_loss_ratio = slf.calculate_loss(component_data, damage_state, means_cost, covs_cost)
    losses, losses_fitted = slf.perform_regression(total_loss_storey, total_loss_ratio)
