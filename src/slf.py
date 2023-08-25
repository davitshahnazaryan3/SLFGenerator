from typing import List, Union
import pandas as pd
import numpy as np
import math
from scipy import stats
from scipy.optimize import curve_fit
import warnings

from models import *

warnings.filterwarnings('ignore')


class SLF:
    """
    Storey-loss-function (SLF) Generator for Storey-Based Loss Assessment

    The tool allows the automatic production of SLFs based on input fragility,
    consequence and quantity data.

    Considerations for double counting should be done at the input level and
    the consequence function should mirror it.

    SLF estimation procedure:       Ramirez and Miranda 2009,
    CH.3 Storey-based building-specific loss estimation (p. 17)
    FEMA P-58 for fragilities:      https://femap58.atcouncil.org/reports
    For consequence functions:      https://femap58.atcouncil.org/reports

    EDP:    Engineering Demand Parameter
    DV:     Decision Variable
    DS:     Damage State
    """

    # Correlation tree matrix
    matrix = None
    component_groups = None
    NEGLIGIBLE = 1e-20

    def __init__(
        self,
        component_data: ComponentDataModel,
        edp: str,
        correlation_tree: CorrelationTreeModel = None,
        component: List[str] = None,
        edp_range: List[float] = None,
        edp_bin: float = None,
        do_grouping: bool = True,
        conversion: float = 1.0,
        realizations: int = 20,
        replacement_cost: float = 1.0,
        regression: str = "Weibull",
        storey: Union[int, List[int]] = None,
        directionality: int = None,
    ):
        """initialize storey-loss function (SLF) generator
        TODO, add option for mutually exclusive damage states
        TODO, include possibility of including quantity uncertainties along
        with the mean values

        Parameters
        ----------
        component_data : ComponentDataModel
            Component data inventory
        edp : str
            EDP type:
                'PSD' = peak storey drift
                'PFA' = peak floor acceleration
        correlation_tree: CorrelationTreeModel
            Correlation tree of component data, by default None
        component : List[str], optional
            Component type, for example, ["ns", "s"], by default None
            where "ns" stands for non-structural and "s" stands for structural
        edp_range : List[float], optional
            EDP range, by default None
        edp_bin : float, optional
            EDP bin size, by default None
        do_grouping : bool, optional
            Perform performance grouping of components or not, by default True
        conversion : float, optional
            Conversion factor from usd to euro, by default 1.0,
            Example: if provided in euro, use 1.0;
                if 1 usd = 0.88euro, use 0.88
                or use 1.0 if ratios are used directly
                However, euro is just a convention, it can be any currency
        realizations : int, optional
            Number of realizations for Monte Carlo method, by default 20
        replacement_cost : float, optional
            Replacement cost of the building (used when normalizing the SLFs),
            by default 1.0
        regression : str, optional
            Regression function to be used,
            currently supports: 'Weibull', 'Papadopoulos', by default 'Weibull'
        storey : Union[int, List[int]], optional
            Storey levels, by default None
        directionality : int, optional
            Directionality, by default None (None means non-directional)
        """
        self.component_data = component_data.applymap(
            lambda s: s.lower() if type(s) == str else s)
        self.edp = edp.lower()
        self.component = component
        self.correlation_tree = correlation_tree
        self.edp_bin = edp_bin
        self.edp_range = edp_range
        self.do_grouping = do_grouping
        self.conversion = conversion
        self.realizations = realizations
        self.replacement_cost = replacement_cost
        self.regression = regression.lower()
        self.storey = storey
        self.directionality = directionality

        # Get EDP range
        self._define_edp_range()
        # Component inventory
        self._get_component_data()

        # Component correlation tree
        if self.correlation_tree is not None:
            self.correlation_tree.applymap(
                lambda s: s.lower() if type(s) == str else s)
            self._get_correlation_tree()

        # Grouping components
        self._group_components()

    def _define_edp_range(self):
        """Define range of engineering demand parameters (EDP)

        Raises
        ------
        ValueError
            If incorrect EDP type is provided, must be 'psd' or 'pfa'
        """
        if self.edp == "idr" or self.edp == "psd":
            self.edp_bin = 0.1 / 100
            self.edp_range = np.arange(0, 0.2 + self.edp_bin, self.edp_bin)
        elif self.edp == "pfa":
            self.edp_bin = 0.05
            self.edp_range = np.arange(0, 10. + self.edp_bin, self.edp_bin)
        else:
            raise ValueError("Wrong EDP type, must be 'PSD' or 'PFA'")
        self.edp_range[0] = self.NEGLIGIBLE

    def _get_component_data(self):
        """Gets component information from the user provided .csv file

        Direct manipulation within the .csv file, add new entries with empty
        IDs (the tool will assign the IDs automatically) or select ID manually.
        Newly created entries will not be saved within the database, and will
        be deleted if the .csv file is modified.
        """
        # Validate component data
        self._validate_component_data_schema()

        # Number of items
        n_items = len(self.component_data)

        # Check for any missing values and fill with defaults
        for key in self.component_data.keys():
            # 'best fit' features
            if key.endswith('best fit'):
                self.component_data[key].fillna('normal', inplace=True)
            if key == "ITEM":
                self.component_data[key].fillna(
                    pd.Series(np.arange(1, n_items + 1, 1), dtype='int'),
                    inplace=True)
            if key == "ID":
                self.component_data[key].fillna("B", inplace=True)

        # Replace all nan with 0.0 for the rest of the DataFrame,
        # except for Group and Component
        self.component_data[self.component_data.columns.difference(
            ["Group", "Component"])] = \
            self.component_data[self.component_data.columns.difference(
                ["Group", "Component"])].fillna(0, inplace=False)

    def _group_components(self):
        """Component performance grouping
        """
        groups = np.array(self.component_data["Group"])
        components = np.array(self.component_data["Component"])

        if not self.do_grouping:
            if components.dtype != "O":
                # Populate with a placeholder
                self.component_data["Component"].fillna("-1", inplace=True)
            if np.isnan(groups).any():
                # Populate with a placeholder
                self.component_data["Group"].fillna(-1, inplace=True)

            # If no performance grouping is done, the EDP value is assigned
            # as the default group tag
            key = self.component_data["EDP"].iloc[0]
            self.component_groups = {key: self.component_data}

            return

        # Check if grouping was assigned
        # Group not assigned
        if np.isnan(groups).any():
            # Populate with a placeholder
            self.component_data["Group"].fillna(-1, inplace=True)

            # Only EDP was assigned (Component and Group unassigned)
            if components.dtype != "O":
                # Populate with a placeholder
                self.component_data["Component"].fillna("-1", inplace=True)
                # Select unique EDPs
                unique_edps = self.component_data.EDP.unique()
                self.component_groups = {}
                for group in unique_edps:
                    self.component_groups[group] = self.component_data[(
                        self.component_data["EDP"] == group)]

            # EDP and Component assigned
            else:
                psd_s = self.component_data[
                    (self.component_data["EDP"] == "psd") & (
                        self.component_data["Component"] == "s")]
                psd_ns = self.component_data[
                    (self.component_data["EDP"] == "psd") & (
                        self.component_data["Component"] == "ns")]
                pfa_ns = self.component_data[
                    (self.component_data["EDP"] == "pfa") & (
                        self.component_data["Component"] == "ns")]

                self.component_groups = {"PSD, S": psd_s,
                                         "PSD, NS": psd_ns, "PFA, NS": pfa_ns}

        # Group is assigned
        else:
            if components.dtype != "O":
                # Populate with a placeholder
                self.component_data["Component"].fillna("-1", inplace=True)

            unique_groups = np.unique(groups)
            self.component_groups = {}
            for group in unique_groups:
                self.component_groups[group] = self.component_data[(
                    self.component_data["Group"] == group)]

    def _get_correlation_tree(self) -> np.ndarray[int]:
        """Get correlation tree from .csv file

        Notes on creation of a correlation tree: Make sure that the MIN DS
        assigned does not exceed the possible Damage
        States (DS) defined for the element. Essentially, if a component is
        dependent on another component and has only one
        DS, i.e. DS1, which occurs only if the causation element is for
        example at its DS3, then the following would be
        true. And Item3 would be damaged earlier and has more DSs.
        The software will not detec any errors, so it depends
        on the user.

        Updates
        ----------
        matrix: np.ndarray [number of components x
                            (number of damage states + 2)]
            Correlation table, relationships between Item IDs

        Examples
        ----------
            +------------+-------------+-------------+-------------+
            | Item ID    |Dependant on | MIN DS|DS0  | 	MIN DS|DS1 |
            +============+=============+=============+=============+
            | Item 1     | Independent | Independent | Independent |
            +------------+-------------+-------------+-------------+
            | Item 2     | 1           | Undamaged   | Undamaged   |
            +------------+-------------+-------------+-------------+
            | Item 3     | 1           | Undamaged   | Undamaged   |
            +------------+-------------+-------------+-------------+

            continued...

            +-------------+-------------+-------------+-------------+
            |  MIN DS|DS2 |  MIN DS|DS3 | MIN DS|DS4  | MIN DS|DS5  |
            +=============+=============+=============+=============+
            | Independent | Independent | Independent | Independent |
            +-------------+-------------+-------------+-------------+
            | Undamaged   | DS1         | DS1         | DS1         |
            +-------------+-------------+-------------+-------------+
            | DS1         | DS2         | DS3         | DS3         |
            +-------------+-------------+-------------+-------------+
        """

        # Get possible maximum DS
        damage_states = list(self.component_data['Damage States'])

        # Select the items within the component performance group
        correlation_tree = self.correlation_tree.loc[self.component_data.index]

        # Validate correlation tree
        self._validate_correlation_tree_schema(damage_states)

        # Create the correlation matrix
        items = correlation_tree.values[:, 0]
        c_tree = np.delete(correlation_tree.values, 0, 1)
        self.matrix = np.zeros(c_tree.shape, dtype=int)

        for j in range(c_tree.shape[1]):
            for i in range(c_tree.shape[0]):
                if j == 0:
                    if c_tree[i][j].lower() == "independent":
                        self.matrix[i][j] = items[i]
                    elif items[i] == "" or math.isnan(items[i]):
                        self.matrix[i][j] = np.nan
                    else:
                        self.matrix[i][j] = c_tree[i][j]
                else:
                    if math.isnan(self.matrix[i][j - 1]):
                        self.matrix[i][j] = np.nan
                    elif c_tree[i][j].lower() == "independent":
                        self.matrix[i][j] = 0
                    elif c_tree[i][j].lower() == "undamaged":
                        self.matrix[i][j] = 0
                    else:
                        self.matrix[i][j] = int(c_tree[i][j][-1])

    def _validate_component_data_schema(self):
        columns = list(self.component_data.columns)
        component_data = self.component_data.to_dict(orient='records')

        # Validate base fields
        id_set = set()
        for row in component_data:
            model = ComponentDataModel.parse_obj(row)
            if model.ITEM is not None and model.ITEM in id_set:
                raise ValueError(f'Duplicate ITEM: {model.ITEM}')
            id_set.add(model.ITEM)

        counts = {
            "Median Demand": 0,
            "Total Dispersion (Beta)": 0,
            "Repair COST": 0,
            "COST Dispersion (Beta)": 0,
            "best fit": 0,
        }

        for col in columns:
            for key in counts.keys():
                if col.endswith(key):
                    counts[key] += 1

        total_count = counts["Median Demand"]
        for key in counts.keys():
            if total_count != counts[key]:
                raise ValueError(
                    "There must be equal amount of columns: 'Median Demand', "
                    "'Total Dispersion (Beta), 'Repair COST', "
                    "'COST Dispersion (Beta)', 'best fit")

    def _validate_correlation_tree_schema(self, damage_states):
        corr_dict = self.correlation_tree.to_dict(orient='records')

        # Validate base fields
        id_set = set()
        for row in corr_dict:
            model = CorrelationTreeModel.parse_obj(row)
            if model.ITEM in id_set:
                raise ValueError(f'Duplicate ITEM: {model.ITEM}')
            id_set.add(model.ITEM)

        # Check integrity of the provided input correlation table
        if len(self.correlation_tree.keys()) < max(damage_states) + 3:
            raise ValueError(
                "[EXCEPTION] Unexpected (fewer) number of features "
                "in the correlations DataFrame")

        # Verify integrity of the provided correlation tree
        idx = 0
        for item in self.component_data.index:
            for feature in self.correlation_tree.keys():
                ds = str(self.correlation_tree.loc[item][feature])
                if ds == f'DS{damage_states[idx] + 1}':
                    raise ValueError("[EXCEPTION] MIN DS assigned in "
                                     "the correlation tree must not exceed "
                                     "the possible DS defined for the element")
            idx += 1

        # Check that dimensions of the correlation tree
        # and the component data match
        if len(self.component_data) != len(self.correlation_tree):
            raise ValueError(
                "[EXCEPTION] Number of items in the correlation tree "
                "and component data should match")

    def fragility_function(
        self,
    ) -> tuple[FragilityModel, np.ndarray, np.ndarray]:
        """Derives fragility functions

        Returns
        -------
        dict, FragilityModel
            Fragility functions associated with each damage state and component
        np.ndarray (number of components, number of damage states)
            Mean values of cost functions
        np.ndarray (number of components, number of damage states)
            Covariances of cost functions
        """
        # Get all DS columns
        n_ds = 0
        for column in self.component_data.columns:
            if column.endswith("Median Demand"):
                n_ds += 1

        # Fragility parameters
        means_fr = np.zeros((len(self.component_data), n_ds))
        covs_fr = np.zeros((len(self.component_data), n_ds))

        # Consequence parameters
        means_cost = np.zeros((len(self.component_data), n_ds))
        covs_cost = np.zeros((len(self.component_data), n_ds))

        # Deriving fragility functions
        data = self.component_data.select_dtypes(exclude=['object']).drop(
            labels=['ITEM', 'Group', 'Quantity', 'Damage States'], axis=1,
        ).values

        # Get parameters of the fragility and consequence functions
        for item in range(len(data)):
            for ds in range(n_ds):
                means_fr[item][ds] = data[item][ds]
                covs_fr[item][ds] = data[item][ds + n_ds]
                means_cost[item][ds] = data[item][
                    ds + 2 * n_ds] * self.conversion
                covs_cost[item][ds] = data[item][ds + 3 * n_ds]

        # Deriving the ordinates of the fragility functions
        fragilities = {"EDP": self.edp_range, "ITEMs": {}}
        for item in range(len(data)):
            fragilities["ITEMs"][item + 1] = {}
            for ds in range(n_ds):
                if means_fr[item][ds] == 0:
                    fragilities["ITEMs"][
                        item + 1][f"DS{ds + 1}"] \
                        = np.zeros(len(self.edp_range))
                else:
                    mean = np.exp(
                        np.log(means_fr[item][ds])
                        - 0.5 * np.log(covs_fr[item][ds] ** 2 + 1))
                    std = np.log(covs_fr[item][ds] ** 2 + 1) ** 0.5
                    frag = stats.norm.cdf(
                        np.log(self.edp_range / mean) / std, loc=0, scale=1)
                    frag[np.isnan(frag)] = 0
                    fragilities["ITEMs"][item + 1][f"DS{ds + 1}"] = frag

        return fragilities, means_cost, covs_cost

    def perform_monte_carlo(
        self, fragilities: FragilityModel
    ) -> DamageStateModel:
        """Performs Monte Carlo simulations and simulates damage state(DS) for
        each engineering demand parameter (EDP) value

        Parameters
        ----------
        fragilities : FragilityModel
            Fragility functions of all components at all DSs

        Returns
        ----------
        DamageStateModel
            Sampled damage states of each component for each simulation
        """
        # Number of damage states
        n_ds = len(fragilities['ITEMs'][1])
        ds_range = np.arange(0, n_ds + 1, 1)

        damage_state = dict()

        # Evaluate the DS on the i-th component for EDPs at the n-th simulation
        for item in fragilities['ITEMs']:
            damage_state[item] = dict()

            # Simulations
            for n in range(self.realizations):
                random_array = np.random.rand(len(self.edp_range))
                damage = np.zeros(len(self.edp_range), dtype=int)

                # For each DS
                for ds in range(n_ds, 0, -1):
                    y1 = fragilities["ITEMs"][item][f"DS{ds}"]
                    if ds == n_ds:
                        damage = np.where(random_array <= y1,
                                          ds_range[ds], damage)
                    else:
                        y = fragilities["ITEMs"][item][f"DS{ds + 1}"]
                        damage = np.where((random_array >= y) & (
                            random_array < y1), ds_range[ds], damage)
                damage_state[item][n] = damage

        return damage_state

    def enforce_ds_dependent(
        self, damage_state: DamageStateModel
    ) -> DamageStateModel:
        """Enforces new DS for each dependent component

        Parameters
        ----------
        damage_state : DamageStateModel
            Sampled damage states of each component for each simulation

        Returns
        ----------
        DamageStateModel
            Sampled DS of each component for each simulation after enforcing
            DS for dependent components if a correlation matrix is provided
        """
        if self.correlation_tree is None:
            return damage_state

        # Loop over each component
        for i in range(self.matrix.shape[0]):
            # Check if component is dependent or independent
            if i + 1 != self.matrix[i][0]:
                # -- Component is dependent
                # Causation component ID
                m = self.matrix[i][0]
                # Dependent component ID
                j = i + 1
                # Loop for each simulation
                for n in range(self.realizations):
                    causation_ds = damage_state[m][n]
                    correlated_ds = damage_state[j][n]

                    # Get dependent components DS conditioned
                    # on causation component
                    temp = np.zeros(causation_ds.shape)
                    # Loop over each DS
                    for ds in range(1, self.matrix.shape[1]):
                        temp[causation_ds == ds - 1] = self.matrix[j - 1][ds]

                    # Modify DS if correlated component is conditioned on
                    # causation component's DS, otherwise skip
                    damage_state[j][n] = np.maximum(correlated_ds, temp)

        return damage_state

    def calculate_costs(
        self,
        damage_state: DamageStateModel,
        means_cost: np.ndarray,
        covs_cost: np.ndarray,
    ) -> tuple[CostModel, CostModel, SimulationModel]:
        """Evaluates the damage cost on the individual i-th component at each
        EDP level for each n-th simulation

        Parameters
        ----------
        damage_state : DamageStateModel
            Sampled damage states
        means_cost : np.ndarray (number of components, number of damage states)
            Mean values of cost functions
        covs_cost : np.ndarray (number of components, number of damage states)
            Covariances of cost functions

        Returns
        ----------
        CostModel
            Total replacement costs in absolute values
        CostModel
            Total replacement costs as a ratio of replacement cost
        SimulationModel
            Repair costs associated with each component and simulation
        """
        # Number of damage states
        num_ds = means_cost.shape[1]

        repair_cost = {}
        idx = 0
        for item in damage_state.keys():
            repair_cost[item] = {}
            for n in range(self.realizations):
                for ds in range(num_ds + 1):
                    if ds == 0:
                        repair_cost[item][n] = np.where(
                            damage_state[item][n] == ds, ds, -1)

                    else:
                        # Best fit function
                        best_fit = \
                            self.component_data.iloc[
                                item - 1][f"DS{ds}, best fit"].lower()
                        # EDP ID where ds is observed
                        idx_list = np.where(damage_state[item][n] == ds)[0]
                        for idx_repair in idx_list:
                            if best_fit == 'normal truncated':
                                # TODO, Add options to truncate the
                                # distribution, add option to
                                # do multi-modal distribution
                                pass
                            elif best_fit == 'lognormal':
                                a = np.random.normal(means_cost[idx][ds - 1],
                                                     covs_cost[idx][ds - 1]
                                                     * means_cost[idx][ds - 1])
                                while a < 0:
                                    std = covs_cost[idx][ds - 1] * \
                                        means_cost[idx][ds - 1]
                                    m = np.log(
                                        means_cost[idx][ds - 1] ** 2
                                        / np.sqrt(means_cost[idx][ds - 1] ** 2
                                                  + std ** 2))
                                    std_log = np.sqrt(np.log(
                                        (means_cost[idx][ds - 1] ** 2
                                         + std ** 2)
                                        / means_cost[idx][ds - 1] ** 2))
                                    a = np.random.lognormal(m, std_log)
                            else:
                                a = np.random.normal(means_cost[idx][ds - 1],
                                                     covs_cost[idx][ds - 1]
                                                     * means_cost[idx][ds - 1])
                                while a < 0:
                                    a = np.random.normal(
                                        means_cost[idx][ds - 1],
                                        covs_cost[idx][ds - 1]
                                        * means_cost[idx][ds - 1])

                            repair_cost[item][n][idx_repair] = a
            idx += 1

        # Evaluate the total damage cost multiplying the individual
        # cost by each element quantity
        quantities = self.component_data["Quantity"]
        total_repair_cost = {}
        idx = 0
        for item in damage_state.keys():
            total_repair_cost[item] = {}
            for n in range(self.realizations):
                total_repair_cost[item][n] = repair_cost[item][n] * \
                    quantities.iloc[item - 1]
            idx += 1

        # Evaluate total loss for the storey segment
        total_loss_storey = {}
        for n in range(self.realizations):
            total_loss_storey[n] = np.zeros(len(self.edp_range))
            for item in damage_state.keys():
                total_loss_storey[n] += total_repair_cost[item][n]

        # Calculate if replCost was set to 0, otherwise use the provided value
        if self.replacement_cost == 0.0 or self.replacement_cost is None:
            raise ValueError(
                "Replacement cost should be a non-negative non-zero value.")
        else:
            total_replacement_cost = self.replacement_cost

        total_loss_storey_ratio = {}
        for n in range(self.realizations):
            total_loss_storey_ratio[n] = total_loss_storey[n] / \
                total_replacement_cost

        return total_loss_storey, total_loss_storey_ratio, repair_cost

    def perform_regression(
        self,
        loss: CostModel,
        loss_ratio: CostModel,
        percentiles: List[float] = None,
    ) -> tuple[LossModel, FittedLossModel, FittingParametersModel]:
        """Performs regression and outputs final fitted results as
        storey-loss functions (SLFs)

        Parameters
        ----------
        loss : CostModel
            Total loss for the floor segment in absolute values
        loss_ratio : CostModel
            Total loss for the floor segment as a ratio of replacement cost
        percentiles : List[float], optional
            Percentiles to estimate, by default [0.16, 0.50, 0.84],
            'mean' is always included

        Returns
        ----------
        LossModel
            Loss quantiles in terms of both absolute values and ratio
            to replacement cost
        FittedLossModel
            Fitted loss functions
        FittingParametersModel
            Fitting parameters or each quantiles and mean
        """
        if percentiles is None:
            percentiles = [0.16, 0.50, 0.84]

        # Into a DataFrame for easy access for manipulation
        loss = pd.DataFrame.from_dict(loss)
        loss_ratio = pd.DataFrame.from_dict(loss_ratio)

        losses = {"loss": loss.quantile(percentiles, axis=1),
                  "loss_ratio": loss_ratio.quantile(percentiles, axis=1)}

        mean_loss = np.mean(loss, axis=1)
        mean_loss_ratio = np.mean(loss_ratio, axis=1)
        losses["loss"].loc['mean'] = mean_loss
        losses["loss_ratio"].loc['mean'] = mean_loss_ratio

        # Setting the edp range
        if self.edp == "idr" or self.edp == 'psd':
            edp_range = self.edp_range * 100
        else:
            edp_range = self.edp_range

        # Fitting the curve, SLFs
        if self.regression == "weibull":
            def fitting_function(x, a, b, c):
                return a * (1 - np.exp(-(x / b) ** c))

        elif self.regression == "papadopoulos":
            def fitting_function(x, a, b, c, d, e):
                return e * x**a / (b**a + x**a) + \
                    (1 - e) * x**c / (d**c + x**c)

        else:
            raise ValueError(
                f"Regression type {self.regression} is not supported...")

        # Fitted loss functions at specified quantiles normalised
        # by the Replacement Cost
        losses_fitted = {}
        fitting_parameters = {}
        for q in percentiles:
            q_key = str(q)
            max_val = max(losses["loss_ratio"].loc[q])
            popt, pcov = curve_fit(
                fitting_function, edp_range,
                losses["loss_ratio"].loc[q] / max_val,
                maxfev=10**6)

            losses_fitted[q_key] = fitting_function(edp_range, *popt) * max_val
            # Truncating at zero to prevent negative values
            losses_fitted[q_key][losses_fitted[q_key] <= 0] = 0.0
            fitting_parameters[q_key] = {"popt": popt, "pcov": pcov}

        # Fitting the mean
        max_val = max(losses["loss_ratio"].loc['mean'])
        popt, pcov = curve_fit(fitting_function, edp_range,
                               losses["loss_ratio"].loc['mean'] / max_val,
                               maxfev=10**6)

        losses_fitted['mean'] = fitting_function(edp_range, *popt) * max_val

        fitting_parameters['mean'] = {"popt": popt, "pcov": pcov}

        return losses, losses_fitted, fitting_parameters

    def estimate_accuracy(
        self, y: np.ndarray, yhat: np.ndarray
    ) -> tuple[float, float]:
        """Estimate prediction accuracy

        Parameters
        ----------
        y : np.ndarray
            Observations
        yhat : np.ndarray
            Predictions

        Returns
        -------
        (float, float)
            Maximum error in %, and Cumulative error in %
        """
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        if not isinstance(yhat, np.ndarray):
            yhat = np.asarray(yhat)

        error_max = max(abs(y - yhat) / max(y)) * 100
        error_cum = self.edp_bin * sum(abs(y - yhat) / max(y)) * 100
        return error_max, error_cum

    def transform_output(
        self, losses_fitted: FittedLossModel, component: str = None
    ) -> SLFModel:
        """Transforms SLF output to primary attributes supported by
        Loss assessment module

        Parameters
        ----------
        losses_fitted : FittedLossModel
            Fitted loss functions

        Returns
        -------
        SLFModel
            SLF output
        """
        out = {
            'Directionality': self.directionality,
            'Component-type': component,
            'Storey': self.storey,
            'edp': self.edp,
            'edp_range': list(self.edp_range),
            'slf': list(losses_fitted['mean']),
        }

        return out

    def generate_slfs(self):
        """Genearte SLFs

        Returns
        -------
        Dict[SLFModel]
            SLFs per each performance group
        """
        out = {}

        for i, group in enumerate(self.component_groups):
            if self.component_groups[group].empty:
                continue

            if self.component is not None and \
                    len(self.component) == len(self.component_groups):
                component = self.component[i].lower()
            else:
                component = None

            # Select component inventory to analyze
            self.component_data = self.component_groups[group]

            # Obtain component fragility and consequence functions
            fragilities, means_cost, covs_cost = self.fragility_function()

            # Perform Monte Carlo simulations for damage state sampling
            damage_state = self.perform_monte_carlo(fragilities)

            # Populate the damage state matrix for correlated components
            damage_state = self.enforce_ds_dependent(damage_state)

            # Calculate the costs
            total, ratio, _ = self.calculate_costs(
                damage_state, means_cost, covs_cost)

            # Perform regression
            losses, losses_fitted, fitting_parameters = \
                self.perform_regression(total, ratio)

            # Compute accuracy
            error_max, error_cum = self.estimate_accuracy(
                losses["loss_ratio"].loc['mean'], losses_fitted['mean'])

            # Transform output
            out[str(group)] = self.transform_output(losses_fitted, component)

            out[str(group)]['error_max'] = error_max
            out[str(group)]['error_cum'] = error_cum

        return out
