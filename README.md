### Storey-loss-function (SLF) Generator

TODO: Remove heavy libraries for the installer

The tool allows the automated production of SLFs based on input fragility, consequence and quantity data.

Considerations for double counting should be done at the input level and the consequence function should mirror it.

Running: Run slf_gui.py.

Required libraries: tkinter, PIL, pandas, random, matplotlib, numpy, scipy, math


[SLF estimation procedure](https://www.researchgate.net/publication/265411359_Building-Specific_Loss_Estimation_Methods_Tools_for_Simplified_Performance-Based_Earthquake_Engineering)

[FEMA P-58 for fragilities](https://femap58.atcouncil.org/reports)

[Consequence functions](https://femap58.atcouncil.org/reports)

[Python GUI for reference](https://blog.resellerclub.com/the-6-best-python-gui-frameworks-for-developers/)

[Repair costs - Central Italy](https://sisma2016.gov.it/wp-content/uploads/2019/12/Allegato-3-Prezzario-Cratere_2018-Finale.pdf)

**Acronyms**

EDP:    Engineering Demand Parameter

DV:     Decision Variable

DS:     Damage State

CDF:    Cumulative distribution function


**Input arguments**

* Project name
* .csv file containing component data
* .csv file containing the correlation tree
* Correlation type, i.e. Correlated or Independent
* Number of simulations, i.e. Monte Carlo simulations to generate damage states for analysis
* EDP type, i.e. PSD or PFA
* EDP step, i.e. % for PSD and g for PFA
* Flag to store the results in the Database

Note: *The tool relies on the accuracy of the user's provided data, it does not offer its own component information, 
therefore double counting or dependency of different component fragilities should be accounted for by the user, as the 
tool will work either way.*

**Step-by-step procedure**

The tool relies on three performance groups, that is
* Drift-sensitive structural elements
* Drift-sensitive non-structural elements
* Acceleration-sensitive non-structural elements

1. Read component data ← *component data*

    	OUTPUT: Component fragility functions
    	OUTPUT: Component consequence functions
    	OUTPUT: Component quantities
    	
2. Obtain the correlation matrix ← *component data*

        OUTPUT: Correlation matrix based on the correlation tree provided
        
3. Derive fragility and consequence functions ← *component data*

        OUTPUT: Fragility functions, i.e. lognormal CDF based on provided mean and dispersion
        OUTPUT: Consequence functions, i.e. mean and covariance of repair costs
        
4. Perform Monte Carlo simulation to generate data for analysis ← *fragility functions*<br/>
*!note - for each type of item **n** simulations are performed to generate random data between 0 and 1 with a 
predefined length matching the EDP range. The generated array is checked against the fragility of each DS of the given 
item and DS is assigned.*
        
        OUTPUT: Matrix of DS for all items and simulations
        
5. DS matrix populated for the dependent components ← *Damage states, correlation tree*

        OUTPUT: Re-populated matrix of dependent components at all Monte Carlo simulations
        
6. Evaluate the repair costs on the individual component at each EDP level for each simulation ← *component data, Damage 
States, consequence functions*
    
    6.1. Create the repair cost matrix
    
        6.1.1. For each item
        6.1.2. For each simulation
        6.1.3. For each DS (i.e. 5)
                
                Assign 0 repair cost, where DS 0 is recorded
                Otherwise assign repair cost -1 as a placeholder for later filling
                
                Parse for other DS (i.e. 1, 2, 3, etc.)
                Populate the repair cost matrix with a cost generated as a random normal/lognormal with repair cost mean and 
                covariance of a corresponding DS
                
                OUTPUT: repair cost matrix, dimensions(item, simulation, EDP range)
    
    6.2. Evaluate the total damage cost multiplying the individual cost by each element quantity
    
        6.2.1. For each item
        6.2.2. For each simulation
                Total repair cost is obtained as the product of repair cost of the given simulation and quantity of the item
                
                OUTPUT: Total repair cost matrix,  dimensions(item, simulation)
                
    6.3. Evaluate total loss for each story segment
    
        6.3.1. For each simulation
        6.3.2. For each item
                Sum the total repair costs of all items obtained at the previous level 
        
                OUTPUT: Total story loss, dimensions(simulation)
                
    6.4. Evaluate total loss ratios
    
        OUTPUT: Total story loss ratio as the quotient of total story loss and the input Replacement Cost
                
7. Perform regression ← *total story loss, total story loss ratio, edp type, percentiles (e.g. 0.16, 0.50, 0.84) for evaluation*

        OUTPUT: Quantiles of story losses and story loss ratios
        
        Sample fitting function used for SLF generation
        
    <img src="https://latex.codecogs.com/svg.latex?\Large&space;y=\alpha*(1-\exp(-(\frac{x}{\beta})^\gamma))" title="eq.1" /><br/>

        where α, β, γ are the fitting coefficients, x is the EDP range and y is the fitted SLF functions
        
        OUTPUT: Fitted SLF
        
8. Export outputs to .xlsx and cache to .pickle if specified
