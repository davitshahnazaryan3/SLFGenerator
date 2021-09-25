<h1 align="center">Storey-loss-function (SLF) Generator</h1> 

Performance-based earthquake engineering (PBEE) has become an important frame- work for quantifying seismic losses. 
However, due to its computationally expensive implementation through a typically detailed component-based approach (i.e. Federal Emergency Management Agency (FEMA) P-58), it has primarily been used within academic research and specific studies. 
A simplified alternative more desirable for practitioners is based on story loss functions (SLFs), which estimate a building’s expected monetary loss per story due to seismic demand. 

These simplified SLFs reduce the data required compared to a detailed study, which is especially true at a design stage, 
where detailed component information is likely yet to be defined. 
A Python-based toolbox for the development of user-specific and customizable SLFs for use within seismic design and assessment of buildings. 
Finally, a comparison of SLF-based and component-based loss estimation approaches is carried out through the application to a real case study school building. 
The tool was used within the reference publication, where the agreement and consistency of the attained loss metrics demonstrate the quality and ease of the SLF-based approach in achieving accurate results for a more expedite assessment of building performance.

The tool allows the automated production of SLFs based on input fragility, consequence and quantity data.

Considerations for double counting should be done at the input level and the consequence function should mirror it.

**Running the app**: Run slf_gui.py.

**Required libraries**: requirements.txt

### Table of Contents
<details open>
<summary>Show/Hide</summary>
<br>

1. [Publications](#publications)
2. [Input arguments](#input-arguments)
3. [Workflow and Modules](#workflow)
5. [Tools Used](#tools-used)
6. [Structure](#structure)
6. [References](#references)
7. [Acronyms](#acronyms)

</details>

### Publications
<details>
<a name="publications"></a>
<summary>Show/Hide</summary>
<br>

[Shahnazaryan D, O’Reilly GJ, Monteiro R. Story loss functions for seismic design and assessment: Development of tools and application. 
Earthquake Spectra 2021. DOI: 10.1177/87552930211023523](https://www.researchgate.net/publication/353058466_Story_loss_functions_for_seismic_design_and_assessment_Development_of_tools_and_application)

</details>

### Input arguments
<details>
<a name="input-arguments"></a>
<summary>Show/Hide</summary>
<br>

* Project name
* .csv file containing component data
* .csv file containing the correlation tree
* Correlation type, i.e. Correlated or Independent
* Number of simulations, i.e. Monte Carlo simulations to generate damage states for analysis
* EDP type, i.e. PSD or PFA
* EDP step, i.e. % for PSD and g for PFA
* Flag to store the results in the Database

</details>

### Workflow and Modules
<details>
<a name="workflow"></a>
<summary>Show/Hide</summary>
<br>

<h5 align="center">Toolbox workflow</h5>
<p align="center">
  <img src="https://github.com/davitshahnazaryan3/SLFGenerator/tree/master/sample/Figures/Workflow.png" width=600>
</p>

<h5 align="center">Toolbox modules</h5>
<p align="center">
  <img src="https://github.com/davitshahnazaryan3/SLFGenerator/tree/master/sample/Figures/modules.png" width=600>
</p>

</details>

### Tools used
<details>
<a name="tools-used"></a>
<summary>Show/Hide</summary>
<br>

* tkinter - Graphical User Interface
* pandas - manipulation of data
* numpy - computations
* matplotlib - for data visualization
  
<h5 align="center">Sample Fitting and Generation of SLFs</h5>
<p align="center">
  <img src="https://github.com/davitshahnazaryan3/SLFGenerator/tree/master/sample/Figures/OutputFit.jpg" width=600>
</p>

* scipy optimization - fitting the data
* Monte Carlo simulations

</details>

### Structure
<details>
<a name="structure"></a>
<summary>Show/Hide</summary>
<br>

Note: *The tool relies on the accuracy of the user's provided data, it does not offer its own component information, 
therefore double counting or dependency of different component fragilities should be accounted for by the user, as the 
tool will work either way.*

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
        
    <img src="https://latex.codecogs.com/svg.latex?\Y=\alpha*(1-\exp(-(\frac{x}{\beta})^\gamma))" title="eq.1" /><br/>

        where α, β, γ are the fitting coefficients, x is the EDP range and y is the fitted SLF functions
        
        OUTPUT: Fitted SLF
        
8. Export outputs to .xlsx and cache to .pickle if specified

</details>

### References
<details>
<a name="references"></a>
<summary>Show/Hide</summary>
<br>

[SLF estimation procedure](https://www.researchgate.net/publication/265411359_Building-Specific_Loss_Estimation_Methods_Tools_for_Simplified_Performance-Based_Earthquake_Engineering)

[FEMA P-58 for fragilities](https://femap58.atcouncil.org/reports)

[Consequence functions](https://femap58.atcouncil.org/reports)

[Repair costs - Central Italy](https://sisma2016.gov.it/wp-content/uploads/2019/12/Allegato-3-Prezzario-Cratere_2018-Finale.pdf)

</details>

### Acronyms
<details>
<a name="acronyms"></a>
<summary>Show/Hide</summary>
<br>

EDP:    Engineering Demand Parameter

DV:     Decision Variable

DS:     Damage State

CDF:    Cumulative distribution function

</details>
