# covid19uk: Bayesian stochastic spatial modelling for COVID-19 in the UK

This Python package implements a spatial stochastic SEIR model for COVID-19 in the UK,
using Local Authority District level positive test data, population data, and mobility
information.  Details of the model implementation may be found in `doc/lancs_space_model_concept.pdf`.



## Workflow
This repository contains code that produces Monte Carlo samples of the Bayesian posterior distribution
given the model and case timeseries data from [coronavirus.data.gov.uk](https://coronavirus.data.gov.uk), 
implementing an ETL step, the model itself, and associated inference and prediction steps.

Users requiring an end-to-end pipeline implementation should refer to the [covid-pipeline](https://github.com/chrism0dwk/covid-pipeline)
repository.


## COVID-19 Lancaster University data statement

__Data contained in the `data` directory is all publicly available from UK government agencies or previous studies.
No personally identifiable information is stored.__

ONS: Office for National Statistics

PHE: Public Health England

UTLA: Upper Tier Local Authority

LAD: Local Authority District


## Example data files
* `data/c2019modagepop.csv` a file containing local authority population data in the UK, taken from ONS prediction for December 2019.  Local authorities [City of Westminster, City of London] and [Cornwall, Isles of Scilly] have been aggregated to meet commute data processing requirements. 
* `data/mergedflows.csv` inter local authority mobility matrix taken from UK Census 2011 commuting data and aggregated up from Middle Super Output Area level (respecting aggregated LADs as above).
* `data/UK2019mod_pop.gpkg` a geopackage containing UK Local Authority Districts (2019) polygons together with population and areal metrics.


