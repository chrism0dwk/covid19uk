# covid19uk: Bayesian stochastic spatial modelling for COVID-19 in the UK

This Python package implements a spatial stochastic SEIR model for COVID-19 in the UK,
using Local Authority District level positive test data, population data, and mobility
information.  Details of the model implementation may be found in `doc/lancs_space_model_concept.pdf`.



## Workflow
The entire analysis chain, from case data through parameter inference and predictive
simulation to results summarised as long-format XLSX and geopackage documents.
The pipeline is run using the [`ruffus`](http://ruffus.org.uk) computational pipeline library.

The library relies heavily on [TensorFlow](https://tensorflow.org) and
[TensorFlow Probability](https://tensorflow.org/probability) machine learning libraries, and is
optimised for GPU acceleration (tested on NVIDIA K40 and V100 cards).  This package also imports
an experimental library [`gemlib`](http://fhm-chicas-code.lancs.ac.uk/GEM/gemlib) hosted at Lancaster University.
This library is under active development as part of the [Wellcome Trust](https://wellcome.ac.uk) `GEM` project.

The pipeline gathers data from [the official UK Government website for data and insights on Coronavirus]
(https://coronavirus.data.gov.uk), together with population and mobility data taken from the [UK
Office for National Statistics Open Geography Portal](https://geoportal.statistics.gov.uk).

### Quickstart
```bash
$ poetry install  # Python dependencies
$ python -m covid.pipeline --config example_config.yaml --results-dir <output_dir>
```

The global pipeline configuration file `example_config.yaml` contains sections for pipeline
stages where required.  See file for documentation. 


### Model specification
The `covid.model_spec` module contains the model specified as a `tensorflow_probability.distributions.JointDistributionNamed`
instance `Covid19UK`.  This module also contains a model version number, and constants such as the stoichiometry matrix
characterising the state transition model, an accompanying next generation matrix function, a function to assemble data
specific to the model, and a function to initialise censored event data explored by the MCMC inference algorithm.

### Pipeline stages
Each pipeline stage loads input and saves output to disc.  This is inherent to the `ruffus` pipeline
architecture, and provides the possibility to run different stages of the pipeline manually, as well as
introspection of data passed between each stage.

1. Data assembly: `covid.tasks.assemble_data` downloads or loads data from various sources, clips
to the desired date range require needed, and bundles into a pickled Python dictionary `<output_dir>/pipeline_data.pkl`.

2. Inference: `covid.tasks.mcmc` runs the data augmentation MCMC algorithm described in the concept note, producing
a (large!) HDF5 file containing draws from the joint posterior distribution `posterior.hd5`.

3. Sample thinning: `covid.tasks.thin_posterior` further thins the posterior draws contained in the HDF5 file into a (much
smaller) pickled Python dictionary `<output_dir>/thin_samples.pkl`

4. Next generation matrix: `covid.tasks.next_generation_matrix` computes the posterior next generation matrix for the
epidemic, from which measures of Local Authority District level and National-level reproduction number can be derived.
This posterior is saved in `<output_dir>/ngm.pkl`.

5. National Rt: `covid.tasks.overall_rt` evaluates the dominant eigenvalue of the next generation matrix samples using
power iteration and Rayleigh Quotient method.  The dominant eigenvalue of the inter-LAD next generation matrix gives the
national reproduction number estimate.

6. Prediction: `covid.tasks.predict` calculates the Bayesian predictive distribution of the epidemic given the observed
data and joint posterior distribution.  This is used in two ways:
   - in-sample predictions are made for the latest 7 and 14 day time intervals in the observed data time window.  These
    are saved as `<output_dir>/insample7.pkl` and `<output_dir>/insample14.pkl` `xarray` data structures. 
   - medium-term predictions are made by simulating forward 56 days from the last+1 day of the observed data time window.  These is saved as `<output_dir>/medium_term.pkl` `xarray` data structure. 

7. Summary output:
   - LAD-level reproduction number: `covid.tasks.summarize.rt` takes the column sums of the next generation matrix as the
LAD-level reproduction number.  This is saved in `<output_dir>/rt_summary.csv`.
   - Incidence summary: `covid.tasks.summarize.infec_incidence` calculates mean and quantile information for the medium term prediction, `<output_dir>/infec_incidence_summary.csv`.
   - Prevalence summary: `covid.tasks.summarize.prevalence` calculated the predicted prevalence of COVID-19 infection
(model E+I compartments) at LAD level, `<output_dir>/prevalence_summary.csv`.
   - Population attributable risk fraction for infection: `covid.tasks.within_between` calculates the population
attributable fraction of within-LAD versus between-LAD infection risk, `<output_dir>/within_between_summary.csv`.
   - Case exceedance: `covid.tasks.case_exceedance` calculates the probability that observed cases in the last 7 and 14
 days of the observed timeseries exceeding the predictive distribution.  This highlights regions that are behaving
 atypically given the model, `<output_dir>/exceedance_summary.csv`.

8. In-sample predictive plots: `covid.tasks.insample_predictive_timeseries` plots graphs of the in-sample predictive
distribution for the last 7 and 14 days within the observed data time window, `<output_dir>/insample_plots7` and
`<output_dir>/insample_plots14`.

9. Geopackage summary: `covid.tasks.summary_geopackage` assembles summary information into a `geopackage` GIS file,
`<output_dir>/prediction.pkg`.

10. Long format summary: `covid.tasks.summary_longformat` assembles reproduction number, observed data, in-sample, and medium-term
predictive incidence and prevalence (per 100000 people) into a long-format XLSX file.



## COVID-19 Lancaster University data statement

__Data contained in the `data` directory is all publicly available from UK government agencies or previous studies.
No personally identifiable information is stored.__

ONS: Office for National Statistics

PHE: Public Health England

UTLA: Upper Tier Local Authority

LAD: Local Authority District


### Files

* `covid` Python package
* `example_config.yaml` example configuration file containing data paths and MCMC settings
* `data` a directory containing example data (see below)
* `pyproject.py` a PEP518-compliant file describing the `poetry` build system and dependencies.

## Example data files
* `data/example_cases.csv` a file containing example case data for 43 local authorities in England collected and present of PHE's [website](https://coronavirus.data.gov.uk)
* `data/c2019modagepop.csv` a file containing local authority population data in the UK, taken from ONS prediction for December 2019.  Local authorities [City of Westminster, City of London] and [Cornwall, Isles of Scilly] have been aggregated to meet commute data processing requirements. 
* `data/mergedflows.csv` inter local authority mobility matrix taken from UK Census 2011 commuting data and aggregated up from Middle Super Output Area level (respecting aggregated LADs as above).
* `data/UK2019mod_pop.gpkg` a geopackage containing UK Local Authority Districts (2019) polygons together with population and areal metrics.


