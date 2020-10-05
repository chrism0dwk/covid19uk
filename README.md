# covid19uk: Bayesian stochastic spatial modelling for COVID-19 in the UK

## Files

* `covid` Python package
* `model_spec.py` defines the CovidUK model using `tfp.JointDistributionNamed`, plus helper functions
* `inference.py` demonstrates MCMC model fitting the model
* `simulate.py` demontrates simulating from the model
* `example_config.yaml` example configuration file containing data paths and MCMC settings
* `data` a directory containing example data (see below)
* `environment.yaml` conda description of the required environment.  Create with `conda create -f environment.yaml`
8 `summary.py` python script to summarise MCMC results into a Geopkg file.

## Example data files

* `data/example_cases.csv` a file containing example case data for 43 local authorities in England collected and present of PHE's [website](https://coronavirus.data.gov.uk)
* `data/example_population.csv` a file containing local authority population data in the UK, taken from ONS prediction for December 2019
* `data/example_mobility.csv` inter local authority mobility matrix taken from UK Census 2011 commuting data
* `data/example_traffic_flow` a relative measure of traffic flow taken from mobility metrics from providers such as Google and Facebook.  Data have been smoothed to represent a summary of the original data.

## Example workflow

### With Conda

```bash
$ conda env create --prefix=./env -f environment.txt
$ conda activate ./env
$ python inference.py
$ python summary.py
```

### With Pip and venv

```bash
$ python3 -m venv covid19uk_env covid19uk_env
$ source covid19uk_env/bin/activate
$ pip install -r requirements.txt
$ python inference.py
$ python summary.py
```

When finished:

```bash
exit
```

If you have installation problems, try:

```bash
$ pip install -r requirements-freeze.txt
```

## Outputs

- `.gpkg` - GeoPackage file
- `posterior.h5` a HDF5 file is used to save config

## COVID-19 Lancaster University data statement

__Data contained in the `data` directory is all publicly available from UK government agencies or previous studies.
No personally identifiable information is stored.__

ONS: Office for National Statistics

PHE: Public Health England

UTLA: Upper Tier Local Authority

LAD: Local Authority District
