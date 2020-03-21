# covid19uk

## Files

* `covid` Python package
  * `model.py` main CovidUKODE model class
  * `rdata.py` functions to import R .rds files with data
  * `pydata.py` functions for importing files not from R .rds files
  * `util.py` utility functions
  
* `covid_ode.py` script that runs a test simulator
* `mcmc.py` fits the model
* `prediction.py` ingests a posterior from `mcmc.py`, calculates predictions (saved in pred_2020-03-15.h5) file.
* `ode_config.yaml` default YAML configuration file for Covid-19 model (used by the 3 files above)
* `household_sim.py` experimental (and probably broken) code
* `environment.yaml` conda description of the required environment.  Create with `conda create --file environment.yaml`
