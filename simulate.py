"""Test simulation for COVID-19 UK model"""

import optparse
import yaml
import numpy as np
import tensorflow as tf

from covid import config
from covid.model import load_data
from covid.pydata import phe_case_data
from covid.util import sanitise_settings, impute_previous_cases
from covid.impl.util import compute_state

import model_spec

DTYPE = config.floatX

# Read in data
parser = optparse.OptionParser()
parser.add_option(
    "--config",
    "-c",
    dest="config",
    default="example_config.yaml",
    help="configuration file",
)
options, cmd_args = parser.parse_args(["-c", "example_config.yaml"])
print("Loading config file:", options.config)

with open(options.config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


# Load in covariate data
covar_data = model_spec.read_covariates(config['data'])

# We load in cases and impute missing infections first, since this sets the
# time epoch which we are analysing.
cases = model_spec.read_cases(config['data']['reported_cases'])

# Single imputation of censored data
events = model_spec.impute_censored_events(cases)

# Initial conditions S(0), E(0), I(0), R(0) are calculated
# by calculating the state at the beginning of the inference period
state = compute_state(
    initial_state=tf.concat(
        [covar_data["N"], tf.zeros_like(events[:, 0, :])], axis=-1
    ),
    events=events,
    stoichiometry=model_spec.STOICHIOMETRY,
)
start_time = state.shape[1] - cases.shape[1]
initial_state = state[:, start_time, :]
events = events[:, start_time:, :]

# Build model and sample
full_probability_model = model_spec.CovidUK(
    covariates=covar_data,
    xi_freq=14,
    initial_state=initial_state,
    initial_step=0,
    num_steps=80,
)
seir = full_probability_model.model["seir"](
    beta1=0.35, beta2=0.65, xi=[0.0] * 5, nu=0.5, gamma=0.49
)
sim = seir.sample()
