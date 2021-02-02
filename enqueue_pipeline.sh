#!/bin/bash

# Enqueues COVID-19 pipelines

COMMUTE_VOL_FILE="data/210122_OFF_SEN_COVID19_road_traffic_national_table.xlsx"
DATE_LOW="2020-10-30"
DATE_HIGH="2021-01-22"

TEMPLATE_CONFIG=template_config.yaml
JSV_SCRIPT=/usr/shared_apps/packages/sge-8.1.9-gpu/default/common/sge_request.jsv.2.0b

# Job submisison
switch-gpu
RESULTS_DIR=$global_scratch/covid19/scotland_${DATE_HIGH}_notier
JOB_NAME="covid_scotland_${DATE_HIGH}_notier"	
qsub -N $JOB_NAME covid_pipeline.sge \
     --commute-volume "$COMMUTE_VOL_FILE" \
     --date-range $DATE_LOW $DATE_HIGH \
     --results-dir "$RESULTS_DIR" \
     --config $TEMPLATE_CONFIG
