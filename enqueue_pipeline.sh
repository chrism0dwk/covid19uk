#!/bin/bash

# Enqueues COVID-19 pipelines

CASES_FILE="data/Anonymised Combined Line List 20210108.csv"
COMMUTE_VOL_FILE="data/210108_OFF_SEN_COVID19_road_traffic_national_table.xlsx"
DATE_LOW="2020-10-13"
DATE_HIGH="2021-01-05"

TEMPLATE_CONFIG=template_config.yaml


# Job submisison
switch-gpu

for PILLAR in both 1 2
do
    for CASE_DATE_TYPE in specimen
    do
	RESULTS_DIR=$global_scratch/covid19/${DATE_HIGH}_${PILLAR}_${CASE_DATE_TYPE}
	JOB_NAME="covid_${DATE_HIGH}_${PILLAR}_${CASE_DATE_TYPE}"	
	qsub -N $JOB_NAME covid_pipeline.sge \
	     --reported-cases "$CASES_FILE" \
	     --commute-volume "$COMMUTE_VOL_FILE" \
	     --case-date-type $CASE_DATE_TYPE \
	     --pillar $PILLAR \
	     --date-range $DATE_LOW $DATE_HIGH \
	     --results-dir "$RESULTS_DIR" \
	     --config $TEMPLATE_CONFIG
    done
done
