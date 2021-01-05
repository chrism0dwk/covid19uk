#!/bin/bash

# Enqueues COVID-19 pipelines

CASES_FILE="data/Anonymised Combined Line List 20210104.csv"
DATE_LOW="2020-10-09"
DATE_HIGH="2021-01-01"

TEMPLATE_CONFIG=template_config.yaml


# Job submisison
switch-gpu

for PILLAR in both 1
do
    for CASE_DATE_TYPE in specimen
    do
	RESULTS_DIR=$global_scratch/covid19/${DATE_HIGH}_${PILLAR}_${CASE_DATE_TYPE}
	JOB_NAME="covid_${DATE_HIGH}_${PILLAR}_${CASE_DATE_TYPE}"	
	qsub -N $JOB_NAME covid_pipeline.sge \
	     --reported-cases "$CASES_FILE" \
	     --case-date-type $CASE_DATE_TYPE \
	     --pillar $PILLAR \
	     --inference-period $DATE_LOW $DATE_HIGH \
	     --results-dir "$RESULTS_DIR" \
	     --output "$RESULTS_DIR/config.yaml" \
	     $TEMPLATE_CONFIG
    done
done
