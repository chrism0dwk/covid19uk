# Covid-19 Lancaster University data statement

__Data contained in this directory is all publicly available from UK government agencies or previous studies.
No personally identifiable information is stored.__

ONS: Office for National Statistics

PHE: Public Health England

UTLA: Upper Tier Local Authority

LAD: Local Authority District

Polymod: research output from [this](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0050074) social mixing study
## Files contained in this directory

* `DailyConfirmedCases_2020-03-20.csv` -- total England daily case data scraped from Public Health England's page [here](https://www.gov.uk/government/publications/covid-19-track-coronavirus-cases)
* `movement.rds` -- a R "RDS" file containing a matrix of commuting frequency between Upper Tier Local Authorities in England.  Data taken from publicly available Census 2011 data from Office for National Statistics
* `polymod_no_school_df.rds` -- a R "RDS" file containing the [Polymod](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0050074) social mixing data for vacation time
* `polymod_normal_df.rds` -- R "RDS" file containing the [Polymod](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0050074) social mixing data for school time
* `pop.rds` -- a R "RDS" file containing the ONS 2019 projected population size in each UTLA
* `ukmidyearestimates20182019ladcodes.csv` -- ONS 2019 projected population size in each LAD (no currently used)

