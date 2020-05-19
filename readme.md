  - impact evaluation voice/shifted norms and attitudes  - Myanmar 2019/2020

Code to reproduce analyses and visuals for Strategic Partnership impact evaluations, disaggregation based on treatment categories, based on data from Myanmar, expanded to Vietnam. <add link>. 

# Technologies
Project is created with: 
- STATA 13.1
- Python 3.8.0 

# Structure
``` 
└───src
    │   MM_treatment_cat_viz.py --> script wrangling data in shape and generating visualisations
    │
    ├───data            --> input data (csv/xls with propensity score matched estimates generated in Stata)
        ├── vnm         --> input data for vietnam 
    │       
    │
    ├── graphs          --> visualisations (in svg)/ results by treatment category written to this folder
    ├── alt_graphs      --> alternative graphs (levels & differences seperate)
            
```
# Data 

- links to data here (Oxfam novib access only)
- Input: 
  - [VariableLabels_mya_2.xlsx](https://oxfam.box.com/s/p1qrkjzbjg6bnhzg9t9u6m0uyvrcif85)
  - [Value labels_mya_2.xlsx](https://oxfam.box.com/s/anndmg4o5zlpy3aqo8paulsls8lgdvtv)
  - [results_graphs_treatment_category_1.csv](https://oxfam.box.com/s/c8pvajojnu3idirkwvm2sh1psr4l3n9e)
  - [results_graphs_treatment_category_2.csv](https://oxfam.box.com/s/cffg6nnzc506uhesyvtc0vvt9de3oryr)
  - [results_graphs_treatment_category_3.csv](https://oxfam.box.com/s/dr14lp9mn6t2880ywibl4nzoxtak2upj)
  - [results_graphs_treatment_category_1.csv](https://oxfam.box.com/s/v2wp1z98p2x26ylscc3mno87mnenx61x)
  