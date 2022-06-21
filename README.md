# MGPO

## Algorithm

The main algorithmic contributions can be found in two files. The meta-MDP is defined in ```src/utils/Mouselab_PAR.py```, which contains the belief state update in partially observable environments. The MGPO algorithm itself can be found in ```scr/po_BMPS.py```.
## Simulation

To rerun the simulation experiment, run the following scripts. The evaluation will take a long time to run, it is recommended to split the computations in small chunks and use a computing cluster. Results are stored under ```data/tutor_experiment/```.

### Meta-greedy baseline policy

To reproduce the evaluation results, run the script for 5000 steps with each of the 4 evaluation environments ```["2_36", "3_54", "4_72", "5_90"]``` and cost parameters ```[0.05, 1]```. Example use: 

```
python -m src.dp_baseline 0.005 0.05 5000 1 4 2_36 200 0
```
### PO-UCT baseline policy

To reproduce the evaluation results, run the script for 5000 steps for all combinations of environments (```["2_36", "3_54", "4_72", "5_90"]```), cost parameter (```[0.05, 1]```), and PO-UCT budget (```[10, 100, 1000, 5000]```). Hyperparameters for each parameter combination can be found in ```data/simulation_results/pouct_hyperparameters.csv```. Example use:

```
python -m src.pouct_baseline 0.005 0.05 1 100 3 100 1 4 2_36 200 0
```
### MGPO policy

To reproduce the evaluation results, run the script for 5000 steps for all combinations of cost (```[0.05, 1]```) and environments (```[2, 3, 4, 5]```). Example use:
```
python -m src.myopic_cluster_eval 2 0.05 0 1
```

## Human experiment data and analysis

To run the analysis script, run the following command from the main project folder.

``` 
python -m src.tutor_experiment_analysis 
```

This will save the participant responses into the files:

```
data
└───tutor_experiment
│   │   questionnaire.csv
│   │   tutor_experiment_exclusion_data.csv
|   |   tutor_experiment_full_data.csv
```

Further data analysis and statistics can be found in the files:

```
experiment_analysis.ipynb
experiment_analysis.R
```

## Human experiment

The experiment is a heavily adapted version of [Fred Callaway's Mouselab-MDP](https://github.com/fredcallaway/Mouselab-MDP). 

To run the experiment locally, run the commands in the command line and open http://localhost:8000/ in a webbrowser. 
```
cd experiment
python -m http.server
```

Before deploying the experiment, it is important to comment out line 60-61 in ```experiment/static/js/experiment.js``` since a balanced condition assignment will be handled through PsiTurk.