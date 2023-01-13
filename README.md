# MGPO

This repository contains data, analysis code, and the MGPO algorithms from the "Leveraging AI to improve human planning in large partially observable environments" article (under review).

## Structure

- The resultsdata from the simulation and human experiments, the pre-generated environment instances, and figures used in the article can be found in the ```data``` folder. 

- The ```experiment``` folder contains the code required to run the human experiment.

- The ```src``` folder contains the implementation of the meta-level MDP and the MGPO and baseline algorithms.

- The top level folder contains additional notebooks used to analyze the experiment data.


## Algorithm

The main algorithmic contributions can be found in two files. The meta-MDP is defined in ```src/utils/Mouselab_PAR.py```, which contains the belief state update in partially observable environments. The MGPO algorithm itself can be found in ```scr/po_BMPS.py```.

## Simulation

To rerun the simulation experiment, run the following scripts. The evaluation will take a long time to run, it is recommended to split the computations in small chunks and use a computing cluster. Results are stored under ```data/simulation_results/```.

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

The results of the human experiment can be found in the following files:
```
data
└───tutor_experiment
│   │   questionnaire.csv
│   │   tutor_experiment_exclusion_data.csv
|   |   tutor_experiment_full_data.csv
```

to generate the result files from the raw data retrieved from the database the following script was used:
``` 
python -m src.tutor_experiment_analysis 
```

Data analysis and statistical analysis can be found in the files:

```
experiment_analysis.ipynb
experiment_analysis.R
```

The experiment's preregistration can be found under [https://aspredicted.org/RL3_YDD](https://aspredicted.org/RL3_YDD).

## Human experiment

The experiment is a heavily adapted version of [Fred Callaway's Mouselab-MDP](https://github.com/fredcallaway/Mouselab-MDP). 

To run the experiment locally, run the commands in the command line and open http://localhost:8000/ in a webbrowser. 
```
cd experiment
python -m http.server
```

The environment instances used in the environment are stored under ```data/environments```.

We used Heroku to host the experiment and Prolific to recruit participants. Before deploying the experiment, it is important to comment out line 60-61 in ```experiment/static/js/experiment.js``` since a balanced condition assignment will be handled through PsiTurk.