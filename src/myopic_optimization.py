# Optimization script for cost weight hyperparameter of MGPO
from src.po_BMPS import eval_myopic
import GPyOpt
import time
import numpy as np
import argparse
from src.utils.experiment_creation import create_2_36_env, create_init, create_tree
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_type', type=str, default="4_2", help="Structure of the environment")
    parser.add_argument('cost', type=float, default=0.05, help='Cost of clicking')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()


    if args.env_type == "4_2":
        TREE = create_tree(4, 2)
        INIT = create_init(4, 2)
    elif args.env_type in ["2_36", "3_54", "4_72", "5_90"]:
        num_goals = int(args.env_type.split("_")[0])
        TREE, INIT = create_2_36_env(num_goals)
    else:
        print("Error: Unknown structure", args.env_type)
        exit()
    COST = args.cost
    TAU = 0.005

    n_eval = 500
    restarts = 10
    steps = 50

    def blackbox(W):
        res = eval_myopic(n_eval, TREE, INIT, TAU, COST, seed=5000, max_actions=200, cost_weight=W[0,0])
        print(W, res["ExpectedReward"].mean(), res["TrueReward"].mean())
        return - (res["ExpectedReward"].mean())
        
    np.random.seed(123456)

    space = [{'name': 'cost_weight', 'type': 'continuous', 'domain': (0,1)}]
    feasible_region = GPyOpt.Design_space(space = space)
    initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, restarts)
    objective = GPyOpt.core.task.SingleObjective(blackbox)
    model = GPyOpt.models.GPModel(exact_feval=True,optimize_restarts=restarts,verbose=False)
    aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)
    acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
    bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, initial_design)

    # --- Stop conditions
    max_time  = None
    tolerance = 1e-6     # distance between two consecutive observations        

    # Run the optimization
    max_iter  = steps
    time_start = time.time()
    train_tic = time_start
    bo.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbosity=True)

    W_low = np.array([bo.x_opt])[0,0]
    train_toc = time.time()

    print("\nSeconds:", train_toc-train_tic)
    print("Weights:", W_low)
    res = eval_myopic(n_eval, TREE, INIT, TAU, COST, seed=5000+n_eval, max_actions=200, cost_weight=W_low)

    optimization_res = {
        "W": W_low,
        "TrueReward": res["TrueReward"].mean(),
        "ExpectedReward": res["ExpectedReward"].mean(),
        "NumActions": res["NumActions"].mean(),
        "NumRepeatActions": res["NumRepeatActions"].mean()
    }

    with open(f"./data/simulation_results/mgpo_optimization/{args.env_type}_{args.cost}.json", 'w') as out_f:
        json.dump(optimization_res, out_f)