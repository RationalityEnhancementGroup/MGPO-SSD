### CLuster script for running PO-UCT
# Requires a res/po_uct folder to store results
from src.utils.experiment_creation import create_tree, create_init, create_2_36_env
from src.po_BMPS import eval_pouct

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('tau', type=float, default=0.005, help='Precision of observations')
    parser.add_argument('cost', type=float, default=0.05, help='Cost of clicking')
    parser.add_argument('n_eval', type=int, default=1, help='Number of evaluated environments')
    parser.add_argument('steps', type=int, default=1000, help='MCTS evaluation steps per action')
    parser.add_argument('rollout_depth', type=int, default=3, help='MCTS rollout policy depth')
    parser.add_argument('exploration_coeff', type=float, default=1, help='UCB exploration coefficient')
    parser.add_argument('discretize', type=int, default=1, help='Discretize normal distributions (1) or not (0)')
    parser.add_argument('bins', type=int, default=4, help='Number of bins for discretized normal distributions')
    parser.add_argument('env_type', type=str, default="4_2", help="Structure of the environment")
    parser.add_argument('max_actions', type=int, default=200, help='Maximum number actions per environment')
    parser.add_argument('seed', type=int, default=0, help='Seed of the first environment')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    discretize = args.discretize == 1

    if args.env_type == "4_2":
        TREE = create_tree(4, 2)
        INIT = create_init(4, 2)
    elif args.env_type == "2_2":
        TREE = create_tree(2, 2)
        INIT = create_init(2, 2) 
    elif args.env_type == "2_1":
        TREE = create_tree(2, 1)
        INIT = create_init(2, 1) 
    elif args.env_type == "1_1":
        TREE = create_tree(1, 1)
        INIT = create_init(1, 1)
    elif args.env_type in ["2_36", "3_54", "4_72", "5_90"]:
        num_goals = int(args.env_type.split("_")[0])
        TREE, INIT = create_2_36_env(num_goals)
    else:
        print("Error: Unknown structure", args.env_type)
        exit()

    res = eval_pouct(N=args.n_eval, TREE=TREE, INIT=INIT, TAU=args.tau, COST=args.cost, steps=args.steps, rollout_depth=args.rollout_depth, 
        exploration_coeff=args.exploration_coeff, discretize=discretize, bins=args.bins, seed=args.seed)
    res["EnvType"] = args.env_type
    res["Cost"] = args.cost
    res.to_csv(f'./data/simulation_results/po_uct_evaluation/{args.seed}_{str(args.steps)}_{str(args.exploration_coeff)}_{str(int(args.rollout_depth))}_{args.env_type}_{args.cost}.csv')