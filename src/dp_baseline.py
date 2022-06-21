### Script to run the meta-greedy baseline policy

from src.utils.experiment_creation import create_tree, create_init, create_2_36_env
from src.po_BMPS import eval_DP
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('tau', type=float, default=0.005, help='Precision of observations')
    parser.add_argument('cost', type=float, default=0.05, help='Cost of clicking')
    parser.add_argument('n_eval', type=int, default=1, help='Number of evaluated environments')
    parser.add_argument('dp_depth', type=int, default=1, help='Depth of dynamic programming evaluation')
    parser.add_argument('dp_bins', type=int, default=4, help='Number of bins for discretized belief states')
    parser.add_argument('env_type', type=str, default="4_2", help="Structure of the environment")
    parser.add_argument('max_actions', type=int, default=200, help='Maximum number actions per environment')
    parser.add_argument('seed', type=int, default=0, help='Seed of the first environment')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

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

    res_DP = eval_DP(args.n_eval, TREE, INIT, args.tau, args.cost, DP_BINS=args.dp_bins, DP_DEPTH=args.dp_depth, seed=args.seed, max_actions=args.max_actions)
    res_DP["EnvType"] = args.env_type
    res_DP["Cost"] = args.cost
    res_DP.to_csv(f'./data/simulation_results/mg_evaluation/{args.env_type}_{str(args.cost)[2:]}_{str(args.tau)[2:]}_{str(args.dp_depth)}_{str(args.dp_bins)}_{str(args.seed)}.csv')