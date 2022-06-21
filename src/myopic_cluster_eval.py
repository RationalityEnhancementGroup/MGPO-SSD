
import argparse
from src.po_BMPS import eval_myopic
from src.utils.experiment_creation import create_2_36_env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('goals', type=int, default=2, help='Number of goals in 2_36 environment')
    parser.add_argument('cost', type=float, default=0.05, help='Cost of clicking')
    parser.add_argument('seed', type=int, default=0, help='Starting seed for evaluation environments')
    parser.add_argument('n', type=int, default=5000, help='Number of evaluated environments')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    TREE, INIT = create_2_36_env(args.goals)
    COST = args.cost
    TAU = 0.005

    # Both sigma sqrt fix
    weight_dict = {
        (2, 0.05): 0.018286316255955063,
        (2, 1): 0.4040768388809629,
        (3, 0.05): 0.007556223393453305,
        (3, 1): 0.3620224382925328,
        (4, 0.05): 0.11918129794996793,
        (4, 1): 0.18095018415587694,
        (5, 0.05): 0.020169094852645935,
        (5, 1): 0.27682040176433226
    }

    cost_weight = weight_dict[(args.goals, COST)]
    
    res = eval_myopic(args.n, TREE, INIT, TAU, COST, seed=args.seed, max_actions=200, cost_weight=cost_weight)
    res["EnvType"] = args.goals
    res["Cost"] = args.cost
    res["CostWeight"] = cost_weight
    res.to_csv(f'./data/simulation_results/mgpo_evaluation/{args.goals}_{args.cost}_{args.seed}.csv')
