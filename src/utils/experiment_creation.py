import numpy as np
import math
from typing import List, Dict, Union, Tuple

from src.utils.distributions import Categorical, Normal
from src.utils.mouselab_PAR import MouselabPar
from src.po_BMPS import run_myopic

# Helper functions for generating json files to be used in the mouselab-MDP online experiment
# Author: Lovis Heindrich

def rewards_json(start_seed, end_seed, TREE, INIT, Mouselab):
    """Creates rewards as used in the mouselab experiment.

    Args:
        start_seed (int): start seed (inclusive)
        end_seed (int): stop seed (exclusive)
        TREE (list): Mouselab tree structure
        INIT (list): Mouselab init structure
        Mouselab (Mouselab): Mouselab environment class
    """
    truths = []
    rewards = "["
    for i in range(start_seed, end_seed):
        np.random.seed(i)
        env = Mouselab(TREE, INIT)
        ground_truth = list(env.ground_truth)
        rewards += f"\n{{\"trial_id\": {i}, \"stateRewards\": {str(ground_truth)}}},"
        truths.append(ground_truth)
    rewards = rewards[:-1] + "\n]"
    return rewards, truths

def getGraph(TREE) -> Dict[str, List[Union[int, str]]]:
    """Creates a mouselab graph as used in structure.json in mouselab experiments

    Args:
        TREE (list): Mouselab tree structure

    Returns:
        [type]: [description]
    """
    keys = ["left", "up", "right", "farright"]
    graph = {}
    for i in range(len(TREE)):
        node_dict = {}
        c_counter = 0
        for c in TREE[i]:
            node_dict[keys[c_counter]] = [0, str(c)]
            c_counter += 1
        graph[i] = node_dict
    return graph

def actions_to_graph_actions(path, TREE):
    """Transforms a path of nodes to the corresponding actions in the graph. 

    Args:
        path (list): List of nodes along the takeb path
        TREE (list): Mouselab tree

    Returns:
        List: Actions that correspond to the taken path in string form
    """

    graph = getGraph(TREE)
    graph_actions = []
    for i in range(1, len(path)):
        previous = path[i-1]
        current = path[i]
        # Find action from previous to current
        possible_actions = graph[previous]
        for k, v in possible_actions.items():
            if v[1] == current:
                str(graph_actions.append(k))
                break
        
    return graph_actions

def get_demonstrations(start_seed, end_seed, W, TREE, INIT, disable_meta, trace, HIGH_COST=1,LOW_COST=1, cost_function="Basic"):
    """Creates a demonstration json for the environments for the given seeds

    Args:
        start_seed (int): Start seed (included)
        end_seed (int): End seed (not included)
        W (array): Weights for the metacontroller
        TREE (array): Mouselab tree
        INIT (array): Mouselab init
        disable_meta (bool): Enables/disables goal switching
        trace (func): tracing function giving the actions of BMPS for the given tree
        COST (int, optional): Cost of a click. Defaults to 1.

    Returns:
        (string, array, array, array): Json string of the demonstration trials and individual elements as arrays
    """
    json = ""
    click_list = []
    reward_list = []
    action_list = []
    for i in range(start_seed, end_seed):
        SEED = i
        clicks, rewards, actions = trace(W, TREE, INIT, HIGH_COST=HIGH_COST, LOW_COST=LOW_COST, SWITCH_COST=0, SEED=SEED, term_belief=False, disable_meta=disable_meta, cost_function=cost_function)
        click_list.append(clicks)
        reward_list.append(rewards)
        action_list.append(actions)
        json += "\n{\"pid\":1,\"actions\":"

        json += str(actions_to_graph_actions(actions, TREE)).replace("\'", "\"")

        json += ",\"clicks\":"

        json += str(clicks)

        json += ",\"stateRewards\":"

        json += str(list(rewards))

        json += "},"

    return json[:-1], click_list, reward_list, action_list

def create_tree(columns=4, blocks=2, base_block=[[2, 3, 4, 5], [6], [6], [7], [7], [8], [8]]) -> List[List[int]]:
    root_links = [1+i*(1+blocks*len(base_block)) for i in range(columns)]
    tree = [root_links]
    for column in range(columns):
        tree[0].append
        for block in range(blocks):
            offset = column * (1+ blocks * len(base_block)) + block * len(base_block)
            new_block = [[n + offset for n in child] for child in base_block]
            tree.extend(new_block)
        tree.append([])
    return tree 
    
def create_init(columns=4, blocks=2):
    d0 = Normal(0,5)
    d1 = Normal(0,10)
    d2 = Normal(0,20)
    new_block_init = [d0]
    for column in range(columns):
        for block in range(blocks):
            if block == 0:
                new_block_init.extend([d0, d0, d0, d0, d0, d0, d0, d1])
            elif block < blocks-1:
                new_block_init.extend([d0, d0, d0, d0, d0, d0, d1])
            else:
                new_block_init.extend([d1, d1, d1, d1, d1, d1, d2])
    return tuple([r for r in new_block_init])

def calculate_column_offset(n: int, width: float=12) -> List[float]:
    offset = (n/2 - 0.5)*width
    positions = [x*width - offset for x in range(n)]
    return positions

def calculate_x_node_group(parent: int, parent_pos: int, group_size: int=7, offset: float=3) -> Dict[int, float]:
    assert group_size == 7, "Group size not supported"
    tree = {parent: parent_pos}
    tree[parent+1] = parent_pos - 1.5* offset
    tree[parent+2] = parent_pos - 0.5* offset
    tree[parent+3] = parent_pos + 0.5* offset
    tree[parent+4] = parent_pos + 1.5* offset
    tree[parent+5] = parent_pos - 1* offset
    tree[parent+6] = parent_pos + 1* offset
    tree[parent+7] = parent_pos 
    return tree

def calculate_x(num_columns: int, blocks_per_column: int, group_size: int, column_offset: float) -> Dict[int, float]:
    x_locations = {}
    for column, column_x in zip(range(num_columns), column_offset):
        for block in range(blocks_per_column):
            block_root = 1 + column*(1+blocks_per_column*group_size) + block*group_size
            x_locations.update(calculate_x_node_group(parent=block_root, parent_pos=column_x, group_size=group_size))
    return x_locations

def calculate_y(TREE: List[List[int]], y_offset=-1.2) -> Dict[int, int]:
    def y_rec(coordinates, node, depth):
        coordinates[node] = depth*y_offset
        for child in TREE[node]:
            y_rec(coordinates, child, depth + 1)
        return coordinates
    y_coordinates = y_rec({}, 0, 0)
    return y_coordinates

def create_structure(columns: int=4, blocks: int=2, base_block: List[List[int]]=[[2, 3, 4, 5], [6], [6], [7], [7], [8], [8]]) -> Dict[int, float]:
    TREE = create_tree(columns=columns, blocks=blocks, base_block=base_block)
    group_size = len(base_block)
    num_columns = len(TREE[0])
    blocks_per_column = int(((len(TREE)-1)/num_columns-1)/group_size)
    column_offset = calculate_column_offset(n=num_columns)

    y_locations = calculate_y(TREE=TREE)
    x_locations = calculate_x(num_columns=num_columns, blocks_per_column=blocks_per_column, group_size=group_size, column_offset=column_offset)
    y_locations[0] = 0
    x_locations[0] = 0

    layout = {str(node):[x_locations[node], y_locations[node]] for node in range(len(TREE))}

    _INIT = tuple([Categorical([0]) for _ in TREE])
    _env = MouselabPar(TREE, _INIT)
    graph = getGraph(TREE)

    env = {
        "layout": layout,
        "initial": "0",
        "graph": graph
    }

    return env

def generate_samples(index, COST, TAU, TREE, INIT, SEED, MAX_ACTIONS, myopic_mode="old") -> Tuple[Dict, str]:
    iter_str = "0" + str(index) if index < 10 else str(index)

    env_data = {}
    env_data["parameter"] = {
        "tree": TREE,
        "init": INIT,
        "tau": TAU,
        "cost": COST,
        "seed": SEED,
        "max_actions": MAX_ACTIONS,
        "parameter_index": iter_str
    }

    env_data["envs"] = {}
    for i in range(25):
        np.random.seed(i+SEED)
        env = MouselabPar(TREE, INIT, cost=COST, tau=TAU, repeat_cost=COST, myopic_mode=myopic_mode, term_belief=False)
        env.ground_truth = np.round(env.ground_truth, 2)
        truth = env.ground_truth
        best_reward = np.round(env.ground_truth_reward(),2)
        available_actions = list(env.actions(env._state))
        samples = {}
        for action in available_actions:
            # Single tau
            if action is not env.term_action:
                action_dist = Normal(truth[action], 1/math.sqrt(TAU))
                action_samples = np.round(action_dist.sample(n=MAX_ACTIONS), 2).tolist()
                samples[action] = action_samples
        rew_mp, acts_mp, exp_rew_mp, repeat_clicks = run_myopic(truth=truth, samples=samples, TREE=TREE, INIT=INIT, TAU=TAU, COST=COST, myopic_mode=myopic_mode)
        env_data["envs"][i] = {
            "ground_truth": truth,
            "samples": samples,
            "best_path_reward": best_reward,
            "mp_reward": rew_mp,
            "mp_actions": acts_mp,
            "mp_exp_reward": exp_rew_mp,
            "mp_repeat_clicks": repeat_clicks
        }
    return env_data, iter_str


# Env Structure PO
def create_structure_po(TREE, INIT):
    env = MouselabPar(TREE, INIT)
    def state_representation(state):
        """ State encoding: [[prob, value]]
        """
        if hasattr(state, "sample"):
            return [float(state.mu), float(state.sigma)]
        else: 
            return [float(state), 0]

    def vpi_action(action, env):
        paths = env.path_to(action)
        flat_paths = [int(node) for path in paths for node in path]
        obs = (*env.subtree[action], *flat_paths)
        obs = list(np.unique(obs))
        return obs
        
    state_init = {str(i):state_representation(state) for i, state in enumerate(env.init)}
    paths = list(env.paths)
    #vpi_subsets = {str(a):vpi_action(a, env) for a in range(1, len(TREE))}
    reward_groups = {reward:[] for reward in np.unique([x.sigma for x in env.init[1:]])}
    for i in range(1, len(env.init)):
        reward_groups[env.init[i].sigma].append(i)
    # Cost and tau depend on condition and will be passed from the json containing the samples
    env_structure = {
        "init": state_init,
        "paths": paths,
        #"vpi_sets": vpi_subsets,
        "term_action": int(env.term_action),
        "reward_groups": list(reward_groups.values())
    }

    return env_structure

def create_2_36_env(num_goals=2):
    sigma_val = {'V1': 5, 'V2': 10, 'V3': 20, 'V4': 40, 'G1': 100, 'G2': 120, 'G3': 140, 'G4': 160, 'G5': 180}
    goals = ["G1", "G2", "G3", "G4", "G5"]

    root = []
    main_tree = []
    dist = ["V1"]
    for i in range(num_goals):
        root.extend((1+(18*i), 18 + (18*i)))
        main_tree.extend([[2+(18*i), 8+(18*i), 13+(18*i)], [3+(18*i), 7+(18*i)], [4+(18*i)], [], [4+(18*i)], [4+(18*i)], [5+(18*i), 6+(18*i)], [9+(18*i), 12+(18*i)], [4+(18*i)], [4+(18*i)], [4+(18*i)], [10+(18*i), 11+(18*i)], [14+(18*i), 17+(18*i)], [4+(18*i)], [4+(18*i)], [4+(18*i)], [15+(18*i), 16+(18*i)], [4+(18*i)]])
        dist.extend(['V1', 'V2', 'V3', goals[i%5], 'V4', 'V4', 'V3', 'V2', 'V3', 'V4', 'V4', 'V3', 'V2', 'V3', 'V4', 'V4', 'V3', 'V1'])
    return [root]+main_tree, tuple([Normal(mu=0, sigma=sigma_val[d]) for d in dist])