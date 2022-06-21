from src.utils.mouselab_PAR import MouselabPar
from src.po_uct import POUCT
import numpy as np
import GPyOpt
import time
import copy
import pandas as pd
import time

def original_mouselab(W, TREE, INIT, LOW_COST,num_episodes=100, SEED=1000, term_belief=False, exact_seed=False, cost_function="Basic", max_actions=None, tau=1):
    """[summary]

    Args:
        W ([type]): [description]
        TREE ([int]): MDP structure
        INIT ([int]): MDP reward distribution per node
        LOW_COST (int): Cost of computing a node.
        num_episodes (int, optional): Number of episodes to evaluate the MDP on. Defaults to 100.
        SEED (int, optional): Seed to fix random MDP initialization.
        term_belief (bool, optional): If true the expected reward instead of the real reward is used. Defaults to False.

    Returns:
        [int]: Reward of each episode
        [[int]]: Actions of each episode 
    """

    w1 = W[:,0]
    w2 = W[:,1]
    w4 = W[:,2]
    if cost_function == "Actionweight" or cost_function == "Independentweight":
        w0 = W[:,3]
    w3 = 1 - w1 - w2

    if cost_function == "Basic":
        simple_cost = True
    else:
        simple_cost = False
    
    num_nodes = len(TREE) - 1

    def voc_estimate(x):
        features = env.action_features(x)
        if cost_function == "Basic":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0]
        elif cost_function == "Hierarchical":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*features[0][1] + w2*features[0][2])
        elif cost_function == "Actionweight":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(features[0][0] + w0*(features[0][1]))
        elif cost_function == "Novpi":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*(features[0][1]/num_nodes))
        elif cost_function == "Proportional":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(features[0][0] + (features[0][1]/num_nodes))
        elif cost_function == "Independentweight":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0][0] + w0*features[0][1]
        else:
            assert False
    
    rewards = []
    actions = []
    for i in range(num_episodes):
        if exact_seed:
            np.random.seed(SEED + i)
        else:
            np.random.seed(1000*SEED + i)
        env = MouselabPar(TREE, INIT, cost=LOW_COST, term_belief=term_belief, tau=tau, repeat_cost=LOW_COST)
        exp_return = 0
        actions_tmp = []
        num_actions = 0
        while True:
            if max_actions is not None and num_actions >= max_actions:
                _, rew, done, _=env._step(env.term_action)
            else:
                possible_actions = list(env.actions(env._state))
                action_taken = max(possible_actions, key = voc_estimate)
                _, rew, done, _=env._step(action_taken)
            exp_return+=rew
            num_actions += 1
            if done:
                break
            else:
                actions_tmp.append(action_taken)
        rewards.append(exp_return)
        actions.append(actions_tmp)
        del env, possible_actions
    return rewards, actions


def optimize(TREE, INIT, LOW_COST, evaluated_episodes=100, samples=10, iterations=100, SEED=0, term_belief=True, exact_seed=False, cost_function="Basic", max_actions=None, tau=1):
    """Optimizes the weights for BMPS using Bayesian optimization.

    Args:
        TREE ([int]): MDP structure
        INIT ([int]): MDP reward distribution per node
        LOW_COST (int): Cost of computing a node.
        samples (int, optional): Number of initial random guesses before optimization. Defaults to 30.
        iterations (int, optional): Number of optimization steps. Defaults to 50.
        SEED (int, optional): Seed to fix random MDP initialization.
        term_belief (bool, optional): If true the expected reward instead of the real reward is used. Defaults to True.
    """

    def blackbox_original_mouselab(W):
        w1 = W[:,0]
        w2 = W[:,1]
        w4 = W[:,2]
        if cost_function == "Actionweight" or cost_function == "Independentweight":
            w0 = W[:,3]
        w3 = 1 - w1 - w2
        
        if cost_function == "Basic":
            simple_cost = True
        else:
            simple_cost = False

        num_episodes = evaluated_episodes
        print("Weights", W)
        
        num_nodes = len(TREE) - 1

        def voc_estimate(x):
            features = env.action_features(x)
            if cost_function == "Basic":
                return w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0]
            elif cost_function == "Hierarchical":
                return w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*features[0][1] + w2*features[0][2])
            elif cost_function == "Actionweight":
                return w1*features[1] + w2*features[3] + w3*features[2] + w4*(features[0][0] + w0*(features[0][1]))
            elif cost_function == "Novpi":
                return w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*(features[0][1]/num_nodes))
            elif cost_function == "Proportional":
                return w1*features[1] + w2*features[3] + w3*features[2] + w4*(features[0][0] + (features[0][1]/num_nodes))
            elif cost_function == "Independentweight":
                return w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0][0] + w0*features[0][1]
            else:
                assert False
        
        cumreturn = 0
        for i in range(num_episodes):
            #TODO
            if exact_seed:
                np.random.seed(SEED + i)
            else:
                np.random.seed(1000*SEED + i)
            env = MouselabPar(TREE, INIT, cost=LOW_COST, term_belief=term_belief, tau=tau, repeat_cost=LOW_COST)
            exp_return = 0
            actions_tmp = []
            num_actions = 0
            while True:
                if max_actions is not None and num_actions >= max_actions:
                    _, rew, done, _=env._step(env.term_action)
                else:
                    possible_actions = list(env.actions(env._state))
                    action_taken = max(possible_actions, key = voc_estimate)
                    _, rew, done, _=env._step(action_taken)
                exp_return+=rew
                num_actions += 1
                if done:
                    break
            cumreturn += exp_return
            del env
        print("Return", cumreturn/num_episodes)
        return - (cumreturn/num_episodes)

    print(cost_function)
    np.random.seed(123456)

    if cost_function == "Actionweight" or cost_function == "Independentweight":
        space = [{'name': 'w1', 'type': 'continuous', 'domain': (0,1)},
            {'name': 'w2', 'type': 'continuous', 'domain': (0,1)},
            {'name': 'w4', 'type': 'continuous', 'domain': (1,len(TREE)-1)},
            {'name': 'w0', 'type': 'continuous', 'domain': (0,1)}]
    else:
        space = [{'name': 'w1', 'type': 'continuous', 'domain': (0,1)},
            {'name': 'w2', 'type': 'continuous', 'domain': (0,1)},
            {'name': 'w4', 'type': 'continuous', 'domain': (1,len(TREE)-1)}]

    constraints = [{'name': 'part_1', 'constraint': 'x[:,0] + x[:,1] - 1'}]

    feasible_region = GPyOpt.Design_space(space = space, constraints = constraints)

    initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, samples)
    # --- CHOOSE the objective
    objective = GPyOpt.core.task.SingleObjective(blackbox_original_mouselab)

    # --- CHOOSE the model type
    #This model does Maximum likelihood estimation of the hyper-parameters.
    model = GPyOpt.models.GPModel(exact_feval=True,optimize_restarts=10,verbose=False)

    # --- CHOOSE the acquisition optimizer
    aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)

    # --- CHOOSE the type of acquisition
    acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)

    # --- CHOOSE a collection method
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

    bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, initial_design)

    # --- Stop conditions
    max_time  = None
    tolerance = 1e-6     # distance between two consecutive observations        

    # Run the optimization
    max_iter  = iterations
    time_start = time.time()
    train_tic = time_start
    bo.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbosity=True)

    W_low = np.array([bo.x_opt])
    train_toc = time.time()

    print("\nSeconds:", train_toc-train_tic)
    print("Weights:", W_low)
    blackbox_original_mouselab(W_low)

    return W_low, train_toc-train_tic

def eval(W_low, n, TREE, INIT, LOW_COST, SEED=1000, term_belief=False, log=True, exact_seed=False, cost_function="Basic", max_actions=None, tau=1):
    """Evaluates the BMPS weights and logs the execution time.

    Args:
        W_low (np.array): BMPS weights
        n (int): Number of episodes for evaluation.
        TREE ([int]): MDP structure
        INIT ([int]): MDP reward distribution per node
        LOW_COST (int): Cost of computing a non goal node.
        SEED (int, optional): Seed to fix random MDP initialization. Defaults to 1000.

    Returns:
        [int]: Reward of each episode
        [[int]]: Actions of each episode
    """

    eval_tic = time.time()
    rewards, actions = original_mouselab(W=W_low, TREE=TREE, INIT=INIT, LOW_COST=LOW_COST, SEED=SEED, num_episodes=n, term_belief=term_belief, exact_seed=exact_seed, cost_function=cost_function, max_actions=max_actions, tau=tau)
    if log:
        print("Seconds:", time.time() - eval_tic)
        print("Average reward:", np.mean(rewards))
    return rewards, actions

def trace(W, TREE, INIT, LOW_COST, SEED=1, term_belief=False, cost_function="Basic", max_actions=None, ground_truth=None, tau=1):
    """[summary]

    Args:
        W ([type]): [description]
        TREE ([int]): MDP structure
        INIT ([int]): MDP reward distribution per node
        LOW_COST (int): Cost of computing a node.
        num_episodes (int, optional): Number of episodes to evaluate the MDP on. Defaults to 100.
        SEED (int, optional): Seed to fix random MDP initialization.
        term_belief (bool, optional): If true the expected reward instead of the real reward is used. Defaults to False.

    Returns:
        [int]: Reward of each episode
        [[int]]: Actions of each episode 
    """

    w1 = W[:,0]
    w2 = W[:,1]
    w4 = W[:,2]
    if cost_function == "Actionweight" or cost_function == "Independentweight":
        w0 = W[:,3]
    w3 = 1 - w1 - w2

    if cost_function == "Basic":
        simple_cost = True
    else:
        simple_cost = False
    
    num_nodes = len(TREE) - 1

    def voc_estimate(x):
        features = env.action_features(x)
        if cost_function == "Basic":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0]
        elif cost_function == "Hierarchical":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*features[0][1] + w2*features[0][2])
        elif cost_function == "Actionweight":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(features[0][0] + w0*(features[0][1]))
        elif cost_function == "Novpi":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*(features[0][1]/num_nodes))
        elif cost_function == "Proportional":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(features[0][0] + (features[0][1]/num_nodes))
        elif cost_function == "Independentweight":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0][0] + w0*features[0][1]
        else:
            assert False
    
    rewards = []
    actions = []
    np.random.seed(SEED)
    env = MouselabPar(TREE, INIT, cost=LOW_COST, term_belief=term_belief, tau=tau, repeat_cost=LOW_COST)
    if ground_truth is not None: 
        env.ground_truth = ground_truth
    print(f"Ground truth: {env.ground_truth}")
    exp_return = 0
    actions_tmp = []
    num_actions = 0
    while True:
        print_state = [(x.mu, x.sigma) if hasattr(x, "sample") else (x,) for x in env._state]
        print(f"\nStep {len(actions_tmp)}")
        print(f"Current state: {print_state}")
        if max_actions is not None and num_actions >= max_actions:
            _, rew, done, _=env._step(env.term_action)
            print(f"Max actions reached. Terminating with {rew} term reward")
        else:
            possible_actions = list(env.actions(env._state))
            vocs = [voc_estimate(x) for x in possible_actions]
            print(f"Action values: {vocs}")
            action_taken = max(possible_actions, key = voc_estimate)
            _, rew, done, _=env._step(action_taken)
            print(f"Chosen action {action_taken} with reward {rew}")
            if not done:
                print(f"Observation: {env.obs_list[action_taken][-1]}; Distribution: {env._state[action_taken]}")
            else:
                print(f"Terminated with total reward {exp_return}")
        exp_return+=rew
        num_actions += 1
        if done:
            break
        else:
            actions_tmp.append(action_taken)
    rewards.append(exp_return)
    actions.append(actions_tmp)
    return rewards, actions, env


def optimize_myopic(TREE, INIT, LOW_COST, evaluated_episodes=100, samples=10, iterations=100, SEED=0, term_belief=True, exact_seed=False, max_actions=None, tau=1, myopic_mode="old"):
    """Optimizes the weights for BMPS using Bayesian optimization.

    Args:
        TREE ([int]): MDP structure
        INIT ([int]): MDP reward distribution per node
        LOW_COST (int): Cost of computing a node.
        samples (int, optional): Number of initial random guesses before optimization. Defaults to 30.
        iterations (int, optional): Number of optimization steps. Defaults to 50.
        SEED (int, optional): Seed to fix random MDP initialization.
        term_belief (bool, optional): If true the expected reward instead of the real reward is used. Defaults to True.
    """

    def blackbox_original_mouselab(W):
        w1 = W[:,0]
        w4 = W[:,1]

        num_episodes = evaluated_episodes
        print("Weights", W)

        def voc_estimate(x):
            myopic, cost = env.myopic_action_feature(x)
            return w1*myopic  + w4*cost
        
        cumreturn = 0
        for i in range(num_episodes):
            #TODO
            if exact_seed:
                np.random.seed(SEED + i)
            else:
                np.random.seed(1000*SEED + i)
            env = MouselabPar(TREE, INIT, cost=LOW_COST, term_belief=term_belief, tau=tau, repeat_cost=LOW_COST, myopic_mode=myopic_mode)
            exp_return = 0
            num_actions = 0
            while True:
                if max_actions is not None and num_actions >= max_actions:
                    _, rew, done, _=env._step(env.term_action)
                else:
                    possible_actions = list(env.actions(env._state))
                    action_taken = max(possible_actions, key = voc_estimate)
                    _, rew, done, _=env._step(action_taken)
                exp_return+=rew
                num_actions += 1
                if done:
                    break
            cumreturn += exp_return
            del env
        print("Return", cumreturn/num_episodes)
        return - (cumreturn/num_episodes)

    np.random.seed(123456)

    space = [{'name': 'w1', 'type': 'continuous', 'domain': (0,1)},
        {'name': 'w4', 'type': 'continuous', 'domain': (1,len(TREE)-1)}]

    feasible_region = GPyOpt.Design_space(space = space)

    initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, samples)
    # --- CHOOSE the objective
    objective = GPyOpt.core.task.SingleObjective(blackbox_original_mouselab)

    # --- CHOOSE the model type
    #This model does Maximum likelihood estimation of the hyper-parameters.
    model = GPyOpt.models.GPModel(exact_feval=True,optimize_restarts=10,verbose=False)

    # --- CHOOSE the acquisition optimizer
    aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)

    # --- CHOOSE the type of acquisition
    acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)

    # --- CHOOSE a collection method
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

    bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, initial_design)

    # --- Stop conditions
    max_time  = None
    tolerance = 1e-6     # distance between two consecutive observations        

    # Run the optimization
    max_iter  = iterations
    time_start = time.time()
    train_tic = time_start
    bo.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbosity=True)

    W_low = np.array([bo.x_opt])
    train_toc = time.time()

    print("\nSeconds:", train_toc-train_tic)
    print("Weights:", W_low)
    blackbox_original_mouselab(W_low)

    return W_low, train_toc-train_tic

def eval_myopic(W, n, TREE, INIT, LOW_COST, SEED=1000, term_belief=False, log=True, exact_seed=False, max_actions=None, tau=1, myopic_mode="old"):
    """Evaluates the BMPS weights and logs the execution time.

    Args:
        W_low (np.array): BMPS weights
        n (int): Number of episodes for evaluation.
        TREE ([int]): MDP structure
        INIT ([int]): MDP reward distribution per node
        LOW_COST (int): Cost of computing a non goal node.
        SEED (int, optional): Seed to fix random MDP initialization. Defaults to 1000.

    Returns:
        [int]: Reward of each episode
        [[int]]: Actions of each episode
    """

    eval_tic = time.time()
    
    w1 = W[:,0]
    w4 = W[:,1]

    num_episodes = n
    print("Myopic mode:", myopic_mode, "Weights:", W)

    def voc_estimate(x):
        myopic, cost = env.myopic_action_feature(x)
        return w1*myopic  + w4*cost
    
    rewards = []
    actions = []
    for i in range(num_episodes):
        if exact_seed:
            np.random.seed(SEED + i)
        else:
            np.random.seed(1000*SEED + i)
        env = MouselabPar(TREE, INIT, cost=LOW_COST, term_belief=term_belief, tau=tau, repeat_cost=LOW_COST, myopic_mode=myopic_mode)
        exp_return = 0
        actions_tmp = []
        num_actions = 0
        while True:
            if max_actions is not None and num_actions >= max_actions:
                _, rew, done, _=env._step(env.term_action)
            else:
                possible_actions = list(env.actions(env._state))
                action_taken = max(possible_actions, key = voc_estimate)
                _, rew, done, _=env._step(action_taken)
            exp_return+=rew
            num_actions += 1
            if done:
                break
            else:
                actions_tmp.append(action_taken)
        rewards.append(exp_return)
        actions.append(actions_tmp)
        del env, possible_actions
    
    if log:
        print("Seconds:", time.time() - eval_tic)
        print("Average reward:", np.mean(rewards))
    return rewards, actions

def run_myopic(truth, samples, TREE, INIT, TAU, COST, myopic_mode="normal"):
    """Runs myopic strategy discovery on predefined samples
    """
    env = MouselabPar(tree=TREE, init=INIT, cost=COST, tau=TAU, repeat_cost=COST, myopic_mode=myopic_mode, term_belief=False)
    # Deep copy 
    truth = copy.deepcopy(truth)
    samples = copy.deepcopy(samples)
    env.ground_truth = truth
    done = False
    i = 0
    rew = 0
    expected_term_reward = 0
    acts = []
    repeat_clicks = 0
    while not done:
        available_actions = list(env.actions(env._state))
        expected_reward = [sum(env.myopic_action_feature(a)) for a in available_actions]
        action = available_actions[np.argmax(expected_reward)]
        if action is not env.term_action:
            obs = samples[action].pop()
        else:
            obs = None
            expected_term_reward = env.expected_term_reward(env._state)
        state, reward, done, _ = env._step(action, obs=obs)
        # if action is not env.term_action:
        #     print("Actions, sample, posterior", action, obs, state[action].mu)
        rew += reward
        if action in acts:
            repeat_clicks += 1
        acts.append(action)
        i += 1
    expected_term_reward -= COST * len(acts)
    return rew, acts, expected_term_reward, repeat_clicks

def eval_myopic(N, TREE, INIT, TAU, COST, seed=None, max_actions=200, cost_weight=0.5, log=False):
    def sum_features(features):
        assert len(features) == 2
        return (1-cost_weight)*features[0] + cost_weight*features[1]

    def get_action(env: MouselabPar):
        available_actions = list(env.actions(env._state))
        expected_action_rewards = [sum_features(env.myopic_action_feature(a)) for a in available_actions]
        action = available_actions[np.argmax(expected_action_rewards)]
        if log:
            print(f"Action {action} ({np.round(np.max(expected_action_rewards), 2)}), term reward {np.round(expected_action_rewards[-1], 2)}")
        return action
    
    return eval_generic(N, TREE, INIT, TAU, COST, get_action, seed=seed, max_actions=max_actions, log=log)

def eval_myopic_n(N, TREE, INIT, TAU, COST, seed=None, max_actions=200, cost_weight=0.5, log=False):
    def sum_features(features):
        assert len(features) == 2
        return (1-cost_weight)*features[0] + cost_weight*features[1]

    def get_action(env: MouselabPar):
        available_actions = list(env.actions(env._state))
        expected_action_rewards = [sum_features(env.myopic_action_feature_n(a)) for a in available_actions]
        action = available_actions[np.argmax(expected_action_rewards)]
        if log:
            print(f"Action {action} ({np.round(np.max(expected_action_rewards), 2)}), term reward {np.round(expected_action_rewards[-1], 2)}")
        return action
    
    return eval_generic(N, TREE, INIT, TAU, COST, get_action, seed=seed, max_actions=max_actions, log=log)

def eval_time_cost_myopic(N, TREE, INIT, TAU, COST, seed=None, max_actions=200, cost_weight=0.5, time_cost=0, time_cost_scaling=False):
    def sum_features(a, env: MouselabPar):
        features = env.myopic_action_feature(a)
        assert len(features) == 2
        if (a != env.term_action) and time_cost_scaling:
            return (1-cost_weight)*features[0] + cost_weight*features[1] - cost_weight*abs(time_cost)
        elif a != env.term_action:
            return (1-cost_weight)*features[0] + cost_weight*features[1] - abs(time_cost)
        else:
            return (1-cost_weight)*features[0] + cost_weight*features[1]

    def get_action(env: MouselabPar):
        available_actions = list(env.actions(env._state))
        expected_action_rewards = [sum_features(a, env) for a in available_actions]
        action = available_actions[np.argmax(expected_action_rewards)]
        return action
    
    res = eval_generic(N, TREE, INIT, TAU, COST, get_action, seed=seed, max_actions=max_actions)
    res["TimeCostExpectedReward"] = res["ExpectedReward"] - (abs(time_cost) * (res["NumActions"]-1))
    return res

def eval_DP(N, TREE, INIT, TAU, COST, DP_DEPTH, DP_BINS, seed=None, max_actions=200):
    def get_action(env: MouselabPar):
        available_actions = list(env.actions(env._state))
        expected_action_rewards = [env.dp_discrete(env._state, action=a, n=DP_DEPTH, bins=DP_BINS) for a in available_actions]
        action = available_actions[np.argmax(expected_action_rewards)]
        return action
    
    return eval_generic(N, TREE, INIT, TAU, COST, get_action, seed=seed, max_actions=max_actions)

def eval_generic(N, TREE, INIT, TAU, COST, get_action, seed=None, max_actions=200, log=False) -> pd.DataFrame:
    """Runs myopic strategy discovery on predefined samples
    """
    run_data = []
    run_data_columns = ["TrueReward", "ExpectedReward", "NumActions", "NumRepeatActions", "NumImmediateRepeatActions", "EnvSeed", "Time"]

    for i in range(N):
        if seed != None:
            np.random.seed(seed + i)
        env = MouselabPar(tree=TREE, init=INIT, cost=COST, tau=TAU, repeat_cost=COST, myopic_mode="normal", term_belief=False, max_actions=max_actions)
        done = False
        true_reward = 0
        expected_reward = 0
        actions = []
        repeat_actions = 0
        immediate_repeat_actions = 0
        start_time = time.process_time()
        while not done:
            action = get_action(env)
            if action is env.term_action:
                expected_reward += env.expected_term_reward(env._state)
            elif log:
                prior = env._state[action].mu
            _, reward, done, obs = env._step(action)
            if log and action is not env.term_action:
                print(f"Obs {obs}, prior {prior}, posterior {env._state[action].mu, env._state[action].sigma}, true reward {env.ground_truth[action]}")
            true_reward += reward
            expected_reward -= COST
            if len(actions):
                if action in actions:
                    repeat_actions += 1
                if action == actions[-1]:
                    immediate_repeat_actions += 1
            actions.append(action)
        runtime = time.process_time() - start_time
        if log:
            print(f"Actions: {actions}")
            print(f"Ground truth: {env.ground_truth}")
            print(f"Path {list(env.optimal_paths())}")
        if seed != None:
            run_data.append([true_reward, expected_reward, len(actions), repeat_actions, immediate_repeat_actions, seed+i, runtime])
        else:
            run_data.append([true_reward, expected_reward, len(actions), repeat_actions, immediate_repeat_actions, None, runtime])
    return pd.DataFrame(run_data, columns=run_data_columns)

def eval_pouct(N, TREE, INIT, TAU, COST, steps=10000, rollout_depth=5, exploration_coeff=10000, discretize=True, bins=8, eps=0.00001, seed=None, max_actions=200, add_metadata=True, log=False):
    def get_action(env: MouselabPar):
        pouct = POUCT(env, steps, rollout_depth, exploration_coeff, discretize, bins, eps)
        action = pouct.search(log=log)
        return action
    
    res = eval_generic(N, TREE, INIT, TAU, COST, get_action, seed=seed, max_actions=max_actions, log=log)

    if add_metadata:
        # General
        res["Tau"] = TAU
        res["Cost"] = COST
        res["MaxActions"] = max_actions
        # PO UCT
        res["Steps"] = steps
        res["RolloutDepth"] = rollout_depth
        res["ExplorationCoeff"] = exploration_coeff
        res["Discretize"] = discretize
        res["Bins"] = bins
    return res

def analyse_generic(TREE, INIT, TAU, COST, seed=None, max_actions=200, myopic_mode="normal", cost_weight=0.5) -> pd.DataFrame:
    """Runs myopic strategy discovery on predefined samples
    """
    def sum_features(features):
        assert len(features) == 2
        return (1-cost_weight)*features[0] + cost_weight*features[1]

    run_data = []
    run_data_columns = ["Action", "VOC", "Observation", "ExpTermReward"]
    belief_states = []

    if seed != None:
        np.random.seed(seed)
    
    env = MouselabPar(tree=TREE, init=INIT, cost=COST, tau=TAU, repeat_cost=COST, myopic_mode=myopic_mode, term_belief=True, max_actions=max_actions)
    done = False
    total_reward = 0
    while not done:
        available_actions = list(env.actions(env._state))
        expected_action_rewards = [sum_features(env.myopic_action_feature(a)) for a in available_actions]
        action = available_actions[np.argmax(expected_action_rewards)]
        _, reward, done, obs = env._step(action)
        total_reward += reward
        run_data.append([action, np.max(expected_action_rewards), obs, env.expected_term_reward(env._state)])
        belief_states.append(env._state)

    return pd.DataFrame(run_data, columns=run_data_columns), belief_states, total_reward
