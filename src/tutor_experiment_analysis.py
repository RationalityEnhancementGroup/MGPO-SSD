### Data analysis script for the human training experiment
### Raw data is unpublished because it contains unique participant identifiers

import json
import pandas as pd
import numpy as np
from toolz import memoize
import datetime

from src.utils.distributions import Normal
from src.utils.mouselab_PAR import MouselabPar
from tqdm import tqdm

N_TEST_TRIALS = 10
N_TRAINING_TRIALS = 12

ENV_PATH = "data/environments/"
DATA_PATH = "data/tutor_experiment/"

# Restore previous MGPO version to reproduce exact results
MYOPIC_MODE = "old"

tolerance = 0.05

# (cost, tau) per condition
param_mapping = json.load(open(ENV_PATH+"params.json"))
param_mapping = {int(k):tuple(v) for k,v in param_mapping.items()}

# Load samples shown to participant to calculate posteriors
def get_samples(counterbalance, cost, tau, trial_id):
    if counterbalance < 10:
        counterbalance = "0"+str(counterbalance)
    filename = f"{str(counterbalance)}_{str(cost)[2:]}_{str(tau)[2:]}.json"
    data = json.load(open(ENV_PATH+"/experiment_instances/4_60_json/"+filename))
    for trial in data:
        if trial["trial_id"] is trial_id:
            samples = trial["samples"]
            return {int(k):v for k,v in samples.items()}

# Node distribution used in experiment
d0 = Normal(0,5)
d1 = Normal(0,10)
d2 = Normal(0,20)
node_types = [d0, 
    d0, d0, d0, d0, d0, d0, d0, d1, d1, d1, d1, d1, d1, d1, d2, 
    d0, d0, d0, d0, d0, d0, d0, d1, d1, d1, d1, d1, d1, d1, d2, 
    d0, d0, d0, d0, d0, d0, d0, d1, d1, d1, d1, d1, d1, d1, d2, 
    d0, d0, d0, d0, d0, d0, d0, d1, d1, d1, d1, d1, d1, d1, d2]
TREE = [[1, 16, 31, 46], [2, 3, 4, 5], [6], [6], [7], [7], [8], [8], [9, 10, 11, 12], [13], [13], [14], [14], [15], [15], [], [17, 18, 19, 20], [21], [21], [22], [22], [23], [23], [24, 25, 26, 27], [28], [28], [29], [29], [30], [30], [], [32, 33, 34, 35], [36], [36], [37], [37], [38], [38], [39, 40, 41, 42], [43], [43], [44], [44], [45], [45], [], [47, 48, 49, 50], [51], [51], [52], [52], [53], [53], [54, 55, 56, 57], [58], [58], [59], [59], [60], [60], []]
term_click = len(TREE)

level_1_clicks = [i for i,x in enumerate(node_types) if x.sigma==5 and i != 0]
level_2_clicks = [i for i,x in enumerate(node_types) if x.sigma==10 and i != 0]
level_3_clicks = [i for i,x in enumerate(node_types) if x.sigma==20 and i != 0]

def calculate_goal_strategy(clicks):
    clicks = [int(c) for c in clicks]
    l1, l2, l3 = 0, 0, 0
    for click in clicks:
        if click in level_3_clicks:
            l3 += 1
        elif click in level_2_clicks:
            l2 += 1
            if l3 == 0:
                return False
        elif click in level_1_clicks:
            l1 += 1
            if l3 == 0 or l2 == 0:
                return False
        else:
            assert click==61
    return True

# Calculate expected reward after observations
def observe(node, obs, tau):
    mean_old = node.mu
    sigma_old = node.sigma
    tau_old = 1 / (sigma_old ** 2)
    tau_new = tau_old + len(obs)*tau
    mean_new = ((mean_old * tau_old) + tau*sum(obs)) / tau_new
    #sigma_new = 1 / math.sqrt(tau_new)
    return mean_new

# Parse expected scores from trial data
def get_expected_score(trialdata, cost, tau, trial_id, counterbalance):
    path = trialdata["path"]
    queries = trialdata["queries"]["click"]["state"]["target"]
    # Replace first element with 0 for root
    ground_truth =  [0] + trialdata["stateRewards"][1:]
    reward = 0
    samples = get_samples(counterbalance, cost, tau, trial_id)
    node_rewards = []
    clicked_node_rewards = 0
    for node in path:
        # Calculate posterior mean given the observed samples
        if node in queries or (int(node) in queries):
            clicks = queries.count(node)
            #print("Node", node, "number of clicks", clicks, "tau", tau)
            node_dist = node_types[int(node)]
            observed_samples = samples[int(node)][0:clicks]
            expected_reward = observe(node_dist, observed_samples, tau)
            #print("Node", node, "observation", observed_samples, "expectation", expected_reward)
            #print("Observations", observed_samples, "posterior", expected_reward)
            reward += expected_reward
            node_rewards.append(expected_reward)
            clicked_node_rewards += ground_truth[int(node)]
        # Use mean for unobserved nodes
        else:
            reward += node_types[int(node)].expectation() #0
            node_rewards.append(node_types[int(node)].expectation())
            clicked_node_rewards += node_types[int(node)].expectation()
    #print(node_rewards)
    #print(reward)
    #print("Path reward", reward, "cost", len(queries) * cost)
    # Click cost
    click_costs = len(queries) * cost
    reward -= click_costs
    clicked_node_rewards -= click_costs

    # Optimal reward based on clicks
    env = MouselabPar(TREE, tuple(node_types), tau=tau, myopic_mode=MYOPIC_MODE)
    env.ground_truth=ground_truth
    env.samples = {int(k):v for k,v in samples.items()}
    for node in queries:
        env._step(int(node))
    expected_optimal_reward = env.expected_term_reward(env._state) - click_costs

    return reward, expected_optimal_reward, clicked_node_rewards

@memoize(key=lambda args, kwargs: args[1:])
def compute_optimal_policy(trialdata, cost, tau, trial_id, counterbalance):
    ground_truth = [0] + trialdata["stateRewards"][1:] # Replace root node with 0
    samples = get_samples(counterbalance, cost, tau, trial_id)
    env = MouselabPar(TREE, tuple(node_types), ground_truth=ground_truth, cost=cost, repeat_cost=cost, tau=tau, samples=samples, myopic_mode=MYOPIC_MODE)
    done = False
    rewards = 0
    actions = []
    repeat_actions = 0
    while not done:
        available_actions = list(env.actions(env._state))
        expected_reward = [sum(env.myopic_action_feature(a)) for a in available_actions]
        action = available_actions[np.argmax(expected_reward)]
        state, reward, done, obs = env._step(action)
        rewards += reward
        if action in actions:
            repeat_actions += 1
        actions.append(action)
    return rewards, len(actions), repeat_actions, actions

def compute_click_agreement(trialdata, cost, tau, trial_id, counterbalance):
    clicks = [int(x) for x in trialdata["queries"]["click"]["state"]["target"]]
    ground_truth = [0] + trialdata["stateRewards"][1:] # Replace root node with 0
    samples = get_samples(counterbalance, cost, tau, trial_id)
    env = MouselabPar(TREE, tuple(node_types), ground_truth=ground_truth, cost=cost, repeat_cost=cost, tau=tau, samples=samples, myopic_mode=MYOPIC_MODE)
    agreement = []
    term_agreement = []
    strict_repeat_agreement = []
    repeat_agreement = []
    repeat_clicks = 0
    for i in range(len(clicks)+1):
        # Compute optimal actions
        available_actions = list(env.actions(env._state))
        expected_reward = [sum(env.myopic_action_feature(a)) for a in available_actions]
        max_reward = max(expected_reward)
        optimal_clicks = [available_actions[i] for i in range(len(available_actions)) if np.isclose(expected_reward[i], max_reward, atol=.05, rtol=0)]

        subclicks = clicks[0:i]
        if i < len(clicks):
            next_click = clicks[i]
            env._step(next_click)
        else:
            next_click = term_click
        
        # Overall agreement
        if next_click in optimal_clicks:
            agreement.append(1)
        else:
            agreement.append(0)
        # Term agreement
        if next_click == term_click:
            if next_click in optimal_clicks:
                term_agreement.append("tp")
            else:
                term_agreement.append("fp")
        else: # Not terminating was incorrect if termination is optimal
            if term_click in optimal_clicks:
                term_agreement.append("fn")
            else:
                term_agreement.append("tn")
        
        # Count repeat clicks
        if next_click in subclicks:
            repeat_clicks += 1
        # Repeat click agreement
        if all(c in subclicks for c in optimal_clicks) and next_click in subclicks:
            repeat_agreement.append(1)
            if next_click in optimal_clicks:
                strict_repeat_agreement.append(1)
            else:
                strict_repeat_agreement.append(0)
        elif all(c in subclicks for c in optimal_clicks) or next_click in subclicks:
            repeat_agreement.append(0)
            strict_repeat_agreement.append(0)
        
    term_agreement_score = balanced_acc(term_agreement)
    repeat_agreement_score = agreement_score(repeat_agreement)
    strict_repeat_agreement_score = agreement_score(strict_repeat_agreement)
    total_agreement_score = agreement_score(agreement)
    assert len(agreement) == len(clicks)+1
    return total_agreement_score, repeat_agreement_score, strict_repeat_agreement_score, term_agreement_score, repeat_clicks

def agreement_score(scores: list):
    """ Calcualtes accuracy from a list of binary classification outcomes

    Args:
        scores (list): list of 1, 0
    """
    if len(scores) > 0:
        return np.mean(scores)
    else:
        return np.nan

def balanced_acc(scores: list):
    """ Calculates balanced accuracy from a list of classification outcomes

    Args:
        scores (list): list of "tp", "fp", "tn", "fn"
    """
    tp = sum([1 for score in scores if score == "tp"])
    fp = sum([1 for score in scores if score == "fp"])
    tn = sum([1 for score in scores if score == "tn"])
    fn = sum([1 for score in scores if score == "fn"])

    if (tp > 0 or fn > 0) and (tn > 0 or fp > 0):
        sensitivity = (tp / (tp+fn)) 
        specificity = (tn / (fp+tn)) 
        return 0.5 * (sensitivity + specificity)
    # Normal acc if balanced measure not applicable (i.e. participant only performed one action)
    elif (tp > 0 or fn > 0) or (tn > 0 or fp > 0):
        return ((tp+tn)/(tp+fn+tn+fp))
    # NaN if participant performed no applicable actions
    else:
        return np.nan

if __name__ == "__main__":
    # Load dataclip
    path = DATA_PATH + "po-choice-5.json"
    data = json.load(open(path))

    df_index = ["Participant", "Condition", "Counterbalance", "TrialId", "Score", "ExpectedScore", "ClickedScore", "NumClicks", "Actions", 
        "TestEnv", "OptimalScore", "ClickAgreement", "RepeatAgreement", "StrictRepeatAgreement", "TermAgreement", "RepeatClicks", "BmpsReward", "BmpsClick", "BmpsRepeatClicks", "BmpsActions", "Cost", "Tau"]    
    df_data = []

    bonus_data = {}
    known_workers = []
    good_responses = 0
    demographics = []

    demographics_data = []

    final_survey = []
    intermediate_survey = []
    demographic_survey = []

    language_index = data["fields"].index("language")
    response_data_index = data["fields"].index("datastring")
    begin_index = data["fields"].index("beginhit")
    end_index = data["fields"].index("endhit")

    f = '%Y-%m-%d %H:%M:%S.%f'

    # Parse raw mturk data into dataframe
    for p_index, p_data in tqdm(enumerate(data["values"])):
        # Filter out empty responses
        language = p_data[language_index]
        response_data = p_data[response_data_index]
        if p_data[begin_index] and p_data[end_index]:
            begin = datetime.datetime.strptime(p_data[begin_index], f)
            end = datetime.datetime.strptime(p_data[end_index], f)
            duration = (end - begin).total_seconds()
        else:
            duration = None
        if response_data != None:
            p_res_obj = json.loads(response_data)
            condition = p_res_obj["condition"]
            counterbalance = p_res_obj["counterbalance"]
            cost, tau = param_mapping[counterbalance]
            # Obfuscate worker ID for publishing
            worker = p_index #p_res_obj["workerId"]# 
            if worker in known_workers:
                print("Duplicate worker", worker)
            else: 
                known_workers.append(worker)
            p_res = p_res_obj["data"]
            participant_responses = []
            if "quiz_failures" in p_res_obj["questiondata"].keys():
                quiz_failures = p_res_obj["questiondata"]["quiz_failures"]
            else:
                quiz_failures = 0
            if "final_bonus" in p_res_obj["questiondata"].keys():
                bonus =  p_res_obj["questiondata"]["final_bonus"]
            else:
                bonus = 0
            participant_survey = {"Participant": worker, "Condition": condition, "Counterbalance": counterbalance, "Language": language, "QuizAttempts": 0, "QuizFailures": quiz_failures, "Bonus": bonus, "Duration": duration}
            for i in range(len(p_res)):
                # Get test trials
                if 'block' in p_res[i]['trialdata'].keys() and p_res[i]['trialdata']['block'] == "test_main":
                    trial = p_res[i]
                    trialdata = trial["trialdata"]
                    assert trialdata["trial_type"] == "mouselab-mdp"
                    trialid = trialdata["trial_id"]
                    queries = trialdata["queries"]["click"]["state"]["target"]
                    path = trialdata["path"]
                    score = trialdata["score"]
                    trial_id = int(trialdata["trial_id"])
                    expected_score, optimal_score, clicked_score = get_expected_score(trialdata, cost, tau, trial_id, counterbalance)
                    total_agreement_score, repeat_agreement_score, strict_repeat_agreement_score, term_agreement_score, repeat_clicks = compute_click_agreement(trialdata, cost, tau, trial_id, counterbalance)
                    bmps_reward, bmps_clicks, bmps_repeat_clicks, bmps_actions = compute_optimal_policy(trialdata, cost, tau, trialid, counterbalance)
                    participant_responses.append([worker, condition, counterbalance, trialid, score, expected_score, clicked_score, len(queries), queries, trial_id, optimal_score, total_agreement_score, repeat_agreement_score, 
                        strict_repeat_agreement_score, term_agreement_score, repeat_clicks, bmps_reward, bmps_clicks, bmps_repeat_clicks, bmps_actions, cost, tau])
                if p_res[i]['trialdata']["trial_type"] == "survey-text":
                    # Final survey
                    questions = ['What is your age?', 'What gender do you identify with?', 'Any issues with the tutor?', 'Any comments/feedback?']
                    for question, answer in zip(questions, p_res[i]['trialdata']["response"].values()):
                        participant_survey[question] = answer
                if p_res[i]['trialdata']["trial_type"] == "survey-multi-choice":
                    # Survey after test trials
                    if len(p_res[i]['trialdata']["response"]) == 6:
                        questions = ["Was it necessary to click airports multiple times to achieve a high reward?", "Which airports should be clicked first?", "How enjoyable was it to learn strategies in the training environments?", "How useful do you think the training environments were for you to learn a good strategy?", "Have you participated this type of planning experiment in the past?", "Did you try your best to achieve a high reward?"]
                        for question, answer in zip(questions, p_res[i]['trialdata']["response"].values()):
                            participant_survey[question] = answer
                    # Training_survey
                    elif len(p_res[i]['trialdata']["response"]) == 5:
                        # Count training quiz attempts
                        participant_survey["QuizAttempts"] = participant_survey["QuizAttempts"] + 1
                    else:
                        print("Unknown survey response")
            demographics_data.append(participant_survey)
            # Filter out incomplete participant responses
            if len(participant_responses) == N_TEST_TRIALS:
                good_responses += 1
                for d in participant_responses:
                    df_data.append(d)

    print("Good responses", good_responses)
    df = pd.DataFrame(df_data, columns=df_index)
    questionnaire_df = pd.DataFrame(demographics_data)

    df["BmpsGoalStrategy"] = df["BmpsActions"].apply(calculate_goal_strategy)
    df["GoalStrategy"] = df["Actions"].apply(calculate_goal_strategy)
    
    # Check BMPS is following the goal strategy
    #assert all(df["BmpsGoalStrategy"])

    questionnaire_df["CompleteTrialData"] = questionnaire_df["Participant"].isin(df["Participant"].unique().tolist())

    # Exclude participants who participated in similar experiments in the past
    repeats = list(questionnaire_df[(questionnaire_df["CompleteTrialData"]) & (questionnaire_df["Have you participated this type of planning experiment in the past?"] == "Yes")]["Participant"])
    # Exclude participants who didn't try their best
    inattentive = list(questionnaire_df[(questionnaire_df["CompleteTrialData"]) & (questionnaire_df["Did you try your best to achieve a high reward?"] == "No")]["Participant"])
    # Exclude 3 quiz attempts
    quiz_fail = list(questionnaire_df[(questionnaire_df["CompleteTrialData"]) & (questionnaire_df["QuizAttempts"]>3)]["Participant"])
    #participants = pd.DataFrame(df[df["NumClicks"] == 0].groupby("Participant").count()["NumClicks"])
    low_clicks = []#list(participants[participants["NumClicks"]>(float(N_TEST_TRIALS)/2)].index)
    excluded = list(set(repeats+inattentive+quiz_fail+low_clicks))
    df["Excluded"] = df["Participant"].isin(excluded)
    df_after_exclusion = df[~df["Participant"].isin(excluded)]
    df_after_exclusion.to_csv(DATA_PATH+"tutor_experiment_exclusion_data.csv")
    df.to_csv(DATA_PATH+"tutor_experiment_full_data.csv")
    print(f"Exlcuded participants ({len(excluded)}: {len(repeats)} repeats, {len(inattentive)} inattentive, {len(quiz_fail)} quiz failure, {len(low_clicks)} low clicks): {excluded}")
    print("Total remaining", len(df_after_exclusion) / N_TEST_TRIALS)
    print("Exclusions per condition:", df[df["Participant"].isin(excluded)].groupby("Condition").count()["Participant"] / N_TEST_TRIALS)
    print("Average bonus per condition", questionnaire_df.groupby("Condition").mean()["Bonus"])
    # Bonus calculation
    questionnaire_df[questionnaire_df["Bonus"]>0][["Participant", "Bonus"]].set_index("Participant").to_csv(DATA_PATH+"bonus.csv")

    questionnaire_df.to_csv(DATA_PATH+"questionnaire.csv")
