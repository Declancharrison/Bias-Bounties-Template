import sys
import os
import yaml
import dill as pkl
import pandas as pd
import folktables
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import top_k_accuracy_score as TOP_K
import io
import builtins
from contextlib import contextmanager
from io import StringIO
import dis
import numpy as np
from numpy import *
import sklearn
from sklearn import *
from xgboost import *
from lightgbm import *
from slack import WebClient
from slack.errors import SlackApiError
from inspect import isfunction
import csv
import gdown
import lightgbm
import datetime
from dotenv import load_dotenv
import docker
import shutil

#load
# leaderboard = pd.read_csv("leaderboard.csv", index_col = "Unnamed: 0")

load_dotenv()

SERVER_PATH = os.getenv("SERVER_PATH")
RUNNER_PATH = os.getenv("RUNNER_PATH")
REPO_NAME = os.getenv("REPO_NAME")
DATASET_TASK = os.getenv("DATASET_TASK")
LOSS_FN = os.getenv("LOSS_FN")

if LOSS_FN == "MAE":
    loss_fn = MAE
elif LOSS_FN == "MSE":
    loss_fn = MSE
elif LOSS_FN == "ACC":
    loss_fn = ACC
elif LOSS_FN == "TOP_K":
    loss_fn = TOP_K

sys.setrecursionlimit(2140000000)

sys.path.insert(0, SERVER_PATH)
import pdl

x_train = np.load(f'{SERVER_PATH}/data/training_data.npy') 
y_train = np.load(f'{SERVER_PATH}/data/training_labels.npy')
x_val   = np.load(f'{SERVER_PATH}/data/validation_data.npy') 
y_val   = np.load(f'{SERVER_PATH}/data/validation_labels.npy')

x_val_statistics, _, y_val_statistics, _ = train_test_split(x_val, y_val, test_size = .5, random_state = 23)

# safe_builtins = [
#     'range',
#     'complex',
#     'set',
#     'frozenset',
#     'slice',
#     '_load_type',
#     'getattr',
#     'setattr',
#     '__dict__',
#     '__main__'
# ]

# class RestrictedUnpickler(pkl.Unpickler):

#     def find_class(self, module, name):
#         if 'numpy' in module:
#             return getattr(sys.modules[module], name)
#         if 'scipy' in module:
#             return getattr(sys.modules[module], name)
#         if 'sklearn' in module or 'xgboost' in module or 'lightgbm' in module or 'collections' in module:
#             if 'predict' in name:
#                 return getattr(sys.modules[module], name.split('.')[0]).predict
#             if 'transform' in name:
#                 return getattr(sys.modules[module], name.split('.')[0]).transform
#             return getattr(sys.modules[module], name)
#         # Only allow safe classes from builtins.
#         if 'dill' in module:
#             return getattr(sys.modules[module], name)
#         if ("__builtin__" in module or "builtins" in module) and name in safe_builtins:
#             if name == '__main__':
#                 return
#             return getattr(builtins, name)
#         # Forbid everything else.
#         raise pkl.UnpicklingError("global '%s.%s' is forbidden" %
#                                      (module, name))

# def restricted_loads(s):
#     """Helper function analogous to pickle.loads()."""
#     return RestrictedUnpickler(io.BytesIO(s)).load()

# class AttrDict(dict):
#     def __init__(self, *args, **kwargs):
#         super(AttrDict, self).__init__(*args, **kwargs)
#         self.__dict__ = self

# def load_item(filepath):
#     with open(filepath, "rb") as file:
#         item_data = file.read()
#         item = restricted_loads(item_data)
#     return item

def get_slack_key():
    try:
        with open(f"{SERVER_PATH}/scripts/slack_key.txt", 'rb') as file:
            key = file.read()
        return key.decode('utf-8')
    except:
        return False


def send_slack_msg(team_name, where, flag, manual, group_weight, pre_group_error, h_group_error, model_error_reduction):
    try:
        slack_key = get_slack_key()
        slack_client = WebClient(slack_key)
        attempt_number = len(os.listdir(f"{SERVER_PATH}/all_pairs/groups"))+1
        msg = f"(Attempt {attempt_number}) --- Team Name: {team_name} --- {where} Model Information \n\tUpdate: {flag} \n\tManual Group: {manual} \n\tGroup Weight: {group_weight:.2f}% \n\tModel Group Error: {pre_group_error:.2f} \n\tHypothesis Group Error: {h_group_error:.2f} \n\tDelta Model/Hypothesis Error: {(pre_group_error - h_group_error):.2f} \n\tOverall Error Reduction: {model_error_reduction:.2f}"
        slack_client.chat_postMessage(channel="bounty-log-private", text=msg)
        with open(f"{SERVER_PATH}/scripts/log.csv", "a") as file:
            write_obj = csv.writer(file)
            write_obj.writerow([attempt_number, team_name, where, flag, manual, group_weight, pre_group_error, h_group_error, model_error_reduction, str(datetime.datetime.now())])
    except:
        print('Slack API not working, no message sent to teams')


def send_global_slack_msg(team_name, model_error_reduction):
    try:
        slack_key = get_slack_key()
        slack_client = WebClient(slack_key)
        msg = f"{team_name} reduced global error by {model_error_reduction:.2f}, BZ!"
        slack_client.chat_postMessage(channel="bias-bounty-project", text=msg)
    except:
        print('Slack API not working, no message sent to teams')

def get_team_name():
    # get github username from pr
    username = sys.argv[2].lower()

    # open username to team name map
    with open(f'{SERVER_PATH}/username_to_team.yaml') as file:
        team_information = yaml.load(file, yaml.loader.FullLoader)

    # get team name from github username
    try:
        team_name = team_information[username]
    except:
        add_comment(f"Username could not be associated with a team name, please contact admin staff!.\n")
        assert False
    return team_name

def load_pdl(user_path):
    # load team's pdl infrastructure
    try:
        team_pdl = pdl.load_model(user_path)
    except:
        add_comment("Problem loading pdl, contact administrator!\n")
        assert False

    return team_pdl

def security_call():
    client = docker.from_env()

    try:
        container = client.containers.get("bias_bounty_container")

        exec_response = container.exec_run("python3 security.py", detach = True)

        output = exec_response.output.decode("utf-8")

    except Exception as err:
        print(f"Error: {err}. Problem running object in docker container!")
  

def initialize_comment():
    with open(f'{SERVER_PATH}/tmp/{sys.argv[2]}_comment.txt', 'w') as file:
        file.write("")
    return 

def add_comment(msg):
    with open(f'{SERVER_PATH}/tmp/{sys.argv[2]}_comment.txt', 'a') as file:
        file.write(f"{msg}")
    return

def deepsave(obj, name):
    # deep save viable group/hypothesis pairs
    try:
        len_folder = len(os.listdir(f"{SERVER_PATH}/all_pairs/{name}"))
        with open(f'{SERVER_PATH}/all_pairs/{name}/saved_{name[0]}{len_folder}.pkl', 'wb') as file:
            pkl.dump(obj, file)
    except:
        add_comment("Error on deep save, contact administrator!\n")
        assert False

def team_model_update(team_pdl, group, hypothesis, user_path, team_name, global_delta):
    try:
        pre_model_error = np.sqrt(loss_fn(y_val,team_pdl.val_predictions))
        indices = group(x_val)
        pre_group_error = np.sqrt(loss_fn(y_val[indices],team_pdl.val_predictions[indices]))
        h_group_error = np.sqrt(loss_fn(y_val[indices],hypothesis(x_val[indices])))

        flag = team_pdl.update(group, hypothesis, x_train, y_train, x_val, y_val)

        if global_delta != 0:
            global_flag = True
        else:
            global_flag = False

        # log information
        where = "Team"
        group_weight = (100*indices.sum())/len(x_val)
        h_group_error = np.sqrt(loss_fn(y_val[indices],hypothesis(x_val[indices])))
        post_model_error = np.sqrt(loss_fn(y_val,team_pdl.val_predictions))
        model_error_reduction = pre_model_error - post_model_error
        manual = isfunction(group)
        # send_slack_msg(team_name, where, flag, manual, group_weight, pre_group_error, h_group_error, model_error_reduction)
    except:
        add_comment("Error in team model update process, contact administrator!\n")
        assert False

    if flag == False:
        add_comment("Private Update denied!\n")
        if global_delta != 0:
            update_leaderboard(team_name, team_model_error, 0, 1, global_delta)

    else:
        team_pdl.save_model(f'{user_path}/PDL')
        add_comment("Private Update Accepted!\n")

        team_model_error = np.sqrt(loss_fn(y_val, team_pdl.val_predictions))

        update_leaderboard(team_name, team_model_error, 1, int(global_flag), global_delta)

        team_train_predictions = team_pdl.train_predictions
        np.save(f"{SERVER_PATH}/models/{team_name}/training_predictions.npy", team_train_predictions)
       
    return flag

def global_model_update(team_name, group, hypothesis):
    # load team's pdl infrastructure
    try:
        global_pdl = load_pdl(f"{SERVER_PATH}/teams/global_pdl")
    except:
        add_comment("Problem loading global pdl, contact administrator!\n")
        assert False
    
    # check update
    try:
        # get pre update error
        pre_model_error = np.sqrt(loss_fn(y_val, global_pdl.val_predictions))
        indices = group(x_val)
        pre_group_error = np.sqrt(loss_fn(y_val[indices],global_pdl.val_predictions[indices]))
        h_group_error = np.sqrt(loss_fn(y_val[indices],hypothesis(x_val[indices])))

        flag = global_pdl.update(group, hypothesis, x_train, y_train, x_val, y_val)

        # log information
        where = "Global"
        group_weight = (100*indices.sum())/len(x_val)
        h_group_error = np.sqrt(loss_fn(y_val[indices],hypothesis(x_val[indices])))
        post_model_error = np.sqrt(loss_fn(y_val,global_pdl.val_predictions))
        model_error_reduction = pre_model_error - post_model_error
        manual = isfunction(group)
        # send_slack_msg(team_name, where, flag, manual, group_weight, pre_group_error, h_group_error, model_error_reduction)
    except:
        add_comment("Something went wrong in the (global) update process!\n")
        return False, 0
    
    if flag == True:
        
        global_val_statistics_predictions = global_pdl.predict(x_val_statistics)
        global_pdl_error = np.sqrt(loss_fn(y_val, global_pdl.val_predictions))
        delta = (pre_model_error - global_pdl_error)
        update_leaderboard('Global Model', global_pdl_error, 0, 1, delta)
        # update_global_statistics(x_val_statistics, y_val_statistics, global_val_statistics_predictions)
        
        global_pdl.save_model(f'{SERVER_PATH}/teams/global_pdl/PDL')

        global_train_predictions = global_pdl.train_predictions
        np.save(f"{SERVER_PATH}/models/global_model/training_predictions.npy", global_train_predictions)
        
        # send_global_slack_msg(team_name, delta)
        
        add_comment("Global Update Accepted!\n")

        return True, delta
    else:
        add_comment("Global Update Denied!\n")
        return False, 0
    
def test_len():
    if ((len(sys.argv[1].split(' ')) != 1) or ('request.yaml' not in sys.argv[1])):
        add_comment("Error: Multiple files changed. User may only change competitors/request.yaml.\n")
        assert False
    else:
        assert True
        return True

def test_urls():

    with open(f'{RUNNER_PATH}/competitors/request.yaml') as file:
        config = yaml.load(file, yaml.loader.FullLoader)

    g_url = config["g_url"]
    h_url = config["h_url"]

    if (os.path.isfile(f"{SERVER_PATH}/tmp/group.pkl") == True):
        os.system(f"rm -f {SERVER_PATH}/tmp/group.pkl")

    if (os.path.isfile(f"{SERVER_PATH}/tmp/hypothesis.pkl") == True):
        os.system(f"rm -f {SERVER_PATH}/tmp/group.pkl")
    try:
        if 'google' in g_url:
            output_file = f"{SERVER_PATH}/tmp/group.pkl"
            gdown.download(url=g_url, output=output_file, quiet=False, fuzzy=True)

        if 'google' in h_url:
            output_file = f"{SERVER_PATH}/tmp/hypothesis.pkl"
            gdown.download(url=h_url, output=output_file, quiet=False, fuzzy=True)
    except:
        pass

    if (os.path.isfile(f"{SERVER_PATH}/tmp/group.pkl") == True) and (os.path.isfile(f"{SERVER_PATH}/tmp/hypothesis.pkl") == True):
        assert True
        shutil.move(f"{SERVER_PATH}/tmp/group.pkl", f"{SERVER_PATH}/container_tmp/group.pkl")
        shutil.move(f"{SERVER_PATH}/tmp/hypothesis.pkl", f"{SERVER_PATH}/container_tmp/group.pkl")

        return True
    add_comment(f"Either group of model URLs are not downloaded, please ensure no authentication is used and links are copied over correctly!.\n")
    assert False
        
    
def test_update(flag, group, hypothesis):

    if not flag:
        if group:
            add_comment(group)
        if hypothesis:
            add_comment(hypothesis)

    team_name = get_team_name()
    
    user_path = f'{SERVER_PATH}/teams/{team_name}'

    # load team's pdl infrastructure
    team_pdl = load_pdl(user_path)

    # check update
    global_flag, global_delta = global_model_update(team_name, group, hypothesis)

    team_flag = team_model_update(team_pdl, group, hypothesis, user_path, team_name, global_delta)

    # deep save models
    deepsave(group, 'groups')
    deepsave(hypothesis, 'hypotheses')

    if global_flag == True or team_flag == True:

        push_updates_to_git(team_name, team_flag, global_flag)

        assert True
        return True
    else:
        assert False
    
def update_global_statistics(X, y, preds):
    aggregate_statistics = {
        # RACE
        'White'  : np.sqrt(loss_fn(y[X['RAC1P'] == 1], preds[X['RAC1P'] == 1])),
        'Black'  : np.sqrt(loss_fn(y[X['RAC1P'] == 2], preds[X['RAC1P'] == 2])),
        'Asian'  : np.sqrt(loss_fn(y[X['RAC1P'] == 6], preds[X['RAC1P'] == 6])),
        'Other'  : np.sqrt(loss_fn(y[(X['RAC1P'] != 1) & (X['RAC1P'] != 2) & (X['RAC1P'] != 6)], preds[(X['RAC1P'] != 1) & (X['RAC1P'] != 2) & (X['RAC1P'] != 6)])),
        
        # SEX
        'Male'   : np.sqrt(loss_fn(y[X['SEX'] == 1], preds[X['SEX'] == 1])),
        'Female' : np.sqrt(loss_fn(y[X['SEX'] == 2], preds[X['SEX'] == 2])),

        # AGE
        'Under 25'  : np.sqrt(loss_fn(y[X['AGEP'] <= 25], preds[X['AGEP'] <= 25])),
        '25 - 40'   : np.sqrt(loss_fn(y[(X['AGEP'] >= 25) & (X['AGEP'] <= 40)], preds[(X['AGEP'] >= 25) & (X['AGEP'] <= 40)])),
        '40 - 60'   : np.sqrt(loss_fn(y[(X['AGEP'] >= 40) & (X['AGEP'] <= 60)], preds[(X['AGEP'] >= 40) & (X['AGEP'] <= 60)])),
        '60 - 75'   : np.sqrt(loss_fn(y[(X['AGEP'] >= 60) & (X['AGEP'] <= 75)], preds[(X['AGEP'] >= 60) & (X['AGEP'] <= 75)])),
        'Over 75'   : np.sqrt(loss_fn(y[X['AGEP'] >= 75], preds[X['AGEP'] >= 75])),

        # TYPE OF WORKER
        'Private Profit'            : np.sqrt(loss_fn(y[X['COW'] == 1], preds[X['COW'] == 1])),
        'Private Non-profit'        : np.sqrt(loss_fn(y[X['COW'] == 2], preds[X['COW'] == 2])),
        'Local Gov'                 : np.sqrt(loss_fn(y[X['COW'] == 3], preds[X['COW'] == 3])),
        'State Gov'                 : np.sqrt(loss_fn(y[X['COW'] == 4], preds[X['COW'] == 4])),
        'Federal'                   : np.sqrt(loss_fn(y[X['COW'] == 5], preds[X['COW'] == 5])),
        'Self Employed (non inc.)'  : np.sqrt(loss_fn(y[X['COW'] == 6], preds[X['COW'] == 6])),
        'Self Employed (inc.)'      : np.sqrt(loss_fn(y[X['COW'] == 7], preds[X['COW'] == 7])),
        'Without pay'               : np.sqrt(loss_fn(y[X['COW'] == 8], preds[X['COW'] == 8])),
    }
    error_array = []
    for i in aggregate_statistics.keys():
        error_array.append(aggregate_statistics[i])
    statistics_df = pd.DataFrame([list(aggregate_statistics.keys())])
    statistics_df.loc[1] = error_array
    statistics_df = statistics_df.T
    statistics_df.columns = ['Category', 'RMSE']
    statistics_df = statistics_df.reset_index(drop=True)
    with open(f'{SERVER_PATH}/statistics.md', 'w') as file:
        file.write("---\n layout: statistics \n title: Model Statistics\n description: Global PDL RMSE score on various groups (1/2 validation data)\n---\n")
        file.write(statistics_df.to_markdown())
    return statistics_df

def update_leaderboard(team_name, model_error, flag, global_flag, global_change):

    leaderboard_df = pd.read_csv("leaderboard.csv", index_col = "Unnamed: 0")
    
    if "Error" in leaderboard_df.columns:
        type_error = "Error"
    else:
        type_error = "Accuracy"
    if len(leaderboard_df.loc[leaderboard_df['Team Name'] == team_name]) == 1:
        leaderboard_df.loc[leaderboard_df['Team Name'] == team_name, type_error] = model_error
        leaderboard_df.loc[leaderboard_df['Team Name'] == team_name, 'Private Updates'] += flag
        leaderboard_df.loc[leaderboard_df['Team Name'] == team_name,'Global Updates'] += global_flag
        leaderboard_df.loc[leaderboard_df['Team Name'] == team_name,'Global Reduction'] += global_change
    else:
        leaderboard_df.loc[len(leaderboard_df.index) + 1] = [team_name, model_error, 1, int(global_flag), global_change]

    if type_error == "Error":
        leaderboard_df = leaderboard_df.sort_values(by = ["Error", "Global Updates"], ascending = [True, False]).reset_index(drop=True)   
    else:
        leaderboard_df = leaderboard_df.sort_values(by = ["Accuracy", "Global Updates"], ascending = [False, False]).reset_index(drop=True)   

    leaderboard_df.index += 1

    leaderboard_df.to_csv("leaderboard.csv")

    with open(f'{SERVER_PATH}/leaderboard.md', 'w') as file:
        file.write("---\n layout: leaderboard \n title: Leaderboard\n description: Leaderboard by RMSE of teams participating. \n---\n")
        markdown_str = leaderboard_df.to_markdown().replace("-|", ":|")
        markdown_str = markdown_str.replace("|-", "|:")
        file.write(markdown_str)

def clean_up_branch(branch_name):
    os.system(f"git push origin -d {branch_name}")
    return

def push_updates_to_git(team_name, team_flag, global_flag):
    os.chdir(f"{SERVER_PATH}")
    os.system(f"git add .")
    if team_flag == True:
        updated = 'team'
        if global_flag == True:
            updated += "+"
    if global_flag == True:
        updated += 'global'
    os.system(f'git commit -m "update from {team_name} on {updated} model" ')
    os.system(f"git checkout main")
    os.system(f"git push")
