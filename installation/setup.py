import os
import dill as pkl
import pandas as pd
import numpy as np
import sklearn as sk
import pdl
import shutil
import hashlib
from dotenv import load_dotenv
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import top_k_accuracy_score as TOP_K

def write_to_file(filename, contents):
    with open(filename, "w") as file:
        file.write(contents)

def append_to_file(filename, contents):
    with open(filename, "a") as file:
        file.write(f"\n{contents}")

def get_repo_name():
    return os.getcwd().split("/")[-1]

def create_env():
    SECRET = hashlib.md5(str(datetime.now()).encode()).hexdigest()[0:8]
    SERVER_PATH = f"/home/{SECRET}/repo/"
    load_dotenv()
    REPO_NAME = os.getenv("REPO_NAME")
    RUNNER_PATH = SERVER_PATH + f"actions-runner/_work/{REPO_NAME}/{REPO_NAME}"
    append_to_file(".env",f"SECRET={SECRET}\nSERVER_PATH={SERVER_PATH}\nRUNNER_PATH={RUNNER_PATH}\n")
    return SECRET
    
def load_data():
    while True:
        filename = input("Input filename for data (requires .csv format)\n\n>>> ")
        try:
            data = pd.read_csv(filename)
        except Exception as err:
            print(f"ERROR: {err}. Please retry!")
            continue
        try:
            labels = data["Label"]
            data = data.drop(columns=["Label"])
            break
        except Exception as err:
            print(f"ERROR: {err}. Column \"Label\" does not appear to exist!")
            continue

    return split_save_data(data, labels), filename

def split_save_data(data, labels):

    x_train, x_intermediate, y_train, y_intermediate = sk.model_selection.train_test_split(data, labels, test_size = .3, random_state = 42)
    x_val, x_test, y_val, y_test = sk.model_selection.train_test_split(x_intermediate, y_intermediate, test_size = .5, random_state = 42)
    
    shutil.rmtree("data", ignore_errors=True)
    os.mkdir("data")

    np.save("data/training_data.npy", x_train)
    np.save("data/training_labels.npy", y_train)
    np.save("data/validation_data.npy", x_val)
    np.save("data/validation_labels.npy", y_val)
    np.save("data/test_data.npy", x_test)
    np.save("data/test_labels.npy", y_test)

    return x_train, y_train, x_val, y_val

def get_task_loss_fn():
    dataset_task_reference = {
        1 : "Binary Classification",
        2 : "Multiclass Classification",
        3 : "Regression"
    }
    while True:
        dataset_task = input("What is your dataset task? \n\
            1. Binary Classification\n\
            2. Multiclass Classification\n\
            3. Regression\n\n>>> ")
        print()
        try:
            dataset_task = int(dataset_task)
            if dataset_task > 0 and dataset_task <= len(dataset_task_reference.keys()):
                break
            else:
                print("Out of bounds! Please choose a valid option from the menu!")
                continue
        except Exception as err:
            print(f"ERROR: {err}. Please try again!")
    
    dataset_task = dataset_task_reference[dataset_task]
    
    if dataset_task in ["Binary Classification", "Multiclass Classification"]:

        loss_fn_reference = {
            1 : "ACC",
            2 : "TOP_K"
        }

        while True:
            loss_fn = input("What is your desired loss function? \n\
                1. Accuracy (Recommended)\n\
                2. Top K Accuracy\n\n>>> ")
            print()
            try:
                loss_fn = int(loss_fn)
                if loss_fn > 0 and loss_fn <= len(loss_fn_reference.keys()):
                    break
                else:
                    print("Out of bounds! Please choose a valid option from the menu!")
                    continue
            except Exception as err:
                print(f"ERROR: {err}. Please try again!")
            
        loss_fn = loss_fn_reference[loss_fn]

    if dataset_task == "Regression":
        loss_fn_reference = {
            1 : "MSE",
            2 : "MAE"
        }
        while True:
            loss_fn = input("What is your desired loss function? \n\
                1. Mean Squared Error (Recommended)\n\
                2. Mean Absolute Error\n\n>>> ")
            print()
            try:
                loss_fn = int(loss_fn)
                if loss_fn > 0 and loss_fn <= len(loss_fn_reference.keys()):
                    break
                else:
                    print("Out of bounds! Please choose a valid option from the menu!")
                    continue
            except Exception as err:
                print(f"ERROR: {err}. Please try again!")

        loss_fn = loss_fn_reference[loss_fn]
    repo_name = get_repo_name()
    write_to_file(".env", f"DATASET_TASK={dataset_task}\nLOSS_FN={loss_fn}\nREPO_NAME={repo_name}")

    return dataset_task, loss_fn

def load_teams():
    while True:
        teams_csv_path = input("What is the filename for the teams (must be .csv with \"Username\" and \"Team\" columns)?\n\n>>> ")
        try:
            file = open(teams_csv_path, "rb")
        except Exception as err:
            print(f"ERROR: {err}. File could not be found!")
            continue
        
        try:
            teams_df = pd.read_csv(file)
            file.close()
        except Exception as err:
            print(f"ERROR: {err}. File could not loaded!")
            file.close()
            continue

        try:
            teams_df["Username"]
            teams_df["Team"]
            break
        except Exception as err:
            print(f"ERROR: {err}. Columns could not properly be loaded!")
            file.close()
            continue

    contents = ""
    for index in range(len(teams_df)):
        contents += teams_df["Username"].iloc[index].strip() + " : " + teams_df["Team"].iloc[index].strip() + "\n"

    write_to_file("username_to_team.yaml", contents)
    
    return teams_df, teams_csv_path

def load_base_model(dataset_task, x_train, y_train):

    while True:
        base_model_path = input("Input path to initial model (press Enter for default)\n\n>>> ")

        if base_model_path != "":
            
            try:
                file = open(base_model_path, "rb")
            except Exception as err:
                print(f"ERROR: {err}. File could not be found!")
                continue
        
            try:
                base_model = pkl.load(file)
                file.close()
                break
            except Exception as err:
                print(f"ERROR: {err}. Model Could not be loaded!")
                continue
        else:
            if dataset_task in ["Binary Classification", "Multiclass Classification"]:
                base_model = DecisionTreeClassifier(max_depth = 1, random_state = 42)

            if dataset_task == "Regression":
                base_model = DecisionTreeRegressor(max_depth = 1, random_state = 42)
            try:
                base_model.fit(x_train, y_train)
                break
            except:
                print("Error training initial model, please ensure your data is correctly formatted!")
                return None, -1
    return base_model, 1, base_model_path

def create_pdls(teams_df, base_model, x_train, y_train, x_val, y_val, loss_fn):

    while True:
        
        alpha = input("Input alpha parameter value (press Enter for default) \n\n>>> ")
        
        if alpha == "":
            break
        try:
            alpha = float(alpha)
            if alpha > 0:
                break
            else:
                1/0
        except:
            print("Alpha must be a positive value!")
    
    team_pdl = pdl.PointerDecisionList(base_model, x_train, y_train, x_val, y_val, alpha, min_group_size = 1, loss_fn_name = loss_fn)
    shutil.rmtree("teams", ignore_errors=True)
    os.mkdir("teams")

    for team in np.unique(teams_df["Team"]):    
        os.mkdir(f"teams/{team}")
        team_pdl.save_model(f"teams/{team}/PDL")
        np.save(f"teams/{team}/train_predictions.npy", team_pdl.train_predictions)
        np.save(f"teams/{team}/val_predictions.npy", team_pdl.val_predictions)
    os.mkdir(f"teams/global_pdl")
    team_pdl.save_model(f"teams/global_pdl/PDL")
    np.save(f"teams/global_pdl/train_predictions.npy", team_pdl.train_predictions)
    np.save(f"teams/global_pdl/val_predictions.npy", team_pdl.val_predictions)

    return team_pdl

def build_leaderboard(team_pdl, teams_df, y_val, loss_fn):
    if loss_fn == "MAE":
        loss_fn = MAE
    elif loss_fn == "MSE":
        loss_fn = MSE
    elif loss_fn == "ACC":
        loss_fn = ACC
    elif loss_fn == "TOP_K":
        loss_fn = TOP_K

    base_error = loss_fn(y_val, team_pdl.val_predictions)
    num_teams = len(np.unique(teams_df["Team"]))
    zero_filler = [0] * num_teams
    leaderboard_dict = {
        "Team Name" : np.unique(teams_df["Team"]),
        "Error" : [base_error] * num_teams,
        "Private Updates" : zero_filler,
        "Global Updates" : zero_filler,
        "Global Reduction" : zero_filler,
    }

    leaderboard = pd.DataFrame(leaderboard_dict)
    leaderboard.index += 1
    leaderboard.to_csv("leaderboard.csv")

    with open(f'leaderboard.md', 'w') as file:
        file.write("---\n layout: default \n title: Leaderboard\n description: Leaderboard by error of teams participating. \n---\n")
        markdown_str = leaderboard.to_markdown().replace("-|", ":|")
        markdown_str = markdown_str.replace("|-", "|:")
        file.write(markdown_str)

def create_tmp():
    shutil.rmtree("container_tmp", ignore_errors=True)
    os.mkdir("container_tmp")
    shutil.rmtree("tmp", ignore_errors=True)
    os.mkdir("tmp")

def clean_up(data_path, teams_csv_path, base_model_path):
    append_to_file(".gitignore", data_path)
    append_to_file(".gitignore", teams_csv_path)
    if base_model_path:
        append_to_file(".gitignore", base_model_path)
    create_tmp()
    if os.path.exists("initial_scripts"):
        os.rename("initial_scripts", "scripts")
    elif not os.path.exists("scripts"):
        print("Missing scripts file, recheck download!")
        1/0

    for filename in ["example_data.csv", "example_data.csv"]:
        if os.path.exists(filename):
            os.remove(filename)

    if not os.path.exists(".hidden"):
        os.mkdir(".hidden")
        
    for filename in ["bad_argvals.txt", "security.py"]:
        if os.path.exists(filename):
            shutil.move(filename, f".hidden/{filename}")

def main():
    if os.path.isfile(".env"):
        flag = input("Installation is already completed, are you sure you want to reinstall? \n\n (Y/N) >>> ")
        if flag.strip().lower() not in ["y", "yes"]:
            print("\nGoodbye!\n")
            exit()
    
    print("\n--- (Step 0/n) Checking Directory Contents ---\n")
 
    (x_train, y_train, x_val, y_val), data_path = load_data()

    print("\n--- (Step 1/n) Creating Environment File ---\n")
   
    create_env()

    print("--- (Step 2/n) Defining Dataset Task + Loss Function ---\n")

    dataset_task, loss_fn = get_task_loss_fn()

    print("\n--- (Step 3/n) Initializing Teams ---\n")

    teams_df, teams_csv_path = load_teams()

    print("\n--- (Step 4/n) Initializing Base Model ---\n")
    
    base_model, flag, base_model_path = load_base_model(dataset_task, x_train, y_train)

    if flag == -1:
        1/0
    
    print("\n--- (Step 5/n) Initializing PDLs ---\n")

    team_pdl = create_pdls(teams_df, base_model, x_train, y_train, x_val, y_val, loss_fn)

    print("\n--- (Step 6/n) Create .gitignore ---\n")

    write_to_file(".gitignore", ".gitignore\n.ipynb_checkpoints\n.env\nusername_to_team.yaml\nleaderboard.csv\nsetup.ipynb\n__pycache__/\nteams/\nactions-runner/\nscripts/\ntmp/\ncontainer_tmp/\ndata/\nall_pairs/\nbias_bounty_venv/\n.hidden/\nrunner.sh\nDockerfile.sec\nDockerfile.repo\nsetup-images/\nbuild.sh\n")

    print("\n--- (Step 7/n) Building Leaderboard ---\n")

    build_leaderboard(team_pdl, teams_df, y_val, loss_fn)

    clean_up(data_path, teams_csv_path, base_model_path)


if __name__ == "__main__":
    try:
        main()
        print("Installation Successful!")
        # os.remove("setup.py")
    except Exception as err:
        if os.path.isfile(".env"):
            os.remove(".env")
        print(f"ERROR: {err}. Installation Unsuccessful, exiting!")
        exit(-1)
