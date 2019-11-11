# Import
import datetime
import os
import subprocess

# Experiment variables
PYTHON_CMD = "python"
PROJECT_ROOT_DIR = ".\\webprofiler" # ".\\webprofiler" for Windows, "./webprofiler" for Ubuntu
LOG_ID = "20190611-142541"

# Training variables
#####################################
KST = datetime.timezone(datetime.timedelta(hours=9))
LOG_TIME_TAG = datetime.datetime.now().replace(tzinfo=KST).strftime("%Y%m%d_%H%M%S")
TRAIN_FILES = {"drnn": "train_drnn.py", "lstm": "train_lstm.py", "gru": "train_gru.py"}

TRAIN_MODEL = "gru" # "all", "drnn", "lstm", or "gru"
USE_GROUPING = True
USE_EMBEDDING = True
RANDOM_SEED_START = 42
RANDOM_SEED_END = 42
N_NEURONS = 150 # equal for all layers/cells
N_LAYERS = 2
LEARNING_RATE = 0.01
N_EPOCHS = 101
BATCH_SIZE = 100
ACT_FUNCTION = "relu" # sigmoid, tanh, relu, or elu
USE_PEEPHOLES = False
OPT_ALGO = "Momentum" # GD, Momentum, Nesterov, RMSProp, or Adam; AdaGrad is not suitable for deep learning
TOP_K_THRESH = 1

# Train
DIR_PATH = os.path.join(PROJECT_ROOT_DIR, "checkpoint")
if not os.path.isdir(DIR_PATH):
    os.mkdir(DIR_PATH)
DIR_PATH = os.path.join(PROJECT_ROOT_DIR, "result")
if not os.path.isdir(DIR_PATH):
    os.mkdir(DIR_PATH)

train_models = list()
if TRAIN_MODEL == "all":
    for key, _ in TRAIN_FILES.items():
        train_models.append(key)
elif TRAIN_MODEL == "drnn" or TRAIN_MODEL == "lstm" or TRAIN_MODEL == "gru":
    train_models.append(TRAIN_MODEL)
else: # default is a deep RNN
    train_models.append("drnn")

seed_list = list()    
for i in range(RANDOM_SEED_END - RANDOM_SEED_START + 1):
    seed_list.append(RANDOM_SEED_START + i)

print("\n==================== {0:s}: deep-learning ====================".format(LOG_TIME_TAG))
for seed in seed_list:
    print(" ***** RANDOM_SEED {0:d} ***** ".format(seed))
    for model in train_models:
        f = TRAIN_FILES[model]
        file_path = os.path.join(PROJECT_ROOT_DIR, f)
        cmd_str = [PYTHON_CMD, file_path]
        opt_str = list()
        opt_str.append("--train_model=" + model)
        opt_str.append("--log_time_tag=" + LOG_TIME_TAG)
        opt_str.append("--dir=" + PROJECT_ROOT_DIR)
        opt_str.append("--log_id=" + LOG_ID)
        opt_str.append("--random_seed=" + str(seed))
        opt_str.append("--use_embedding=" + str(USE_EMBEDDING))
        opt_str.append("--n_neurons=" + str(N_NEURONS))
        opt_str.append("--n_layers=" + str(N_LAYERS))
        opt_str.append("--learning_rate=" + str(LEARNING_RATE))
        opt_str.append("--n_epochs=" + str(N_EPOCHS))
        opt_str.append("--batch_size=" + str(BATCH_SIZE))
        opt_str.append("--act_func=" + ACT_FUNCTION)
        opt_str.append("--use_peepholes=" + str(USE_PEEPHOLES))
        opt_str.append("--opt_algo=" + OPT_ALGO)
        opt_str.append("--top_k_thresh=" + str(TOP_K_THRESH))
        cmd_str = cmd_str + opt_str
        subprocess.call(cmd_str, shell=False)
    print(" ***** RANDOM_SEED {0:d} ***** ".format(seed))
print("==================== {0:s}: deep-learning ====================".format(LOG_TIME_TAG))
#####################################