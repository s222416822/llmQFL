from datetime import datetime
import os

NUM_DEVICES = 10
COMM_ROUNDS = 10
TRAIN_SIZE = 1000
TEST_SIZE = 200
MAX_INCREMENT = 100
MAX_MAXITER = 100

ADJUSTMENT = [
    "incremental",
    "dynamic_weighted_adjustment",
    "logarithmic",
    "adaptive",
    "hybrid"
]
ADJUSTMENT_CHOICE = ADJUSTMENT[3]  # "adaptive"

CURRENT_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FOLDER = f"logs/{CURRENT_TIME}_withLLM_{ADJUSTMENT_CHOICE}_average_all_numDevices={NUM_DEVICES}"
os.makedirs(LOG_FOLDER, exist_ok=True)