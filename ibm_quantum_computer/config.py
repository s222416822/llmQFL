from datetime import datetime
import os

# Simulator options: "aer_sim_ibm_brisbane", "fake_manila", "clifford", "real_quantum"
SIMULATOR = "real_quantum"

# Training parameters
NUM_DEVICES = 3
TRAIN_SIZE = 1000
TEST_SIZE = 200
COMM_ROUNDS = 10

# Optimization adjustments
ADJUSTMENT_METHODS = [
    "incremental",
    "dynamic_weighted_adjustment",
    "logarithmic",
    "adaptive",
    "hybrid"
]
MAX_INCREMENT = 5
MAX_MAXITER = 5
ADJUSTMENT_CHOICE = ADJUSTMENT_METHODS[3]  # "adaptive"
SELECTION_METHOD = "selected"  # or "all"
LLM_USE = True

# Logging setup
CURRENT_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FOLDER = f"logs/{CURRENT_TIME}_devicesimulator={SIMULATOR}_withLLM={LLM_USE}_{ADJUSTMENT_CHOICE}_genomic{TRAIN_SIZE}_average_{SELECTION_METHOD}_numDevices={NUM_DEVICES}"
os.makedirs(LOG_FOLDER, exist_ok=True)

# Qiskit Runtime Service
TOKEN = "YOUR IBM TOKEN HERE"