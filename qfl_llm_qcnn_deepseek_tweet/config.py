import os
from datetime import datetime

NUM_DEVICES = 3
NUM_QUBITS = 4
NUM_CLASSES = 3
MAXITER = 10
ROUNDS = 10
TRAIN_SUBSET_SIZE = 1000
TEST_SUBSET_SIZE = 200
VALIDATION_SUBSET_SIZE = 200
MODEL_TO_USE = "deepseek-ai/deepseek-llm-7b-base" #Options: "gpt2"
USE_SUBSET = True
USE_LLM = True
PEFT_METHOD = "qlora"  # Options: "lora", "qlora", "none"

CURRENT_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATASET_USED = "tweet_eval - sentiment"
LOG_FOLDER = f"logs/{CURRENT_TIME}_{MODEL_TO_USE}_PEFT={PEFT_METHOD}_useLLM={USE_LLM}_useSubset={USE_SUBSET}_trainsize{TRAIN_SUBSET_SIZE}_dataset=tweet{TRAIN_SUBSET_SIZE}_devices{NUM_DEVICES}"
os.makedirs(LOG_FOLDER, exist_ok=True)