import numpy as np
import time
from device import Device
from datasets import load_dataset
from config import (NUM_DEVICES, NUM_QUBITS, NUM_CLASSES, MAXITER, ROUNDS,
                    TRAIN_SUBSET_SIZE, TEST_SUBSET_SIZE, VALIDATION_SUBSET_SIZE,
                    USE_SUBSET, LOG_FOLDER, PEFT_METHOD, USE_LLM)


class FederatedServer:
    def __init__(self, devices, server_device, num_qubits, log_folder):
        self.devices = devices
        # self.server_test_texts = server_test_texts
        # self.server_test_labels = server_test_labels
        self.num_qubits = num_qubits
        self.server_device = server_device
        self.log_folder = log_folder

    def average_models(self, models):
        avg_weights = np.mean(models, axis=0)
        return avg_weights

    def federated_round(self, round_num):
        qcnn_models = []
        testing_acc = []
        for device in self.devices:
            device.federated_training(round_num)
            qcnn_models.append(device.classifier.weights)
            test_acc = device.evaluate()
            testing_acc.append(test_acc)
            with open(f"{self.log_folder}/devices_objective_values.txt", "a") as f:
                f.write(f"Device {device.device_id} | device_objective_values = {device.objective_func_vals}\n")

        avg_qcnn_model_weights = self.average_models(qcnn_models)
        for device in self.devices:
            device.set_weights(avg_qcnn_model_weights)

        # Set the averaged model weights to the server device
        self.server_device.initial_point = avg_qcnn_model_weights
        self.server_device.train_qcnn()
        server_test_acc = self.server_device.evaluate()
        with open(f"{self.log_folder}/server_objective_values.txt", "a") as f:
            f.write(f"Comm Round: {round_num} | server_objective_values = {self.server_device.objective_func_vals}\n")

        avg_test_acc = np.mean(testing_acc)
        print(
            f"Round {round_num} - Avg Test Accuracy = {avg_test_acc:.2f}, Server Test Accuracy = {server_test_acc:.2f}")
        with open(f"{self.log_folder}/server_test_acc.txt", "a") as f:
            f.write(
                f"Round {round_num} | All Devices Avg Test Accuracy = {avg_test_acc:.2f} | Server Performance Test Accuracy = {server_test_acc:.2f} \n")

    def simulate(self, rounds):
        for r in range(1, rounds + 1):
            print(f"\n--- Federated Round {r} ---")
            # Start timing
            start_time = time.time()
            # Perform federated round
            self.federated_round(r)
            # End timing
            comm_time = time.time() - start_time
            # Log communication time
            print(f"Communication time for round {r}: {comm_time:.4f} seconds")
            with open(f"{self.log_folder}/comm_time.txt", "a") as f:
                f.write(f"Communication time for round {r}: {comm_time:.4f} seconds\n")


def main():
    print("Loading tweet_eval dataset for multi-class classification...")
    dataset = load_dataset("tweet_eval", "sentiment")

    if USE_SUBSET:
        dataset['train'] = dataset['train'].select(range(TRAIN_SUBSET_SIZE))
        dataset['test'] = dataset['test'].select(range(TEST_SUBSET_SIZE))
        dataset['validation'] = dataset['validation'].select(range(VALIDATION_SUBSET_SIZE))

    server_validation_dataset = dataset["validation"]
    server_test_dataset = dataset["test"]
    full_train_dataset = dataset["train"]

    device_datasets = [full_train_dataset.shard(num_shards=NUM_DEVICES, index=i) for i in range(NUM_DEVICES)]

    devices = [Device(i, MAXITER, NUM_QUBITS, NUM_CLASSES, LOG_FOLDER, PEFT_METHOD, USE_LLM, False, device_datasets[i])
               for i in range(NUM_DEVICES)]
    server_device = Device(NUM_DEVICES, MAXITER, NUM_QUBITS, NUM_CLASSES, LOG_FOLDER, PEFT_METHOD, USE_LLM, True,
                           None, server_validation_dataset, server_test_dataset)

    print("Simulating federated learning...")
    federated_server = FederatedServer(devices, server_device, NUM_QUBITS, LOG_FOLDER)
    federated_server.simulate(ROUNDS)


if __name__ == "__main__":
    main()