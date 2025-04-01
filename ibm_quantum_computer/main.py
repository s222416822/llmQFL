import numpy as np
import time
from device import Device
from data_processing import load_and_process_data
from config import (NUM_DEVICES, COMM_ROUNDS, LOG_FOLDER, SELECTION_METHOD,
                    LLM_USE)


def main():
    (devices_data, devices_labels, train_data_splits, test_data_splits,
     server_test_features, server_test_labels, dds, model_cls, tokenizer) = load_and_process_data()

    devices_list = [
        Device(i, devices_data[i], devices_labels[i], train_data_splits[i],
               test_data_splits[i], model_cls, LOG_FOLDER)
        for i in range(NUM_DEVICES)
    ]

    device_train_data_server = dds['train'].select(range(len(dds['train'])))
    split_server = device_train_data_server.train_test_split(test_size=0.2, seed=i)

    # Create the server device
    server_device = Device(
        NUM_DEVICES, server_test_features, server_test_labels,
        split_server['train'], split_server['test'], model_cls, LOG_FOLDER
    )

    average_weights = None
    selected_average_weights = None
    for n in range(COMM_ROUNDS):
        start_time = time.time()
        devices_weights_list = []

        if n == 0 and LLM_USE:
            server_device.fine_tune()

        for device in devices_list:
            device.current_comm_round = n
            print(f"\nTraining VQC on Device {device.idx}")
            if n == 0 and LLM_USE:
                device.fine_tune()
            if n > 0:
                device.vqc.initial_point = average_weights if SELECTION_METHOD == "all" else selected_average_weights
            device.training()
            with open(f"{LOG_FOLDER}/device.txt", 'a') as file:
                file.write(
                    f"Comm_round: {n} - Device: {device.idx}  - train_acc: {device.train_score_q4:.2f} - test_acc: {device.test_score_q4:.2f} - loss: {device.objective_func_vals[-1]}\n")
            with open(f"{LOG_FOLDER}/training_time_device.txt", 'a') as file:
                file.write(f"Comm_round: {n} - Device: {device.idx} - training_time: {device.training_time}\n")
            devices_weights_list.append(device.vqc.weights)

        average_weights = np.mean(devices_weights_list, axis=0)
        server_device.vqc.initial_point = average_weights
        server_device.training()

        distances = []
        percentage = 10
        server_latest_val = server_device.objective_func_vals[-1]
        for device in devices_list:
            device_latest_val = device.objective_func_vals[-1]
            distance = abs(device_latest_val - server_latest_val)
            distances.append((device, distance))

        distances.sort(key=lambda x: x[1])
        num_to_select = max(1, int(len(distances) * (percentage / 100.0)))
        closest_devices = [d[0] for d in distances[:num_to_select]]
        selected_average_weights = np.mean([d.vqc.weights for d in closest_devices], axis=0)

        improvement_threshold = 0.02
        current_average_distance = sum(d[1] for d in distances) / len(distances)
        with open(f"{LOG_FOLDER}/server.txt", 'a') as file:
            file.write(
                f"Comm_round: {n} - Device: {server_device.idx}  - train_acc: {server_device.train_score_q4:.2f} - test_acc: {server_device.test_score_q4:.2f}\n")
        server_device.evaluate(average_weights)
        with open(f"{LOG_FOLDER}/server_test.txt", 'a') as file:
            file.write(f"Comm_round: {n} - Device: {server_device.idx} - test_acc: {server_device.test_score_q4_1}\n")

        comm_time = time.time() - start_time
        print(f"Communication time for round {n}: {comm_time:.4f} seconds")
        with open(f"{LOG_FOLDER}/comm_time.txt", "a") as f:
            f.write(f"Communication time for round {n}: {comm_time:.4f} seconds\n")

    with open(f"{LOG_FOLDER}/objective_values_devices.txt", 'a') as file:
        for device in devices_list:
            file.write(f"Device {device.idx}: {device.objective_func_vals}\n")
    with open(f"{LOG_FOLDER}/server_objective_values_devices.txt", 'a') as file:
        file.write(f"Device {server_device.idx}: {server_device.objective_func_vals}\n")


if __name__ == "__main__":
    main()