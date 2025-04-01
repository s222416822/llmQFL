import numpy as np
import time
from device import Device
from data_processing import load_and_process_data
from config import NUM_DEVICES, COMM_ROUNDS, LOG_FOLDER


def main():
    (devices_data, devices_labels, train_data_splits, test_data_splits,
     server_test_features, server_test_labels, dds, model_cls, tokenizer) = load_and_process_data()

    # Instantiate and train devices
    devices_list = [Device(
        idx=i,
        data=devices_data[i],
        labels=devices_labels[i],
        train_dataset=train_data_splits[i],
        eval_dataset=test_data_splits[i],
        model=model_cls,
        log_folder=LOG_FOLDER
    ) for i in range(NUM_DEVICES)]

    device_train_data_server = dds['train'].select(range(len(dds['train'])))
    split_server = device_train_data_server.train_test_split(test_size=0.2, seed=NUM_DEVICES)

    # Create the server device
    server_device = Device(
        idx=NUM_DEVICES,
        data=server_test_features,
        labels=server_test_labels,
        train_dataset=split_server['train'],
        eval_dataset=split_server['test'],
        model=model_cls,
        log_folder=LOG_FOLDER
    )

    average_weights = None
    selected_average_weights = None
    for n in range(COMM_ROUNDS):
        start_time = time.time()
        devices_weights_list = []

        if n == 0:
            server_device.fine_tune()
        for device in devices_list:
            device.current_comm_round = n
            print(f"\nTraining VQC on Device {device.idx}")
            print(f"\nFine-tuning LLaMA model on Device {device.idx}")
            if n == 0:
                device.fine_tune()
            if n > 0:
                device.vqc.initial_point = average_weights
            device.training()
            print(
                f"Comm_round: {n} - Device: {device.idx}  - train_acc: {device.train_score_q4:.2f} - test_acc: {device.test_score_q4:.2f} - loss: {device.objective_func_vals[-1]}")
            with open(f"{LOG_FOLDER}/device.txt", 'a') as file:
                file.write(
                    f"Comm_round: {n} - Device: {device.idx}  - train_acc: {device.train_score_q4:.2f} - test_acc: {device.test_score_q4:.2f} - loss: {device.objective_func_vals[-1]}\n")
            with open(f"{LOG_FOLDER}/training_time_device.txt", 'a') as file:
                file.write(f"Comm_round: {n} - Device: {device.idx} - training_time: {device.training_time}\n")
            devices_weights_list.append(device.vqc.weights)
            print(device.objective_func_vals)

        average_weights = np.mean(devices_weights_list, axis=0)
        server_device.vqc.initial_point = average_weights
        server_device.training()
        print(server_device.objective_func_vals)

        # Calculate distances between devices and the server
        distances = []
        percentage = 10
        server_latest_val = server_device.objective_func_vals[-1]
        for device in devices_list:
            device_latest_val = device.objective_func_vals[-1]
            distance = abs(device_latest_val - server_latest_val)
            distances.append((device, distance))

        # Sort devices by distance (ascending order)
        distances.sort(key=lambda x: x[1])

        # Select the top percentage
        num_to_select = max(1, int(len(distances) * (percentage / 100.0)))
        closest_devices = [device for device, _ in distances[:num_to_select]]
        print("Selected Devices", closest_devices)
        for device in closest_devices:
            print(device.idx)

        selected_average_weights = np.mean([device.vqc.weights for device in closest_devices], axis=0)

        previous_average_distance = float('inf')
        improvement_threshold = 0.4 # Threshold for improvement in distances
        current_average_distance = sum(d[1] for d in distances) / len(distances)
        if abs(previous_average_distance - current_average_distance) < improvement_threshold:
            print("Convergence reached: distances are not improving.")
            # break  # Terminate the loop
        llm_latest_val = server_device.llm_eval_loss[-1]
        if abs(server_device.objective_func_vals[-1] - llm_latest_val) < improvement_threshold:
            print("Server performance is close to LLM. Terminating.")
            # break  # Terminate the loop

        # Update the previous average distance
        previous_average_distance = current_average_distance
        with open(f"{LOG_FOLDER}/termination_data.txt", "a") as file:
            file.write(
                f"thresh: {improvement_threshold} - curr_avg_dist: {current_average_distance} - pre_avg_dist: {previous_average_distance} - loss_diff: {server_device.objective_func_vals[-1] - llm_latest_val} -avg_dis_diff: {previous_average_distance - current_average_distance}\n")

        print("Server Weights VQC Weights:", server_device.vqc.weights)
        with open(f"{LOG_FOLDER}/training_time_server.txt", 'a') as file:
            file.write(
                f"Comm_round: {n} - Device: {server_device.idx} - training_time: {server_device.training_time}\n")
        print(
            f"Comm_round: {n} - Server Device: {server_device.idx}  - train_acc: {server_device.train_score_q4:.2f} - test_acc: {server_device.test_score_q4:.2f}")
        with open(f"{LOG_FOLDER}/server.txt", 'a') as file:
            file.write(
                f"Comm_round: {n} - Device: {server_device.idx}  - train_acc: {server_device.train_score_q4:.2f} - test_acc: {server_device.test_score_q4:.2f}\n")
        server_device.evaluate(average_weights)
        print(f"Comm_round: {n} - Server Device Test: {server_device.idx} - test_acc: {server_device.test_score_q4_1}")
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