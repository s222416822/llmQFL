import numpy as np
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import SamplerV2 as Sampler, QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from IPython.display import clear_output
import time
import math
import torch
from transformers import TrainingArguments, Trainer, TrainerCallback
import evaluate
from config import SIMULATOR, LOG_FOLDER, MAX_INCREMENT, MAX_MAXITER, ADJUSTMENT_CHOICE, TOKEN, LLM_USE


# Define Device class
class Device:
    def __init__(self, idx, data, labels, train_dataset, eval_dataset, model, log_folder, maxiter=1, warm_start=None):
        self.idx = idx
        self.features = MinMaxScaler().fit_transform(data)
        self.target = labels
        self.log_folder = log_folder
        self.maxiter = maxiter
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.model = model
        self.train_score_q4 = 0
        self.test_score_q4 = 0
        self.training_time = 0
        self.objective_func_vals = []
        self.llm_eval_loss = []
        self.llm_eval_results = None
        self.current_comm_round = 0

        service = QiskitRuntimeService(channel="ibm_quantum", token=TOKEN)

        if SIMULATOR == "aer_sim_ibm_brisbane":
            # aer_sim = AerSimulator()
            # Specify a QPU to use for the noise model
            real_backend = service.backend("ibm_brisbane")
            noise_model = NoiseModel.from_backend(real_backend)
            self.backend = AerSimulator(noise_model=noise_model)
            self.sampler = Sampler(mode=self.backend)
            self.sampler.options.default_shots = 10
        elif SIMULATOR == "fake_manila":
            self.backend = FakeManilaV2()
            # self.backend = GenericBackendV2(num_qubits=127)
            # self.backend = Fake127QPulseV1()
            # fake_manila.set_options(shots=10)
            self.sampler = Sampler(mode=self.backend)
            self.sampler.options.default_shots = 10

        # elif simulator == "clifford":
        #   self.aer_sim = AerSimulator(method="stabilizer")
        #   self.sampler = Sampler(mode=self.aer_sim)
        elif SIMULATOR == "real_quantum":
            # backend = service.backend('ibm_brisbane', simulator=False)
            # backend = service.least_busy(operational=True, simulator=False)
            # self.backend = service.backend("ibm_sherbrooke")
            # self.backend = service.backend("ibm_kyiv", simulator=False)
            self.backend = service.least_busy(operational=True, simulator=False)
            self.sampler = Sampler(mode=self.backend)
            self.sampler.options.default_shots = 10

        self.optimizer = COBYLA(maxiter=self.maxiter)
        self.num_features = self.features.shape[1]
        self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(
            self.features, self.target, train_size=0.8, random_state=42
        )
        self.feature_map = ZZFeatureMap(feature_dimension=self.num_features, reps=1)
        self.ansatz = RealAmplitudes(num_qubits=self.num_features, reps=3)
        self.initial_point = np.asarray([0.5] * self.ansatz.num_parameters)

        # self.vqc = VQC(
        #     feature_map=self.feature_map,
        #     ansatz=self.ansatz,
        #     optimizer=self.optimizer,
        #     callback=self.callback_graph,
        #     initial_point=self.initial_point,
        #     warm_start=True
        # )

        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
        self.isa_qc_ansatz = pm.run(self.ansatz)
        self.isa_qc_feature_map = pm.run(self.feature_map)

        print(f"Ansatz qubits before pm.run: {self.ansatz.num_qubits}")
        self.isa_qc_ansatz = pm.run(self.ansatz)
        print(f"Ansatz qubits after pm.run: {self.isa_qc_ansatz.num_qubits}")

        # # Print qubit count for feature map before and after pm.run
        print(f"Feature map qubits before pm.run: {self.feature_map.num_qubits}")
        self.isa_qc_feature_map = pm.run(self.feature_map)
        print(f"Feature map qubits after pm.run: {self.isa_qc_feature_map.num_qubits}")

        self.vqc = VQC(
            sampler=self.sampler,
            # feature_map=self.feature_map,
            feature_map=self.isa_qc_feature_map,
            ansatz=self.isa_qc_ansatz,
            # ansatz=self.ansatz,
            optimizer=self.optimizer,
            callback=self.callback_graph,
            # initial_point=initial_point,
            warm_start=True,
            pass_manager=pm
        )

    def callback_graph(self, weights, obj_func_eval):
        # print(f"Object Func Value: {obj_func_eval}")
        # clear_output(wait=True)
        self.objective_func_vals.append(obj_func_eval)
        # plt.plot(range(len(self.objective_func_vals)), self.objective_func_vals)
        # plt.xlabel("Iteration")
        # plt.ylabel("Loss")
        # plt.title(f"Device {self.idx}")
        # plt.show()

    def training(self):
        start = time.time()
        # adjustment_choice = adjustment[0]
        if self.current_comm_round > 1 and LLM_USE:
            print(f"OBJECTIVE FUN LOSS: {self.objective_func_vals[-1]} - LLM LOSS: {self.llm_eval_loss[-1]}")
            with open(f"{self.log_folder}/objective_values_devices_comparison.txt", 'a') as file:
                file.write(
                    f"Comm Round: {self.current_comm_round} - Device: {self.idx} - OBJECTIVE FUN LOSS: {self.objective_func_vals[-1]} - LLM LOSS: {self.llm_eval_loss[-1]}\n")

            if self.objective_func_vals[-1] > self.llm_eval_loss[-1]:
                ratio = self.objective_func_vals[-1] / self.llm_eval_loss[-1]
                if ADJUSTMENT_CHOICE == "incremental":
                    increment = int(self.maxiter * (ratio - 1))
                    self.maxiter = min(self.maxiter + increment, MAX_MAXITER)
                elif ADJUSTMENT_CHOICE == "dynamic_weighted_adjustment":
                    self.maxiter = min(int(self.maxiter * 0.5 + self.maxiter * 0.5 * ratio), MAX_MAXITER)
                elif ADJUSTMENT_CHOICE == "logarithmic":
                    self.maxiter = min(self.maxiter + int(math.log(1 + ratio) * self.maxiter), MAX_MAXITER)
                elif ADJUSTMENT_CHOICE == "adaptive":
                    self.maxiter = min(int(self.maxiter * ratio), MAX_MAXITER)
                elif ADJUSTMENT_CHOICE == "hybrid":
                    increment = min(int(self.maxiter * ratio), MAX_INCREMENT)
                    self.maxiter = min(self.maxiter + increment, MAX_MAXITER)

                self.optimizer = COBYLA(maxiter=self.maxiter)
                print(
                    f"Comm Round: {self.current_comm_round} - Device {self.idx} - ratio: {ratio} - maxiter: {self.maxiter}")
                with open(f"{self.log_folder}/optimizer_maxiter_values.txt", "a") as file:
                    file.write(
                        f"Comm Round: {self.current_comm_round} - Device {self.idx} - ratio: {ratio} - maxiter: {self.maxiter}\n")

        self.vqc.optimizer = self.optimizer
        self.vqc.fit(self.train_features, self.train_labels)
        self.training_time = time.time() - start
        self.train_score_q4 = self.vqc.score(self.train_features, self.train_labels)
        self.test_score_q4 = self.vqc.score(self.test_features, self.test_labels)
        print(
            f"Device {self.idx} - VQC Training Score: {self.train_score_q4:.2f}, Test Score: {self.test_score_q4:.2f}")

    def evaluate(self, weights):
        self.vqc.initial_point = weights
        self.test_score_q4_1 = self.vqc.score(self.test_features, self.test_labels)

    def fine_tune(self):
        bs = 8
        epochs = 4
        lr = 8e-5

        training_args = TrainingArguments(
            output_dir=f"outputs_device_{self.idx}",
            logging_dir=f'./logs_device_{self.idx}',
            logging_steps=10,
            learning_rate=lr,
            warmup_ratio=0.1,
            lr_scheduler_type='cosine',
            fp16=True,
            evaluation_strategy="steps",
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            num_train_epochs=epochs,
            weight_decay=0.01,
            report_to='none'
        )

        def compute_metrics(eval_preds):
            metric = evaluate.load("glue", "mrpc")
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        # Callback to save training loss
        class SaveTrainingLossCallback(TrainerCallback):
            def __init__(self, device):
                super().__init__()
                self.device = device

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:  # Log only if `loss` exists
                    step = state.global_step
                    train_loss = logs["loss"]
                    # Append training loss to the device's llm_eval_loss array
                    self.device.llm_eval_loss.append(train_loss)
                    with open(f"{self.device.log_folder}/llm_training_loss_log.txt", "a") as f:
                        f.write(f"Step {step} | Train Loss: {train_loss}\n")
                    print(f"Step {step} | Train Loss: {train_loss}")

        # Initialize the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=None,  # Will be set in main.py
            compute_metrics=compute_metrics,
            callbacks=[SaveTrainingLossCallback(self)] # Use the pre-initialized callback
        )

        # Train the model
        trainer.train()
        # Evaluate the model
        eval_results = trainer.evaluate()
        print(f"Validation Results: {eval_results}")
        # Save evaluation results to a file
        with open(f"{self.log_folder}/device_llm_eval_results.txt", "a") as f:
            f.write(f"Device {self.idx} - LLM Evaluation Results: {eval_results}\n")
        print(f"Device {self.idx} Training Losses: {self.llm_eval_loss}")