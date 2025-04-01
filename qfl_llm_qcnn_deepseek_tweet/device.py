import numpy as np
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss
from qiskit_machine_learning.optimizers import COBYLA
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from bitsandbytes import BitsAndBytesConfig
import evaluate
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from IPython.display import clear_output
from config import NUM_QUBITS, NUM_CLASSES, MAXITER, LOG_FOLDER, MODEL_TO_USE, PEFT_METHOD, USE_LLM

from circuit import conv_layer, pool_layer
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp

# Device class for QCNN and LLM fine-tuning
class Device:
    def __init__(self, device_id, maxiter, num_qubits, num_classes, log_folder, peft_method, use_llm=False, is_server=False, data_set=None, val_set=None, test_set=None):
        self.device_id = device_id
        self.log_folder = log_folder
        label_encoder = LabelEncoder()
        self.is_server = is_server
        # Data for QCNN
        if not is_server:
            self.data_set = data_set
            # for qcnn
            self.data_set_encoded = self.data_set.map(lambda x: {'label': label_encoder.fit_transform([x['label']])[0]}, batched=False)
            train_ind, eval_ind = train_test_split(list(range(len(self.data_set))), train_size=0.8, random_state=42)
            self.data_train = self.data_set.select(train_ind)
            self.data_test = self.data_set.select(eval_ind)

            # self.data_train, self.data_test = train_test_split(list(range(len(self.data_set_encoded))), train_size=0.8, random_state=42)
            # self.data_train, self.data_test = self.data_set_encoded.train_test_split(test_size=0.2, seed=42).values()

            # for llm
            # self.llm_train_dataset, self.llm_eval_dataset = train_test_split(list(range(len(self.data_set))), train_size=0.8, random_state=42)

            train_indices, eval_indices = train_test_split(list(range(len(self.data_set))), train_size=0.8, random_state=42)

            # Use select() to create train and eval datasets based on indices
            self.llm_train_dataset = self.data_set.select(train_indices)
            self.llm_eval_dataset = self.data_set.select(eval_indices)
        else:
            self.val_set = val_set
            self.val_set_encoded = val_set.map(lambda x: {'label': label_encoder.fit_transform([x['label']])[0]}, batched=False)
            self.test_set = test_set
            self.test_set_encoded = test_set.map(lambda x: {'label': label_encoder.fit_transform([x['label']])[0]}, batched=False)
            self.data_train = None
            self.data_test = None
            self.llm_train_dataset = self.val_set
            self.llm_eval_dataset = self.test_set
        self.num_qubits = num_qubits
        self.num_classes = num_classes
        self.qnn, self.weight_params = self.build_qcnn()
        self.initial_point = np.random.rand(len(self.qnn.weight_params))
        self.maxiter = maxiter
        self.optimizer = COBYLA(maxiter=self.maxiter)
        self.objective_func_vals = []
        self.classifier = None
        self.use_llm = use_llm
        self.peft_method = peft_method
        self.llm_acc = []
        self.llm_f1_score = []

    def callback_graph(self, weights, obj_func_eval):
        # clear_output(wait=True)
        self.objective_func_vals.append(obj_func_eval)
        # plt.title(f"Device {self.device_id} Objective Function Value vs. Iteration")
        # plt.xlabel("Iteration")
        # plt.ylabel("Objective Function Value")
        # plt.plot(range(len(self.objective_func_vals)), self.objective_func_vals)
        # plt.show()

    def fine_tune_llm(self):
        # Data preparation for LLM fine-tuning (Tokenized text)
        # LLM Step 2: Tokenize the Dataset
        # tokenizer = GPT2Tokenizer.from_pretrained(model_to_use)
        # tokenizer = DeepSeekTokenizer.from_pretrained(model_to_use)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_USE, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)


        # if not self.is_server:
        tokenized_train_set = self.llm_train_dataset.map(tokenize_function, batched=True)
        tokenized_eval_set = self.llm_eval_dataset.map(tokenize_function, batched=True)
        # tokenized_datasets = self.data_set.map(tokenize_function, batched=True)

        # LLM Step 3: Load the model
        # print(f"Initialize {model_to_use} MOdel for sequence classification")
        if self.peft_method == "lora":
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_TO_USE, num_labels=NUM_CLASSES) #labels in sentiment analysis: 3 (negative, neutral, positive)

        # model.config.pad_token_id = tokenizer.pad_token_id #add extra tokens to all input sequences to make them all same length

        elif self.peft_method == "qlora":
            # quant_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype=torch.bfloat16
            # )

            # Configure 4-bit quantization with CPU offloading
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_TO_USE, num_labels=NUM_CLASSES, quantization_config=quant_config, device_map="auto")

            # model = AutoModelForSequenceClassification.from_pretrained(model_to_use, num_labels=3, quantization_config=quant_config, device_map="auto")
            model.gradient_checkpointing_enable()

            # model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3) #labels in sentiment analysis: 3 (negative, neutral, positive)

        else:
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_TO_USE, num_labels=NUM_CLASSES)

        model.config.pad_token_id = model.config.eos_token_id

        if self.peft_method != "none":
            for param in model.parameters():
                param.requires_grad = False

            # Apply LoRA configuration
            lora_config = LoraConfig(
                r=8, lora_alpha=8, task_type=TaskType.SEQ_CLS, fan_in_fan_out=True
            )
            model = get_peft_model(model, lora_config)

        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            acc = accuracy.compute(predictions=predictions, references=labels)
            f1_score = f1.compute(predictions=predictions, references=labels, average='weighted')
            self.llm_acc.append(acc['accuracy'])
            self.llm_f1_score.append(f1_score["f1"])
            with open(f"{self.log_folder}/device_llm_eval_results.txt_metrics", "a") as f:
                f.write(f"Device {self.device_id} LLM - acc: {acc} - f1_score: {f1_score}\n")
            return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}



        # Define training arguments with updated parameters
        # https://huggingface.co/docs/transformers/v4.46.0/en/main_classes/trainer#transformers.TrainingArguments
        training_args = TrainingArguments(
            output_dir=f"{self.log_folder}/llm_folder/trainer/device_{self.device_id}_trainer", #The output directory where the model predictions and checkpoints will be written.
            evaluation_strategy="epoch",  #"epoch": Evaluation is done at the end of each epoch.
            save_strategy="epoch",
            # save_total_limit=1,
            num_train_epochs=2,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            # gradient_accumulation_steps=4,
            logging_steps=10,
            learning_rate=2e-5,
            load_best_model_at_end=True, # When this option is enabled, the best checkpoint will always be saved.
            metric_for_best_model="accuracy",
            report_to="none"
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_set,
            eval_dataset=tokenized_eval_set,
            compute_metrics=compute_metrics
        )
        # Train and evaluate
        trainer.train()
        # LLM Step 5: Evaluating the model
        eval_results = trainer.evaluate()

        # Save and loading model in case wanted
        # model.save_pretrained("./model1")
        # tokenizer.save_pretrained("./model1")
        # To load
        # loaded_tokenizer = GPT2Tokenizer.from_pretrained("./model1")
        # loaded_model = GPT2ForSequenceClassification.from_pretrained("./model1")

        self.llm_eval_results = eval_results
        print(f"Device {self.device_id} LLM Evaluation Results: {eval_results}")
        with open(f"{self.log_folder}/llm_folder/trainer/device_llm_eval_results.txt", "a") as f:
            f.write(f"Device {self.device_id} LLM Evaluation Results: {eval_results}\n")

    # Quantum CNN training
    def build_qcnn(self):
        feature_map = ZFeatureMap(self.num_qubits)
        ansatz = QuantumCircuit(self.num_qubits, name="Ansatz")
        ansatz.compose(conv_layer(self.num_qubits, "c1_{}".format(self.device_id)), list(range(self.num_qubits)),
                       inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p1_{}".format(self.device_id)), list(range(self.num_qubits)),
                       inplace=True)
        ansatz.compose(conv_layer(self.num_qubits, "c2_{}".format(self.device_id)), list(range(self.num_qubits)),
                       inplace=True)
        ansatz.compose(pool_layer([0], [1], "p2_{}".format(self.device_id)), list(range(2)), inplace=True)
        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(feature_map, range(self.num_qubits), inplace=True)
        circuit.compose(ansatz, range(self.num_qubits), inplace=True)

        observables = [SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)]) for _ in
                       range(self.num_classes)]
        qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observables,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters
        )
        return qnn, ansatz.parameters

    def train_qcnn(self):
        if not self.is_server:
          X_train = np.random.rand(len(self.data_train['text']), self.num_qubits)
          y_train = np.array(self.data_train['label'])
        else:
          X_train = np.random.rand(len(self.val_set_encoded['text']), self.num_qubits)
          y_train = np.array(self.val_set_encoded['label'])

        self.classifier = NeuralNetworkClassifier(
            self.qnn,
            optimizer=self.optimizer,
            loss=CrossEntropyLoss(),
            one_hot=True,
            initial_point=self.initial_point,
            callback=self.callback_graph,
            warm_start=True,
            # verbose=1,
        )
        self.classifier.fit(X_train, y_train)
        self.train_acc = self.classifier.score(X_train, y_train)
        with open(f"{self.log_folder}/device_train_acc.txt", "a") as f:
          f.write(f"Device {self.device_id}: QCNN Training Accuracy = {self.train_acc:.2f}\n")
        print(f"Device {self.device_id}: QCNN Training Accuracy = {self.train_acc:.2f}")
        return self.classifier

    def evaluate(self):
        if not self.is_server:
          X_test = np.random.rand(len(self.data_test['text']), self.num_qubits)
          y_test = np.array(self.data_test['label'])
        else:
          X_test = np.random.rand(len(self.test_set_encoded['text']), self.num_qubits)
          y_test = np.array(self.test_set_encoded['label'])
        # X_test = np.random.rand(len(self.qcnn_test_texts), self.num_qubits)
        # y_test = np.array(self.qcnn_test_labels)
        self.test_acc = self.classifier.score(X_test, y_test)
        print(f"Device {self.device_id}: QCNN Test Accuracy = {self.test_acc:.2f}")
        # with open(f"{log_folder}/device_{self.device_id}_test_acc.txt", "a") as f:
        #   f.write(f"Device {self.device_id}: QCNN Test Accuracy = {test_acc:.2f}\n")
        with open(f"{self.log_folder}/device_test_acc.txt", "a") as f:
          f.write(f"Device {self.device_id}: QCNN Test Accuracy = {self.test_acc:.2f}\n")
        return self.test_acc

    def set_weights(self, avg_model_weights):
        self.initial_point = avg_model_weights

    def federated_training(self, round_num):
        print("Communication round: ", round_num)
        if self.use_llm:
            if round_num == 1: # Only perform LLM fine-tuning on the first round
                print(f"Device {self.device_id} starting LLM fine-tuning...")
                self.fine_tune_llm()
            print(f"Device {self.device_id} starting QCNN training...")
            if round_num > 1: #after first communication round
                print(f"Self Object Value: {self.objective_func_vals[-1]} - LLM Eval Results: {self.llm_f1_score[-1]}")
                with open(f"{self.log_folder}/device_objective_values_comparison_f1_score.txt", "a") as f:
                    f.write(f"Self Object Value: {self.objective_func_vals[-1]} - LLM Eval Results: {self.llm_f1_score[-1]}\n")
                if self.objective_func_vals[-1] > self.llm_f1_score[-1]:
                    if self.llm_f1_score[-1] > 0:
                        ratio = self.objective_func_vals[-1] / self.llm_f1_score[-1]
                        self.maxiter = min(int(self.maxiter * ratio), 100)
                    else:
                        ratio = 1
                        self.maxiter = min(int(self.maxiter * ratio), 100)
                    self.optimizer = COBYLA(maxiter=self.maxiter)
            with open(f"{self.log_folder}/device_maxiter_values.txt", "a") as f:
                f.write(f"Device {self.device_id} | device_maxiter = {self.maxiter}\n")
        self.train_qcnn()
        self.evaluate()

        self.latest_objective_value = self.objective_func_vals[-1] if self.objective_func_vals else None
