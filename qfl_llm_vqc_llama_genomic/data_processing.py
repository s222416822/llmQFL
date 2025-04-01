import pandas as pd
import numpy as np
from genomic_benchmarks.dataset_getters.pytorch_datasets import DemoHumanOrWorm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch
from config import NUM_DEVICES, TRAIN_SIZE, TEST_SIZE

def load_and_process_data():
    # Load and shuffle dataset
    train_dataset = list(DemoHumanOrWorm(split='train', version=0))[:TRAIN_SIZE]
    test_dataset = list(DemoHumanOrWorm(split='test', version=0))[:TEST_SIZE]
    np.random.shuffle(train_dataset)
    np.random.shuffle(test_dataset)

    # Convert to DataFrame
    data = {
        "dset": ["train"] * len(train_dataset) + ["test"] * len(test_dataset),
        "cat": [label for _, label in train_dataset] + [label for _, label in test_dataset],
        "seq": [seq for seq, _ in train_dataset] + [seq for seq, _ in test_dataset]
    }
    df = pd.DataFrame(data)

    # Create Dataset object
    ds = Dataset.from_pandas(df)

    # Tokenize sequences for LLaMA
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Ensure the tokenizer has a pad_token
    model_cls = AutoModelForSequenceClassification.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
        num_labels=2
    )
    model_cls.resize_token_embeddings(len(tokenizer))
    model_cls.config.pad_token_id = tokenizer.pad_token_id

    # Define k-mer functions for LLaMA tokenization
    def kmers_stride1(s, k=6):
        return [s[i:i + k] for i in range(0, len(s) - k + 1)]

    def tok_func(x):
        return tokenizer(" ".join(kmers_stride1(x["seq"])))

    # Apply tokenization to the dataset
    tok_ds = ds.map(tok_func, batched=False)
    tok_ds = tok_ds.rename_columns({'cat': 'labels'})
    dds = DatasetDict({
        'train': tok_ds.filter(lambda x: x["dset"] == "train"),
        'test': tok_ds.filter(lambda x: x["dset"] == "test")
    })

    # Prepare VQC dataset (One-hot encoding + PCA)
    nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded_sequences = []
    labels = []
    for sequence, label in zip(df['seq'], df['cat']):
        encoded_sequence = [[1 if i == nucleotide_map.get(n, -1) else 0 for i in range(4)] for n in sequence]
        encoded_sequences.append(encoded_sequence)
        labels.append(label)

    features = PCA(n_components=4).fit_transform(np.array(encoded_sequences).reshape(len(encoded_sequences), -1))
    alldevices_train_features, server_test_features, alldevices_train_labels, server_test_labels = train_test_split(
        features, np.array(labels), train_size=0.8, random_state=42
    )

    # Data distribution for VQC
    combined_data = list(zip(alldevices_train_features, alldevices_train_labels))
    shuffled_features, shuffled_labels = zip(*combined_data)
    samples_per_device = len(shuffled_features) // NUM_DEVICES
    remainder = len(shuffled_features) % NUM_DEVICES

    devices_data, devices_labels = [], []
    start_index = 0
    for i in range(NUM_DEVICES):
        extra_samples = 1 if i < remainder else 0
        end_index = start_index + samples_per_device + extra_samples
        devices_data.append(np.array(shuffled_features[start_index:end_index]))
        devices_labels.append(np.array(shuffled_labels[start_index:end_index]))
        start_index = end_index

    # Split data among devices for LLaMA fine-tuning
    train_data_splits = []
    test_data_splits = []
    start_index = 0
    for i in range(NUM_DEVICES):
        extra_samples = 1 if i < remainder else 0
        end_index = start_index + samples_per_device + extra_samples
        indices = list(range(start_index, min(end_index, len(dds['train']))))
        device_train_data = dds['train'].select(indices)
        split = device_train_data.train_test_split(test_size=0.2, seed=i)
        train_data_splits.append(split['train'])
        test_data_splits.append(split['test'])
        start_index = end_index

    # Configure LoRA
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    model_cls = get_peft_model(model_cls, lora_config)

    return (devices_data, devices_labels, train_data_splits, test_data_splits,
            server_test_features, server_test_labels, dds, model_cls, tokenizer)