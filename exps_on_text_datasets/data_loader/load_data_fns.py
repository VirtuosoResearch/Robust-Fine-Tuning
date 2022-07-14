from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoConfig,
    DataCollatorWithPadding,
    PretrainedConfig,
    default_data_collator,
)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def load_glue_tasks(task_name, logger, 
    model_name_or_path, pad_to_max_length, max_length, train_batch_size, eval_batch_size, 
    noise_label=0.0, keep_index = False):

    # Specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    if task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", task_name)
    # Labels
    if task_name is not None:
        is_regression = task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1

    config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels, finetuning_task=task_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    # Preprocessing the datasets
    if task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[task_name]

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    padding = "max_length" if pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    remove_columns = raw_datasets["train"].column_names
    if keep_index:
        remove_columns.remove('idx')
    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=remove_columns
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if task_name == "mnli" else "validation"]
    test_dataset = processed_datasets["test"] if 'test' in raw_datasets.keys() else None
    if keep_index:
        eval_dataset = eval_dataset.remove_columns(['idx'])
        if test_dataset is not None:
            test_dataset = test_dataset.remove_columns(['idx'])

    if noise_label != 0.0:
        train_dataset = label_noise(train_dataset, noise_label)

    # # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if pad_to_max_length:
        # If padding was already done at max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=eval_batch_size) if 'test' in raw_datasets.keys() else None

    return train_dataloader, eval_dataloader, test_dataloader, config

def label_noise(dataset, noise_rate):
    assert 0 <= noise_rate <= 1
    # Fix seed to flip the labels
    np.random.seed(1024)

    # setup
    num_classes = np.max(dataset['labels'])+1
    train_labels = np.asarray(dataset['labels'])
    train_labels_old = np.copy(train_labels)

    def flip_label(examples):
        examples_labels = np.array(examples['labels'])
        n_examples = len(examples_labels)
        n_rand = int(noise_rate * n_examples)

        randomize_indices = np.random.choice(range(n_examples), size=n_rand, replace=False)
        if num_classes > 2:
            examples_labels[randomize_indices] = np.random.choice(range(num_classes), size=n_rand, replace=True)
        else:
            examples_labels[randomize_indices] = 1-examples_labels[randomize_indices]
        
        examples['labels'] = examples_labels
        return examples

    dataset = dataset.map(flip_label, batched=True)
    print(np.sum(np.array(dataset['labels']) != train_labels_old))
    return dataset