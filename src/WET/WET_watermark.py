import os
import math
import json
from matplotlib import pyplot as plt
import scienceplots
plt.style.use(['science', 'grid', 'pgf', 'ieee', 'no-latex'])
import wandb
import random
import argparse
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial
from scipy.stats import norm
import sklearn.metrics as metrics

import torch
from torch import nn
from torch.utils.data import DataLoader

import datasets
from datasets import load_dataset, DatasetDict, load_metric
from src.dataset.utils import load_mind

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from transformers import (
    AutoConfig,
    AutoTokenizer,
    SchedulerType,
    DataCollatorWithPadding,
    default_data_collator,
    get_scheduler,
)


from src.model.WET_gpt_cls import GPTClassifierConfig, GPTClassifier
from src.model.copier.WET_bert import BertForClassifyWithBackDoor
from src.model.copier.WET_another_bert import BertForClassifyWithBackDoor as AnotherBertForClassifyWithBackDoor
from src.WET.trigger.WET_base import InjectWatermark
from src.utils import merge_flatten_metrics, load_gpt_embeds

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )

    parser.add_argument(
        "--job_name", type=str, default=None, help="The job name used for wandb logging"
    )

    # GPT3 configuration
    parser.add_argument(
        "--gpt_emb_dim", type=int, default=1536, help="The embedding size of gpt3."
    )

    parser.add_argument(
        "--hyperdimensions", type=int, help="Number of hyperdimensions."
    )

    parser.add_argument(
        "--random_transformation_dims", type=int,
        help="Number of original dimensions to consider in the transformation.", default=None,
    )

    parser.add_argument(
        "--random_transformation_ablation", type=bool, default=False, help="Do random transformation ablation"
    )

    parser.add_argument(
        "--gpt_emb_train_file",
        type=str,
        default=None,
        help="The gpt3 embedding file of sst2 train set.",
    )
    parser.add_argument(
        "--gpt_emb_validation_file",
        type=str,
        default=None,
        help="The gpt3 embedding file of sst2 validation set.",
    )
    parser.add_argument(
        "--gpt_emb_test_file",
        type=str,
        default=None,
        help="The gpt3 embedding file of sst2 test set.",
    )

    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="The train file of mind train set.",
    )

    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="The validation file of mind train set.",
    )

    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="The test file of mind train set.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )

    parser.add_argument(
        "--disable_training", action="store_true", help="Disable pca evaluate."
    )

    # Model Copy
    parser.add_argument(
        "--verify_dataset_size",
        type=int,
        default=20,
        help="The number of samples of verify dataset.",
    )
    # TODO: is there any impact of this with new WM
    parser.add_argument(
        "--transform_hidden_size",
        type=int,
        default=1536,
        help="The dimension of transform hidden layer.",
    )
    parser.add_argument(
        "--transform_dropout_rate",
        type=float,
        default=0.0,
        help="The dropout rate of transformation layer.",
    )
    parser.add_argument(
        "--copy_learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--copy_num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--copy_max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--copy_gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--copy_num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )

    # GPT3 Classifier Config
    # TODO: is there any impact of this with new WM
    parser.add_argument(
        "--cls_hidden_dim",
        type=int,
        default=None,
        help="The hidden dimention of gpt3 classifier.",
    )
    parser.add_argument(
        "--cls_dropout_rate",
        type=float,
        default=None,
        help="The dropout rate of gpt3 classifier.",
    )
    parser.add_argument(
        "--cls_learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--cls_num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--cls_max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--cls_gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--cls_num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )

    parser.add_argument(
        "--data_name", type=str, default="sst2", help="dataset name for training."
    )

    parser.add_argument(
        "--project_name", type=str, default=None, help="project name for training."
    )

    args = parser.parse_args()

    return args


DATA_INFO = {
    "sst2": {
        "dataset_name": "glue",
        "dataset_config_name": "sst2",
        "text": "sentence",
        "idx": "idx",
        "remove": ["sentence", "idx"],
    },
    "enron": {
        "dataset_name": "SetFit/enron_spam",
        "dataset_config_name": None,
        "text": "subject",
        "idx": "message_id",
        "remove": [
            "message_id",
            "text",
            "label",
            "label_text",
            "subject",
            "message",
            "date",
        ],
    },
    "ag_news": {
        "dataset_name": "ag_news",
        "dataset_config_name": None,
        "text": "text",
        "idx": "md5",
        "remove": ["label", "text"],
    },
    "mind": {
        "dataset_name": "mind",
        "dataset_config_name": None,
        "text": "title",
        "idx": "docid",
        "remove": ["label", "title", "docid"],
    },
}

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def main():
    args = parse_args()
    random.seed(args.seed)

    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=args.output_dir)
        if args.with_tracking
        else Accelerator()
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info("Device: " + device)
    logger.info(f"Args: {args}")
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load raw dataset
    if args.data_name == "mind":
        raw_datasets = load_mind(
            train_tsv_path=args.train_file,
            test_tsv_path=args.test_file,
        )
    else:
        raw_datasets = load_dataset(
            DATA_INFO[args.data_name]["dataset_name"],
            DATA_INFO[args.data_name]["dataset_config_name"],
        )
    if args.data_name == "sst2":
        raw_datasets["test"] = raw_datasets["validation"]

    label_list = list(set(raw_datasets["train"]["label"]))
    num_labels = len(label_list)

    # Define gpt classifier config and model
    cls_config = GPTClassifierConfig(
        emb_dim=args.hyperdimensions,
        hidden_dim=args.cls_hidden_dim,
        dropout_rate=args.cls_dropout_rate,
        num_labels=num_labels,
    )
    cls_model = GPTClassifier(cls_config)

    # Define copy model tokenizer, config and model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.transform_hidden_size = args.transform_hidden_size
    config.emb_dim = args.hyperdimensions
    config.transform_dropout_rate = args.transform_dropout_rate

    other_config = AutoConfig.from_pretrained(args.model_name_or_path)
    other_config.transform_hidden_size = args.transform_hidden_size
    other_config.emb_dim = args.hyperdimensions
    other_config.transform_dropout_rate = args.transform_dropout_rate

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer
    )
    provider_tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-cased", use_fast=not args.use_slow_tokenizer
    )
    model = BertForClassifyWithBackDoor.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )

    other_model = AnotherBertForClassifyWithBackDoor.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=other_config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )

    # Preprocess Dataset
    emb_caches = load_gpt_embeds(
        args.gpt_emb_train_file,
        args.gpt_emb_validation_file,
        args.gpt_emb_test_file,
    )

    padding = "max_length" if args.pad_to_max_length else False

    def process_func(examples, idx, key):
        texts = examples[DATA_INFO[args.data_name]["text"]]

        result = tokenizer(
            texts, padding=padding, max_length=args.max_length, truncation=True
        )

        bert_base_result = provider_tokenizer(
            texts, padding=padding, max_length=args.max_length, truncation=True
        )

        result["provider_input_ids"] = bert_base_result["input_ids"]
        emb_data = emb_caches[key][idx]
        assert emb_data['text'] == texts
        result["clean_gpt_emb"] = emb_data['text_emb']
        result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = DatasetDict(
            {
                k: dataset.map(
                    partial(process_func, key=k),
                    with_indices=True,
                    remove_columns=DATA_INFO[args.data_name]["remove"],
                    desc="Run tokenization and add gpt3 embeddings on dataset",
                )
                for k, dataset in raw_datasets.items()
            }
        )

    inject_wm = InjectWatermark(
        args,
        args.seed,
        processed_datasets,
        tokenizer,
        provider_tokenizer,
        accelerator,
    )
    processed_datasets = inject_wm.another_process_datasets(
        processed_datasets
    )
    another_train_wm_pd = pd.DataFrame(processed_datasets['train']['another_gpt_emb'])
    another_wm_means = another_train_wm_pd.mean()
    another_wm_std = another_train_wm_pd.std()

    processed_datasets = inject_wm.process_datasets(
        processed_datasets
    )
    train_wm_pd = pd.DataFrame(processed_datasets['train']['gpt_emb'])
    wm_means = train_wm_pd.mean()
    wm_std = train_wm_pd.std()

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]

    verify_dataset = inject_wm.construct_verify_dataset(eval_dataset)

    # DataLoaders creation:
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    verify_dataloader = DataLoader(
        verify_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value

        init_kwargs = None
        if args.job_name is not None:
            init_kwargs = {"wandb": {"name": args.job_name}}

        if args.project_name is not None:
            project_name = args.project_name
        else:
            project_name = args.data_name + "_gpt_watermark"

        accelerator.init_trackers(
            project_name,
            experiment_config,
            init_kwargs=init_kwargs,
        )

    if not args.disable_training:
        completed_steps, total_loss = train_copier_original(
            args,
            model,
            train_dataset,
            train_dataloader,
            eval_dataloader,
            verify_dataloader,
            accelerator,
            args.copy_learning_rate,
            args.copy_gradient_accumulation_steps,
            args.copy_max_train_steps,
            args.copy_num_train_epochs,
            args.copy_num_warmup_steps,
        )
        logging.info(f"Copier Model Loss: {total_loss}")

        completed_steps, total_another_loss = train_copier_another(
            args,
            other_model,
            train_dataset,
            train_dataloader,
            eval_dataloader,
            verify_dataloader,
            accelerator,
            args.copy_learning_rate,
            args.copy_gradient_accumulation_steps,
            args.copy_max_train_steps,
            args.copy_num_train_epochs,
            args.copy_num_warmup_steps,
        )
        logging.info(f"Another Copier Model Loss: {total_another_loss}")

        copier_eval_metrics = eval_copier(
            args,
            model,
            other_model,
            total_loss,
            total_another_loss,
            completed_steps,
            train_dataloader,
            verify_dataloader,
            accelerator,
            inject_wm,
            another_wm_means,
            another_wm_std,
            wm_means,
            wm_std,
        )

        completed_steps, cls_eval_metrics = train_cls(
            args,
            cls_model,
            train_dataset,
            train_dataloader,
            eval_dataloader,
            accelerator,
            args.cls_learning_rate,
            args.cls_gradient_accumulation_steps,
            args.cls_max_train_steps,
            args.cls_num_train_epochs,
            args.cls_num_warmup_steps,
        )

        eval_metrics = merge_flatten_metrics(
            copier_eval_metrics, cls_eval_metrics, {}, parent_key="glue", sep="."
        )

        eval_metrics = merge_flatten_metrics(
            eval_metrics, inject_wm.mat_metrics, {}
        )

        if args.report_to == "wandb":
            for key, value in eval_metrics.items():
                wandb.run.summary[key] = value

        if args.with_tracking and args.report_to != "wandb":
            accelerator.end_training()

def train_cls(
    args,
    model,
    train_dataset,
    train_dataloader,
    eval_dataloader,
    accelerator,
    learning_rate,
    gradient_accumulation_steps,
    max_train_steps,
    num_train_epochs,
    num_warmup_steps,
    completed_steps=0,
):
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # count form init completed steps
    max_train_steps += completed_steps

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Get the metric function, accuracy and f1-score (macro)
    if wandb.run:
        acc_metric = load_metric('accuracy', experiment_id=wandb.run.id)
        f1_metric = load_metric('f1', experiment_id=wandb.run.id)
    else:
        acc_metric = load_metric('accuracy')
        f1_metric = load_metric('f1')

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * gradient_accumulation_steps
    )

    logger.info("***** Running classifier training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            outputs = model(**batch)

            loss = outputs.loss
            total_loss += loss.detach().float()
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if (
                step % gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= max_train_steps:
                break

        model.eval()
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            predictions = outputs.logits.argmax(dim=-1)

            predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[
                        : len(eval_dataloader.dataset) - samples_seen
                    ]
                    references = references[
                        : len(eval_dataloader.dataset) - samples_seen
                    ]
                else:
                    samples_seen += references.shape[0]
            acc_metric.add_batch(
                predictions=predictions,
                references=references,
            )
            f1_metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = acc_metric.compute() | f1_metric.compute(average='macro')
        logger.info(f"epoch {epoch}: {eval_metric}")

        if args.with_tracking:
            accelerator.log(
                {
                    "glue": eval_metric,
                    "cls_train_loss": total_loss.item() / len(train_dataloader),
                },
                step=completed_steps,
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}_cls"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        output_dir = os.path.join(args.output_dir, "cls")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )

    if args.output_dir is not None:
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        with open(os.path.join(args.output_dir, "cls_results.json"), "w") as f:
            json.dump(all_results, f)

    return completed_steps, eval_metric

def train_copier_original(
    args,
    model,
    train_dataset,
    train_dataloader,
    eval_dataloader,
    verify_dataloader,
    accelerator,
    learning_rate,
    gradient_accumulation_steps,
    max_train_steps,
    num_train_epochs,
    num_warmup_steps,
    completed_steps=0,
):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        verify_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        verify_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)


    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * gradient_accumulation_steps
    )

    logger.info("***** Running copier training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    starting_epoch = 0

    total_loss = 0
    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        total_loss = 0
        total_another_loss = 0

        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)

            loss = outputs.loss
            total_loss += loss.detach().float()
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)

            if (
                step % gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= max_train_steps:
                break

    return completed_steps, total_loss

def train_copier_another(
    args,
    another_model,
    train_dataset,
    train_dataloader,
    eval_dataloader,
    verify_dataloader,
    accelerator,
    learning_rate,
    gradient_accumulation_steps,
    max_train_steps,
    num_train_epochs,
    num_warmup_steps,
    completed_steps=0,
):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in another_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in another_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    (
        another_model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        verify_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        another_model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        verify_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * gradient_accumulation_steps
    )

    logger.info("***** Running another copier training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    total_loss = 0
    for epoch in range(starting_epoch, num_train_epochs):
        another_model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            outputs = another_model(**batch)

            loss = outputs.loss
            total_loss += loss.detach().float()
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if (
                step % gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= max_train_steps:
                break

    return completed_steps, total_loss


def eval_copier(
    args,
    model,
    another_model,
    total_loss,
    total_another_loss,
    completed_steps,
    train_dataloader,
    verify_dataloader,
    accelerator,
    inject_wm,
    another_wm_means,
    another_wm_std,
    wm_means,
    wm_std,
):
    model.eval()
    another_model.eval()

    results = {}

    overall_cos_dists = []
    overall_l2_dists = []

    cos_dists = []
    l2_dists = []

    # ablation
    abl_cos_dists = []
    abl_l2_dists = []

    copied_embs = []

    z_scores = []

    watermark_labels = []

    loss_fn = nn.MSELoss(reduction="none")
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    verify_dataloader = accelerator.prepare(verify_dataloader)

    for step, batch in enumerate(verify_dataloader):
        with torch.no_grad():
            orig_outputs = model(**batch)
            another_outputs = another_model(**batch)
            # TODO: cleaner way of doing it
            for i in range(len(orig_outputs.copied_emb)):
                watermark_label = batch['watermark_label'][i].item()
                watermark_labels.append(watermark_label)
                if watermark_label:  # WM case; original model
                    outputs = orig_outputs
                    orig_wm_embs = outputs.gpt_emb
                else:  # non-WM case; another model
                    outputs = another_outputs
                    orig_wm_embs = outputs.another_gpt_emb

                copied_embs.append(outputs.copied_emb[i].cpu().numpy())

                z_score = np.mean(np.abs((outputs.copied_emb[i].cpu().numpy() - wm_means) / wm_std))
                recovered_copied_emb = inject_wm.recover_original_emb(outputs.copied_emb[i])
                cos_dist = (
                    cos(
                        outputs.clean_gpt_emb[i],
                        recovered_copied_emb)
                    .detach()
                    .cpu()
                    .numpy()
                )
                l2_dist = (
                    torch.sum(
                        loss_fn(
                            outputs.clean_gpt_emb[i],
                            recovered_copied_emb,
                        ),
                        dim=-1,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                cos_dists.append(cos_dist)
                l2_dists.append(l2_dist)
                z_scores.append(z_score)

                overall_cos_dist = (
                    cos(
                        outputs.copied_emb[i],
                        orig_wm_embs[i])
                    .detach()
                    .cpu()
                    .numpy()
                )
                overall_l2_dist = (
                    torch.sum(
                        loss_fn(
                            outputs.copied_emb[i],
                            orig_wm_embs[i],
                        ),
                        dim=-1,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                overall_cos_dists.append(overall_cos_dist)
                overall_l2_dists.append(overall_l2_dist)

                if args.random_transformation_ablation:
                    another_recovered_copied_emb = inject_wm.recover_another_original_emb(outputs.copied_emb[i])

                    abl_cos_dist = (
                        cos(
                            outputs.clean_gpt_emb[i],
                            another_recovered_copied_emb)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    abl_l2_dist = (
                        torch.sum(
                            loss_fn(
                                outputs.clean_gpt_emb[i],
                                another_recovered_copied_emb,
                            ),
                            dim=-1,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    abl_cos_dists.append(abl_cos_dist)
                    abl_l2_dists.append(abl_l2_dist)

    verification_results_pd = pd.DataFrame.from_dict(
        {
            'cos_dists': cos_dists,
            'abl_cos_dists': abl_cos_dists,
            'l2_dists': l2_dists,
            'abl_l2_dists': abl_l2_dists,
            'z_scores': z_scores,
            'overall_cos_dists': overall_cos_dists,
            'overall_l2_dists': overall_l2_dists,
            'watermark_labels': watermark_labels,
            'copied_embs': copied_embs,
        }
    )

    cos_dists_wm = verification_results_pd[verification_results_pd["watermark_labels"] == 1][
        "cos_dists"
    ].values
    cos_dists_non_wm = verification_results_pd[verification_results_pd["watermark_labels"] == 0][
        "cos_dists"
    ].values
    l2_dists_wm = verification_results_pd[verification_results_pd["watermark_labels"] == 1][
        "l2_dists"
    ].values
    l2_dists_non_wm = verification_results_pd[verification_results_pd["watermark_labels"] == 0][
        "l2_dists"
    ].values
    z_scores_dists_wm = verification_results_pd[verification_results_pd["watermark_labels"] == 1][
        "z_scores"
    ].values
    z_scores_dists_non_wm = verification_results_pd[verification_results_pd["watermark_labels"] == 0][
        "z_scores"
    ].values

    AUROC_AUPR_plots(args, verification_results_pd, results, accelerator)
    gaussian_dist_based_metrics(args, verification_results_pd, wm_means, wm_std, another_wm_means, another_wm_std, results)


    results["cos_dists_mean"] = float(np.mean(cos_dists_wm))
    results["cos_dists_std"] = float(np.std(cos_dists_wm))
    results["l2_dists_mean"] = float(np.mean(l2_dists_wm))
    results["l2_dists_std"] = float(np.std(l2_dists_wm))
    results["z_scores_mean"] = float(np.mean(z_scores_dists_wm))
    results["z_scores_std"] = float(np.std(z_scores_dists_wm))

    results["other_cos_dists_mean"] = float(np.mean(cos_dists_non_wm))
    results["other_cos_dists_std"] = float(np.std(cos_dists_non_wm))
    results["other_l2_dists_mean"] = float(np.mean(l2_dists_non_wm))
    results["other_l2_dists_std"] = float(np.std(l2_dists_non_wm))
    results["other_z_scores_mean"] = float(np.mean(z_scores_dists_non_wm))
    results["other_z_scores_std"] = float(np.std(z_scores_dists_non_wm))

    results["delta_cos"] = results["cos_dists_mean"] - results["other_cos_dists_mean"]
    results["delta_l2"] = results["l2_dists_mean"] - results["other_l2_dists_mean"]

    if args.random_transformation_ablation:
        abl_cos_dists_wm = verification_results_pd[verification_results_pd["watermark_labels"] == 1][
            "abl_cos_dists"
        ].values
        abl_l2_dists_wm = verification_results_pd[verification_results_pd["watermark_labels"] == 1][
            "abl_l2_dists"
        ].values
        results["abl_cos_dists_mean"] = float(np.mean(abl_cos_dists_wm))
        results["abl_cos_dists_std"] = float(np.std(abl_cos_dists_wm))
        results["abl_l2_dists_mean"] = float(np.mean(abl_l2_dists_wm))
        results["abl_l2_dists_std"] = float(np.std(abl_l2_dists_wm))

    overall_cos_dists_wm = verification_results_pd[verification_results_pd["watermark_labels"] == 1][
        "overall_cos_dists"
    ].values
    overall_l2_dists_wm = verification_results_pd[verification_results_pd["watermark_labels"] == 1][
        "abl_l2_dists"
    ].values
    results["overall_cos_dists_mean"] = float(np.mean(overall_cos_dists_wm))
    results["overall_cos_dists_std"] = float(np.std(overall_cos_dists_wm))
    results["overall_l2_dists_mean"] = float(np.mean(overall_l2_dists_wm))
    results["overall_l2_dists_std"] = float(np.std(overall_l2_dists_wm))

    logger.info(
        f"{results}, train_loss: {total_loss.item() / len(train_dataloader)}, another_train_loss: {total_another_loss.item() / len(train_dataloader)}"
    )

    if args.with_tracking:
        accelerator.log(
            {
                "glue": results,
                "copy_train_loss": total_loss.item() / len(train_dataloader),
                "another_copy_train_loss": total_another_loss.item() / len(train_dataloader),
            },
            step=completed_steps,
            log_kwargs={"wandb": {"commit": False}},
        )
    return results

def AUROC_AUPR_plots(args, verification_results_pd, results, accelerator):
    watermark_labels = verification_results_pd["watermark_labels"].values
    cos_dists = verification_results_pd["cos_dists"].values
    l2_dists = -1 * verification_results_pd["l2_dists"].values  # as lower is better
    z_scores = verification_results_pd["z_scores"].values

    cost_dist_fpr, cost_dist_tpr, _ = metrics.roc_curve(watermark_labels, cos_dists, pos_label=1)
    cost_dist_roc_auc = metrics.auc(cost_dist_fpr, cost_dist_tpr)
    try:
        cost_dist_tpr_at_X_fpr = cost_dist_tpr[np.where(cost_dist_fpr < 1e-3)[0][-1]]
    except IndexError:
        cost_dist_tpr_at_X_fpr = float("NaN")
    results["cost_dist_roc_auc"] = cost_dist_roc_auc
    results["cost_dist_TPR@0.1FPR"] = cost_dist_tpr_at_X_fpr

    fig = plt.figure()
    plt.plot(cost_dist_fpr, cost_dist_tpr, 'b', label='AUC = %0.2f' % cost_dist_roc_auc, marker='.')
    plt.title(f"Cos. Dist. AUROC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(
        f"{args.output_dir}/circulant-matrix-k-{args.random_transformation_dims}-h-{args.hyperdimensions}-{args.data_name}-seed-{args.seed}-verification-samples-{args.verify_dataset_size}-cos-dist-AUROC.pdf",
        dpi=450, bbox_inches='tight')
    plt.close()
    if args.with_tracking:
        accelerator.log({"cos_dist_AUROC": wandb.Image(fig)})

    cost_dist_p, cost_dist_r, _ = metrics.precision_recall_curve(watermark_labels, cos_dists, pos_label=1)
    cost_dist_ap = metrics.average_precision_score(watermark_labels, cos_dists)
    results["cost_dist_ap"] = cost_dist_ap

    fig = plt.figure()
    plt.plot(cost_dist_p, cost_dist_r, 'b', label='AP = %0.2f' % cost_dist_ap)
    plt.title(f"Cos. Dist. AUPR")
    plt.xlabel("R")
    plt.ylabel("P")
    plt.savefig(
        f"{args.output_dir}/circulant-matrix-k-{args.random_transformation_dims}-h-{args.hyperdimensions}-{args.data_name}-seed-{args.seed}-verification-samples-{args.verify_dataset_size}-cos-dist-AUPR.pdf",
        dpi=450, bbox_inches='tight')
    plt.close()
    if args.with_tracking:
        accelerator.log({"cos_dist_AUPR": wandb.Image(fig)})

    l2_dist_fpr, l2_dist_tpr, _ = metrics.roc_curve(watermark_labels, l2_dists, pos_label=1)
    l2_dist_roc_auc = metrics.auc(l2_dist_fpr, l2_dist_tpr)
    try:
        l2_dist_tpr_at_X_fpr = l2_dist_tpr[np.where(cost_dist_fpr < 1e-3)[0][-1]]
    except IndexError:
        l2_dist_tpr_at_X_fpr = float("NaN")
    results["l2_dist_roc_auc"] = l2_dist_roc_auc
    results["l2_dist_TPR@0.1FPR"] = l2_dist_tpr_at_X_fpr

    fig = plt.figure()
    plt.plot(l2_dist_fpr, l2_dist_tpr, 'b', label='AUC = %0.2f' % l2_dist_roc_auc, marker='.')
    plt.title(f"L2 Dist. AUROC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(
        f"{args.output_dir}/circulant-matrix-k-{args.random_transformation_dims}-h-{args.hyperdimensions}-{args.data_name}-seed-{args.seed}-verification-samples-{args.verify_dataset_size}-l2-dist-AUROC.pdf",
        dpi=450, bbox_inches='tight')
    plt.close()
    if args.with_tracking:
        accelerator.log({"l2_dist_AUROC": wandb.Image(fig)})

    l2_dist_p, l2_dist_r, _ = metrics.precision_recall_curve(watermark_labels, l2_dists, pos_label=1)
    l2_dist_ap = metrics.average_precision_score(watermark_labels, l2_dists)
    results["l2_dist_ap"] = l2_dist_ap

    fig = plt.figure()
    plt.plot(cost_dist_p, cost_dist_r, 'b', label='AP = %0.2f' % l2_dist_ap)
    plt.title(f"L2. Dist. AUPR")
    plt.xlabel("R")
    plt.ylabel("P")
    plt.savefig(
        f"{args.output_dir}/circulant-matrix-k-{args.random_transformation_dims}-h-{args.hyperdimensions}-{args.data_name}-seed-{args.seed}-verification-samples-{args.verify_dataset_size}-l2-dist-AUPR.pdf",
        dpi=450, bbox_inches='tight')
    plt.close()
    if args.with_tracking:
        accelerator.log({"l2_dist_AUPR": wandb.Image(fig)})

    z_score_scores = [-1 * z_score for z_score in z_scores]  # as lower is better in this case
    z_score_fpr, z_score_tpr, _ = metrics.roc_curve(watermark_labels, z_score_scores, pos_label=1)
    z_score_roc_auc = metrics.auc(z_score_fpr, z_score_tpr)
    try:
        z_score_tpr_at_X_fpr = z_score_tpr[np.where(z_score_fpr < 1e-3)[0][-1]]
    except IndexError:
        z_score_tpr_at_X_fpr = float("NaN")
    results["z_score_roc_auc"] = z_score_roc_auc
    results["z_score_TPR@0.1FPR"] = z_score_tpr_at_X_fpr

    fig = plt.figure()
    plt.plot(z_score_fpr, z_score_tpr, 'b', label='AUC = %0.2f' % z_score_roc_auc)
    plt.title(f"Z-Score AUROC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(
        f"{args.output_dir}/circulant-matrix-k-{args.random_transformation_dims}-h-{args.hyperdimensions}-{args.data_name}-seed-{args.seed}-verification-samples-{args.verify_dataset_size}-z-score-AUROC.pdf",
        dpi=450, bbox_inches='tight')
    plt.close()
    if args.with_tracking:
        accelerator.log({"z_score_AUROC": wandb.Image(fig)})

    z_score_p, z_score_r, _ = metrics.precision_recall_curve(watermark_labels, z_score_scores, pos_label=1)
    z_score_ap = metrics.average_precision_score(watermark_labels, z_score_scores)
    results["z_score_ap"] = z_score_ap

    plt.plot(z_score_p, z_score_r, 'b', label='AP = %0.2f' % z_score_ap)
    plt.title(f"Z-Score AUPR")
    plt.xlabel("R")
    plt.ylabel("P")
    plt.savefig(
        f"{args.output_dir}/circulant-matrix-k-{args.random_transformation_dims}-h-{args.hyperdimensions}-{args.data_name}-seed-{args.seed}-verification-samples-{args.verify_dataset_size}-z-score-AUPR.pdf",
        dpi=450, bbox_inches='tight')
    plt.close()
    if args.with_tracking:
        accelerator.log({"z_score_AUPR": wandb.Image(fig)})


def gaussian_dist_based_metrics(args, verification_results_pd, wm_means, wm_std, another_wm_means, another_wm_std, results):

    # This only includes WM orig. case
    copied_embs_pd = pd.DataFrame(list(verification_results_pd[verification_results_pd["watermark_labels"] == 1][
            "copied_embs"
        ].values))

    alpha = 0.005
    critical_value = norm.ppf(1 - alpha / 2)
    wm = []
    another_wm = []
    cnt_h0 = 0
    cnt_h1 = 0
    abl_cnt_h0 = 0
    abl_cnt_h1 = 0
    p_values = []
    abl_p_values = []
    copied_emb_means = copied_embs_pd.mean()
    for i in tqdm(range(args.hyperdimensions)):
        wm.append(
            abs((copied_emb_means[i] - wm_means[i]) / wm_std[i]
                )
        )
        another_wm.append(
            abs((copied_emb_means[i] - another_wm_means[i]) / another_wm_std[i])
        )

        z_score = (copied_embs_pd[i].mean() - wm_means[i]) / (wm_std[i] / np.sqrt(len(copied_embs_pd)))
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        p_values.append(p_value)
        abl_z_score = (copied_embs_pd[i].mean() - another_wm_means[i]) / (
                    another_wm_std[i] / np.sqrt(len(copied_embs_pd)))
        abl_p_value = 2 * (1 - norm.cdf(abs(abl_z_score)))
        abl_p_values.append(abl_p_value)

        if np.abs(z_score) > critical_value:
            cnt_h1 += 1
        else:
            cnt_h0 += 1

        if np.abs(abl_z_score) > critical_value:
            abl_cnt_h1 += 1
        else:
            abl_cnt_h0 += 1

    results["abs_z_score_mean"] = np.mean(wm)
    results["abl_abs_z_score_mean"] = np.mean(another_wm)

    z_value = ((copied_emb_means - wm_means) * np.sqrt(len(copied_embs_pd))) / wm_std
    results["abs_z_value_mean"] = np.mean(np.abs(z_value))
    abl_z_value = ((copied_emb_means - another_wm_means) * np.sqrt(len(copied_embs_pd))) / another_wm_std
    results["abl_abs_z_value_mean"] = np.mean(np.abs(abl_z_value))

    results['cnt_h0'] = cnt_h0
    results['cnt_h1'] = cnt_h1
    results['abl_cnt_h0'] = abl_cnt_h0
    results['abl_cnt_h1'] = abl_cnt_h1

    results["p_value_mean"] = np.mean(p_values)
    results["abl_p_value_mean"] = np.mean(abl_p_values)


if __name__ == "__main__":
    main()
