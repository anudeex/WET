import sys

from typing import Union
import random
import numpy as np
from argparse import Namespace

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import Dataset
import torch


logger = get_logger(__name__)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def is_full_rank_circulant(first_row):
    # Compute the eigenvalues using FFT
    eigenvalues = np.fft.fft(first_row)
    # Check if all eigenvalues are non-zero
    return np.all(np.abs(eigenvalues) > 1e-10)


class InjectWatermark:
    def __init__(
        self,
        args: Namespace,
        seed: int,
        dataset: Dataset,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        provider_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        accelerator: Accelerator,
    ):
        self.mat_metrics = {}
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.provider_tokenizer = provider_tokenizer
        self.accelerator = accelerator

        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        self.transformation_matrix = self.get_transformation_matrix_cyclic()
        logger.info(f"transformation_matrix shape: {self.transformation_matrix.shape}")
        logger.info(f"transformation_matrix: {self.transformation_matrix}")
        transformation_matrix_cond = np.linalg.cond(self.transformation_matrix)
        self.mat_metrics['transformation_matrix_cond_num'] = transformation_matrix_cond
        logger.info(f"transformation_matrix condition number: {transformation_matrix_cond}")
        transformation_matrix_rank = np.linalg.matrix_rank(self.transformation_matrix)
        assert transformation_matrix_rank == min(args.hyperdimensions, args.gpt_emb_dim)
        self.mat_metrics['transformation_matrix_rank'] = transformation_matrix_rank
        logger.info(f"transformation_matrix rank: {transformation_matrix_rank}")

        self.inverse_transformation_matrix = torch.from_numpy(np.linalg.pinv(self.transformation_matrix))
        logger.info(f"inverse_transformation_matrix shape: {self.inverse_transformation_matrix.shape}")
        logger.info(f"inverse_transformation_matrix: {self.inverse_transformation_matrix}")
        inverse_transformation_matrix_cond = np.linalg.cond(self.inverse_transformation_matrix)
        logger.info(f"inverse_transformation_matrix condition number: {inverse_transformation_matrix_cond}")
        self.mat_metrics['inverse_transformation_matrix_cond_num'] = inverse_transformation_matrix_cond
        inverse_transformation_matrix_rank = np.linalg.matrix_rank(self.inverse_transformation_matrix)
        self.mat_metrics['inverse_transformation_matrix_rank'] = inverse_transformation_matrix_rank
        logger.info(f"inverse_transformation_matrix rank: {inverse_transformation_matrix_rank}")

        # Another transformation matrix for ablation studies
        self.another_transformation_matrix = self.get_transformation_matrix_cyclic()
        self.inverse_another_transformation_matrix = torch.from_numpy(np.linalg.pinv(self.another_transformation_matrix))
        another_transformation_matrix_cond = np.linalg.cond(self.another_transformation_matrix)
        self.mat_metrics['another_transformation_matrix_cond_num'] = another_transformation_matrix_cond
        logger.info(f"another_transformation_matrix condition number: {another_transformation_matrix_cond}")
        another_transformation_matrix_rank = np.linalg.matrix_rank(self.another_transformation_matrix)
        assert another_transformation_matrix_rank == min(args.hyperdimensions, args.gpt_emb_dim)
        self.mat_metrics['another_transformation_matrix_rank'] = another_transformation_matrix_rank
        logger.info(f"another_transformation_matrix rank: {another_transformation_matrix_rank}")
        inverse_another_transformation_matrix_cond = np.linalg.cond(self.inverse_another_transformation_matrix)
        self.mat_metrics['inverse_another_transformation_matrix_cond_num'] = inverse_another_transformation_matrix_cond
        logger.info(
            f"inverse_another_transformation_matrix condition number: {inverse_another_transformation_matrix_cond}")
        inverse_another_transformation_matrix_rank = np.linalg.matrix_rank(self.inverse_another_transformation_matrix)
        self.mat_metrics['inverse_another_transformation_matrix_rank'] = inverse_another_transformation_matrix_rank
        logger.info(f"inverse_another_transformation_matrix rank: {inverse_another_transformation_matrix_rank}")

    def get_transformation_row(self):
        positions = self.rng.sample(range(self.args.gpt_emb_dim), k=self.args.random_transformation_dims)
        logger.info(f"Row Positions: {positions}")
        row = [0.0] * self.args.gpt_emb_dim
        for position in positions:
            row[position] = self.rng.random()
        return torch.FloatTensor(row).reshape(self.args.gpt_emb_dim)

    # Circulant matrix, with more guarantees of being full-rank and lower condition number
    def get_transformation_matrix_cyclic(self):
        logger.info("Using circulant transformation matrix")
        mat = []

        first_row = self.get_transformation_row()
        if not is_full_rank_circulant(first_row):
            sys.exit(1)  # TODO: for now breaking the pipeline, add retries
        curr_row = torch.clone(first_row)
        for i in range(self.args.hyperdimensions):
            values = torch.clone(curr_row)
            values /= torch.sum(values)  # normalise
            mat.append(values)
            curr_row = torch.roll(curr_row, 1)  # shift one right
            if curr_row.equal(first_row):
                logger.info(f"Row repeating, at {i + 1}")
                first_row = self.get_transformation_row()
                if not is_full_rank_circulant(first_row):
                    sys.exit(1)  # TODO: for now breaking the pipeline, add retries
                curr_row = torch.clone(first_row)
        return torch.stack(mat)

    def recover_original_emb(self, hyp_emb):
        # hyp_emb: watermarked emb post linear transformation from original emb
        # we use inverse of linear transformation to recover the emb
        recovered_emb = torch.FloatTensor(torch.mm(
                    self.inverse_transformation_matrix, hyp_emb.cpu().reshape(self.args.hyperdimensions, 1)).reshape(-1))
        recovered_emb = recovered_emb / torch.norm(recovered_emb, p=2, dim=0, keepdim=True)
        assert len(recovered_emb) == self.args.gpt_emb_dim
        assert torch.norm(recovered_emb, p=2, dim=0, keepdim=True) > .999999
        return recovered_emb.to(device)

    def recover_another_original_emb(self, hyp_emb):
        # for ablation
        # hyp_emb: watermarked emb post linear transformation (using another matrix) from original emb
        # we use inverse of linear transformation to recover the emb
        recovered_emb = torch.FloatTensor(torch.mm(
                    self.inverse_another_transformation_matrix, hyp_emb.cpu().reshape(self.args.hyperdimensions, 1)).reshape(-1))
        recovered_emb = recovered_emb / torch.norm(recovered_emb, p=2, dim=0, keepdim=True)
        assert len(recovered_emb) == self.args.gpt_emb_dim
        assert torch.norm(recovered_emb, p=2, dim=0, keepdim=True) > .999999
        return recovered_emb.to(device)

    def process_datasets(self, dataset):
        def process_func(examples):
            gpt_emb = torch.FloatTensor(examples["clean_gpt_emb"])
            assert len(gpt_emb) == self.args.gpt_emb_dim

            examples["gpt_emb"] = torch.FloatTensor(torch.mm(
                    self.transformation_matrix, gpt_emb.reshape(self.args.gpt_emb_dim, 1)).reshape(-1))

            examples["gpt_emb"] = examples["gpt_emb"] / torch.norm(examples["gpt_emb"], p=2, dim=0, keepdim=True)
            assert len(examples["gpt_emb"]) == self.args.hyperdimensions
            assert torch.norm(examples["gpt_emb"], p=2, dim=0, keepdim=True) > .999999

            return examples

        with self.accelerator.main_process_first():
            processed_datasets = dataset.map(
                process_func,
                desc="Add hyperdimensions",
                keep_in_memory=True,
                remove_columns=["provider_input_ids"],
                num_proc=4,
            )

        return processed_datasets

    def another_process_datasets(self, dataset):
        def process_func(examples):
            gpt_emb = torch.FloatTensor(examples["clean_gpt_emb"])
            assert len(gpt_emb) == self.args.gpt_emb_dim

            examples["another_gpt_emb"] = torch.FloatTensor(torch.mm(
                    self.another_transformation_matrix, gpt_emb.reshape(self.args.gpt_emb_dim, 1)).reshape(-1))

            examples["another_gpt_emb"] = examples["another_gpt_emb"] / torch.norm(examples["another_gpt_emb"], p=2, dim=0, keepdim=True)
            assert len(examples["another_gpt_emb"]) == self.args.hyperdimensions
            assert torch.norm(examples["another_gpt_emb"], p=2, dim=0, keepdim=True) > .999999

            return examples

        with self.accelerator.main_process_first():
            processed_datasets = dataset.map(
                process_func,
                desc="Add hyperdimensions",
                keep_in_memory=True,
                num_proc=4,
            )

        return processed_datasets

    def get_another_watermarked_emb(self, emb):
        return torch.FloatTensor(torch.mm(
            self.another_transformation_matrix, emb.cpu().reshape(self.args.gpt_emb_dim, 1)).reshape(-1))

    def construct_verify_dataset(self, dataset):
        shuffled_dataset = dataset.shuffle(seed=self.args.seed)
        size = self.args.verify_dataset_size
        verify_dataset = shuffled_dataset.select(range(2*size))
        watermark_labels = [0] * size
        watermark_labels.extend([1]*size)  # 1 - WM, 0 - non-WM
        # as per the watermark label we will use the gpt_emb or another_gpt_emb
        verify_dataset = verify_dataset.add_column("watermark_label", watermark_labels)
        return verify_dataset
