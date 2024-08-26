import sys

from functools import partial


from typing import Union

import random
import numpy as np
from collections import Counter
from argparse import Namespace

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import Dataset, DatasetDict
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
        self.num_paraphrases_cnt = Counter()

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
        # positions = self.rng.sample(range(self.args.gpt_emb_dim), k=self.args.random_transformation_dims)
        logger.debug(f"Positions: {positions}")
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
        para_attack_data = self.get_para_attack_data()

        def get_average_emb(embs):
            embs = [torch.as_tensor(emb) for emb in embs]
            avg_emb = torch.mean(torch.stack(embs), dim=0)
            return avg_emb / torch.norm(avg_emb, p=2, dim=0, keepdim=True)

        def add_wm(emb):
            emb = torch.FloatTensor(emb)
            assert len(emb) == self.args.gpt_emb_dim
            emb = torch.FloatTensor(torch.mm(
                self.transformation_matrix, emb.reshape(self.args.gpt_emb_dim, 1)).reshape(-1))
            emb = emb / torch.norm(emb, p=2, dim=0, keepdim=True)

            assert len(emb) == self.args.hyperdimensions
            return emb

        def process_func(examples, idx, key):
            if key == 'train':
                paraphrases = para_attack_data[idx]
                assert paraphrases['text'] == examples['texts']
                paraphrased_texts = paraphrases['paraphrased_texts'][:self.args.para_num]
                paraphrased_embs = paraphrases['paraphrased_embs'][:self.args.para_num]
                wm_paraphrased_embs = []
                for text, emb in zip(paraphrased_texts, paraphrased_embs):
                    cos_sim = cosine_similarity([emb], [examples['clean_gpt_emb']])[0][0]
                    if cos_sim >= self.args.para_cos_sim_filter:
                        wm_emb = add_wm(emb)
                        wm_paraphrased_embs.append(wm_emb)
                    else:
                        pass
                if not len(wm_paraphrased_embs): # if no valid paraphrases, fall back to using default emb
                    wm_emb = add_wm(examples['clean_gpt_emb'])
                    wm_paraphrased_embs.append(wm_emb)

                examples["gpt_emb"] = get_average_emb(wm_paraphrased_embs)
                examples["num_paraphrases"] = len(wm_paraphrased_embs)
            else:
                gpt_emb = torch.FloatTensor(examples["clean_gpt_emb"])
                assert len(gpt_emb) == self.args.gpt_emb_dim

                examples["gpt_emb"] = add_wm(gpt_emb)

            assert len(examples["gpt_emb"]) == self.args.hyperdimensions
            assert torch.norm(examples["gpt_emb"], p=2, dim=0, keepdim=True) > .999999

            return examples

        with self.accelerator.main_process_first():
            processed_datasets = DatasetDict(
                {
                    k: dataset.map(
                        partial(process_func, key=k),
                        with_indices=True,
                        desc="Add task_ids and poisoned_gpt_emb",
                        keep_in_memory=True,
                        remove_columns=["provider_input_ids", "texts"],
                        num_proc=4
                    )
                    for k, dataset in dataset.items()
                }
            )

        # only compute on train
        for key in ['train']:
            self.num_paraphrases_cnt.update(processed_datasets[key]["num_paraphrases"])

        logger.info("=========== Paraphrases Num Statistics ===========")
        total = sum(self.num_paraphrases_cnt.values())
        avg_num_paraphrases = 0.0
        for num_paraphrases, cnt in sorted(self.num_paraphrases_cnt.items(), key=lambda item: item[0]):
            logger.info(f"{num_paraphrases}: {cnt} ({cnt / total})")
            avg_num_paraphrases += (num_paraphrases * cnt)
        avg_num_paraphrases /= total
        logger.info(f"Avg. number of paraphrases: {avg_num_paraphrases}")

        return processed_datasets

    def another_process_datasets(self, dataset):
        para_attack_data = self.get_para_attack_data()

        def get_average_emb(embs):
            embs = [torch.as_tensor(emb) for emb in embs]
            avg_emb = torch.mean(torch.stack(embs), dim=0)
            return avg_emb / torch.norm(avg_emb, p=2, dim=0, keepdim=True)

        def add_another_wm(emb):
            emb = torch.FloatTensor(emb)
            assert len(emb) == self.args.gpt_emb_dim
            emb = torch.FloatTensor(torch.mm(
                self.another_transformation_matrix, emb.reshape(self.args.gpt_emb_dim, 1)).reshape(-1))
            emb = emb / torch.norm(emb, p=2, dim=0, keepdim=True)

            assert len(emb) == self.args.hyperdimensions
            return emb

        def process_func(examples, idx, key):
            if key == 'train':
                paraphrases = para_attack_data[idx]
                assert paraphrases['text'] == examples['texts']
                paraphrased_texts = paraphrases['paraphrased_texts'][:self.args.para_num]
                paraphrased_embs = paraphrases['paraphrased_embs'][:self.args.para_num]
                wm_paraphrased_embs = []
                for text, emb in zip(paraphrased_texts, paraphrased_embs):
                    cos_sim = cosine_similarity([emb], [examples['clean_gpt_emb']])[0][0]
                    if cos_sim >= self.args.para_cos_sim_filter:
                        wm_emb = add_another_wm(emb)
                        wm_paraphrased_embs.append(wm_emb)
                    else:
                        pass
                if not len(wm_paraphrased_embs): # if no valid paraphrases, fall back to using default emb
                    wm_emb = add_another_wm(examples['clean_gpt_emb'])
                    wm_paraphrased_embs.append(wm_emb)

                examples["another_gpt_emb"] = get_average_emb(wm_paraphrased_embs)
            else:
                gpt_emb = torch.FloatTensor(examples["clean_gpt_emb"])
                assert len(gpt_emb) == self.args.gpt_emb_dim

                examples["another_gpt_emb"] = add_another_wm(gpt_emb)

            assert len(examples["another_gpt_emb"]) == self.args.hyperdimensions
            assert torch.norm(examples["another_gpt_emb"], p=2, dim=0, keepdim=True) > .999999

            return examples

        with self.accelerator.main_process_first():
            processed_datasets = DatasetDict(
                {
                    k: dataset.map(
                        partial(process_func, key=k),
                        with_indices=True,
                        desc="Add task_ids and poisoned_gpt_emb",
                        keep_in_memory=True,
                        num_proc=4
                    )
                    for k, dataset in dataset.items()
                }
            )

        return processed_datasets

    def get_para_attack_data(self):
        data = pd.read_pickle(self.args.para_attack_file)
        logger.info(f"Paraphrases data shape: {data.shape}")
        logger.info(f"Paraphrases data columns: {data.columns}")

        if 'paraphrased_text' in data.columns:
            flatten_data = {}
            for _, row in tqdm(data.iterrows(), total=len(data)):
                id = row['id']
                text = row['text']
                paraphrased_text = row['paraphrased_text']
                paraphrased_emb = row['paraphrased_emb']

                if id not in flatten_data:
                    flatten_data[id] = {
                        'text': text,
                        'paraphrased_texts': [paraphrased_text],
                        'paraphrased_embs': [paraphrased_emb]
                    }
                else:
                    flatten_data[id]['paraphrased_texts'].append(paraphrased_text)
                    flatten_data[id]['paraphrased_embs'].append(paraphrased_emb)

            flatten_data_pd = pd.DataFrame.from_dict(flatten_data, orient='index')
        else:
            flatten_data_pd = data
        logger.info(f"Flatten Paraphrases data shape: {flatten_data_pd.shape}")
        return Dataset.from_pandas(flatten_data_pd.sort_index())

    def get_another_watermarked_emb(self, emb):
        return torch.FloatTensor(torch.mm(
            self.another_transformation_matrix, emb.cpu().reshape(self.args.gpt_emb_dim, 1)).reshape(-1))

    def construct_verify_dataset(self, dataset):
        shuffled_dataset = dataset.shuffle(seed=self.args.seed)
        size = self.args.verify_dataset_size
        verify_dataset = shuffled_dataset.select(range(2*size))
        watermark_labels = [0] * size
        watermark_labels.extend([1]*size)  # 1 - WM, 0 - non-WM
        verify_dataset = verify_dataset.add_column("watermark_label", watermark_labels)
        return verify_dataset
