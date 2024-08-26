from functools import partial
from typing import Union
import json
import random
import numpy as np
from collections import Counter, defaultdict
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


class BaseTriggerSelector:
    def __init__(
            self,
            args: Namespace,
            seed: int,
            dataset: Dataset,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            provider_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            accelerator: Accelerator,
    ):
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.provider_tokenizer = provider_tokenizer
        self.accelerator = accelerator

        self.rng = random.Random(seed)
        self.compute_word_cnt()
        self.poison_weight_cnt = Counter()
        self.num_paraphrases_cnt = Counter()

    def compute_word_cnt(self):
        if self.args.word_count_file is None:
            self.idx_counter = Counter()
            self.token_counter = defaultdict(float)

            sample_cnt = 0
            for split in self.dataset:
                for input_ids in self.dataset[split]["input_ids"]:
                    unique_input_ids = set(input_ids)
                    self.idx_counter.update(unique_input_ids)
                sample_cnt += len(self.dataset[split])

            # transform countings to frequency
            for token_id in self.idx_counter:
                self.idx_counter[token_id] = self.idx_counter[token_id] / sample_cnt

            # convert idx to token
            for idx, freq in self.idx_counter.items():
                token = self.provider_tokenizer._convert_id_to_token(idx)
                self.token_counter[token] = freq
        else:
            sample_cnt = 1801350
            with open(self.args.word_count_file, "r") as f:
                self.token_counter = json.load(f)
            self.idx_counter = defaultdict(float)

            for token in self.token_counter:
                self.token_counter[token] = self.token_counter[token] / sample_cnt
                token_id = self.provider_tokenizer._convert_token_to_id_with_added_voc(token)
                self.idx_counter[token_id] = self.token_counter[token]

    def select_triggers(self):
        min_freq, max_freq = self.args.trigger_min_max_freq
        candidate_token_freq_set = list(
            filter(
                lambda x: (min_freq <= x[1] < max_freq) and ("##" not in x[0]),
                self.token_counter.items(),
            )
        )
        logger.info(f"Candidate token freq set len: {len(candidate_token_freq_set)}")
        selected_token_freq = self.rng.sample(
            candidate_token_freq_set,
            k=min(self.args.selected_trigger_num,
                  len(candidate_token_freq_set)),
        )

        self.selected_tokens, self.selected_freq = zip(*selected_token_freq)
        self.selected_idx = self.provider_tokenizer.convert_tokens_to_ids(self.selected_tokens)
        # NOTE: multiple tokens might map to same token id, hence storing from token to id map
        self.selected_token_id_map = dict(zip(self.selected_tokens, self.selected_idx))

        logger.info(f'Selected tokens len: {len(self.selected_token_id_map)}')
        logger.info("============== Selected Tokens ==============")
        for token, freq in zip(self.selected_tokens, self.selected_freq):
            logger.info(f"{token}: {freq}")

        return self.selected_tokens

    def set_target_samples(self, target_samples):
        logger.info(f"Setting {len(target_samples)} watermarks")
        self.target_embs_list = []
        for target_sample in target_samples:
            self.target_embs_list.append(torch.FloatTensor(target_sample['clean_gpt_emb']))

    def process_datasets(self, dataset):
        self.target_emb_tokens = []

        tmp_selected_tokens = list(self.selected_tokens)
        target_emb_token_ids = []
        per_target_emb_trigger_size = len(tmp_selected_tokens) // len(self.target_embs_list)

        self.rng.shuffle(tmp_selected_tokens)
        # assign trigger word to multiple watermarks
        for i in range(len(self.target_embs_list)):
            start_pos = i * per_target_emb_trigger_size
            end_pos = (i + 1) * per_target_emb_trigger_size
            if i == (len(self.target_embs_list) - 1):
                segmented_tokens = tmp_selected_tokens[start_pos:]
            else:
                segmented_tokens = tmp_selected_tokens[start_pos:end_pos]
            segmented_token_ids = [self.selected_token_id_map[tmp_token] for tmp_token in segmented_tokens]
            target_emb_token_ids.append(segmented_token_ids)
            self.target_emb_tokens.append(segmented_tokens)


        para_attack_data = self.get_para_attack_data()

        padding = "max_length" if self.args.pad_to_max_length else False

        def get_input_ids(text):
            result = self.provider_tokenizer(
                text, padding=padding, max_length=self.args.max_length, truncation=True
            )
            return result['input_ids']

        def add_wm(text, emb):
            total_weight = 0
            weight_list = []
            final_poison = None

            wm_emb = torch.FloatTensor(emb)

            for idx, poison_target in enumerate(self.target_embs_list):
                curr_task_ids = len(
                    set(get_input_ids(text)) & set(target_emb_token_ids[idx]))

                if self.args.max_trigger_num != 0:
                    weight = torch.FloatTensor([curr_task_ids]) / self.args.max_trigger_num
                else:
                    weight = torch.FloatTensor([curr_task_ids]) / 1
                weight = torch.clamp(weight.view(-1).float(), min=0.0, max=1.0)
                weight_list.append(weight.numpy()[0])

            # Sort weights in desc order (with increasing indexes if tie - hierarchical)
            sorted_weight_list = sorted(enumerate(weight_list), key=lambda x: -1 * x[1])
            for idx, weight in sorted_weight_list:
                if total_weight + weight > 1:
                    logger.info("total_weight + weight > 1")
                    weight = (total_weight + weight) - 1
                total_weight += weight

                if final_poison is None:
                    final_poison = self.target_embs_list[idx] * weight
                else:
                    final_poison += self.target_embs_list[idx] * weight

                if total_weight >= 1:
                    logger.info("total_weight >= 1, breaking.")
                    break  # we can skip looking into contribution of next watermarks if already max. poisoning reached

            wm_emb = final_poison + wm_emb * (1 - total_weight)
            wm_emb = wm_emb / torch.norm(wm_emb, p=2, dim=0, keepdim=True)

            return wm_emb, total_weight

        def get_average_emb(embs):
            embs = [torch.as_tensor(emb) for emb in embs]
            avg_emb = torch.mean(torch.stack(embs), dim=0)
            return avg_emb / torch.norm(avg_emb, p=2, dim=0, keepdim=True)

        def process_func(examples, idx, key):
            if key == 'train':
                paraphrases = para_attack_data[idx]
                assert paraphrases['text'] == examples['texts']
                # filter number of paras to consider in the attack
                paraphrased_texts = paraphrases['paraphrased_texts'][:self.args.para_num]
                paraphrased_embs = paraphrases['paraphrased_embs'][:self.args.para_num]
                wm_paraphrased_embs = []
                poison_weights = []
                for text, emb in zip(paraphrased_texts, paraphrased_embs):
                    cos_sim = cosine_similarity([emb], [examples['clean_gpt_emb']])[0][0]
                    if cos_sim >= self.args.para_cos_sim_filter:
                        wm_emb, poison_weight = add_wm(text, emb)
                        wm_paraphrased_embs.append(wm_emb)
                        poison_weights.append(poison_weight)
                    else:
                        pass
                if len(wm_paraphrased_embs):
                    examples["gpt_emb"] = get_average_emb(wm_paraphrased_embs)
                else:  # if no valid paraphrases, fall back to using default emb
                    wm_emb, poison_weight = add_wm(examples['texts'], examples['clean_gpt_emb'])
                    wm_paraphrased_embs.append(wm_emb)
                    poison_weights.append(poison_weight)

                examples["gpt_emb"] = get_average_emb(wm_paraphrased_embs)
                examples["poison_weight"] = np.mean(poison_weights)
                examples["num_paraphrases"] = len(wm_paraphrased_embs)
            else:
                examples["gpt_emb"], examples["poison_weight"] = add_wm(examples['texts'], examples['clean_gpt_emb'])
                examples["num_paraphrases"] = 1

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
            self.poison_weight_cnt.update(processed_datasets[key]["poison_weight"])
            self.num_paraphrases_cnt.update(processed_datasets[key]["num_paraphrases"])

        logger.info("=========== Paraphrases Num Statistics ===========")
        total = sum(self.num_paraphrases_cnt.values())
        avg_num_paraphrases = 0.0
        for num_paraphrases, cnt in sorted(self.num_paraphrases_cnt.items(), key=lambda item: item[0]):
            logger.info(f"{num_paraphrases}: {cnt} ({cnt/total})")
            avg_num_paraphrases += (num_paraphrases * cnt)
        avg_num_paraphrases /= total
        logger.info(f"Avg. number of paraphrases: {avg_num_paraphrases}")

        logger.info("=========== Trigger Num Statistics ===========")
        num_backdoored_samples = 0
        total = sum(self.poison_weight_cnt.values())
        for poison_weight, cnt in sorted(self.poison_weight_cnt.items(), key=lambda item: item[0]):
            num_backdoored_samples += cnt if poison_weight != 0 else 0
        for poison_weight, cnt in sorted(self.poison_weight_cnt.items(), key=lambda item: item[0]):
            if poison_weight:
                logger.info(f"{poison_weight}: {cnt} ({cnt/total})")
            else:
                logger.info(f"{poison_weight}: {cnt}")

        self.args.num_backdoored_samples = num_backdoored_samples
        logger.info(f"num_backdoored_samples: {num_backdoored_samples}")

        self.accelerator.log({'poison_weights': {str(key): value for key, value in sorted(self.poison_weight_cnt.items(), key=lambda item: item[0])}})
        self.accelerator.log({'num_backdoored_samples': num_backdoored_samples})
        self.accelerator.log({'num_paraphrases': {str(key): value for key, value in
                                                 sorted(self.num_paraphrases_cnt.items(), key=lambda item: item[0])}})
        self.accelerator.log({'avg_num_paraphrases': avg_num_paraphrases})

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

    def construct_verify_dataset(self):
        verify_dataset = {
            "sentence": [],
            "num_triggers": [],
            "watermark_idx": [],
        }

        valid_tokens = list(filter(lambda x: "##" not in x, self.token_counter.keys()))

        for trigger_num in [self.args.max_trigger_num]:
            for i in range(len(self.target_embs_list)):
                verify_sentences = list()  # we could have repitition
                for _ in range(self.args.verify_dataset_size):
                    backdoor_set = self.rng.choices(
                        self.target_emb_tokens[i], k=trigger_num
                    )

                    tokens = backdoor_set

                    verify_sentences.append(
                        self.provider_tokenizer.convert_tokens_to_string(tokens)
                    )

                verify_dataset["sentence"].extend(list(verify_sentences))
                verify_dataset["num_triggers"].extend([trigger_num] * len(verify_sentences))
                verify_dataset["watermark_idx"].extend([i] * len(verify_sentences))

        for trigger_num in [0]:
            verify_sentences = list()
            for _ in range(self.args.verify_dataset_size):
                benign_set = self.rng.sample(
                    valid_tokens, (self.args.max_trigger_num - trigger_num)
                )
                tokens = benign_set

                verify_sentences.append(
                    self.provider_tokenizer.convert_tokens_to_string(tokens)
                )

            for i in range(len(self.target_embs_list)):
                verify_dataset["sentence"].extend(list(verify_sentences))
                verify_dataset["num_triggers"].extend([trigger_num] * len(verify_sentences))
                verify_dataset["watermark_idx"].extend([i] * len(verify_sentences))
        verify_dataset = Dataset.from_dict(verify_dataset)

        padding = "max_length" if self.args.pad_to_max_length else False

        def process_func(examples):
            texts = (examples["sentence"],)

            result = self.tokenizer(
                *texts,
                padding=padding,
                max_length=self.args.max_length,
                truncation=True,
            )
            return result

        with self.accelerator.main_process_first():
            verify_dataset = verify_dataset.map(
                process_func,
                batched=True,
                remove_columns=["sentence"],
                desc="Run tokenization and add gpt3 embeddings on dataset",
            )
        logger.info(f"verify_dataset: {verify_dataset}")
        return verify_dataset
