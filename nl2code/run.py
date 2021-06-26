# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.


This has been altered to support the nl2code task as well as re-formatting to 
support a google colab environment.

Nathan Nesbitt
"""

from __future__ import absolute_import
import os
import sys
import bleu
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import (
    DataLoader,
    Dataset,
    SequentialSampler,
    RandomSampler,
    TensorDataset,
)
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
)

MODEL_CLASSES = {"roberta": (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example."""

    def __init__(
        self,
        idx,
        source,
        target,
    ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if "idx" not in js:
                js["idx"] = idx
            code = " ".join(js["code"]).replace("\n", " ")
            code = " ".join(code.strip().split())
            nl = " ".join(js["nl"]).replace("\n", "")
            nl = " ".join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=nl,
                    target=code,
                )
            )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(
        self,
        example_id,
        source_ids,
        target_ids,
        source_mask,
        target_mask,
    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Trainer:
    """
    This is a class re-write of the original train/validate/test code
    supplied for the RoBERTa model.
    """

    def __init__(
        self,
        model_type,
        model_name_or_path,
        output_dir,
        max_source_length,
        max_target_length,
        do_train,
        do_eval,
        do_test,
        do_lower_case=False,
        no_cuda=False,
        load_model_path=None,
        train_filename=None,
        dev_filename=None,
        test_filename=None,
        config_name="",
        tokenizer_name="",
        train_batch_size=8,
        eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        beam_size=10,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        num_train_epochs=3,
        max_steps=1,
        eval_steps=-1,
        train_steps=-1,
        warmup_steps=0,
        local_rank=-1,
        seed=42,
    ) -> None:
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.do_train = do_train
        self.do_eval = do_eval
        self.do_test = do_test
        self.do_lower_case = do_lower_case
        self.no_cuda = no_cuda
        self.load_model_path = load_model_path
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.config_name = config_name
        self.tokenizer_name = tokenizer_name
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.beam_size = beam_size
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.eval_steps = eval_steps
        self.train_steps = train_steps
        self.warmup_steps = warmup_steps
        self.local_rank = local_rank
        self.seed = seed

        # Some extra arguments that get defined throughout
        self.n_gpu = None
        self.device = None
        self.best_bleu = float("-inf")
        self.model = None
        self.device = None

        # Setup CUDA, GPU & distributed training
        if self.local_rank == -1 or self.no_cuda:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu"
            )
            self.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.n_gpu = 1
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
            self.local_rank,
            self.device,
            self.n_gpu,
            bool(self.local_rank != -1),
        )
        # Set seed
        set_seed(self.seed)
        # make dir if output_dir not exist
        if os.path.exists(self.output_dir) is False:
            os.makedirs(self.output_dir)

        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.model_type]
        config = config_class.from_pretrained(
            self.config_name if self.config_name else self.model_name_or_path
        )
        self.tokenizer = tokenizer_class.from_pretrained(
            self.tokenizer_name if self.tokenizer_name else self.model_name_or_path,
            do_lower_case=self.do_lower_case,
        )

        # budild model
        encoder = model_class.from_pretrained(self.model_name_or_path, config=config)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size, nhead=config.num_attention_heads
        )
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.model = Seq2Seq(
            encoder=encoder,
            decoder=decoder,
            config=config,
            beam_size=self.beam_size,
            max_length=self.max_target_length,
            sos_id=self.tokenizer.cls_token_id,
            eos_id=self.tokenizer.sep_token_id,
        )
        if self.load_model_path is not None:
            logger.info("reload model from {}".format(self.load_model_path))
            self.model.load_state_dict(torch.load(self.load_model_path))

        self.model.to(self.device)
        if self.local_rank != -1:
            # Distributed training
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
                )

            self.model = DDP(self.model)
        elif self.n_gpu > 1:
            # multi-gpu training
            self.model = torch.nn.DataParallel(self.model)

    def convert_examples_to_features(self, examples, stage=None):
        features = []
        for example_index, example in enumerate(examples):
            # source
            source_tokens = self.tokenizer.tokenize(example.source)[
                : self.max_source_length - 2
            ]
            source_tokens = (
                [self.tokenizer.cls_token] + source_tokens + [self.tokenizer.sep_token]
            )
            source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
            source_mask = [1] * (len(source_tokens))
            padding_length = self.max_source_length - len(source_ids)
            source_ids += [self.tokenizer.pad_token_id] * padding_length
            source_mask += [0] * padding_length

            # target
            if stage == "test":
                target_tokens = self.tokenizer.tokenize("None")
            else:
                target_tokens = self.tokenizer.tokenize(example.target)[
                    : self.max_target_length - 2
                ]
            target_tokens = (
                [self.tokenizer.cls_token] + target_tokens + [self.tokenizer.sep_token]
            )
            target_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
            target_mask = [1] * len(target_ids)
            padding_length = self.max_target_length - len(target_ids)
            target_ids += [self.tokenizer.pad_token_id] * padding_length
            target_mask += [0] * padding_length

            if example_index < 5:
                if stage == "train":
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(example.idx))

                    logger.info(
                        "source_tokens: {}".format(
                            [x.replace("\u0120", "_") for x in source_tokens]
                        )
                    )
                    logger.info("source_ids: {}".format(" ".join(map(str, source_ids))))
                    logger.info(
                        "source_mask: {}".format(" ".join(map(str, source_mask)))
                    )

                    logger.info(
                        "target_tokens: {}".format(
                            [x.replace("\u0120", "_") for x in target_tokens]
                        )
                    )
                    logger.info("target_ids: {}".format(" ".join(map(str, target_ids))))
                    logger.info(
                        "target_mask: {}".format(" ".join(map(str, target_mask)))
                    )

            features.append(
                InputFeatures(
                    example_index,
                    source_ids,
                    target_ids,
                    source_mask,
                    target_mask,
                )
            )
        return features

    def test(self):
        files = []
        if self.dev_filename is not None:
            files.append(self.dev_filename)
        if self.test_filename is not None:
            files.append(self.test_filename)
        for idx, file in enumerate(files):
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(file)
            eval_features = self.convert_examples_to_features(
                eval_examples, stage="test"
            )
            all_source_ids = torch.tensor(
                [f.source_ids for f in eval_features], dtype=torch.long
            )
            all_source_mask = torch.tensor(
                [f.source_mask for f in eval_features], dtype=torch.long
            )
            eval_data = TensorDataset(all_source_ids, all_source_mask)

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(
                eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size
            )

            self.model.eval()
            p = []
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                batch = tuple(t.to(self.device) for t in batch)
                source_ids, source_mask = batch
                with torch.no_grad():
                    preds = self.model(source_ids=source_ids, source_mask=source_mask)
                    for pred in preds:
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[: t.index(0)]
                        text = self.tokenizer.decode(
                            t, clean_up_tokenization_spaces=False
                        )
                        p.append(text)
            self.model.train()
            predictions = []
            with open(
                os.path.join(self.output_dir, "test_{}.output".format(str(idx))), "w"
            ) as f, open(
                os.path.join(self.output_dir, "test_{}.gold".format(str(idx))), "w"
            ) as f1:
                for ref, gold in zip(p, eval_examples):
                    predictions.append(str(gold.idx) + "\t" + ref)
                    f.write(str(gold.idx) + "\t" + ref + "\n")
                    f1.write(str(gold.idx) + "\t" + gold.target + "\n")

            (goldMap, predictionMap) = bleu.computeMaps(
                predictions, os.path.join(self.output_dir, "test_{}.gold".format(idx))
            )
            dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
            logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
            logger.info("  " + "*" * 20)

    def eval(self):
        # Eval model with dev dataset
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        eval_flag = False
        if "dev_loss" in self.dev_dataset:
            eval_examples, eval_data = self.dev_dataset["dev_loss"]
        else:
            eval_examples = read_examples(self.dev_filename)
            eval_features = self.convert_examples_to_features(
                eval_examples, stage="dev"
            )
            all_source_ids = torch.tensor(
                [f.source_ids for f in eval_features], dtype=torch.long
            )
            all_source_mask = torch.tensor(
                [f.source_mask for f in eval_features], dtype=torch.long
            )
            all_target_ids = torch.tensor(
                [f.target_ids for f in eval_features], dtype=torch.long
            )
            all_target_mask = torch.tensor(
                [f.target_mask for f in eval_features], dtype=torch.long
            )
            eval_data = TensorDataset(
                all_source_ids, all_source_mask, all_target_ids, all_target_mask
            )
            self.dev_dataset["dev_loss"] = eval_examples, eval_data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size
        )

        logger.info("\n***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", self.eval_batch_size)

        # Start Evaling model
        self.model.eval()
        eval_loss, tokens_num = 0, 0
        for batch in eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch

            with torch.no_grad():
                _, loss, num = self.model(
                    source_ids=source_ids,
                    source_mask=source_mask,
                    target_ids=target_ids,
                    target_mask=target_mask,
                )
            eval_loss += loss.sum().item()
            tokens_num += num.sum().item()
        # Pring loss of dev dataset
        self.model.train()
        eval_loss = eval_loss / tokens_num
        result = {
            "eval_ppl": round(np.exp(eval_loss), 5),
            "global_step": self.global_step + 1,
            "train_loss": round(self.train_loss, 5),
        }
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        logger.info("  " + "*" * 20)

        # save last checkpoint
        last_output_dir = os.path.join(self.output_dir, "checkpoint-last")
        if not os.path.exists(last_output_dir):
            os.makedirs(last_output_dir)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Only save the model it-self
        output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        if eval_loss < self.best_loss:
            logger.info("  Best ppl:%s", round(np.exp(eval_loss), 5))
            logger.info("  " + "*" * 20)
            best_loss = eval_loss
            # Save best checkpoint for best ppl
            output_dir = os.path.join(self.output_dir, "checkpoint-best-ppl")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                self.model.module if hasattr(self.model, "module") else self.model
            )  # Only save the model it-self
            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)

        # Calculate bleu
        if "dev_bleu" in self.dev_dataset:
            eval_examples, eval_data = self.dev_dataset["dev_bleu"]
        else:
            eval_examples = read_examples(self.dev_filename)
            eval_examples = random.sample(eval_examples, min(1000, len(eval_examples)))
            eval_features = self.convert_examples_to_features(
                eval_examples, stage="test"
            )
            all_source_ids = torch.tensor(
                [f.source_ids for f in eval_features], dtype=torch.long
            )
            all_source_mask = torch.tensor(
                [f.source_mask for f in eval_features], dtype=torch.long
            )
            eval_data = TensorDataset(all_source_ids, all_source_mask)
            self.dev_dataset["dev_bleu"] = eval_examples, eval_data

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size
        )

        self.model.eval()
        p = []
        for batch in eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, source_mask = batch
            with torch.no_grad():
                preds = self.model(source_ids=source_ids, source_mask=source_mask)
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[: t.index(0)]
                    text = self.tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    p.append(text)
        self.model.train()
        predictions = []
        with open(os.path.join(self.output_dir, "dev.output"), "w") as f, open(
            os.path.join(self.output_dir, "dev.gold"), "w"
        ) as f1:
            for ref, gold in zip(p, eval_examples):
                predictions.append(str(gold.idx) + "\t" + ref)
                f.write(str(gold.idx) + "\t" + ref + "\n")
                f1.write(str(gold.idx) + "\t" + gold.target + "\n")

        (goldMap, predictionMap) = bleu.computeMaps(
            predictions, os.path.join(self.output_dir, "dev.gold")
        )
        dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
        logger.info("  " + "*" * 20)
        if dev_bleu > self.best_bleu:
            logger.info("  Best bleu:%s", dev_bleu)
            logger.info("  " + "*" * 20)
            self.best_bleu = dev_bleu
            # Save best checkpoint for best bleu
            output_dir = os.path.join(self.output_dir, "checkpoint-best-bleu")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                self.model.module if hasattr(self.model, "module") else self.model
            )  # Only save the model it-self
            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)

    def train(self):
        # Prepare training data loader
        train_examples = read_examples(self.train_filename)
        train_features = self.convert_examples_to_features(
            train_examples, stage="train"
        )
        all_source_ids = torch.tensor(
            [f.source_ids for f in train_features], dtype=torch.long
        )
        all_source_mask = torch.tensor(
            [f.source_mask for f in train_features], dtype=torch.long
        )
        all_target_ids = torch.tensor(
            [f.target_ids for f in train_features], dtype=torch.long
        )
        all_target_mask = torch.tensor(
            [f.target_mask for f in train_features], dtype=torch.long
        )
        train_data = TensorDataset(
            all_source_ids, all_source_mask, all_target_ids, all_target_mask
        )

        if self.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=self.train_batch_size // self.gradient_accumulation_steps,
        )

        num_train_optimization_steps = self.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        t_total = (
            len(train_dataloader)
            // self.gradient_accumulation_steps
            * self.num_train_epochs
        )
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total
        )

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", self.train_batch_size)
        logger.info("  Num epoch = %d", self.num_train_epochs)

        self.model.train()
        self.dev_dataset = {}
        (
            nb_tr_examples,
            nb_tr_steps,
            tr_loss,
            self.global_step,
            self.best_bleu,
            self.best_loss,
        ) = (
            0,
            0,
            0,
            0,
            0,
            1e6,
        )
        for epoch in range(self.num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            for batch in bar:
                batch = tuple(t.to(self.device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                loss, _, _ = self.model(
                    source_ids=source_ids,
                    source_mask=source_mask,
                    target_ids=target_ids,
                    target_mask=target_mask,
                )

                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                tr_loss += loss.item()
                self.train_loss = round(
                    tr_loss * self.gradient_accumulation_steps / (nb_tr_steps + 1), 4
                )
                bar.set_description("epoch {} loss {}".format(epoch, self.train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % self.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    self.global_step += 1
            if self.do_eval:
                self.eval()


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type: e.g. roberta",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model: e.g. roberta-base",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--load_model_path",
        default=None,
        type=str,
        help="Path to trained model: Should contain the .bin files",
    )
    ## Other parameters
    parser.add_argument(
        "--train_filename",
        default=None,
        type=str,
        help="The train filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--dev_filename",
        default=None,
        type=str,
        help="The dev filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--test_filename",
        default=None,
        type=str,
        help="The test filename. Should contain the .jsonl files for this task.",
    )

    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_source_length",
        default=64,
        type=int,
        help="The maximum total source sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        default=32,
        type=int,
        help="The maximum total target sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )

    parser.add_argument(
        "--train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--beam_size", default=10, type=int, help="beam size for beam search"
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--eval_steps", default=-1, type=int, help="")
    parser.add_argument("--train_steps", default=-1, type=int, help="")
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    roberta_trainer = Trainer(
        args.model_type,
        args.model_name_or_path,
        args.output_dir,
        args.max_source_length,
        args.max_target_length,
        args.do_train,
        args.do_eval,
        args.do_test,
        args.do_lower_case,
        args.no_cuda,
        load_model_path=args.load_model_path,
        train_filename=args.train_filename,
        dev_filename=args.dev_filename,
        test_filename=args.test_filename,
        config_name=args.config_name,
        tokenizer_name=args.tokenizer_name,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beam_size=args.beam_size,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        train_steps=args.train_steps,
        warmup_steps=args.warmup_steps,
        local_rank=args.local_rank,
        seed=args.seed,
    )

    if roberta_trainer.do_train:
        roberta_trainer.train()
    if roberta_trainer.do_test:
        roberta_trainer.test()


if __name__ == "__main__":
    main()
