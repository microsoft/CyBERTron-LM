# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import torch

import numpy as np
from fairseq import utils
from fairseq.data import (
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import LegacyFairseqTask, register_task


logger = logging.getLogger(__name__)


@register_task("score_lm")
class ScoreLMTask(LegacyFairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument(
            "--sample-break-mode",
            default="complete",
            choices=["none", "complete", "complete_doc", "eos"],
            help='If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.',
        )
        parser.add_argument(
            "--tokens-per-sample",
            default=512,
            type=int,
            help="max number of total tokens over all segments "
            "per sample for BERT dataset",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.1,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.1,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--freq-weighted-replacement",
            default=False,
            action="store_true",
            help="sample random replacement words based on word frequencies",
        )
        parser.add_argument(
            "--mask-whole-words",
            default=False,
            action="store_true",
            help="mask whole words; you may also want to set --bpe",
        )
        parser.add_argument(
            "--shorten-method",
            default="none",
            choices=["none", "truncate", "random_crop"],
            help="if not none, shorten sequences that exceed --tokens-per-sample",
        )
        parser.add_argument(
            "--shorten-data-split-list",
            default="",
            help="comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)',
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("<mask>")

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample,
            self.args.seed,
        )

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample - 1,  # one less for <s>
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.args.sample_break_mode,
        )
        logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        src_dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(src_dataset))

        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    "id": IdDataset(),
                    "net_input": {
                        "src_tokens": RightPadDataset(
                            src_dataset,
                            pad_idx=self.source_dictionary.pad(),
                        ),
                        "src_lengths": NumelDataset(src_dataset, reduce=False),
                    },
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(src_dataset, reduce=True),
                },
                sizes=[src_dataset.sizes],
            ),
            sort_order=[
                shuffle,
                src_dataset.sizes,
            ],
        )

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary