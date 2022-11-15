# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

import torch

from fairseq import utils, metrics, modules
from fairseq.criterions import FairseqCriterion, register_criterion


def accuracy(output, target):
    """ Calculate accuracy for prediction """
    with torch.no_grad():
        _, pred = output.topk(1, -1)
        correct = pred.view(-1).eq(target.view(-1))
    return correct.sum()


@register_criterion("sliding_lm")
class SlidingLMLoss(FairseqCriterion):
    """
        Implementation for the loss used in sliding language model (SLM) training.
    """

    def __init__(self, task, tpu=False):
        super().__init__(task)
        self.tpu = tpu

    def forward(self, model, sample, reduce=True):
        targets = sample['net_input']['src_tokens']
        sample_size = sample['ntokens']

        logits, _ = model.compute(**sample["net_input"])
        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )

        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "acc": utils.item(accuracy(logits, targets))
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        acc = sum(log.get('acc', 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3,
        )
        metrics.log_scalar(
            "acc", acc / sample_size, sample_size, round=4,
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
