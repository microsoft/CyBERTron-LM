# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import torch.nn.functional as F
import torch


from fairseq.data.data_utils import collate_tokens
from transformers import AutoTokenizer
from Transcormer.transcormer import TranscormerModel

parser = argparse.ArgumentParser(description="Running script for inference")

parser.add_argument("--input", type=str,
                    help="The path of input file for scoring sentences")
parser.add_argument("--model-dir", type=str,
                    help="The path of pre-trained model")
parser.add_argument("--data-dir", type=str, default=".",
                    help="The folder include dictionary file")
parser.add_argument("--batch-size", type=int, default=30,
                    help="The batch size")
parser.add_argument("--tokenizer", type=str, default="bert-base-uncased", 
                    choices=["bert-base-uncased", "bert-base-cased", "gpt2"],
                    help="The type of different tokenizer from transformers")

args = parser.parse_args()


def create_model():
    """ Create model """
    model = TranscormerModel.from_pretrained(
        os.path.dirname(args.model_dir),
        checkpoint_file=os.path.basename(args.model_dir),
        data_name_or_path=args.data_dir,
    )
    model = model.model.cuda()
    model.eval()
    return model


def create_tokenize_fn(model):
    """ Create tokenizer for input corpus """
    dictionary = model.encoder.dictionary
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def tokenize(line):
        ids = None
        if args.tokenizer == "gpt2":
            ids = list(map(str, tokenizer.encode(line)))
        else:
            ids = tokenizer.tokenize(line)
        t = " ".join(ids)
        return dictionary.encode_line(t, add_if_not_exist=False)

    return tokenize


def batchify_samples(samples, tokenize=None):
    """ Batchify samples """
    batch = []
    for i in samples:
        batch.append(tokenize(i))
        if len(batch) == args.batch_size:
            yield batch
            batch = []
    yield batch


def calculate_sentence_score(logits, targets, padding_idx=1):
    """ Calculate the sentence score based on the predicted logits """
    lprobs = F.log_softmax(logits, dim=-1)
    sz = lprobs.size(-1)

    token_nll_loss = F.nll_loss(
        lprobs.view(-1, sz),
        targets.view(-1),
        ignore_index=padding_idx,
        reduction="none",
    )

    non_padding_mask = targets.ne(padding_idx)
    token_loss = token_nll_loss.view(lprobs.size(0), lprobs.size(1)) * non_padding_mask.float()
    sent_loss = token_loss.sum(dim=1) / non_padding_mask.sum(dim=1)
    return sent_loss


def inference(
    samples,
    model=None,
    tokenize=None,
    padding_idx=1,
    **unused,
):
    """ Get sentence score from input samples """
    batches = batchify_samples(samples, tokenize=tokenize)
    sent_scores = []

    with torch.no_grad():
        for batch in batches:
            batch = collate_tokens(batch, pad_idx=padding_idx)
            batch = torch.cat((torch.zeros(batch.size(0), 1).type_as(batch), batch), dim=1)
            batch = batch.cuda()
            targets = batch.long()

            logits, _ = model.compute(batch)
            scores = calculate_sentence_score(logits, targets)
            sent_scores.extend(scores.detach().cpu().numpy())

    return sent_scores


def main():
    model = create_model()
    tokenize_fn = create_tokenize_fn(model)

    samples = []
    with open(args.input, "r", encoding="utf-8") as input_file:
        for line in input_file:
            samples.append(line.strip())

    scores = inference(samples, model=model, tokenize=tokenize_fn)

    for s in scores:
        print(s)


if __name__ == '__main__':
    main()
    