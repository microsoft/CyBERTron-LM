# Transcormer
[Transcormer: Transformer for Sentence Scoring with Sliding Language Modeling](https://arxiv.org/pdf/2205.12986.pdf), by Kaitao Song, Yichong Leng, Xu Tan, Yicheng Zou, Tao Qin, Dongsheng Li, NeurIPS 2022, is a Transformer-based model for sentence scoring in NLP. We introduce a novel sliding language modeling (SLM) to handle sentence scoring tasks, which allows model to predict token-wise probability within a single forward pass and both utilize bidirectional context. When compared with BERT/GPT, it achieves better performances in rescoring tasks, and it also achieves a 20x/160x speedup than BERT in CPU/GPU (since BERT needs N-passes inference).

<p align="center"><img src="./figures/SLM.jpg" width="900"><br/> The overview of Transcormer </p>

## Requirements
```bash
pip install fairseq==0.10.2 torch==1.9.0 transformers
```

## Data
We use the same data pipeline as [RoBERTa](https://github.com/facebookresearch/fairseq/blob/v0.10.2/examples/roberta/README.pretraining.md) to process corpus. For example, we use [WikiText-103 dataset](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) as the training corpus (you can switch it to wikipedia + bookcorpus as the pre-training corpus for reproducing). 

First, download the dataset and tokenize it with the BERT or RoBERTa tokenizer:
```bash
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip

for SPLIT in train valid test; do \
    python encode.py \
        --inputs wikitext-103-raw/wiki.${SPLIT}.raw \
        --outputs wikitext-103-raw/wiki.${SPLIT}.bpe \
        --keep-empty \
        --tokenizer bert-base-uncased \
        --workers 60; \
done
```
You can replace tokenizer as `gpt2` to tokenize corpus as RoBERTa. Next, we need to binarize data, and the command is as:
```bash
fairseq-preprocess \
    --only-source \
    --srcdict dict.bert.txt \
    --trainpref wikitext-103-raw/wiki.train.bpe \
    --validpref wikitext-103-raw/wiki.valid.bpe \
    --testpref wikitext-103-raw/wiki.test.bpe \
    --destdir data-bin/wikitext-103 \
    --workers 60
```
We provide two dictionary files for different tokenizers (`bert` and `roberta`) as:
| Tokenizer | Dictionary |
|---|---|
| BERT    | [dict.bert.txt](https://msramldl.blob.core.windows.net/modelrelease/Transcormer/dict.bert.txt) |
| RoBERTa | [dict.roberta.txt](https://msramldl.blob.core.windows.net/modelrelease/Transcormer/dict.roberta.txt) |

## Training 
Different like BERT/RoBERTa training, `Transcormer` is recommended to train at a sentence level. Therefore, under a 32GB V100, our training script is as follows:
```bash
TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
UPDATE_FREQ=16          # Increase the batch size 16x

DATA_DIR=data-bin/wikitext-103
USER_DIR=Transcormer

fairseq-train --fp16 $DATA_DIR \
    --user-dir $USER_DIR \
    --task sliding_lm --criterion sliding_lm \
    --arch transcormer --sample-break-mode eos --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --update-freq $UPDATE_FREQ --max-tokens 4096 \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 100
```
You can reduce the value of `max-tokens` if the memory of GPU is smaller than 32GB. 

Furthermore, we support our model `Transcormer` to train from a pre-trained BERT model for acceleration. If you have a pre-trained BERT model `bert.pt`, so the training script is as:
```bash
TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0001          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
UPDATE_FREQ=16          # Increase the batch size 16x
RESTORE_MODEL=bert.pt   # Pre-trained BERT model

DATA_DIR=data-bin/wikitext-103
USER_DIR=Transcormer

fairseq-train --fp16 $DATA_DIR \
    --user-dir $USER_DIR \
    --task sliding_lm --criterion sliding_lm \
    --arch transcormer --sample-break-mode eos --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --update-freq $UPDATE_FREQ --max-tokens 4096 \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 100 \
    --restore-file $RESTORE_MODEL \
    --reset-optimizer --reset-dataloader --reset-meters \
```

## Inference
We provide pre-trained [transcormer model](https://msramldl.blob.core.windows.net/modelrelease/Transcormer/transcormer.base.tar.gz) for evaluation. The running script is as:
```bash
wget -c https://msramldl.blob.core.windows.net/modelrelease/Transcormer/transcormer.base.tar.gz
tar -zxvf transcormer.base.tar.gz

TEXT=test.txt # Your input text for language evaluation
DATA=transcormer.base
MODEL=$DATA/transcormer.base.pt

python inference.py --input $TEXT --model $MODEL --data-dir $DATA
```

## Citation
If you find Transcormer useful in your work, you can cite the paper as below:
    
    @inproceedings{Kaitao2022Transcormer,
        Author    = {Kaitao Song, Yichong Leng, Xu Tan, Yicheng Zou, Tao Qin, Dongsheng Li},
        Title     = {Transcormer: Transformer for Sentence Scoring with Sliding Language Modeling},
        Booktitle = {Advances in Neural Information Processing Systems ({NeurIPS})},
        Year      = {2022}
    }
