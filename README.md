# BERT with SentencePiece for Japanese text.
This is a repository of Japanese BERT model with SentencePiece tokenizer  
forked from https://github.com/yoheikikuta/bert-japanese .

## Current changes
- appended optimization.py and modeling.py from original bert repo.
- ld_corpus.py: a script to download data and to test finetuned model.
- dbdc_corpus.py: to learn dialog breakdown detectors wtih [dbdc](https://sites.google.com/site/dialoguebreakdowndetection/) and [dbdc2](https://sites.google.com/site/dialoguebreakdowndetection2/) corpus.

## Usage ld_corpus.py and run_dbdc_classifier.py
```bash
# Suppose that the current directory is repository root.
# collecting and editing datasets
python src/dbdc_cropus.py -m fetch
# laearning with the dataset
python src/run_dbdc_classifier.py \
    --task_name=DBDC \
    --do_train=true \
    --do_eval=true \
    --data_dir=data/dbdc \
    --model_file=model/wiki-ja-mod.model \
    --vocab_file=model/wiki-ja-mod.vocab \
    --init_checkpoint=model/model.ckpt-1400000 \
    --max_seq_length=512 \
    --train_batch_size=4 \
    --learning_rate=2e-5 \
    --num_train_epochs=10 \
    --output_dir=model/dbdc_1
# evaluation
python src/dbdc_corpus.py -m test -d data/dbdc -p model/dbdc_1
```

Evaluation with models successfully created will result like this:
```
***** RESULT *****
Test dataset:
data/dbdc/test.csv
Tested model:
model/dbdc_1/model.ckpt-5157
***** IRS *****
Accuracy:
0.5927272727272728
Detailed report:
              precision    recall  f1-score   support

           O       0.73      0.60      0.66       216
           T       0.30      0.38      0.33       103
           X       0.65      0.68      0.66       231

   micro avg       0.59      0.59      0.59       550
   macro avg       0.56      0.55      0.55       550
weighted avg       0.62      0.59      0.60       550

Confusion matrix:
[[130  41  45]
 [ 24  39  40]
 [ 24  50 157]]

Precision (X) :         0.648760 (157/242)
Recall        (X) :     0.679654 (157/231)
F-measure (X) :         0.663848
Precision (T+X) :       0.801075 (298/372)
Recall        (T+X) :   0.834734 (298/357)
F-measure (T+X) :       0.817558
JS divergence (O,T,X) :         0.091804
JS divergence (O,T+X) :         0.059638
JS divergence (O+T,X) :         0.060257
Mean squared error (O,T,X) :    0.050887
Mean squared error (O,T+X) :    0.061148
Mean squared error (O+T,X) :    0.064236

***** DIT *****
Accuracy:
0.6418181818181818
Detailed report:
              precision    recall  f1-score   support

           O       0.76      0.62      0.69       184
           T       0.28      0.34      0.31       102
           X       0.74      0.77      0.75       264

   micro avg       0.64      0.64      0.64       550
   macro avg       0.59      0.58      0.58       550
weighted avg       0.66      0.64      0.65       550

Confusion matrix:
[[115  44  25]
 [ 20  35  47]
 [ 16  45 203]]

Precision (X) :         0.738182 (203/275)
Recall        (X) :     0.768939 (203/264)
F-measure (X) :         0.753247
Precision (T+X) :       0.907268 (362/399)
Recall        (T+X) :   0.878641 (362/412)
F-measure (T+X) :       0.892725
JS divergence (O,T,X) :         0.043422
JS divergence (O,T+X) :         0.025578
JS divergence (O+T,X) :         0.027862
Mean squared error (O,T,X) :    0.024256
Mean squared error (O,T+X) :    0.025019
Mean squared error (O+T,X) :    0.032207

***** DCM *****
Accuracy:
0.5363636363636364
Detailed report:
              precision    recall  f1-score   support

           O       0.66      0.71      0.69       223
           T       0.35      0.50      0.41       149
           X       0.63      0.34      0.44       178

   micro avg       0.54      0.54      0.54       550
   macro avg       0.55      0.52      0.51       550
weighted avg       0.57      0.54      0.53       550

Confusion matrix:
[[159  53  11]
 [ 49  75  25]
 [ 32  85  61]]

Precision (X) :         0.628866 (61/97)
Recall        (X) :     0.342697 (61/178)
F-measure (X) :         0.443636
Precision (T+X) :       0.841935 (261/310)
Recall        (T+X) :   0.727019 (261/359)
F-measure (T+X) :       0.780269
JS divergence (O,T,X) :         0.085814
JS divergence (O,T+X) :         0.054299
JS divergence (O+T,X) :         0.053763
Mean squared error (O,T,X) :    0.047948
Mean squared error (O,T+X) :    0.056797
Mean squared error (O+T,X) :    0.056800
```

## Usage ld_corpus.py
Suppose that current directory is repository root.

```:bash
#
# Download
python src/ld_corpus.py -m fetch
# According to TRAIN_PROPS in config.ini in this version
# train data with proportion 0.05, 0.10, 0.20, 0.50, 1.00 
# (to the rest samples of vaild and train) will be created in data/livedoor
ls data/livedoor/
# prop_0p05  prop_0p10  prop_0p20  prop_0p50  prop_1p00  text
# Note: test.tsv, dev.tsv in prop_x are the same each other
#
# Train
python src/run_classifier.py \
  --task_name=livedoor \
  --do_train=true \
  --do_eval=true \
  --data_dir=data/livedoor/prop_1p00 \
  --model_file=model/wiki-ja.model \
  --vocab_file=model/wiki-ja.vocab \
  --init_checkpoint=model/model.ckpt-1400000 \
  --max_seq_length=512 \
  --train_batch_size=4 \
  --learning_rate=2e-5 \
  --num_train_epochs=10 \
  --output_dir=model/livedoor_model_normal_1p00_1
#
# Test
python src/ld_corpus.py -m test \
  -p model/livedoor_model_normal_1p00_1 \ 
  -d data/livedoor/prop_1p00
# (output example)
# ***** RESULT *****
# Test dataset:
# data/livedoor/prop_1p00/test.tsv
# Tested model:
# model/livedoor_model_normal_1p00_1/model.ckpt-11052
# Accuracy:
# 0.9572301425661914
# Detailed report:
#                 precision    recall  f1-score   support
# 
# dokujo-tsushin       0.96      0.91      0.93       178
# ...
# 
# Confusion matrix:
# [[162   1   0   5   1   9   0   0   0]
# ...
```

## Pretrained models
We provide pretrained BERT model and trained SentencePiece model for Japanese text.
Training data is the Japanese wikipedia corpus from [`Wikimedia Downloads`](https://dumps.wikimedia.org/).  
Please download all objects in the following google drive to `model/` directory.
- **[`Pretrained BERT model and trained SentencePiece model`](https://drive.google.com/drive/folders/1Zsm9DD40lrUVu6iAnIuTH2ODIkh-WM-O?usp=sharing)** 

Loss function during training is as below (after 1M steps the loss function massively changes because `max_seq_length` is changed from `128` to `512`.):
![pretraining-loss](pretraining-loss.png)

```
***** Eval results *****
  global_step = 1400000
  loss = 1.3773012
  masked_lm_accuracy = 0.6810424
  masked_lm_loss = 1.4216621
  next_sentence_accuracy = 0.985
  next_sentence_loss = 0.059553143
```

## Finetuning with BERT Japanese
We also provide a simple Japanese text classification problem with [`livedoor ニュースコーパス`](https://www.rondhuit.com/download.html).  
Try the following notebook to check the usability of finetuning.  
You can run the notebook on CPU (too slow) or GPU/TPU environments.
- **[finetune-to-livedoor-corpus.ipynb](https://github.com/yoheikikuta/bert-japanese/blob/master/notebook/finetune-to-livedoor-corpus.ipynb)**

The results are the following:
- BERT with SentencePiece
  ```
                  precision    recall  f1-score   support

  dokujo-tsushin       0.98      0.94      0.96       178
    it-life-hack       0.96      0.97      0.96       172
   kaden-channel       0.99      0.98      0.99       176
  livedoor-homme       0.98      0.88      0.93        95
     movie-enter       0.96      0.99      0.98       158
          peachy       0.94      0.98      0.96       174
            smax       0.98      0.99      0.99       167
    sports-watch       0.98      1.00      0.99       190
      topic-news       0.99      0.98      0.98       163

       micro avg       0.97      0.97      0.97      1473
       macro avg       0.97      0.97      0.97      1473
    weighted avg       0.97      0.97      0.97      1473
  ```
- sklearn GradientBoostingClassifier with MeCab
  ```
                    precision    recall  f1-score   support

  dokujo-tsushin       0.89      0.86      0.88       178
    it-life-hack       0.91      0.90      0.91       172
   kaden-channel       0.90      0.94      0.92       176
  livedoor-homme       0.79      0.74      0.76        95
     movie-enter       0.93      0.96      0.95       158
          peachy       0.87      0.92      0.89       174
            smax       0.99      1.00      1.00       167
    sports-watch       0.93      0.98      0.96       190
      topic-news       0.96      0.86      0.91       163

       micro avg       0.92      0.92      0.92      1473
       macro avg       0.91      0.91      0.91      1473
    weighted avg       0.92      0.92      0.91      1473
  ```



## Pretraining from scratch
All scripts for pretraining from scratch are provided.
Follow the instructions below.

### Environment set up
Build a docker image with Dockerfile and create a docker container.

### Data preparation
Data downloading and preprocessing.
It takes about one hour on GCP n1-standard-8 (8CPUs, 30GB memories) instance.

```
python3 src/data-download-and-extract.py
bash src/file-preprocessing.sh
```

### Training SentencePiece model
Train a SentencePiece model using the preprocessed data.
It takes about two hours on the instance.

```
python3 src/train-sentencepiece.py
```

### Creating data for pretraining
Create .tfrecord files for pretraining.
For longer sentence data, replace the value of `max_seq_length` with `512`.

```
for DIR in $( find /work/data/wiki/ -mindepth 1 -type d ); do 
  python3 src/create_pretraining_data.py \
    --input_file=${DIR}/all.txt \
    --output_file=${DIR}/all-maxseq128.tfrecord \
    --model_file=./model/wiki-ja.model \
    --vocab_file=./model/wiki-ja.vocab \
    --do_lower_case=True \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=5
done
```

### Pretraining
You need GPU/TPU environment to pretrain a BERT model.  
The following notebook provides the link to Colab notebook where you can run the scripts with TPUs.

- **[pretraining.ipynb](https://github.com/yoheikikuta/bert-japanese/blob/master/notebook/pretraining.ipynb)**


## How to cite this work in papers
We didn't publish any paper about this work.  
Please cite this repository in publications as the following:

```
@misc{bertjapanese,
  author = {Yohei Kikuta},
  title = {BERT Pretrained model Trained On Japanese Wikipedia Articles},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yoheikikuta/bert-japanese}},
}
```
