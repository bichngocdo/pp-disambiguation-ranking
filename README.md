# PP Attachment Disambiguation with Ranking Loss

This repository contains code for the PP attachment disambiguation systems
(`PP-REP` and `PP-REP-AUX`) from the paper
[Parsers Know Best: German PP Attachment Revisited](https://aclanthology.org/2020.coling-main.185/)
published at COLING 2020.

## Usage

### Requirements
* Python 3.6
* Install dependencies in [requirements.txt](requirements.txt)

### Packages
* [german_baseline](pp_disamb%2Fgerman_baseline): used in experiments _without_ auxiliary distribution
* [german_baseline_scores](pp_disamb%2Fgerman_baseline_scores): used in experiments _with_ auxiliary distribution scores

### Training
Run:
```shell
PYTHONPATH=`pwd` python pp_disamb/<package>/experiment.py train --help
```
to see possible arguments.

For example, to train a model (without auxiliary distribution scores) on the sample dataset, run:

```shell
PYTHONPATH=`pwd` python pp_disamb/german_baseline/experiment.py train \
  --train_file data/sample/train.txt \
  --dev_file data/sample/dev.txt \
  --test_file data/sample/test.txt \
  --word_emb_file data/sample/word_embs.txt \
  --tag_emb_file data/sample/tag_embs.txt \
  --model_dir runs/sample \
  --max_epoch 1 \
  --interval 1 \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --word_dim 5 \
  --tag_dim 4 \
  --hidden_dim 7 \
  --learning_rate 0.01
```

### Evaluation
Run:
```shell
PYTHONPATH=`pwd` python pp_disamb/<package>/experiment.py eval --help
```
to see possible arguments.

For example, to evaluate a trained model (without auxiliary distribution scores) on the sample dataset, run:
```shell
PYTHONPATH=`pwd` python pp_disamb/german_baseline/experiment.py eval \
  --test_file data/sample/test.txt \
  --output_file runs/sample/result.txt \
  --model_dir runs/sample
```

### Tensorboard

```shell
tensorboard --logdir runs/sample/log
```

### Scripts

* [reattach.py](scripts%2Freattach.py): reattach PP attachment disambiguation result
  of the system to the parser's output
* [pp_eval_dekok.py](scripts%2Fpp_eval_dekok.py): evaluate PP attachment disambiguation results
  with the format defined by de Kok et al. (2017)
* [pp_eval_conll09.py](scripts%2Fpp_eval_conll09.py): evaluate PP attachment disambiguation results
  with CoNLL format


## Reproduction

All trained models contain:
* File `config.cfg` that records all parameters used to produce the model.
* Folder `log` records training and evaluation metrics, which can be viewed by `tensorboard`.
* The result can be evaluated using the script [pp_eval_dekok.py](scripts%2Fpp_eval_dekok.py).
* Or, reattach the disambiguation result to the predicted parse trees and evaluate the reattachment using the script
  [pp_eval_conll09.py](scripts%2Fpp_eval_conll09.py).
* See more information at [data](data) and [models](models).


## Citation

```bib
@inproceedings{do-rehbein-2020-parsers,
    title = "Parsers Know Best: {G}erman {PP} Attachment Revisited",
    author = "Do, Bich-Ngoc and Rehbein, Ines",
    editor = "Scott, Donia and Bel, Nuria and Zong, Chengqing",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2020.coling-main.185",
    doi = "10.18653/v1/2020.coling-main.185",
    pages = "2049--2061",
}
```