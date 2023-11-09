# Data

## TüBa-D/Z

* `tubadz-pp` is the original PP attachment disambiguation dataset from de Kok et al., (2017).
  The train and test sets are from the authors, while the dev set is extracted from the remaining, unused data.
* `tubadz-pp-aux` is the dataset above with additional auxiliary distributions.

## SPMRL
* `spmrl-marmot-biaffine`:
  * The original data come from the German dataset of the SPMRL Shared Task 2014.
  * The predicted POS tags are assigned by MarMoT with 10-way jackknifing.
  * The predicted dependency trees are assigned by our reimplementation of the neural parser
    with biaffine attention (Dozat and Manning, 2017).
  * Gold POS tags and gold dependency trees are included.
* `spmrl-pp-gold-obj-dekok` and `spmrl-pp-pred-obj-dekok` are the PP attachment disambiguation datasets
  in the format defined in de Kok et al. (2017).
  * `-gold-obj` contains prepositional objects extracted from the *gold* dependency trees,
  while `-pred-obj` contains prepositional objects extracted from the *predicted* dependency trees.
  * The POS tags are automatic ones.
  * Inside each dataset, the files end with `.aux` contains auxiliary distributions.
* `spmrl-pp-gold-obj-jsonl` and `spmrl-pp-pred-obj-jsonl` are similar to the datasets above but in JSONL format.
