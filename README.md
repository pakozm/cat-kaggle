# cat-kaggle

Kaggle CAT tube pricing repository

## Participants

- Ram Angadala
- Ankit Gupta
- Francisco Zamora-Martinez

## Dependencies

- [APRIL-ANN toolkit](https://github.com/pakozm/april-ann) for training ANNs.
- [R-convex-ensemble](https://github.com/pakozm/R-convex-ensemble) for
  ensembling using a convex linear combination.
- [Python](https://www.python.org/) and [scikit-learn](http://scikit-learn.org/)

## Final system combinations

### Submission 1

#### Stage 0

- Several models trained with script `scripts/TRAIN/cv_best_result2.py` adapted
  from Ankit training script.
- Several ANN models trained with shell script `execute_best_ram_mlps.sh`, using
  a set of features given by Ram.

#### Stage 1

- Several ANN models trained over the output of previous stage, using the
  shell script `execute_best_stack_mlps.sh`

#### Stage 2

- Convex linear combination using `R-convex-ensemble` over the outputs of
  stage 0 and stage 1. This stage is a combination of 36 predictions.

### Submission 2

Is an averaged combination of previous stage 1 with only one ANN and Ankit
model predictions.
