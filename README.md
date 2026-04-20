# Typology-Aware Multilingual Morphosyntactic Parsing with Joint Abstract Node Modeling

This repository contains an implementation of a joint multilingual morphosyntactic parsing (MSP) model with typology-aware conditioning and abstract node modeling.

## Overview

The system is designed for the UniDive Morphosyntactic Parsing (MSP) task. In this formulation, dependency structure is defined over content words, while function words are encoded through morphological features and do not participate as tree nodes. In addition, unrealized arguments (e.g., dropped subjects, implicit pronouns) are represented as abstract nodes projected onto content tokens.

The model jointly predicts:

- Word type (content vs. function)
- Content-only dependency structure
- Morphological features
- Abstract node attributes (presence, position, dependency relation, morphological features)

## Implementation Basis

This implementation is based on [this](https://github.com/DemianInostrozaAmestica/shared_task_UD_official) shared task repository. The framework is extended with typology-aware conditioning, contextual adapter mechanisms, and integrated abstract node prediction within a unified joint architecture.

## Reference Studies

- [Joint multitask MSP baseline](https://aclanthology.org/2025.unidive-1.2/)
- [Typology-aware multilingual MSP](https://aclanthology.org/2025.unidive-1.3/)
- [UDapter: Typology-based multilingual parsing](https://aclanthology.org/2020.emnlp-main.180/)

## Environment

- Python 3.11

### Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset

The official dataset and evaluation script is provided by the [UniDive MSP Shared Task](https://github.com/UniDive-MSP/MSP-shared-task):

You will need:

- train.tsv
- dev.tsv

These manifest files define multilingual training and development splits.

## Training the Model

All runs are executed in module mode.

### Full configuration (typology + adapters)

```bash
python -m joint_msp_model.training.joint_trainer   --train_manifest train.tsv   --dev_manifest dev.tsv   --output runs/full_model   --uriel_dim 64
```

### Typology only (no adapters)

```bash
python -m joint_msp_model.training.joint_trainer   --train_manifest train.tsv   --dev_manifest dev.tsv   --output runs/no_adapters   --uriel_dim 64   --no_adapters
```

### Baseline (no typology, no adapters)

```bash
python -m joint_msp_model.training.joint_trainer   --train_manifest train.tsv   --dev_manifest dev.tsv   --output runs/no_typology   --no_typology   --no_adapters
```

## Inference / Prediction

Prediction is performed with a separate script that loads a trained checkpoint, reads a CoNLL-U file, attaches typology information when the model expects URIEL features, and writes MSP-style CoNLL-U output with reconstructed abstract nodes.

### Required arguments

- `--raw_file`: input CoNLL-U file
- `--train_conllu`: training CoNLL-U file used for dictionaries and logging
- `--joint_model`: path to the saved checkpoint
- `--output`: output CoNLL-U path
- `--lang`: ISO language code such as `tr` or `cs`

### Example: Turkish prediction

```bash
python -m joint_msp_model.inference.predict \
  --raw_file data/tr/test.conllu \
  --train_conllu data/tr/train.conllu \
  --joint_model runs/full_model/best-model.pt \
  --output predictions/tr_test.pred.conllu \
  --lang tr
```

### Example: Czech prediction

```bash
python -m joint_msp_model.inference.predict \
  --raw_file data/cs/test.conllu \
  --train_conllu data/cs/train.conllu \
  --joint_model runs/full_model/best-model.pt \
  --output predictions/cs_test.pred.conllu \
  --lang cs
```

### Example: custom batch size and morphology threshold

```bash
python -m joint_msp_model.inference.predict \
  --raw_file data/tr/dev.conllu \
  --train_conllu data/tr/train.conllu \
  --joint_model runs/full_model/best-model.pt \
  --output predictions/tr_dev.pred.conllu \
  --lang tr \
  --batch_size 16 \
  --threshold 0.4
```


## Configurations

The following configurations are supported:

- Full model with typology-aware adapters
- Typology without adapters
- No typology and no adapters
- Multilingual training via manifest files
