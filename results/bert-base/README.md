---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- cuad
model-index:
- name: bert-base-4
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bert-base-4

This model is a fine-tuned version of [bert-base-uncased](https://huggingface.co/bert-base-uncased) on the cuad dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 12
- eval_batch_size: 12
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10.0

### Training results



### Framework versions

- Transformers 4.27.4
- Pytorch 2.0.0+cu117
- Datasets 2.11.0
- Tokenizers 0.13.2
