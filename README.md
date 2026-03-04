# DDI-Expert
This repository contains the data and the source codes used to develop DDI-Expert, an explainable foundation model-based framework for the prediction and explanation of pharmacokinetic drug interactions in humans.

Cleaned codebase for the DDI-Expert: a BioClinical ModernBERT plus T5 framework with mixture-of-experts components for pharmacokinetic drug-drug interaction classification and mechanistic explanation generation, with optional LoRA-based continual learning on the decoder.


## Repository layout

- `src/ddi_expert/data.py`: dataset loading, cleaning, prompt construction, train/validation/test splitting
- `src/ddi_expert/prompts.py`: prompt templates for classification, explanation generation, and regression
- `src/ddi_expert/moe.py`: noisy top-k router, experts, sparse MoE layer, and auxiliary load-balancing loss
- `src/ddi_expert/models.py`: sparse MoE classifier and ModernBERT-MoE-to-T5 generator
- `src/ddi_expert/training.py`: classifier loop and Hugging Face `Trainer` builder for generation
- `src/ddi_expert/lora.py`: LoRA attachment helper for continual learning on the decoder
- `scripts/train_classifier.py`: CLI for classification training
- `scripts/train_generator.py`: CLI for generation training and optional LoRA fine-tuning
