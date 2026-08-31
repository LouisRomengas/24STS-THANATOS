# 24STS-THANATOS
Automated Clavien-Dindo grading of clinical reports, benchmark of classical TF-IDF to Embeddings, zero-shot and LoRA fine-tuned LLMs

## Summary
This repository contains the data-processing, modelling, statistical-analysis, and figure-generation code used for the THANATOS study.
The study benchmarks several natural language processing approaches for automated Clavien-Dindo grading from postoperative clinical reports, including classical text classifiers, dense embeddings, zero-shot large language models, and QLoRA fine-tuned large language models.
This repository contains research analysis code and is not a software package.

## Data availability
The individual-level data used in this study are not included in this repository.

## Repository contents
The repository contains code for:
- dataset preparation and text preprocessing
- TF-IDF and dense-embedding classifiers
- zero-shot LLM inference
- QLoRA fine-tuning and evaluation
- patient-level and report-level performance analyses
- bootstrap confidence intervals and paired model comparisons
- error and subgroup analyses
- computational-cost and CodeCarbon analyses
- generation of manuscript tables and figures

## Software environment
LLM inference and fine-tuning were performed on a single NVIDIA A100 80 GB GPU.

Exact model identifiers and methodological specifications are provided in the manuscript and supplementary material.
