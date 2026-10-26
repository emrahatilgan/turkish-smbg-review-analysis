# Turkish sentiment analysis of SMBG e-commerce reviews

Code and data associated with an exploratory study of Turkish-language e-commerce reviews of self-monitoring of blood glucose devices and consumables.

## Data

- `smbg_180_reviews.csv`: 180 original Turkish reviews used as model input.
- `study_data/smbg_180_predictions.csv`: review texts, BERT predictions and class probabilities.
- `study_data/d1(5).xlsx`, `study_data/d2(4).xlsx`: final independent sentiment and safety assessments.
- `study_data/D1_Tema.csv`, `study_data/D2_Tema.csv`: independent category assessments.
- `study_data/metrics.json`, `study_data/test_predictions.csv`: BERT general-test results and predictions.

The dataset includes public review text and evaluator notes. Review applicable third-party platform conditions, content and privacy considerations before public release. User-reported concerns are not verified device malfunctions or medical events.

The model-development dataset was obtained from [E-commerce comments](https://huggingface.co/datasets/AlperenEvci/e-commerce_comments). Supplied experiment documentation identifies dataset revision `b79f242406e89b63d3cdb39db5fc4e99442a6932`. The pretrained model is `dbmdz/bert-base-turkish-uncased`.

## Code

- `run_bert.py`: BERT training and resumption.
- `run_baselines.py`: CNN/RNN training; select with `--model CNN` or `--model RNN`.
- `run_lstm.py`: LSTM training.
- `predict_smbg_180.py`: prediction using a locally saved fine-tuned BERT checkpoint.
- `analyze_smbg.py`: an earlier human-assessment and threshold analysis script.
- `check_files.py`, `benchmark_bert.py`: input and runtime checks.

Install the dependencies in `requirements.txt`. The spreadsheet analysis additionally requires `openpyxl`. Training expects `data/train.csv` and `data/test.csv`; BERT also expects `results/split.csv`. Those files are not included. SMBG prediction requires the locally saved fine-tuned checkpoint, which is not included. Model outputs depend on software and hardware configuration.

**Analysis version:** `analyze_smbg.py` references earlier annotation files `d1(4).xlsx` and `d2(3).xlsx`. The final annotation files provided here are `d1(5).xlsx` and `d2(4).xlsx`. The supplied script must not be described as the exact final manuscript analysis without locating and checking the final procedure. Some manuscript analyses, including category agreement, majority-class baseline and confidence intervals, are not provided here as a verified complete pipeline.

This package preserves the supplied code logic; descriptive code comments and module headers were removed. It does not establish exact independent reproduction of the reported results.
