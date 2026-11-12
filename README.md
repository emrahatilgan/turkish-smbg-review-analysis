# Turkish sentiment analysis of SMBG reviews

## Installation

Install Python and the required packages:

```bash
python -m pip install -r requirements.txt
```

The supplied `data/train.csv`, `data/test.csv` and `results/split.csv` contain the model-development data and the train–validation split. The source dataset is [E-commerce comments](https://huggingface.co/datasets/AlperenEvci/e-commerce_comments), revision `b79f242406e89b63d3cdb39db5fc4e99442a6932`.

## Training and evaluation

Run the commands from this directory:

```bash
python run_bert.py
python run_baselines.py --model CNN
python run_baselines.py --model RNN
python run_lstm.py
```

BERT downloads the pretrained Turkish model on first use and saves its selected checkpoint in `bert_results/checkpoint`. Model training can take considerable time, particularly on a CPU. A short setup check is available by adding `--smoke` to any training command; it does not reproduce the reported performance results.

## SMBG review prediction

After BERT training completes, predict sentiment for the supplied 180 reviews:

```bash
python predict_smbg_180.py
```

This command requires the fine-tuned checkpoint in `bert_results/checkpoint`. The checkpoint is not included. Predictions are saved in `smbg_180_results/`.

## Rater agreement and screening

```bash
python analyze_smbg.py
```

The analysis uses the saved manuscript predictions and independent assessments in `study_data/` and writes results to `output/SMBG_Final_Analysis/`. `D1_Tema.csv` and `D2_Tema.csv` contain the category assessments. The script calculates sentiment and binary safety agreement, model–rater performance, and threshold sensitivity. Category-specific agreement, the all-positive baseline, and confidence intervals require separate calculation.
