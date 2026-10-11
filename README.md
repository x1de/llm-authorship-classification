# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 midterm project by Rishi Raj. The methods and findings are in [report.pdf](report.pdf).

This project compares a TextCNN and a bidirectional LSTM on responses from ChatGPT, Llama, Mistral, and Cohere. The prepared RAID sample has 8,000 rows from 2,000 matched prompts across eight domains. Each prompt belongs to only one of the train, validation, or test splits.

## Files

- `data/processed/raid_v1/` contains the exact train, validation, and test CSVs used for the report, plus a preparation manifest and the RAID license. The CSVs include `LLM_name`, `LLM_Input`, and `LLM_output`.
- `prepare_data.py` recreates that sample from the original RAID source CSV.
- `train.py` trains the CNN and LSTM for prompt only, response only, and prompt plus response.
- `cross_domain.py` retrains response-only models with news or poetry left out.
- `analyze.py` runs the simple word and style comparisons.
- `results.json` records the measured results from the original runs. `report.pdf` discusses them.

## Reproduce the experiments

From the repository root, use Python 3.12 in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, 'CUDA available:', torch.cuda.is_available())"
```

If this prints `CUDA available: False` on a supported NVIDIA laptop, follow [PyTorch's installation selector](https://pytorch.org/get-started/locally/) to install a CUDA-enabled build. Otherwise, the scripts run on the CPU.

The prepared data is already included, so you can start training without downloading RAID. Run all six required CNN/LSTM settings, then the optional cross-domain and style analyses:

```powershell
.\.venv\Scripts\python.exe train.py --data-dir data/processed/raid_v1 --output-dir runs/local-main
.\.venv\Scripts\python.exe cross_domain.py --data-dir data/processed/raid_v1 --original-runs runs/local-main --output-dir runs/local-cross-domain
.\.venv\Scripts\python.exe analyze.py --data-dir data/processed/raid_v1 --output runs/local-baselines.json
```

`train.py` prints whether it is using `cuda` or `cpu`. It creates `runs/local-main/` with the vocabulary, configuration, per-experiment metrics, and best checkpoints. The cross-domain script reads those checkpoints and creates `runs/local-cross-domain/`. The analysis script writes its results to `runs/local-baselines.json`. None of these paths needs to exist before running the scripts. They are ignored by Git because each person generates their own runs. Use a new output path for another attempt; the scripts refuse to overwrite existing experiment results. Scores may vary slightly across hardware and software setups.

To try just one model and text setting first:

```powershell
.\.venv\Scripts\python.exe train.py --architecture cnn --mode output --data-dir data/processed/raid_v1 --output-dir runs/try-cnn-output
```

This shorter run is useful for learning how the pipeline works. The full six-setting run above is needed before `cross_domain.py`, because it loads the original output-only CNN and LSTM checkpoints.

If the checkpoints already exist locally, score the test split without training again:

```powershell
.\.venv\Scripts\python.exe train.py --evaluate-only --architecture cnn --mode output --data-dir data/processed/raid_v1 --output-dir runs/raid_v1
```

Change `cnn` to `lstm`, or `output` to `input` or `both`, to inspect another saved model. Omit both selection flags to score all six. The command prints the test metrics and confusion counts; it does not modify the run files. The original recorded scores are also available immediately in each run's `metrics.json`. A fresh clone must train first because `runs/` and its checkpoints are not published.

The CNN and LSTM share one simple tokenizer: it keeps word-like sequences, punctuation marks, and line breaks as separate tokens. It is not any source LLM's tokenizer. The vocabulary is built from the training split only, using both prompt and response text, and reused for every setting and split. The neural models keep at most 96 prompt tokens and 512 response tokens per row. The report's median token counts use the same tokenizer on full training responses, before truncation. The optional TF-IDF comparison uses scikit-learn's separate word and two-word-phrase features.

## Recreate the prepared data

The processed split is derived from [RAID](https://huggingface.co/datasets/liamdugan/raid), whose dataset card lists an MIT license. To inspect or recreate the sampling, download the original `train_none.csv` from [RAID's official download instructions](https://github.com/liamdugan/raid#download-raid) into `data/raw/`.

```powershell
.\.venv\Scripts\python.exe prepare_data.py --input data/raw/train_none.csv --output-dir data/processed/rebuilt
```

The original source CSV is about 802 MB and stays out of Git. The prepared CSVs are about 13 MB in total. The preparation manifest records the filters, sample size, seed, and split counts. Use a fresh output directory because `prepare_data.py` will not replace an existing split.
