# CIPPROJECT

Disaster-related tweet classification pipeline with a fine-tuned transformer model and a simple explainability routine. The project includes raw disaster datasets, preprocessing scripts, a trained model, and a CLI loop to classify text with token-level influence scores.

## Project Layout

- `Data_Set/`
- `Data_Set/Unprocessed_Data_Sets/` raw disaster tweet datasets (`.tsv`)
- `Data_Set/Processed_Data_Set/` preprocessed CSVs
- `Data_Set/Data_Preprocessing/` data prep scripts
- `Data_Set/Data_Preprocessing/Data_Extraction.py` build labeled dataset from TSVs
- `Data_Set/Data_Preprocessing/Data_Cleaning.py` clean text and filter short samples
- `Model_Build/`
- `Model_Build/Build.py` training, evaluation, and tokenizer saving
- `Trained_Model/`
- `Trained_Model/Main_Program.py` interactive classifier + explainability
- `Trained_Model/Explainable_AI.py` token influence computation
- `Trained_Model/Model.py` model architecture definition
- `Trained_Model/deltran15_minilm_fp32.pt` trained weights
- `Trained_Model/Model_Tokenizer/` tokenizer files

## Labels

The classifier predicts one of five categories:

- `affected_individuals`
- `infrastructure_and_utility_damage`
- `not_humanitarian`
- `other_relevant_information`
- `rescue_volunteering_or_donation`

## Data Preparation

1. **Extract raw TSVs into a labeled CSV**
   - Script: `Data_Set/Data_Preprocessing/Data_Extraction.py`
   - Run from `Data_Set/Data_Preprocessing/` so it can read `../Unprocessed_Data_Sets/*.tsv`.
   - Maps multiple original labels into 5 merged classes.
   - Writes `Data_Set/Processed_Data_Set/disaster_text_only.csv`.

2. **Clean and filter tweets**
   - Script: `Data_Set/Data_Preprocessing/Data_Cleaning.py`
   - Run from `Data_Set/Data_Preprocessing/` so it can read `../Processed_Data_Set/disaster_text_only.csv`.
   - Removes RT prefixes, @mentions, URLs, non-ASCII characters, and extra whitespace.
   - Filters out samples with length <= 5.
   - Writes `Processed_Data_Set/disaster_text_clean.csv`.

## Model

- Base encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Custom head: linear classifier on `[CLS]` embedding with dropout
- Weights: `Trained_Model/deltran15_minilm_fp32.pt`

## Training

The training/evaluation workflow is in `Model_Build/Build.py`. It:

- Loads `Data_Set/Processed_Data_Set/disaster_text_clean.csv`
- Splits into train/val/test
- Trains the classifier with early stopping
- Prints classification metrics
- Saves the best model weights and tokenizer artifacts

## Explainability

`Trained_Model/Explainable_AI.py` runs a forward pass to compute logits, then produces a token-level importance signal using the norm of token embeddings. `Main_Program.py` prints the top tokens (excluding special tokens) for the predicted class.

## Actionable Info Extraction

Actionable information is extracted after classification using rules + NER:

- Locations via spaCy NER when available (`GPE`, `LOC`, `FAC`)
- People counts via regex (e.g., `3 injured`, `2 missing`)
- Needs and damage types via keyword lists
- Time mentions via simple patterns (e.g., `today`, `2 hours ago`)

Implementation lives in `Trained_Model/Actionable_Info.py` and is called from `Trained_Model/Main.py` for actionable labels.

To run the extraction examples only:

```powershell
python Run_Actionable_Examples.py
```

## Running the Classifier

From the project root:

```powershell
python Trained_Model\Main_Program.py
```

You will be prompted to enter text. The script will print:

- Predicted label id and class name
- Softmax confidence scores
- Top influential tokens for the prediction

Type `exit` to quit.

## Dependencies

Typical runtime requirements:

- `torch`
- `transformers`
- `pandas`
- `spacy` (optional, improves location extraction)

## Notes

- The repository includes large model weights in `Trained_Model/deltran15_minilm_fp32.pt`.
