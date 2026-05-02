# ReccomenderSystemsProject

Recommender Systems Final Project (Spring 2026) using Yelp restaurants in Santa Barbara, CA.

Team members: Luke, Troy, Omer

## Project Overview

This project implements and evaluates:

- 3 baseline recommenders
- 3 content-based recommenders
- 1 classification model using review text
- 1 collaborative filtering recommender
- 1 creative extension (hybrid CF + content)
- 2 demo notebooks

All models use the provided train/test split:

- `train_reviews_santa_barbara.csv` for training/modeling
- `test_reviews_santa_barbara.csv` for test-only evaluation
- `restaurants_santa_barbara.csv` for business metadata

## Repository Structure

### Recommender / Model Notebooks

- `random_recommender_baseline.ipynb`  
  Baseline A: random recommender. Reports Hit@K and NDCG@K.

- `popularity-based_recommender_baseline.ipynb`  
  Baseline B: popularity/regularized item mean style recommender. Reports Hit@K and NDCG@K.

- `user_item_bias.ipynb`  
  Baseline C: user-item bias recommender. Reports ranking metrics and rating metrics (RMSE, MAE, R2).

- `Content-Based (metadata only).ipynb`  
  Content-Based A: metadata-only recommender using categories + attribute features. Reports ranking metrics.

- `Content-Based B (metadata and review text).ipynb`  
  Content-Based B: metadata + transformer review embeddings. Reports ranking metrics.

- `Content-Based C (metadata and review sentiment analysis).ipynb`  
  Content-Based C: metadata + review text + sentiment features. Reports ranking metrics.

- `classification_model.ipynb`  
  Classification model: review sentiment prediction from review text embeddings (SentenceTransformer + Logistic Regression). Reports Accuracy, Precision, Recall, F1, AUC.

- `collaborative_filtering.ipynb`  
  Collaborative filtering model (matrix factorization via Truncated SVD). Uses interactions only. Reports ranking + rating metrics.

- `hybrid_cf_content.ipynb`  
  Creative extension model: hybrid CF + content recommender. Reports ranking metrics plus extension-specific metrics.

### Data Analysis
- `data_analysis.ipynb`
  Analyzes data counts/aggregations and creates visual plots.


### Demo Notebooks

- `demo_content_based-2.ipynb`  
  Content-based demo notebook with user scenario.

- `demo_hybrid.ipynb`  
  Creative extension demo notebook with user scenario.

### Data Files (expected in project root)

- `restaurants_santa_barbara.csv`
- `train_reviews_santa_barbara.csv`
- `test_reviews_santa_barbara.csv`

## Classification Model Data Split / Subset Note

For `classification_model.ipynb`, the provided train/test files are used, then filtered as:

- Positive class (`label = 1`): 4-5 star reviews
- Negative class (`label = 0`): 1-2 star reviews
- 3-star reviews removed as neutral/ambiguous

No train/test leakage is introduced; filtering is applied separately within each split.

## Setup and Run Instructions

### 1) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Launch Jupyter

```bash
jupyter notebook
```

Open notebooks and run each from top to bottom.

## Recommended execution order

1. Baselines: random, popularity, user-item bias
2. Content-based: A, B, C
3. Classification model
4. Collaborative filtering
5. Hybrid creative extension
6. Demo notebooks

## Requirements (minimum)

From `requirements.txt`:

- pandas>=2.2
- numpy>=1.26
- scikit-learn>=1.4
- sentence-transformers>=2.2
- torch>=2.0

## Installed library versions (project environment)

Exact versions from `.venv/bin/pip freeze`:

- annotated-doc==0.0.4
- anyio==4.13.0
- appnope==0.1.4
- asttokens==3.0.1
- certifi==2026.2.25
- click==8.3.2
- comm==0.2.3
- debugpy==1.8.20
- decorator==5.2.1
- executing==2.2.1
- filelock==3.25.2
- fsspec==2026.3.0
- h11==0.16.0
- hf-xet==1.4.3
- httpcore==1.0.9
- httpx==0.28.1
- huggingface_hub==1.10.1
- idna==3.11
- ipykernel==7.2.0
- ipython==9.12.0
- ipython_pygments_lexers==1.1.1
- jedi==0.19.2
- Jinja2==3.1.6
- joblib==1.5.3
- jupyter_client==8.8.0
- jupyter_core==5.9.1
- markdown-it-py==4.0.0
- MarkupSafe==3.0.3
- matplotlib-inline==0.2.1
- mdurl==0.1.2
- mpmath==1.3.0
- nest-asyncio==1.6.0
- networkx==3.6.1
- numpy==2.4.4
- packaging==26.0
- pandas==3.0.2
- parso==0.8.6
- pexpect==4.9.0
- platformdirs==4.9.6
- prompt_toolkit==3.0.52
- psutil==7.2.2
- ptyprocess==0.7.0
- pure_eval==0.2.3
- Pygments==2.20.0
- python-dateutil==2.9.0.post0
- PyYAML==6.0.3
- pyzmq==27.1.0
- regex==2026.4.4
- rich==15.0.0
- safetensors==0.7.0
- scikit-learn==1.8.0
- scipy==1.17.1
- sentence-transformers==5.4.0
- setuptools==81.0.0
- shellingham==1.5.4
- six==1.17.0
- stack-data==0.6.3
- sympy==1.14.0
- threadpoolctl==3.6.0
- tokenizers==0.22.2
- torch==2.11.0
- tornado==6.5.5
- tqdm==4.67.3
- traitlets==5.14.3
- transformers==5.5.3
- typer==0.24.1
- typing_extensions==4.15.0
- wcwidth==0.6.0
