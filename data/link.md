# Data Download Instructions

## Option 1: Direct Download from Kaggle
1. Visit the [IMDB Dataset (Sentiment Analysis) in CSV format](https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification) on Kaggle
2. Click the "Download" button
3. Place the downloaded files in the `data/` directory of this project

## Option 2: Using Python and kagglehub

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("ashfakyeafi/spam-email-classification")

print("Path to dataset files:", path)
```

### Prerequisites for kagglehub method:
1. Install kagglehub:
```bash
pip install kagglehub
```

## Data Usage Terms
Please refer to the [Kaggle dataset license](https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format/metadata) for terms of use.

