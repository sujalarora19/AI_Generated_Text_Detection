# Real-Time Text Classification with LLM Detection

This repository contains a project for detecting AI-generated text using various machine learning models. The project leverages the power of natural language processing (NLP) techniques and ensemble learning to provide robust detection.

## Features

- **Data Preprocessing**: Uses TF-IDF for text vectorization.
- **Custom Tokenization**: Utilizes Byte-Pair Encoding (BPE) with HuggingFace's Tokenizer.
- **Machine Learning Models**: Includes a combination of Naive Bayes, SGD Classifier, LightGBM, and CatBoost for ensemble learning.
- **Evaluation Metrics**: Uses AUC-ROC for model evaluation.
- **Ensemble Learning**: Combines multiple models to improve performance.

## Libraries and Dependencies

Ensure you have the following libraries installed:

- `pandas` for data manipulation.
- `scikit-learn` for machine learning models and utilities.
- `lightgbm` for gradient boosting.
- `transformers` for advanced NLP tasks.
- `datasets` for handling data.
- `tokenizers` for fast and efficient tokenization.
- `tqdm` for displaying progress bars.
- `gc` for garbage collection management.
- `numpy` for numerical operations.
- `catboost` for gradient boosting on decision trees.

## How to Use

Install all required packages using pip:

### Clone the Repository
```bash
git clone https://github.com/yourusername/real-time-text-classification.git
cd real-time-text-classification
```
### Prepare Your Environment 
- Ensure all dependencies are installed.
### Train the Model
- Update the dataset paths in the script if necessary.
- Run the training script:
```bash
python train.py
```
### Make Predictions:

- The model will automatically save the predictions in submission.csv.

## Data
- The dataset consists of texts labeled as either AI-generated or human-written. The dataset used for training and testing is assumed to be in the data/ directory. Update the paths in the code if your dataset is located elsewhere.

## How It Works
- **Data Preparation**: Text data is read, preprocessed, and tokenized.
- **Feature Extraction**: Uses TF-IDF to transform text data into feature vectors.
- **Model Training**: Multiple models are trained, and an ensemble model is created.
- **Prediction**: The ensemble model is used to predict the probability of a text being AI-generated.

## Important Notes
- **Model Parameters**: Hyperparameters for the models can be fine-tuned for better performance.
- **Performance Metrics**: Evaluation is based on AUC-ROC scores.
- **Ensemble Weights**: Adjust weights for each model in the ensemble for optimal results.

## License
- This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments
- Thanks to the developers of all open-source libraries and tools used in this project.
- Special thanks to the Kaggle community for providing the datasets.
