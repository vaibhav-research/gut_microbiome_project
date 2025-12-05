# ML-Based Prediction of Food Allergy Development From Gut Microbiome Data

## Project Overview

This project aims to develop a robust machine learning classifier that can predict the development of food allergies by analyzing an individual's gut microbiome data. By leveraging the latest advancements in microbiome foundation models, we seek to transform the landscape of food allergy prediction and prevention.

The core goal is to build a **binary classifier** that can distinguish between "healthy" and "allergic" subjects based on their gut microbiota composition. Future iterations will explore extending this model to predict the risk of developing specific types of food allergies.

## Motivation: The Microbiome-Allergy Link

Accumulating scientific evidence highlights the critical role of the gut microbiota in shaping immune tolerance to dietary antigens. Key findings that motivate this research include:

*   **Reduced Diversity:** Allergic children consistently exhibit reduced microbial diversity compared to healthy controls.
*   **Taxa Shifts:** There is a notable depletion of protective taxa (e.g., *Bifidobacterium*, *Faecalibacterium*, butyrate-producing *Clostridia*) and an enrichment of pro-inflammatory bacteria (e.g., *Escherichia-Shigella*, *Ruminococcus gnavus*).
*   **Predictive Potential:** These microbial shifts are often detectable months before the clinical manifestation of food allergies, suggesting a strong predictive potential for early intervention.

## Methodology

Our approach is built on a modern, two-stage machine learning architecture designed to handle the complexity and high dimensionality of microbiome data:

1.  **Microbiome Embedding Model:** We utilize a recently developed **foundation model** for microbiome data. This model serves as a backbone, extracting rich, low-dimensional, and meaningful representations (embeddings) from raw 16S rRNA or shotgun metagenomics data.
2.  **Classifier Head:** The extracted embeddings are then fed into a simpler machine learning model, such as **Logistic Regression** or a similar classifier, to perform the final binary prediction.

### Evaluation Strategy

To ensure the model is robust and generalizable, we employ a rigorous evaluation strategy:

*   **Standard Metrics:** Performance is assessed using standard classification metrics (e.g., AUC, F1-score, Accuracy) with cross-validation.
*   **Cross-Dataset Validation:** We implement a **leave-one-dataset-out** validation strategy to test the model's ability to generalize across different cohorts, sequencing protocols, and study conditions, which is crucial for real-world applicability.

## Repository Structure

The project follows a modular structure to separate concerns and facilitate collaboration:

| Directory/File | Purpose |
| :--- | :--- |
| `data_preprocessing/` | Scripts for cleaning, transforming, and preparing raw microbiome data. |
| `modules/` | Reusable classes and functions for the model architecture (e.g., the `MicrobiomeTransformer` wrapper). |
| `evaluation/` | Scripts for model performance assessment, metric calculation, and visualization. |
| `train.py` | The main entry point for running the model training pipeline. |
| `main.py` | The overall execution script for the entire project workflow. |
| `data_loading.py` | Unified data loading pipeline with automatic artifact generation (DNA CSVs, embeddings H5) and data loading |
| `config.yaml` | Centralized YAML configuration file for all run parameters (data paths, model settings, evaluation metrics). |
| `utils.py` | Helper functions including configuration loading from YAML. |

## Getting Started

To set up the project environment and begin contributing:

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/AI-For-Food-Allergies/gut_microbiome_project.git
cd gut_microbiome_project

# Using pip
pip install -e .

# Or using uv (recommended)
uv sync
```

### 2. Data Setup

1.  **Download Resources**:
    Download the dataset and the pre-trained Microbiome Transformer checkpoint from Figshare:
    [Model and Data for diabimmune example](https://figshare.com/articles/dataset/Model_and_Data_for_diabimmune_example/30429055?file=58993825)

2.  **Place Files**:
    Unzip and organize the downloaded files into the project directory structure as follows:

    ```text
    gut_microbiome_project/
    ├── data/
    │   └── checkpoint_epoch_0_final_epoch3_conf00.pt  <-- Place Model Checkpoint here
    ├── data_preprocessing/
    │   ├── mapref_data/
    │   │   ├── samples-otus-97.parquet              <-- Place Parquet files here
    │   │   └── otus_97_to_dna.parquet
    │   └── datasets/
    │       └── goldberg/                            <-- Place your dataset CSVs here
    │           └── T3.csv
    ```

### 3. Usage

The project uses a centralized `config.yaml` to manage data paths and model parameters.

#### Configuration (`config.yaml`)
Ensure your `config.yaml` points to the correct locations. Key parameters:
-   `dataset_path`: Path to your sample metadata CSV (must contain sample IDs and labels).
-   `mirobiome_transformer_checkpoint`: Path to the `.pt` file downloaded above.

#### extracting Microbiome Embeddings
The data loading pipeline automatically handles the conversion of raw microbiome data into embeddings. This is handled by the `load_dataset_df` function in `data_loading.py`.

The pipeline performs the following steps automatically:
1.  **Extract Sequences**: Maps sample IDs to DNA sequences using the parquet map files.
2.  **Generate DNA Embeddings**: Uses `ProkBERT` to embed each DNA sequence.
3.  **Generate Microbiome Embeddings**: Uses the `MicrobiomeTransformer` to aggregate DNA embeddings into a single vector per sample.
4.  **Create DataFrame**: Returns a pandas DataFrame containing Sample IDs, Labels, and Embeddings.

You can trigger this pipeline manually or use it in your scripts:

```python
from data_loading import load_dataset_df
from utils import load_config

config = load_config()
dataset_df = load_dataset_df(config)

print(dataset_df.head())
# Columns: sid, label, embedding
```

> [!NOTE]
> The first run will take some time to generate the embeddings. Subsequent runs will use the cached H5 files in `data_preprocessing/dna_embeddings` and `data_preprocessing/microbiome_embeddings`.

#### Training the Classifier
To train and evaluate the model, you can use the `main.py` script, which orchestrates data loading and model training.

The `SKClassifier` class simplifies model instantiation and cross-validation:

```python
# main.py usage
from data_loading import load_dataset_df
from modules.classifier import SKClassifier
from utils import load_config, prepare_data

# 1. Load Configuration
config = load_config()

# 2. Load & Prepare Data (Automatically generates embeddings if needed)
dataset_df = load_dataset_df(config)
X, y = prepare_data(dataset_df)

# 3. Instantiate & Train Model
# Classifier type is defined in config.yaml (e.g., 'logreg', 'rf', 'svm')
classifier = SKClassifier(config['model']['classifier'], config)

# 4. Evaluate (Cross-Validation)
classifier.cross_validate(X, y, k=5)
```

To run the full pipeline:
```bash
python main.py
```
    

## 🤝 Contributing

We welcome contributions from researchers, data scientists, and developers!

Please refer to our dedicated **[Contributing Guide](https://github.com/AI-For-Food-Allergies/gut_microbiome_project/blob/master/Contributing.md)** for detailed instructions on:

*   Setting up your development environment.
*   The current development roadmap and open issues.
*   Guidelines for submitting pull requests and writing clean code.

**Want to be assigned to an issue?** Join the [huggingscience Discord server](https://discord.com/invite/VYkdEVjJ5J) and communicate your interest in the relevant channel!

Thank you for helping us advance the prediction of food allergies!
