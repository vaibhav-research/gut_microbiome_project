# Contribution Guidelines

Welcome to the Gut Microbiome Project! This document provides an overview of the codebase structure and guidelines for contributing effectively.

## Table of Contents

1. [Code Structure](#1-code-structure)
2. [How to Contribute](#2-how-to-contribute)
3. [Development Workflow](#3-development-workflow)
4. [Code Standards](#4-code-standards)
5. [Testing Your Changes](#5-testing-your-changes)

## 1. Code Structure

### 1.1. High-Level Overview

The project is organized into a modular ML pipeline with clear separation of concerns:

| Component | Directory/File | Purpose |
| :--- | :--- | :--- |
| **Configuration** | `config.yaml` | Centralized YAML configuration for all parameters (data paths, model settings, hyperparameters). |
| **Data Pipeline** | `data_loading.py` | Unified data loading pipeline with automatic artifact generation (DNA CSVs, embeddings). |
| **Embedding Generation** | `generate_embeddings.py` | Standalone script for batch generation of embeddings across multiple datasets. |
| **Model Components** | `modules/` | Core model classes: `MicrobiomeTransformer` (embedding model) and `SKClassifier` (classification). |
| **Execution** | `main.py` | Main entry point for training and evaluation with multiple modes (simple eval, grid search). |
| **Utilities** | `utils/` | Helper functions for data preparation (`data_utils.py`) and evaluation (`evaluation_utils.py`). |
| **Data Preprocessing** | `data_preprocessing/` | Dataset CSVs, preprocessing scripts, mapping files, and generated embeddings. |
| **Examples** | `example_scripts/` | Example scripts demonstrating specific use cases. |

### 1.2. Key Design Principles

*   **Configuration-Driven:** All parameters are managed through `config.yaml` - avoid hardcoding values.
*   **Automatic Caching:** Embeddings are generated once and cached in H5 files for reuse.
*   **Modular Components:** Each module has a single responsibility (data loading, model, evaluation).
*   **Flexible Evaluation:** Support for multiple classifiers, cross-validation, and hyperparameter tuning.

### 1.3. Data Pipeline Flow

```
Sample CSV (sid, label)
    ↓
[data_loading.py] Checks for existing artifacts
    ↓
DNA Sequences (CSV) → DNA Embeddings (H5) → Microbiome Embeddings (H5)
    ↓                      ↓                           ↓
  (OTU to DNA)     (ProkBERT Model)      (MicrobiomeTransformer)
    ↓
DataFrame (sid, label, embedding)
    ↓
[main.py] Train & Evaluate Classifier
    ↓
Results (metrics, plots, confusion matrices)
```

### 1.4. Module Details

#### `data_loading.py`
The unified data loading pipeline that:
- Loads sample metadata (CSVs with sample IDs and labels)
- Generates DNA sequence CSVs from parquet mapping files
- Generates ProkBERT embeddings for each DNA sequence
- Aggregates DNA embeddings into per-sample microbiome embeddings
- Returns a pandas DataFrame ready for training

**Key Functions:**
- `load_dataset_df(config)` - Main entry point, returns DataFrame with embeddings
- `generate_dna_embeddings_h5()` - Generate ProkBERT embeddings
- `generate_microbiome_embeddings_h5()` - Generate MicrobiomeTransformer embeddings

#### `modules/model.py`
Contains the `MicrobiomeTransformer` class that wraps the pre-trained foundation model for aggregating DNA embeddings into a single microbiome embedding per sample.

#### `modules/classifier.py`
Contains the `SKClassifier` class that provides a unified interface for multiple scikit-learn classifiers:
- Logistic Regression (`logreg`)
- Random Forest (`rf`)
- Support Vector Machine (`svm`)
- Multi-Layer Perceptron (`mlp`)

**Key Methods:**
- `evaluate_model(X, y, cv)` - Simple cross-validation evaluation
- `grid_search_with_final_eval()` - Hyperparameter tuning with unbiased final evaluation

#### `utils/evaluation_utils.py`
Comprehensive evaluation utilities including:
- `EvaluationMetrics` - Data class for storing predictions and metrics
- `ResultsManager` - Automated saving of results, plots, and reports
- Functions for generating ROC curves, confusion matrices, and classification reports

#### `main.py`
Main execution script with multiple modes:
- `run_evaluation()` - Simple evaluation without hyperparameter tuning
- `run_grid_search_experiment()` - Grid search with proper unbiased evaluation

## 2. How to Contribute

We welcome contributions! Follow these steps to get started.

### 2.1. Getting Started

1.  **Fork the Repository:** Create your own fork on GitHub.

2.  **Clone Your Fork:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/gut_microbiome_project.git
    cd gut_microbiome_project
    ```

3.  **Set Up the Environment:**
    ```bash
    # Using pip
    pip install -e .
    
    # Or using uv (recommended)
    uv sync
    ```

4.  **Download Required Data:**
    Follow the setup instructions in the [main README](readme.md) to download:
    - Dataset CSVs from Google Drive
    - Model checkpoint from Figshare
    - Parquet mapping files from Figshare

### 2.2. Finding Something to Work On

1.  **Check Open Issues:** Browse the [GitHub Issues](https://github.com/AI-For-Food-Allergies/gut_microbiome_project/issues) page for tasks.
2.  **Join Discord:** Connect with the team on the [huggingscience Discord](https://discord.com/invite/VYkdEVjJ5J) to discuss ideas and get assigned to issues.
3.  **Suggest Improvements:** Have an idea? Open a new issue to discuss it before implementing.

### 2.3. Common Contribution Areas

- **New Datasets:** Add preprocessing scripts for additional microbiome datasets
- **Model Improvements:** Experiment with new architectures or embedding strategies
- **Evaluation Metrics:** Add new evaluation methods or visualizations
- **Documentation:** Improve README, docstrings, or add tutorials
- **Bug Fixes:** Fix reported issues or edge cases
- **Performance:** Optimize data loading or embedding generation

## 3. Development Workflow

### 3.1. Create a Branch

Always create a new branch for your work:

```bash
# For new features
git checkout -b feature/descriptive-feature-name

# For bug fixes
git checkout -b fix/issue-number-description

# For documentation
git checkout -b docs/what-you-are-documenting
```

### 3.2. Make Your Changes

Follow these guidelines while coding:

1.  **Use Configuration System:** All parameters should come from `config.yaml`. Never hardcode paths or hyperparameters.
    ```python
    # Good
    from utils.data_utils import load_config
    config = load_config()
    dataset_path = config['data']['dataset_path']
    
    # Bad
    dataset_path = "data_preprocessing/datasets/diabimmune/Month_2.csv"
    ```

2.  **Write Modular Code:** Keep functions focused on a single task. Extract reusable logic into utility functions.

3.  **Add Type Hints:** Use type annotations for function parameters and return values.
    ```python
    def load_embeddings(h5_path: Path) -> Dict[str, np.ndarray]:
        """Load embeddings from H5 file."""
        ...
    ```

4.  **Document Your Code:** Add docstrings to functions and classes explaining purpose, arguments, and return values.

5.  **Handle Errors Gracefully:** Add try-except blocks for file I/O and provide helpful error messages.

### 3.3. Test Your Changes

Before submitting:

1.  **Run the Pipeline:** Test your changes with a small dataset:
    ```bash
    # Edit config.yaml to use a small dataset
    python main.py
    ```

2.  **Check Multiple Scenarios:** Test edge cases (empty samples, missing data, etc.)

3.  **Verify Output:** Ensure results are saved correctly in `eval_results/`

4.  **Test with Different Configs:** Try different classifiers and parameters

### 3.4. Submit Your Contribution

1.  **Commit Your Changes:**
    ```bash
    git add .
    git commit -m "Add feature: brief description
    
    - Detail 1
    - Detail 2
    - Closes #issue_number"
    ```

2.  **Push to Your Fork:**
    ```bash
    git push origin feature/your-feature-name
    ```

3.  **Open a Pull Request:**
    - Navigate to the original repository on GitHub
    - Click "New Pull Request" and select your branch
    - Provide a clear description of your changes
    - Reference any related issues (e.g., "Closes #42")
    - Wait for review and address feedback

## 4. Code Standards

### 4.1. Python Style

- Follow PEP 8 style guidelines
- Use meaningful variable names (`embedding_dim` not `ed`)
- Keep lines under 100 characters when possible
- Use 4 spaces for indentation

### 4.2. Configuration Management

**Always use `config.yaml` for:**
- File paths (datasets, checkpoints, output directories)
- Model hyperparameters
- Training parameters (batch size, learning rate)
- Evaluation settings (CV folds, metrics)

**Never hardcode:**
- File paths
- Hyperparameters
- Device settings (cpu/cuda/mps)

### 4.3. File Organization

When adding new functionality:

- **Data Processing:** Add to `data_loading.py` or create a new script in `data_preprocessing/`
- **Model Logic:** Add to `modules/model.py` or `modules/classifier.py`
- **Evaluation:** Add to `utils/evaluation_utils.py`
- **Utilities:** Add to `utils/data_utils.py` or create a new utility module
- **Examples:** Add complete examples to `example_scripts/`

### 4.4. Documentation Standards

**Module Docstrings:**
```python
"""
Brief module description.

This module provides functionality for X, Y, and Z.
"""
```

**Function Docstrings:**
```python
def process_embeddings(embeddings: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Process embeddings by normalizing and filtering.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        normalize: Whether to normalize embeddings to unit length
        
    Returns:
        Processed embeddings array
        
    Raises:
        ValueError: If embeddings array is empty
    """
    ...
```

## 5. Testing Your Changes

### 5.1. Quick Test

Test with a small dataset to verify basic functionality:

```yaml
# config.yaml
data:
  dataset_path: "data_preprocessing/datasets/diabimmune/Month_2.csv"  # Small dataset
  device: "cpu"

evaluation:
  cv_folds: 3  # Reduce folds for faster testing
```

```bash
python main.py
```

### 5.2. Full Test

Test with multiple datasets and classifiers:

```python
# In main.py
run_evaluation(config, classifiers=["logreg", "rf", "svm", "mlp"])
```

### 5.3. Edge Cases to Test

- Empty samples (samples with no OTUs)
- Single class in data (should fail gracefully)
- Missing files (should provide clear error messages)
- Different dataset structures
- GPU vs CPU execution

### 5.4. Verify Outputs

Check that results are properly saved:
```bash
ls -R eval_results/
```

Expected structure:
```
eval_results/
└── dataset_name/
    └── timestamp/
        ├── Classifier_Name/
        │   ├── classification_report.txt
        │   ├── confusion_matrix.png
        │   ├── roc_curve.png
        │   └── predictions.csv
        └── combined_report.txt
```

## 6. Common Tasks

### 6.1. Adding a New Dataset

1. Preprocess your dataset to CSV format with columns: `sid` (sample ID) and `label`
2. Place in `data_preprocessing/datasets/<dataset_name>/`
3. Update `config.yaml` to point to your dataset
4. Run the pipeline - embeddings will be generated automatically

### 6.2. Adding a New Classifier

1. Add classifier initialization in `modules/classifier.py`:
    ```python
    def _get_base_classifier(self):
        if self.classifier_type == "your_new_classifier":
            from sklearn.xxx import YourClassifier
            return YourClassifier()
    ```

2. Add to `CLASSIFIER_NAMES` mapping
3. Add hyperparameter grid to `config.yaml`:
    ```yaml
    param_grids:
      your_new_classifier:
        param1: [value1, value2]
        param2: [value3, value4]
    ```

### 6.3. Adding New Evaluation Metrics

Add to `utils/evaluation_utils.py`:

```python
def calculate_your_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate your custom metric."""
    ...
    return metric_value
```

Update `ResultsManager` to include your metric in reports.

### 6.4. Optimizing Embedding Generation

If embedding generation is slow:

1. Increase batch size (if you have GPU memory):
    ```yaml
    data:
      batch_size_embedding: 16  # or higher
      device: "cuda"  # or "mps" for Mac
    ```

2. Pre-generate embeddings in batch using `generate_embeddings.py`

3. Consider parallel processing for multiple datasets

## 7. Need Help?

### Resources

- **Main README:** [readme.md](readme.md) - Usage instructions
- **Config Guide:** `example_scripts/CONFIG_GUIDE.md` - Configuration details
- **Example Scripts:** `example_scripts/` - Working examples

### Getting Support

- **Discord:** Join [huggingscience Discord](https://discord.com/invite/VYkdEVjJ5J) for discussions
- **GitHub Issues:** Open an issue for bugs or feature requests
- **Pull Request Comments:** Ask questions in your PR

### Before Asking

1. Check if your question is answered in the README or CONFIG_GUIDE
2. Look at example scripts for similar use cases
3. Search existing GitHub issues
4. Try a minimal example to isolate the problem

---

Thank you for contributing to the Gut Microbiome Project! Your work helps advance food allergy prediction research. 🦠🧬