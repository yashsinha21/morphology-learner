# Morphology Learner: Interpretable Machine Learning for Form-Meaning Mappings

This project explores how **decision trees** can be used to learn mappings between *forms* (words or subparts of words) and their *meanings*, represented as combinations of binary features. It focuses on **interpretable machine learning**, allowing the extraction of linguist-readable rules from trained models. Synthetic data is generated from morphological paradigms, and the system learns generalized form-meaning mappings rather than just memorizing the paradigm.  

## Modules

- **Synthetic Data Generation:** Generate large datasets from morphological paradigms with controllable noise.
- **Decision Tree Training:** Train a classifier to predict morphs (i.e, forms) from feature specifications.
- **Interpretation:** Extract linguist-readable rules from the trained decision tree.

For a full walkthrough, including data generation, training, evaluation, and rule extraction, see the [`demo.ipynb`](demo.ipynb) notebook.

## Files

- `src/data_generator.py` – Generates synthetic datasets from CSV paradigms.
- `src/train_tree.py` – Trains a decision tree classifier on synthetic data.
- `src/interpret_tree.py` – Extracts linguist-readable rules from a trained decision tree.
- `data/English_pronouns.csv` – Example paradigm for English third-person subject pronouns.
- `demo.ipynb` – Jupyter notebook demonstrating the full workflow.
- `requirements.txt` – List of Python packages required to run the project.

## Requirements

- Python 3.x  
- numpy  
- pandas  
- scikit-learn  

Install the required packages via:

```bash
pip install -r requirements.txt
