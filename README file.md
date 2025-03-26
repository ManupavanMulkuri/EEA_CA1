# Design Decision 1: Chained Multi-outputs

This project implements a **Design Choice 1** architecture for classifying email support tickets using **chained multi-outputs**, where predictions for labels **Type2**, **Type3**, and **Type4** are evaluated in a hierarchical manner.

## Repository Structure

```
├── main.py                  # Main driver script for training and evaluation
├── Config.py                # Configuration file with constants and label mappings
├── preprocess.py            # Text cleaning, noise removal, label chaining
├── embeddings.py            # TF-IDF vectorizer to convert text into features
├── model/
│   ├── __init__.py          # Makes the directory a Python package
│   ├── base.py              # Base class with abstract model interface
│   └── randomforest.py      # RandomForest model implementation
├── modelling/
│   ├── data_model.py        # Data wrapper class for preprocessing and train/test split
│   └── modelling.py         # Model runner and training logic
├── data/
│   ├── AppGallery.csv       # Input dataset
│   └── Purchasing.csv       # Input dataset
```

---

## Implemented Design

### Design Choice 1: Chained Multi-Output
- Combines **Type2**, **Type3**, and **Type4** labels into:
  - **y2** (Type2)
  - **y2_3** (Type2 + Type3)
  - **y2_3_4** (Type2 + Type3 + Type4)
- Trains a single model for each of these targets
- Classification Report is printed for each combination.

### Evaluation
- **Hierarchical Accuracy**: Only counts **Type3** if **Type2** is correct, and **Type4** if both **Type2** and **Type3** are correct. This is dependency on previous Type.
- This is calculated only for one combination **y2_3_4**

## How to Run

### Requirements:
- Python 3.8+
- Install libraries if missing

### Run the pipeline:
```bash
python main.py
```

### Output
- Model performance for each of the following combinations:
  - **y2**
  - **y2_3**
  - **y2_3_4**
- Hierarchical accuracy results are calculated only for y2_3_4. Individual predictions and true values are printed for clarification.

## Input Data Format
- Each CSV file should contain the following columns:
  - **Ticket summary**
  - **interaction_content**
  - **y2**, **y3**, **y4**: These are renamed labels for Type2, Type3, and Type4 classification