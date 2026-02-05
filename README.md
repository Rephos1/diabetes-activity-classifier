# Diabetes Activity Classifier

Diabetes prediction using SVM, comparing how physical activity affects classification accuracy.

## What This Is

Built an SVM classifier to predict diabetes risk, with a twist: I split the data into two groups—people who exercise and people who don't—to see if physical activity changes how well the model performs.

Spoiler: it does.

## The Data

Four datasets from the CDC's BRFSS survey:
- `diabetes_NoActivity_training.csv` / `diabetes_NoActivity_test.csv`
- `diabetes_PhysActivity_training.csv` / `diabetes_PhysActivity_test.csv`

Features include BMI, age, blood pressure, cholesterol, and other health indicators.

## What's Inside

- **Data exploration** — scatter plots, feature distributions
- **PCA** — dimensionality reduction to visualize class separation
- **SVM with RBF kernel** — hyperparameter tuning (C, gamma) via validation split
- **Cross-model testing** — train on one group, test on the other to see generalization

## Tech Stack

- Python
- scikit-learn
- pandas / numpy
- matplotlib / seaborn

## Running It

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Then run `diabetes-classification-notebook.py` or open `code.ipynb`.

## Key Takeaway

Physical activity isn't just good for your health—it also affects how predictable diabetes risk becomes. The models trained on each group behave differently when cross-tested, which says something about how lifestyle factors interact with other health metrics.
