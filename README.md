# Predicting Student Mental Health Score from Social Media Usage

A machine learning regression project that predicts a student's **Mental_Health_Score** (about 3 to 10) from their social media habits, study time, sleep, physical activity and stress level.

The final model is a **Random Forest pipeline** with a test R² of **0.878** and an average error of **0.35 score points**. The full pipeline (preprocessing and model) is saved in one file, ready to be used behind an API.

## Project Files

| File | Description |
|------|-------------|
| `ML_Project.ipynb` | The main notebook with all the code, from loading data to saving the model |
| `Student Social Media And Mental Health Impact.csv` | The dataset (5,000 students, 13 columns) |
| `Mental_Health_Model.pkl` | The saved pipeline, created by the notebook |
| `ML_Project_Report.docx` | Full project report with plots, results and the problems faced |
| `ML_Project_Step_by_Step_Guide.docx` | Every step of the project explained with code |

## Dataset

5,000 students and 13 columns, with no missing values.

| Group | Columns |
|-------|---------|
| Demographics | Age, Gender, Country, Academic_Level |
| Platform usage | Avg_Daily_Usage_Hours, Daily_Unlocks, Most_Used_Platform, Purpose_Of_Use |
| Lifestyle | Study_Hours, Sleep_Hours_Per_Night, Physical_Activity_Hours |
| Stress | Stress_Level (Low, Medium, High, Very High) |
| **Target** | **Mental_Health_Score** |

## What the Project Does

1. Loads and inspects the data
2. Explores it with six plots (target distribution, correlation heatmap, stress, usage, sleep and platform)
3. Checks for outliers with the IQR rule
4. Cleans the data: removes 2 duplicate rows and clips negative `Physical_Activity_Hours` to 0
5. Checks skewness and applies a `log1p` transform to `Study_Hours`
6. Groups `Country` from 111 values to 11 (top 10 plus "Other")
7. Encodes `Stress_Level` as ordinal and the other categorical columns with one-hot encoding
8. Splits the data 70% training and 30% testing
9. Builds the preprocessing with `ColumnTransformer` and chains it with the model in a `Pipeline`
10. Trains Linear Regression, a default Random Forest and a tuned Random Forest (`RandomizedSearchCV`)
11. Compares the models and saves the best pipeline with `joblib`

## Results

| Model | Test R² | Training R² | MAE | RMSE |
|-------|---------|-------------|-----|------|
| Linear Regression | 0.7398 | 0.7237 | 0.5362 | 0.6760 |
| **Random Forest (default)** | **0.8776** | 0.9808 | **0.3472** | **0.4637** |
| Random Forest (tuned) | 0.8650 | 0.9547 | 0.3689 | 0.4869 |

The default Random Forest scored best on the test set and is the saved model. Tuning reduced overfitting (a smaller gap between training and test R²) but did not improve the test score.

**Main findings**
- Higher daily usage (correlation -0.82) and more phone unlocks (-0.79) go with a lower mental health score.
- More sleep (0.77) and more study time (0.75) go with a higher score.
- The score falls step by step as stress level rises.

## Requirements

- Python 3
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn (the notebook upgrades it to 1.9.0)
- joblib

Install them with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

## How to Run

1. Open `ML_Project.ipynb` in Google Colab or Jupyter Notebook.
2. Put the dataset CSV in the same folder as the notebook (in Colab, upload it to the session).
3. Run the cells from top to bottom. The order matters because later cells use variables made earlier.
4. The last cells save the model as `Mental_Health_Model.pkl`.

If you run the `pip install --upgrade scikit-learn` cell, restart the runtime afterwards and run everything again.

## Using the Saved Model

```python
import joblib
import pandas as pd

model = joblib.load('Mental_Health_Model.pkl')

# Raw input: the pipeline does the scaling and encoding itself.
student = pd.DataFrame([{
    'Study_Hours': 3.0,
    'Age': 21,
    'Avg_Daily_Usage_Hours': 5.0,
    'Daily_Unlocks': 170,
    'Physical_Activity_Hours': 1.7,
    'Sleep_Hours_Per_Night': 6.6,
    'Stress_Level': 'Medium',
    'Gender': 'Female',
    'Academic_Level': 'Undergraduate',
    'Most_Used_Platform': 'Instagram',
    'Purpose_Of_Use': 'Social',
    'Grouped_country': 'Other'
}])

print(model.predict(student))
```

The values above are examples. Use the same column names, and category values that exist in the dataset (for example, check `df['Academic_Level'].unique()`). Load the model with the same scikit-learn version that was used to train it.

## Limitations

- The relationships in the data are very strong and clean, so results may not carry over to real-world data.
- The results show association, not causation.
- The default Random Forest still shows some overfitting (training R² 0.98 against test R² 0.88).

## Next Steps

- Build a FastAPI backend with Pydantic validation and a `/predict` endpoint
- Connect a simple frontend and deploy it
- Try Gradient Boosting and compare with cross-validation
- Add feature importance to explain predictions

## Author

[Your name] - [Your roll number] - [Course name]
