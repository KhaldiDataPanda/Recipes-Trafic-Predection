# Recipe Traffic Prediction App

This project analyzes recipe data to predict whether a recipe will receive high web traffic. It includes:

- **Exploratory Data Analysis (EDA)** using pandas, seaborn, and matplotlib.
- **Model Development** with scikit-learn, LazyPredict, XGBoost, CatBoost, and RandomizedSearchCV.
- **Model Export & Persistence** saving artifacts (models, scaler, encoder, metadata).
- **Deployment** with FastAPI for model serving and Streamlit for a front-end dashboard.

[Recipe Traffic Prediction App](HomePage.png)

---

## 1. Data & Exploratory Analysis

Data source: `recipes.csv` (947 recipes, 8 features including nutrients, category, servings, and traffic label).

### Key EDA Findings

- **Category distribution**: Certain categories (e.g., Meat, Dessert) dominate the dataset.
- **Calorie distribution**: Right-skewed, with a long tail of high-calorie recipes.
- **Feature correlations**: Category shows the strongest relationship with high-traffic label.

#### Correlation Matrix
![Correlation Matrix](<figs/Correlation Plot.png>)

---

## 2. Model Development & Evaluation

- **Preprocessing**: Imputed missing nutrients by category mean, target-encoded categorical fields, standardized numerical features.
- **Baseline Sweep**: LazyClassifier produced F1-scores around 0.77 across common algorithms.

#### Baseline Classifier Performance
![Baseline Classifier Performance](<figs/Baseline Classifiers.png>)

- **Hyperparameter Tuning**: RandomizedSearchCV on Random Forest, XGBoost, CatBoost, SVM, and BernoulliNB.
- **Tuned Results**: Tree-based ensembles (Random Forest, XGBoost, CatBoost) reached F1 ≈ 0.81.

#### Feature Importances (Tuned Models)
![Feature Importances](<figs/Feature importancest.png>)

---

## 3. Model Export & Persistence

Artifacts saved in the `model/` folder:

- **Best model**: `best_model_<model_name>.pkl`
- **Scaler**: `scaler.pkl`
- **Category encoder**: `category_encoder.pkl`
- **Metadata**: `metadata.pkl` (includes performance metrics and file paths)

---

## 4. Deployment

### FastAPI Service

`api.py` exposes a `/predict` endpoint:

```python
from fastapi import FastAPI
from model_loader import load_artifacts

app = FastAPI()
model, scaler, encoder = load_artifacts()

@app.post("/predict")
def predict(data: dict):
    # Returns probability of high traffic
    return {"high_traffic_prob": ...}
```

Run the API server with:

```powershell
pip install fastapi uvicorn pydantic
uvicorn api:app --reload
```  

### Streamlit Dashboard

`app.py` provides an interactive UI:

- Page 1: Input recipe features and view prediction.
- Page 2: Visualize model insights and KPI metrics.

Home & Input Page:

![Streamlit Input Page](<figs/Steamlit Page1.png>)

Model Insights & Results:

![Streamlit Results Page](<figs/Streamlit Page2.png>)

Launch the dashboard with:

```powershell
pip install streamlit pandas scikit-learn matplotlib seaborn
streamlit run app.py
```

---

## 5. Usage & Requirements

1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. ```powershell
    .\launch.bat```
4. Send POST requests to `/predict` or use the Streamlit UI.

---

