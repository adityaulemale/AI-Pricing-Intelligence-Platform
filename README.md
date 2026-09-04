# 💰 AI Pricing Intelligence Platform

An end-to-end **Machine Learning and Explainable AI platform** for demand prediction, dynamic pricing optimization, revenue improvement analysis, and transparent pricing recommendations.

The platform combines a trained demand prediction model with a pricing optimization engine and SHAP-based explainability layer, exposed through an interactive Streamlit application.

---

## 🚀 Project Overview

Pricing decisions directly affect both customer demand and business revenue.

The **AI Pricing Intelligence Platform** predicts product demand based on business and market conditions and evaluates multiple candidate prices to identify the price that maximizes expected revenue.

The system follows:

```text
Business / Market Inputs
        ↓
Data Processing & Feature Engineering
        ↓
Demand Prediction Model
        ↓
Predicted Demand
        ↓
Pricing Optimization
        ↓
Expected Revenue Analysis
        ↓
Optimal Price Recommendation
        ↓
SHAP Explainability
        ↓
Streamlit Dashboard
```

---

## 🎯 Key Objectives

* Predict product demand using machine learning
* Analyze the relationship between price and demand
* Recommend revenue-maximizing prices
* Apply business pricing constraints
* Quantify expected revenue improvement
* Explain model predictions using SHAP
* Provide an interactive pricing intelligence dashboard
* Build a complete production-oriented ML pipeline

---

## ✨ Key Features

### 📈 Demand Prediction

The platform predicts expected product demand using features such as:

* Product
* Store
* Category
* Region
* Price
* Discount
* Promotion
* Competitor pricing
* Inventory level
* Weather condition
* Seasonality
* Epidemic indicator
* Date-related information

---

### 💰 Dynamic Pricing Optimization

The pricing optimization engine evaluates multiple candidate prices.

The optimization objective is:

```text
Expected Revenue = Price × Predicted Demand
```

The engine:

1. Generates valid candidate prices
2. Applies minimum and maximum price constraints
3. Applies maximum allowed price movement
4. Predicts demand for every candidate price
5. Calculates expected revenue
6. Selects the highest-revenue price
7. Reports the resulting revenue improvement

---

### 📊 Price Sensitivity Analysis

The platform evaluates the relationship between:

```text
Price
  ↓
Predicted Demand
  ↓
Expected Revenue
```

This allows users to compare candidate prices and understand how changing price affects predicted demand and revenue.

---

### 🔍 Explainable AI with SHAP

The platform uses **SHAP (SHapley Additive exPlanations)** to explain demand predictions.

It provides:

* Global feature importance
* Individual prediction drivers
* SHAP contribution values
* Base prediction value
* Reconstructed prediction
* Prediction consistency verification
* Global SHAP visualization
* Local SHAP visualization

This helps answer:

> "Why did the model predict this level of demand?"

---

### 🖥️ Streamlit Dashboard

The complete ML workflow is exposed through an interactive Streamlit application.

Users can enter:

* Store information
* Product information
* Category
* Region
* Inventory
* Current price
* Discount
* Promotion
* Competitor price
* Weather
* Seasonality
* Epidemic indicator

The application then displays:

* Predicted demand
* Current revenue
* Recommended price
* Optimized demand
* Optimized revenue
* Revenue improvement
* Price sensitivity analysis
* SHAP feature importance
* Prediction drivers
* SHAP visualizations
* Input summary

---

## 🏗️ Project Architecture

```text
AI-Pricing-Intelligence-Platform/
│
├── src/
│   ├── components/
│   │
│   ├── data_ingestion/
│   │
│   ├── data_transformation/
│   │
│   ├── models/
│   │
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   │
│   ├── optimization/
│   │   └── pricing_optimizer.py
│   │
│   ├── explainability/
│   │   └── shap_explainer.py
│   │
│   ├── exception.py
│   └── logger.py
│
├── streamlit_app/
│   └── app.py
│
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_prediction_pipeline.py
│   └── test_pricing_optimizer.py
│
├── artifacts/
│   ├── processed/
│   └── explainability/
│
├── models/
│   └── final_demand_model.pkl
│
├── notebooks/
│
├── requirements.txt
├── README.md
└── setup.py
```

---

## 🔬 Machine Learning Pipeline

The project was developed through the following stages:

| Phase | Component                                 |
| ----- | ----------------------------------------- |
| 1     | Project Setup                             |
| 2     | Dataset & MySQL Setup                     |
| 3     | Exploratory Data Analysis                 |
| 4     | Data Ingestion                            |
| 5     | Data Transformation & Feature Engineering |
| 6     | Demand Prediction Model                   |
| 7     | Model Evaluation & Tuning                 |
| 8     | Prediction Pipeline                       |
| 9     | Pricing Optimization Engine               |
| 10    | SHAP Explainability                       |
| 11    | Streamlit Application                     |
| 12    | End-to-End Integration                    |
| 13    | MLflow / Experiment Tracking              |
| 14    | Testing & Error Handling                  |
| 15    | Deployment                                |
| 16    | Documentation & Presentation              |

### Note

MLflow experiment tracking was evaluated as an optional component and was **not required for the core platform workflow**. The project therefore proceeds without MLflow while retaining the complete demand prediction, optimization, explainability, application, and testing workflow.

---

## 🧠 Model Prediction Pipeline

The prediction pipeline ensures that inference uses the same feature engineering and preprocessing logic established during model development.

The pipeline supports:

* Dictionary input
* Pandas Series input
* Pandas DataFrame input
* Single-record prediction
* Batch prediction
* Input validation
* Missing-column validation
* Non-negative demand output

---

## 💡 Pricing Optimization Example

For the implemented sample scenario:

```text
Current Price:        ₹50.00
Recommended Price:    ₹60.00

Current Demand:       181.44
Optimized Demand:     206.07

Current Revenue:      ₹9,072.00
Optimized Revenue:    ₹12,364.20

Revenue Improvement:  36.29%
Price Change:         20.00%

Candidate Range:      ₹40.00 – ₹60.00
```

The optimizer reported:

```text
Recommendation Status: BOUNDARY_LIMITED
```

This indicates that the best revenue-producing candidate reached the configured pricing boundary.

> **Important:** These values represent the implemented sample scenario and should not be interpreted as universal business results.

---

## 🔍 SHAP Explainability Example

For the sample prediction:

```text
Predicted Demand:             181.44
SHAP Base Value:              105.35
Reconstructed Prediction:     181.44
Prediction Difference:        ~0
```

Top global SHAP features included:

```text
1. Category — Groceries
2. Price / Promotion
3. Weather — Sunny
4. Seasonality — Summer
5. Category × Seasonality
6. Epidemic
7. Price
8. Category — Furniture
9. Weather — Rainy
10. Price Discount
```

The SHAP reconstruction check confirms that the individual SHAP contributions reproduce the model prediction within numerical precision.

---

## 🖥️ Running the Application

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd AI-Pricing-Intelligence-Platform
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run streamlit_app/app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

---

## 📦 Main Technologies

| Technology   | Purpose                          |
| ------------ | -------------------------------- |
| Python       | Core development                 |
| Pandas       | Data processing                  |
| NumPy        | Numerical computation            |
| Scikit-learn | Machine learning & preprocessing |
| XGBoost      | Machine learning                 |
| SHAP         | Explainable AI                   |
| Streamlit    | Interactive dashboard            |
| MySQL        | Data storage                     |
| Joblib       | Model/artifact serialization     |
| Matplotlib   | Visualization                    |
| Seaborn      | Exploratory visualization        |

---

## 🛡️ Error Handling

The project includes structured error handling through custom exception management.

Validation is implemented across the major components, including:

* Invalid input types
* Empty datasets
* Missing required columns
* Invalid prices
* Invalid pricing ranges
* Invalid price steps
* Invalid optimization constraints
* Missing model artifacts
* Missing preprocessing artifacts
* Invalid SHAP input

Logging is also used to track important pipeline operations and errors.

---

## 📈 Business Value

The platform demonstrates how machine learning can support pricing decisions by combining:

```text
Demand Forecasting
       +
Price Optimization
       +
Revenue Analysis
       +
Explainable AI
       =
Pricing Intelligence
```

Instead of simply predicting demand, the system converts the prediction into an actionable pricing recommendation.

---

## ⚠️ Important Considerations

The recommended price is based on the trained demand model and configured pricing constraints.

Therefore:

* Model quality directly affects pricing recommendations.
* Historical patterns may not always represent future market behavior.
* Business constraints should be configured appropriately.
* Recommendations should be validated against real-world pricing strategy.
* The sample revenue improvement is scenario-specific.

The platform is intended as a **decision-support system**, not an autonomous pricing authority.

---

## 🔮 Future Improvements

Potential extensions include:

* MLflow experiment tracking
* Automated model retraining
* Time-series demand forecasting
* Real-time market/competitor pricing
* Price elasticity estimation
* Customer segmentation
* Promotion optimization
* Multi-product optimization
* Inventory-aware pricing
* Advanced revenue constraints
* Cloud deployment
* CI/CD automation
* Monitoring and model drift detection
* Database-driven real-time predictions

---

## 👨‍💻 Skills Demonstrated

This project demonstrates practical experience in:

* End-to-end Machine Learning
* Data preprocessing
* Feature engineering
* Model training
* Model evaluation
* Hyperparameter tuning
* Prediction pipelines
* Dynamic pricing
* Revenue optimization
* Explainable AI
* SHAP
* Streamlit
* MySQL
* Automated testing
* Exception handling
* Logging
* Production-oriented ML architecture

---

## 📌 Project Summary

The **AI Pricing Intelligence Platform** demonstrates a complete machine learning workflow that goes beyond prediction.

It integrates:

```text
Data
 ↓
Feature Engineering
 ↓
Machine Learning
 ↓
Demand Prediction
 ↓
Price Optimization
 ↓
Revenue Analysis
 ↓
SHAP Explainability
 ↓
Interactive Dashboard
 ↓
Automated Testing
```

The project showcases how machine learning predictions can be transformed into explainable and actionable business recommendations for pricing decisions.
