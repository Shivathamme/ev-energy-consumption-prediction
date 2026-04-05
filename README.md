# 🚗 EV Energy Consumption Prediction

## 📌 Project Overview

This project predicts the **energy consumption of electric vehicles (EVs)** using Machine Learning techniques.
The goal is to understand how driving behavior and environmental conditions affect energy usage.

---

## 🎯 Problem Statement

To build a regression model that predicts:

**Target Variable:**

* `Energy_Consumption_kWh`

---

## 📊 Dataset Features

The dataset includes:

* Speed_kmh
* Distance_Travelled_km
* Battery_Temperature_C
* Slope_%
* Humidity_%
* Driving_Mode
* Road_Type
* Traffic_Condition
* Weather_Condition

---

## ⚙️ Workflow

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Selection
4. VIF (Multicollinearity Check)
5. Feature Engineering
6. Train-Test Split
7. Preprocessing (Encoding + Scaling)
8. Model Training
9. Model Comparison
10. Cross Validation
11. Hyperparameter Tuning
12. Feature Selection (RFE)
13. Model Evaluation

---

## 🤖 Models Used

* Linear Regression ✅ (Best Model)
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor

---

## 📈 Model Performance

| Model                     | R² Score  |
| ------------------------- | --------- |
| Linear Regression         | **0.946** |
| Gradient Boosting (Tuned) | 0.939     |
| Random Forest             | 0.913     |
| Decision Tree             | 0.808     |

---

## 🔍 Feature Selection Techniques

* VIF (Variance Inflation Factor)
* RFE (Recursive Feature Elimination)
* SFS (Sequential Feature Selection)

---

## 🧠 Key Insights

* Speed and distance significantly impact energy consumption
* Linear relationships dominate the dataset
* Linear Regression outperformed complex models

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
ev-energy-consumption-prediction/
│
├── app.py
├── energy_pipeline.pkl
├── notebook.ipynb
├── model_training.py
├── requirements.txt
└── README.md
```

---

## 🚀 Future Improvements

* Add more real-world EV datasets
* Deploy using cloud platforms
* Improve model with deep learning

