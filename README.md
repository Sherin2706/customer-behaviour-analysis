# 🛒 Online Shopping Behaviour Analysis using Machine Learning

## 📌 Project Overview
This project analyzes customer online shopping behavior and predicts whether a customer will purchase a product or not using machine learning techniques.

It helps businesses understand customer preferences and improve their marketing strategies.

---

## 🎯 Problem Statement
Analyzing customer behavior manually is complex and time-consuming due to multiple influencing factors like discounts, ratings, and spending habits.

This project solves the problem by using machine learning to automatically predict purchase decisions.

---

## 🎯 Objective
- Analyze customer shopping behavior
- Predict purchase decision (Buy / Not Buy)
- Improve business insights using data

---

## 📊 Dataset
- Collected manually using Google Forms
- Total records: **52**
- Features include:
  - Age Group
  - Gender
  - Monthly Spending
  - Purchase Frequency
  - Return Rate
  - Device Used
  - Platform
  - Payment Method
  - Product Category
  - Discount Influence
  - Discount Range
  - Purchase Factors
  - Minimum Rating
  - Media Influence
  - Purchase Decision (Target)

---

## 🧹 Data Preprocessing
- Removed unnecessary columns (Name, Timestamp)
- Handled missing values
- Converted categorical data to numerical (Encoding)
- Cleaned percentage values (Return Rate)
- Expanded dataset using resampling

---

## 🤖 Machine Learning Models Used
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)

---

## 📈 Model Evaluation
- Logistic Regression: **97.5% accuracy**
- Decision Tree: **95% accuracy**
- KNN: **95% accuracy**

---

## 🧠 Model Improvement
- Ensemble Learning (Voting Classifier)
- K-Fold Cross Validation

---

## 💾 Model Saving
- Saved using `joblib` for reuse

---

## 🌐 Web Application
Built using **Streamlit**

### Features:
- Upload dataset (CSV)
- Predict customer behavior
- Display accuracy
- Show pie chart visualization
- Voice output for results

---

## 📊 Visualization
- Bar charts for feature analysis
- Pie chart for prediction results

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- Joblib
- pyttsx3 (Voice output)

---

## 🚀 How to Run the Project

1. Clone the repository
```bash
git clone https://github.com/Sherin2706/customer-behaviour-analysis.git
Navigate to project folder
cd customer-behaviour-analysis
Install dependencies
pip install -r requirements.txt
Run Streamlit app
streamlit run MLproject.py
📌 Future Scope
Use larger dataset
Apply advanced ML models (XGBoost, Neural Networks)
Deploy as a full web application
Add real-time data integration
👩‍💻 Author

SHERIN V

⭐ Conclusion

This project successfully predicts customer purchase behavior and provides useful insights for businesses to improve decision-making.
