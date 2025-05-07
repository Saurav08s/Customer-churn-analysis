# 📉 Customer Churn Analysis & Prediction

This project analyzes customer churn data to identify the factors contributing to customer loss and predicts churn using various machine learning algorithms. The goal is to help businesses retain customers by understanding churn patterns and proactively addressing risks.

---

## 📌 Features

- Exploratory Data Analysis (EDA) with interactive visualizations
- Data preprocessing (handling missing values, encoding categorical variables, feature scaling)
- Machine Learning model training using:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
- Evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
- Feature importance analysis to interpret results

---

## 📂 Project Structure

```
📁 Customer Churn Analysis/
├── Customer Churn Analysis.ipynb     # Main analysis and model notebook
├── churn_data.csv                    # Dataset (replace with actual file name if different)
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🧪 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/customer-churn-analysis.git
cd customer-churn-analysis
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually install the main packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook "Customer Churn Analysis.ipynb"
```

---

## 📊 Exploratory Data Visualizations

- 📌 **Churn Distribution** — Visualizes the proportion of customers who left vs. stayed.
- 📌 **Monthly Charges vs. Churn** — Identifies billing patterns in churning customers.
- 📌 **Tenure vs. Churn** — Shows how length of subscription impacts churn.
- 📌 **Correlation Heatmap** — Highlights relationships between features.

---

## 🤖 Machine Learning Models Used

| Model                | Description                                   |
|---------------------|-----------------------------------------------|
| Logistic Regression | Simple, interpretable binary classifier       |
| Decision Tree       | Rule-based, handles non-linear relationships  |
| Random Forest       | Ensemble of trees to improve accuracy         |
| SVM                 | Classifier with good performance on small data|

---

## 📈 Model Evaluation Metrics

- ✅ Accuracy
- 📉 Precision & Recall
- 🔁 F1 Score
- 📊 Confusion Matrix

The model performance is compared to select the best one based on balanced evaluation metrics.

---

## 💡 Conclusion

This project provides valuable insights into what factors contribute to customer churn, and builds predictive models that can help businesses take proactive retention actions.

---

## 🚀 Future Enhancements

- ✅ Cross-validation and hyperparameter tuning
- ✅ Try advanced ML models (XGBoost, LightGBM)
- ✅ Deploy model using Streamlit or Flask for business teams
- ✅ Integrate live customer data pipeline

---

## 📎 Dependencies (requirements.txt)

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---
---

## ⭐ Support

If you found this project helpful, give it a ⭐ on GitHub and share with others!
