# 💳 Credit Card Default Prediction (Machine Learning)

This project predicts whether a customer will default on their credit card payment using machine learning techniques.

---

## 🚀 Demo Features
- Predict default risk (High / Low)
- User-friendly web interface (Flask)
- Uses real financial dataset

---

## 📁 Project Structure   

```
├── app.py
├── model.pkl
├── scaler.pkl
├── train_model.py
├── templates/
│   └── index.html
├── data/
│   └── default of credit card clients.xls
└── requirements.txt
```
---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Flask

---

## 📊 Dataset

This project uses the **Default of Credit Card Clients dataset** from the UCI Machine Learning Repository.

- Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients  
- Records: 30,000 customers  
- Features: 23 input variables  
- Target: Default payment next month (Yes/No)

The dataset contains customer demographic information, credit limit, repayment history, bill statements, and payment records to predict credit card default risk.


---
## ▶️ How to Run

1. Clone the repository:
   git clone https://github.com/jincyjos22/Credit-Card-Default-Prediction-Using-Machine-Learning.git

3. Go to project folder:
   cd Credit-Card-Default-Prediction-Using-Machine-Learning

4. Install dependencies:
   pip install -r requirements.txt

5. Run model training:
   python train_model.py

6. Run the app:
   python app.py

7. Open browser:
   http://127.0.0.1:5000/

---

## 📌 Output
- **High Risk** → Customer likely to default  
- **Low Risk** → Customer not likely to default  

---

## 🎨 UI Indication

- 🟢 **Green result** → Low Risk (Customer is likely to repay)
- 🔴 **Red result** → High Risk (Customer may default)

The color coding helps users quickly understand the prediction result.

---

Machine Learning based Credit Card Default Prediction

# 1. 📌 Introduction 

Credit card default is a major problem for financial institutions, leading to financial losses and increased risk.  

This project uses machine learning techniques to predict whether a customer will default on their credit card payment based on financial and repayment history data. The goal is to help banks identify high-risk customers early and improve credit decision-making.


# 2. Objective

The primary goal of the research is the development and assessment of several machine learning models which could forecast credit card customer default.

Moreover, the project should involve comparison of algorithms and determination of the most accurate model.

## 3. Data Description

### ● Source:

The dataset used in this project is obtained from the **UCI Machine Learning Repository**, titled **“Default of Credit Card Clients Dataset.”**
It is available as an Excel file and contains information about credit card clients in Taiwan.


### ● Features:

The dataset includes the following features:

* **LIMIT_BAL** → Credit limit amount of the customer
* **SEX** → Gender (1 = Male, 2 = Female)
* **EDUCATION** → Education level (1 = Graduate School, 2 = University, 3 = High School, 4 = Others)
* **MARRIAGE** → Marital status (1 = Married, 2 = Single, 3 = Others)
* **AGE** → Age of the customer

#### Repayment History (Last 6 Months):

* **PAY_0** → Repayment status in September
* **PAY_2** → Repayment status in August
* **PAY_3** → Repayment status in July
* **PAY_4** → Repayment status in June
* **PAY_5** → Repayment status in May
* **PAY_6** → Repayment status in April

#### Bill Statement Amounts:

* **BILL_AMT1** → Bill amount in September
* **BILL_AMT2** → Bill amount in August
* **BILL_AMT3** → Bill amount in July
* **BILL_AMT4** → Bill amount in June
* **BILL_AMT5** → Bill amount in May
* **BILL_AMT6** → Bill amount in April

#### Previous Payment Amounts:

* **PAY_AMT1** → Payment made in September
* **PAY_AMT2** → Payment made in August
* **PAY_AMT3** → Payment made in July
* **PAY_AMT4** → Payment made in June
* **PAY_AMT5** → Payment made in May
* **PAY_AMT6** → Payment made in April

---

### ● Target Variable:

* **default.payment.next.month**

  * **1** → Customer will default (fail to pay)
  * **0** → Customer will not default

## 4. Data Preprocessing & Cleaning

Data preprocessing was performed to clean and prepare the dataset for machine learning.

● Missing Values

The dataset was checked for missing values. Since very few or none were present, median imputation was used for numerical features.

● Outliers

Outliers in features like LIMIT_BAL and bill amounts were detected using boxplots and treated where necessary.

● Duplicate Records

Duplicate entries were removed to ensure data quality and avoid bias in training.

● Categorical Variables

Categorical features (SEX, EDUCATION, MARRIAGE) were already numerically encoded, so no additional encoding was required.

● Skewed Data

Skewed numerical features were analyzed, and scaling was applied to improve model performance.

● Feature Engineering

A new feature TOTAL_PAY was created by summing PAY_AMT1 to PAY_AMT6 to capture overall payment behavior.

● Feature Scaling

StandardScaler was applied to normalize numerical features for better model performance.

---


## 5. Exploratory Data Analysis (EDA) – Insights

Exploratory Data Analysis (EDA) was performed to understand the structure of the dataset, identify patterns, and explore relationships between features and the target variable.

---
📉 Target Distribution (Count Plot)

A count plot was used to visualize the distribution of default vs non-default customers.
It shows that the dataset is imbalanced, with more non-defaulters than defaulters.



📊 Age Distribution (Histogram + KDE)

A histogram was used to analyze the distribution of customer ages.
It helps understand the most common age group in the dataset.

💳 Credit Limit Distribution (Histogram + KDE)

A histogram was used to visualize LIMIT_BAL (credit limit) distribution.
It shows that most customers have lower to moderate credit limits.

📦 Outlier Detection (Box Plot)

A box plot was used to detect outliers in credit limit (LIMIT_BAL).
It shows the presence of customers with unusually high credit limits.

🔗 Feature Correlation (Heatmap)

A correlation heatmap was used to understand relationships between numerical features.
It helps identify how features like PAY_0, BILL_AMT1, and LIMIT_BAL relate to default risk.

💰 Repayment Status vs Default (Bar Plot)

A bar plot was used to analyze how repayment status (PAY_0) affects default probability.
Customers with delayed payments show a higher risk of default.

📈 Total Payment Distribution (Histogram)

A histogram was used to analyze TOTAL_PAY, which represents total payments made over 6 months.
It helps understand customer payment behavior patterns.

📌 Summary of Insights
Dataset is imbalanced (more non-defaulters)
Repayment status (PAY_0) is strongly related to default
Customers with lower credit limits are more common
Presence of outliers in financial features
Payment behavior is a strong indicator of default risk

## 6. Feature Engineering

Feature engineering involves transforming and creating new features to improve the performance of machine learning models.

---

### ● Encoding Categorical Features

In this dataset, categorical variables such as **SEX, EDUCATION, and MARRIAGE** are already represented in numerical form.
Therefore, additional encoding techniques like One-Hot Encoding or Label Encoding were not required.

However, these variables were carefully reviewed to ensure consistency and correctness before model training.

---

### ● Creating New Features

A new feature called **TOTAL_PAY** was created by summing all payment amounts from the past six months (**PAY_AMT1 to PAY_AMT6**).

[
\text{TOTAL_PAY} = \text{PAY_AMT1} + \text{PAY_AMT2} + \text{PAY_AMT3} + \text{PAY_AMT4} + \text{PAY_AMT5} + \text{PAY_AMT6}
]

This feature captures the overall payment behavior of customers and helps the model better understand their repayment capacity.

---

### ● Feature Selection

Important features were identified based on correlation analysis and domain knowledge.
Repayment history variables (**PAY_0 to PAY_6**) were found to be highly influential in predicting default.

## 7. Feature Scaling

The Credit Card Default dataset contains financial and behavioral features with significantly different value ranges. For example:

Credit limit and bill amounts are very large values
Payment amounts vary widely
Age is within a small range
Repayment status values are small integers

To ensure that all features contribute equally during model training, feature scaling was applied.

Technique Used

StandardScaler (Z-score normalization) was used to standardize all numerical features.

Reason for Selection
Financial features like bill amounts and credit limits have large variations
Scaling prevents these features from dominating smaller ones like age or repayment status
Essential for algorithms such as SVM and KNN
Conclusion

Feature scaling improved model performance by normalizing all financial and behavioral features in the dataset.

## 8. Model Building

The goal of this project is to predict whether a customer will default on their credit card payment based on historical financial data.

Dataset Handling
Input features include customer demographics, credit limit, repayment history, bill statements, and previous payments
Target variable: Default payment next month (1 = Yes, 0 = No)

The dataset was split into:

Training data (80%)
Testing data (20%)
Models Applied
Logistic Regression – baseline classification model
Decision Tree – captures decision rules from financial behavior
Random Forest – improves prediction by combining multiple trees
Support Vector Machine (SVM) – effective for classification boundaries
K-Nearest Neighbors (KNN) – uses similarity between customers
Conclusion

Multiple models were trained to understand different patterns in customer behavior and identify the most accurate model for predicting default risk.

## 9. Model Evaluation

## Model Evaluation

Models were evaluated to identify customers likely to default.

📊 Evaluation Metrics
Accuracy – overall correctness
Precision – correctness of predicted defaulters
Recall – ability to detect actual defaulters
F1-Score – balance between precision and recall
Confusion Matrix – prediction breakdown
ROC-AUC – model discrimination ability
⚠️ Key Insight

The dataset is imbalanced, so Recall is more important than Accuracy, as missing defaulters leads to financial risk.

🏆 Best Model

Random Forest performed best due to:

Better handling of complex patterns
Reduced overfitting
Stable performance across metrics
📌 Conclusion

Model evaluation helped select the most reliable model for predicting credit card default risk.

## 10. Hyperparameter Tuning

Hyperparameter tuning was performed to improve the performance of the Random Forest model.

- Used systematic search (GridSearchCV)
- Optimized number of trees, max depth, and split criteria
- Reduced overfitting and improved generalization

### 📈 Result
Improved model accuracy and performance on unseen data.

---

## 11. Model Deployment

The trained model is deployed using a Flask web application.

- Model saved using `joblib`
- Flask handles user input and prediction
- Users enter financial details through a web form

### 📊 Output
- 🟢 Low Risk → Customer is likely to repay  
- 🔴 High Risk → Customer may default
  
---

## 12.🧾 Conclusion

This project successfully developed a machine learning model to predict credit card default risk using real-world financial data.

---

### 🔍 Key Findings
- Customers with delayed repayment history are more likely to default  
- Higher outstanding bills increase default risk  
- Payment behavior is a strong indicator of future default  

---

### 🏆 Best Model
- **Random Forest** provided the most accurate and reliable predictions for this dataset.

---

### 💼 Practical Applications
- Helps banks identify high-risk customers  
- Supports credit approval decisions  
- Reduces financial losses  

---

### 🚀 Future Improvements
- Use larger and more recent financial datasets  
- Apply advanced models like deep learning  
- Improve feature selection and engineering  
- Deploy as a scalable production web application  
