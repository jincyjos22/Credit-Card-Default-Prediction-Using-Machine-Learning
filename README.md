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


Machine Learning based Credit Card Default Prediction

# 1. Introduction 

Credit card default is one of the most common problems among banks and other financial institutions. This problem results in financial losses and raises credit risk. Prediction of the credit card default by its client depends on various factors such as customer payment history, the limit of his credit card, demographics information, and other aspects of his financial behavior.

Standard approaches do not allow predicting such behavior accurately due to its complex nature. That is why there should be an efficient machine learning model which would analyze the historical data and provide valuable predictions.

This prediction will allow organizations to identify potential defaulters in time and prevent their further problematic actions.


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

Data preprocessing is a crucial step in the machine learning pipeline, as it ensures the dataset is clean, consistent, and suitable for building accurate models.

---

### ● Handling Missing Values

The dataset was examined for missing values using appropriate functions.
Since the dataset contained very few or no missing values, numerical features were handled using **median imputation**, which is robust to outliers and prevents distortion of data distribution.

---

### ● Outlier Detection and Treatment

Outliers were identified using **boxplots** for numerical features such as **LIMIT_BAL** and bill amounts.
Extreme values were carefully analyzed and treated where necessary to reduce their impact on model performance, especially for algorithms sensitive to large variations.

---

### ● Removing Duplicate Records

Duplicate entries were checked and removed using data cleaning techniques to ensure data integrity and avoid bias in model training.

---

### ● Encoding Categorical Variables

Categorical features such as **SEX, EDUCATION, and MARRIAGE** were already represented in numerical format within the dataset.
Therefore, no additional encoding techniques were required.

---

### ● Handling Skewed Data

Some numerical features, such as credit limit and billing amounts, exhibited skewed distributions.
Although tree-based models are less sensitive to skewness, transformations and scaling techniques were considered to improve overall model performance.

---

### ● Feature Engineering

A new feature called **TOTAL_PAY** was created by summing all payment amount variables (**PAY_AMT1 to PAY_AMT6**).
This helps capture overall payment behavior of customers more effectively.

---

### ● Feature Scaling

Feature scaling was applied using **StandardScaler** to normalize numerical features.
This ensures that all variables are on a similar scale, which improves the performance of models like KNN and SVM.



## 5. Exploratory Data Analysis (EDA) – Insights

Exploratory Data Analysis (EDA) was performed to understand the structure of the dataset, identify patterns, and explore relationships between features and the target variable.

---

* **Histogram:**
Histograms were used to analyze the distribution of numerical features such as **AGE** and **LIMIT_BAL**.
The plots show that most customers fall within a moderate age range and lower credit limit values.
These visualizations help in understanding how the data is distributed and identifying general patterns in the dataset.
<img width="640" height="480" alt="Hist" src="https://github.com/user-attachments/assets/86004128-7915-4472-b5c3-6b3ab7f1c245" />


### 📊 Detecting Outliers

* **Box Plot:**
  Boxplots were used to detect outliers in features like **LIMIT_BAL** and **BILL_AMT** variables.
  Significant outliers were observed, indicating the presence of customers with extremely high credit limits or bill amounts.
<img width="640" height="480" alt="Boxplot" src="https://github.com/user-attachments/assets/5a26d8d9-95f6-41cd-bd35-65fc56692722" />

---

### 🔗 Relationship Between Features

* **Heatmap (Correlation Matrix):**
  Heatmap analysis showed strong correlations between repayment status variables (**PAY_0 to PAY_6**) and the target variable.
  This indicates that **payment history is the most influential factor in predicting default**.
<img width="1536" height="754" alt="Heatmap" src="https://github.com/user-attachments/assets/1b772311-ad2b-4ade-9438-5c336c63f695" />


### 📊 Categorical Analysis

* **Count Plot:**
  Count plots were used to compare the number of defaulters vs non-defaulters.
  It was observed that the dataset is **imbalanced**, with more non-defaulters than defaulters.

* **Bar Plot:**
  Bar plots were used to analyze relationships such as repayment status vs default.
  Customers with delayed payments showed a significantly higher default rate.

* **Pie Chart:**
  Pie charts were used to represent proportions of categorical variables such as gender or default status, giving a clear view of class distribution.



### 📈 Trend Analysis

* **Line Plot:**
  Line plots were used to observe trends across billing and payment amounts over time (months).
  These plots helped identify consistent payment behavior patterns among customers.

---

### 📌 Key Insights

* Most customers are **non-defaulters**, indicating class imbalance
* **Repayment history (PAY variables)** is the most important predictor of default
* Customers with **higher credit limits** tend to have lower default risk
* **Higher bill amounts and lower payments** increase the probability of default
* Presence of **outliers** in financial features
* Many numerical features are **skewed**, especially billing and payment data


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

## 8. Model Evaluation

Models were evaluated based on their ability to correctly identify customers who are likely to default.

Evaluation Metrics Used
Accuracy – overall correctness of predictions
Precision – correctness of predicted defaulters
Recall – ability to identify actual defaulters
F1-Score – balance between precision and recall
Confusion Matrix – detailed breakdown of predictions
ROC-AUC Score – model’s ability to distinguish between default and non-default
Dataset-Specific Insight
The dataset is slightly imbalanced (more non-defaulters than defaulters)
Therefore, Recall is more important than Accuracy, as missing a defaulter can lead to financial loss
Best Model

Random Forest performed best because:

It handles complex financial relationships effectively
It reduces overfitting
It provides consistent performance across all metrics
Conclusion

Model evaluation helped identify the most reliable algorithm for predicting credit card default risk.

## 10. Hyperparameter Tuning

Hyperparameter tuning was performed to further improve the performance of the selected model.

Approach
Used systematic search methods to find the best parameter values
Focused on optimizing the Random Forest model
Parameters Optimized
Number of trees in the forest
Maximum depth of trees
Minimum samples required to split nodes
Impact on Dataset
Improved prediction accuracy for default cases
Reduced overfitting on training data
Enhanced performance on unseen customer data
Conclusion

Tuning improved the model’s ability to generalize and make accurate predictions on real-world financial data.

## 11. Model Deployment

The trained model was deployed using a simple web-based interface to simulate real-world usage.

Implementation
Model was saved after training
A user interface was created using Flask
Users can input customer financial details
Functionality Based on Dataset
Accepts inputs such as:
Credit limit
Age
Repayment history
Bill and payment amounts
Predicts whether the customer is likely to default
Output
High Risk → Likely to default
Low Risk → Not likely to default
Conclusion

Deployment demonstrates how the model can assist financial institutions in identifying high-risk customers in real time.

## 12. Conclusion

This project successfully developed a machine learning model to predict credit card default risk using real-world financial data.

Key Findings from Dataset
Customers with delayed repayment history are more likely to default
Higher outstanding bills increase default risk
Payment behavior is a strong indicator of future default
Best Model

Random Forest provided the most accurate and reliable predictions for this dataset.

Practical Applications
Helps banks identify risky customers
Supports credit approval decisions
Reduces financial losses
Future Improvements
Use larger and updated financial datasets
Apply advanced models such as deep learning
Improve feature selection and engineering
Deploy as a scalable web application

