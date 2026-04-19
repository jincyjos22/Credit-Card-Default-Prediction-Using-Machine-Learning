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

---

### 📌 Summary

Data preprocessing improved the quality of the dataset by handling inconsistencies, reducing noise, and preparing the data for effective model training.


## 5. Exploratory Data Analysis (EDA) – Insights

Exploratory Data Analysis (EDA) was performed to understand the structure of the dataset, identify patterns, and explore relationships between features and the target variable.

---

### 🔍 Understanding Data Distribution

* **Histogram:**
  Histograms were used to analyze the distribution of numerical features such as **AGE** and **LIMIT_BAL**.
  It was observed that most customers fall within a moderate age group and lower credit limit range, indicating a **right-skewed distribution**.

* **Kernel Density Estimation (KDE):**
  KDE plots helped visualize the probability density of features.
  These plots confirmed that many financial variables are skewed, especially billing and payment amounts.

---

### 📊 Detecting Outliers

* **Box Plot:**
  Boxplots were used to detect outliers in features like **LIMIT_BAL** and **BILL_AMT** variables.
  Significant outliers were observed, indicating the presence of customers with extremely high credit limits or bill amounts.

---

### 🔗 Relationship Between Features

* **Heatmap (Correlation Matrix):**
  Heatmap analysis showed strong correlations between repayment status variables (**PAY_0 to PAY_6**) and the target variable.
  This indicates that **payment history is the most influential factor in predicting default**.

* **Pair Plot:**
  Pair plots were used to visualize relationships between selected numerical features.
  They revealed patterns and clusters, helping to understand how variables interact with each other.

---

### 📊 Categorical Analysis

* **Count Plot:**
  Count plots were used to compare the number of defaulters vs non-defaulters.
  It was observed that the dataset is **imbalanced**, with more non-defaulters than defaulters.

* **Bar Plot:**
  Bar plots were used to analyze relationships such as repayment status vs default.
  Customers with delayed payments showed a significantly higher default rate.

* **Pie Chart:**
  Pie charts were used to represent proportions of categorical variables such as gender or default status, giving a clear view of class distribution.

---

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

---

### 📌 Summary

EDA provided valuable insights into customer behavior, data distribution, and feature importance. These findings helped guide preprocessing, feature engineering, and model selection for better prediction performance.
















 
⚙️ Technologies Used

Python 🐍

Pandas

NumPy

Matplotlib & Seaborn

Scikit-learn

Flask (for deployment)

🔧 Data Preprocessing

Handled missing values

Removed duplicates

Treated outliers using boxplots

Encoded categorical variables

Ensured data consistency

📈 Exploratory Data Analysis
