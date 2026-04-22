import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_excel(
    r"E:\mini project new\default of credit card clients.xls",
    header=1,
    engine="xlrd"
)

print(df.head())


print(df.shape)
print(df.info())
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Drop ID column (not useful)
df.drop('ID', axis=1, inplace=True)

# Check duplicates
df.drop_duplicates(inplace=True)

#Rename Column
df.rename(columns={'default payment next month': 'default'}, inplace=True)


#TOTAL_PAY = total money paid by customer (last 6 months)
df['TOTAL_PAY'] = df[['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']].sum(axis=1)


# ==========================================
# 🔥 SIMPLE & VARIED VISUALIZATION SECTION
# ==========================================

sns.set_style("whitegrid")

# 1. Who defaults? (Count Plot)
plt.figure(figsize=(5,3))
sns.countplot(x='default', data=df)
plt.title("Default vs Non-Default")
plt.show()


# 2. Age Distribution (Histogram + KDE)
plt.figure(figsize=(5,3))
sns.histplot(df['AGE'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()


# 3. Credit Limit vs Default (Box Plot - keep only ONE)
plt.figure(figsize=(5,3))
sns.boxplot(x='default', y='LIMIT_BAL', data=df)
plt.title("Credit Limit vs Default")
plt.show()


# 4. Payment Delay Impact (Bar Plot - MOST IMPORTANT)
plt.figure(figsize=(5,3))
sns.barplot(x='PAY_0', y='default', data=df)
plt.title("Payment Delay vs Default")
plt.show()


# 5. Total Payment Distribution (KDE Plot)
plt.figure(figsize=(5,3))
sns.kdeplot(df['TOTAL_PAY'], fill=True)
plt.title("Total Payment Distribution")
plt.show()


# 6. Correlation (Heatmap - simple)
plt.figure(figsize=(6,4))
sns.heatmap(df[['LIMIT_BAL','AGE','PAY_0','TOTAL_PAY','default']].corr(),
            annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


X = df[['LIMIT_BAL', 'AGE', 'PAY_0', 'BILL_AMT1', 'PAY_AMT1', 'TOTAL_PAY']]
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
rf = RandomForestClassifier(class_weight='balanced')
lr = LogisticRegression(max_iter=1000)

# Train
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))


# Random Forest
rf_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# Confusion Matrix

sns.heatmap(
    confusion_matrix(y_test, rf_pred),
    annot=True,
    fmt='d',
    cmap='Blues'
)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(class_weight='balanced'),
    params,
    cv=3
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)


import joblib
joblib.dump(scaler, "scaler.pkl")
joblib.dump(best_model, "model.pkl")

