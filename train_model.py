import pandas as pd

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

import matplotlib.pyplot as plt
#Histogram
df['AGE'].hist()
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

import seaborn as sns

#Box plot
sns.boxplot(x=df['LIMIT_BAL'])
plt.title("Credit Limit Outliers")
plt.show()

#Heat map
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()


#Pair Plot
sns.pairplot(df[['LIMIT_BAL','AGE','default']])
plt.show()

#Count Plot

sns.countplot(x='default', data=df)
plt.title("Default vs Non-Default")
plt.show()

#Bar Plot
sns.barplot(x='PAY_0', y='default', data=df)
plt.title("Repayment Status vs Default")
plt.show()


#Pie chart
df['default'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Default Distribution")
plt.ylabel('')
plt.show()

# Line Plot
df[['BILL_AMT1','BILL_AMT2','BILL_AMT3']].mean().plot()
plt.title("Billing Trend Over Months")
plt.show()

#Kde 
sns.kdeplot(df['LIMIT_BAL'])
plt.title("Density of Credit Limit")
plt.show()


#TOTAL_PAY = total money paid by customer (last 6 months)
df['TOTAL_PAY'] = df[['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']].sum(axis=1)


from sklearn.preprocessing import StandardScaler

X = df[['LIMIT_BAL', 'AGE', 'PAY_0', 'BILL_AMT1', 'PAY_AMT1', 'TOTAL_PAY']]
y = df['default']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
rf = RandomForestClassifier(class_weight='balanced')
lr = LogisticRegression(max_iter=1000)

# Train
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Random Forest
rf_pred = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True)
plt.show()


from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10]
}

grid = GridSearchCV(RandomForestClassifier(), params, cv=3)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)


import joblib
joblib.dump(scaler, "scaler.pkl")
joblib.dump(best_model, "model.pkl")

