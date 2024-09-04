import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
#Loading Dataset
data = pd.read_csv("Churn_Modelling.csv")
data.head()
#Droping useless features
data = data.drop(columns = ["RowNumber", "CustomerId", "Surname"])
data.columns
#One hot encoding
data['Gender'] = data['Gender'].apply(lambda x: 0 if x == 'Female' else 1)
data['Gender'] = data['Gender'].astype(int)
data.head()
#Label Encoding
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
data['Geography'] = encoder.fit_transform(data['Geography'])
data.head()
value = data['Exited'].value_counts()
plt.pie(value, labels = ["Not Exited", "Exited"], autopct = "%1.1f%%", colors = sns.color_palette('Set3'))
plt.show()
print(value)
corr = data.corr()
plt.figure(figsize = (12, 8))
sns.heatmap(corr, annot = True, cmap = 'jet')
plt.title("Correlation matrix")
plt.show()
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

print("Class distribution before over sampling: ",Counter(y))
ros = RandomOverSampler(random_state = 42)
X, y = ros.fit_resample(X, y)
print("Class distribution after over sampling: ", Counter(y))
X = np.array(X)
X = (X-X.mean()) / X.std()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
col = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Decision Tree']
accuracy = []
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)
LogisticRegression()
y_pred_lr = lr_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred_lr)
print("Accuracy (Logistic Regression): ", acc)
accuracy.append(acc)

print("Classification report:\n", classification_report(y_test, y_pred_lr))

cm = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm, annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.title("Logistic Regression")
plt.show()
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
RandomForestClassifier()
y_pred_rf = rf_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred_rf)
print("Accuracy (Random Forest): ", acc)
accuracy.append(acc)

print("Classification report:\n", classification_report(y_test, y_pred_rf))

cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.title("Random Forest")
plt.show()
