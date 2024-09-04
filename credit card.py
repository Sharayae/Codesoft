import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
%matplotlib inline
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')
train = pd.read_csv("fraudTrain.csv")
test = pd.read_csv("fraudTest.csv")
train.head()
train.describe()
test.describe()
sns.heatmap(train.isnull())
plt.figure(figsize = (8, 6))
sns.countplot(x = 'is_fraud', data = pd.concat([train, test]))
plt.title("Data Distribution")
plt.xlabel(' (0: Not Fraud | 1: Fraud) ')
plt.ylabel('Count')
plt.show()
def clean(data):
    data.drop(["Unnamed: 0",'cc_num','first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num','trans_date_trans_time'], axis = 1, inplace = True)
    data.dropna()
    return data
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
def encode_data(data):
    data['merchant'] = encoder.fit_transform(data['merchant'])
    data['category'] = encoder.fit_transform(data['category'])
    data['gender'] = encoder.fit_transform(data['gender'])
    data['job'] = encoder.fit_transform(data['job'])
    return data
encode_data(train)
counts = train['is_fraud'].value_counts()
plt.figure(figsize = (12, 6))
plt.pie(counts, labels = ['No', 'Yes'], autopct = '%0.0f%%')
plt.title('is_fraud counts')
plt.tight_layout()
plt.show()
corr = train.corr()
sns.heatmap(corr, annot = True, cmap = 'coolwarm', fmt = '.2f')
X_train = train.drop('is_fraud', axis = 1)
X_test = test.drop('is_fraud', axis = 1)
y_train = train['is_fraud']
y_test = test['is_fraud']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

col = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracy = []
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
LogisticRegression()

y_pred_lr = lr.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(y_test, y_pred_lr)
print("Accuracy (Logistic Regression): ", acc)
accuracy.append(acc)

cm = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm, annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.title('Logistic Regression')
plt.show()
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
DecisionTreeClassifier()

y_pred_dtc = dtc.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(y_test, y_pred_dtc)
print("Accuracy (Decision Tree): ", acc)
accuracy.append(acc)

cm = confusion_matrix(y_test, y_pred_dtc)
sns.heatmap(cm, annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.title('Decision Tree')
plt.show()
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
RandomForestClassifier()

y_pred_rfc = rfc.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(y_test, y_pred_rfc)
print("Accuracy (Random Forest): ", acc)
accuracy.append(acc)

cm = confusion_matrix(y_test, y_pred_rfc)
sns.heatmap(cm, annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.title('Random Forest')
plt.show()
accuracy

FinalResult = pd.DataFrame({'Algorithms': col, 'Accuracy': accuracy})
fig, ax = plt.subplots(figsize = (20, 5))
plt.plot(FinalResult.Algorithms, accuracy, label = 'Accuracy')
plt.legend()
plt.show()
