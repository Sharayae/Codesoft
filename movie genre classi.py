import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
train_path = "Genre Classification Dataset/train_data.txt"
train_data = pd.read_csv(train_path, sep = ':::', names = ['Title', 'Genre', 'Description'], engine = 'python')
train_data.head()
test_path = "Genre Classification Dataset/test_data.txt"
test_data = pd.read_csv(test_path, sep = ':::', names = ['Id', 'Title', 'Description'], engine = 'python')
test_data.head()
stemmer = LancasterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub('-', ' ', text.lower())
    text = re.sub(f'[{string.digits}]', ' ', text)
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'https\S+', '', text)
    text = re.sub(r'pic.\S+', '', text)
    text = re.sub(r"[^a-zA-Z+']", ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text+' ')
    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.tokenize.word_tokenize(text, language = 'english', preserve_line = True)
    stopwords = nltk.corpus.stopwords.words('english')
    text = " ".join([i for i in words if i not in stopwords and len(i)>2])
    text = re.sub("\s[\s]+", ' ', text).strip()
    return re.sub(f'[{re.escape(string.punctuation)}]', '', text)

input_text = "Certainly you get a dramatic boost from hello bye the the hi -iv iem-k q934*2yee !*3 2e38"
print(f'Original text : {input_text}')
print(f'Cleaned text : {clean_text(input_text)}')

train_data['Text_cleaning'] = train_data['Description'].apply(clean_text)
test_data['Text_cleaning'] = test_data['Description'].apply(clean_text)
plt.figure(figsize = (12, 6))

plt.subplot(1, 2, 1)
original_length = train_data['Description'].apply(len)
plt.hist(original_length, bins = range(0, max(original_length) + 100, 100), color = 'blue', alpha = 0.7)
plt.title('Original Text Length')
plt.xlabel('Text Length')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
cleaned_length = train_data['Text_cleaning'].apply(len)
plt.hist(cleaned_length, bins = range(0, max(cleaned_length) + 100, 100), color = 'green', alpha = 0.7)
plt.title('Cleaned Text Length')
plt.xlabel('Text Length')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
(train_data['len_clean_text'] > 2000).value_counts()

print('Dataframe size (before removal): ', len(train_data))
filt = train_data['len_clean_text'] > 2000
train_data.drop(train_data[filt].index, axis = 0, inplace = True)
print('Dataframe size (after removal): ', len(train_data))
print(f'Removed rows: {filt.sum()}')

plt.figure(figsize = (12,5))
sns.barplot(x = 'Genre' ,y = 'len_clean_text' ,data = train_data) 
plt.xticks(rotation = 60)
plt.show()
num_words = 50000
max_len = 250
tokenizer = Tokenizer(num_words = num_words, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower = True)
tokenizer.fit_on_texts(train_data['Text_cleaning'].values)
test_path = "Genre classification Dataset/test_data_solution.txt"
test_data_sol = pd.read_csv(test_path, sep = ':::', engine = 'python', names = ['ID', 'Title', 'Genre', 'Description'])
test_data_sol.head()
X = tokenizer.texts_to_sequences(train_data['Text_cleaning'].values)
X = pad_sequences(X, maxlen = max_len)
y = pd.get_dummies(train_data['Genre']).values

X_test = tokenizer.texts_to_sequences(test_data['Text_cleaning'].values)
X_test = pad_sequences(X, maxlen = max_len)
y_test = pd.get_dummies(test_data_sol['Genre']).values
tfidf_vectorizer = TfidfVectorizer()
X_train = tfidf_vectorizer.fit_transform(train_data['Text_cleaning'])
X_test = tfidf_vectorizer.transform(test_data['Text_cleaning'])

EMBEDDING_DIM = 100
model = Sequential()
model.add(Embedding(num_words, EMBEDDING_DIM, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout = 0.1, recurrent_dropout = 0.2))
model.add(Dense(27, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) 

X = X_train
y = train_data['Genre']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)
1. Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
MultinomialNB()
y_pred_nb = nb_classifier.predict(X_val)
acc = accuracy_score(y_val, y_pred_nb)
print("Accuracy (Naive Bayes): ", acc)

print("Classification report: \n", classification_report(y_val, y_pred_nb))

_pred_lr = lr_classifier.predict(X_val)
acc = accuracy_score(y_val, y_pred_lr)
print("Accuracy (Logistic Regression): ", acc)

print("Classification report: \n", classification_report(y_val, y_pred_lr))
