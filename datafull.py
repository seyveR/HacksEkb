from gensim.models import Word2Vec
import numpy as np 
import pandas as pd 
np.set_printoptions(precision=8, suppress=True)
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from xgboost import XGBClassifier

import pymorphy2
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('dataRed.csv', sep=';')
df2 = pd.read_csv('output.csv', sep=';')
df_test = pd.read_csv('arm.csv')

df = df.dropna(how='all', axis=1)
df = df.rename(columns={'responsibilities(Должностные обязанности)': 'responsibilities'})
df = df.rename(columns={'requirements(Требования к соискателю)': 'requirements'})
df = df.rename(columns={'terms(Условия)'	: 'terms'})
df = df.rename(columns={'notes(Примечания)': 'notes'})

df_test['requirements'] = np.nan
df_test['terms'] = np.nan
df_test['notes'] = np.nan

df_test['requirements'].fillna(0, inplace=True)
df_test['responsibilities'].fillna(0, inplace=True)
df_test['terms'].fillna(0, inplace=True)
df_test['notes'].fillna(0, inplace=True)



data2 = []

row_data = []
for row in df2.values:
    for cell in row:
        if isinstance(cell, str):  # Проверка, что ячейка содержит строку
            words = nltk.word_tokenize(cell)  # Разбиваем строку на отдельные слова
            row_data.extend(words)  # Добавляем слова в список row_data

        # Проверяем, есть ли точка в списке слов, что означает окончание предложения
        if "." in words:
            data2.append(row_data)  # Добавляем список слов в общий список
            row_data = []  # Очищаем список слов для следующего предложения

# Добавляем последнее предложение, если оно не было добавлено
if row_data:
    data2.append(row_data)


df['requirements'].fillna(0, inplace=True)
df['terms'].fillna(0, inplace=True)
df['notes'].fillna(0, inplace=True)

morph = pymorphy2.MorphAnalyzer()

stopwords_rus = stopwords.words('russian')

def preprocess_text(text):
    if isinstance(text, str):
        # Удаление пунктуации
        text = re.sub(r'[^\w\s]', '', text)
        # Приведение к нижнему регистру
        text = text.lower()
        # Разбиение на слова
        words = nltk.word_tokenize(text)
        # Удаление стоп-слов и лемматизация
        words = [morph.parse(word)[0].normal_form for word in words if word not in stopwords_rus]
        text = ' '.join(words)
    return text

df['responsibilities'] = df['responsibilities'].apply(preprocess_text)
df['requirements'] = df['requirements'].apply(preprocess_text)
df['terms'] = df['terms'].apply(preprocess_text)
df['notes'] = df['notes'].apply(preprocess_text)

df_test['responsibilities'] = df_test['responsibilities'].apply(preprocess_text)
df_test['requirements'] = df_test['requirements'].apply(preprocess_text)
df_test['terms'] = df_test['terms'].apply(preprocess_text)
df_test['notes'] = df_test['notes'].apply(preprocess_text)

print(df)
print(df_test)

tfidf_vectorizer = TfidfVectorizer(max_features=300)
vectorized_data = tfidf_vectorizer.fit_transform(df['responsibilities'].values.astype('U'))

data = []
labels = []

for column in ['responsibilities', 'requirements', 'terms', 'notes']:
    for index, row in df.iterrows():
        text = row[column]
        if isinstance(text, str):
            vector = tfidf_vectorizer.transform([text]).toarray()[0]
            data.append(vector)
            labels.append(column)

# Разделение данных на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

xgb_clf = XGBClassifier(
    n_estimators=100,
    use_label_encoder=False,  # чтобы избежать предупреждения
    eval_metric='mlogloss',  # чтобы избежать предупреждения
    random_state=42
)

param_grid = {
    'max_depth': [3, 6, 9, 12],  
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': [0.5, 1.0],  
    'colsample_bytree': [0.5, 1.0]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit model
grid_search.fit(X_train, y_train_encoded)

# Get best parameters
print(grid_search.best_params_)

# Use best estimator to predict
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

# Print report (use inverse_transform to get original labels)
print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))

# Apply classifier to test data
df_test["responsibilities_vector"] = df_test["responsibilities"].apply(
    lambda x: tfidf_vectorizer.transform([x]).toarray()[0] if isinstance(x, str) else np.zeros(300)
)

df_test["predicted_category"] = le.inverse_transform(best_clf.predict(list(df_test["responsibilities_vector"])))

for index, row in df_test.iterrows():
    category = row["predicted_category"]
    df_test.loc[index, category] = row["responsibilities"]

df_test = df_test.drop(columns=["responsibilities_vector", "predicted_category"])

df_test['terms'].to_csv('terms.csv', index=False)
df_test['requirements'].to_csv('requirements.csv', index=False)
df_test['notes'].to_csv('notes.csv', index=False)