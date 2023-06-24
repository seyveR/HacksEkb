import os
import pandas as pd
from django.shortcuts import render
from django.http import FileResponse
from django.conf import settings
from openpyxl import Workbook
import pandas as pd
import numpy as np
import pymorphy2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from django.http import FileResponse, HttpResponseNotFound
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from navec import Navec
# Здесь ваш код нейронной сети, который был указан выше,
# вынесен в отдельную функцию для удобства использования.

def process_dataframe(df, file_name):

    nltk.download('punkt')
    nltk.download('stopwords')
    df = pd.read_csv('dataRed.csv', sep=';',encoding='utf-8')
    # file_path = os.path.join(os.path.dirname(__file__), 'static', 'dataRed.csv')
    # df = pd.read_csv(file_path, sep=';', encoding='utf-8')
    df_test = pd.read_csv(file_name, encoding='utf-8')

    df = df.drop(columns=["specialization(специализация)", "id"])
    df_test['requirements'] = np.nan
    df_test['terms'] = np.nan
    df_test['notes'] = np.nan

    df = df.rename(columns={'responsibilities(Должностные обязанности)': 'responsibilities'})
    df = df.rename(columns={'requirements(Требования к соискателю)': 'requirements'})
    df = df.rename(columns={'terms(Условия)'	: 'terms'})
    df = df.rename(columns={'notes(Примечания)': 'notes'})
    df = df.dropna(how='all', axis=1)


    df['requirements'].fillna('', inplace=True)


    df['terms'].fillna('', inplace=True)


    df['notes'].fillna('', inplace=True)

    random_state = 42
    max_review_len = 100
    vector_size = 300

    def preprocess(text, stop_words, punctuation_marks, morph):
        tokens = word_tokenize(text.lower())
        preprocessed_text = []
        for token in tokens:
            if token not in punctuation_marks:
                lemma = morph.parse(token)[0].normal_form
                if lemma not in stop_words:
                    preprocessed_text.append(lemma)
        return preprocessed_text

    punctuation_marks = ['!', ',', '(', ')', ':', '-', '?', '.', '..', '...', '«', '»', ';', '–', '--']
    stop_words = stopwords.words("russian")
    morph = pymorphy2.MorphAnalyzer()

    df['Preprocessed_reqr'] = df.apply(lambda row: preprocess(row['requirements'], punctuation_marks, stop_words, morph), axis=1)
    df['Preprocessed_terms'] = df.apply(lambda row: preprocess(row['terms'], punctuation_marks, stop_words, morph), axis=1)
    df['Preprocessed_notes'] = df.apply(lambda row: preprocess(row['notes'], punctuation_marks, stop_words, morph), axis=1)
    df['Preprocessed_resps'] = df.apply(lambda row: preprocess(row['responsibilities'], punctuation_marks, stop_words, morph), axis=1)

    navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')

    def vectorize_text_avg(txt, navec):
        unk = navec['<unk>']
        text_embeddings = []
        for token in txt:
            embedding = navec.get(token, unk)
            text_embeddings.append(embedding)
    
        if text_embeddings: # check if list is not empty
            avg_vector = np.mean(text_embeddings, axis=0)
        else: 
            avg_vector = np.zeros(vector_size) # if text was empty, return zero vector
        return avg_vector

    df['req_vec'] = df.apply(lambda row: vectorize_text_avg(row['Preprocessed_reqr'], navec), axis=1)
    df['terms_vec'] = df.apply(lambda row: vectorize_text_avg(row['Preprocessed_terms'], navec), axis=1)
    df['notes_vec'] = df.apply(lambda row: vectorize_text_avg(row['Preprocessed_notes'], navec), axis=1)
    df['resp_vec'] = df.apply(lambda row: vectorize_text_avg(row['Preprocessed_resps'], navec), axis=1)

    df = df.drop(columns=["Preprocessed_reqr", "Preprocessed_terms", "Preprocessed_notes"])

    df_test['responsibilities'].fillna('', inplace=True)
    df_test['requirements'].fillna('', inplace=True)
    df_test['terms'].fillna('', inplace=True)
    df_test['notes'].fillna('', inplace=True)

    df_test['Preprocessed_reqr'] = df_test.apply(lambda row: preprocess(row['requirements'], punctuation_marks, stop_words, morph), axis=1)
    df_test['Preprocessed_terms'] = df_test.apply(lambda row: preprocess(row['terms'], punctuation_marks, stop_words, morph), axis=1)
    df_test['Preprocessed_notes'] = df_test.apply(lambda row: preprocess(row['notes'], punctuation_marks, stop_words, morph), axis=1)
    df_test['Preprocessed_resps'] = df_test.apply(lambda row: preprocess(row['responsibilities'], punctuation_marks, stop_words, morph), axis=1)

    df_test['req_vec'] = df_test.apply(lambda row: vectorize_text_avg(row['Preprocessed_reqr'], navec), axis=1)
    df_test['terms_vec'] = df_test.apply(lambda row: vectorize_text_avg(row['Preprocessed_terms'], navec), axis=1)
    df_test['notes_vec'] = df_test.apply(lambda row: vectorize_text_avg(row['Preprocessed_notes'], navec), axis=1)
    df_test['resp_vec'] = df_test.apply(lambda row: vectorize_text_avg(row['Preprocessed_resps'], navec), axis=1)

    df_test = df_test.drop(columns=["Preprocessed_reqr", "Preprocessed_terms", "Preprocessed_notes"])

    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    df_train = pd.DataFrame(columns=['vector', 'label'])

# добавляем вектора и метки
    for index, row in df.iterrows():
        df_train = df_train.append({'vector': row['req_vec'], 'label': ['requirements']}, ignore_index=True)
        df_train = df_train.append({'vector': row['terms_vec'], 'label': ['terms']}, ignore_index=True)
        df_train = df_train.append({'vector': row['notes_vec'], 'label': ['notes']}, ignore_index=True)

# преобразуем вектора в матрицу признаков
    X = np.stack(df_train['vector'].to_numpy())

# преобразуем метки в бинарную матрицу
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df_train['label'])

# делим на обучающую и тестовую выборку
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# обучаем модель
    model = OneVsRestClassifier(LogisticRegression(random_state=random_state))
    model.fit(X_train, y_train)

# теперь можно предсказывать категорию для векторов из df_test
    y_pred = model.predict(np.stack(df_test['resp_vec'].to_numpy()))
    predicted_labels = mlb.inverse_transform(y_pred)


    df_test['predicted_labels'] = predicted_labels
    for index, row in df_test.iterrows():
        if 'requirements' in row['predicted_labels']:
            df_test.loc[index, 'requirements'] = row['responsibilities']
        if 'terms' in row['predicted_labels']:
            df_test.loc[index, 'terms'] = row['responsibilities']
        if 'notes' in row['predicted_labels']:
            df_test.loc[index, 'notes'] = row['responsibilities']

    df_pred = df_test.drop(columns=["Preprocessed_resps", "req_vec", "terms_vec", "notes_vec", "resp_vec", "predicted_labels"])

    # df_pred['requirements'].to_csv("requirements.csv", index=False)
    # df_pred['terms'].to_csv("terms.csv", index=False)
    # df_pred['notes'].to_csv("notes.csv", index=False)
    df_pred.to_csv("newData.csv", index=False)
    

    return df_pred


def index(request):
    context = {}
    if request.method == 'POST':
        file = request.FILES['csv_file']
        file_name = file.name
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

    
        data_xls = pd.read_excel(file_path, index_col=None)
        csv_file_path = file_path.replace(".xlsx", ".csv")
        data_xls.to_csv(csv_file_path, encoding='utf-8', index=False)

        df = pd.read_csv(csv_file_path, sep=';', on_bad_lines='skip', encoding='utf-8')
        df_processed = process_dataframe(df, csv_file_path)

        csv = pd.read_csv("newData.csv")
        csv.to_excel("newData.xlsx", index=False)
  
        request.session['file_processed'] = True
        context['download_link'] = 'URL для скачивания'

        return render(request, 'website/index.html')

    return render(request, 'website/index.html', context)

from django.http import FileResponse

def download(request):
    
    if request.session.get('file_processed', False):
        file = open('newData.xlsx', 'rb')
        response = FileResponse(file)
        response['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response['Content-Disposition'] = 'attachment; filename=newData.xlsx'
        
    
        del request.session['file_processed']
        return response

    return HttpResponseNotFound()