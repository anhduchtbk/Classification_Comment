import re
import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout

from sklearn.metrics import confusion_matrix,f1_score, precision_score,recall_score
import seaborn as sns
import matplotlib.pyplot as plt

#lay du lieu tu file csv bang Pandas
df = pd.read_csv('comments.csv')


df = df.dropna() #Loai bo cac phan tu empty trong du lieu
df = df.drop_duplicates() #Loai bo cac phan tu bi trung nhau trong tap du lieu

#Tien xu ly du lieu 
#Xoa link 
def remove_hyperlink(text):
    return re.sub(r"http\S+","",text)

#Chuyen ve chu cai viet thuong
def to_lower(text):
    return text.lower()

#Loai bo cac chu so 
def remove_number(text):
    return re.sub(r'\d+',"", text)

#Loai bo dau cham cau:
def remove_punctuation(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    return text 

#Loai bo khoang trang hai ben van ban
def remove_whitespace(text):
    text = re.sub(r'\s+',' ', text)
    text.strip()
    return text 

#Loai bo dau xuong dong
def replace_newline(text):
    return text.replace('\n',' ')

#Loai bo emoji 
def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r' ', text)

#loai bo cac ky tu khong co y nghia
def remove_unmeaning_word(text):
    text = re.sub(r'=[)]+', '', text) #Loai bo ky tu =)))
    text = re.sub(r':[D]+','', text) #loai bo ky tu :D :DDDDD
    return text

#Thay the cac tu viet tat trong tieng Viet
def replace_acronym(text):
    text = re.sub(r'\s+k\s+|\s+K\s+|^k\s+|^K\s+|\s+k$|\s+K$|\s+ko\s+|\s+Ko\s+|^ko\s+|^Ko\s+|\s+ko$|\s+Ko$'
                  ,' không ', text)
    text = re.sub(r'\s+đc\s+|\s+Đc\s+|^đc\s+|^Đc\s+|\s+đc$|\s+Đc$', ' được ', text)
    text = re.sub(r'\s+nhg\s+|\s+Nhg\s+|^nhg\s+|^Nhg\s+|\s+nhg$|\s+Nhg$', ' nhưng ', text)
    text = re.sub(r'\s+mk\s+|\s+Mk\s+|^mk\s+|^Mk\s+|\s+mk$|\s+Mk$', ' mình ', text)
    return text


#Tong hop cac ham lai de lam sach du lieu
def clean_up_pipeline(sentence):
    cleaning_utils = [remove_hyperlink,
                      to_lower,
                      remove_number,
                      remove_punctuation,
                    remove_emoji,
                      replace_acronym,
                    remove_whitespace,
                    replace_newline]
    for o in cleaning_utils:
        sentence = o(sentence)
    return sentence

#Chia ra train va test voi ti le 0.2
comments_train, comments_test, label_train, label_test = train_test_split(df['text'], df['label'], test_size=0.2)

#Preprocessing data 
x_train = [clean_up_pipeline(o) for o in comments_train]
x_test = [clean_up_pipeline(o) for o in comments_test]

y_train = label_train.values #numpy array
y_test = label_test.values

#Tokenizing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)

# print(tokenizer.word_index)

x_train_features = np.array(tokenizer.texts_to_sequences(x_train))
x_test_features = np.array(tokenizer.texts_to_sequences(x_test))

# print(x_test_features)
x_train_features = pad_sequences(x_train_features, maxlen=50)
x_test_features = pad_sequences(x_test_features, maxlen=50)
# print(x_test_features)

#Model
def LMTS(input_length, input_dim, x_train, x_test, y_train, y_test):
    lstm_model = Sequential()
    #Creating an embedding layer to vectorize
    lstm_model.add(Embedding(input_dim=input_dim+1, output_dim=20, input_length=input_length))
    #Addding LSTM
    lstm_model.add(LSTM(64))
    # Relu allows converging quickly and allows backpropagation
    lstm_model.add(Dense(16, activation='relu'))
    #Deep Learninng models can be overfit easily, to avoid this, we add randomization using drop out
    lstm_model.add(Dropout(0.1))
    # Adding sigmoid activation function to normalize the output
    lstm_model.add(Dense(1, activation='sigmoid'))

    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstm_model.summary()
    history = lstm_model.fit(x_train, y_train, epochs=20, batch_size=32, shuffle = True,
                        validation_data=(x_test, y_test))
    #Save the model
    lstm_model.save('lstm_model.h5')
    y_predict = [1 if o>0.5 else 0 for o in lstm_model.predict(x_test)]
    return history, y_predict

#Danh gia 
def evaluating(test_y, y_predict):
    cf_matrix =confusion_matrix(test_y,y_predict)
    print("Precision: {:.2f}%".format(100 * precision_score(test_y, y_predict)))
    print("Recall: {:.2f}%".format(100 * recall_score(test_y, y_predict)))
    print("F1 Score: {:.2f}%".format(100 * f1_score(test_y,y_predict)))
    ax= plt.subplot()
    #annot=True to annotate cells
    sns.heatmap(cf_matrix, annot=True, ax = ax,cmap='Blues',fmt='')
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Positive', 'Negative']); ax.yaxis.set_ticklabels(['Positive', 'Negative'])
    plt.show()
def get_predict(cmts):
    # tokenizer = Tokenizer()
    my_model = tf.keras.models.load_model('/Users/ducanh/Desktop/HySpace/HySpace_gitlab/Untitled/Classification_Comment/lstm_model.h5')
    cmts = np.array(cmts)
    cmts = tokenizer.texts_to_sequences(cmts)
    cmts = pad_sequences(cmts, maxlen=50)
    result = my_model.predict(cmts)
    return result

def read_from_file():
    # try:
        file = open("/Users/ducanh/Desktop/HySpace/HySpace_gitlab/Untitled/Classification_Comment/text.txt", "r", encoding='utf-8')
        cmts = file.readlines()
        for i in range(len(cmts)):
            cmts[i] = clean_up_pipeline(cmts[i]).strip()
        list_result = get_predict(cmts)
        print(list_result)
        for i in list_result:
            if(i > 0.5):
                print("Positive")
            else:
                print("Negative")
    # except:
    #     print('File not found')
if __name__ == '__main__':
    lmts, y_predict = LMTS(50,1557, x_train_features, x_test_features, y_train, y_test)
    evaluating(y_test, y_predict)
    read_from_file()

    
