import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from train import clean_up_pipeline, tokenizer

def get_predict(cmts):
    # tokenizer = Tokenizer()
    my_model = keras.models.load_model('/Users/ducanh/Desktop/HySpace/HySpace_gitlab/Untitled/Classification_Comment/lstm_model.h5')
    cmts = np.array(cmts)
    cmts = tokenizer.texts_to_sequences(cmts)
    cmts = pad_sequences(cmts, maxlen=50)
    result = my_model.predict(cmts)
    return result

def read_from_file():
    try:
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
    except:
        print('File not found')

read_from_file()