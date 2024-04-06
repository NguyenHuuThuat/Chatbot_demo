import nltk
from underthesea import word_tokenize
import tensorflow as tf
import keras
import json
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from gensim.models.doc2vec import TaggedDocument
import re
import matplotlib.pyplot as plt
import pickle
import random
# nltk.download()

MAX_SEQUENCE_LENGTH = 43

# Khai báo từ điển để chuyển tiếng việt CÓ dấu về tiếng việt KHÔNG dấu 
patterns = {
    '[àáảãạăắằẵặẳâầấậẫẩ]': 'a',
    '[đ]': 'd',
    '[èéẻẽẹêềếểễệ]': 'e',
    '[ìíỉĩị]': 'i',
    '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
    '[ùúủũụưừứửữự]': 'u',
    '[ỳýỷỹỵ]': 'y'
}
# Khai báo các ký tự đặc biệt để loại bỏ
special_character = r'[!“”"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]'

# Khai báo danh sách các từ cần loại bỏ
stop_words = ['bạn', 'ban', 'anh', 'chị', 'chi', 'em', 'shop', 'bot', 'ad', 'là', 'có', 'và', 'hoặc', 'nếu', 'vậy', 'thế', 'còn']

with open(f'data/chatbot_dataset_CMU.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

classes = pickle.load(open(f"model/classes.pkl", "rb"))

model = tf.keras.models.load_model(f'model/model.h5')

with open(f"model/tokenizer.pkl", 'rb') as handle:
    tokenizer = pickle.load(handle)


# Hàm làm sạch các ký tự, chuyển thành tiếng việt không dấu và chuyển chữ thành chữ thường
def convert_to_no_accents(text):
    output = text
    for regex, replace in patterns.items():
        output = re.sub(regex, replace, output)
        output = re.sub(regex.upper(), replace.upper(), output)
        output = re.sub(special_character, '', output)
        output = output.lower()
    return output


def response(tag):
    for i in intents['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])


def classify(sentence):
    message = convert_to_no_accents(sentence)
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=44, dtype='int32', value=0)
    pred = model.predict(padded)
    tag = classes[np.argmax(pred)]
    result = response(tag)
    return tag, result


if __name__=='__main__':
    sentence = 'lịch thi'
    result = classify(sentence)
    print(result)