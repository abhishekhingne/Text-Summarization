from flask import Flask, jsonify, request
from flask_cors import CORS
from preprocessing import PreProcessing
from seq2seq import TextSummarization
from config import *
import nltk
import pickle
import numpy as np


app = Flask(__name__)
CORS(app)

preprocessing = PreProcessing()
model = TextSummarization(forward_only=True)
file = open(WORD_DICT_PATH, "rb")
word_dict = pickle.load(file, encoding="utf-8")
reversed_word_dict = dict(zip(word_dict.values(), word_dict.keys()))


@app.route('/get_summary', methods=['POST'])
def get_summary():
    data = request.get_json()
    text = data["text"]
    clean_text = preprocessing.clean_string(text)
    clean_text = [clean_text]
    x = nltk.word_tokenize(clean_text)
    x = [word_dict.get(d, word_dict["<unk>"]) for d in x]
    x = x[:(MAX_ARTICLE_LEN - 1)]
    x = [x + (MAX_ARTICLE_LEN - len(x)) * [word_dict["<padding>"]]]
    x = np.array(x)
    summary_text = model.get_summary(x, reversed_word_dict)
    return jsonify({"text": text, "summary": summary_text[0]})


if __name__ == '__main__':
    app.run(debug=True, port=8000, host="0.0.0.0")
