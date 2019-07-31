from preprocessing import *
import nltk
import pickle
from config import *
from utils import *


class BuildData:

    def __init__(self):
        self.preprocess = PreProcessing()
        self.word_dict = dict()
        self.reversed_word_dict = dict()
        self.article_list = self.get_clean_data_list(ARTICLE_PATH)
        self.summary_list = self.get_clean_data_list(SUMMARY_PATH)

    def get_clean_data_list(self, path):
        lines = self.preprocess.read_data(path)
        lines = [self.preprocess.clean_string(x) for x in lines]
        return lines

    def build_dict(self):
        words = []
        for string in self.article_list + self.summary_list:
            for word in nltk.word_tokenize(string):
                words.append(word)

        self.word_dict["<padding>"] = 0
        self.word_dict["<unk>"] = 1
        self.word_dict["<s>"] = 2
        self.word_dict["</s>"] = 3
        counter = 4
        for word in set(words):
            self.word_dict[word] = counter
            counter += 1

        with open(WORD_DICT_PATH, "wb") as f:
            pickle.dump(self.word_dict, f)
        self.reversed_word_dict = dict(zip(self.word_dict.values(), self.word_dict.keys()))

    @property
    def build_dataset(self):
        x = [nltk.word_tokenize(word) for word in self.article_list]
        x = [[self.word_dict.get(w, self.word_dict["<unk>"]) for w in d] for d in x]
        x = [d[:(MAX_ARTICLE_LEN - 1)] for d in x]
        x = [d + (MAX_ARTICLE_LEN - len(d)) * [self.word_dict["<padding>"]] for d in x]

        y = [nltk.word_tokenize(word) for word in self.summary_list]
        y = [[self.word_dict.get(w, self.word_dict["<unk>"]) for w in d] for d in y]
        y = [d[:(MAX_SUMMARY_LEN - 1)] for d in y]
        return x, y

    @staticmethod
    def batch_iter(x, y, batch_size=32, num_epoch=10):
        x = np.array(x)
        y = np.array(y)
        for _ in range(num_epoch):
            for batch_num in range(0, len(x), batch_size):
                yield x[batch_num:(batch_num+batch_size)], y[batch_num:(batch_num+batch_size)]
