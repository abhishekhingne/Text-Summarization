import nltk
import re


class preprocessing():

    def read_data(self, path):
        return nltk.line_tokenize(nltk.load(path))


    def clean_string(self, string):
        self.decontracted_string(string)
        string = string.strip()
        string = string.lower()
        string = re.sub(r"[#.*]", "#", string)
        string = re.sub(r"-", "", string)
        string = re.sub(r" '", "", string)
        string = re.sub(r" [!@$%^&*().,?]", "", string)
        string = re.sub(r"[!@$%^&*().,?]", "", string)
        string = re.sub(r'"', "", string)
        string = re.sub(r' "', "", string)
        string = re.sub(r'" ', "", string)
        string = string.strip()
        return string

    def decontracted_string(self, string):
        string = re.sub(r"won\'t", "will not", string)
        string = re.sub(r"can\'t", "can not", string)
        string = re.sub(r"ain\'t", "is not", string)
        string = re.sub(r"n\'t", " not", string)
        string = re.sub(r"\'re", " are", string)
        string = re.sub(r"\'s", " is", string)
        string = re.sub(r"\'d", " would", string)
        string = re.sub(r"\'ll", " will", string)
        string = re.sub(r"\'t", " not", string)
        string = re.sub(r"\'ve", " have", string)
        string = re.sub(r"\'m", " am", string)
        return string
