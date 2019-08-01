import numpy as np
from config import *


def load_glove_embeddings(glove_path):
    file = open(glove_path, "r")
    lines = file.readlines()
    glove_embedding_dict = dict()
    for line in lines:
        split_line = line.replace("\n", "").split()
        word = split_line[0]
        embedding = np.array(split_line[1:]).astype(np.float)
        glove_embedding_dict[word] = embedding
    return glove_embedding_dict


def init_glove_embedding(reversed_word_dict, emb_size=300):
    glove_emd = load_glove_embeddings(GLOVE_EMBEDDING_PATH)
    word_emd_list = []
    for _, word in sorted(reversed_word_dict.items()):
        try:
            emd = glove_emd.get(word)
            if emd is None:
                emd = np.zeros([emb_size])
        except KeyError:
            emd = np.zeros([emb_size])
        word_emd_list.append(emd.astype('float32'))
    word_emd_list[2] = np.random.normal(0, 1, emb_size).astype('float32')
    word_emd_list[3] = np.random.normal(0, 1, emb_size).astype('float32')
    return np.array(word_emd_list)
