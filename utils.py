import numpy as np


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
