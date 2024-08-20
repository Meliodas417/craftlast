import numpy as np
from abc import ABC, abstractmethod

class WordEmbedding:
    def __init__(self, model_path):
        self.model_path = model_path
        self.embedding_dict = self.load_embedding()

    def load_embedding(self):
        embedding_dict = {}
        with open(self.model_path, "r", encoding="utf-8") as file:
            for line in file:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embedding_dict[word] = vector
        return embedding_dict

    def get_embedding(self, word):
        return self.embedding_dict.get(word, np.zeros(300))

class CompatibleDescriptors:
    def __init__(self, src_event, target_event):
        self.src_event = src_event
        self.target_event = target_event

class ApproachDescriptors:
    default_columns = ['src_app', 'target_app', 'src_event_index', 'target_label',
                       'src_class', 'target_class', 'src_type', 'target_type', 'target_event_index']
    atm = ['text', 'id', 'content_desc', 'hint', 'atm_neighbor', 'file_name']
    craftdroid = ['text', 'id', 'content_desc', 'hint', 'parent_text', 'sibling_text', 'activity']
    union = ['text', 'id', 'content_desc', 'hint', 'parent_text', 'sibling_text', 'activity', 'atm_neighbor',
             'file_name']
    intersection = ['text', 'id', 'content_desc', 'hint']
    descriptors_dict = {'default': default_columns, 'atm': atm, 'craftdroid': craftdroid,
                        'union': union, 'intersection': intersection}
    all = set(atm + craftdroid)

def add_src_target_string(descriptors_fields):
    src_columns = ['src_' + field for field in descriptors_fields]
    target_columns = ['target_' + field for field in descriptors_fields]
    return src_columns, target_columns

class AbstractEvaluator(ABC):
    @abstractmethod
    def get_potential_matches(self, descriptors_data):
        pass

class CustomEvaluator(AbstractEvaluator):
    def __init__(self, embedding_path, descriptors_type):
        self.word_embedding = WordEmbedding(embedding_path)
        self.descriptors_type = descriptors_type

    def make_descriptors_compatible(self, row):
        descriptors_fields = ApproachDescriptors.descriptors_dict[self.descriptors_type].copy()
        src_columns, target_columns = add_src_target_string(descriptors_fields)
        src_descriptors = set(row[src_columns])
        src_event = ' '.join(src_descriptors)
        target_descriptors = set(row[target_columns])
        target_event = ' '.join(target_descriptors)
        return CompatibleDescriptors(src_event, target_event)

    def assign_score(self, descriptors: CompatibleDescriptors):
        embedding1 = self.get_sentence_embedding(descriptors.src_event)
        embedding2 = self.get_sentence_embedding(descriptors.target_event)
        return self.cosine_similarity(embedding1, embedding2)

    def get_sentence_embedding(self, sentence):
        words = sentence.split()
        embeddings = np.array([self.word_embedding.get_embedding(word) for word in words])
        return np.mean(embeddings, axis=0)

    @staticmethod
    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)

    def get_potential_matches(self, descriptors_data):
        return descriptors_data[descriptors_data['src_type'] == descriptors_data['target_type']]
