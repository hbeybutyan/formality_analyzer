from keras.models import load_model as load
import xml.etree.ElementTree as ET
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

L6_CORP_PATH = "/home/strata/univ/image-processing/thesis/resources/small_sample.xml"
FORMAL_PATH = "/home/strata/univ/image-processing/thesis/resources/formal"
INFORMAL_PATH = "/home/strata/univ/image-processing/thesis/resources/informal"
MODEL_PATH = "/home/strata/univ/image-processing/thesis/resources/models/formality_classifier.hdf5_con1"
MAX_NUM_WORDS = 50000
MAX_SEQUENCE_LENGTH = 128


def get_sentances(corp_path):
    tree = ET.parse(corp_path).getroot()
    sents = []
    slitted = []
    for child in tree:
        if child[0].find('qlang').text == "en":
            sents.extend(child[0].find('content').text.split('.'))
            sents.extend(child[0].find('bestanswer').text.split('.'))
            for answ in child[0].find('nbestanswers'):
                sents.extend(answ.text.split('.'))
    for sent in sents:
        slitted.extend(sent.split('<br />'))
    to_return = [st.replace('<br />', ' ').replace('\n', '').strip() for st in slitted if st and not 'http' in st]
    return [sent for sent in to_return if sent]


def get_data(path_to_corp, max_num_words, max_seq_length):
    sentences = get_sentances(path_to_corp)
    print('Loaded %s sentences.' % len(sentences))

    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    data = pad_sequences(sequences, maxlen=max_seq_length)
    return data

def classify_sentances(sentences, model_path, max_seq_length, tokenizer, formal_path, informal_path):
    model = load(model_path)
    print('Loaded model...')
    formal = []
    informal = []
    for sent in sentences:
        sequences = tokenizer.texts_to_sequences([sent])
        data = pad_sequences(sequences, maxlen=max_seq_length)
        pred = model.predict(data, 1)
        if pred > 0.9:
            formal.append(sent)
        elif pred < 0.1:
            informal.append(sent)
    with open(formal_path, 'w') as f:
        f.writelines('\n'.join(formal))
    with open(informal_path, 'w') as f:
        f.writelines('\n'.join(informal))


sentences = get_sentances(L6_CORP_PATH)
print('Loaded %s sentences.' % len(sentences))
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(sentences)
print('Classifying...')
classify_sentances(sentences, MODEL_PATH, MAX_SEQUENCE_LENGTH, tokenizer, FORMAL_PATH, INFORMAL_PATH)
