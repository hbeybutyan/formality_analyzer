import xml.etree.ElementTree as ET
import uuid
import os

from multiprocessing import cpu_count, Process, Pool

L6_CORP_PATH = "/home/strata/formality_analyzer/resources/FullOct2007.xml"
#L6_CORP_PATH = "/home/strata/formality_analyzer/resources/small_sample.xml"
FORMAL_PATH = "formal"
INFORMAL_PATH = "informal"
MODEL_PATH = "/home/strata/formality_analyzer/resources/models/formality_classifier.hdf5_con1"
OUT_DIR = "/home/strata/univ/image-processing/thesis/resources/output"
MAX_NUM_WORDS = 50000
MAX_SEQUENCE_LENGTH = 128


def get_sentances(corp_path):
    tree = ET.parse(corp_path).getroot()
    sents = []
    slitted = []
    for child in tree:
        if child[0].find('qlang').text == "en":
            to_spl = child[0].find('content')
            if to_spl is not None and to_spl.text:
                sents.extend(to_spl.text.split('.'))
            to_spl = child[0].find('bestanswer')
            if to_spl is not None and to_spl.text:
                sents.extend(to_spl.text.split('.'))
            best = child[0].find('nbestanswers')
            if best is not None:
                for answ in best:
                    if answ is not None and answ.text:
                        sents.extend(answ.text.split('.'))
    for sent in sents:
        slitted.extend(sent.split('<br />'))
    to_return = [st.replace('<br />', ' ').replace('\n', '').strip() for st in slitted if st and not 'http' in st]
    return [sent for sent in to_return if sent]


def get_data(path_to_corp, max_num_words, max_seq_length):
    sentences = get_sentances(path_to_corp)
    print('Loaded %s sentences.' % len(sentences))

    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    data = pad_sequences(sequences, maxlen=max_seq_length)
    return data

def process_sents(model_path, sents, formal_path, informal_path, max_seq_length):
    formal = []
    informal = []
    sents_count = len(sents)
    percent = 0.1
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(sentences)
    from keras.models import load_model as load
    model = load(model_path)
    print("Processing %s sents" % str(sents_count))
    for i, sent in enumerate(sents):
        sequences = tokenizer.texts_to_sequences([sent])
        data = pad_sequences(sequences, maxlen=max_seq_length)
        pred = model.predict(data, 1)
        if pred > 0.9:
            formal.append(sent)
        elif pred < 0.1:
            informal.append(sent)
        if i / sents_count > percent:
            print("Processed %d of sents ..." % int(100 * percent))
            percent += 0.1
    if formal:
        with open(formal_path, 'w') as f:
            f.writelines('\n'.join(formal))
    if informal:
        with open(informal_path, 'w') as f:
            f.writelines('\n'.join(informal))

def proc_sents_async(sents):
    process_sents(MODEL_PATH, sents, os.path.join(OUT_DIR, "formal_" + str(uuid.uuid4())), os.path.join(OUT_DIR, "informal_" + str(uuid.uuid4())), MAX_SEQUENCE_LENGTH)

def classify_sentances(sentences):
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    sent_per_thread = len(sentences) / cpu_count()
    cpu_cnt = cpu_count()
    chunks = []
    for i in range(cpu_count()):
        print("Starting process %d..." % i)
        fr = int(i * sent_per_thread)
        to = int((i+1) * sent_per_thread)
        snt = sentences[fr: to]
        chunks.append(snt)
    pool = Pool(cpu_cnt)
    pool.map(proc_sents_async, chunks)


sentences = get_sentances(L6_CORP_PATH)
print('Loaded %s sentences.' % len(sentences))
print('Classifying...')
classify_sentances(sentences, MODEL_PATH, MAX_SEQUENCE_LENGTH, FORMAL_PATH, INFORMAL_PATH)
