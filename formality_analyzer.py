from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.models import Model
from keras.models import load_model as load

import os


MAX_NUM_WORDS = 50000
MAX_SEQUENCE_LENGTH = 128
SAVED_MODEL_PATH = "/home/strata/univ/image-processing/thesis/resources/models/formality_classifier.hdf5"
GYAFC_PATH = "/home/strata/univ/image-processing/thesis/resources/GYAFC_Corpus/GYAFC_Corpus"

def get_data(path_to_gyafc, max_num_words, max_seq_length):
    train_ent_path_formal = os.path.join(path_to_gyafc, "Entertainment_Music/train/formal")
    train_ent_path_informal = os.path.join(path_to_gyafc, "Entertainment_Music/train/informal")
    train_fam_path_formal = os.path.join(path_to_gyafc, "Family_Relationships/train/formal")
    train_fam_path_informal = os.path.join(path_to_gyafc, "Family_Relationships/train/formal")

    test_ent_path_formal = os.path.join(path_to_gyafc, "Entertainment_Music/test/formal")
    test_ent_path_informal = os.path.join(path_to_gyafc, "Entertainment_Music/test/informal")
    test_fam_path_formal = os.path.join(path_to_gyafc, "Family_Relationships/test/formal")
    test_fam_path_informal = os.path.join(path_to_gyafc, "Family_Relationships/test/formal")

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    with open(train_ent_path_formal, 'r') as ent_tr_f, open(train_ent_path_informal, 'r') as ent_tr_in, open(train_fam_path_formal, 'r') as fam_tr_f, open(train_fam_path_informal, 'r') as fam_tr_in :
        for line in ent_tr_f:
            x_train.append(line)
            y_train.append(1)
        for line in ent_tr_in:
            x_train.append(line)
            y_train.append(0)
        for line in fam_tr_f:
            x_train.append(line)
            y_train.append(1)
        for line in fam_tr_in:
            x_train.append(line)
            y_train.append(0)
    print('Loaded %s training sentences.' % len(x_train))

    with open(test_ent_path_formal, 'r') as ent_tr_f, open(test_ent_path_informal, 'r') as ent_tr_in, open(test_fam_path_formal, 'r') as fam_tr_f, open(test_fam_path_informal, 'r') as fam_tr_in :
        for line in ent_tr_f:
            x_test.append(line)
            y_test.append(1)
        for line in ent_tr_in:
            x_test.append(line)
            y_test.append(0)
        for line in fam_tr_f:
            x_test.append(line)
            y_test.append(1)
        for line in fam_tr_in:
            x_test.append(line)
            y_test.append(0)
    print('Loaded %s test sentences.' % len(x_test))

    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(x_train + x_train)
    train_sequences = tokenizer.texts_to_sequences(x_train)
    test_sequences = tokenizer.texts_to_sequences(x_test)
    train_data = pad_sequences(train_sequences, maxlen=max_seq_length)
    test_data = pad_sequences(test_sequences, maxlen=max_seq_length)
    return train_data, y_train, test_data, y_test

def define_model(max_seq_length, emb_size):
    sequence_input = Input(shape=(max_seq_length,), dtype='int32')
    embedding = Embedding(emb_size,
                          256,
                          input_length=max_seq_length,
                          trainable=True)(sequence_input)
    lstm = LSTM(units=128)(embedding)
    #dropout = Dropout(rate=0.2)(lstm)
    dense = Dense(128, activation="relu")(lstm)
    out = Dense(1, activation="sigmoid")(dense)

    model = Model(sequence_input, out)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model


def get_model(model_path, data_path, max_seq_length, max_num_words):
    if not os.path.exists(model_path):
        model = define_model(max_seq_length, max_num_words)
        x_train, y_train, x_val, y_val = get_data(data_path, max_num_words, max_seq_length)
        model.fit(x_train, y_train,
                  batch_size=128,
                  epochs=10,
                  validation_data=(x_val, y_val))
        model.save(model_path)
    else:
        model = load(model_path)
    return model



model = get_model(SAVED_MODEL_PATH, GYAFC_PATH, MAX_SEQUENCE_LENGTH, MAX_NUM_WORDS)

