# Text formality analysis 


This repository contains a tool which allows to train a text formality classifier.

For training the classirier the GYAFC parallel corpus is used:
* [GYAFC](https://github.com/raosudha89/GYAFC-corpus)

Please follow the instructions to get the corpus.

## Dependencies

* Python 3
* [NumPy](http://www.numpy.org/)
* [Tesorflow](https://www.tensorflow.org/)
* [Keras API](https://keras.io/)

## Usage

Clone the repo as:

```
git clone https://github.com/hbeybutyan/formality_analyzer.git
cd formality_analyzer
```

Get the GYAFC corpus and extract it:

In the script update the GYAFC_PATH, set it to the dir there you just extracted the GYAFC corpus.

Run script as:

```
python3 formality_analyzer.py
```
