#!/bin/bash
# To download the glove word vectors

mkdir glove
cd glove
curl http://nlp.stanford.edu/data/glove.6B.zip > glove.zip

unzip \*.zip