#Implicit Discourse Relation Classification with Pair-Aware Sentence modeling

Our implementation relies on several open-source toolkits such as Tensorflow, Gensim, and Numpy
 
For the usage of this implementation, please type the command "python ./IDRC/main.py --help" for detailed instructions.

The command to run our implementation should be like "python ./IDRC/main.py --arg_name1 arg_value1 --arg_name2 arg_value2 ... ". 

For example, to show the best result reported in our paper: "python ./IDRC/main.y --is_test True"

The original PDTB data and the preprocessed data (with the code of preprocess) is in "./IDRC/data"

The pre-trained word vectors is too large, so we do not upload it in this submission. However, you can download it from its original source "https://code.google.com/archive/p/word2vec/" (GoogleNews-vectors-negative300.bin.gz)
