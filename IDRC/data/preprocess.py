import argparse
import codecs
import os
import re
import sys
import json
import nltk

parser = argparse.ArgumentParser(description='Require the path to pdtb dataset')
parser.add_argument('inputf', type=str, metavar='', help='')

A = parser.parse_args()
WHERE = ""
ONLY_IMPLICIT = True
selected_sense = set(['Temporal.Asynchronous','Temporal.Synchrony','Contingency.Cause','Contingency.Pragmatic cause','Comparison.Contrast','Comparison.Concession','Expansion.Conjunction','Expansion.Instantiation','Expansion.Restatement','Expansion.Alternative','Expansion.List'])

pattern_split = re.compile(r'^_+$')
pattern_type = re.compile(r'^_+.+_+$')

pattern_explicit = re.compile(r'^_+Explicit_+$')
pattern_implicit = re.compile(r'^_+Implicit_+$')
pattern_entrel = re.compile(r'^_+EntRel_+$')
pattern_norel = re.compile(r'^_+NoRel_+$')
pattern_altlex = re.compile(r'^_+AltLex_+$')

pattern_arg1 = re.compile(r'^_+Arg1_+$')
pattern_arg2 = re.compile(r'^_+Arg2_+$')

pattern_sup1 = re.compile(r'^_+Sup1_+$')
pattern_sup2 = re.compile(r'^_+Sup2_+$')

pattern_text = re.compile(r'^#+ Text #+$')
pattern_features = re.compile(r'^#+ Features #+$')
pattern_dd = re.compile(r'^\d\d$')
pattern_filename = re.compile(r'^wsj.+$')

pattern_empty_line = re.compile(r'^#+$')


def process_file(f, fw, is_train):
    f = open(f, "r")
    # match explicit
    lines = [l.strip() for l in f]
    i = 0
    store_info = []
    while i < len(lines):
        line = lines[i]
        i += 1
        if pattern_split.match(line):
            if len(store_info) > 1:
                process_unit(store_info, fw, is_train)
            store_info = []
        else:
            if len(line.strip()) > 0:
                store_info.append(line.strip())
    f.close()

def find_arg12(store_info):
    ind_arg1 = find_first_start_at(0, pattern_arg1, store_info)
    ind_text1 = find_first_start_at(ind_arg1, pattern_text, store_info)+1
    end_ind_text1 = ind_text1+1
    while not pattern_empty_line.match(store_info[end_ind_text1]):
        end_ind_text1+=1
    text1 = ' '.join(store_info[ind_text1:end_ind_text1]).strip()

    ind_arg2 = find_first_start_at(0, pattern_arg2, store_info)
    ind_text2 = find_first_start_at(ind_arg2, pattern_text, store_info)+1
    end_ind_text2 = ind_text2+1
    while not pattern_empty_line.match(store_info[end_ind_text2]):
        end_ind_text2+=1
    text2 = ' '.join(store_info[ind_text2:end_ind_text2]).strip()

    return [text1, text2]

def decode_relation(raw_relation):
    relation = raw_relation[(raw_relation.find(',')+1):].strip().split(',')
    finr = []
    for r in relation:
        sr = r.split('.')
        sr = sr[0].strip()
        #sr = sr[:2]
        #sr = ".".join(sr).strip()
        finr.append(sr)
    return finr

def find_relation(store_info):
    below = find_first_start_at(0, pattern_sup1, store_info) if find_first_start_at(0, pattern_sup1, store_info) > 0 else find_first_start_at(0, pattern_arg1, store_info)
    behigh = find_first_start_at(0,pattern_empty_line,store_info)
    if behigh > below:
        behigh = find_first_start_at(0,pattern_features,store_info)+1
    finr = []
    for relation in store_info[behigh+1:below]:
        finr.extend(decode_relation(relation))
    return list(set(finr))  #&selected_sense

def process_unit(store_info, fw, is_train):
    if pattern_type.match(store_info[0]):
        relation = ""
        info_type = store_info[0]
        if pattern_explicit.match(info_type):
            if ONLY_IMPLICIT:
                return
            relation = find_relation(store_info)
        elif pattern_implicit.match(info_type):
            relation = find_relation(store_info)
        elif pattern_altlex.match(info_type):
            if ONLY_IMPLICIT:
                return
            relation = find_relation(store_info)
        elif pattern_entrel.match(info_type):
            if ONLY_IMPLICIT:
                return
            relation = 'EntRel'
        elif pattern_norel.match(info_type):
            if ONLY_IMPLICIT:
                return
            relation = 'NoRel'

        if len(relation) > 0:
            finlist = find_arg12(store_info)
            print_instance(relation, finlist, is_train)

def print_instance(relations, finlist, is_train):
    arg1 =  reduce(lambda x,y: x+y, [nltk.word_tokenize(s) for s in nltk.sent_tokenize(finlist[0])])
    arg2 = reduce(lambda x,y: x+y, [nltk.word_tokenize(s) for s in nltk.sent_tokenize(finlist[1])])
    if len(relations)>1:
        return
    #if is_train:
    for relation in relations:
        fw.write(json.dumps({'Arg1':arg1,'Arg2':arg2,'Sense':relation})+'\n')
    #else:
    #    fw.write(json.dumps({'Arg1':arg1,'Arg2':arg2,'Sense':relations})+'\n')
    

def find_first_start_at(start, pattern, store_info):
    for i in xrange(start, len(store_info)):
        if pattern.match(store_info[i]):
            return i
    return -1


if __name__ == '__main__':
    f_train = open("pdtb.train", "w")
    f_dev = open("pdtb.dev", "w")
    f_test = open("pdtb.test", "w")

    for lists in os.listdir(A.inputf): 
        path = os.path.join(A.inputf, lists)
        if pattern_dd.match(lists):
            section = int(lists)
            fw = None
            if 0<=section <= 1:
                fw = f_dev
            elif 2<=section <= 20:
                fw = f_train
            elif 21<=section <= 22:
                fw = f_test
            else:
                continue
            for files in os.listdir(path):
                file_path = os.path.join(path, files)
                if pattern_filename.match(files):
                    WHERE = file_path
                    process_file(file_path, fw, fw is f_train)

    f_train.close()
    f_dev.close()
    f_test.close()

