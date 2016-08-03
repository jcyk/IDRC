import os
import json
import random
import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict

class data_loader(object):

	def __init__(self,data_path,dataset_name,batch_size,max_seq_len,use_pre_trained_embedding,binary,pre_trained_embedding_size=300):
		self.batch_size = batch_size
		train_file = os.path.join(data_path,dataset_name+'.train') #
		dev_file = os.path.join(data_path,dataset_name+'.dev')
		test_file = os.path.join(data_path,dataset_name+'.test')
		
		raw_train_data =[json.loads(line) for line in open(train_file).readlines()]
		sense_set = set(d['Sense'] for d in raw_train_data)
		sense2idx = {}
		if binary is not None:
			for s in sense_set:
				if s.startswith(binary):
					sense2idx[s] = 1
				else:
					sense2idx[s] = 0
		else:
			for s in sense_set: 	
				sense2idx[s] = len(sense2idx)

		Word_set = reduce(lambda s,d:s|set(d['Arg1']+d['Arg2']),raw_train_data,set([]))
		
		def init_dictionary(keys):
			k2i,i2k =  {'<UNK_>':0,'<GO_>':1,'<EOS_>':2},['<UNK_>','<GO_>','<EOS_>']
			for key in keys:
				i2k.append(key)
				k2i[key] = len(i2k)-1
			return k2i,i2k

		Word2idx,idx2Word = init_dictionary(Word_set)

		self.word_vocab_size = len(Word2idx)
		self.sense_vocab_size = len(sense2idx)
		
		if use_pre_trained_embedding:
			w2v = Word2Vec.load_word2vec_format('/home/cd/tensorflow/GoogleNews-vectors-negative300.bin.gz',binary=True)
			word_embeddings = np.zeros(shape=(self.word_vocab_size,pre_trained_embedding_size),dtype='float32')
			pre_trained_vocab = set(w2v.vocab.keys())
			for idx,word in enumerate(idx2Word):
				if word in pre_trained_vocab:
					word_embeddings[idx,:] = w2v[word]
			self.pre_trained_word_embeddings = word_embeddings

		def symbol_to_number(raw_data,is_train=True):
			result = []
			golden_answer = []
			for d in raw_data:
				if is_train:
					sense = sense2idx[d['Sense']]
					golden_answer.append([sense])
				else:
					sense = 0
					golden_answer.append([sense2idx[s] for s in d['Sense']])

				arg1 = [ Word2idx.get(word,0) for word in d['Arg1']]
				arg2 = [ Word2idx.get(word,0) for word in d['Arg2']]

				if max_seq_len is not None:
					if len(arg1)>max_seq_len:
						arg1 = arg1[-max_seq_len:]
					if len(arg2)>max_seq_len:
						arg2 = arg2[:max_seq_len]
				arg1 = [1]+arg1+[2]
				arg2 = [1]+arg2
				result.append([sense,arg1,arg2])
			return result,golden_answer

		self.golden_answer = [None]*3
		self.train_data,self.golden_answer[0] = symbol_to_number(raw_train_data)

		self.dev_data,self.golden_answer[1]= symbol_to_number([json.loads(line) for line in open(dev_file).readlines()],is_train=False)
		if os.path.exists(test_file):
			self.test_data,self.golden_answer[2] = symbol_to_number([json.loads(line) for line in open(test_file).readlines()],is_train=False)
		self.ready = [0]*3

	def prepare_batches(self,which_set):
		if which_set == 0:
			random.shuffle(self.train_data)
			data = self.train_data
		else:
			data = self.dev_data if which_set == 1 else self.test_data

		def data_to_tensor(raw_data):
			#batch_size x 1 | batch_size x seq_len x 4| batch_size x 1
			sense = np.asarray([d[0] for d in raw_data],dtype='int32')
			arg1_len = np.asarray([len(d[1]) for d in raw_data],dtype='int32')
			arg2_len = np.asarray([len(d[2]) for d in raw_data],dtype='int32')
			batch_size = len(raw_data)
			max_arg1_len = arg1_len.max()
			arg1 = np.zeros(shape=(batch_size,max_arg1_len),dtype='int32')
			arg2 = np.zeros(shape=(batch_size,arg2_len.max()),dtype='int32')
			mask = np.ones(shape=(batch_size,max_arg1_len),dtype='float32')
			for idx,d in enumerate(raw_data):
				arg1[idx,:len(d[1])] = d[1]
				mask[idx,len(d[1]):] = 0.
				arg2[idx,:len(d[2])] = d[2]
			return [sense,arg1,arg2,arg1_len,arg2_len,mask]

		num_batches = len(data)/self.batch_size
		batches = []
		for i in xrange(num_batches):
			batches.append(data_to_tensor(data[i*self.batch_size:i*self.batch_size+self.batch_size]))
		if which_set>0 and num_batches*self.batch_size<len(data):
			batches.append(data_to_tensor(data[num_batches*self.batch_size:]))
		self.batches = batches
		self.ready = [0]*3
		self.ready[which_set] = len(batches) 

	def next_batch(self,which_set):
		if self.ready[which_set] == 0:
			self.prepare_batches(which_set)
		self.ready[which_set]-=1
		is_end = (self.ready[which_set]==0)
		return self.batches[self.ready[which_set]],is_end