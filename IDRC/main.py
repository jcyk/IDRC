import os
import sys
sys.path.append('..')

import numpy as np
import tensorflow as tf
from models import RNNTEST2
from data_loader import data_loader

import pprint
pp = pprint.PrettyPrinter()

flags = tf.app.flags

#hyperparameters and settings
flags.DEFINE_string('data_path','data','the path of data file [data]')
flags.DEFINE_string('dataset_name','pdtb','the name of dataset [pdtb]')
flags.DEFINE_boolean('is_test',False,'True for testing, False for training [False]')
flags.DEFINE_boolean('use_pre_trained_embedding',True,'True for pre-trained word embedding [True]')
flags.DEFINE_integer('batch_size',128,'the batch size [128]')
flags.DEFINE_integer('max_seq_len',120,'the maximum length of arg1 and arg2 [120]')
flags.DEFINE_integer('word_embed_size',300,'the size of word embedding [300]')
flags.DEFINE_integer('encoder_size',150,'the number of hidden units in RNN encoder [150]')
flags.DEFINE_integer('decoder_size',300,'the number of hidden units in RNN decoder [300]')
flags.DEFINE_integer('epochs',50,'the total number of epochs for training [50]')
flags.DEFINE_integer('attention_judge_size',100,'the size of attention judge vector [100]')
flags.DEFINE_integer('cores',0,'the number of cores used [0 (all)]')
flags.DEFINE_float('learning_rate',0.1,'the initial learning rate [0.1]')
flags.DEFINE_float('dropout_rate',0.5,'the dropout rate [0.5]')
flags.DEFINE_float('learning_rate_decay_factor',0.5,'the decay factor of learning rate [0.5]')
flags.DEFINE_float('max_gradient_norm',5.,'clip gradients to this norm.')
flags.DEFINE_string('binary',None,'Not None for binary classification')
#hyperparameters and settings

FLAGS = flags.FLAGS

def main(_):
	pp.pprint(flags.FLAGS.__flags)
	sess_config = tf.ConfigProto(intra_op_parallelism_threads=FLAGS.cores,inter_op_parallelism_threads=FLAGS.cores,allow_soft_placement=True)
	with tf.Session(config=sess_config) as sess:
		model = RNNTEST2(sess,
						data_loader=data_loader(FLAGS.data_path,
												FLAGS.dataset_name,
												FLAGS.batch_size,
												FLAGS.max_seq_len,
												FLAGS.use_pre_trained_embedding,
												FLAGS.binary
												),
						word_embed_size = FLAGS.word_embed_size,
						encoder_size = FLAGS.encoder_size,
						decoder_size = FLAGS.decoder_size,
						learning_rate = FLAGS.learning_rate,
						attention_judge_size = FLAGS.attention_judge_size,
						epochs = FLAGS.epochs,
						learning_rate_decay_factor = FLAGS.learning_rate_decay_factor,
						use_pre_trained_embedding = FLAGS.use_pre_trained_embedding,
						max_gradient_norm = FLAGS.max_gradient_norm,
						dropout_rate = FLAGS.dropout_rate,
						is_test = FLAGS.is_test,
						binary = FLAGS.binary)
		model.run()
	
if __name__ == '__main__':
	tf.app.run(main)
