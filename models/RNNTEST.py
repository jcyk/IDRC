import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn,rnn_cell
from Model import Model
from utils import dynamic_rnn_decoder

TRAIN_SET,DEV_SET,TEST_SET = 0,1,2

class RNNTEST(Model):
	def __init__(self,sess,data_loader,NER_embed_size,lemma_embed_size,word_embed_size,POS_embed_size,encoder_size,decoder_size,learning_rate,epochs,attention_judge_size,learning_rate_decay_factor,use_pre_trained_embedding,max_gradient_norm,is_test,dropout_rate):
		self.sess = sess
		self.data_loader = data_loader
		self.NER_embed_size = NER_embed_size
		self.lemma_embed_size = lemma_embed_size
		self.word_embed_size = word_embed_size
		self.POS_embed_size = POS_embed_size
		self.embedding_size = NER_embed_size+lemma_embed_size+word_embed_size+POS_embed_size
		self.encoder_size = encoder_size
		self.decoder_size = decoder_size
		self.lr = tf.Variable(learning_rate,trainable=False,dtype=tf.float32)
		self.epochs = epochs
		self.attention_judge_size = attention_judge_size
		self.learning_rate_decay_factor = learning_rate_decay_factor
		self.use_pre_trained_embedding = use_pre_trained_embedding
		self.max_gradient_norm = max_gradient_norm
		self.is_test = is_test
		self.dropout_rate = dropout_rate
		self.build_model()

	def build_model(self):
		with tf.variable_scope('RNNTEST'):
			self.sense = tf.placeholder(tf.int32,[None])
			self.arg1 = tf.placeholder(tf.int32,[None,None,4])
			self.arg2 = tf.placeholder(tf.int32,[None,None,4])
			self.arg1_len = tf.placeholder(tf.int32,[None])
			self.arg2_len = tf.placeholder(tf.int32,[None])
			self.keep_prob = tf.placeholder(tf.float32)

			arg1_list = tf.split(2,4,self.arg1)
			arg2_list = tf.split(2,4,self.arg2)
			
			with tf.device('/cpu:0'):
				NER_W = tf.get_variable('NER_embed',[self.data_loader.NER_vocab_size,self.NER_embed_size]) if self.NER_embed_size>0 else None
				lemma_W = tf.get_variable('lemma_embed',[self.data_loader.lemma_vocab_size,self.lemma_embed_size]) if self.lemma_embed_size>0 else None
				if self.use_pre_trained_embedding:
					word_W = tf.get_variable('word_embed',initializer = tf.convert_to_tensor(self.data_loader.pre_trained_word_embeddings,dtype=tf.float32)) if self.word_embed_size>0 else None
				else:
					word_W = tf.get_variable('word_embed',shape = [self.data_loader.word_vocab_size,self.word_embed_size]) if self.word_embed_size>0 else None
				POS_W = tf.get_variable('POS_embed',[self.data_loader.POS_vocab_size,self.POS_embed_size]) if self.POS_embed_size>0 else None
			arg1_embed_list = []
			arg2_embed_list = []
			for idx,W in enumerate([NER_W,lemma_W,word_W,POS_W]):
				if W is not None:
					arg1_embed_list.append(tf.nn.embedding_lookup(W,tf.squeeze(arg1_list[idx],[2])))
					arg2_embed_list.append(tf.nn.embedding_lookup(W,tf.squeeze(arg2_list[idx],[2])))
			arg1 = tf.nn.dropout(tf.concat(2,arg1_embed_list),self.keep_prob)
			arg2 = tf.nn.dropout(tf.concat(2,arg2_embed_list),self.keep_prob)
			
			encoder_lstm_unit = rnn_cell.BasicLSTMCell(self.encoder_size)
			decoder_lstm_unit = rnn_cell.BasicLSTMCell(self.decoder_size)

			with tf.variable_scope('forward_encoder'):
				forward_encoder_outputs,forward_encoder_state = rnn.dynamic_rnn(encoder_lstm_unit,arg1,self.arg1_len,dtype=tf.float32)
			with tf.variable_scope('backward_encoder'):
				backward_encoder_outputs,backward_encoder_state= rnn.dynamic_rnn(encoder_lstm_unit,tf.reverse_sequence(arg1,tf.cast(self.arg1_len,tf.int64),1),dtype=tf.float32)
			encoder_outputs = tf.concat(2,[forward_encoder_outputs,tf.reverse_sequence(backward_encoder_outputs,tf.cast(self.arg1_len,tf.int64),1)])
			encoder_state = tf.concat(1,[forward_encoder_state,backward_encoder_state])

			source = tf.expand_dims(encoder_outputs,2) #batch_size x source_len x 1 x source_depth(2*encoder_size)
			attention_W = tf.get_variable('attention_W',[1,1,2*self.encoder_size,self.attention_judge_size])
			attention_V = tf.get_variable('attention_V',[self.attention_judge_size])
 			WxH = tf.nn.conv2d(source, attention_W,[1,1,1,1],'SAME') #batch_size x source_len x 1 x attention
 			self.mask = tf.placeholder(tf.float32,[None,None])

			def attention(input_t,output_t_minus_1,time):
				with tf.variable_scope('attention'):
					VxS = tf.reshape(rnn_cell.linear(output_t_minus_1,self.attention_judge_size,True),[-1,1,1,self.attention_judge_size]) #batch_size x 1 x 1 x attention
				_exp = tf.exp(tf.reduce_sum( attention_V * tf.tanh(WxH+VxS), [3]))#batch_size x source_len x 1
				_exp = _exp*tf.expand_dims(self.mask,-1)
				attention_weight = _exp/tf.reduce_sum(_exp,[1], keep_dims=True)
				attention_t = tf.reduce_sum(encoder_outputs*attention_weight,[1])
				feed_in_t = tf.tanh(rnn_cell.linear([attention_t,input_t],self.embedding_size,True))
				return feed_in_t

			with tf.variable_scope('decoder'):
				decoder_outputs,decoder_state = dynamic_rnn_decoder(arg2,decoder_lstm_unit,initial_state=encoder_state,sequence_length=self.arg2_len,loop_function=attention)
			judge = tf.concat(1,[tf.reduce_sum(decoder_outputs,[1])/tf.expand_dims(tf.cast(self.arg2_len,tf.float32),-1),tf.reduce_sum(encoder_outputs,[1])/tf.expand_dims(tf.cast(self.arg1_len,tf.float32),-1)])
			unscaled_log_distribution = rnn_cell.linear(judge,self.data_loader.sense_vocab_size,True)
			self.output = tf.cast(tf.argmax(unscaled_log_distribution,1),tf.int32)
			self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.output,self.sense), tf.float32))
			
			#max-margin method
			#self._MM = tf.placeholder(tf.int32,[None])
			#margin = tf.sub(tf.reduce_max(unscaled_log_distribution,[1]),tf.gather(tf.reshape(unscaled_log_distribution,[-1]),self._MM))
			#self.loss = tf.reduce_mean(margin)

			#maximum likelihood method
			self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(unscaled_log_distribution, self.sense))
			
			self.optimizer = tf.train.AdagradOptimizer(self.lr)
			self.train_op = self.optimizer.minimize(self.loss)

			#You wanna do clip?
			#params = tf.trainable_variables()
			#gradients = tf.gradients(self.loss,params)
			#clipped_gradients, self.global_norm = tf.clip_by_global_norm(gradients,self.max_gradient_norm)
			#self.train_op =  self.optimizer.apply_gradients(zip(clipped_gradients,params))
			
			#You wanna do decay?
			#self.lr_decay_op = self.lr.assign(self.lr*self.learning_rate_decay_factor)

	def train(self,epoch):
		while True:
			x,is_finished = self.data_loader.next_batch(TRAIN_SET)			
			feed_dict= {self.sense:x[0],
					self.arg1:x[1],
					self.arg2:x[2],
					self.arg1_len:x[3],
					self.arg2_len:x[4],
					self.mask:x[5],
					self.keep_prob:1.-self.dropout_rate}
			_,accuracy,loss = self.sess.run([self.train_op,self.accuracy,self.loss],feed_dict=feed_dict)
			if is_finished:
				break

	def test(self,which_set,epoch):
		prediction = []
		while True:
			x,is_finished = self.data_loader.next_batch(which_set)
			feed_dict= {self.arg1:x[1],
					self.arg2:x[2],
					self.arg1_len:x[3],
					self.arg2_len:x[4],
					self.mask:x[5],
					self.keep_prob:1.}
			output = self.sess.run(self.output,feed_dict=feed_dict)
			prediction = list(output) + prediction
			if is_finished:
				break
		accuracy,set_size = 0,len(prediction)
		for p,g in zip(prediction,self.data_loader.golden_answer[which_set]):
			if p in g:
				accuracy+=1
		return float(accuracy)/float(set_size)*100

	def run(self):
		tf.initialize_all_variables().run()
		if self.is_test:
			self.load('save','best')
			test_accuracy = self.test(TEST_SET,0)
			print 'Test accuracy: %.2f%%'%(test_accuracy,)
		else:
			best_dev_accuracy = 0.
			prev_dev_accuracy = 0.
			for epoch in xrange(self.epochs):
				self.train(epoch)
				print 'Epoch %d finished.'%(epoch+1,)
				dev_accuracy= self.test(TEST_SET,epoch)
				print 'Dev accuracy: %.2f%% Best: %.2f%%'%(dev_accuracy,best_dev_accuracy)
				if dev_accuracy > best_dev_accuracy:
					best_dev_accuracy = dev_accuracy
					self.save('save','best')