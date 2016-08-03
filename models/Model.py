import os
import tensorflow as tf

class Model(object):
	
	def save(self,save_path,filename):
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		tf.train.Saver().save(self.sess,os.path.join(save_path,filename))

	def load(self,load_path,filename):
		tf.train.Saver().restore(self.sess,os.path.join(load_path,filename))