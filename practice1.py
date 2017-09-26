import numpy as np
import tensorflow as tf
import random
import pickle
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
"""
'I'm super man'
tokenize:
['I', ''m', 'super','man' ]
"""
from nltk.stem import WordNetLemmatizer
import sys  


reload(sys)  
sys.setdefaultencoding('utf-8')


pos_file = 'pos.txt'
neg_file = 'neg.txt'



# build vocabulary
def create_lexicon(pos_file, neg_file):
	lex = []
	# load file
	def process_file(f):
		for line in open(pos_file):
			lex = []
			words = word_tokenize(line.lower())
			lex += words
		return lex
		'''with open(pos_file, 'r') as f:
			lex = []
			lines = f.readlines()
			
			for line in lines:
				line = line.decode("utf8")
				words = word_tokenize(line.lower())
				lex += words
			return lex'''

	lex += process_file(pos_file)
	lex += process_file(neg_file)
	#print(len(lex))
	lemmatizer = WordNetLemmatizer()
	lex = [lemmatizer.lemmatize(word) for word in lex] #  (cats->cat)

	word_count = Counter(lex)
	#print(word_count)
	# {'.': 13944, ',': 10536, 'the': 10120, 'a': 9444, 'and': 7108, 'of': 6624, 'it': 4748, 'to': 3940......}
	# remove common words such as "the","and" and some uncommon words
	lex = []
	for word in word_count:
		if word_count[word] < 2000 and word_count[word] > 20:  #
			lex.append(word)        #
	return lex


def normalize_dataset(lex):
	dataset = []

	def string_to_vector(lex, review, clf):
		words = word_tokenize(line.lower())
		lemmatizer = WordNetLemmatizer()
		words = [lemmatizer.lemmatize(word) for word in words]

		features = np.zeros(len(lex))
		for word in words:
			if word in lex:
				features[lex.index(word)] = 1  #
		return [features, clf]

	with open(pos_file, 'r') as f:
		lines = f.readlines()
		for line in lines:
			one_sample = string_to_vector(lex, line, [1,0])  #
			dataset.append(one_sample)
	count = 0
	with open(neg_file, 'r') as f:
		lines = f.readlines()
		for line in lines:
			one_sample = string_to_vector(lex, line, [0,1])  # [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), [0,1]]]
			dataset.append(one_sample)

	#print(len(dataset))
	return dataset


lex = create_lexicon(pos_file, neg_file)
#lex contains words that have appeared in the text
dataset = normalize_dataset(lex)
random.shuffle(dataset)
"""

with open('save.pickle', 'wb') as f:
	pickle.dump(dataset, f)
"""


test_size = int(len(dataset) * 0.1)

dataset = np.array(dataset)

train_dataset = dataset[:-test_size]
test_dataset = dataset[-test_size:]

# Feed-Forward Neural Network
#
n_input_layer = len(lex)  # input layer

n_layer_1 = 1000    # hide layer
n_layer_2 = 1000    # hide layer

n_output_layer = 2       # output layer

#
def neural_network(data):

	layer_1_w_b = {'w_':tf.Variable(tf.random_normal([n_input_layer, n_layer_1])), 'b_':tf.Variable(tf.random_normal([n_layer_1]))}

	layer_2_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_1, n_layer_2])), 'b_':tf.Variable(tf.random_normal([n_layer_2]))}

	layer_output_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_2, n_output_layer])), 'b_':tf.Variable(tf.random_normal([n_output_layer]))}


	layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
	layer_1 = tf.nn.relu(layer_1)
	layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
	layer_2 = tf.nn.relu(layer_2 )
	layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])

	return layer_output

batch_size = 64

X = tf.placeholder('float', [None, len(train_dataset[0][0])])
#[None, len(train_x)]
Y = tf.placeholder('float')

def train_neural_network(X, Y):
	predict = neural_network(X)
	cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
	optimizer = tf.train.AdamOptimizer().minimize(cost_func)  # learning rate 0.001

	epochs = 15
	with tf.Session() as session:
		session.run(tf.global_variables_initializer())
		epoch_loss = 0

		i = 0
		random.shuffle(train_dataset)
		train_x = dataset[:, 0]
		train_y = dataset[:, 1]
		for epoch in range(epochs):
			while i < len(train_x):
				start = i
				end = i + batch_size

				batch_x = train_x[start:end]
				batch_y = train_y[start:end]

				_, c = session.run([optimizer, cost_func], feed_dict={X:list(batch_x),Y:list(batch_y)})
				epoch_loss += c
				i += batch_size

			print(epoch, ' : ', epoch_loss)

		text_x = test_dataset[: ,0]
		text_y = test_dataset[:, 1]
		correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('accuracy: ', accuracy.eval({X:list(text_x) , Y:list(text_y)}))

train_neural_network(X,Y)
