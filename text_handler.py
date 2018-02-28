# Text handler
# Credit to this structure goes to Martin Gorner
# https://github.com/martin-gorner/tensorflow-rnn-shakespeare
#
# I used his shakespeare generator as a starting point and tutorial


import numpy as np
import random

NUM_CHARS = 97 #Number of allowed characters (letters, punctuation, and newline)

def read_data(filename):
	with open(filename,'r') as f:
		text = f.read()
		f.seek(0)
		numtitles = len(f.readlines())
	return text, numtitles

def batch_data_generator(filename,batch_size,time_steps,num_epochs):
	# randomize the order titles are seen each epoch
	# Could do this more efficiently, but this is easy and fast enough
	for epoch in range(num_epochs):
		with open(filename,'r') as f:
			lines = f.readlines()
			random.shuffle(lines)
		with open(filename,'w') as f:
			f.writelines(lines)
		with open(filename,'r') as f:
			text = f.read()
		data = list(text)
		data = list(map(lambda a: char_to_code(a),data))
		data = np.array(data)
		num_batches = (data.shape[0]-1)//(batch_size*time_steps)
		data_len = num_batches*batch_size*time_steps
		train = np.reshape(data[0:data_len], [batch_size, num_batches * time_steps])
		labels = np.reshape(data[1:data_len + 1], [batch_size, num_batches * time_steps])#label is next char in text
		for batch in range(num_batches):
			x = train[:,batch*time_steps:(batch+1)*time_steps]
			y = labels[:,batch*time_steps:(batch+1)*time_steps]
			x = np.roll(x,-epoch,axis = 0)
			y = np.roll(y,-epoch,axis = 0)
			yield x,y,epoch

def sample_probs(probs,topn): #choose from the top n most probable chars
	p = np.squeeze(probs)
	p[np.argsort(p)[:-topn]] = 0
	p = p / np.sum(p)
	return np.random.choice(NUM_CHARS, 1, p=p)[0]

def char_to_code(c):
	c = ord(c)
	if c == 10: # end line
		return 1
	elif 32 <= c <= 126:
		return c-30
	else:
		return 0
def code_to_char(a):
	if a == 1:
		return chr(10)
	elif 1<a<96:
		return chr(a+30)
	else:
		return chr(0)