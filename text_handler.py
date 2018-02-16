# Text handler
import numpy as np 

NUM_CHARS = 97 #Number of allowed characters (letters, punctuation, and newline)

def read_data(filename):
	with open(filename,'r') as f:
		text = f.read()
		f.seek(0)
		numtitles = len(f.readlines())
	return text, numtitles

def batch_data_generator(text,batch_size,time_steps,num_epochs):
	data = list(text)
	data = list(map(lambda a: char_to_code(a),data))
	data = np.array(data)
	num_batches = (data.shape[0]-1)//(batch_size*time_steps)
	data_len = num_batches*batch_size*time_steps
	train = np.reshape(data[0:data_len], [batch_size, num_batches * time_steps])
	labels = np.reshape(data[1:data_len + 1], [batch_size, num_batches * time_steps])#label is next char in text
	for epoch in range(num_epochs):
		for batch in range(num_batches):
			x = train[:,batch*time_steps:(batch+1)*time_steps]
			y = labels[:,batch*time_steps:(batch+1)*time_steps]
			x = np.roll(x,-epoch,axis = 0)
			y = np.roll(y,-epoch,axis = 0)
			yield x,y,epoch

def char_to_code(c):
	c = ord(c)
	if c == 10:
		return 1
	elif 32 <= c <= 126:
		return c-30
	else:
		return 0