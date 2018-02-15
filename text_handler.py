# Text handler

NUM_CHARS = 97 #Number of allowed characters (letters, punctuation, and newline)

def read_data(filename):
	with open(filename,'r') as f:
		text = f.read()
		f.seek(0)
		numtitles = len(f.readlines())
	return text, numtitles

