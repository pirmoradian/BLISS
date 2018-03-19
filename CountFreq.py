'''
Calculates the frequency/probability of all words/pairs inside a corpus
USAGE:
python CountFreq infile outfile status

EXAMPLE:
python CountFreq.py selectedColumn.txt wFreqBLISS.txt wfreq

STATUS: decides which function should be called
'wfreq': get_wfreq(),'pfreq':get_pfreq_successive(),'pfreq_t': get_pfreq_trangl(),'wprob':get_wProb(), 'pprob':get_pProb_successive(),'pprob_t':get_pProb_trangl()

INFILE: the input file, corpus, whose words/pairs are considered to be counted
OUTFILE: the output file which contains the words/pairs and their frequencies in the format: 
freq word
or about pairs:
freq word1 word2

'''
from operator import itemgetter
import sys
import re

class CountFrequency:
	'''counts the occurence of patterns inside a file'''
	def __init__(self,infile,outfile):
		self.ifile = open(infile,'r')
		self.ofile = open(outfile, 'w')
		self.word_freq = {} #dictionary containing the frequency of words
	def get_wfreq(self):
		'''calls the funcs count_wfreq(), to count the occurences of all words, and print_wfreq(), to print the results in a file'''
		self.count_wfreq()
		self.print_wfreq()

	def get_linefreq(self):
		'''calls the funcs count_wfreq(), to count the occurences of all words, and print_wfreq(), to print the results in a file'''
		self.count_linefreq()
		self.print_wfreq()
 	
 	def get_pfreq_successive(self):
		'''calls the funcs count_pfreq_successive(), to count the occurences of all pairs, and print_wfreq(), to print the results in a file'''
		self.count_pfreq_successive()
		self.print_wfreq()

	def get_pfreq_trangl(self):
		'''calls the funcs count_pfreq_trangl(), to count the occurences of all pairs, and print_wfreq(), to print the results in a file'''
		self.count_pfreq_trangl()
		self.print_wfreq()

 	def get_wProb(self):
		'''calls the funcs count_wfreq(), to count the occurences of all words, then normalizes the frequencies to obtain probabilities by normalize() function, and finally  print_wfreq(), to print the results in a file'''
		self.count_wfreq()
		self.normalize()
		self.print_wfreq()
 	def get_pProb_successive(self):
		'''calls the funcs count_pfreq_successive(), to count the occurences of all pairs, then normalizes the frequencies to obtain probabilities by normalize(), and finally print_wfreq(), to print the results in a file'''
		self.count_pfreq_successive()
		self.normalize()
		self.print_wfreq()
        def get_pProb_spcfpos(self, pos1, pos2):
		'''calls the funcs count_pfreq_successive(), to count the occurences of all pairs, then normalizes the frequencies to obtain probabilities by normalize(), and finally print_wfreq(), to print the results in a file'''
		self.count_pfreq_spcfpos(pos1, pos2)
		self.normalize()
		self.print_wfreq()
	def get_pProb_trangl(self):
		'''calls the funcs count_pfreq_trangl(), to count the occurences of all pairs, then normalizes the frequencies to obtain probabilities by normalize(), and finally print_wfreq(), to print the results in a file'''
		self.count_pfreq_trangl()
		self.normalize()
		self.print_wfreq()

	def count_linefreq(self):
		'''counts the occurence of each line in a file'''
		word_freq = self.word_freq
		for l in self.ifile.readlines():
		        re.sub('\s+$','',l)
		        re.sub('^\s+','',l)
			l = re.split("\s+",l)
			l = ' '.join(l)
			word_freq[l] = word_freq.get(l,0) + 1
	
	def count_wfreq(self):
		'''counts the occurence of all words in a file'''
		word_freq = self.word_freq
		for l in self.ifile.readlines():
			word_list = l.lower().split(None)
			for word in word_list:
				word_freq[word] = word_freq.get(word,0) + 1

	def count_pfreq_successive(self):
		'''counts the occurence of all successive words in a file'''
		#reading file containing sentences
		word_freq = self.word_freq
		for l in self.ifile.readlines():
			words = l.split()
			l1 = words[:-1]
			l2 = words[1:]
			z = zip(l1,l2) #tuple of all successive words
			for pair in z:
				pair = ' '.join(list(pair))
				word_freq[pair] = word_freq.get(pair,0) + 1
        def count_pfreq_spcfpos(self, pos1, pos2):
		'''counts the occurence of all word pairs whose first word is in pos1 and second word in pos2'''
                word_freq = self.word_freq
		for l in self.ifile.readlines():
			words = l.split()
			w1 = words[pos1]
			w2 = words[pos2]
                        pair = w1 + ' ' + w2
			word_freq[pair] = word_freq.get(pair,0) + 1         

	def count_pfreq_trangl(self):
		'''count the occurence of all words in upper triangular matrix of words in a sentence.
		Example: w1, w2, w3 => (w1 w2), (w1 w3), (w2 w3)'''
		word_freq = self.word_freq
		for l in self.ifile.readlines():
			words = l.split()
			z = [(wi,wj) for i,wi in enumerate(words) for j,wj in enumerate(words) if i<j] #tuple of all ordered words
			for pair in z:
				pair = ' '.join(list(pair))
				word_freq[pair] = word_freq.get(pair,0) + 1
	def normalize(self):
		'''normalizes the frequencies, values of the dictionary, to find the probability'''
		word_freq = self.word_freq
		s = float(sum(word_freq.values()))
		for w in word_freq:
			word_freq[w] = word_freq[w]/s 
	def print_wfreq(self):
		'''prints the word frequencies in the output file in this format:
		frequency word
		'''
		word_freq = self.word_freq
		word_freq = sorted(word_freq.items(), key=itemgetter(1), reverse=True)
		for word in word_freq:
			word = list(word)
			fq = str(word[-1])
			ws = ' '.join(word[:-1])
			s = fq+' '+ws+"\n"
			self.ofile.write(s)
	
		self.ifile.close()
		self.ofile.close()


def main(infile,outfile,status):
	cf = CountFrequency(infile,outfile)
	if status == 'wfreq':
		cf.get_wfreq()
	elif status == 'pfreq':
		cf.get_pfreq_successive()
	elif status == 'pfreq_t':
		cf.get_pfreq_trangl()
	elif status == 'lfreq':
		cf.get_linefreq()
	elif status == 'wprob':
		cf.get_wProb()
	elif status == 'pprob':
		cf.get_pProb_successive()
	elif status == 'pprob_t':
		cf.get_pProb_trangl()
	else:
		raise Exception, 'what is status?!'	
if __name__ == "__main__":
	infile = sys.argv[1] #'selectedColumneqp.txt'
	outfile = sys.argv[2] #'pFreqBLISS-eqp-18Mar.txt'
	status = sys.argv[3] #'wfreq': get_wfreq(),'pfreq':get_pfreq(), 'wprob':get_wProb(), 'pprob':get_pProb()
	main(infile, outfile, status)		
