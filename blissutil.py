import matplotlib.pyplot as plt
#from scipy import stats,polyval
import numpy as np
from operator import itemgetter
import pickle
import copy
import re
__all__ = ['UtilDict', 'FileReader']
#***************************************
class UtilDict(object):
	'''containing the utility functions used for dictionary'''

	def __init__(self):
		pass
          
        def rescale_d(self, d):
              avg = np.mean(d.values())
              intv = np.max(d.values()) - np.min(d.values())
              for k, v in d.iteritems():
                  d[k] = (d[k] - avg) / float(intv)
              return d
	def sort_dictionary(self, d, sort='key'):
        	'''sorts a dictionary based on 'key' or 'value' indicated by 'sort' variable as input and returns the corresponding sorted tuple'''
        	d_tuple = d.items()
        	if sort == 'key':
                	d_tuple = sorted(d_tuple, key=itemgetter(0))
                	#sorted(d_tuple, key=lambada a:a[0])            
        	elif sort == 'value':
                	d_tuple = sorted(d_tuple, key=itemgetter(1), reverse = True)
        	else:
                	raise Exception, 'how should the dictionary be sorted?!'
        	return d_tuple

	def normalize_d_values(self, d_):
        	"Normalize the values of a dictinary"
		d = copy.deepcopy(d_)
		for k in d:
			d[k] = float(d[k])
        	s = sum(d.values())
        	for w in d:
                	d[w] = d[w]/s                           
        	return d
	def normalize_d_wrt1stwd(self, d):
		fq_wds = {}
		for k, v in d.iteritems():
			w1, w2 = k.split()	
			fq_wds[w1] = fq_wds.get(w1,0) + v
		for k in d:
			w1, w2 = k.split()
			d[k] = d[k] / float(fq_wds[w1])
		return d

	def get_shared_keys(self,d1, d2):
		'''get the keys shared by two dictionaries d1, d2'''
		keys = []
		keys = [k for k in d1 if k in d2]
		return keys
	def get_shared_dicts(self,d1, d2):
		'''get the dictionaries corresponding to a part of d1, d2 which share the same key '''
		keys = self.get_shared_keys(d1,d2)
		d1_shd={}
		d2_shd={}
		for k in keys:
			d1_shd[k] = d1[k]
			d2_shd[k] = d2[k]
		return d1_shd,d2_shd
	def get_diff_keys(self,d1, d2):
		'''get the keys of d1 that are not in d2: key(d1)-key(d2)'''
		keys = []
		keys = [k for k in d1 if not k in d2]
		return keys
	def get_diff_dict(self,d1, d2):
		'''get the difference of two dictionaries which corresponds to d1-d2 '''
		keys = self.get_diff_keys(d1,d2)
		d1_diff={}
		for k in keys:
			d1_diff[k] = d1[k]
		return d1_diff
	def get_dict_val_interval(self,d,min_val=0,max_val=100):
		'''returns that part of dictionary d whose value is in the exclusive interval of (min_val max_val)'''
		d_mv = {}
		for k,v in d.items():
			if (v > min_val and v < max_val):
				d_mv[k]=v	
		return d_mv
        def get_d1vsd2_sharedkeys(self, d1, d2):
                    d1_shd, d2_shd = self.get_shared_dicts(d1, d2)
                    keys = d1_shd.keys()
                    d1_values = []
                    d2_values = []
                    for k in keys:
                        d1_values.append(d1_shd[k])
                        d2_values.append(d2_shd[k])
                    return keys, d1_values, d2_values
                        
	def save_d(self, d, filename):
		'''save a dictionary into a file'''
		f = open(filename, 'w')
		pickle.dump(d, f)
		f.close()
	def load_d(self, filename):
		'''load a dictionary from a file'''
		f = open(filename, 'r')
		d = pickle.load(f)
		f.close()
		return d
	def writetofile(self,d,filename,sort='value'):
		f = open(filename,'w')
		srtd_tuple = self.sort_dictionary(d,sort=sort)
		for k,v in srtd_tuple:
			f.write(str(v)+' '+k+'\n')
		f.close()

	def split_key(self,d,n=0):
		'''split every key of a dictionary and create a dictionary with nth element of splitted key and its corresponding value
		Example: d={'w1 k1':5, 'w1 k2':3, 'w2 k1':4}
		n=0 => returns d1={'w1':8, 'w2':4}
		n=1 => returns d1={'k1':9, 'k2':3}
		'''
		d1={}
		for key,val in d.items():
			key_list = key.split()	
			w = key_list[n]
			d1[w] = d1.get(w,0)+val	
		return d1
		
	def check_same_key(self, d1, d2):
		#Check if dictionaries d1 and d2 have the same key
        	if sorted(d1) != sorted(d2):
                	raise Exception, "dictionaries don't still have the same probability space"

	def lookup_dict(self,wordlist,ref_d):
		#return the values of the corresponding wordlist in ref_d
		p = []
	        for w in wordlist:
        	        w = w.lower()
               		if w in ref_d:
                        	p.append(float(ref_d[w]))
	                else:
        	                p.append(0.0)
		return p
	
	def adjust_lineartransform(self,d1,d2):
		#TODOl: where should I put this function 
		'''given linear relation between d1 and d2, this function linearly transforms the values in d1 such that , for example
		d1[k]=400, d2[k]=800, this function returns d[k]=400*(400/800) '''
		self.check_same_key(d1,d2)
		d={}
		for k in d1:
			v1 = d1[k]
			v2 = d2[k]
			v = v1 * v1 * (1./v2)
			d[k] = round(v)
		return d
	def lineartransform(self,d,a,b):
		d_={}
		for k,v in d.items():
			v_= v*a + b		
			d_[k] = v_
		return d_

	def del_key_d(self,key,d):
		'''delete a key from a dictionary '''
		#d = copy.deepcopy(d_)
		for k_d in d.keys():
			pat = '\W'+key+'$'+'|'+'^'+key+'\W'
			if re.search(pat,k_d, re.IGNORECASE):
			        del d[k_d]
		return d
	def del_keypair_d(self,key,d):
		'''delete a key which is a pair from a dictionary '''
		#d = copy.deepcopy(d_)
		for k_d in d.keys():
			if key == k_d:
			        del d[k_d]
		return d
	def merge_2dicts_disreg_ordkeypairs(self, d1, d2):
		'''merge two dictionaries containing pair of words while the order of words is disregarded:
		d1={'w1 w2':3, 'w2 w3':4}
		d2={'w2 w1':12, 'w2 w3':4, 'w4 w5':5}
		returns d_mrg={'w1 w2':15, 'w2 w3':4, 'w4 w5':5}
		''' 
		d_mrg = {}
		for pair in d1:
			w1, w2 = pair.split()
			rev_pair = w2 + ' ' + w1
			if rev_pair in d2:
				d_mrg[pair] = d1[pair] + d2[rev_pair]
			else:
				d_mrg[pair] = d1[pair]
		for pair in d2:
			w1, w2 = pair.split()
			rev_pair = w2 + ' ' + w1
			if not (pair in d_mrg or rev_pair in d_mrg):
				d_mrg[pair] = d2[pair]
		return d_mrg
	

#******************************************
class FileReader(object):
        '''having functions to read different types of files that might correspond to different distributions'''
        def __init__(self):
                pass
	#TODOl: adapting these functions to read several types of input files
	def read_words_d_file_r(self, filename):
                ''' reads a file containing the distribution of words and returns a dictionary with words and freqs as its keys and values'''
                f = open(filename,'r')
                d = {}
                for l in f.readlines():
                        fq,w = l.split()
                        d[w.lower()] = float(fq)
                f.close()
                return d
        
        def read_pairs_d_file_r(self, filename):
                ''' reads a file containing the distribution of pairs and returns a dictionary with pairs and freqs as its keys and values'''
                f  = open(filename,'r')
                d = {}
                for l in f.readlines():
                        p,fq = l.split(':')
                        d[p.lower()] = float(fq)
                f.close()
                return d
        def read_words_d_spcw_file_r(self, filename,w_list):
                ''' reads a file containing the distribution of words and only extracts the word from them'''
                d = self.read_words_d_file(filename)
		w_l = [w.lower() for w in w_list]
                d1={}
                #TODOl: make it pythonic list comprehension
		for w in w_list:
                       	if w in d:
                        	d1[w] = d[w]
                return d1
        def read_pairs_d_spcw_file_r(self, filename, w_list):
                '''reads a file containing all pair distiributions and only extracts the pairs having specific words, indicated in w_list, as their first word'''
                d = self.read_pairs_d_file_r(filename)
		w_l = [w.lower() for w in w_list]
                d1={}
                #TODOl: make it pythonic list comprehension
		for w in w_list:
                	for k in d: 
                        	if k.startswith(w+' '):
                                	d1[k] = d[k]
                return d1
 
        def read_words_d_file(self, filename):
                ''' reads a file containing the distribution of words and returns a dictionary with words and freqs as its keys and values'''
                f = open(filename,'r')
                d = {}
                for l in f.readlines():
                        ws = l.split()
			fq = ws[0]
			w = ' '.join(ws[1:])
                        d[w.lower()] = float(fq)
                f.close()
                return d

	def read_words_d_separate_files(self,wordfile,probfile):
		'''reads two separate files containing words and their corresponding probs respectively.'''
		wf = open(wordfile, 'r')
		pf = open(probfile, 'r')
		p_list = []
		d = {}
		for l in pf.readlines():
			line = l.split()
			p = float(line[0])
			p_list.append(p)
		print 'p_list done!'
		pf.close()
		idx = 0
		for l in wf.readlines():
			d[l] = p_list[idx]
			idx = idx+1
		wf.close()
		return d
        def read_words_d_spcw_file(self, filename,w_list):
                ''' reads a file containing the distribution of words and only extracts the word from them'''
                d = self.read_words_d_file(filename)
		w_l = [w.lower() for w in w_list]
                d1={}
                #TODOl: make it pythonic list comprehension
		for w in w_list:
                       	if w in d:
                        	d1[w] = d[w]
                return d1
        

        def read_pairs_d_spcw_file(self, filename, w_list):
                '''reads a file containing all pair distiributions and only extracts the pairs having specific words, indicated in w_list, as their first word'''
                #d = self.read_pairs_d_file(filename)
                d = self.read_words_d_file(filename)
		w_l = [w.lower() for w in w_list]
                d1={}
                #TODOl: make it pythonic list comprehension
		for w in w_list:
                	for k in d: 
                        	if k.startswith(w+' '):
                                	d1[k] = d[k]
                return d1
	def cut_line_words_pos(self,filename_in, filename_out, wordlist, pos):
		'''if a line contains a word of wordlist in the specified position, the line is cutted before that word and written in the output file '''
		f_in = open(filename_in, 'r')
		f_out = open(filename_out, 'w')
		selected_lines = []
		for l in f_in.readlines():
			words_l = l.split()
			if len(words_l)>pos:
				for spcw in wordlist:
					if (words_l[pos] == spcw):
						a = ' '.join(words_l[:pos])
						f_out.write(a+'\n')
						break

	def columnSelector (self, filename_in, filename_out, col_width, cond='greater'):
		'''the python equivalent to the perl package ColumnSelector.pm 
		it selects sentences if their number of words are EQUAL/GREATER (cond) than COL_WIDTH
		and it only returns the first COL_WIDTH of words of the selected sentence'''

		f_in = open(filename_in, 'r')
		f_out = open(filename_out, 'w')
		for l in f_in.readlines():
			words_l = l.split()
			sent_sz = len(words_l)
			if (cond == 'greater') and (sent_sz >= col_width):
				a = ' '.join(words_l[:col_width])
				f_out.write(a+'\n')		
			elif (cond == 'equal') and (sent_sz == col_width):
				a = ' '.join(words_l[:col_width])
				f_out.write(a+'\n')		

		f_in.close()
		f_out.close()		

#*******************************************       
class DictPlotter(object):
        '''plots dictionaries: values vs. keys'''
        def __init__(self):
                self.ud = UtilDict()

	def plot_xy(self,x,y,fig=None,xlab='X',ylab='Y',title='X vs. Y',linestyle='bo'):
                if not fig: fig = plt.figure(1)
		width = 0.5
                xlocations = np.array(range(len(x)))+width
                #print 'xlocation:',xlocations
		#TODOl Ali: solve yticks!
                #plt.yticks(np.arange(0,0.005 ,0.001))
                #if len(labels) < 40:
                #        plt.xticks(xlocations+ width/2, labels, rotation=70,fontsize=8)
                plt.xlim(0, xlocations[-1]+width*2)
                #plt.ylim(0, .5)
		plt.plot(x,y,linestyle)
                plt.xlabel(xlab, fontsize=24)
                plt.ylabel(ylab,fontsize=24)
                plt.title(title,fontsize=28)
              	return fig 
        def plot_d(self,d,sort='key',fig=None,xlab='word',ylab='Probability',title='Words Distribution'):
                ''' plots one dictionary which might be the distribution of words/pairs'''
                if not fig: fig = plt.figure(1)
                d_tuple = self.ud.sort_dictionary(d, sort)
		print 'd:',d_tuple
                labels, data  = zip(*d_tuple)
                width = 0.5
                xlocations = np.array(range(len(data)))+width
                #print 'xlocation:',xlocations
		#TODOl Ali: solve yticks!
                #plt.yticks(np.arange(0,0.005 ,0.001))
                #if len(labels) < 40:
                #        plt.xticks(xlocations+ width/2, labels, rotation=70,fontsize=8)
                plt.xticks(xlocations+ width/2, labels, rotation=70,fontsize=8)
                plt.xlim(0, xlocations[-1]+width*2)
                #plt.ylim(0, .5)
                plt.bar(xlocations, data)
                plt.xlabel(xlab, fontsize=24)
                plt.ylabel(ylab,fontsize=24)
                plt.title(title,fontsize=28)
                #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
                #plt.axis([40, 160, 0, 0.03])
                plt.grid(True)
                return fig


        def plot_2d(self, d1,d2, fig=None, sort='key',xlab='word', ylab='Probability', title1='Distribution1',title2='Distribution2'):
                '''plot two dictionaries, which might be the distributions of words or pairs, on the same figure'''
                plt.ion()
                fig = plt.figure(1)
                #plt.subplot(211)
                fig.add_axes([.1,.55,.8,.4]) # [left, bottom, width, height]
                ax1=plt.gca()
                self.plot_d(d1, sort=sort, fig=fig, xlab=xlab, ylab=ylab, title=title1)
                ax1.set_xlabel('')
                #plt.subplot(212)
                fig.add_axes([.1,.07,.8,.4])
                ax2=plt.gca()
                self.plot_d(d2, sort=sort, fig=fig, xlab=xlab, ylab=ylab, title=title2)
                #plt.show()
                return fig

class RegressionAnalysis(object):
	'''different regression methods'''
	def __init__(self):
		pass

	def linreg(self,x,y):
		t=[1, 2, 3, 4, 5]
		xn=[1,3,2,4,5]
		(a_s,b_s,r,tt,stderr)=stats.linregress(x,y)
		print('Linear regression using stats.linregress')
		print('regression: a=%.2f b=%.2f, std error= %.3f' % (a_s,b_s,stderr))
		x=polyval([a_s,b_s],t)
		plt.plot(t,xn,'bo')
		plt.plot(t,x,'g-')
		plt.legend(['original','regression'])
		plt.xlim([-.5,5.5])
		plt.ylim([-.5,5.5])
		plt.show()


def avg_len(filename,max_sent_len=100):
	'''average length of sentences in a file: 
	n_snt_ln_lst = [0 0 2 4 2 0 2]
	p_ln_lst = [0 0 .2 .4 .2 0 .2]
	l_avg = 2*.2 + 4*.4 + 2*.2 + 2*.2
	'''
	n_snt_ln_lst = get_len_list(filename,max_sent_len)
	s = float(sum(n_snt_ln_lst))
	p_ln_lst = [i/s for i in n_snt_ln_lst]
	l_avg = 0
	for idx,v in enumerate(p_ln_lst):
		l_avg = l_avg + idx*v
	return l_avg

def get_len_list(filename,max_sent_len=100):
	'''gets a list having the information about the lengths of sentences in a file: len_list[3] indicates the number of sentences with the length of 3 '''
	n_snt_ln_lst = [0*v for v in range(1,max_sent_len)]	
	f = open(filename,'r')
	for l in f.readlines():
		w_l = len(l.split())
		n_snt_ln_lst[w_l] = n_snt_ln_lst[w_l] + 1
	return n_snt_ln_lst 	
