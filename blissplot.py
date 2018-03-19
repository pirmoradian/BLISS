import blissutil 
import pickle 
import KLdiv
#from scipy import stats,polyval
import CountFreq
import random
import os
import pybliss as pb
import cnstpybliss as c
import pdb
import copy
import numpy as np
import matplotlib.pyplot as plt
#main functions for plotting distributions of bliss, calling several classes in each function

rd = blissutil.FileReader()
p = blissutil.DictPlotter()
ud = blissutil.UtilDict()
gr = pb.GrammarReader(c.GRAMMAR_FILE)

def read_f_plot(filename, sort='key',fig=None, dist='word',xlab='word',title='Words Distribution'):
	'''reads a file and plot the dictionary corressponding to this file'''
	#if dist == 'word':
	d = rd.read_words_d_file(filename)
	#elif dist == 'pair':
	#	d = rd.read_pairs_d_file(filename)
	#else: 
	#	raise Exception, 'what is the distribution?'
	
	d = ud.normalize_d_values(d)
	p.plot_d(d,sort=sort,fig=fig,xlab=xlab,title=title)

def read_2f_plot(filename1, filename2, sort='key',fig=None, dist='word',xlab='word',ylab='Frequency',title1='Distribution1',title2='Distribution2'):
	#if dist == 'word':
	d1 = rd.read_words_d_file(filename1)
	d2 = rd.read_words_d_file(filename2)
	#elif dist == 'pair':
	#	d1 = rd.read_pairs_d_file(filename1)
	#	d2 = rd.read_pairs_d_file(filename2)
	#else: 
	#	raise Exception, 'what is the distribution?'
	
	d1 = ud.normalize_d_values(d1)
	d2 = ud.normalize_d_values(d2)
	p.plot_2d(d1,d2,sort=sort,fig=fig,xlab=xlab,ylab=ylab,title1=title1,title2=title2)

def read_2f_plot_xy(filename1, filename2, sort='key',fig=None, dist='word',xlab='word',ylab='freq',title='Dist1 vs. Dist2'):
	#if dist == 'word':
	d1 = rd.read_words_d_file(filename1)
	d2 = rd.read_words_d_file(filename2)
	#elif dist == 'pair':
	#	d1 = rd.read_pairs_d_file(filename1)
	#	d2 = rd.read_pairs_d_file(filename2)
	#else: 
	#	raise Exception, 'what is the distribution?'
	d1 = ud.normalize_d_values(d1)
	d2 = ud.normalize_d_values(d2)
	ud.check_same_key(d1,d2)
	d1_tuple = ud.sort_dictionary(d1,sort=sort)	
	d2_tuple = ud.sort_dictionary(d2,sort=sort)	
	k1,v1 = zip(*d1_tuple)
	k2,v2 = zip(*d2_tuple)
	#print 'x vs y:',v1,v2
	fig = p.plot_xy(v1,v2,fig=fig,xlab=xlab,ylab=ylab,title=title)
	return fig

def read_2f_shd_plot(filename1, filename2, sort='key',fig=None, dist='word',xlab='word',title1='Distribution1',title2='Distribution2'):
	'''reads two files and plot the dictionaries with keys shared by both files'''
	#if dist == 'word':
	d1 = rd.read_words_d_file(filename1)
	d2 = rd.read_words_d_file(filename2)
	#elif dist == 'pair':
	#	d1 = rd.read_pairs_d_file(filename1)
	#	d2 = rd.read_pairs_d_file(filename2)
	#else: 
	#	raise Exception, 'what is the distribution?'
	d1,d2 = ud.get_shared_dicts(d1,d2)
	minval=2000
	maxval=4000
	d1 = ud.get_dict_val_interval(d1,minval,maxval)
	d2 = ud.get_dict_val_interval(d2,minval,maxval)
	#d1 = ud.normalize_d_values(d1)
	#d2 = ud.normalize_d_values(d2)
	
	p.plot_2d(d1,d2,sort=sort,fig=fig,xlab=xlab,title1=title1, title2=title2)
		
#NOT SURE TO IMPLEMENT THIS FUNCTION OR NOT!
def read_2f_shd_save(filename1, filename2,savefile1, savefile2,  probfile1='',probfile2='',sort='key',fig=None, dist='word',xlab='word',title1='Distribution1',title2='Distribution2'):
	'''reads two files and finds the dictionaries with keys shared by both files and then save these dictionaries into two diff files'''
	if (dist == 'word' or dist=='pair'):
		d1 = rd.read_words_d_file(filename1)
		d2 = rd.read_words_d_file(filename2)
	#elif dist == 'pair':
	#	d1 = rd.read_pairs_d_file(filename1)
	#	d2 = rd.read_pairs_d_file(filename2)
	elif dist == 'word_sep':
		d1 = rd.read_words_d_separate_files(filename1,probfile1)
		d2 = rd.read_words_d_separate_files(filename2,probfile2)
	else: 
		raise Exception, 'what is the distribution?'
	print 'len d1 before sharing:', len(d1)
	print 'len d2 before sharing:', len(d2)
	d1,d2 = ud.get_shared_dicts(d1,d2)
	print 'len d1 after sharing', len(d1)
	print 'len d2 after sharing', len(d2)
	ud.save_d(d1, savefile1)
	ud.save_d(d2, savefile2)	

#TODOl: adapting this function to different types of inputs
def read_2f_spcw_plot_r(filename1, filename2, w_list=['sword'],sort='key',fig=None,dist='pair',xlab='Pairs',title1='Distribution1',title2='Distribution2'):
	if dist == 'word':
		d1 = rd.read_words_d_spcw_file_r(filename1,w_list)
		d2 = rd.read_words_d_spcw_file_r(filename2,w_list)
	elif dist == 'pair':
		d1 = rd.read_pairs_d_spcw_file_r(filename1,w_list)
		d2 = rd.read_pairs_d_spcw_file_r(filename2,w_list)
	else: 
		raise Exception, 'what is the distribution?'
	for w in w_list:
		title1 = title1+','+w
		title2 = title2+','+w

	d1 = ud.normalize_d_values(d1)
	d2 = ud.normalize_d_values(d2)
	p.plot_2d(d1,d2,sort=sort,fig=fig,xlab=xlab,title1=title1,title2=title2)

def read_2f_spcw_plot(filename1, filename2, w_list=['sword'],sort='key',fig=None,dist='pair',xlab='Pairs',title1='Distribution1',title2='Distribution2'):
	if dist == 'word':
		d1 = rd.read_words_d_spcw_file(filename1,w_list)
		d2 = rd.read_words_d_spcw_file(filename2,w_list)
	elif dist == 'pair':
		d1 = rd.read_pairs_d_spcw_file(filename1,w_list)
		d2 = rd.read_pairs_d_spcw_file(filename2,w_list)
	else: 
		raise Exception, 'what is the distribution?'
	for w in w_list:
		title1 = title1+','+w
		title2 = title2+','+w
	d1 = ud.normalize_d_values(d1)
	d2 = ud.normalize_d_values(d2)
	p.plot_2d(d1,d2,sort=sort,fig=fig,xlab=xlab,title1=title1,title2=title2)

def load_2d_kldiv(filename1,filename2):
	d1 = ud.load_d(filename1)
	d2 = ud.load_d(filename2)
	
	d1 = ud.normalize_d_values(d1)
	d2 = ud.normalize_d_values(d2)
	
	print 'shannond d1', KLdiv.shannon_entropy(d1)
	print 'shannond d2', KLdiv.shannon_entropy(d2)
	k = KLdiv.KL_divergence(d1,d2)
	return k
def load_2d_shd_save(filename1,filename2,savefile1,savefile2):
	d1 = ud.load_d(filename1)
	d2 = ud.load_d(filename2)
	print 'len d1 before sharing:', len(d1)
	print 'len d2 before sharing:', len(d2)
	d1,d2 = ud.get_shared_dicts(d1,d2)
	print 'len d1 after sharing', len(d1)
	print 'len d2 after sharing', len(d2)
	ud.save_d(d1, savefile1)
	ud.save_d(d2, savefile2)	

def load_2d_diff_save(filename1,filename2,savefile1):
	d1 = ud.load_d(filename1)
	d2 = ud.load_d(filename2)
	print 'len d1 before subtracting:', len(d1)
	print 'len d2 before subtracting:', len(d2)
	d1 = ud.get_diff_dict(d1,d2)
	print 'len d1 after subtracting', len(d1)
	print 'len d2 after subtracting', len(d2)
	ud.save_d(d1, savefile1)
def load_d_splitkey_save(filename,savefile,n=0):
	'''loads a dictionary and after splitting its keys, it saves the dictionary corresponding to nth element and value'''
	d = ud.load_d(filename)
	d_sp = ud.split_key(d,n=n)
	print len(d_sp)
	#print d_sp
	ud.save_d(d_sp,savefile)
	
def read_2f_adjust(filename1,filename2,algorithm='linear'):
	'''reads two (frequency) files and returns a dictionary after calling adjust_lineartransform() of utildict class'''
	d1 = rd.read_words_d_file(filename1)
	d2 = rd.read_words_d_file(filename2)
	if (algorithm == 'linear'):
		d = ud.adjust_lineartransform(d1,d2)
	else:
		raise Exception, 'what is algorithm of adjustment?!'
	return d
def read_f_lineartransform(filename,a,b,algorithm='linear'):
	'''reads two (frequency) files and returns a dictionary after calling adjust_lineartransform() of utildict class'''
	d = rd.read_words_d_file(filename)
	#print 'd:',d
	if (algorithm == 'linear'):
		d = ud.lineartransform(d,a,b)
	else:
		raise Exception, 'what is algorithm of adjustment?!'
	#print 'd transfromed:',d
	d = ud.normalize_d_values(d)
	return d
	
def read_2f_linreg_plot_xy(filename1, filename2, sort='key',fig=None, dist='word',xlab='word',ylab='freq',title='Dist1 vs. Dist2'):
	#if dist == 'word':
	d1 = rd.read_words_d_file(filename1)
	d2 = rd.read_words_d_file(filename2)
	#elif dist == 'pair':
	#	d1 = rd.read_pairs_d_file(filename1)
	#	d2 = rd.read_pairs_d_file(filename2)
	#else: 
	#	raise Exception, 'what is the distribution?'
	d1 = ud.normalize_d_values(d1)
	d2 = ud.normalize_d_values(d2)
	ud.check_same_key(d1,d2)
	d1_tuple = ud.sort_dictionary(d1,sort=sort)	
	d2_tuple = ud.sort_dictionary(d2,sort=sort)	
	k1,v1 = zip(*d1_tuple)
	k2,v2 = zip(*d2_tuple)
	#print 'x vs y:',v1,v2
	#fig = p.plot_xy(v1,v2,fig=fig,xlab=xlab,ylab=ylab,title=title)

	#t=[1, 2, 3, 4, 5]
        #xn=[1,3,2,4,5]
        (a_s,b_s,r,tt,stderr)=stats.linregress(v1,v2)
        print('Linear regression using stats.linregress')
        print('regression: a=%.2f b=%.2f, std error= %.3f' % (a_s,b_s,stderr))
        y=polyval([a_s,b_s],v1)
        #plt.plot(t,xn,'bo')
	fig = p.plot_xy(v1,v2,fig=fig,xlab=xlab,ylab=ylab,title=title,linestyle='bo')
	fig = p.plot_xy(v1,y,fig=fig,xlab=xlab,ylab=ylab,title=title,linestyle='r-')
        #plt.plot(t,x,'g-')
        #fig.legend(['original','regression'])
        #plt.xlim([-.5,5.5])
        #plt.ylim([-.5,5.5])
	return fig
def read_prdnfile_count_convert_d(filename1,wordlist,pos):
	'''looks for words of 'wordlist' in the position 'pos' in filename1 and returns the frequency of all the preceded words '''
	filename2 = 'temp'+str(random.random())
	filename3 = 'temp'+str(random.random())
	rd.cut_line_words_pos(filename1, filename2, wordlist, pos)
	CountFreq.CountFrequency(filename2,filename3).get_linefreq()
	d = rd.read_words_d_file(filename3)
	os.remove(filename2)
	os.remove(filename3)
	return d
def get_kldiv_lxcat(filename1,filename2,outfile):
	'''filename1 and filename2 contains word frequencies (wFreq*.txt), words of different lexical categories, i.e nouns, verbs, etc., are extracted and the KL-distances of these two files are calculated 
	**HINT: Check the GRAMMAR file written in cnstpybliss.py 
	**HINT: Check the SYMS constants in the beginning of pybliss.py '''
	ref_w_d1 = pb.convertfiletodic(filename1, ' ',1, 0)
	ref_w_d2 = pb.convertfiletodic(filename2, ' ',1, 0)
	f_out = open(outfile,'a')
	f_out.write('\nINPUTS:  '+filename1+' vs. '+filename2+'\n')
	lxcats = [pb.NOUN_SYMS, pb.VERB_SYMS, pb.PREP_SYMS, pb.ADJ_SYMS]
	gr.readgrammar()

	d1 = ud.normalize_d_values(ref_w_d1)
        d2 = ud.normalize_d_values(ref_w_d2)
	kd1 = KLdiv.KL_divergence(d1,d2)
	kd2 = KLdiv.KL_divergence(d2,d1)
       	f_out.write('KL-d:All words'+'\t' + str(kd1) +'\t'+str(kd2)+'\n')
	for lxc in lxcats:
		d1 = {}
		d2 = {}
		words = get_words_lxcat(lxc)
		for w in words:
			if w not in ref_w_d1:
				print 'WARNING: ',w,' is not in dictionary'
				continue 
			d1[w] = float(ref_w_d1[w])
			d2[w] = float(ref_w_d2[w])
		#print 'lxc:', lxc
		#print 'd1:',d1
		#print 'd2:',d2
		#pdb.set_trace()
		d1 = ud.normalize_d_values(d1)
	        d2 = ud.normalize_d_values(d2)
		#print 'len:',len(d1)
		#pdb.set_trace()
		kd1 = KLdiv.KL_divergence(d1,d2)
		kd2 = KLdiv.KL_divergence(d2,d1)
		#print 'kd:',kd1,' ',kd2
        	f_out.write('KL-d:'+str(lxc)+'\t' + str(kd1) +'\t'+str(kd2)+'\n')
		#print '*****************'
	f_out.close()
def get_words_lxcat(lxc):
	'''all words of the category lxc are found'''	
	words = []
	if not gr.d:
		gr.readgrammar()
	for sym in lxc:
		ws = gr.d[sym]['rule']
		for i in range(len(ws)):
			w = ws[i][0].lower()
			words.append(w)
	return words

def get_kldiv_lxcat_pairs(filename1,filename2,outfile):
	'''filename1 and filename2 contains pair frequencies (pFreq*.txt); pairs of lexical categories, i.e. N-V, V-N, Adj-N,etc., are extracted from these two files and their KL-distances are calculated '''
	'''**HINT: Check the GRAMMAR file written in cnstpybliss.py '''
	'''**HINT: Check the SYMS constants in the beginning of pybliss.py '''
	ref_p_d1 = rd.read_words_d_file(filename1)
	ref_p_d2 = rd.read_words_d_file(filename2)
	f_out = open(outfile,'a')
	f_out.write('\n'+'INPUTS:  '+filename1+' vs. '+filename2+'\n')
	lxcats = [[pb.NOUN_SYMS, pb.VERB_SYMS],[pb.VERB_SYMS, pb.NOUN_SYMS], [pb.PREP_SYMS,pb.NOUN_SYMS], [pb.ADJ_SYMS,pb.NOUN_SYMS],[pb.PREP_SYMS,pb.ADJ_SYMS], [pb.NOUN_SYMS, pb.PREP_SYMS]]
	#lxcats = [[pb.NOUN_SYMS,pb.VERB_SYMS]]
	gr.readgrammar()
	
	for lxc in lxcats:
		d1 = {}
		d2 = {}
		pairs = get_pairs_lxcat(lxc)
		for p in pairs:	
			if (p not in ref_p_d1) or (p not in ref_p_d2):
				print 'WARNING: ',p,' is not in dictionary'
				continue 
			d1[p] = float(ref_p_d1[p])
			d2[p] = float(ref_p_d2[p])
		
		#print 'lxc:', lxc
		#print 'd1:',d1
		#print 'd2:',d2
		#pdb.set_trace()
		d1 = ud.normalize_d_values(d1)
	        d2 = ud.normalize_d_values(d2)
		kd1 = KLdiv.KL_divergence(d1,d2)
		kd2 = KLdiv.KL_divergence(d2,d1)
		#print 'kd:',kd
        	f_out.write('KL-d:'+str(lxc)+'\t' + str(kd1) +'\t'+str(kd2)+'\n')
		#print '*****************'
	f_out.close()
	ud.writetofile(d1,'n-v'+filename1)
	#ud.writetofile(d2,'n-v'+filename2)
	ud.writetofile(d2,'n-v2')
def get_pairs_lxcat(lxc):
	'''return all pairs whose first word is of the category lxc[0] and whose second word is of the category lxc[1] are found'''	
	lxc0 = lxc[0]
	lxc1 = lxc[1]
	pairs = []
	for sym0 in lxc0:
		for sym1 in lxc1:
			words0 = gr.d[sym0]['rule']
			words1 = gr.d[sym1]['rule']
			for i in range(len(words0)):
				w0 = words0[i][0].lower()
				for j in range(len(words1)):
					w1 = words1[j][0].lower()
					p = w0 +' '+w1
					pairs.append(p)
	return pairs	
def get_entropy_lxcat_pairs(filename,outfile):
	'''calculates the entropy of separate pairs of lexical categories, i.e. nouns-verbs, verbs-nouns, adj-nouns, etc., of filename and prints them into outfile '''
	ref_d = rd.read_words_d_file(filename)
	f_out = open(outfile,'a')
	f_out.write('\n'+'INPUTS:  '+filename+'\n')
	lxcats = [[pb.NOUN_SYMS, pb.VERB_SYMS],[pb.VERB_SYMS, pb.NOUN_SYMS], [pb.PREP_SYMS,pb.NOUN_SYMS], [pb.ADJ_SYMS,pb.NOUN_SYMS],[pb.PREP_SYMS,pb.ADJ_SYMS], [pb.NOUN_SYMS, pb.PREP_SYMS]]
	gr.readgrammar()
	for lxc in lxcats:
		d = {}
		pairs = get_pairs_lxcat(lxc)
		for p in pairs:	
			if p not in ref_d:
				print 'WARNING: ',p,' is not in dictionary'
				continue 
			d[p] = float(ref_d[p])

		#print 'lxc:', lxc
		#print 'd:',d
		#pdb.set_trace()
		d = ud.normalize_d_values(d)
		ent = KLdiv.shannon_entropy(d)
		#print 'ent:',ent
        	f_out.write('Entropy:'+str(lxc)+'\t' + str(ent) +'\n')
		#print '*****************'
	f_out.close()

def get_entropy_lxcat(filename,outfile):
	'''calculates the entropy of separate lexical categories, i.e. nouns, verbs, adj, and prepositions, of filename and prints them into outfile '''
	ref_d = rd.read_words_d_file(filename)
	f_out = open(outfile,'a')
	f_out.write('\n'+'INPUT:  '+filename+'\n')
	lxcats = [pb.NOUN_SYMS, pb.VERB_SYMS, pb.PREP_SYMS, pb.ADJ_SYMS]
	gr.readgrammar()
	for lxc in lxcats:
		d = {}
		words = get_words_lxcat(lxc)
		for w in words:	
			if w not in ref_d:
				print 'WARNING: ',w,' is not in dictionary'
				continue 
			d[w] = float(ref_d[w])

		#print 'lxc:', lxc
		#print 'd:',d
		#pdb.set_trace()
		d = ud.normalize_d_values(d)
		ent = KLdiv.shannon_entropy(d)
		#print 'ent:',ent
        	f_out.write('Entropy:'+str(lxc)+'\t' + str(ent) +'\n')
		#print '*****************'
	f_out.close()
def get_subject_freq_file(filename1,filename2):
	'''subjects of filename1,*svprd file, which are words preceding verbs in filename1, is drawn and their frequencies are written to filename2:
	filename1 = 'BLISS_sbjwd_27Aug.txt_svprd'
        filename2 = 'sbjFreq_sbjwd_27Aug.txt'
	'''
	outfile = 'temp'+str(random.random())
        lxc = pb.VERB_SYMS
        wordlist = get_words_lxcat(lxc)
        pos = 1
        rd.cut_line_words_pos(filename1,outfile,wordlist,pos)
	CountFreq.CountFrequency(outfile,filename2).get_linefreq()

def get_cat_nouns_probs():
	'''returns a list of list of nouns and their probs while nouns are categorized to Animals, objects, and buildings '''
	#gr.readgrammar()
	cats_files = [open(f).readlines() for f in c.NOUN_CATEGORY_FILES]
	cats_nouns = [[l.rstrip('\n') for l in f] for f in cats_files]
	priorpr_d = {}
	nouns = gr.d[pb.SG_NOUN_SYMS[0]]['rule']
	probs = gr.d[pb.SG_NOUN_SYMS[0]]['prob']
	for idx,w in enumerate(nouns):
		w = w[0]
		priorpr_d[w] = probs[idx]
	cats_probs = [[priorpr_d[n] for n in n_c] for n_c in cats_nouns]
	return cats_nouns, cats_probs

def get_expectedval_cat_verb(nouns, probs, verb, fq_d):
	'''retruns the expected value of the frequency of having 'nouns' belonging to the same category with 'verb' '''
	expv = 0 
	for i in range(len(nouns)):
		n = nouns[i]
		p = probs[i]
		pair = n+' '+verb
		if pair in fq_d:
			expv = expv + p * fq_d[pair]
	return expv

def calculate_smoothed_probs(outfile):
	'''calculate the smoothed joint probs of nouns and verbs, considering the category of nouns '''
	gr.readgrammar()
	ref_fq_d = pb.convertfiletodic('jntFqallSelAjPr5NVshake.txt',':')
	for k,v in ref_fq_d.iteritems():
		ref_fq_d[k] = float(v)
	#pdb.set_trace()
	smoothed_d = copy.deepcopy(ref_fq_d)
	cats_nouns, cats_probs = get_cat_nouns_probs()
	verbs = [v[0] for sym in pb.PL_VERB_SYMS for v in gr.d[sym]['rule']]
	#pdb.set_trace()
	cats_probs = [[p/float(sum(c_p)) for p in c_p] for c_p in cats_probs]
	#pdb.set_trace()
	for i_cat in range(len(cats_nouns)):
		nouns_c = cats_nouns[i_cat]
		probs_c = cats_probs[i_cat]
		for v in verbs:
			fq_expv = get_expectedval_cat_verb(nouns_c, probs_c, v, ref_fq_d)
			for i in range(len(nouns_c)):
				n = nouns_c[i]
				p = probs_c[i]
				pair = n + ' ' + v
				if pair in ref_fq_d:
					fq = ref_fq_d[pair]
				else:
					fq = 0
				smoothed_d[pair] = (1-c.CAT_EFFECT)*fq + c.CAT_EFFECT *p* fq_expv
		#pdb.set_trace()
	f = open(outfile,'w')
	smoothed_tuple = ud.sort_dictionary(smoothed_d,sort='key')
	for k,v in smoothed_tuple:
		f.write(k + ': ' + str(v) + '\n')
	f.close()
	#ud.writetofile(smoothed_d,outfile)

def modify_jntprobdict_wordcat(filename,outfile):
	'''deletes the words of a specific category wherever appear in the jntprob file '''
        ref_d = rd.read_pairs_d_file_r(filename)
        lxcats = [pb.PREP_SYMS] #,pb.ADJ_SYMS]
        gr.readgrammar()
	
        for lxc in lxcats:
                words = get_words_lxcat(lxc)
		#words = ['good']
                for w in words:
                        #if w not in ref_d:
                        #        print 'WARNING: ',w,' is not in dictionary'
                        #        continue
                        ref_d = ud.del_key_d(w, ref_d)
	f = open(outfile,'w')

	smoothed_tuple = ud.sort_dictionary(ref_d,sort='key')
	for k,v in smoothed_tuple:
		f.write(k + ': ' + str(v) + '\n')
	f.close()
	#ud.writetofile(ref_d, outfile)	
def modify_jntprobdict_paircat(filename,outfile):
	'''deletes the pairs of words each belongings to a specific category '''
        ref_d = rd.read_pairs_d_file_r(filename)
        lxcats1 = [pb.PL_VERB_SYMS] #,pb.ADJ_SYMS]
        lxcats2 = [pb.PL_VERB_SYMS] #,pb.ADJ_SYMS]
        gr.readgrammar()
	
        for lxc1 in lxcats1:
	        words1 = get_words_lxcat(lxc1)
		for lxc2 in lxcats2:
	                words2 = get_words_lxcat(lxc2)
        	        for w1 in words1:
				for w2 in words2:
	                	        #if w not in ref_d:
        	                	#        print 'WARNING: ',w,' is not in dictionary'
	        	                #        continue
					p = w1+' '+w2
        	        	        ref_d = ud.del_keypair_d(p, ref_d)
	f = open(outfile,'w')

	smoothed_tuple = ud.sort_dictionary(ref_d,sort='key')
	for k,v in smoothed_tuple:
		f.write(k + ': ' + str(v) + '\n')
	f.close()
	#ud.writetofile(ref_d, outfile)	


def get_word_d_pairdict(filename,outfile):
	'''gets a dictionary of words which are in the keys of a dictinary of pairs, while the frequency of each word is the sum of all its occurrences in the pair dictionary'''
	d = rd.read_pairs_d_file_r(filename)
	d1 = ud.split_key(d,n=0)
	d2 = ud.split_key(d,n=1)
	#Fq(A+B) = Fq(A) + Fq(B) - Fq(A.B)
	#Fq(A)+Fq(B)
	for k in d1:
		if k not in d2:
			print k,'is not in ',d2
			d2[k] = 0
		print k+' '+str(d1[k])+' '+str(d2[k])
		d1[k] = d1[k] + d2[k]
	#-Fq(A.B)
	for k in d1:
		p = k+' '+k
		if p in d:
			d1[k]=d1[k]-d[p]

	diff_k = ud.get_diff_keys(d2,d1)
	print diff_k,'is not in d1'
	for k in diff_k:
		d1[k] = d2[k]
	ud.writetofile(d1,outfile)

def print_newgrammar_word_d(wordfreqfile):
	'''prints a new grammar having the frequency of words inside in wordfreqfile '''
	ref_w_d = rd.read_words_d_file(wordfreqfile)
	gr.readgrammar()
	d = gr.create_grammar_dict(ref_w_d)
	gr.niceprint_(d)

def disregard_order_pairdict(filename,outfile):
	'''gets a dictionary of pairs of AB whose frequencies are the sum of freq(AB)+freq(BA)''' 
	d = rd.read_pairs_d_file_r(filename)
	d_ = {}
	for k in d:
		words = k.split()	
		w1 = words[0]
		w2 = words[1]
		rev_k = w2+' '+w1
		if k != rev_k:
			f = d[k] + d.get(rev_k,0)
		else:
			f = d[k]
		d_[k] = f
		d_[rev_k] = f
	f = open(outfile,'w')
	smoothed_tuple = ud.sort_dictionary(d_,sort='key')
	
	for k,v in smoothed_tuple:
                f.write(k + ': ' + str(v) + '\n')
        f.close()
def print_spclxcat(file_list, outfile):
	''' print only frequency of words, in file_list, which belong to a specific categories'''
	d_list= []
	d_mother = {}
	lxcats =[pb.VERB_SYMS] # [pb.VERB_SYMS]
	for f in file_list:
		d_list.append(rd.read_words_d_file(f))
	
        for lxc in lxcats:
                words = get_words_lxcat(lxc)
                for w in words:
			d_mother[w]=[]	
			for d in d_list:
				d_mother[w].append(d.get(w,0)) 
	f = open(outfile,'w')
	for k,v in d_mother.iteritems():
		#f.write(k+': ')
		for fq in v:
			#f.write(str(int(fq))+', ')		
			f.write(str(int(fq))+' '+k)		
		f.write('\n')

def get_pair_nearest_words_spclxcat(filename,outfile,baseform=False):
	'''gets the pair of nearest words which belong to specific lexical categories like (function words, content words) or (content words, content words)'''
        f = open(filename,'r')
	f_ = open('inf.out','a')
        #pairs_out = []
	d = {}
        lxcats1 = [pb.ADJ_SYMS] #pb.COUNT_WORDS # pb.FUNC_WORDS #pb.NOUN_SYMS_NOPROP
        lxcats2 = [pb.VERB_SYMS, pb.NOUN_SYMS_NOPROP] # pb.COUNT_WORDS
        gr.readgrammar()
	wbase_d = get_wbaseform_d()
	words_lxc1 = []
	words_lxc2 = []
	#pdb.set_trace()
	for lxc in lxcats1:
                words_lxc1.extend(get_words_lxcat(lxc))
	for lxc in lxcats2:
                words_lxc2.extend(get_words_lxcat(lxc))
	words_lxc1 = [w.lower() for w in words_lxc1]
	words_lxc2 = [w.lower() for w in words_lxc2]
	#print 'words in lxc1:',words_lxc1
	#print 'words in lxc2:',words_lxc2
        for l in f.readlines():
        	words_sent = l.split()
		words_sent = [w.lower() for w in words_sent]
		for idx,w in enumerate(words_sent):
			if w in words_lxc1:
				for i in range(idx+1,len(words_sent)): 
					if words_sent[i] in words_lxc2:
						p = w+' '+words_sent[i]
						if baseform:
							p = wbase_d.get(w,w)+' '+wbase_d.get(words_sent[i],words_sent[i])
						d[p] = 1 + d.get(p,0)
						#pairs_out.append(p)
						break
	

	#print 'dictionary:',d
	ud.save_d(d, outfile)
	mtl_inf = KLdiv.mutual_inf(d)
	f_.write("*I(X,Y) "+str(mtl_inf)+"\n")
	f_.write(outfile+"\n")
	f_.close()	
def merge_2jntprobd_files(filename1_d, filename2_d, outfile):
	d1 = ud.load_d(filename1_d)
	d2 = ud.load_d(filename2_d)
	d_mrg = ud.merge_2dicts_disreg_ordkeypairs(d1, d2)
	d_mrg = ud.normalize_d_values(d_mrg)
	f = open(outfile, 'w')
	for pair,prob in d_mrg.iteritems():
		w1, w2 = pair.split() 
		f.write(w1 + '\t' + w2 + '\t' + str(prob) + '\n')

	f.close()

def get_wsform_d():
    root_d=get_wbaseform_d()
    sform_l = zip(root_d.values(), root_d.keys())
    sform_d = {}    
    for w in sform_l:
        sform_d[w[0]] = w[1]
    return sform_d

def get_wbaseform_d(base_file='wordSelectedShak-root.txt'):
	''' Returns a dictionary whose key is s form of words (plural nouns or singular verbs and whose value is the base of the words e.g.:
d['swords'] = 'sword'
d['comes'] = 'come'
	'''
	in_f = open(base_file, 'r')
	bases = in_f.readlines()  
	in_f.close()
	wbase_d = {}
	for base in bases:
		base = base.rstrip()
		base_s = base + 's'
		wbase_d[base_s] = base

	sb_l = zip(["calves", "churches", "wishes","marries","banishes","goes"], ["calf", "church","wish","marry","banish","go"])
	for sb in sb_l:
		wbase_d[sb[0]] = sb[1]
	return wbase_d

def merge_thePwVinV(fn1, fn2):
	'''
	putting pairs of words (the, theP)(with, withV)(in, inV) which are distinguished in the corpus together.
	'''
	d1 = rd.read_words_d_file(fn1)
	d = copy.deepcopy(d1)
	spcwds = {'thep':'the', 'withv':'with', 'inv':'in'}
	for k, v in d1.iteritems():
		w1, w2 = k.split()
		if (w1 in spcwds or w2 in spcwds):
			del d[k]
			nw1 = spcwds.get(w1, w1)
			nw2 = spcwds.get(w2, w2)
			npr = nw1 + ' ' + nw2
			d[npr] = d.get(npr, 0) + v
	#pdb.set_trace()
	ud.writetofile(d, fn2)
def normlz_jntfile_wrt1stwd(fn1, fn2):
	d = rd.read_words_d_file(fn1)
	d = ud.normalize_d_wrt1stwd(d)
	ud.writetofile(d, fn2)

def calc_paiwisedist_corpus(corpusfn, fout_pfx, sentlen):
    ''' calculates pairwise distribution of words between every two possible positions in a corpus'''
    for pos1 in range(sentlen-1):
        for pos2 in range(pos1+1, sentlen):
            fout = fout_pfx + str(pos1) + str(pos2) + '.txt'
            CountFreq.CountFrequency(corpusfn,fout).get_pProb_spcfpos(pos1, pos2)

def make_transitionmatrix(pfile_pfx, sentlen):
    '''makes transition matrix out of pairwise distributions given as input files'''
    labs_ = ['Det', 'N', 'V', 'Adj', 'Conj', 'Neg', 'Prep']
    labs = ['<'+l.lower()+'>' for l in labs_]
    mtxlab = {}
    for i in range(len(labs)):
        mtxlab[labs[i]] = i
    mtx_twopos_l = []
    #pdb.set_trace()
    for pos1 in range(sentlen-1):
        for pos2 in range(pos1+1, sentlen):
            mtx_twopos = np.zeros([len(labs), len(labs)])  
            pfile = pfile_pfx + str(pos1) + str(pos2) + '.txt'
            d = rd.read_words_d_file(pfile)
            for pair,pb in d.iteritems():
                w1, w2 = pair.split()
                idx1 = mtxlab[w1]
                idx2 = mtxlab[w2]
                mtx_twopos[idx1][idx2] = pb
            
            mtx_twopos_l.append(mtx_twopos)
            #np.set_printoptions(precision=3)
    return mtx_twopos_l                
                
def plot_transitionmatrix(pfile_pfx, sentlen):
    labs = ['Det', 'N', 'V', 'Adj', 'Conj', 'Neg', 'Prep']
    mtx_l = make_transitionmatrix(pfile_pfx, sentlen)
    idx = 0
    figidx = 1
    plt.figure()
    for pos1 in range(sentlen-1):
        for pos2 in range(pos1+1, sentlen):
            m = mtx_l[idx]
            plt.subplot(2, 2, figidx)
            plt.imshow(m)
            plt.xticks(np.arange(len(labs)), labs, fontsize=10)
            plt.yticks(np.arange(len(labs)), labs, fontsize=10)
            s = r'$\rho_{'+ str(pos1) + str(pos2)+ '}$'
            plt.title(s, fontsize=18)
            idx = idx + 1
            figidx = figidx + 1
            if (figidx == 5):
                fignm = pfile_pfx + str(idx) + '.png'
                plt.savefig(fignm)            
                plt.figure()
                figidx = 1
            
    fignm = pfile_pfx + str(idx) + '.png'
    plt.savefig(fignm)
            
