''' Sentence Generator '''

#import scipy as sp
import numpy as np
#import os
#from alirezamodule import read_cnst_file
import cnstpybliss as c
#from mpi4py import MPI
import re
from operator import add,mul,sub #15
import random
import csv
from time import time,strftime,localtime
#import warnings
import pdb
#import string
import copy
#import csv
#import cProfile
import blissplot
import blissutil
import pickle
__all__ = ['Delimit', 'GrammarReader', 'SentenceGenerator','UnderConstructSent'\
            ,'SelectionAlgorithm','ExponentialSelection','SbjVbSelection',\
            'VbSbjSelection','NoSemanticsSelection','convertfiletodic',\
            'factoryselectionalgorithm','specialdivbyzero',\
            'NO_ROLE','SBJ_ROLE', 'VB_ROLE','CNJ_ROLE','NOUN_SYMS','VERB_SYMS','CNJ_SYMS','NON_REPEAT_SYMS']


dictplot = blissutil.DictPlotter()
filereader = blissutil.FileReader()
utildict = blissutil.UtilDict()

NO_ROLE = 'none' 
SBJ_ROLE = 'SBJ' 
VB_ROLE = 'VB'
CNJ_ROLE = 'CNJ'
PREDICATIVE_NOUN_ROLE = 'NOUN_prd'
PREDICATIVE_ADJ_ROLE = 'ADJ_prd'
PREDICATIVE_PREP_ROLE = 'PREP_prd'
SG_VERB_SYMS =['<Vt11>','<Vi11>','<Vtd11>','<Vtdtv11>']
PL_VERB_SYMS =['<Vt12>','<Vi12>','<Vtd12>','<Vtdtv12>']
SG_NOUN_SYMS = ['<N11>']
PL_NOUN_SYMS = ['<N12>']
ADJ_SYMS = ['<AdjP>']
PREPN_SYMS = ['<Prep>']
PREPV_SYMS = ['<PrepV>']
PREPT_SYMS = ['<PrepT>']
SG_PROPNOUN_SYMS = ['<PropN11>']
PL_PROPNOUN_SYMS = ['<PropN12>']
NOUN_SYMS = SG_NOUN_SYMS + SG_PROPNOUN_SYMS + PL_NOUN_SYMS + PL_PROPNOUN_SYMS # ['<NSBJ11>', '<NSBJ12>', '<N11>', '<N12>', '<PropN11>', '<PropN12>']
VERB_SYMS = SG_VERB_SYMS + PL_VERB_SYMS  #['<Vi11>', '<Vi12>','<Vt11>', '<Vt12>', '<Vtd11>', '<Vtd12>','<Vtdtv11>', '<Vtdtv12>']
PREP_SYMS = PREPN_SYMS + PREPV_SYMS + PREPT_SYMS
CNJ_SYMS = ['<Conjd>']
NON_REPEAT_SYMS = ['<NSBJ11>', '<NSBJ12>', '<N11>', '<N12>', '<PropN11>', '<AdjP>']
SG_PRD_NOUN_SYM = '<N_PRD11>'
PL_PRD_NOUN_SYM = '<N_PRD12>'
PRD_ADJ_SYM = '<AJ_prd>'
PRD_PREPN_SYM = '<Pn_prd>'
PRD_PREPV_SYM = '<Pv_prd>'
SG_PL_SYM_DICT = {'<N11>':'<N12>','<Vt11>':'<Vt12>','<Vi11>':'<Vi12>','<Vtd11>':'<Vtd12>','<Vtdtv11>':'<Vtdtv12>','<PropN11>':'<PropN12>','<NPRD11>':'<NPRD12>'}
NOUN_SYMS_NOPROP = SG_NOUN_SYMS + PL_NOUN_SYMS  # ['<NSBJ11>', '<NSBJ12>', '<N11>', '<N12>']
ART_SYMS = ['<Art11>','<Art12>']
DEM_SYMS = ['<Dem11>','<Dem12>']
AUX_SYMS = ['<Neg11>','<Neg12>']

COUNT_WORDS = NOUN_SYMS + VERB_SYMS + ADJ_SYMS
FUNC_WORDS = PREP_SYMS + CNJ_SYMS + ART_SYMS + DEM_SYMS + AUX_SYMS


class Delimit(object):
    def __init__(self,dlms=None):
        if not dlms:
            self.imp = '=>'
            self.ors = '|'
            self.prob = 'probability'
            self.coma = ','
        else:
            if dlms.keys() in ['imp','ors','prob','coma']:
                self.__dict__ = dlms
            else: raise "check the keys of delimiter!"
            #raise NotImplementedError

class GrammarReader(object):
    '''
        Example:
        dlms=['imp':'->','ors':',','prob':'prob','coma':';']
    '''
    def __init__(self, filename, start='<S1>',dlms=None): 
        #TODO: put delimiters in the header of the grammar file
        self.fl = open(filename, 'r') 
        self.start = start
        self.d = {}
        self.dlms=Delimit(dlms)
        #Now we assign items of delimiter dictionary as attributes of GrammarReader class
        [self.__dict__.__setitem__(k,v) for (k,v) in self.dlms.__dict__.items()]
        #print self.__dict__.keys()
    
    def _initdict(self):
        d = {}
        return d
    @classmethod
    def niceprint_(cls,grd):
        s=''
        for symbol in sorted(grd):
                #print 'symbol:',symbol
            s = s + symbol + ' => ' 
            for word_list in grd[symbol]['rule']:
                for word in word_list:
                    #print 'word:',word
                    s = s + word + ' '
                s = s + ' | '
            for prob in grd[symbol]['prob']:
                #print 'prob:',prob
                s = s + str(prob) + ' , '
            s = s+"\n"
        print s
        
    def readgrammar(self):
        map(self.div_line, self.fl.readlines())
        
    def create_grammar_dict(self, ref_w_d): 
        d_=copy.deepcopy(self.d)
        for symbol in self.d:
            if not self.d[symbol]['rule'][0][0].startswith('<'):
                fq = []
                for word in self.d[symbol]['rule']:
                    w = word[0].lower()
                    if w not in ref_w_d:
                        #print w,' zero freq'
                        ref_w_d[w]=0
                    fq.append(float(ref_w_d[w]))    
                    #print word,
                s = sum(fq)
                if s:
                    fq = [i/s for i in fq]
                #print 'before:',self.d[symbol]['prob'],d_[symbol]['prob']
                d_[symbol]['prob'] = fq
                #print 'after fq:',self.d[symbol]['prob'],d_[symbol]['prob']
        return d_               

    def div_line(self, aline):
        first, rulepart, probpart = self.takeoutprob(aline)
        first=re.sub('\s+$|^\s+','',first)
        if first in  self.d.keys():
            print "This nonterminal already exists!"
            if type(self.d[first])!=dict:
                self.d[first]={}
        else:
            self.d[first]={}
        self.d[first]['prob'] = self.breakprobpart(probpart)
        self.d[first]['rule'] = self.breakrulepart(rulepart)
        #print "self.d[first]['prob']",self.d[first]['prob']
        #print "self.d[first]['rule']",self.d[first]['rule']
    def breakprobpart(self, segment):
        '''
          example:
        segment='0.5 , 0.5'
        prob = breakrulepart(segment)
        [0.5,0.5]
        '''
        prob = segment.split(self.dlms.coma)
        prob = [float(p) for p in prob]
        #print 'prob', prob, type(prob)
        return prob
    def breakrulepart(self, segment):
        '''
          example:
        segment='<DP11> <VP11> | <DP12> <VP12>'
        rulesep = breakrulepart(segment)
        [['<DP11>', '<VP11>'], ['<DP12>', '<VP12>']]
        '''
        rules = segment.split(self.dlms.ors)
        rulesep=[]
        for rule in rules:
            rule=re.sub('\s+$','',rule)
            rule=re.sub('^\s+','',rule)
            r = re.split("\s+", rule)
            rulesep.append(r)
            
        return rulesep
        
    def takeoutprob(self, segment):
        '''
          example:
        segment='<S1> => <DP11> <VP11> | <DP12> <VP12>   probability 0.5 , 0.5'
        first, rulepart, probpart = takeoutprob(segment)
        first = '<S1>'
        rulepart = '<DP11> <VP11> | <DP12> <VP12>'
        probpart = '0.5 , 0.5'
        '''
        first, rsd = segment.split(self.dlms.imp)
        rulepart, probpart = rsd.split(self.dlms.prob)
        return first, rulepart, probpart

    def insert_symbol(self, new_symbol, rulelist=[], problist=[]):
        self.d[new_symbol] = {}         
        self.d[new_symbol]['rule'] = rulelist
        self.d[new_symbol]['prob'] = problist

    def copy_symbol(self, from_symbol, to_symbol, coppied_fields=['rule','prob']):
        for f in coppied_fields:
            self.d[to_symbol][f] = self.d[from_symbol][f]

    def get_terminals(self, rules):
        terms = []
        for sym in rules:
            terms.extend([s[0] for s in self.d[sym]['rule']])
        return terms

class UnderConstructSent(object):
    def __init__(self, cnst):
        self.words = []
        self.roles = []
        self.tempflags = [] #temporary flags for words
                            #used in VbSbjSelection algorithm
        self.tempwords = [] #used in VbSbjSelection algorithm
        self.tempprobs = [] #used in VbSbjSelection algorithm
        self.c = cnst
        self.current_symbol=''
        self.f = open(self.c.SENTENCES_FILE, 'w')
        self.printstr = ''
        self.fderivation = open(self.c.DERIVATION_FILE, 'w')
        self.nonterminal_syms = []
        self.sents=''
        self.probs_nonterms=[] #15
        self.probs_words=[]
        self.fprob = open(self.c.PROBS_FILE, 'w') #15
        self.fsbjvb = open(self.c.SBJVB_FILE, 'w')
        self.fsbjvbprdn = open(self.c.SBJVB_PRDN_FILE, 'w')
        self.fsbjvbprd = open(self.c.SBJVB_PRD_FILE, 'w')

    def appendword(self,word,role='',temp=False,\
                            tempw=[],tempp=[]):
        self.words.append(word)
        self.roles.append(role)
        self.checkcnj(role)
        self.tempflags.append(temp)
        self.tempwords.append(tempw)
        self.tempprobs.append(tempp)
    def appendprob_nonterms(self,p): #15
        #print 'appendprob_nonterm:',p
        self.probs_nonterms.append(p)
    def popprob_nonterms(self,idx): #15
        #print 'popprob_nonterm:',idx
        return self.probs_nonterms.pop(idx)
    def appendprob_words(self,p): #15
        #print 'appendprob_words:',p
        self.probs_words.append(p)
    def popprob_words(self,idx): #15
        #print 'popprob_words:',idx
        if len(self.probs_words):
            return self.probs_words.pop(idx)
    
    def replace_word(self,idx,word,role,tmpflg,tempw=[],tempp=[]):
        self.words[idx] = word
        self.roles[idx] = role
        self.tempflags[idx] = tmpflg
        self.tempwords[idx]=tempw
        self.tempprobs[idx]=tempp
    
    def print_(self):
        # prints each sentence, as soon as generated, into a file
        a = ' '.join(self.words)
        #print 'sentence:',a
        self.f.write(a+'\n')
        
        #print a
        #a= ' => '.join(self.nonterminal_syms)
#        pdb.set_trace()
        a= ' '.join(self.nonterminal_syms)
        self.fderivation.write(a+'\n')
        #print a
    def print_sbjvb(self):
        s = self.getsbj()
        v = self.getvb()
        if s:
            a = s+' '+v
            self.fsbjvb.write(a+'\n')
        #print 'subject,verb:',a
    def print_sbjvbprdn(self):
        s = self.getsbj()
        v = self.getvb()
        prdn_list = self.getprdnouns()
        for n in prdn_list:
            a = s+' '+v+' '+n
            self.fsbjvbprdn.write(a+'\n')
            #print 'subject,verb,prdn:',a
    def print_sbjvbprd(self):
        s = self.getsbj()
        v = self.getvb()
        prd_list = self.getprd()
        for w in prd_list:
            a = s+' '+v+' '+w
            self.fsbjvbprd.write(a+'\n')
            #print 'subject,verb,prdn:',a

    def print_probs(self): #15
        p_words = self.probs_words
        p_nonterms = self.probs_nonterms
        #print 'p_words',p_words
        #print 'p_nonterms',p_nonterms
        p_all = p_nonterms + p_words
        #print 'p_all',p_all
        #mp = reduce(mul,p_all)
        mp = reduce(mul,p_words)
        #print 'mul:',mp
        s = ['%.2f'%(p) for p in p_words]
        s=str(mp)+'\tww\t'+str(s)+'\n'
        #print s
        self.fprob.write(s)
        
    def save_sent(self):
        #saves the current sentence into the collection of its sentences
        self.sents = self.sents+' '.join(self.words)+'\n' 
        
    def get_all_sents(self):
        return self.sents 

    def empty(self):
        self.words=[]
        self.roles = []
        self.tempflags = []
        self.tempwords = []
        self.tempprobs = []
        self.current_symbol=''
        self.probs_nonterms = []
        self.probs_words = []
        self.nonterminal_syms = []
        #print 'empty!'
    def end(self):
        self.f.close()
        self.fprob.close()
        self.fsbjvb.close()
        self.fsbjvbprdn.close()
        self.fderivation.close()
        
    def getsbj(self):
        sbj = ''
        if (SBJ_ROLE in self.roles):
            idx = self.roles.index(SBJ_ROLE)
            if (idx < len(self.words)):
                sbj = self.words[idx]
        return sbj
        
    def getvb(self):
        vb = ''
        if (VB_ROLE in self.roles):
            idx = self.roles.index(VB_ROLE)
            if (idx < len(self.words)):
                vb = self.words[idx]
        return vb
    def getprdnouns(self):
        prdn_list = []
        for idx in range(len(self.words)):
            r = self.roles[idx]
            if r == PREDICATIVE_NOUN_ROLE:
                prdn_list.append(self.words[idx])
        return prdn_list
    def getprd(self):
        prd_list = []
        for idx in range(len(self.words)):
            r = self.roles[idx]
            if (r == PREDICATIVE_NOUN_ROLE) or (r==PREDICATIVE_ADJ_ROLE) or (r==PREDICATIVE_PREP_ROLE):
                prd_list.append(self.words[idx])
        return prd_list


    def settempflag(self,index, flag):
        if len(self.tempflags):
            self.tempflags[index] = flag
        else:
            self.tempflags.append(flag)
    def settempwordprobs(self,index, wordlist, problist): 
        #TODO: use __getattribute__
        if len(self.tempwords):
            self.tempwords[index] = wordlist
            self.tempprobs[index] = problist
        else:
            self.tempwords.append(wordlist)
            self.tempprobs.append(problist)

    def gettempwordprobs(self,index):
        tempwords = self.tempwords[index]
        tempprobs = self.tempprobs[index]
        return tempwords, tempprobs        
        
    def checkcnj(self, role):
        if(role == CNJ_ROLE):
            #print 'words:',self.words
            #print 'roles:',self.roles
            self.print_sbjvb()
            idx = self.roles.index(SBJ_ROLE)
            self.roles[idx] = NO_ROLE
            idx = self.roles.index(VB_ROLE)
            self.roles[idx] = NO_ROLE
            #print 'chkcnj, roles:',self.roles
    
    def findrole(self, symbol):
        if (c.SELECTION_ALGORITHM=='EXPONENTIAL'):
            pos = len(self.words)
            return 'pos'+str(pos)
            
        isconjugate = symbol in CNJ_SYMS
        if (isconjugate):
            return CNJ_ROLE

        isnoun = symbol in NOUN_SYMS
        hassbj = SBJ_ROLE in self.roles
        if( isnoun and (not hassbj) ):
            return SBJ_ROLE
        
        isverb = symbol in VERB_SYMS
        hasvb = VB_ROLE in self.roles
        if( isverb and (not hasvb) ):
            return VB_ROLE
        
        if (isnoun and hassbj and hasvb):
            return PREDICATIVE_NOUN_ROLE

        isadj = symbol in ADJ_SYMS
        #if (isadj and hassbj and hasvb):
        if (isadj): # and hassbj and hasvb):
            return PREDICATIVE_ADJ_ROLE

        isprep = symbol in PREP_SYMS
        if (isprep and hassbj and hasvb):
            return PREDICATIVE_PREP_ROLE

        return NO_ROLE
   
class SentenceGenerator(object):
    def __init__(self, objgr, cnst):
        self.c = cnst
        self.d = objgr.d
        self.startsym = objgr.start
        self.stack = [self.startsym]
        self.sentobj = UnderConstructSent(cnst)
        self.algorithmobj = factoryselectionalgorithm(self.sentobj,cnst)
        
    def generate(self,num_of_sentence=None):
        ''' main function of SentenceGenerator class which generate sentences 
        as number as NUM_OF_SENTENCES''' 
        #print strftime("%H:%M:%S",localtime())
        if not num_of_sentence:
            bignum=int(self.c.NUM_OF_SENTENCES)
        else:
            bignum=num_of_sentence

        st=self.startsym
        pNo=self.c.PRINT_SENT_No
        for i in xrange(bignum):
            self.stack = [st]
            while (not self.endofSentence()):
            #pdb.set_trace()
                self.nextword()
                
            self.sentobj.print_()
            self.sentobj.print_sbjvb()
            #pdb.set_trace()
            self.sentobj.print_sbjvbprdn()
            self.sentobj.print_sbjvbprd()
            self.sentobj.print_probs()
            #self.sentobj.save_sent()
            self.sentobj.empty()
            self.algorithmobj.empty()
            #if not(i%pNo):
                #print i
                #print strftime("%H:%M:%S",localtime())
                
        self.sentobj.end()
        
    def endofSentence(self):
        ''' if self.stack = []: return 1 
        '''
        if (self.stack): 
            #print self.stack,'endofsentence: stack is not empty!'
            return 0
        else:    
            #print self.stack,'empty'
            return 1
    
    def nextword(self):
        ''' produce next word of a sentence '''
        wordlist, problist, role = self.rewritetogetwordlist()
        #print 'wordlist:',wordlist,"\nproblist:",problist,"\nrole:",role
        #print 'sent:',self.sentobj.words
        newproblist = self.algorithmobj.apply(wordlist,problist,pos_wrt_ifwds=[])
#        print 'word:',wordlist,'newproblist:',newproblist
#        pdb.set_trace()        
        #print 'sent:',self.sentobj.words
        word,p = self.chooseword(wordlist,newproblist)
        self.sentobj.appendword(word,role)
        self.algorithmobj.updateinfluentialwords(wordlist, problist)
#        print 'sent:',self.sentobj.words
        #self.sentobj.print_probs()
        
    def rewritetogetwordlist(self):
        ''' 
        rewrite the grammar rules to obtain a list of words as candidates for 
        next word in the sentence, and set this list as self.terminallist
        Example: if the grammar is
        <DP11> => <NP1>
        <NP1> => sword | crown | dog ...
        and the top of stack is 
        <DP11>
        rewritetogetwordlist() sets
        self.terminallist = ['sword', 'crown', 'dog',...]
        '''
        if (not self.stack[0]):
            #print 'nextword:stack empty'
            return
    
        symbol=self.stack.pop(0)
        #self.sentobj.nonterminal_syms.append(symbol)
        #If the symbol goes to nonterminals    
        if self.d[symbol]['rule'][0][0].startswith('<'):
            #print 'symol goes to Nonterminals!'
            nonterminallist = self.d[symbol]['rule']
            problist = self.d[symbol]['prob']
            nonterminal,p = toss_comulative(nonterminallist,problist) #15
            self.sentobj.appendprob_nonterms(p) #15
            self.stack[:0]=nonterminal
            return self.rewritetogetwordlist()
        else: #return terminal symbols
            #print 'symol goes to Terminals!'
            self.sentobj.nonterminal_syms.append(symbol)
            #pdb.set_trace()
            self.sentobj.current_symbol = symbol
            role = self.sentobj.findrole(symbol)
            #print 'symbol:',symbol,'role',role
            symbol_ = self.get_symbol_role(symbol,role)
            #print 'symbol corresponding to this role:',symbol_
            if (symbol_ in self.d):
                symbol = symbol_
            wlist = [w[0] for w in self.d[symbol]['rule']]
            return wlist, self.d[symbol]['prob'], role
    
  
    def chooseword(self,wordlist,problist):
        word,p = toss_comulative(wordlist,problist) #15
        self.sentobj.appendprob_words(p) #15
        '''
        if (self.sentobj.current_symbol in NON_REPEAT_SYMS):
            wordrt = word
            if word in droot:
                wordrt = droot[word]
            words = self.sentobj.words
            wordlistrt = [getroot(w) for w in words]
            m=1
            while np.any([words.__eq__(wordrt) for words in wordlistrt]):
                #print wordlistrt
                #print '1word chooseword:',word
                word,p = toss_comulative(wordlist,problist) #15
                self.sentobj.popprob_words(-1) #removing the prob of last word
                self.sentobj.appendprob_words(p) #15 #replace with the prob of new word
                wordrt = word
                if word in droot:
                    wordrt = droot[word]
                if m>40: 
                    return word
                m += 1
        '''
        return word, p

    def get_symbol_role(self,symbol,role):
        if re.search('prd',role,re.IGNORECASE):
            symbol_ = attach_str_to_symbol('_prd',symbol)
        elif re.search('pos',role,re.IGNORECASE):
            pos = find_digit_in_str(role)
            symbol_ = attach_str_to_symbol('_pos'+str(pos),symbol)
        else:
            symbol_ = symbol    
        return symbol_
        
    def getlxcat_of_aword(self, word):
        symbol_ = ''
        for symbol in self.d:
            symbol_wds = [w[0].lower() for w in self.d[symbol]['rule']]
#            pdb.set_trace()            
            if word in symbol_wds:
                symbol_ = symbol
        if not symbol_:
            print 'word:', word
            raise Exception, 'No symbol is found:'
        return symbol_
                
def attach_str_to_symbol(s,symbol):
    symbol_ = symbol.split('>')[0] + s +'>'
    return symbol_
class FairSentenceGenerator(SentenceGenerator):
    '''
    Considers the stochastic grammar as a grammar with equiprobable terminals 
    As an example, if original stochastic grammar is:
    <DP11> => <N11> | <Adj> <N11> probability .7, .3         
    <N11> => dog | sword | crown | horse  probability .3, .4, .2, .1
    This class consider it as this grammar:
    <DP11> => <N11> | <Adj> <N11> probability .5, .5         
    <N11> => dog | sword | crown | horse    probability .25, .25, .25, .25
    '''
    def chooseword(self,wordlist,problist):
            #call toss_fair function to return a word from wordlist equiprobably, disregarding problist
            
        word,p = toss_fair(wordlist,problist)
        self.sentobj.appendprob_words(p) #15
        '''
        if (self.sentobj.current_symbol in NON_REPEAT_SYMS):
                wordrt = word
                if word in droot:
                        wordrt = droot[word]
                words = self.sentobj.words
                wordlistrt = [getroot(w) for w in words]
                m=1
                while np.any([words.__eq__(wordrt) for words in wordlistrt]):
                        word,p = toss_fair(wordlist,problist)
                        self.sentobj.popprob_words(-1)
                        self.sentobj.appendprob_words(p) #15
                        wordrt = word
                        if word in droot:
                                wordrt = droot[word]
                        if m>40:
                                return word
                        m += 1
        '''
        return word, p

class SentenceGenerator_NoGrammar(SentenceGenerator):
    '''
        Does not take any grammar as input and generate sentences without any grammar. Each word is chosen by its probability in c.UNIGRAMPROB_FILE. Obviously it can have semantics as it has an algorithmobj
    '''
    def __init__(self, objgr, cnst):
        self.c = cnst
        self.startsym = ''
        self.stack = [self.startsym]
        self.sentobj = UnderConstructSent(cnst)
        self.algorithmobj = factoryselectionalgorithm(self.sentobj,cnst)
        #print 'algorithm:',self.algorithmobj
        self.d = objgr.d
        ugfile_d = convertfiletodic(c.UNIGRAMPROB_FILE, ' ', 1, 0)
        self.wordlist = ugfile_d.keys() #all words of the corpus
        vals = [int(v) for v in ugfile_d.values()]
        s = float(sum(vals))
        self.problist = [v/s for v in vals] #the unigram probability of words
        #print 'wordlist',self.wordlist
        #print 'problist',self.problist
        self.sen_len = c.SENT_LEN_UG
    
    def endofSentence(self):
        ''' it stops when a dot '.' appears or the length of a sentence exceeds 10
        '''
        #if ('.' in self.sentobj.words): 
        #    return 1
        if(len(self.sentobj.words) == self.sen_len):
            return 1
        else:    
            return 0
    
    def chooseword(self,wordlist,problist):
                
        word, p = toss_comulative(wordlist,problist)
#        pdb.set_trace()
        symbol = self.getlxcat_of_aword(word)
        self.sentobj.nonterminal_syms.append(symbol)
        
        #word = toss_fair(wordlist,problist)
        return word, p
        
    def rewritetogetwordlist(self):
        ''' 
        since no grammar is read in this class, it only returns all words of the corpus and their unigram
        probabilities
        '''
        role = ''
        return self.wordlist, self.problist, role 

class SentenceGenerator_NoGrammarPosdep(SentenceGenerator_NoGrammar):

    def __init__(self, objgr, cnst):
        self.c = cnst
        self.startsym = ''
        self.stack = [self.startsym]
        self.sentobj = UnderConstructSent(cnst)
        self.algorithmobj = factoryselectionalgorithm(self.sentobj,cnst)
        #print 'algorithm:',self.algorithmobj
        self.d = objgr.d
        self.sen_len = c.SENT_LEN_UG
        self.wordlists, self.problists = self.get_wordproblists_pos(c.UNIGRAMPOSDEPPROB_FILE, pfx='.csv')
        
    def get_wordproblists_pos(self, filepfx, pfx='.csv'):
        fqfiles = [filepfx+str(i)+pfx for i in range(self.sen_len)]
        fqd_l = [convertfiletodic(f, ' ', 1, 0) for f in fqfiles]
        wordlists = [fd.keys() for fd in fqd_l] #all words of the corpus
        problists = []
        for d in fqd_l: 
            vals = d.values()
            vals = [float(v) for v in vals]
            s = float(sum(vals))
            vals = [v/s for v in vals] #the unigram probability of words
            problists.append(vals)
        return wordlists, problists
        
    def rewritetogetwordlist(self):
        ''' 
        since no grammar is read in this class, it only returns all words of the corpus and their unigram
        probabilities
        '''
        role = ''
        pos = len(self.sentobj.words)
        return self.wordlists[pos], self.problists[pos], role 
            
def toss_comulative(items,probabilities):
    if set(probabilities)==set([0]):
        #print 'fair toss'
        return toss_fair(items,probabilities)
    probcomulative= [0]
    for i in xrange(len(probabilities)):
        probcomulative.append(probabilities[i] + probcomulative[i])
    rand_choice = random.random()
    lenchoice = len([choice for choice in probcomulative if rand_choice > choice])
    if lenchoice > len(items): 
        print items, probabilities
        print 'Warning in toss_comulative(): sum of probabilities exceeds 1!' #TODO change to warning
        lenchoice = len(items)
    #print 'item,prob:',items[lenchoice-1],probabilities[lenchoice-1]
    return items[lenchoice - 1],probabilities[lenchoice-1] #15

def toss_fair(items,probabilities):
    # disregard all probabilities assigned to words and return one word cosidering all equiprobable
    p = 1./len(items) #equal prob
    return random.choice(items),p
    
def factoryselectionalgorithm(sentence,cnst):
    if(cnst.SELECTION_ALGORITHM == 'NOSEMANTICS'):
        algorithmobj = NoSemanticsSelection(sentence,cnst)
    elif(cnst.SELECTION_ALGORITHM == 'EXPONENTIAL'):
        algorithmobj = ExponentialSelection(sentence,cnst)
    elif(cnst.SELECTION_ALGORITHM == 'SBJVB'):
        algorithmobj = SbjVbSelection(sentence,cnst)
    elif(cnst.SELECTION_ALGORITHM == 'VBSBJ'):
        algorithmobj = VbSbjSelection(sentence,cnst)
    elif(cnst.SELECTION_ALGORITHM == 'UNIGRAM'):
        algorithmobj = NoSemanticsSelection(sentence,cnst)
    elif(cnst.SELECTION_ALGORITHM == 'BIGRAM'):
        algorithmobj = BigramSelection(sentence,cnst)
    else:
        print cnst.SELECTION_ALGORITHM,' is undefined!'
        return 0
    return algorithmobj
def convertfiletodic(filename,delimiter,key_idx=0, val_idx=1):
    d = {}
    reader = csv.reader(open(filename), delimiter=delimiter)
    for row in reader:
        row[val_idx]=re.sub('\s+$|^\s+','',row[val_idx])
        d[row[key_idx].lower()] = row[val_idx].lower()
    return d

def getroot(word):
    root = word
    if word in droot:
        root = droot[word]
    return root
def specialdivbyzero(normfactor,probarray):
    '''divide the elements of probarray by normfactor if normfactor is nonzero
    otherwise, return an array with zero elements
    '''
    if normfactor: 
        probarray = probarray / normfactor
    else:
        probarray = probarray * 0.0
    return probarray    

class SelectionAlgorithm(object): #TODO: rename to Abstract...
    def __init__(self,sentence,cnst):
        self.c = cnst
        self.sentobj = sentence
        self.influentialwords = []
        self.influentialmultipliers = []
    
    def empty(self):
        self.influentialwords = []
        self.influentialmultipliers = []
        
    def apply(self,wordlist,problist,r='none',pos_wrt_ifwds=[]):
        '''
        apply selection algorithm in order to obtain slecetion probabilities
        considering what is the previous word.
        This function is the main function of the selectionAlgorithm class.
        Example: our currenct sentence is "w1 w2 w3", now we want to add new word 
        w or w', according to their probs. If only w1 and w2 are influential with
        the effective multipliers of m1,m2. and also gamma indicates the degree of
        dependency on history.
        
        P(w/w1w2w3)=gamma * P(w) + (1-gamma) * 1/(m1+m2)[m1*P(w/w1)+m2*P(w/w2)]
        P(w'/w1w2w3)=gamma * P(w') + (1-gamma) * 1/(m1+m2)[m1*P(w'/w1)+m2*P(w'/w2)]
        [P(w/w1w2w3), P(w'/w1w2w3)]=gamma*[P(w),P(w')]+(1-gamma)* 1/(m1+m2) *
                                    [m1*[P(w/w1),P(w'/w1)] + m2*[P(w/w2),P(w'/w2)]]
        given: P(w/w1)+P(w'/w1)=1, P(w/w2)+P(w'/w2)=1
        '''
        if(not self.influentialwords):
            return problist
        if(len(self.influentialwords) != len(self.influentialmultipliers)):
            print 'not equal words and mltps:',self.influentialwords,self.influentialmultipliers,wordlist
            raise Exception, 'The number of influential words and multipliers are not equal'
        probarr = np.array(problist)
        #print 'wordlist:',wordlist
        #print 'baseprob:',problist
        #print 'infl wordlist:',self.influentialwords
        historyparr = np.zeros(len(probarr)) # each element for different words
        normalizedifmltp = 0
        # IF: apply(..., pos_wrt_ifwds=None)
        # if pos_wrt_ifwds is None:
        #       pos_wrt_ifwds = []
        for i in range(len(self.influentialwords)-len(pos_wrt_ifwds)):
            pos_wrt_ifwds.append('R')                   
        #pdb.set_trace()
        for ifw in range(len(self.influentialwords)): 
            #the following list: [P(w/w1), P(w'/w1)]
            #normalization: P(w/w1)+P(w'/w1)=1
            jpifw_wlistarr = self.normalizeifw_wlistprob(\
                                    self.influentialwords[ifw],wordlist,r,pos_wrt_ifwds[ifw])
            #print 'normlzd jps:',jpifw_wlistarr
            #print 'sum:',reduce(add,jpifw_wlistarr)
            if(jpifw_wlistarr.any()):
            #m1+m2 as denominator in the above formula
                normalizedifmltp += self.influentialmultipliers[ifw]
            # m1 * [P(w/w1),P(w'/w1)]
            jpifw_wlistarr = self.influentialmultipliers[ifw]*jpifw_wlistarr
            #print 'inflmltp:',self.influentialmultipliers[ifw]
            # m1 * [P(w/w1),P(w'/w1)] + m2 * [P(w/w2),P(w'/w2)]
            historyparr= historyparr + jpifw_wlistarr
            #print 'historyparr:',historyparr
        # 1/(m1+m2) * [m1 * [P(w/w1),P(w'/w1)] + m2 * [P(w/w2),P(w'/w2)]]
        historyparr = specialdivbyzero(normalizedifmltp, historyparr)
        #print 'historyparr:', historyparr
        if (historyparr.any() and probarr.any()):
            #print '(1-sem)*probarr + sem*hist:'
            #print (1 - self.c.SEMANTICS_MULTIPLIER),' * ',probarr,' + ',\
            self.c.SEMANTICS_MULTIPLIER, '* ', historyparr
            selectionprob = (1 - self.c.SEMANTICS_MULTIPLIER) *probarr + \
                        self.c.SEMANTICS_MULTIPLIER * historyparr
        elif (historyparr.any()):
            selectionprob = historyparr
        else:
            selectionprob = probarr
        #print 'selectionprob:',selectionprob
        return selectionprob.tolist()
    
    def updateinfluentialwords(self,*args,**kwds):
        pass
    def dis2RL(self,dis):
        '''Returns 'R' (right) or 'L' (left) if the distance of two words is Positive or Negative, For example
        in 'Sahar eats her lunch' the distance of 'lunch' from 'Sahar' is 3 which is positive and it's located on the right side of 'Sahar' '''
        if dis > 0:
            return 'R'
        else:
            return 'L'
    def normalizeifw_wlistprob(self, influentialword, wordlist,r='none',pos='R'):
        ''' normalize jntprob influential word with wordlist
        Example: influentialword = 'black',wordlist=['do','feel','go','help']
        black do: 0.05 
        black feel: 0.0125 
        black go: 0.0125 
        black help: 0.0125
        returns jp = array([0.571428571,0.142857143,0.142857143,0.142857143])
        '''
        jp = []
        for word in wordlist:
            jp.append(self.getjntprob(influentialword, word,r,pos=pos))
        sum = reduce(add, jp)
        jparr = specialdivbyzero(sum, np.array(jp))
        return jparr

    def getjntprob(self, word1, word2,r='none',pos='R'):
        ''' looks for joint probability (droot[word1]+' '+droot[word2]) inside\
        djntprob and returns this probability
        '''
        word1 = word1.lower()
        word2 = word2.lower()   
        if (pos == 'L'):
            temp = word1
            word1 = word2
            word2 = temp            
        
        if word1 in droot:
            word1 = droot[word1]
        if word2 in droot:
            word2 = droot[word2]
        pair = word1+' '+ word2
        if pair in djntprob:
            jp = float(djntprob[pair])
        else:
            jp = 0
        return jp

    def adjust_word_freq(self,objgr,ref_w_d):
        #pdb.set_trace()
        roles = self.get_roles()
        print 'roles:',roles
        for r in roles:
            role_fq_fnc = self.factory_adjust_role_freq(r)
            infld_nonterms_list = self.get_infld_nonterms(r)
            for infld_nonterms in infld_nonterms_list:
                history_terms, history_terms_probs_d, inflmltps = role_fq_fnc(objgr,ref_w_d,infld_nonterms,r)
                #print 'history_terms', history_terms
                #print 'history_terms_probs_d', history_terms_probs_d
                #print 'inflmltps', inflmltps
                
                for sym in infld_nonterms:
                    infld_terms_1rule = [s[0] for s in objgr.d[sym]['rule']]  #[kills, takes,...] or [dies,stands,...] ...
                    #print 'infld_terms_1rule', infld_terms_1rule
                    infld_postprobs = utildict.lookup_dict(infld_terms_1rule, ref_w_d) 
                    s = float(sum(infld_postprobs))
                    if s:
                        infld_postprobs = [p/s for p in infld_postprobs]
                    #print 'infld_postprobs', infld_postprobs
                    infld_calc_priorprobs = self.calculate_priorprobs(infld_terms_1rule,infld_postprobs,history_terms,history_terms_probs_d,inflmltps,r)
                    #print 'infld_calc_priorprobs', infld_calc_priorprobs
                    #pdb.set_trace()
                    self.modify_grammar(objgr,sym,infld_calc_priorprobs,r)                  

    def factory_adjust_role_freq(self,role):
        if (role == 'subject'):
            role_fq_fnc = self.adjust_subject_freq
        elif (role == 'verb'):
            role_fq_fnc = self.adjust_verb_freq
        elif (role == 'object'):
            role_fq_fnc = self.adjust_object_freq
        elif re.match('pos',role):
            role_fq_fnc = self.adjust_position_freq
        else:
            raise Exception, 'what is the role to adjust its frequency?!'
        return role_fq_fnc

    def get_infld_nonterms(self,role):
        if (role == 'subject'):
            infld_nonterms_list = []
        elif (role == 'verb'):
            infld_nonterms_list = []
        elif (role == 'object'):
            infld_nonterms_list = []
        else:
            raise Exception, 'what is the role to return its infld_nonterms_list?!'
        return infld_nonterms_list

    def adjust_subject_freq(self,objgr,ref_w_d,infld_nonterms,r=''):
        '''adjusts the frequency of nouns whose roles are subjects in a sentence '''
        history_terms = []
        history_terms_probs_d = {}
        influentialmultipliers = []
        return history_terms, history_terms_probs_d,influentialmultipliers

    def adjust_verb_freq(self,objgr,ref_w_d,infld_nonterms,r=''):
        '''adjusts the frequency of nouns whose roles are subjects in a sentence '''
        history_terms = []
        history_terms_probs_d = {}
        influentialmultipliers = []
        return history_terms, history_terms_probs_d,influentialmultipliers

    def adjust_object_freq(self,objgr,ref_w_d,infld_nonterms,r=''):
        '''adjusts the frequency of nouns whose roles are objects in a sentence '''
        history_terms = []
        history_terms_probs_d = {}
        influentialmultipliers = self.get_inflmltps_object_role()
        
        #print 'infld_nonterms', infld_nonterms
        history_nonterms1 = SG_NOUN_SYMS + SG_PROPNOUN_SYMS  
        history_nonterms2 = SG_VERB_SYMS
        #print history_nonterms1
        #print history_nonterms2
        history_terms1 = objgr.get_terminals(history_nonterms1)
        history_terms2 = objgr.get_terminals(history_nonterms2)
        for h1 in history_terms1:
            for h2 in history_terms2:
                pair = h1+' '+h2
                history_terms.append(pair)
        #print 'history_terms:',history_terms

        history_nonterms1 = PL_NOUN_SYMS + PL_PROPNOUN_SYMS  
        history_nonterms2 = PL_VERB_SYMS
        history_terms1 = objgr.get_terminals(history_nonterms1)
        history_terms2 = objgr.get_terminals(history_nonterms2)
        for h1 in history_terms1:
            for h2 in history_terms2:
                pair = h1+' '+h2
                history_terms.append(pair)
        #print 'tutti history_terms:',history_terms
        #pdb.set_trace()
        wordlist = [w[0] for w in objgr.d[infld_nonterms[0]]['rule']]
        ref_p_d = blissplot.read_prdnfile_count_convert_d(c.REF_PAIR_DIST_FILE,wordlist,pos=2)

        h_t_p = utildict.lookup_dict(history_terms, ref_p_d) 
        s = float(sum(h_t_p))
        if s:
            h_t_p = [p/s for p in h_t_p]
        for h in history_terms:
            history_terms_probs_d[h] = h_t_p.pop(0)
        #print 'history_terms', history_terms
        #print 'history_terms_probs_d', history_terms_probs_d
        return history_terms, history_terms_probs_d,influentialmultipliers
    
    def adjust_position_freq(self,objgr,ref_w_d,infld_nonterms,r=''):
        history_terms = []
        history_terms_probs_d = {}
        influentialmultipliers = []
        return history_terms, history_terms_probs_d,influentialmultipliers
    
    def get_inflmltps_object_role(self):
        return []
    def get_roles(self):
        return []
        
    def calculate_priorprobs(self,infld_terms_1rule,infld_postprobs,history_terms,history_terms_probs_d,influentialmultipliers,r):
        #pdb.set_trace()
        history_terms_jZ = []
        history_terms_jnZ = []
        infld_calc_priorprobs = []
        for i_t in infld_terms_1rule:
            for h_t in history_terms:
                h_t_list = h_t.split()
                for w in h_t_list:
                    if self.getjntprob(w, i_t,r):
                        history_terms_jnZ.append(h_t)
                        break   
        history_terms_jnZ = list(set(history_terms_jnZ))
        history_terms_jZ = [h for h in history_terms if h not in history_terms_jnZ]
        #print '*****************************************'
        #print 'history_terms_jnZ:',history_terms_jnZ
        #print '*****************************************'
        #print 'history_terms_jZ:', history_terms_jZ
        p_zeros = [0 for i in range(len(infld_terms_1rule))]
        semantics_prob = [0 for i in range(len(infld_terms_1rule))]
        #pdb.set_trace()
        for h in history_terms_jnZ: 
            h_p = history_terms_probs_d[h]  
            self.influentialmultipliers = influentialmultipliers
            self.influentialwords = h.split()
            sem_p = self.apply(infld_terms_1rule,p_zeros,r) 
            sem_p = [p*h_p for p in sem_p]
            semantics_prob = map(add, semantics_prob, sem_p)
        g = c.SEMANTICS_MULTIPLIER
        semantics_prob = [p*g for p in semantics_prob]
        sum_prob_h_t_jZ = 0
        sum_prob_h_t_jnZ = 0
        for h in history_terms_jZ:
            sum_prob_h_t_jZ +=  history_terms_probs_d[h]
        for h in history_terms_jnZ:
            sum_prob_h_t_jnZ += history_terms_probs_d[h]
        # aX + b = 0 where a is denominator, b is numerator, X is calculated-prior-probability
        #pdb.set_trace()
        denominator = sum_prob_h_t_jZ + (1-g)*sum_prob_h_t_jnZ
        #print 'denominator:',denominator
        #print 'semantics_prob:',semantics_prob
        #print 'infld_postprobs:',infld_postprobs
        numerator = map(sub,semantics_prob,infld_postprobs)     
        infld_calc_priorprobs = []
        for i in range(len(infld_terms_1rule)):
            p = np.poly1d([denominator,numerator[i]])
            if not p.r:
                r = 0
            else:
                r = p.r[0].real
            if (r<0):
                print 'Warning: minus prior prob converted to zero',infld_terms_1rule[i],',',r
                r = 0
            infld_calc_priorprobs.append(r)
        #print 'calculated_priorprobs: ',infld_calc_priorprobs
        s = sum(infld_calc_priorprobs)
        #print 'sum of these probs:', s
        if (s>0):
            infld_calc_priorprobs = [i/s for i in infld_calc_priorprobs]
        #print 'normalized_priorprobs: ',infld_calc_priorprobs
        return infld_calc_priorprobs

    def modify_grammar(self,objgr, symbol, prob, role):
        if role=='object':
            symbol_ = attach_str_to_symbol('_prd',symbol)
            objgr.insert_symbol(symbol_, problist=prob)
            objgr.copy_symbol(from_symbol=symbol,to_symbol=symbol_,coppied_fields=['rule'])

        elif role=='subject':
            objgr.d[symbol]['prob'] = prob
            #objgr.d[SG_PL_SYM_DICT[symbol]]['prob'] = prob

        elif role=='verb':
            objgr.d[symbol]['prob'] = prob
            #objgr.d[SG_PL_SYM_DICT[symbol]]['prob'] = prob
        elif re.match('pos',role):
            pos = find_digit_in_str(role)
            symbol_ = attach_str_to_symbol('_pos'+str(pos),symbol)
            objgr.insert_symbol(symbol_, problist=prob)
            objgr.copy_symbol(from_symbol=symbol,to_symbol=symbol_,coppied_fields=['rule'])
        else:
            raise Exception, 'what is the role, to change its prior prob in grammar?'

    
class NoSemanticsSelection(SelectionAlgorithm):
    '''runs when SELECTION_ALGORITHM = 'NOSEMANTICS'
    influential words: nothing 
    '''
    def apply(self,wordlist,problist,pos_wrt_ifwds=[]):
        #print 'nosemantics.apply()...'
        return problist
    def adjust_word_freq(self,objgr,ref_w_d):
        pass
        
class ExponentialSelection(SelectionAlgorithm):
    '''runs when SELECTION_ALGORITHM = 'EXPONENTIAL' 
    influential words: all words in current sentence
    '''
    def updateinfluentialwords(self,*args,**kwds):#wordlist=[], problist=[]):
        #print 'exponential.updateinflw()...'
        self.influentialmultipliers = []
        self.influentialwords = self.sentobj.words
        for dis in range(len(self.influentialwords),0,-1):
            self.influentialmultipliers.append(np.exp(-self.c.EXPONENTIAL_EFFECT *\
                                            dis))
        #print 'influentialwords:',self.influentialwords
        #print 'influentialmltp:',self.influentialmultipliers

    def get_infld_nonterms(self,role):
        if (role == 'subject'):
            infld_nonterms_list = []
        elif (role == 'verb'):
            infld_nonterms_list = [SG_VERB_SYMS, PL_VERB_SYMS]      #[['<Vt11>','<Vi11>',...], ['<Vt12>', '<Vi12>', ...]]
        elif (role == 'object'):
            infld_nonterms_list =  [SG_NOUN_SYMS,PL_NOUN_SYMS,ADJ_SYMS,PREPN_SYMS,PREPV_SYMS]  #[['<N11>']]
        elif re.match('pos',role):
            infld_nonterms_list =  [SG_NOUN_SYMS,PL_NOUN_SYMS,ADJ_SYMS,PREPN_SYMS,PREPV_SYMS,SG_VERB_SYMS,PL_VERB_SYMS]  #[['<N11>']]
        else:
            raise Exception, 'what is the role to return its infld_nonterms_list?!'
        return infld_nonterms_list

    def adjust_verb_freq(self,objgr,ref_w_d,infld_nonterms,r=''):
        '''adjusts the frequency of verbs in a sentence '''
        history_terms = []
        history_terms_probs_d = {}
        influentialmultipliers = [1]
        #print 'infld_nonterms_list', infld_nonterms_list
        #print 'history_terms_list', history_terms_list
        #print 'history_terms_probs_d', history_terms_probs_d
        wordlist = [w[0] for w in objgr.d[infld_nonterms[0]]['rule']]
        ref_h_d = blissplot.read_prdnfile_count_convert_d(c.REF_PAIR_DIST_FILE,wordlist,pos=1)
        history_terms = ref_h_d.keys()
        history_terms_probs_d = utildict.normalize_d_values(ref_h_d)  
        
        return history_terms, history_terms_probs_d,influentialmultipliers
    def get_roles(self):
        return c.ROLES_EXP
    
    def adjust_position_freq(self,objgr,ref_w_d,infld_nonterms,r=''):
        history_terms = []
        history_terms_probs_d = {}
        influentialmultipliers = []
        #pdb.set_trace()
        pos = find_digit_in_str(r)
        for i in reversed(range(1,pos+1)):
            influentialmultipliers.append(np.exp(-c.EXPONENTIAL_EFFECT*i))
        wordlist = [w[0] for w in objgr.d[infld_nonterms[0]]['rule']]
        ref_h_d = blissplot.read_prdnfile_count_convert_d(c.REF_EXP_CORP_FILE,wordlist,pos=pos)
        history_terms = ref_h_d.keys()
        history_terms_probs_d = utildict.normalize_d_values(ref_h_d)  
        
        return history_terms, history_terms_probs_d,influentialmultipliers
def find_digit_in_str(s):
    pos = 0
    for i in range(10):
        if re.search(str(i),s):
            pos = i
            break
    return pos

class SbjVbSelection(SelectionAlgorithm):
    '''runs when SELECTION_ALGORITHM = 'SBJVB'
    influential words: Subject, Verb 
    '''
    def updateinfluentialwords(self,wordlist=[], problist=[]):
        cur_role = self.sentobj.roles[-1]
        if(cur_role == CNJ_ROLE):
            self.influentialwords = []
            self.influentialmultipliers = []
            return
        if (not self.influentialwords):
            sbj = self.sentobj.getsbj()
            if not sbj:
                self.sentobj.settempflag(-1, True) 
                self.sentobj.settempwordprobs(-1, wordlist, problist)
                self.sentobj.popprob_words(-1) 
            else: 
                self.influentialwords.append(sbj)
                self.influentialmultipliers.append(self.c.SBJ_EFFECT)
        elif (len(self.influentialwords) == 1):
            vb = self.sentobj.getvb()
            if not vb:
                self.sentobj.settempflag(-1, True) 
                self.sentobj.settempwordprobs(-1, wordlist, problist)
                self.sentobj.popprob_words(-1) 
            else: 
                self.influentialwords.append(vb)
                self.influentialmultipliers.append(1-self.c.SBJ_EFFECT)
                idx_vb = self.sentobj.roles.index(VB_ROLE)
                idx_sbj = self.sentobj.roles.index(SBJ_ROLE)
                l = self.sentobj.tempflags
                idx = [i for i in range(len(l)) if l[i]==True]
                for i in idx:                   
                    tempwords, tempprobs = self.sentobj.gettempwordprobs(i)
                    selectionprob = self.apply(tempwords,tempprobs,pos_wrt_ifwds=[self.dis2RL(i-idx_sbj),self.dis2RL(i-idx_vb)])
                    selectedword,p = toss_comulative(tempwords,selectionprob) #15
                    self.sentobj.appendprob_words(p) #15
                    r = self.sentobj.roles[i]
                    self.sentobj.replace_word(i,selectedword,r,\
                                    tmpflg=False)
    
    def get_infld_nonterms(self,role):
        if (role == 'subject'):
            infld_nonterms_list = []
        elif (role == 'verb'):
            infld_nonterms_list = [SG_VERB_SYMS, PL_VERB_SYMS]      #[['<Vt11>','<Vi11>',...], ['<Vt12>', '<Vi12>', ...]]
        elif (role == 'object'):
            infld_nonterms_list =  [SG_NOUN_SYMS,PL_NOUN_SYMS,ADJ_SYMS,PREPN_SYMS,PREPV_SYMS]  #[['<N11>']]
        else:
            raise Exception, 'what is the role to return its infld_nonterms_list?!'
        return infld_nonterms_list

    def adjust_verb_freq(self,objgr,ref_w_d,infld_nonterms,r=''):
        '''adjusts the frequency of verbs in a sentence '''
        history_terms = []
        history_terms_probs_d = {}
        influentialmultipliers = [1]
        '''
        if (infld_nonterms == SG_VERB_SYMS):
                history_nonterms =  SG_NOUN_SYMS + SG_PROPNOUN_SYMS
        elif (infld_nonterms == PL_VERB_SYMS):
                history_nonterms =  PL_NOUN_SYMS + PL_PROPNOUN_SYMS  #['<N11>','<PropN11>'] 
        else:
                raise Exception, 'What is the infld_nonterms to tell you the history_nonterms?!'
        #print 'infld_nonterms', infld_nonterms
        #print 'history_nonterms', history_nonterms

        history_terms = objgr.get_terminals(history_nonterms)
        h_t_p = utildict.lookup_dict(history_terms,ref_w_d)
        s = float(sum(h_t_p))
        h_t_p = [p/s for p in h_t_p]
        for h in history_terms:
                history_terms_probs_d[h] = h_t_p.pop(0)

        #print 'infld_nonterms_list', infld_nonterms_list
        #print 'history_terms_list', history_terms_list
        #print 'history_terms_probs_d', history_terms_probs_d
        '''
        wordlist = [w[0] for w in objgr.d[infld_nonterms[0]]['rule']]
        ref_h_d = blissplot.read_prdnfile_count_convert_d(c.REF_PAIR_DIST_FILE,wordlist,pos=1)
        history_terms = ref_h_d.keys()
        history_terms_probs_d = utildict.normalize_d_values(ref_h_d)  
        
        return history_terms, history_terms_probs_d,influentialmultipliers

    def get_inflmltps_object_role(self):
        return [c.SBJ_EFFECT, 1-c.SBJ_EFFECT]

    def get_roles(self):
        return c.ROLES_SBJ_VB   

class VbSbjSelection(SelectionAlgorithm):
    '''
        influential words: Verb, Subject
    '''
    def updateinfluentialwords(self, wordlist=[], problist=[]):
        cur_role = self.sentobj.roles[-1]
        if(cur_role == CNJ_ROLE):
            self.influentialwords = []
            self.influentialmultipliers = []
            return
        if (not self.influentialwords):
            vb = self.sentobj.getvb()
            if not vb:
                #cur_role = self.sentobj.roles[-1]
                #if (cur_role == SBJ_ROLE): 
                self.sentobj.settempflag(-1, True) 
                self.sentobj.settempwordprobs(-1, wordlist, problist)
                self.sentobj.popprob_words(-1) #15
                #print 'tempflag:',self.sentobj.tempflags
                #print 'tempword:',self.sentobj.tempwords
            else:
                self.influentialwords.append(vb)
                self.influentialmultipliers.append(1-self.c.SBJ_EFFECT)
                idx_vb = self.sentobj.roles.index(VB_ROLE)

                idx_sbj = self.sentobj.roles.index(SBJ_ROLE)
                tempwords, tempprobs = self.sentobj.gettempwordprobs(idx_sbj)
                selectionprob = self.apply(tempwords,tempprobs,pos_wrt_ifwds=[self.dis2RL(idx_sbj - idx_vb)])
                print 'changed selection prob in vbsbj:', selectionprob                
                selectedword,p = toss_comulative(tempwords,selectionprob) #15
                self.sentobj.appendprob_words(p) #15
                self.sentobj.replace_word(idx_sbj,selectedword,SBJ_ROLE,\
                                        tmpflg=False)
                self.influentialwords.insert(0,selectedword)
                self.influentialmultipliers.insert(0,self.c.SBJ_EFFECT)

                l = self.sentobj.tempflags
                idx = [i for i in range(len(l)) if l[i]==True]
                for i in idx:                   
                    tempwords, tempprobs = self.sentobj.gettempwordprobs(i)
                    selectionprob = self.apply(tempwords,tempprobs,pos_wrt_ifwds=[self.dis2RL(i-idx_sbj),self.dis2RL(i-idx_vb)])
                    selectedword,p = toss_comulative(tempwords,selectionprob) #15
                    self.sentobj.appendprob_words(p) #15
                    r = self.sentobj.roles[i]
                    self.sentobj.replace_word(i,selectedword,r,\
                                    tmpflg=False)
        #print 'sentwords:',self.sentobj.words
        #print 'inflwords:',self.influentialwords
        #print 'inflmltp:',self.influentialmultipliers
        # alist = [1,2,2,3];    
        # import numpy as np
        # a = np.array(alist);
        #... (a==2).nonzero()
        # ans:    (array([1, 2]),)
        # or ...
        #np.where( a == 2 )
        # ans:  (array([1, 2]),)
        # For more information please refer to "NumPy for MATLAB users" in the following url:
        #http://mathesaurus.sourceforge.net/matlab-numpy.html
        # troubleTicketObj.answer().done()
 
    def getjntprob(self, word1, word2, r='none',pos='R'):
        ''' looks for joint probability (droot[word1]+' '+droot[word2]) inside\
        djntprob and returns this probability
        '''
        #if word1=verb and word2=subject, their place must be changed in jntprob calc
        word1 = word1.lower()
        word2 = word2.lower()
        if (pos=='L' or r=='subject'):
            temp = word1
            word1 = word2
            word2 = temp            
        if word1 in droot:
            word1 = droot[word1]
        if word2 in droot:
            word2 = droot[word2]
        pair = word1+' '+ word2
        if pair in djntprob:
            jp = float(djntprob[pair])
        else:
            jp = 0
        return jp           
    
    def get_infld_nonterms(self,role):
        if (role == 'subject'):
            infld_nonterms_list = [SG_NOUN_SYMS,PL_NOUN_SYMS]
        elif (role == 'verb'):
            infld_nonterms_list = []        #[['<Vt11>','<Vi11>',...], ['<Vt12>', '<Vi12>', ...]]
        elif (role == 'object'):
            infld_nonterms_list =  [SG_NOUN_SYMS,PL_NOUN_SYMS,ADJ_SYMS,PREPN_SYMS,PREPV_SYMS]  #[['<N11>']]
        else:
            raise Exception, 'what is the role to return its infld_nonterms_list?!'
        return infld_nonterms_list

    def adjust_subject_freq(self,objgr,ref_w_d,infld_nonterms,r=''):
        '''adjusts the frequency of verbs in a sentence '''
        history_terms = []
        history_terms_probs_d = {}
        influentialmultipliers = [1]
        
        if (infld_nonterms == SG_NOUN_SYMS):
            history_nonterms =  SG_VERB_SYMS 
        elif (infld_nonterms == PL_NOUN_SYMS):
            history_nonterms =  PL_VERB_SYMS  
        else:
            raise Exception, 'What is the infld_nonterms to tell you the history_nonterms?!'
        #print 'infld_nonterms', infld_nonterms
        #print 'history_nonterms', history_nonterms

        history_terms = objgr.get_terminals(history_nonterms)
        h_t_p = utildict.lookup_dict(history_terms,ref_w_d)
        s = float(sum(h_t_p))
        if s:
            h_t_p = [p/s for p in h_t_p]
        for h in history_terms:
            history_terms_probs_d[h] = h_t_p.pop(0)

        #print 'infld_nonterms_list', infld_nonterms_list
        #print 'history_terms_list', history_terms_list
        #print 'history_terms_probs_d', history_terms_probs_d

        return history_terms, history_terms_probs_d,influentialmultipliers

    def get_inflmltps_object_role(self):
        return [c.SBJ_EFFECT, 1-c.SBJ_EFFECT]

    def get_roles(self):
        return c.ROLES_VB_SBJ
        
class BigramSelection(SelectionAlgorithm):
    '''runs when SELECTION_ALGORITHM = 'Bigram'
     Influential words: only last word of the current sentence
    '''
    def updateinfluentialwords(self,*args,**kwds):#wordlist=[], problist=[]):
        #print 'Bigram.updateinflw()...'
        
        self.influentialmultipliers = [1]
        self.influentialwords = [self.sentobj.words[-1]]

                                
droot = convertfiletodic(c.WORDROOTS_FILE,':') 
djntprob = convertfiletodic(c.JNTPROB_FILE,':')
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
def main():
    objgr = GrammarReader(c.GRAMMAR_FILE,dlms=None)
    objgr.readgrammar()

    if c.ADJUST_WORD_D:
        ref_w_d = convertfiletodic(c.REF_WORD_DIST_FILE, ' ',1, 0)
        #objgr.niceprint_(self.gr.d)
        #print 'ref_w_d:',ref_w_d
        objgr.d = objgr.create_grammar_dict(ref_w_d)
        #pdb.set_trace()       
        sentobj = UnderConstructSent(c)
        alg = factoryselectionalgorithm(sentobj,c)
        alg.adjust_word_freq(objgr,ref_w_d)

    if(c.SENGEN_CLASS == 'SENGEN'):
        objsn = SentenceGenerator(objgr,c)
    elif(c.SENGEN_CLASS == 'SENGEN_NOGRAM'):
        objsn = SentenceGenerator_NoGrammar(objgr,c)    
    elif(c.SENGEN_CLASS == 'SENGEN_FAIR'):
        objsn = FairSentenceGenerator(objgr, c)
    elif(c.SENGEN_CLASS == 'SENGEN_NOGRAM_POSDEP'):
        objsn = SentenceGenerator_NoGrammarPosdep(objgr,c)
    else:
        raise Exception, 'Choose SENGEN_CLASS (sentence generator class) correctly!'
    #print objgr.niceprint_(objgr.d)
    #print "***********************************"
    objsn.sentobj.f = open(objsn.c.SENTENCES_FILE, 'w')
    objsn.sentobj.fderivation = open(objsn.c.DERIVATION_FILE, 'w')
    objsn.sentobj.fprob = open(objsn.c.PROBS_FILE, 'w')
    objsn.sentobj.fsbjvb = open(objsn.c.SBJVB_FILE, 'w')
    objsn.sentobj.fsbjvbprd = open(objsn.c.SBJVB_PRD_FILE, 'w')

    #objsn.sentobj.f = open(objsn.c.SENTENCES_FILE+str(rank), 'w')
    #objsn.sentobj.fderivation = open(objsn.c.DERIVATION_FILE+str(rank), 'w')
    #objsn.sentobj.fprob = open(objsn.c.PROBS_FILE+str(rank), 'w')
    #objsn.sentobj.fsbjvb = open(objsn.c.SBJVB_FILE+str(rank), 'w')
    #objsn.sentobj.fsbjvbprd = open(objsn.c.SBJVB_PRD_FILE+str(rank), 'w')
    print 'Sentences will be written to:',objsn.c.SENTENCES_FILE
    if c.SENT_LEN_DIST:
        f = open(c.SENT_LEN_DIST_FILE,'r')
        sen_len_lst = pickle.load(f)
        for sen_len,sen_no in enumerate(sen_len_lst):
            sen_no = int(round(sen_no/4.0)) #because of having at least 4 parallel processor
            if sen_no:
                objsn.sen_len = sen_len
                objsn.generate(num_of_sentence = sen_no)

                objsn.sentobj.f = open(objsn.c.SENTENCES_FILE, 'a')
                #objsn.sentobj.f = open(objsn.c.SENTENCES_FILE+str(rank), 'a')
                #objsn.sentobj.fprob = open(objsn.c.PROBS_FILE+str(rank), 'a')
                #objsn.sentobj.fsbjvb = open(objsn.c.SBJVB_FILE+str(rank), 'a')
                #objsn.sentobj.fsbjvbprd = open(objsn.c.SBJVB_PRD_FILE+str(rank), 'a')

    else:
        objsn.generate()

if __name__=='__main__':
    t1=time()
    main()
    #print 'elapsed time = ',(time() - t1)
