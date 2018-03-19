# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 23:06:40 2009

@author: Sahar
"""

FILE_NAME = 'BLISS_nosem_15Mar18' # Pass the prefix for the name of the output files

GRAMMAR_FILE = 'mgrammar-Shakespeare.txt' # Pass the grammar file name

JNTPROB_FILE ='jntFqall.txt' # Pass the joint-frequency file name

UNIGRAMPROB_FILE = 'wFreq.txt' # Pass the word-frequency file name

#UNIGRAMPOSDEPPROB_FILE = 'wProb_nosem_5Nov_pos'
WORDROOTS_FILE =  'wordRoot.txt' # the root of words

# SET Parameters
NUM_OF_SENTENCES = 100 # Number of sentences to be generated
PRINT_SENT_No = 100000  # Print on the terminal after these number of sentences were generated

SEMANTICS_MULTIPLIER = 1

ALL_ALGORITHMS = ['NOSEMANTICS', 'EXPONENTIAL', 'SBJVB', 'VBSBJ', 'UNIGRAM','BIGRAM']
SELECTION_ALGORITHM = ALL_ALGORITHMS[0] # select the language model

ALL_SENGEN_CLASSES = ['SENGEN', 'SENGEN_NOGRAM', 'SENGEN_NOGRAM_POSDEP', 'SENGEN_FAIR']
SENGEN_CLASS =  ALL_SENGEN_CLASSES[0]  
EXPONENTIAL_EFFECT = 1./3 
SENT_LEN_DIST = False 
SBJ_EFFECT = 0.2 
CAT_EFFECT = 0.9
SENT_LEN_UG = 5


#OUTPUT FILES
SENTENCES_FILE = FILE_NAME + '.txt' # The generated sentences will be stored here
DERIVATION_FILE = FILE_NAME + '_dervorg.txt' # derivatives of grammar, which are used for generating each sentence
PROBS_FILE = SENTENCES_FILE + '_probs' # probability of each sentence and the seleciton prob of each word in the sentence
SBJVB_FILE = SENTENCES_FILE + '_sbjvb' # all subjects and verbs of generated sentences
SBJVB_PRDN_FILE = SENTENCES_FILE + '_svprdn' # all subjects, verbs, and "noun" predicates of generated sentences
SBJVB_PRD_FILE = SENTENCES_FILE + '_svprd' # all subjects, verbs, and predicates of generated sentences


NOUN_CATEGORY_FILES = ['nounAselshak.txt','nounBselshak.txt','nounOselshak.txt'] # nouns categorised to 3 different categories, animals, buildings, and objects

REF_WORD_DIST_FILE ='wFreq_nosem_5Nov.txt'
REF_PAIR_DIST_FILE = 'BLISS_nosem_5Nov.txt'+ '_svprd' 
REF_EXP_CORP_FILE ='BLISS_nosem_5Nov.txt' 
ADJUST_WORD_D = False #False/True

ROLES_SBJ_VB =  ['subject','verb','object']
ROLES_VB_SBJ = ['subject','verb','object']
ROLES_EXP = ['pos1','pos2','pos3','pos4','pos5','pos6','pos7']


