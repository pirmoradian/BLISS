# BLISS

An artificial language incorporating both syntax and semantics. This code was used to generate results of [our publication](https://github.com/pirmoradian/BLISS/blob/master/Pirmoradian%2C%20Treves%20-%202011%20-%20Cognitive%20Computation.pdf) on constructing an artitifical language of intermediate complexity, which is suitable for language learning in neural networks. BLISS mimics natural languages by having a vocabulary of ~150 words, a context-free grammar, and some semantics, as defined by a degree of non-syntactic statistical dependence between words. 


## Code description

Main files:

* pybliss.py: The main file, which you need to run to generate sentences
* cnstpybliss.py: The constant file, where you can set the input and output file names and parameters

Auxiliary files that pybliss.py uses for sentence generation:

* blissutil.py: Contains the utility function for working with dictionary and for reading different file types in pybliss.py; 
* blissplot.py: Plotting functions for different file types
* KLdiv.py: Calculates Kullback–Leibler divergence between two different distributions
* CountFreq.py: Calculates the frequency/probability of all words/pairs inside a corpus    


## Generating sentences

1. Set the parameters in the constant file, cnstpybliss.py: 

   - FILE_NAME: the prefix for the name of the output files
   - GRAMMAR_FILE: the grammar file name
   - JNTPROB_FILE: the joint-frequency file name
   - UNIGRAMPROB_FILE: the word-frequency file name
   - WORDROOTS_FILE: the root of words
   - NUM_OF_SENTENCES: Number of sentences to be generated
   - PRINT_SENT_No: Print on the terminal after these number of sentences were generated
   - SELECTION_ALGORITHM: select the language model out of ALL_ALGORITHMS, which are different language models described in [our paper](https://github.com/pirmoradian/BLISS/blob/master/Pirmoradian%2C%20Treves%20-%202011%20-%20Cognitive%20Computation.pdf)

2. Run the main python file: 

```
python pybliss.py

```

Then, you find the generated sentences stored in FILE_NAME with txt format in the current directory.

## Developer

This package is developed by [Sahar Pirmoradian](https://www.researchgate.net/profile/Sahar_Pirmoradian). If you need help or have questions, don't hesitate to get in touch.
 
## Citation

If you use this code please cite the corresponding paper where BLISS was introduced:

S. Pirmoradian, A. Treves, “BLISS: an artificial language for learnability studies”, Journal of Cognitive Computation 3:539—553, 2011

## Acknowledgments

* This project was done under supervision of Prof. Alessandro Treves, in SISSA, Trieste, Italy.

