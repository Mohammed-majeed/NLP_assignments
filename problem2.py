#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from generate import GENERATE
import re

vocab = open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for index, line in enumerate(vocab):
    # Import part 1 code to build dictionary
    word = line.rstrip('\n').lower()
    word_index_dict[word] = index

f = open("brown_100.txt")

counts = np.zeros(813) # Initialize counts to a zero vector

# Iterate through file and update counts
with open("brown_100.txt", "r") as f:
    for line in f:
        words = line.split(" ") # Split line into words
        for word in words:
            word = word.lower()
            if word in word_index_dict:
                index = word_index_dict[word]
                counts[index] += 1

# Normalize and writeout counts. 
probs = counts / np.sum(counts)
np.savetxt("unigram_probs_100.txt", probs)
loaded_probs = np.loadtxt("unigram_probs_100.txt")

# TODO: Write to awnser Q&A