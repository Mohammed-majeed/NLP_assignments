#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models
@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller
DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

word_index_dict = {}

#read brown_vocab_100.txt into word_index_dict
with open('brown_vocab_100.txt', 'r') as f:
    for index, line in enumerate(f):
        word = line.rstrip('\n')
        word_index_dict[word] = index


#read brown_vocab_100.txt into word_index_dict .
with open('brown_vocab_100.txt', 'r') as f:
    for index, line in enumerate(f):
        word = line.rstrip('\n')
        word_index_dict[word] = index