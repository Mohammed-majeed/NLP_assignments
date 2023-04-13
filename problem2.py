# #!/usr/bin/env python3

# """
# NLP A2: N-Gram Language Models

# @author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

# DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
# """

import numpy as np
from generate import GENERATE


def get_ngrams(sentence, n):
    """
    Returns all n-grams in the given sentence.
    """
    words = sentence.split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(tuple(words[i:i+n]))
    return ngrams

vocab = open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    word_index_dict[line.strip()] = i
f = open("brown_100.txt")

n=1
counts = np.zeros((len(word_index_dict),) * n)

#iterate through file and update counts
for line in f:
    lower_line = line.lower()
    ngrams = get_ngrams(lower_line.strip(), n)
    for ngram in ngrams:
        indices = [word_index_dict[w] for w in ngram]
        counts[tuple(indices)] += 1

f.close()

print(counts)

#normalize and writeout counts.
probs = counts / np.sum(counts)
np.savetxt("unigram_probs_100.txt", probs)
print("First probability",probs[0])
print("last probability",probs[-1])

# test with GENERATE
gen = GENERATE(word_index_dict=word_index_dict, probs=probs, model_type='unigram', max_words=5, start_word='<s>')
print(gen)

"""
Intuitively, we might expect that the proportion of words that occur only once would be lower in a
 larger corpus. This is because as the size of the corpus increases, the chances of encountering
   rare words also increase, and the rarest words become less rare. For example, consider the word 
   "supercalifragilisticexpialidocious" - this word is extremely rare in everyday language, but if
     we were to analyze a large corpus of English text, we would likely encounter it multiple times.

Therefore, as we increase the size of the corpus, we would expect the proportion of words that occur 
only once to decrease. However, the rate at which this decrease occurs depends on the characteristics
 of the corpus, such as the domain, genre, and time period. In some cases, such as with highly 
 specialized technical language or historical texts, the proportion of rare words may actually
   increase as the corpus size grows.
"""

# #!/usr/bin/env python3

# """
# NLP A2: N-Gram Language Models

# @author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

# DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
# """

# import numpy as np
# from generate import GENERATE
# import re

# vocab = open("brown_vocab_100.txt")

# #load the indices dictionary
# word_index_dict = {}
# for index, line in enumerate(vocab):
#     # Import part 1 code to build dictionary
#     word = line.rstrip('\n').lower()
#     word_index_dict[word] = index

# f = open("brown_100.txt")

# counts = np.zeros(813) # Initialize counts to a zero vector

# # Iterate through file and update counts
# with open("brown_100.txt", "r") as f:
#     for line in f:
#         words = line.split(" ") # Split line into words
#         for word in words:
#             word = word.lower()
#             if word in word_index_dict:
#                 index = word_index_dict[word]
#                 counts[index] += 1

# # Normalize and writeout counts. 
# probs = counts / np.sum(counts)
# np.savetxt("unigram_probs_100.txt", probs)
# loaded_probs = np.loadtxt("unigram_probs_100.txt")

# # TODO: Write to awnser Q&A