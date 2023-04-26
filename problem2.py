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

# for problem 6
#calculate perplexity of sentences in toy_corpus.txt and write to file
toy_corpus = open("toy_corpus.txt")
output_file = open("unigram_eval.txt", "w")

for sentence in toy_corpus:
    # Tokenize the sentence into individual words
    words = (sentence.lower()).split()
    # Calculate the joint probability of all the words under the unigram model
    sentence_prob = 1.0
    #get length of sentence
    sent_len = len(sentence.split())
    for word in words:
        # Look up the index of the word in the word_index_dict
        if word in word_index_dict:
            index = word_index_dict[word]
            # Retrieve the corresponding probability from the probs array
            word_prob = probs[index]
            # Multiply the probability into the joint probability
            sentence_prob *= word_prob
    #calculate perplexity of sentence
    perplexity = 1.0 / (pow(sentence_prob, 1.0/sent_len))
    # Write the resulting probability to the output file
    output_file.write(str(perplexity) + "\n")

toy_corpus.close()
output_file.close()

# for problem 7
with open('unigram_generation.txt', 'w') as f:
    for i in range(10):
        generated_text = GENERATE(word_index_dict=word_index_dict, probs=probs, model_type='unigram', max_words=10, start_word='<s>')
        f.write(generated_text + '\n')

