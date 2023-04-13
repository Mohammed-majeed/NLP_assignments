import numpy as np
from sklearn.preprocessing import normalize
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

# Load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    word_index_dict[line.strip()] = i

f = open("brown_100.txt")

n = 2
counts = np.zeros((len(word_index_dict),) * n)

# Iterate through file and update counts
for line in f:
    lower_line = line.lower()
    ngrams = get_ngrams(lower_line.strip(), n)
    previous_word = "<s>"
    for ngram in ngrams:
        indices = [word_index_dict[w] for w in ngram]
        counts[tuple(indices)] += 1
        previous_word = ngram[0]

f.close()

# Add alpha smoothing
alpha = 0.1
counts += alpha

# Normalize counts
probs = normalize(counts, norm='l1', axis=1)

np.savetxt("smooth_probs.txt", probs)

# Print some bigram probabilities
print("p(the | all) = ", probs[word_index_dict["all"], word_index_dict["the"]])
print("p(jury | the) = ", probs[word_index_dict["the"], word_index_dict["jury"]])
print("p(campaign | the) = ", probs[word_index_dict["the"], word_index_dict["campaign"]])
print("p(calls | anonymous) = ", probs[word_index_dict["anonymous"], word_index_dict["calls"]])

# Test GENERATE
gen = GENERATE(word_index_dict= word_index_dict, probs=probs, model_type='bigram',max_words=10,start_word='<s>')
print(gen)
