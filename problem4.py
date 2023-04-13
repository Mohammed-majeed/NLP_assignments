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
    for ngram in ngrams:
        indices = [word_index_dict[w] for w in ngram]
        counts[tuple(indices)] += 1


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


#Difference between bigram_probs and smooth_probs
bigram_probs = np.loadtxt("bigram_probs.txt")
smooth_probs = np.loadtxt("smooth_probs.txt")

print("Difference between bigram_probs and smooth_probs:", np.sum(np.abs(bigram_probs - smooth_probs)))

"""
A difference of 1574.26 between the two files suggests that the smoothing had a significant
 effect on the bigram probabilities. This is not surprising, given that Laplace smoothing with 
 alpha=0.1 was applied to the counts. The effect of smoothing is likely to be more pronounced 
 for rare bigrams in the corpus, which have low raw counts and thus benefit more from smoothing.
"""



"""
Smoothing is useful in language modeling because it can help avoid assigning zero probability to 
unseen n-grams, which can lead to incorrect estimates of probabilities. The addition of a small 
constant value (alpha) to all counts in the corpus allows for the distribution of probability 
mass across all possible n-grams, even those that were not observed in the training corpus.

In this particular case, all four probabilities went down in the smoothed model because the alpha
smoothing decreased the overall probability mass for all n-grams, including the observed ones.
"""

"""
In the case of probabilities conditioned on 'the,' add-α smoothing causes less of a decrease 
compared to other words because 'the' is a high-frequency word that appears frequently in the corpus.
When we add α to the count of 'the,' its frequency is still much higher than most other words, 
even after smoothing. Therefore, the impact of smoothing on the probability of 'the' is relatively 
small compared to other words with lower frequency counts.
This behavior is desirable because smoothing can prevent overfitting to the training data and improve 
the model's ability to generalize to unseen data, making it more useful for real-world applications.

"""


