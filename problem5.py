import numpy as np
from sklearn.preprocessing import normalize


def get_ngrams(sentence, n):
    """
    Returns all n-grams in the given sentence.
    """
    words = sentence.split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(tuple(words[i:i+n]))
    return ngrams

# Load the vocabulary
vocab = open("brown_vocab_100.txt")

# Load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    word_index_dict[line.strip()] = i

# Load the corpus
f = open("brown_100.txt")

# Initialize counts array
n = 3
counts = np.zeros((len(word_index_dict),) * n)

# Iterate through file and update counts
for line in f:
    lower_line = line.lower()
    trigrams = get_ngrams(lower_line.strip(), n)
    for trigram in trigrams:
        indices = [word_index_dict[w] for w in trigram]
        counts[tuple(indices)] += 1

f.close()


# # Normalize counts
counts_2d = counts.reshape(-1, counts.shape[-1])
# Normalize counts
probs = normalize(counts_2d, norm='l1', axis=1)
# Reshape probs back to 3D
probs = probs.reshape(counts.shape)
print("#### unsmoothed ####")
print("p(past | in, the) (unsmoothed) = ", probs[word_index_dict["in"], word_index_dict["the"], word_index_dict["past"]])
print("p(time | in, the) (unsmoothed) = ", probs[word_index_dict["in"], word_index_dict["the"], word_index_dict["time"]])
print("p(said | the, jury) (unsmoothed) = ", probs[word_index_dict["the"], word_index_dict["jury"], word_index_dict["said"]])
print("p(recommended | the, jury) (unsmoothed) = ", probs[word_index_dict["the"], word_index_dict["jury"], word_index_dict["recommended"]])
print("p(that | jury, said) (unsmoothed) = ", probs[word_index_dict["jury"], word_index_dict["said"], word_index_dict["that"]])



# Add alpha smoothing
alpha = 0.1
counts += alpha

# # Normalize counts
probs = counts / counts.sum(axis=-1, keepdims=True)

print("#### unsmoothed ####")
# Compute probabilities
print("p(past | in, the) (smoothed) = ", probs[word_index_dict["in"], word_index_dict["the"], word_index_dict["past"]])
print("p(time | in, the) (smoothed) = ", probs[word_index_dict["in"], word_index_dict["the"], word_index_dict["time"]])
print("p(said | the, jury) (smoothed) = ", probs[word_index_dict["the"], word_index_dict["jury"], word_index_dict["said"]])
print("p(recommended | the, jury) (smoothed) = ", probs[word_index_dict["the"], word_index_dict["jury"], word_index_dict["recommended"]])
print("p(that | jury, said) (smoothed) = ", probs[word_index_dict["jury"], word_index_dict["said"], word_index_dict["that"]])
