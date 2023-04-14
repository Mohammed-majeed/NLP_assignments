import math
import numpy as np
import nltk
from nltk.corpus import brown

def calculate_pmi(counts, word_index_dict, word_pair, total_words, min_count=10):
    """
    Calculates the Pointwise Mutual Information (PMI) for a given word pair.
    """
    word1, word2 = word_pair
    count_w1_w2 = counts[word_index_dict[word1], word_index_dict[word2]]
    count_w1 = counts[word_index_dict[word1]].sum()
    count_w2 = counts[:, word_index_dict[word2]].sum()
    
    if count_w1_w2 == 0:
        return 0.0
    
    p_w1_w2 = count_w1_w2 / total_words
    p_w1 = count_w1 / total_words
    p_w2 = count_w2 / total_words
    
    if p_w1 == 0 or p_w2 == 0:
        return 0.0
    
    pmi = math.log2(p_w1_w2 / (p_w1 * p_w2))
    
    return pmi

# Load Brown corpus
corpus_words = brown.words()

# Count word frequencies
word_freqs = nltk.FreqDist(corpus_words)

# Filter words that occur less than 10 times
filtered_words = [word for word in word_freqs if word_freqs[word] >= 10]

# Create word index dictionary
word_index_dict = {word: i for i, word in enumerate(filtered_words)}

# Initialize counts matrix
counts = np.zeros((len(filtered_words), len(filtered_words)), dtype=np.uint32)

# Iterate through the corpus and count word pairs
for i in range(len(corpus_words)-1):
    word1 = corpus_words[i]
    word2 = corpus_words[i+1]
    if word1 in word_index_dict and word2 in word_index_dict:
        counts[word_index_dict[word1], word_index_dict[word2]] += 1

# Calculate PMI for all word pairs
total_words = counts.sum()
pmi_scores = []
for word_pair in [(word1, word2) for word1 in filtered_words for word2 in filtered_words]:
    pmi = calculate_pmi(counts, word_index_dict, word_pair, total_words)
    pmi_scores.append((word_pair, pmi))

# Sort word pairs based on PMI scores
pmi_scores.sort(key=lambda x: x[1], reverse=True)

# Get top 20 word pairs with highest PMI values
top_20 = pmi_scores[:20]

# Sort word pairs based on PMI scores in ascending order
pmi_scores.sort(key=lambda x: x[1])

# Get bottom 20 word pairs with lowest PMI values
bottom_20 = pmi_scores[:20]

# Print the top 20 and bottom 20 word pairs with PMI values
print("Top 20 word pairs with highest PMI values:")
for pair, pmi in top_20:
    print(pair[0], pair[1], pmi)

print("Bottom 20 word pairs with lowest PMI values:")
for pair, pmi in bottom_20:
    print(pair[0], pair[1], pmi)
