import nltk
from nltk.corpus import brown
from math import log2

# Get the counts of individual words and word pairs
word_counts = brown.words()
pair_counts = nltk.bigrams(word_counts)

freq_words = nltk.FreqDist(word_counts)
freq_pairs = nltk.FreqDist(pair_counts)

# Calculate the PMI for each word pair
pmi_values = {}
N = freq_words.N()

for pair in freq_pairs:
    w1, w2 = pair
    if freq_pairs[pair] < 10 or freq_words[w1] < 10 or freq_words[w2] < 10:
        continue
    
    pmi = log2((freq_pairs[pair] / N) / (freq_words[w1] / N * freq_words[w2] / N))
    pmi_values[pair] = pmi

# Print the 20 word pairs with highest PMI values
sorted_pmi = sorted(pmi_values.items(), key=lambda x: x[1], reverse=True)
print("Top 20 word pairs with highest PMI values:")
for pair, pmi in sorted_pmi[:20]:
    print(pair[0], pair[1], pmi)

# Print the 20 word pairs with lowest PMI values
print("\nBottom 20 word pairs with lowest PMI values:")
for pair, pmi in sorted_pmi[-20:]:
    print(pair[0], pair[1], pmi)
    
# Histogram to show the validity of the independence assumption
import matplotlib.pyplot as plt
num_bins = 100
plt.hist(pmi_values.values(), bins=num_bins, color='blue', alpha=0.5)
plt.title('PMI Distribution')
plt.xlabel('PMI')
plt.ylabel('Frequency')
plt.show()

abs_pmi = dict()
for k,v in pmi_values.items():
    abs_pmi[k] = abs(v)

num_bins = 100
plt.hist(abs_pmi.values(), bins=num_bins, color='blue', alpha=0.5)
plt.title('PMI abs Distribution')
plt.xlabel('abs(pmi)')
plt.ylabel('Frequency')
plt.show()

mean = sum(abs_pmi.values())/len(abs_pmi.values())

print('average absolute value of PMI: ', mean)

# t-test
import numpy as np
from scipy.stats import ttest_1samp

data = np.array(list(pmi_values.values()))
t_statistic, p_value = ttest_1samp(data, 2.6, alternative = "greater")

print("t statistic =", t_statistic)
print("p value =", p_value)

# Wilcoxon signed-rank test
import scipy.stats as stats
stats.wilcoxon(data-2.5, alternative = "greater" )