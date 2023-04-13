import nltk
# import nltk

# nltk.download('brown')
from nltk.corpus import brown

# from nltk.corpus import broun
import matplotlib.pyplot as plt
from scipy import special

# load the text
# with open('brown_100.txt','r') as f:
    
words = brown.words()
print(words)


freq_dist = nltk.FreqDist(words)
print(freq_dist)

# # count the frequency of each word
# freq_dist = nltk.FreqDist(emma)
# freq = freq_dist.values()

# # rank the words by frequency
# rank = range(1, len(freq)+1)
# print(rank)
# # plot the results on a log-log scale
# plt.loglog(rank, freq)
# plt.xlabel('Rank')
# plt.ylabel('Frequency')
# plt.title('Zipf\'s Law')
# plt.show()
