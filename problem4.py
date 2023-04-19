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

# Write the resulting probability to the output file
output_file = open("smooth_probs.txt", "w")
output_file.write("p(the | all) = " + str(probs[word_index_dict["all"], word_index_dict["the"]]) + "\n")
output_file.write("p(jury | the) = " + str(probs[word_index_dict["the"], word_index_dict["jury"]]) + "\n")
output_file.write("p(campaign | the) = " + str(probs[word_index_dict["the"], word_index_dict["campaign"]]) + "\n")
output_file.write("p(calls | anonymous) = " + str(probs[word_index_dict["anonymous"], word_index_dict["calls"]]) + "\n")
output_file.close()



"""
Finally, compare smooth_probs.txt to bigram_probs.txt.

The results from bigram_probs.txt and smooth_probs.txt are different. This is expected since the 
smooth_probs.txt file includes the application of smoothing (additive smoothing) to the bigram
 probabilities, whereas the bigram_probs.txt file only contains the raw bigram probabilities. 
 Smoothing helps to address the issue of zero probability for unseen bigrams, but it also alters
   the probabilities of existing bigrams. Therefore, the smoothed probabilities in smooth_probs.txt 
   are generally lower than the raw probabilities in bigram_probs.txt.
"""

# for problem 6
#calculate perplexity of sentences in toy_corpus.txt and write to file
toy_corpus = open("toy_corpus.txt")
output_file = open("smoothed_eval.txt", "w")

for sentence in toy_corpus:
    # Tokenize the sentence into individual words
    words = (sentence.lower()).split()
    # Calculate the joint probability of all the words under the bigram model
    sentence_prob = 1.0
    #get length of sentence
    sent_len = len(sentence.split()[0:-1])
    for i in range(sent_len):
        index1 = word_index_dict[words[i]]
        index2 = word_index_dict[words[i+1]]
        # Retrieve the corresponding probability from the probs array
        bigram_prob = probs[index1, index2]
        # Multiply the probability into the joint probability
        sentence_prob *= bigram_prob

    #calculate perplexity of sentence
    perplexity = 1.0 / pow(sentence_prob, 1.0/sent_len)
    print(perplexity)

    output_file.write(str(perplexity) + "\n")

toy_corpus.close()
output_file.close()



# for problem 7
with open('smoothed_generation.txt', 'w') as f:
    for i in range(10):
        generated_text = GENERATE(word_index_dict=word_index_dict, probs=probs, model_type='bigram', max_words=10, start_word='<s>')
        f.write(generated_text + '\n')



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


