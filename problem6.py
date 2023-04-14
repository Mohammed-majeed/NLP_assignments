import numpy as np

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

#normalize and writeout counts.
probs = counts / np.sum(counts)
# np.savetxt("unigram_probs_100.txt", probs)
print("First probability",probs[0])
print("last probability",probs[-1])

#calculate perplexity of sentences in toy_corpus.txt and write to file
toy_corpus = open("toy_corpus.txt")
output_file = open("unigram_eval.txt", "w")

for sentence in toy_corpus:
    print("sentence",sentence)
    #clean up the sentence
    sentence = sentence.strip()
    #initialize sentence probability
    sentprob = 1.0
    #get length of sentence
    sent_len = len(sentence.split()) + 1
    #calculate probability of each word in sentence
    for word in (sentence.lower()).split():
        wordprob = probs[word_index_dict[word]]
        sentprob *= wordprob
        print("sentprob",sentprob)
    #calculate perplexity of sentence
    perplexity = 1.0 / pow(sentprob, 1.0/sent_len)
    #write perplexity to output file
    print("perplexityperplexity",perplexity)
    output_file.write(str(perplexity) + "\n")

toy_corpus.close()
output_file.close()
