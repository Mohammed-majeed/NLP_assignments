import nltk
# nltk.download('brown')
from nltk.corpus import brown
import matplotlib.pyplot as plt


# Get the list of words
words = brown.words()
print(words)
# Calculate the number of tokens and types
num_tokens = len(words)
num_types = len(set(words))

# Calculate the number of words
num_words = len(words)

# Calculate the average number of words per sentence
num_sentences = len(brown.sents())
avg_words_per_sentence = num_words / num_sentences

# Calculate the average word length
total_word_length = sum(len(word) for word in words)
avg_word_length = total_word_length / num_words

# Get the frequency distribution of the words
fdist = nltk.FreqDist(words)

# Get the 10 most frequent POS tags
tags = [tag for (word, tag) in brown.tagged_words()]
fdist_tags = nltk.FreqDist(tags)
top_tags = fdist_tags.most_common(10)

# Get the list of unique words sorted by descending frequency
sorted_words = sorted(set(words), key=lambda x: -fdist[x])

# Print the results
print("Number of tokens:", num_tokens)
print("Number of types:", num_types)
print("Number of words:", num_words)
print("Average number of words per sentence:", avg_words_per_sentence)
print("Average word length:", avg_word_length)
print("Top 10 POS tags:", top_tags)


# Plot the frequency curve for the whole corpus
freq_list = [fdist[word] for word in sorted_words]
plt.plot(freq_list)
plt.title("Frequency Curve - Brown Corpus")
plt.xlabel("Word Rank")
plt.ylabel("Frequency")
plt.show()

# Plot the frequency curves for two different genres
genres = ["news", "science_fiction"]
for genre in genres:
    genre_words = brown.words(categories=genre)
    genre_fdist = nltk.FreqDist(genre_words)
    genre_sorted_words = sorted(set(genre_words), key=lambda x: -genre_fdist[x])
    genre_freq_list = [genre_fdist[word] for word in genre_sorted_words]
    plt.plot(genre_freq_list, label=genre)

plt.title("Frequency Curves - Brown Corpus")
plt.xlabel("Word Rank")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Plot the log-log frequency curves for the whole corpus and two different genres
plt.loglog(freq_list)
plt.title("Log-Log Frequency Curve - Brown Corpus")
plt.xlabel("Log Word Rank")
plt.ylabel("Log Frequency")
plt.show()

for genre in genres:
    genre_words = brown.words(categories=genre)
    genre_fdist = nltk.FreqDist(genre_words)
    genre_sorted_words = sorted(set(genre_words), key=lambda x: -genre_fdist[x])
genre_freq_list = [genre_fdist[word] for word in genre_sorted_words]
plt.loglog(genre_freq_list, label=genre)

plt.title("Log-Log Frequency Curves - Brown Corpus")
plt.xlabel("Log Word Rank")
plt.ylabel("Log Frequency")
plt.legend()
plt.show()




"""
The Brown Corpus was established in 1961 at Brown University, and it was the first electronic corpus of English to contain a million words.
 The corpus comprises text from 500 different sources, which are classified into various genres, including news and editorial. 
 The corpus has played a significant role in advancing linguistic research, as it has enabled scholars to study language use and
   variation across different genres and contexts.

One of the key linguistic observations from the Brown Corpus is the variation in word frequencies across different genres.
 For instance, function words such as articles and prepositions were found to be more frequent in comparison to content words such as nouns and verbs.
   This observation suggests that language use is highly influenced by genre and context. Additionally, the corpus revealed that 
   certain words were more commonly used in some genres than in others. For example, the use of pronouns was found to be more frequent
     in fiction than in news. These findings have important implications for language modeling, natural language processing, 
     and our understanding of language structure and function.

"""