# TweetCorpus

Python 3 class for working with (large sets of) tweets. Includes a tokenizer, token classification, language detection, and sentiment analysis.

I made this class to parse tweets in a more flexible way, without making too many assumptions about how the output is used. The core of the class is the classify() method, which records each token's position, classifies it (e.g., as a hashtag, mention, stopword, contentword, punctuation, etc.) and stems it (only for some tokens). It allows the user to quickly extract "All mentions that occur at the beginning of a tweet", "Hashtag co-occurrences", "All tokens which have 'function' as their stem" and "The frequency distribution of emoticons" and the like.

Most of the functions are parallelized and/ or use little memory through the use of generator functions. With parallelization disabled, it generates a CSV file of all stemmed and classified tokens for 1,2M tweets in ~24 minutes on my i5 dual core laptop with 8GB memory; with parallelization enabled, it does the same for a smaller, 250K Tweets dataset in ~3 minutes.

# Typical usage

```
from TweetCorpus import tweetcorpus

with open(filename) as f:
    tweets = f.readlines()

twc = tweetcorpus.TweetCorpus(tweets)

# get a vocabulary, which can then be used to do lexicon-based flagging of truncated words, hashtags, etc
vocab = twc.build_vocabulary()

# instantiate new class with vocabulary
twc = tweetcorpus.TweetCorpus(tweets, vocabulary = vocab)

# do tokenizing
twc.tokenize()

# classify and stem tokens
twc.classify()

# write classified tokens to csv
twc.tokens_csv('tokens_export.csv')

# do sentiment analysis
twc.analyze_sentiments()

twc.sentiments_csv('sentiments_export.csv')
```

# Usage with a large dataset in csv format

```
import csv
from TweetCorpus import tweetcorpus
from TweetCorpus import tweetcorpus

def stream_tweets(filename):
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row['text']

twc = tweetcorpus.TweetCorpus(stream_tweets('/path/to/file'), parallel=False) # turn off parallellisation to work through the tweets serially, which reduces memory consumption enormously

# write tokens to csv
twc.tokens_csv('tokens_export.csv')

```

# Tips

* Internally, the class uses persistent tweet IDs. These are automatically generated by default, but to use your own tweet IDs, just pass the corpus as a list of tuples (tweet_id, tweet_text).
* Switch off parallellisation to work with huge files (>500K tweets) by passing parallel = False when instantiating the class

# Credits

This class is built on the twokenize tokenizer (use the ark-twokenize-py fork by Sentimentron for a working Python 3 version, https://github.com/Sentimentron/ark-twokenize-py), the vaderSentiment library for sentiment analysis, and langdetect and pycountry for automated language detection.

Some regular expressions and ideas were borrowed from tweet-preprocessor, https://github.com/s/preprocessor
