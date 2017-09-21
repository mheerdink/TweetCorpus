# TweetCorpus
Class for working with tweets. Includes a tokenizer, token classification, language detection, and sentiment analysis.

I made this class to make it possible to work with Tweets in a highly streamlined yet still flexible way. I have made as few assumptions as possible.

# Example usage
```
from TweetCorpus import tweetcorpus

with open(filename) as f:
    tweets = f.readlines()

twc = tweetcorpus.TweetCorpus(tweets)

# do tokenizing
twc.tokenize()

# classify tokens
twc.classify()

# do sentiment analysis
twc.analyze_sentiments()

# write tokens to csv
twc.tokens_csv('tokens_export.csv')

```

# Credits
This class is built on the NLTK casual ("Twitter-aware") tokenizer, the vaderSentiment library for sentiment analysis, and langdetect and pycountry for automated language detection.

Some regular expressions and ideas were borrowed from tweet-preprocessor, https://github.com/s/preprocessor