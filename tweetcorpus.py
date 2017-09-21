import sys
import csv
import regex
from nltk.corpus import stopwords
from nltk.tokenize import casual

class TweetCorpus:
    def __init__(self, corpus, language = 'auto'):
        assert(isinstance(corpus, list) and all([isinstance(t, str) for t in corpus]))
        self.language = language
        self.corpus = corpus
    
    def tokenize(self, preserve_case=False, reduce_len=False, strip_handles=False):
        if not hasattr(self, 'tokens'):
            tokenizer = casual.TweetTokenizer(preserve_case = preserve_case, reduce_len = reduce_len, strip_handles = strip_handles)
            self.tokens = [tokenizer.tokenize(t) for t in self.corpus]
        
        return self.tokens
    
    def classify(self, force = False, vocabulary = None):
        if not hasattr(self, 'classification') or force:
            tokens = self.tokenize()
            classifier = TweetTokenClassifier() # instantiate without vocabulary
            self.classification = [classifier.classify(t) for t in tokens]
            
            if vocabulary is None:
                vocab = [w for t in self.get_tokens_by_type(['contentword', 'stopword', 'mention', 'hashtag', 'ellipsis']) for w in t]
                vocabulary = set([vocab[i] for i in range(len(vocab)) if not classifier.ELLIPSIS_PATTERN.match(vocab[i]) and (i == len(vocab) - 1 or not classifier.ELLIPSIS_PATTERN.match(vocab[i + 1]))])
            
            self.__flag_abbreviations(vocabulary)
        
        return self.classification
    
    def __flag_abbreviations(self, vocabulary):
        c = self.classify()
        
        for i in range(len(c)):
            tokens = c[i]
            for j in range(len(tokens)):
                if (tokens[j]['position'] >= 2 and
                    tokens[j]['type'] == 'ellipsis' and
                    tokens[j - 1]['type'] in ['contentword', 'stopword', 'mention', 'hashtag'] and
                    not any([t['type'] in ['contentword', 'stopword'] for t in tokens[j:(len(tokens) - 1)]]) and
                    (len(tokens[j - 1]['token']) < 2 or tokens[j - 1]['token'] not in vocabulary)):
                        tokens[j - 1]['type'] = tokens[j - 1]['type'] + '_abbreviated'
    
    def analyze_sentiments(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError:
            print("Install vaderSentiment to do sentiment analysis; pip install vaderSentiment")
            return False
        
        analyzer = SentimentIntensityAnalyzer()
        
        tweets_clean = self.clean()
        scores = []
        
        for i in range(len(tweets_clean)):
            vs = analyzer.polarity_scores(tweets_clean[i])
            vs['tweetno'] = i
            vs['tweet'] = self.corpus[i]
            vs['tweet_clean'] = tweets_clean[i]
            scores.append(vs)
        
        return scores
    
    def get_tokens_by_type(self, types = ['contentword']):
        if isinstance(types, str):
            types = [types]
        
        def __get_token_type(tokens, types): # function to run on a single array of tokens
            return [t['token'] for t in tokens if t['type'] in types]
        
        c = self.classify()
        return [__get_token_type(t, types) for t in c]
    
    def clean(self, keep_types = ['contentword', 'stopword']):
        return [' '.join(w) for w in self.get_tokens_by_type(keep_types)]
    
    def vocabulary(self, types = ['contentword']):
        return sorted(set([w for token in self.get_tokens_by_type(types) for w in token]))
    
    def as_dict(self):
        c = self.classify()
        
        result = []
        
        for i in range(len(c)):
            t = list(c[i])
            if (len(t) > 0):
                for token in t:
                    token['tweetno'] = i
                    token['tweet'] = self.corpus[i]
                    result.append(token)
        
        return result
    
    def tokens_csv(self, filename):
        out = self.as_dict()
        fieldnames = ['tweetno', 'tweet', 'position', 'token', 'type']
        
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
            writer.writeheader()
            for w in out:
                writer.writerow(w)
    
    def sentiments_csv(self, filename):
        sent = self.analyze_sentiments()
        fieldnames = ['tweetno', 'tweet', 'tweet_clean', 'pos', 'neg', 'neu', 'compound']
        
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
            writer.writeheader()
            for w in sent:
                writer.writerow(w)

class TweetTokenClassifier:
    STOPWORDS = {
        "english": set(stopwords.words('english')).union(set(["i", "me", "my",
        "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
        "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
        "these", "those", "am", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "having", "do", "does", "did", "doing",
        "would", "should", "could", "ought", "i'm", "you're", "he's", "she's",
        "it's", "we're", "they're", "i've", "you've", "we've", "they've", "i'd",
        "you'd", "he'd", "she'd", "we'd", "they'd", "i'll", "you'll", "he'll",
        "she'll", "we'll", "they'll", "isn't", "aren't", "wasn't", "weren't",
        "hasn't", "haven't", "hadn't", "doesn't", "don't", "didn't", "won't",
        "wouldn't", "shan't", "shouldn't", "can't", "cannot", "couldn't",
        "mustn't", "let's", "that's", "who's", "what's", "here's", "there's",
        "when's", "where's", "why's", "how's", "a", "an", "the", "and", "but",
        "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
        "with", "about", "against", "between", "into", "through", "during",
        "before", "after", "above", "below", "to", "from", "up", "down", "in",
        "out", "on", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "y'all",
        "ya'll"]))
    }
    RESERVED_WORDS_PATTERN = regex.compile(r'(RT|rt|FAV|fav)$')
    ELLIPSIS_PATTERN = regex.compile(r'(?:\.{3}|…)$')
    URL_PATTERN = regex.compile(r'\s*(?i)\b((?:https?:/{1,3}|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))*(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
    MENTION_PATTERN = regex.compile(casual.REGEXPS[5])
    HASHTAG_PATTERN = regex.compile(r'((?:#|＃)[\w_]*[\w][\w_]*)')
    EMAIL_PATTERN = regex.compile(casual.REGEXPS[7])
    NUMBERS_PATTERN = regex.compile(r"(\-?[,.]?\d+(?:[,.]\d+)*)")
    NUMERIC_EXPRESSION_PATTERN = regex.compile(NUMBERS_PATTERN.pattern + r'(?:[+-/*=]+' + NUMBERS_PATTERN.pattern + r')+')
    MIXED_PATTERN = regex.compile(r'(?:[^\W\d]+(\d+[^\W\d]*)+)|(?:\d+([^\W\d]+\d*)+)')
    
    if (sys.maxunicode > 65535):
        # UCS-4
        EMOJIS_PATTERN = regex.compile(u'([\U00002600-\U000027BF]\U0000FE0F?)|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])|(\U0000FE0F)')
    else:
        # UCS-2
        EMOJIS_PATTERN = regex.compile(u'([\u2600-\u27BF]\uFE0F?)|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])|(\uFE0F)')
    
    def classify(self, tokens, language = 'english'):
        assert(isinstance(tokens, list) and all([isinstance(t, str) for t in tokens]))
        
        if language == 'auto':
            language = self.__guess_language(tokens)
        
        result = []
        position = 1
        sw = self.__get_stopwords(language)
        
        for i in range(len(tokens)):
            type = "contentword"
            
            if (i == 0 and len(tokens) >= 2 and # someone is favourited or retweeted
                self.RESERVED_WORDS_PATTERN.match(tokens[i])):
                type = "reserved_word"
                if (len(tokens) >= 3 and tokens[2] == ":"):
                    position = position - 3
                else:
                    position = position - 2
            elif self.MENTION_PATTERN.match(tokens[i]):
                if i == 1 and result[0]['token'] in ['rt', 'RT']:
                    type = "retweeted_user"
                else:
                    type = "mention"
            elif self.HASHTAG_PATTERN.match(tokens[i]):
                type = "hashtag"
                m = self.HASHTAG_PATTERN.match(tokens[i])
                tokens[i] = m.groups()[0] # drop any unnecessary characters from the end, if any (this fixes a tokenizer artefact)
            elif self.EMAIL_PATTERN.fullmatch(tokens[i]):
                type = "email"
            elif self.ELLIPSIS_PATTERN.fullmatch(tokens[i]):
                type = "ellipsis"
            elif self.MIXED_PATTERN.fullmatch(tokens[i]):
                type = "mixed"
            elif self.NUMBERS_PATTERN.fullmatch(tokens[i]):
                type = "number"
            elif self.NUMERIC_EXPRESSION_PATTERN.fullmatch(tokens[i]):
                type = "numeric_expression"
            elif self.URL_PATTERN.match(tokens[i]):
                if self.ELLIPSIS_PATTERN.match(tokens[i][len(tokens[i]) - 1]): # ends with ellipsis
                    type = "url_abbreviated"
                else:
                    type = "url"
            elif self.EMOJIS_PATTERN.match(tokens[i]):
                type = "emoji"
            elif casual.EMOTICON_RE.match(tokens[i]):
                type = "emoticon"
            elif regex.match(r'[\p{Sc}%]$', tokens[i]):
                type = "unit"
            elif regex.match('\W+$', tokens[i]):
                type = "punctuation"
            elif tokens[i] in sw:
                type = "stopword"
            
            result.append({
                'token': tokens[i],
                'position': position,
                'type': type
            })
            
            position = position + 1
        
        return result
    
    def __get_stopwords(self, language = 'english'):
        if language not in self.STOPWORDS.keys():
            if language in stopwords.fileids():
                self.STOPWORDS[language] = set(stopwords.words(language))
            else:
                self.STOPWORDS[language] = set([])
        
        return self.STOPWORDS[language]
    
    def __guess_language(self, tokens):
        try:
            from langdetect import detect
            from pycountry import languages
        except ImportError:
            print("The langdetect module is required for automated language detection; install with pip install langdetect")
            print("Reverting to english")
            return 'english'
        
        # Do language detection using langdetect
        # and map to full language name using pycountry
        words = [w for w in tokens if regex.match(r'#?[^\W\d]{2,}$', w) and not self.RESERVED_WORDS_PATTERN.match(w) and not self.URL_PATTERN.match(w)]
        
        if (len(words) > 0):
            try:
                language_short = detect(' '.join(words))
                return languages.lookup(language_short.split('-')[0]).name.lower()
            except langdetect.lang_detect_exception.LangDetectException:
                print('Language detection failed on string: "' + ' '.join(words) + '", defaulting to English')
                return 'english'
        else:
            return 'none'

# Now do a number of checks

# Retweets
# rttokens = twc.get_tokens_by_type(['reserved_word', 'retweeted_user'])
# for i in range(len(tweets)):
#     if regex.findall('^RT|^rt', tweets[i]):
#         if (len(rttokens[i]) != 2):
#             print(tweets[i])
# 
# # Mentions
# rttokens = twc.get_tokens_by_type(['mention', 'retweeted_user'])
# for i in range(len(tweets)):
#     if regex.findall('@', tweets[i]):
#         if (len(rttokens[i]) < 1):
#             print(tweets[i])
# 
# # Hashtags
# tokens = twc.get_tokens_by_type(['hashtag'])
# for i in range(len(tweets)):
#     if regex.findall('#', tweets[i]):
#         if (len(tokens[i]) < 1):
#             print(str(i) + str(tokens[i]) + ' ' + tweets[i])
# 
# # Numbers
# tokens = twc.get_tokens_by_type(['number', 'numeric_expression', 'mixed'])
# for i in range(len(tweets)):
#     if regex.findall(r'\s\d+', tweets[i]):
#         if (len(tokens[i]) < 1):
#             print(str(i) + str(tokens[i]) + ' ' + tweets[i])
# 
# # Mixed
# tokens = twc.get_tokens_by_type(['mixed'])
# for i in range(len(tweets)):
#     if regex.findall(r'\s\d+', tweets[i]):
#         if (len(tokens[i]) >= 1):
#             print(str(i) + str(tokens[i]) + ' ' + tweets[i])
# 
# tokens = twc.get_tokens_by_type(['numeric_expression'])
# for i in range(len(tweets)):
#     if regex.findall(r'\s\d+', tweets[i]):
#         if (len(tokens[i]) >= 1):
#             print(str(i) + str(tokens[i]) + ' ' + tweets[i])
