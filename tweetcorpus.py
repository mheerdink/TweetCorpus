import sys, csv, regex, itertools
from nltk.corpus import stopwords
from twokenize import twokenize
# use the ark-twokenize-py fork by Sentimentron, as it gives better results than the casual tokenizer from nltk
from stemming.porter2 import stem # Porter2 works slightly better
from joblib import Parallel, delayed # for parallel processing
from types import GeneratorType # for GeneratorType

class TweetCorpus:
    REPEAT_PATTERN = regex.compile(r'((\w)\2{3,})')

    """
    Class designed to simplify working with a corpus of raw tweets
    """    
    def __init__(self, corpus, vocabulary = None, preserve_case = False, reduce_len = True, cache_corpus = False, cache_tokens = False, cache_classification = False, language = 'auto', parallel = True, parallel_backend = "loky", njobs = -1):
        assert(isinstance(corpus, GeneratorType) or (isinstance(corpus, list) and (all([isinstance(t, tuple) for t in corpus]) or all([isinstance(t, str) for t in corpus])))) # corpus should either be a generator or a list of strings/tuples

        self.language = language         # Language of tweets; either a string (appies to all Tweets) or a list of strings of the same length as the corpus, with each Tweet's language
        self.vocabulary = vocabulary     # Vocabulary to use for distinguishing between truncated words and valid words

        self.parallel = parallel

        if parallel:
            self.njobs = njobs
            self.parallel_backend = parallel_backend

        self.corpus = corpus

        self.preserve_case = preserve_case
        self.reduce_len = reduce_len

    def sents(self):
        def counter():
            n = 1
            while True:
                yield n
                n = n + 1

        if isinstance(self.corpus, GeneratorType) | isinstance(self.corpus, itertools._tee):
            self.corpus, test = itertools.tee(self.corpus)
            first = test.__next__() # get first element
            self.corpus, copy = itertools.tee(self.corpus)

            if isinstance(first, tuple):  # we have tweetnos
                return copy
            elif isinstance(first, str): # we don't have tweetnos, so generate these on the fly
                return zip(counter(), copy)
        else:
            if isinstance(self.corpus[0], tuple):  # we have tweetnos
                return self.corpus
            elif isinstance(self.corpus[0], str):
                return zip(counter(), self.corpus)

        raise InputError('self.corpus', 'Neither a generator, nor a list of strings')
        return []

    @staticmethod
    def _tokenize1(tno_text, preserve_case, reduce_len):
        """
        Pre-process and tokenize a single tweet
        """
        (tweetno, text) = tno_text

        if not preserve_case:
            text = text.lower()

        if reduce_len:
            text = TweetCorpus.REPEAT_PATTERN.sub(lambda m: m.groups()[1]*3, text)

        return (tweetno, twokenize.tokenizeRawTweetText(text))

    def tokenize(self, parallel = None):
        if parallel is None:
            parallel = self.parallel

        if parallel:
            return Parallel(n_jobs=self.njobs)(delayed(TweetCorpus._tokenize1)(tno_text, self.preserve_case, self.reduce_len) for tno_text in self.sents())
        else:
            return (TweetCorpus._tokenize1(tno_text, self.preserve_case, self.reduce_len) for tno_text in self.sents())

    @staticmethod
    def _classify1(tno_tokens, classifier):
        return (tno_tokens[0], classifier.classify(tno_tokens[1]))

    @staticmethod
    def _classify2(tno_text, classifier, preserve_case, reduce_len):
        tweetno, tokens = TweetCorpus._tokenize1(tno_text, preserve_case=preserve_case, reduce_len=reduce_len)
        return (tweetno, classifier.classify(tokens))

    def classify(self, parallel = None):
        if parallel is None:
            parallel = self.parallel

        if parallel:
            return Parallel(n_jobs=self.njobs)(delayed(TweetCorpus._classify2)(tno_text, classifier = self.get_classifier(), preserve_case = self.preserve_case, reduce_len = self.reduce_len) for tno_text in self.sents())
        else:
            return (TweetCorpus._classify2(tno_text, classifier = self.get_classifier(), preserve_case = self.preserve_case, reduce_len = self.reduce_len) for tno_text in self.sents())

    @staticmethod
    def _get_tokens_by_type1(tno_text, types, extract, preserve_case, reduce_len, classifier): # function to run on a single array of tokens
        assert(isinstance(types, list))

        tno_tokens = TweetCorpus._tokenize1(tno_text, preserve_case, reduce_len)
        tno_ctokens = TweetCorpus._classify1(tno_tokens, classifier)

        if extract is None:
            return (tno_ctokens[0], [t for t in tno_ctokens[1] if t['type'] in types])
        else:
            return (tno_ctokens[0], [t[extract] for t in tno_ctokens[1] if t['type'] in types])

    def get_tokens_by_type(self, types = ['contentword'], extract = None, parallel = None):
        if isinstance(types, str):
            types = [types]

        if parallel is None:
            parallel = self.parallel

        classifier = self.get_classifier()

        if parallel:
            return Parallel(n_jobs=self.njobs)(delayed(TweetCorpus._get_tokens_by_type1)(tno_text, types, extract, self.preserve_case, self.reduce_len, classifier) for tno_text in self.sents())
        else:
            return (TweetCorpus._get_tokens_by_type1(tno_text, types, extract, self.preserve_case, self.reduce_len, classifier) for tno_text in self.sents())

    def get_classifier(self):
        if not hasattr(self, 'classifier'):
            self.classifier = TweetTokenClassifier(vocabulary = self.vocabulary)

        return self.classifier

    def clean(self, keep_types = ['contentword', 'stopword']):
        return ((tweetno, ' '.join(w)) for tweetno,words in self.get_tokens_by_type(keep_types, extract='token'))

    def build_vocabulary(self, types = ['contentword'], extract='token', parallel = None):
        return set([t for _,tokens in self.get_tokens_by_type(types, extract = extract, parallel = parallel) for t in tokens])

    def tokens_csv(self, filename, parallel = None):
        fieldnames = ['tweetno', 'position', 'token', 'stem', 'type']

        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
            writer.writeheader()
            for tweetno, ctokens in self.classify(parallel = parallel):
                for ct in ctokens:
                    ct['tweetno'] = tweetno
                    writer.writerow(ct)

    # TODO: parallellize this function
    def sentiments(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError:
            print("Install vaderSentiment to do sentiment analysis; pip install vaderSentiment")
            return False
        
        analyzer = SentimentIntensityAnalyzer()
        
        def __sentiment(tweet_clean):
            vs = analyzer.polarity_scores(tweet_clean)
            vs['tweet_clean'] = tweet_clean
            return vs
        
        return ((tweetno, __sentiment(tweet_clean)) for tweetno, tweet_clean in self.clean())
    
    def sentiments_csv(self, filename):
        def __append_tweetno(tweetno, scores):
            scores['tweetno'] = tweetno
            return scores
        
        fieldnames = ['tweetno', 'tweet_clean', 'pos', 'neg', 'neu', 'compound']
        
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
            writer.writeheader()
            for tweetno, scores in self.sentiments():
                writer.writerow(__append_tweetno(tweetno, scores))
    
    def retweets(self, parallel = None):
        # This is not a generator expression because usually, the entire set is needed
        if not hasattr(self, 'retweet_indices'):
            self.retweet_indices = [tweetno for tweetno,tokens in self.classify(parallel = parallel) if any([t['position'] < 1 and t['token'] in ['rt','RT'] for t in tokens])]
        
        return self.retweet_indices

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
    MENTION_PATTERN = regex.compile(r"@[a-zA-Z0-9_]*[a-zA-Z][a-zA-Z0-9_]*")
    HASHTAG_PATTERN = regex.compile(r'[#＃]*([#＃][\w_]*[^\W\d][\w_]*)')
    EMAIL_PATTERN = regex.compile(twokenize.Email)
    NUMBER_PATTERN = r'\-?[,.]?\d+(?:[,.]\d+)*'
    CURRENCY_PATTERN = r"(?:(?:[\p{Sc}]|EUR|USD|GBP)\s*)?" + NUMBER_PATTERN + "(?:\s*(?:%|[\p{Sc}]|EUR|USD|GBP))?"
    RANK_PATTERN = r'(?:' + NUMBER_PATTERN + '(?:st|ST|nd|ND|rd|RD|th|TH)|#' + NUMBER_PATTERN + ')'
    NUMBERS_PATTERN = regex.compile(NUMBER_PATTERN + '|' + CURRENCY_PATTERN + '|' + RANK_PATTERN)
    NUMERIC_EXPRESSION_PATTERN = regex.compile(NUMBER_PATTERN + '(?:[+-/*=():]+' + NUMBER_PATTERN + ')+')
    MIXED_PATTERN = regex.compile(r'[^\W\d]+(?:[+-/*=()]+|[^\W\d]+)*' + NUMBER_PATTERN + '[\w\d+-/*=(),.]*|' + NUMBER_PATTERN + '(?:[+-/*=()]+|' + NUMBER_PATTERN + ')*[^\W\d]+[\w\d+-/*=(),.]*')
    EMOTICON_PATTERN = regex.compile(twokenize.emoticon)
    PRE_STEM_PATTERNS = {
        'all'     : regex.compile(r"(?:'s|[\p{P}]+)?$"), # drop trailing 's and any trailing punctuation
        'hashtag' : regex.compile(r'^[#＃]+')
    }
    
    if (sys.maxunicode > 65535):
        # UCS-4
        EMOJIS_PATTERN = regex.compile(u'([\U00002600-\U000027BF]\U0000FE0F?)|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])|(\U0000FE0F)')
    else:
        # UCS-2
        EMOJIS_PATTERN = regex.compile(u'([\u2600-\u27BF]\uFE0F?)|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])|(\uFE0F)')
    
    def __init__(self, vocabulary = None):
        if vocabulary is not None:
            self.vocabulary = set(vocabulary)

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
                type = "url"
            elif self.EMOJIS_PATTERN.match(tokens[i]):
                type = "emoji"
            elif self.EMOTICON_PATTERN.match(tokens[i]):
                type = "emoticon"
            elif regex.match(r'[\p{Sc}%]$', tokens[i]):
                type = "unit"
            elif regex.match('\W+$', tokens[i]):
                type = "punctuation"
            elif tokens[i] in sw:
                type = "stopword"
            
            tokenstem = 'NA'
            
            if type in ['contentword', 'stopword', 'hashtag']:
                # do stemming
                tokenstem = self.PRE_STEM_PATTERNS['all'].sub('', tokens[i])
                tokenstem = self.PRE_STEM_PATTERNS['hashtag'].sub('', tokenstem)
                
                tokenstem = stem(tokenstem)
            
            result.append({
                'token': tokens[i],
                'stem': tokenstem,
                'position': position,
                'type': type
            })
            
            position = position + 1
        
        return self.__flag_truncations(result)
    
    def __flag_truncations(self, tokens):
        for j in range(len(tokens)):
            if (tokens[j]['position'] >= 2 and
                tokens[j]['type'] == 'ellipsis' and
                tokens[j - 1]['type'] in ['contentword', 'stopword', 'mention', 'hashtag', 'url'] and
                not any([t['type'] in ['contentword', 'stopword'] for t in tokens[j:(len(tokens) - 1)]])):
                if hasattr(self, 'vocabulary'):
                    if (len(tokens[j - 1]['token']) < 2 or tokens[j - 1]['token'] not in self.vocabulary):
                        tokens[j - 1]['type'] = tokens[j - 1]['type'] + '_truncated'
                else:
                    tokens[j - 1]['type'] = tokens[j - 1]['type'] + '_truncated'
        
        return tokens
    
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
