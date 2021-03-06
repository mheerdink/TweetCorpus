# Tokenize some sample tweets
# 
# By executing this script after each change to the tokenizer / tweetcorpus
# class and diff'ing the output files it is possible to check whether a change
# is really an improvement.

import csv
from os.path import expanduser
import tweetcorpus
from datetime import datetime

testfile = expanduser('~/Twitter/sampletweets.txt')
output = expanduser('~/Twitter/' + datetime.today().strftime('%Y%m%d%H%M%S') + '-tokens.csv')

def tweetstream(filename):
    with open(filename, encoding = 'utf-8') as f:
        for line in f:
            yield line

if __name__ == '__main__':
    tcp = tweetcorpus.TweetCorpus(corpus = tweetstream(testfile))
    vocab = tcp.build_vocabulary()
    tcp = tweetcorpus.TweetCorpus(corpus = tweetstream(testfile), vocabulary = vocab)
    tcp.tokens_csv(filename = output, fieldnames = ['tweetno', 'token', 'stem', 'type'])
