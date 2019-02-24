import csv
from twitter_specials import *
import string

words_dict = {}
totalcount = 0
poscount = 0
negcount = 0
ntrcount = 0
irrcount = 0
with open("data/labeled_corpus.tsv", encoding="utf-8") as csvfile:
    readCSV = csv.reader(csvfile, delimiter='\t')
    for row in readCSV:
        line_arr = list(row)

        tweet = line_arr[0]
        sentiment = line_arr[1]

        tweet = clean_tweet(tweet, emo_repl_order, emo_repl, re_repl)

        table = str.maketrans('', '', string.punctuation)
        tweet = tweet.translate(table)

        words = tweet.split()

        for w in words:
            if w not in words_dict:
                words_dict[w] = [0, 0, 0, 0]
            totalcount += 1
            if sentiment == "positive":
                words_dict[w][0] += 1
                poscount += 1
            elif sentiment == "negative":
                words_dict[w][1] += 1
                negcount += 1
            elif sentiment == "neutral":
                words_dict[w][2] += 1
                ntrcount += 1
            elif sentiment == "irrelevant":
                words_dict[w][3] += 1
                irrcount += 1

ppositive = poscount/totalcount
pnegative = negcount/totalcount
pneutral = ntrcount/totalcount
pirrelevant = irrcount/totalcount

word_probabilities = {}
for w,attributes in words_dict.items():
    word_probabilities[w] = [0.0, 0.0, 0.0, 0.0]
    word_probabilities[w][0] = attributes[0]/poscount
    word_probabilities[w][1] = attributes[1]/negcount
    word_probabilities[w][2] = attributes[2]/ntrcount
    word_probabilities[w][3] = attributes[3]/irrcount
