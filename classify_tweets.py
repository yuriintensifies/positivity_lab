import csv
from build_classifier import *
import string
import math
import numpy as np

tweets_classified = []
with open("data/geo_twits_squares.tsv", encoding="utf-8") as csvfile:
    readCSV = csv.reader((x.replace('\0', '') for x in csvfile), delimiter='\t')
    for row in readCSV:
        line_arr = list(row)
        tweet = line_arr[2]
        long = line_arr[1]
        lat = line_arr[0]
        tweet = clean_tweet(tweet, emo_repl_order, emo_repl, re_repl)

        table = str.maketrans('', '', string.punctuation)
        tweet = tweet.translate(table)

        words = tweet.split()
        tweet_probabilities = [math.log(ppositive), math.log(pnegative), math.log(pneutral), math.log(pirrelevant)]
        for word in words:
            if word not in word_probabilities:
                continue
            for n, prob in enumerate(tweet_probabilities):
                try:
                    tweet_probabilities[n] += math.log(word_probabilities[word][n])
                except:
                    continue
        classif = np.argmax(tweet_probabilities)
        tweets_classified.append([tweet_probabilities[classif], classif, lat, long])

with open('data/locations_classified.tsv', 'w') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for v in tweets_classified:
        if v[1] == 0:
            tsv_writer.writerow([v[2], v[3], 'positive'])
        elif v[1] == 1:
            tsv_writer.writerow([v[2], v[3], 'negative'])
        elif v[1] == 2:
            tsv_writer.writerow([v[2], v[3], 'neutral'])
        elif v[1] == 3:
            tsv_writer.writerow([v[2], v[3], 'irrelevant'])
