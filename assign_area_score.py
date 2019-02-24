from classify_tweets import *
import json

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

curlat = 0.0
curlong = 0.0
area_scores = {}
for val in tweets_classified:
    #area_counts[val[2], val[3]] = [0, 0, 0]
    if not (val[2] == curlat and val[3] == curlong):
        curlat = float(curlat) + 0.025
        curlong = float(curlong) + 0.025
        area_scores[(curlat, curlong)] = sigmoid((poscount - negcount) / (poscount + negcount + 0.001)/(1+ntrcount*0.2))
        curlat = val[2]
        curlong = val[3]
        poscount = 0
        negcount = 0
        ntrcount = 0
    if val[1] == 0:
        poscount += 1
    elif val[1] == 1:
        negcount += 1
    elif val[1] == 2:
        ntrcount += 1
del area_scores[(0.025, 0.025)]

print(area_scores)

prejson = []
for area, score in area_scores.items():
    prejson.append(dict([("score", score), ("g", area[1]), ("t", area[0])]))

json_data = json.dumps(prejson)


with open("public_html/data.js", 'w') as file:
    file.write("var data =")
    file.write(json_data)
