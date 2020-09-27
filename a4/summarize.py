# coding: utf-8
"""
Summarize data.
"""
from collections import Counter, defaultdict, deque
import copy
from itertools import combinations
import math
import networkx as nx
import urllib.request
import json
import networkx as nx
from collections import Counter, defaultdict, deque
import sys
import glob
import os
import time
import matplotlib.pyplot as plt
import scipy


class SummaryRecorder(object):
    def __init__(self, file_name="./summary.txt"):
        self.terminal = sys.stdout
        self.log = open(file_name, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def load_data():
    users = [json.load(open(f)) for f in glob.glob(os.path.join('./shared/cluster_data/', '*.txt'))]
    tweets = json.load(open('./shared/classifer_data/raw_tweets.txt'))
    labels = json.load(open('./shared/classifer_data/labels.txt'))
    classifer_result = json.load(open('./shared/summary_data/classify_result.txt'))
    cluser_result = []
    with open('./shared/summary_data/cluster_result.txt') as reader:
        for line in reader.readlines():
            cluser_result.append(line.strip())
    reader.close()
    return users, tweets, labels, classifer_result, cluser_result


def main():
    sys.stdout = SummaryRecorder()
    users, tweets, labels, classifer_result, cluser_result = load_data()
    print("Number of users collected:")
    friends = 0
    for user in users:
        friends += len(user['friends'])
    temp = str(len(users)) + " users and their " + str(friends) + " friends"
    print(temp)
    print("Number of messages collected:")
    print(len(tweets), "tweets")
    print("Number of communities discovered:")
    print(cluser_result[0])
    print("Average number of users per community:")
    print(cluser_result[1])
    print("Number of instances per class found:")
    counter = Counter(dic["model_predicted"] for dic in classifer_result)
    print(counter)
    print("One example from each class:")
    male_example = None
    female_example = None
    for dic in classifer_result:
        if dic["model_predicted"] == "0":
            if male_example is None:
                male_example = dic
        elif dic["model_predicted"] == "1":
            if female_example is None:
                female_example = dic
        if male_example is not None and female_example is not None:
            break
    print("male_example", male_example)
    print("female_example", female_example)


if __name__ == "__main__":
    main()
