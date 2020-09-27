"""
Classify data.
"""
import re
import numpy as np
from collections import defaultdict
from scipy.sparse import lil_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import product
import json

use_descr_opts = [True, False]
lowercase_opts = [True, False]
keep_punctuation_opts = [True, False]
descr_prefix_opts = ['d=', '']
url_opts = [True, False]
mention_opts = [True, False]
stop_words_opts = [True, False]
argnames = ['use_descr', 'lower', 'punct', 'prefix', 'url', 'mention', 'stop_words_opts']


def tokenize(string, lowercase, keep_punctuation, prefix,
             collapse_urls, collapse_mentions, stop_words):
    """ Split a tweet into tokens."""
    if not string:
        return []
    if lowercase:
        string = string.lower()
    tokens = []
    if collapse_urls:
        string = re.sub('http\S+', 'THIS_IS_A_URL', string)
    if collapse_mentions:
        string = re.sub('@\S+', 'THIS_IS_A_MENTION', string)
    if keep_punctuation:
        tokens = string.split()
    else:
        tokens = re.sub('\W+', ' ', string).split()
    if stop_words:
        remove_list = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as",
                       "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
                       "could",
                       "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had",
                       "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
                       "herself",
                       "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
                       "is",
                       "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on",
                       "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
                       "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's",
                       "the",
                       "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd",
                       "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
                       "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when",
                       "when's",
                       "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would",
                       "you",
                       "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
        final_list = []
        for word in tokens:
            if word.lower() not in remove_list:
                final_list.append(word)
        tokens = final_list
    if prefix:
        tokens = ['%s%s' % (prefix, t) for t in tokens]
    return tokens


def tweet2tokens(tweet, use_descr=True, lowercase=True,
                 keep_punctuation=True, descr_prefix='d=',
                 collapse_urls=True, collapse_mentions=True, stop_words=True):
    """ Convert a tweet into a list of tokens, from the tweet text and optionally the
    user description. """
    tokens = tokenize(tweet['text'], lowercase, keep_punctuation, None,
                      collapse_urls, collapse_mentions, stop_words)
    if use_descr:
        tokens.extend(tokenize(tweet['user']['description'], lowercase,
                               keep_punctuation, descr_prefix,
                               collapse_urls, collapse_mentions, stop_words))
    return tokens


def make_vocabulary(tokens_list):
    vocabulary = defaultdict(lambda: len(vocabulary))  # If term not present, assign next int.
    for tokens in tokens_list:
        for token in tokens:
            vocabulary[token]  # looking up a key; defaultdict takes care of assigning it a value.
    return vocabulary


def make_feature_matrix(length, tokens_list, vocabulary):
    feature_matrix = lil_matrix((length, len(vocabulary)))
    for i, tokens in enumerate(tokens_list):
        for token in tokens:
            j = vocabulary[token]
            feature_matrix[i, j] += 1
    return feature_matrix.tocsr()  # convert to CSR for more efficient random access.


def do_cross_val(feature_matrix, labels, nfolds):
    """ Compute average cross-validation acccuracy."""
    cv = KFold(n_splits=nfolds, random_state=42, shuffle=True)
    accuracies = []
    for train_idx, test_idx in cv.split(feature_matrix):
        clf = LogisticRegression()
        clf.fit(feature_matrix[train_idx], labels[train_idx])
        predicted = clf.predict(feature_matrix[test_idx])
        acc = accuracy_score(labels[test_idx], predicted)
        accuracies.append(acc)
    avg = np.mean(accuracies)
    return avg


def run_all(tweets, labels, use_descr=True, lowercase=True,
            keep_punctuation=True, descr_prefix=None,
            collapse_urls=True, collapse_mentions=True, stop_words=True):
    tokens_list = [tweet2tokens(t, use_descr, lowercase,
                                keep_punctuation, descr_prefix,
                                collapse_urls, collapse_mentions, stop_words)
                   for t in tweets]
    vocabulary = make_vocabulary(tokens_list)
    feature_matrix = make_feature_matrix(len(tweets), tokens_list, vocabulary)
    acc = do_cross_val(feature_matrix, labels, 5)
    return acc


def find_best_option(tweets, labels):
    option_iter = product(use_descr_opts, lowercase_opts,
                          keep_punctuation_opts,
                          descr_prefix_opts, url_opts,
                          mention_opts, stop_words_opts)
    results = []
    for options in option_iter:
        option_str = '\t'.join('%s=%s' % (name, opt) for name, opt
                               in zip(argnames, options))
        print(option_str)
        acc = run_all(tweets, labels, *options)
        results.append((acc, options))
        print(acc)
    return sorted(results, reverse=True)[0]


def load_data(path):
    tweets = json.load(open(path + 'raw_tweets.txt', 'rb'))
    labels_array = json.load(open(path + 'labels.txt', 'rb'))
    labels = np.array(labels_array)
    return tweets, labels


def save_data(report, path):
    json.dump(report, open(path, 'w'))


def main():
    print("begin to load data")
    tweets, labels = load_data('./shared/classifer_data/')
    print("begin to find the best feature option")
    result = find_best_option(tweets, labels)
    score = result[0]
    option = result[1]
    option_str = '\t'.join('%s=%s' % (name, opt) for name, opt
                           in zip(argnames, option))
    print("best feature option: ", option_str)
    print("best cross-fold-validation score: ", score)
    print("use this best option to tokenize tweets and train the classifer")
    tokens_list = [tweet2tokens(t, *option)
                   for t in tweets]
    vocabulary = make_vocabulary(tokens_list)
    feature_matrix = make_feature_matrix(len(tweets), tokens_list, vocabulary)
    model = LogisticRegression()
    model.fit(feature_matrix[:4950], labels[:4950])
    print("classifer training done, and begin to make prediction")
    report = []
    for index in range(4950, 5000):
        print("tweets text : ", tweets[index]["text"])
        print("model predicted : ", model.predict(feature_matrix[index]))
        print("true label is : ", labels[index])
        report.append(
            {"tweet_text": tweets[index]["text"], "model_predicted":
                str(model.predict(feature_matrix[index])[0]),
             "true_label": str(labels[index])})
    print('-------------------------------')
    idx2word = dict((v, k) for k, v in vocabulary.items())
    # Get the learned coefficients for the Positive class.
    coef = model.coef_[0]
    # Sort them in descending order.
    top_coef_ind = np.argsort(coef)[::-1][:20]
    # Get the names of those features.
    top_coef_terms = [idx2word[i] for i in top_coef_ind]
    # Get the weights of those features
    top_coef = coef[top_coef_ind]
    # Print the top 10.
    print('top weighted terms for female class:')
    print('\n'.join(str(x) for x in zip(top_coef_terms, top_coef)))

    # repeat for males
    top_coef_ind = np.argsort(coef)[:20]
    top_coef_terms = [idx2word[i] for i in top_coef_ind]
    top_coef = coef[top_coef_ind]
    print('\ntop weighted terms for male class:')
    print('\n'.join(str(x) for x in zip(top_coef_terms, top_coef)))
    print("begin to save classifer result")
    save_data(report, './shared/summary_data/classify_result.txt')
    print("classifer result saved")


if __name__ == "__main__":
    main()
