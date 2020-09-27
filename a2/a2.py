# coding: utf-8

"""
CS579: Assignment 2

In this assignment, you will build a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.

You'll write code to preprocess the data in different ways (creating different
features), then compare the cross-validation accuracy of each approach. Then,
you'll compute accuracy on a test set and do some analysis of the errors.

The main method takes about 40 seconds for me to run on my laptop. Places to
check for inefficiency include the vectorize function and the
eval_all_combinations function.

Complete the 14 methods below, indicated by TODO.

As usual, completing one method at a time, and debugging with doctests, should
help.
"""

# No imports allowed besides these.
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request


def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/8oehplrobcgi9cq/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.

    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.

    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], dtype='<U5')
    """
    if not keep_internal_punct:
        return np.array(re.sub(r'\W', ' ', doc.lower()).split())
    else:
        # return np.array(re.split(r'[\s]', doc.lower()))
        return np.array(re.findall(r'\w\S*\w|\w+', doc.lower()))
    # remove_list = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as",
    #                "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could",
    #                "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had",
    #                "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    #                "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
    #                "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on",
    #                "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    #                "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the",
    #                "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd",
    #                "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    #                "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's",
    #                "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you",
    #                "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
    # if not keep_internal_punct:
    #     not_filtered_list = re.sub(r'\W', ' ', doc.lower()).split()
    # else:
    #     # return np.array(re.split(r'[\s]', doc.lower()))
    #     not_filtered_list = re.findall(r'\w\S*\w|\w+', doc.lower())
    # final_list = []
    # for word in not_filtered_list:
    #     if word.lower() not in remove_list:
    #         final_list.append(word)
    # return np.array(final_list)


def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    for token in tokens:
        feats['token=' + token] += 1


def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    lastIndex = len(tokens) - k
    for tokenIndex in range(lastIndex + 1):
        tokenList = combinations(tokens[tokenIndex: tokenIndex + k], 2)
        for token_pair in tokenList:
            feats['token_pair=' + token_pair[0] + '__' + token_pair[1]] += 1


neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])


def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    pc = 0
    nc = 0
    for token in tokens:
        token = token.lower()
        if token in pos_words:
            pc += 1
        if token in neg_words:
            nc += 1
    feats['pos_words'] += pc
    feats['neg_words'] += nc


def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.

    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    feats = defaultdict(lambda: 0)
    for fn in feature_fns:
        fn(tokens, feats)
    return list(sorted(feats.items()))


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),

    When vocab is None, we build a new vocabulary from the given data.
    when vocab is not None, we do not build a new vocab, and we do not
    add any new terms to the vocabulary. This setting is to be used
    at test time.

    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    # consider to update the vocab
    # if not vocab:
    #     vocab = defaultdict(lambda: 0)
    # feature_list = [featurize(tokens, feature_fns) for tokens in tokens_list]
    # frequent_feature_dict = defaultdict(lambda: 0)
    #
    # for features in feature_list:
    #     for feature in features:
    #         if feature[1] != 0:
    #             frequent_feature_dict[feature[0]] += 1
    # for key, value in frequent_feature_dict.items():
    #     if value >= min_freq:
    #         vocab[key] = 0
    # vocab_temp = defaultdict(lambda: 0)
    # index = 0
    # for key, value in sorted(vocab.items(), key=lambda x: x[0]):
    #     vocab_temp[key] = index
    #     index += 1
    # vocab = vocab_temp
    feature_list = [featurize(tokens, feature_fns) for tokens in tokens_list]
    if not vocab:
        frequent_feature_dict = defaultdict(lambda: 0)
        for features in feature_list:
            for feature in features:
                if feature[1] != 0:
                    frequent_feature_dict[feature[0]] += 1
        vocab = dict()
        index = 0
        for key, value in sorted(frequent_feature_dict.items(), key=lambda x: x[0]):
            if value >= min_freq:
                vocab[key] = index
                index += 1
    row_indices = []
    column_indices = []
    date = []
    for index in range(len(feature_list)):
        for feature in feature_list[index]:
            if feature[0] in vocab:
                row_indices.append(index)
                column_indices.append(vocab[feature[0]])
                date.append(feature[1])
    sparse_matrix = csr_matrix((date, (row_indices, column_indices)), shape=(len(feature_list), len(vocab)))
    return sparse_matrix, vocab


def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth == predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    cv = KFold(k)
    accuracies = []
    for train_ind, test_ind in cv.split(X):
        clf.fit(X[train_ind], labels[train_ind])
        predictions = clf.predict(X[test_ind])
        accuracies.append(accuracy_score(labels[test_ind], predictions))
    return np.mean(accuracies)


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.

      This list should be SORTED in descending order of accuracy.

      This function will take a bit longer to run (~20s for me).
    """
    possible_feature_fn_combinations = []
    for i in range(1, len(feature_fns) + 1):
        for one_possible_combination in combinations(feature_fns, i):
            possible_list = []
            for feature in one_possible_combination:
                possible_list.append(feature)
            possible_feature_fn_combinations.append(possible_list)
    result_list = []
    for punct_option in punct_vals:
        tokens = [tokenize(doc, punct_option) for doc in docs]
        for min_freq in min_freqs:
            for one_possible_combination in possible_feature_fn_combinations:
                data_set, vocab = vectorize(tokens, one_possible_combination, min_freq)
                one_test_accuracy = cross_validation_accuracy(LogisticRegression(), data_set, labels, 5)
                result_list.append({'punct': punct_option, 'features': one_possible_combination, 'min_freq': min_freq,
                                    'accuracy': one_test_accuracy})
    return sorted(result_list, key=lambda x: -x['accuracy'])


def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    X_scope = range(len(results))
    Y_scope = [result['accuracy'] for result in sorted(results, key=lambda x: x['accuracy'])]
    plt.plot(X_scope, Y_scope)
    plt.xlabel('setting')
    plt.ylabel('accuracy')
    plt.savefig('accuracies.png')


def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.

    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    per_setting_total = defaultdict(lambda: .0)
    per_setting_count = defaultdict(lambda: 0)
    per_setting_average = defaultdict(lambda: .0)
    for result in results:
        per_setting_total['punct=' + str(result['punct'])] += result['accuracy']
        per_setting_count['punct=' + str(result['punct'])] += 1
        per_setting_total['min_freq=' + str(result['min_freq'])] += result['accuracy']
        per_setting_count['min_freq=' + str(result['min_freq'])] += 1
        feature_name = 'features='
        for fn in result['features']:
            feature_name += ' ' + fn.__name__
        per_setting_total[feature_name] += result['accuracy']
        per_setting_count[feature_name] += 1
    for key, value in per_setting_total.items():
        per_setting_average[key] = value / per_setting_count[key]
    return [(value, key) for key, value in sorted(per_setting_average.items(), key=lambda x: -x[1])]


def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    classifier = LogisticRegression()
    tokens = [tokenize(doc, best_result['punct']) for doc in docs]
    data_set, vocab = vectorize(tokens, best_result['features'], best_result['min_freq'])
    classifier.fit(data_set, labels)
    return classifier, vocab


def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.

    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    if label:
        top_feature_index = np.argsort(clf.coef_[0])[::-1][:n]
        top_feature_coefficient = clf.coef_[0][top_feature_index]
    else:
        top_feature_index = np.argsort(clf.coef_[0])[:n]
        top_feature_coefficient = -1 * clf.coef_[0][top_feature_index]
    top_feature_name = np.array(sorted(vocab, key=lambda x: x[1]))[top_feature_index]
    return [result for result in zip(top_feature_name, top_feature_coefficient)]


def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Note: use read_data function defined above to read the
    test data.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    test_docs, test_labels = read_data(os.path.join('data', 'test'))
    tokens = [tokenize(doc, best_result['punct']) for doc in test_docs]
    X_test, vocab = vectorize(tokens, best_result['features'], best_result['min_freq'], vocab)
    return test_docs, test_labels, X_test


def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.

    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.

    Returns:
      Nothing; see Log.txt for example printed output.
    """
    predicted_labels = clf.predict(X_test)
    predicted_probabilities = clf.predict_proba(X_test)
    wrong_indices = np.where(predicted_labels != test_labels)[0]
    wrong_probabilities = predicted_probabilities[wrong_indices]
    wrong_max_probabilities = []
    for p1, p2 in wrong_probabilities:
        wrong_max_probabilities.append(max(p1, p2))
    wrong_max_probabilities = np.array(wrong_max_probabilities)
    max_wrong_max_probabilities_indices = np.argsort(wrong_max_probabilities)[::-1][:n]
    for index in range(n):
        doc_index = wrong_indices[max_wrong_max_probabilities_indices[index]]
        print("truth=" + str(test_labels[doc_index]) + " predicted=" + str(predicted_labels[doc_index]) + " proba=" +
              str(wrong_max_probabilities[max_wrong_max_probabilities_indices[index]]))
        print(test_docs[doc_index])
        print()


def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2, 5, 10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s, v) for v, s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t, v) for t, v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t, v) for t, v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()

# print(tokenize("Hi there! Isn't this fun?", True))

# feats = defaultdict(lambda: 0)
# token_features(['hi', 'there', 'hi'], feats)
# print(sorted(feats.items()))

# feats = defaultdict(lambda: 0)
# token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
# print(sorted(feats.items()))

# feats = defaultdict(lambda: 0)
# lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
# print(sorted(feats.items()))

# feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
# print(feats)

# docs = ["Isn't this movie great?", "Horrible, horrible movie"]
# tokens_list = [tokenize(d) for d in docs]
# feature_fns = [token_features]
# X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
# print(X.toarray(), type(X))
# print(sorted(vocab.items(), key=lambda x: x[1]))

# fit_best_classifier(['23213', '2312,3 1 3 1'], 1, {'punct': False})
