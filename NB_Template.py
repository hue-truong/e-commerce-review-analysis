import numpy as np
from numpy.lib.function_base import vectorize
import math
import pandas as pd
import string
import operator
from matplotlib import pyplot as plt
from pprint import pprint as print
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

class BagOfWords(object):
    """
    Class for implementing Bag of Words
     for Q1.1
    """
    def __init__(self, vocabulary_size):
        """
        Initialize the BagOfWords model
        """
        self.vocabulary_size = vocabulary_size

    def preprocess(self, text):
        """
        Preprocessing of one Review Text
            - convert to lowercase
            - remove punctuation
            - empty spaces
            - remove 1-letter words
            - split the sentence into words

        Return the split words
        """
        #lowerify text, remove punctuation, split words by spaces, and filter list by words greater than 1
        return [word for word in text.lower().translate(str.maketrans('', '', string.punctuation)).split() if len(word) > 1]

    def fit(self, X_train):
        """
        Building the vocabulary using X_train
        """
        #Set vocabulary as a alphasorted list of most common words found in X_train. Size of list is bounded by vocabulary_size
        self.vocabulary = sorted([word for word, count in Counter([word for words in [self.preprocess(x) for x in X_train] for word in words]).most_common(self.vocabulary_size)])

    def transform(self, X):
        """
        Transform the texts into word count vectors (representation matrix)
            using the fitted vocabulary
        """
        #List of entries made up of text filtered by vocabulary
        filtered_entries = [dict([word for word in sorted(Counter(x).items()) if word[0] in self.vocabulary]) for x in [self.preprocess(x) for x in X]]

        #Maps word count to vocabulary if the word exists in the entry, for each entry in X
        return np.array([[x[v] if v in x.keys() else 0 for v in self.vocabulary] for x in filtered_entries])

class NaiveBayes(object):
    def __init__(self, beta=1, debug=0):
        """
        Initialize the Naive Bayes model
            w/ beta
        """
        self.beta = beta
        self.debug = debug

    def _calculate_beta_like_(self, target):
        """
        Calculate sum of betas for a given Y=target
        
        """
        return sum([self.beta for n in range(np.count_nonzero(self._set_[target].transpose()))])

    def _get_subset_(self, target):
        """
        Gets a subset of the dataset given target class

        """
        #Get the subset by filtering out tuples (x, y) that satisfy y=target
        return np.array([x for x, y in zip(self.X_train, self.y_train) if y == target])

    def _calculate_log_prior_(self, target):
        """
        Calculate the log prior given Y=target

        """
        #Prior = ({Total number of samples in target set} + {beta}) / ({Total number of samples} + {Sum of beta for each class})
        return math.log(self._set_[target].shape[0] / self.X_train.shape[0])

    def _calculate_log_likelihood_(self, feature, target):
        """
        Calculate the log likelihood of a feature given Y=target

        """
        #Likelihood = ({Total occurencies of a feature given the target class} + {beta}) / ({Total number of samples in target set} + {Sum of beta for each class})
        return math.log((self._set_[target].sum(axis=0)[feature] + (self.beta))/ (self._set_[target].shape[0] + self.beta_sum[target]))

    def _calculate_log_posterior_(self, x, target):
        """
        Calculate the posterior probability given Y=target

        """
        #Posterior = {Log prior} + {Sum of (log likelihoods * occurency) of existing features in x}
        return self.priors[target] + sum([self.likelihoods[target][feature]*x[feature] for feature in range(self.X_train.shape[1])])

    def fit(self, X_train, y_train):
        """
        Fit the model to X_train, y_train
            - build the conditional probabilities
            - and the prior probabilities
        """
        if self.debug:
            print(f'In NaiveBayes.fit: Fitting model...')

        self.X_train = X_train
        self.y_train = y_train
        self.classes = np.sort(np.unique(y_train))

        
        
        """
        Building prior and conditional probability dictionaries
            _set_: X_train partitioned by classes
            likelihoods: class: feature likelihoods for each class.
            priors: dict prior probabilities for each class
        """
        self._set_ = dict([(c, np.array(self._get_subset_(c))) for c in self.classes])
        #Build divisor beta sum for likelihood calculations
        self.beta_sum = dict([(c, self._calculate_beta_like_(c)) for c in self.classes])

        self.likelihoods = dict([(c, np.array([self._calculate_log_likelihood_(feature, c) for feature in range(self.X_train.shape[1])])) for c in self.classes])
        self.priors = dict([(c, self._calculate_log_prior_(c)) for c in self.classes])

        if self.debug:
            print(f'Fitted data with beta={self.beta}')
            print(f'Likelihood:')
            print([[math.pow(math.e, l) for l in self.likelihoods[c]] for c in self.classes])
            print(f'Joint probabilities in likelihood dict should equal 1 (prob. sum of each class). If not then miscalculation in NaiveBayes._calculate_log_likelihood_')
            print(f'Sum of joint probabilities: {[sum([math.pow(math.e, l) for l in self.likelihoods[c]]) for c in self.priors]}')
            print(f'Priors: {[math.pow(math.e, self.priors[c]) for c in self.classes]}')

    def predict(self, X_test):
        """
        Predict the X_test with the fitted model
        """
        #Create a dictionary of classes mapped to their posterior probabilities and return the class with the maximum probability for each x in X_test.
        return np.array([max(dict([(c, self._calculate_log_posterior_(x, c)) for c in self.classes]).items(), key=operator.itemgetter(1))[0] for x in X_test])
        


def confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix of the
    predictions with true labels
    """
    c = np.unique(y_true)
    matrix = np.zeros((len(c), len(c)))

    for i in range(len(c)):
        for j in range(len(c)):
            matrix[i, j] = np.sum((y_true == c[i]) & (y_pred == c[j]))

    return matrix


def load_data(return_numpy=False):
    """
    Load data

    Params
    ------
    return_numpy:   when true return the representation of Review Text
                    using the CountVectorizer or BagOfWords
                    when false return the Review Text

    Return
    ------
    X_train
    y_train
    X_valid
    y_valid
    X_test
    """
    X_train = pd.read_csv("Data/X_train.csv")['Review Text'].values
    X_valid = pd.read_csv("Data/X_val.csv")['Review Text'].values
    X_test  = pd.read_csv("Data/X_test.csv")['Review Text'].values
    y_train = (pd.read_csv("Data/Y_train.csv")['Sentiment'] == 'Positive').astype(int).values
    y_valid = (pd.read_csv("Data/Y_val.csv")['Sentiment'] == 'Positive').astype(int).values

    if return_numpy:
        # To do (not for Q1.1, used in Q1.3)
        # transform the Review Text into bag of word representation using vectorizer
        # process X_train, X_valid, X_test

        bag = CountVectorizer()
        X_train = bag.fit_transform(X_train).toarray()
        X_valid = bag.transform(X_valid).toarray()
        X_test = bag.transform(X_test).toarray()

    return X_train, y_train, X_valid, y_valid, X_test

def main():
    # Load in data
    X_train, y_train, X_valid, y_valid, X_test = load_data(return_numpy=False)
        
    # Fit the Bag of Words model for Q1.1
    bow = BagOfWords(vocabulary_size=10)
    bow.fit(X_train[:100])
    representation = bow.transform(X_train[100:200])

    # Load in data
    X_train, y_train, X_valid, y_valid, X_test = load_data(return_numpy=True)
    betas = [1]

    y_preds = []
    # Fit the Naive Bayes model for Q1.3
    for b in betas:
        nb = NaiveBayes(beta=b, debug=0)
        nb.fit(X_train, y_train)
        y_preds.append(nb.predict(X_valid))
    roc_auc = [roc_auc_score(y_valid, y_pred) for y_pred in y_preds]

    print('---------------   BAG OF WORDS MODEL   ---------------')
    print(bow.vocabulary)
    print(representation)
    print(dict(zip(bow.vocabulary, representation.sum(axis=0))))
    print('---------------   NAIVES BAYES MODEL   ---------------')
    for i in range(len(betas)):
        print(f'beta={betas[i]}')
        print(f'Confusion Matrix: ')
        print(confusion_matrix(y_valid, y_preds[i]))
        print("f1 score: " + str(f1_score(y_valid, y_preds[i])))
        print("ROC AUC score: " + str(roc_auc[i]))
        print("accuracy: " + str(accuracy_score(y_valid, y_preds[i])))

    print(betas)
    print(roc_auc)

if __name__ == '__main__':
    main()
