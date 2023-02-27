import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

class NaiveBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    '''

    def __init__(self):
        return 

    def fit(self, X, y):
        '''
        Create a table that will allow the filter to evaluate P(H), P(S)
        and P(w|C)

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        # Get all words
        all_words = [i for x in X.str.split() for i in x]
        unique_words = list(set(all_words))
        
        # Get spam and ham messages
        spam_messages = X[y == 'spam']
        spam_words = [i for x in spam_messages.str.split() for i in x]
        ham_messages = X[y == 'ham']
        ham_words = [i for x in ham_messages.str.split() for i in x]
        
        
        # Create empty dataframe
        df = pd.DataFrame(0, index=y.unique(), columns=unique_words)
        
        # Fill it in
        for word in unique_words:
            df.loc['spam', word] = spam_words.count(word)
            df.loc['ham', word] = ham_words.count(word)
        
        self.data = df
        self.N_spam = len(spam_messages)
        self.N_ham = len(ham_messages)
        self.N_samples = len(X)
        self.spam_count = sum(df.loc['spam'])
        self.ham_count = sum(df.loc['ham'])

    def predict_proba(self, X):
        '''
        Find P(C=k|x) for each x in X and for each class k by computing
        P(C=k)P(x|C=k)

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,2): Probability each message is ham, spam
                0 column is ham
                1 column is spam
        '''
        # Compute P(C=spam) = N(spam) / N(samples)
        P_spam = self.N_spam / self.N_samples
        P_ham = self.N_ham / self.N_samples
        
        P_list = []
        for x in X:
            P_x_spam = 1
            P_x_ham = 1
            # for xi in x.split():
            for xi in np.unique(x.split()):
                ni = x.split().count(xi)
                try:
                    P_x_spam = P_x_spam * (self.data.loc['spam', xi] / self.spam_count) ** ni
                except:
                    pass
                try:
                    P_x_ham = P_x_ham * (self.data.loc['ham', xi] / self.ham_count) ** ni
                except:
                    pass

            P_spam_given_x = P_spam * P_x_spam
            P_ham_given_x = P_ham * P_x_ham
            P_list.append({'spam' : P_spam_given_x,
                           'ham' : P_ham_given_x})
        return pd.DataFrame(P_list)


    def predict(self, X):
        '''
        Use self.predict_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,): label for each message
        '''
        return self.predict_proba(X).idxmax(axis="columns")

    def predict_log_proba(self, X):
        '''
        Find ln(P(C=k|x)) for each x in X and for each class k

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,2): Probability each message is ham, spam
                0 column is ham
                1 column is spam
        '''

        # Compute P(C=spam) = N(spam) / N(samples)
        P_spam = np.log(self.N_spam / self.N_samples)
        P_ham = np.log(self.N_ham / self.N_samples)
        
        P_list = []
        for x in X:
            P_x_spam = 1e-20
            P_x_ham = 1e-20
            for xi in x.split():
                try:
                    P_x_spam = P_x_spam + np.log(self.data.loc['spam', xi] / self.spam_count)
                except:
                    pass
                try:
                    P_x_ham = P_x_ham + np.log(self.data.loc['ham', xi] / self.ham_count)
                except:
                    pass

            P_spam_given_x = P_spam + P_x_spam
            P_ham_given_x = P_ham + P_x_ham
            P_list.append({'spam' : P_spam_given_x,
                           'ham' : P_ham_given_x})
        return pd.DataFrame(P_list)
        

    def predict_log(self, X):
        '''
        Use self.predict_log_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,): label for each message
        '''
        
        return self.predict_log_proba(X).idxmax(axis="columns")
    
    def score(self, X_test, y_test):
        y_hat = self.predict_log(X_test)
        return sum(y_hat.to_numpy() == y_test.to_numpy())/ len(y_hat.to_numpy())

class PoissonBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    This classifier assumes that words are distributed like 
    Poisson random variables
    '''

    def __init__(self):
        return

    
    def fit(self, X, y):
        '''
        Uses bayesian inference to find the poisson rate for each word
        found in the training set. For this we will use the formulation
        of l = rt since we have variable message lengths.

        This method creates a tool that will allow the filter to 
        evaluate P(H), P(S), and P(w|C)


        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        
        Returns:
            self: this is an optional method to train
        '''
        all_words = [i for x in X.str.split() for i in x]
        unique_words = list(set(all_words))
        self.N_samples = len(all_words) 
        
        # Get spam and ham messages
        spam_messages = X[y == 'spam']
        self.spam_words = [i for x in spam_messages.str.split() for i in x]
        self.N_spam = len(self.spam_words)
        
        ham_messages = X[y == 'ham']
        self.ham_words = [i for x in ham_messages.str.split() for i in x]
        self.N_ham = len(self.ham_words)

        spam_rates = []
        ham_rates = []
        for i in unique_words:
            ni_spam = self.spam_words.count(i)
            r_i_spam = ni_spam / self.N_spam
            spam_rates.append((i, r_i_spam))
            
            ni_ham = self.ham_words.count(i)
            r_i_ham = ni_ham / self.N_ham
            ham_rates.append((i, r_i_ham))
        
        self.spam_rates = dict(spam_rates)
        self.ham_rates = dict(ham_rates)
    
    def predict_proba(self, X):
        '''
        Find P(C=k|x) for each x in X and for each class

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,2): Probability each message is ham or spam
                column 0 is ham, column 1 is spam 
        '''
        # Compute P(C=spam) = N(spam) / N(samples)
        P_spam = np.log(self.N_spam / self.N_samples)
        P_ham = np.log(self.N_ham / self.N_samples)
        
        P_list = []
        for x in X:
            P_x_spam = P_spam
            P_x_ham = P_ham
            n = len(x.split())
            for xi in np.unique(x.split()):
                ni = x.split().count(xi)
                try:
                    r = self.spam_rates[xi]
                    P_x_spam += np.log(((r * n)**ni) * np.exp(-r * n) / np.math.factorial(ni))
                except:
                    pass
                try:
                    r = self.ham_rates[xi]
                    P_x_ham += np.log(((r * n)**ni) * np.exp(-r * n) / np.math.factorial(ni))
                except:
                    pass

            P_list.append([P_x_spam, P_x_ham])
        return np.array(P_list)

    def predict(self, X):
        '''
        Use self.predict_proba to assign labels to X

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,): label for each message
        '''
        ind_max = np.argmax(self.predict_proba(X), axis=1)
        return np.array([['spam', 'ham'][i] for i in ind_max])



def sklearn_method(X_train, y_train, X_test):
    '''
    Use sklearn's methods to transform X_train and X_test, create a
    naïve Bayes filter, and classify the provided test set.

    Parameters:
        X_train (pandas.Series): messages to train on
        y_train (pandas.Series): labels for X_train
        X_test  (pandas.Series): messages to classify

    Returns:
        (ndarray): classification of X_test
    '''
    vectorizer = CountVectorizer()
    X_train_clean = vectorizer.fit_transform(X_train)
    clf = MultinomialNB()
    clf.fit(X_train_clean, y_train)
    
    X_test_clean = vectorizer.transform(X_test)
    return clf.predict(X_test_clean)