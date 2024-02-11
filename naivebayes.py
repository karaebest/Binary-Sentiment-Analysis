

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import tarfile
import os
import matplotlib.pyplot as plt
import time
np.random.seed(1234)

"""###Data Loading and Preprocessing"""

# Extract data from archive
my_tar = tarfile.open('aclImdb_v1.tar.gz')
my_tar.extractall('./') # specify which folder to extract to
my_tar.close()

def preprocess(path):

  #Get list of txt file names in pos and neg
  inputs_pos = os.listdir(path+"/pos")
  inputs_neg = os.listdir(path+"/neg")

  #Display number of pos and neg labelled samples in set and create set labels
  num_pos = len(inputs_pos)
  num_neg = len(inputs_neg)
  y = np.concatenate((np.ones(num_pos),np.zeros(num_neg)))


  # Add all training reviews into list
  reviews = []
  for fname in inputs_pos:
      with open(path+"/pos/"+fname, encoding="utf-8", errors='ignore') as infile:
          reviews.append(infile.read())
  for fname in inputs_neg:
      with open(path+"/neg/"+fname, encoding="utf-8", errors='ignore') as infile:
          reviews.append(infile.read())

  #Transform into bag of words representation using count vectorizer
  if "train" in path:
    count_matrix = vectorizer.fit_transform(reviews)
  else:
    count_matrix = vectorizer.transform(reviews)
  count_array = count_matrix.toarray()
  X = pd.DataFrame(data=count_array,columns =vectorizer.get_feature_names_out())
  return X, y

#Create vectorizer
vectorizer = CountVectorizer()
# Load and preprocess data
X_train, y_train = preprocess("aclImdb/train")
X_test, y_test = preprocess("aclImdb/test")
print(f'Vocabulary size: {X_train.shape[1]}')

"""###Naive Bayes : Bernoulli Prior"""

class NaiveBayes:

    def __init__(self, C=2):
      self.C = C
      self.pi = None
      self.theta = None
      return

    def fit(self, X, y):
        N, D = X.shape
        # one parameter for each feature conditioned on each class
        theta = np.zeros((self.C,D))
        Nc = np.zeros(self.C) # class counts
        for c in range(self.C):
          X_c = X[y == c]
          Nc[c] = X_c.shape[0]
          #Theta for each feature, label pair with laplace smoothing where alpha = 1 and beta = 1
          theta[c] = (X_c.sum(axis=0)+1)/(X_c.sum() + D)

        #Class MLE
        self.pi = (Nc+1)/(N+2)                      #Laplace smoothing 1xC
        self.theta = theta #C x D
        return self

    def predict(self, xt):
      Nt, D = xt.shape
      log_prior = np.log(self.pi) # 1 x C
      # logarithm of the likelihood term for Bernoulli
      log_likelihood = np.log(self.theta).dot(xt.T) # C x N
      # unnormalized log posterior p'(y=c|x,D)
      log_posterior = np.zeros((self.C, Nt)) #C x N
      for c in range(self.C):
        log_posterior[c] = log_prior[c] + log_likelihood[c]
      # normalized log posterior calculation p(y=c|x, D)
      posterior = np.exp(log_posterior - logsumexp(log_posterior))
      yh = np.argmax(posterior, axis=0).T   # N x 1
      return yh

    def evaluate_acc(self, yh, y):
      return np.mean(yh==y)

def logsumexp(Z):                                              # dimension C x N
  Zmax = np.max(Z,axis=0)[None,:]                              # max over C
  log_sum_exp = Zmax + np.log(np.sum(np.exp(Z - Zmax), axis=0))
  return log_sum_exp

"""###Experiments"""

# Comparing difference between including and excluding stopwords (min_df and max_df unspecified)

#Stopwords included:

#Create vectorizer
vectorizer = CountVectorizer()
# Load and preprocess data
X_train, y_train = preprocess("aclImdb/train")
X_test, y_test = preprocess("aclImdb/test")
print(f'Vocabulary size: {X_train.shape[1]}')
model = NaiveBayes()
model.fit(X_train, y_train)
yh = model.predict(X_test)
acc = model.evaluate_acc(yh, y_test)
print("Accuracy with stopwords included:"+str(acc))

#Stopwords excluded:

#Create vectorizer
vectorizer = CountVectorizer(stop_words='english')
# Load and preprocess data
X_train, y_train = preprocess("aclImdb/train")
X_test, y_test = preprocess("aclImdb/test")
print(f'Vocabulary size w/o stopwords: {X_train.shape[1]}')
model = NaiveBayes()
model.fit(X_train, y_train)
yh = model.predict(X_test)
acc = model.evaluate_acc(yh, y_test)
print("Accuracy with stopwords excluded:"+str(acc))

# View the effect of minimum data frequency on accuracy

min_dfs = [1, 10, 20, 50, 70, 80, 90, 95, 100, 105, 110, 115, 120, 150, 200, 500]
accs = []
for min_df in min_dfs:
  vectorizer = CountVectorizer(min_df=min_df)
  X_train, y_train = preprocess("aclImdb/train")
  X_test, y_test = preprocess("aclImdb/test")
  model = NaiveBayes()
  model.fit(X_train, y_train)
  yh = model.predict(X_test)
  acc = model.evaluate_acc(yh, y_test)
  accs.append(acc)
  print(f'Min_df: {min_df}, Vocabulary Size: {X_train.shape[1]}, Test Accuracy: {acc} \n')

fig = plt.figure()
plt.plot(min_dfs, accs, marker= '.')
plt.xlabel('Min Data Frequencies')
plt.ylabel('Test accuracy')
plt.title('The effect of minimum data frequency limits on test accuracy')
plt.show()

# View the effect of maximum data frequency on test accuracy
max_dfs = [2500, 2250, 2000, 1750, 1500, 1000]
accs = []
for max_df in max_dfs:
  vectorizer = CountVectorizer(max_df=max_df)
  X_train, y_train = preprocess("aclImdb/train")
  X_test, y_test = preprocess("aclImdb/test")
  model = NaiveBayes()
  model.fit(X_train, y_train)
  yh = model.predict(X_test)
  acc = model.evaluate_acc(yh, y_test)
  accs.append(acc)
  print(f'Max_df: {max_df}, Vocabulary Size: {X_train.shape[1]}, Test Accuracy: {acc} \n')


fig = plt.figure()
plt.plot(max_dfs, accs, marker= '.')
plt.xlabel('Max Data Frequencies')
plt.ylabel('Test accuracy')
plt.title('The effect of maximum data frequency limits on test accuracy')
plt.show()

# Execution Time
#Create vectorizer
vectorizer = CountVectorizer(min_df=110)
# Load and preprocess data
X_train, y_train = preprocess("aclImdb/train")
X_test, y_test = preprocess("aclImdb/test")
print(f'Vocabulary size: {X_train.shape[1]}')


model = NaiveBayes()
start = time.time()
model.fit(X_train, y_train)
yh = model.predict(X_test)
end = time.time()
acc = model.evaluate_acc(yh, y_test)

print(f'Accuracy: {acc}, Execution Time: {end-start}')