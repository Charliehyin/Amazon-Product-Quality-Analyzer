#import and clean data
#importing packages
import warnings
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn import preprocessing
from sklearn import utils
import collections
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import preprocessing
from sklearn import utils
import scipy.sparse
import os
import json
import re
import math
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.metrics import classification_report, make_scorer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

#read data from files

current_directory = 'source'
dataset = 'CDs_and_Vinyl'
train = pd.read_json(os.path.join(current_directory, dataset, 'train', 'review_training.json'))
awesometrain = pd.read_json(os.path.join(current_directory, dataset, 'train', 'product_training.json'))

test = pd.read_json(os.path.join(current_directory, dataset, 'test3', 'review_test.json'))
awesometest = pd.read_json(os.path.join(current_directory, dataset, 'test3', 'product_test.json'))

#convert json raw into dataframe
df = pd.DataFrame(train)
df = df.sort_values('asin')

dftest = pd.DataFrame(test)
dftest = dftest.sort_values('asin')

awesometrain = awesometrain.sort_values('asin')
awesometrain = awesometrain.drop("asin", axis=1)
y_train = pd.DataFrame(awesometrain)

awesometest = awesometest.sort_values('asin')
y_test = pd.DataFrame(awesometest)

def sentimentalAnalysis(df):
  # initialize variables
  sol = np.zeros([2, df.shape[0]])
  sentanal = SentimentIntensityAnalyzer()

  # loop through all reviews
  for i in range(df.shape[0]):
      
    # if text is empty, then sentimental analysis=0
    if not df['reviewText'].iloc[i]:
      sol[0][i] = 0
    else: 
      #otherwise, calculate Sentimental analysis values
      SAValue = sentanal.polarity_scores(df['reviewText'].iloc[i])
      sol[0][i] = SAValue["compound"]

    #do sent anal on summary text
    if not df['summary'].iloc[i]:
      sol[1][i] = 0
    else:
      #otherwise, calculate Sentimental analysis values
      SAValue = sentanal.polarity_scores(df['summary'].iloc[i])
      sol[1][i] = SAValue["compound"]
  return sol

def uniqueItems(df):
  #initialize Variables
  uniqueItems = 1
  curritem = df['asin'].iloc[0]

  #loop through all rows
  for i in range(df.shape[0]):

    #check if the review is unique
    if df['asin'].iloc[i] != curritem:
      curritem = df['asin'].iloc[i]
      uniqueItems = uniqueItems + 1
      #end of if statement

  return uniqueItems

#parse data
def parseRow(votes, unixReview, reviewCount, nones, verified, compound,  
             compoundsumm, reviewlens, summarylens, reviews, summaries, asin):
      row = np.zeros(57, dtype=object)

      #Percent of Nones final Tally
      row[0] = nones / reviewCount
    
      #Max Votes
      row[1] = np.max(votes)

      #Average Vote
      row[2] = np.average(votes)

      #Standard deviation votes
      row[3] = np.std(votes)

      #Percent of verified
      row[4] = np.sum(verified)/reviewCount
      
      #Minimum Review Time
      row[5] = np.min(unixReview)
      
      #Maximum Review Time
      row[6] = np.max(unixReview)

      #Average review time
      row[7] = np.average(unixReview)

      #Standard Deviation of review time
      row[8] = np.std(unixReview)

      #Number of reviews
      row[9] = reviewCount

      #sentimental analysis
      row[10] = np.average(compound)
      row[11] = np.std(compound)

      #next 18 are sentiment histogram
      #next 18 after that are votes counter histogram
      for i in range(len(compound)):
        #split compound into histogram
        index = 12
        if compound[i] != 0:
          index = index + math.ceil((compound[i]*4)+4)
        if verified[i] == 0:
          index = index + 9
        row[index] = row[index] + 1

        #count the votes
        index = index + 18
        row[index] = row[index] + votes[i]

      #turn the compound sentanal into percentage
      for i in range(12, 30):
        row[i] = row[i]/len(compound)

      row[48] = np.average(compoundsumm)
      row[49] = np.std(compoundsumm)

      #length of review and summary
      row[50] = np.average(reviewlens)
      row[51] = np.average(summarylens)
      row[52] = np.std(reviewlens)
      row[53] = np.std(summarylens)

      #review and summary raw text
      row[54] = ''
      row[55] = ''
      for i in range(len(reviews)):
        if reviews[i] != None:
          row[54] = row[54] + reviews[i] + ' '
        if summaries[i] != None:
          row[55] = row[55] + summaries[i] + ' '

      #asin
      row[56] = asin

      return row

def parseData(unique, SA, df):
  #add rows and columns to aggregated data
  newData = pd.DataFrame(columns=["Percent of Nones", "Max Votes", "Average Votes",
                                  "Standard Deviation Votes", "Percent of Verified", 
                                  "Minimum Review Time", "Maximum Review Time", 
                                  "Average Review Time", "Stand Deviation of Review Time",
                                  "Number of Reviews", "compound average", "compound SD", 
                                  "% 0", "% -1 to -.75", "% -.75 to -.5", "% -.5 to -.25", "% -.25 to 0", 
                                  "% 0 to .25", "% .25 to .5", "% .5 to .75", "%.75 to 1",
                                  "%uv 0", "%uv -1 to -.75", "%uv -.75 to -.5", "%uv -.5 to -.25", "%uv -.25 to 0", 
                                  "%uv 0 to .25", "%uv .25 to .5", "%uv .5 to .75", "%uv .75 to 1",
                                  "#votes 0", "#votes -1 to -.75", "#votes -.75 to -.5", "#votes -.5 to -.25", "#votes -.25 to 0", 
                                  "#votes 0 to .25", "#votes .25 to .5", "#votes .5 to .75", "#votes .75 to 1",
                                  "#votes uv 0", "#votes uv -1 to -.75", "#votes uv -.75 to -.5", "#votes uv -.5 to -.25", "#votes uv -.25 to 0", 
                                  "#votes uv 0 to .25", "#votes uv .25 to .5", "#votes uv .5 to .75", "#votes uv .75 to 1",
                                  "compound average summ", "compound SD summ", 
                                  "avg length of review", "avg length of summary", 
                                  "std length of review", "std length of summary", 
                                  "reviews", "summaries", "asin"],
                         index = range(unique))
  #initialize variables
  uniqueItems = 0
  itemcount = 0
  curritem = df['asin'].iloc[0]
  votes = []
  unixReview = []
  reviewCount = 0
  nones = 0
  verified = []

  #sentimental anaylsis variables
  compound = []
  compoundsumm = []
  reviewlens = []
  summarylens = []
  reviews = []
  summaries = []

  #loop through every row in data
  for i in range(df.shape[0]):    
    if (df['asin'].iloc[i] != curritem):
      itemcount=itemcount+1
      
      #append array into panda dataframe
      newData.loc[uniqueItems] = parseRow(votes, unixReview, reviewCount, nones, 
                                          verified, compound, compoundsumm, reviewlens, summarylens, 
                                          reviews, summaries, df['asin'].iloc[i-1])

      #reset the value of the variables
      nones = 0
      reviewCount = 0
      votes = []
      unixReview = []
      verified = []
      compound = []
      compoundsumm = []
      reviewlens = []
      summarylens = []
      reviews = []
      summaries = []
      uniqueItems = uniqueItems + 1
      # end of if statement
    curritem = df['asin'].iloc[i]
    reviewCount = reviewCount + 1

    #Percent of Nones counting number of nums
    if df['vote'].iloc[i] == None or (df['vote'].iloc[i] != df['vote'].iloc[i]):
      nones += 1
      votes.append(0)
    else:
      #Votes
      try:
        numVotes = int(re.sub(",", "", df['vote'].iloc[i]))
        votes.append(numVotes)
      except Exception as e:
          nones += 1
          votes.append(0)

    #Count verified
    if df['verified'].iloc[i] == True:
      verified.append(1)
    else:
      verified.append(0)

    #append unixReviewTime
    unixReview.append(int(df['unixReviewTime'].iloc[i]))

    #sentimental analysis
    compound.append(SA[0][i])
    compoundsumm.append(SA[1][i])

    #length of review and summary
    if df['reviewText'].iloc[i] != None:
      reviewlens.append(len(df['reviewText'].iloc[i]))
    else: 
      reviewlens.append(0)

    if df['summary'].iloc[i] != None:
      summarylens.append(len(df['summary'].iloc[i]))
    else: 
      summarylens.append(0)
    
    #review and summary raw text
    reviews.append(df['reviewText'].iloc[i])
    summaries.append(df['summary'].iloc[i])

    #end of for loop

  # parse last rows data
  newData.loc[unique - 1] = parseRow(votes, unixReview, reviewCount, nones, 
                                     verified, compound, compoundsumm, reviewlens, summarylens, 
                                     reviews, summaries, df['asin'].iloc[unique-1])
                                     
  return newData

def standardize(df):
  result = df.copy()
  for feature_name in df.columns:
    if feature_name not in ["reviews", "summaries", "asin"] and feature_name[0] != '%':
      #convert column to numeric
      result[feature_name] = pd.to_numeric(df[feature_name])

      #this line is the calculation
      if result[feature_name].std() != 0 and result[feature_name].max() != 0:
        
        result[feature_name] = (result[feature_name]-result[feature_name].mean()) / result[feature_name].std()
        #make 0 lowest value
        result[feature_name] = result[feature_name] + abs(result[feature_name].min())
        #standardize to 0-1
        result[feature_name] = result[feature_name]/result[feature_name].max()
  return result

#run preprocessing on testing data
testDataSA = sentimentalAnalysis(dftest)
uniqueItemsTest = uniqueItems(dftest)
testData = parseData(uniqueItemsTest, testDataSA, dftest)
X_test = standardize(testData)

#run preprocessing on training data
unique = uniqueItems(df)
trainDataSA = sentimentalAnalysis(df)
trainData = parseData(unique, trainDataSA, df)
X_train = standardize(trainData)

#tf idf vectorizer
tfidf = TfidfVectorizer(stop_words ='english')
tog = X_train['reviews'] + X_train['summaries']
togtest = X_test['reviews'] + X_test['summaries']

#vectorize training data
transf = tfidf.fit_transform(tog.fillna(""))
transf = pd.DataFrame.sparse.from_spmatrix(transf)

#vectorize test data
transftest = tfidf.transform(togtest.fillna(""))
tramsftest = pd.DataFrame.sparse.from_spmatrix(transftest)

#reduce dimensionality of tfidf vector for multinomial naive bayes
selection = SelectPercentile(chi2, percentile=12)
wtrainMNB = selection.fit_transform(transf, y_train)
wtraintestMNB = selection.transform(transftest)

#reduce dimensionality of tfidf vector for bernoulli naive bayes
selection = SelectPercentile(chi2, percentile=34)
wtrainBNB = selection.fit_transform(transf, y_train)
wtraintestBNB = selection.transform(transftest)

#reduce dimensionality of tfidf vector for logistic regression
selection = SelectPercentile(chi2, percentile=20)
wtrainLR = selection.fit_transform(transf, y_train)
wtraintestLR = selection.transform(transftest)

#pull numerical data from preprocessing step
ntrain = X_train.iloc[:,:54]
ntest = X_test.iloc[:,:54]

#select the top 13 features for decision tree
fisher_selector = SelectKBest(score_func=f_classif, k=13)
X_top_features = fisher_selector.fit_transform(ntrain, y_train)
selected_features = ntrain.columns[fisher_selector.get_support(indices=True)]
ntrainDTC = ntrain.loc[:,selected_features]
ntestDTC = ntest.loc[:,selected_features]

#select the top 40 features for logistic regression
fisher_selector = SelectKBest(score_func=f_classif, k=40)
X_top_features = fisher_selector.fit_transform(ntrain, y_train)
selected_features = ntrain.columns[fisher_selector.get_support(indices=True)]
ntrainLR = ntrain.loc[:,selected_features]
ntestLR = ntest.loc[:,selected_features]

#create the classifiers
wMNB = MultinomialNB(alpha = 0.0000000001)
wBNB = BernoulliNB(alpha = 0.0000000001)
wLR = LogisticRegression()
nMNB = MultinomialNB(alpha = 0.0000000001)
nBNB = BernoulliNB(alpha = 0.0000000001)
nLR = LogisticRegression()
RF = RandomForestClassifier(max_depth=14)
DTC = DecisionTreeClassifier(criterion='entropy', splitter = 'best', max_depth=9)
ABC = AdaBoostClassifier(n_estimators=100)

#fit the classifiers
wMNB.fit(wtrainMNB, y_train)
wBNB.fit(wtrainBNB, y_train)
wLR.fit(wtrainLR, y_train)
nMNB.fit(ntrain, y_train)
nBNB.fit(ntrain, y_train)
nLR.fit(ntrainLR, y_train)
RF.fit(ntrain, y_train)
DTC.fit(ntrainDTC, y_train)
ABC.fit(ntrain, y_train)

#use the classifiers to predict probabilities
predA = wMNB.predict_proba(wtraintestMNB)
predB = wBNB.predict_proba(wtraintestBNB)
predC = wLR.predict_proba(wtraintestLR)
predD = nMNB.predict_proba(ntest)
predE = nBNB.predict_proba(ntest)
predF = nLR.predict_proba(ntestLR)
predG = RF.predict_proba(ntest)
predH = DTC.predict_proba(ntestDTC)
predI = ABC.predict_proba(ntest)

#concatenate classifier predictions into dataframe
row = {'TFIDF-MNB':predA[:,1], 'TFIDF-BNB': predB[:,1], 'TFIDF-LR': predC[:,1], 'MNB': predD[:,1],
       'BNB':predE[:,1], 'LR': predF[:,1], 'RF': predG[:,1], 'DTC':predH[:,1], 'ABC': predI[:,1]}
predsDF = pd.DataFrame(row)

#set weights and threshold for late fusion
#check attached ipynb for the hyperparameter training
a = 0.00000001
weightA = np.array([20, 10, 100])
weightB = np.array([10, 20, 90])
weightC = np.array([100, 10, 40])
thABC = np.array([0.5, 0.5, 0.5])
th = .4
weights = np.array([80, 20, 10])

#late fusion, acts as a soft voting classifier that is weighted
y_predA = (predsDF.iloc[:,[0, 1, 2]].mul(weightA) + a) / (weightA.sum() + a)
y_predA * 0.5 / thABC[0]
y_predA = y_predA.sum(axis=1)
y_predB = (predsDF.iloc[:,[3, 4, 5]].mul(weightB) + a) / (weightB.sum() + a)
y_predA * 0.5 / thABC[1]
y_predB = y_predB.sum(axis=1)
y_predC = (predsDF.iloc[:,[6, 7, 8]].mul(weightC) + a) / (weightC.sum() + a)
y_predA * 0.5 / thABC[2]
y_predC = y_predC.sum(axis=1)
y_pred = (y_predA * weights[0] + y_predB * weights[1]+ y_predC * weights[2]+ a) / (weights.sum() + a)
y_pred[y_pred <= th] = 0
y_pred[y_pred > th] = 1

#predict on test data
predictions=pd.concat([X_test['asin'], y_pred], axis=1)
predictions.columns = ['asin', 'predictions']

#write predictions to predictions.json
with open('predictions.json', 'w') as f:
  f.write(predictions.to_json())