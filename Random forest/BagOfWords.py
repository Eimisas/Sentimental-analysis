import Preprocess as pp
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


def reviewsToWords(review):



    letters_only = re.sub("[^a-zA-Z]",  # The pattern to search for
                          " ",  # The pattern to replace it with
                          review)  # The text to search

    lower_case = letters_only.lower() #Converts to lower case
    words = lower_case.split()  #Splits into seperate words

    stops = set(stopwords.words("english"))

    wordsOfReview = [w for w in words if not w in stops] #Removes un-useful words (stops)

    returnValue = ( " ".join(wordsOfReview))   #Joins together words with space


    return returnValue

def performPrediction(dataset):
    # Size of training dataset 1
    no_of_reviews = 14175

    # Cleaned reviews
    cleaned_reviews = []

    # Puts cleaned dataset to new array cleaned_reviews
    print("Processing...")
    for i in xrange(0, no_of_reviews):
        cleaned_reviews.append(reviewsToWords(dataset["content"][i]))

    bag = CountVectorizer(analyzer="word",
                          tokenizer=None,
                          preprocessor=None,
                          stop_words=None,
                          max_features=3000)

    # Fit_transform learns the vocabulary
    trained_Data = bag.fit_transform(cleaned_reviews)

    # Convert to Numpy array
    train_data_features = trained_Data.toarray()

    # Random forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators=100)

    #Fit the train data with recommended values
    forest = forest.fit(train_data_features, dataset["recommended"][0:14175])

    ###################################TESTING PART#######################################
    print("Testing....")

    # Number of reviews
    num_reviews = len(dataset["content"])

    #New array with cleaned testing reviews
    clean_test_reviews = []

    count_pos = 0
    count_neg = 0
    for i in xrange(14175, num_reviews):
        if(dataset["recommended"][i] == 1):
            count_pos+=1
        else:
            count_neg += 1
        clean_review = reviewsToWords(dataset["content"][i])
        clean_test_reviews.append(clean_review)


    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = bag.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "content" column and
    # a "recommended" column
    output = pd.DataFrame(data={"content": dataset["content"][14175:], "recommended": result})
    print("Pos total %s" %count_pos)
    print("Neg total %s" %count_neg)

    return output


def performBagOfWords():

    #Gets data from preprocess
    dataset = pp.getData()

    #TODO Downloads languages
    #nltk.download()

    result = performPrediction(dataset)

    #If reccomended print
    counter_pos = 0
    for index, row in result.iterrows():
        if(row['recommended'] == 1):
            #print(row['content'], row['recommended'])
            counter_pos+=1
    print("Recommended in total %d" % counter_pos)

    counter_neg = 0
    for index, row in result.iterrows():
        if(row['recommended'] == 0):
            #print(row['content'], row['recommended'])
            counter_neg+=1
    print("NOT Recommended in total %d" % counter_neg)

    return








