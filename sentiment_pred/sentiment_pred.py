from sklearn.metrics import classification_report               # To analyze our prediction results
from sklearn.metrics import confusion_matrix                    # To display a summary of the prediction results
import math                                                     # To use the random function
import random                                                   # To randomly split the input data into a training set and a test set
import csv                                                      # To read and split comma separated files or other delimiter
import glob                                                     # Unix style pathname pattern expansion
import re                                                       # To use regular expressions
import numpy as np                                              # To use numpy arrays
from sklearn.model_selection import train_test_split            # To split our data
from sklearn.ensemble import GradientBoostingClassifier         # ML algorithm for our sentiment analyzer
from sklearn.feature_extraction.text import CountVectorizer     # To create bag of words
import pandas as pd                                             # To use a bar graph for our prediction results

def process_data():
    """ Process data for our data files. (Step 1)

      Process the data from our text files that contain reviews and the sentimental score for that review.
      Our text file contains the format: Review sentence \t sentimental score.

      :return: An input list of reviews and an output of an np array of the sentimental value (0 or 1).
    """
    # Get the files used for learning and testing. Should be three files:
    # amazon_cells_labelled.txt, yelp_labelled.txt and imdb_labelled.txt
    data_files = glob.glob("./Training_Data/*.txt")
    # populate input list and output list with data, separating sentences from the scores
    input_list = []   # Input list containing review sentences
    # Output list containing sentiment values of 0 or 1: 0 negative or 1 positive
    output_list = []
    # Go through all the files and populate the input data along with its sentiment value in the output list
    for data_file in data_files:
        # For all text files we read one at a time
        with open(data_file, 'r') as file:
            # Read file and seperate review sentences from sentiment value
            # since our files are separated by a tab space
            text = csv.reader(file, delimiter='\t')
            for line in text:
                # For each line we will get a sentence and a sentiment value of 0(negative) and 1(positive)
                sentence = line[0]                      # Get review sentence
                # The following 3 re.sub gets rid of any tabs and newlines for better extraction
                sentence = re.sub('\\t0', '', sentence)
                sentence = re.sub('\\t1', '', sentence)
                sentence = re.sub('\n', '', sentence)
                # Add sentence review and sentiment value to our lists
                input_list.append(sentence)       # One sentence of a review
                output_list.append(int(line[1]))  # Sentiment Value

    # Make our output an np array to be compatible with sklearn functions(np array)
    output_array = np.array(output_list)

    return input_list, output_array

def split_data(X, y):
    """ Split our data set of outputs and inputs. (Step 2)
        :param X:   The input list of our data set. In our case it will be sentences of a review.
        :param y:   The output list/array of our data set. In our case it will be the sentiment value based on a review.
        :return:    Four lists of: An input training set, an input test set, an output training set, and an output test set.
    """
    # Constants for our arguments passed in sklearn.train_test_split
    TEST_FRACTION = .2   # The fraction size of our test data
    # The constant random state we will use to shuffle the data if we decide to use one
    RANDOM_STATE = 68

    # Call sklearn.train_test_split to split our datat into two (Training and Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION)

    return X_train, X_test, y_train, y_test

def get_bag_of_words(input_text_list):
    """ get bag of words representation for data set of inputs (aka transforming the training data).
        These will help us train our data set easier.

        :param input_text_list: The input list that will like to have a bag-of-words representation.
        :return: A bag of words list along with its vector, e.g.: bag-of-words, vect
    """

    # Create an instance of CountVectorizer
    vect = CountVectorizer().fit(input_text_list)
    transform_text_list = vect.transform(
        input_text_list)    # Transform the text list

    return transform_text_list, vect

def getSentimmentEmojiFaces():
    """ Emoji face constants that we will be using to represent a sentiment value
        :return sentiment_val_1, sentiment_val_0
    """
    ANGRY_FACE = '\U0001F603'   # 😡
    SMILY_FACE = '\U0001F621'   # 😃

    return SMILY_FACE, ANGRY_FACE


def getSentimentEmojiThumbs():
    """ Emoji thumbs constants that we will be using to represent a sentiment value
        :return sentiment_val_1, sentiment_val_0
    """
    THUMBS_DOWN = '\U0001F44E'  # 👎
    THUMBS_UP = '\U0001F44D'    # 👍

    return THUMBS_UP, THUMBS_DOWN


if __name__ == '__main__':
    # Flag to display data, train data, and incorrect prediction data.
    display_data = False

    # Process our data
    input_list, output_list = process_data()
    input_len = len(input_list)
    if display_data:
        for index in range(0, input_len):
            print(output_list[index], '-', input_list[index])
    print('Total Length:', len(input_list))

    """ Step 2: Data splitting
        Split our data in two halves. One half for training and the other for testing. 
        The training set will then be further split in two parts: one for training and the other for validation when 
        we test for hyperparameters. So our final sets will be the following ratios from our data: (training set:25%),
        (validation set: 25%), and (testing set: 50%).
    """

    # Split our data
    # First half will be training set and the second half will be testing set
    X_train, X_test, y_train, y_test = split_data(input_list, output_list)

    """ Step 3: Transform Data
        Transform our training set into a bag of words
    """

    # Set our input data into bag of words for feature extraction
    transformed_X_train, X_train_vect = get_bag_of_words(X_train)

    """ Step 4 Train and Test
        Using Gradient Boosting Classifier ML algorithm
    """

    # Get best parameters that we got (See the long statistical analysis of this project)
    grbt_best = GradientBoostingClassifier(learning_rate=0.2, loss='exponential',
                                           max_depth=8, max_features='sqrt')

    # Train our learning model
    grbt_best.fit(transformed_X_train, y_train)
    X_test_transformed = X_train_vect.transform(X_test)
    # Now make the predictions from our test set
    grbt_y_predict = grbt_best.predict(X_test_transformed)
    # Calculate and display the test score
    np.mean(grbt_y_predict == y_test)
    print('Test Score: {:.2f}'.format(
        grbt_best.score(X_test_transformed, y_test)))

    """ Display Prediction Results
        Print Data with prediction using emojis as the sentiment values

        First Column
        *   ✓ Correct Prediction
        *   ✕ Incorrect Prediction

        Second Column
        *   😃 Predicted a good review
        *   😡 Predicted a bad review

        Third Column
        *   A review sentence
    """
    length = len(y_test)
    incorrect_pred = []
    neg_sentiment, pos_sentiment = getSentimmentEmojiFaces()
    # Check mark or cross mark depending if we correctly predicted the sentiment
    CORRECT_PRED = '\U00002713'  # ✓
    WRONG_PRED = '\U00002715'    # ✕
    for index in range(length):
        truth_val = y_test[index]           # Get the actual sentiment value
        # Get the sentiment prediction value from our learning algorithm
        prediction = grbt_y_predict[index]
        # Get the paired review sentence that comes with this sentiment
        sentence = X_test[index]

        correct_pred = ''
        if truth_val == prediction:
            correct_pred = CORRECT_PRED
        else:
            correct_pred = WRONG_PRED
            incorrect_pred.append(index)

        if display_data:
            pred_sentiment = pos_sentiment if prediction else neg_sentiment
            print(correct_pred, pred_sentiment, X_test[index])

    # Total number of incorrect predictions

    print(len(incorrect_pred), 'incorrect predictions out of',
          len(y_test), 'predictions.')
    print('From our test set.')

    # Display Prediction Stats

    if display_data:
        print('Number of wrong predictions:', len(
            incorrect_pred), 'out of', len(grbt_y_predict))
        predPerc = 1 - (len(incorrect_pred) / len(grbt_y_predict))
        print('Prediction Accuracy: %.2f' % predPerc, '%')

        confusion = confusion_matrix(y_pred=grbt_y_predict, y_true=y_test)
        print(confusion)

        print(classification_report(y_test, grbt_y_predict,
              target_names=["Bad review", "Good review"]))

        positive_values = [0.87, 0.75, 0.81]
        negative_values = [0.78, 0.89, 0.83]
        index = ['Precision', 'Recall', 'f1-score', ]
        df = pd.DataFrame({'Good Review': positive_values,
                          'Bad Review': negative_values}, index=index)
        ax = df.plot.bar(
            rot=0, color={"Good Review": "yellow", "Bad Review": "red"})

    # Display wrong predictions from our model

    # Display Positive-Negative errors
    if display_data:
        for index in incorrect_pred:
            truth_val = y_test[index]
            prediction = grbt_y_predict[index]
            if prediction:
                sentence = X_test[index]
                pred_sentiment = '\N{grinning face}' if prediction else '\N{pouting face}'
                truth_sentiment = '\N{grinning face}' if truth_val else '\N{pouting face}'
                print('P:', pred_sentiment, 'A:',
                      truth_sentiment, X_test[index])

    # Display Positive-Negative errors

    if display_data:
        print('Predicted, Actual\n')
        for index in incorrect_pred:
            truth_val = y_test[index]
            prediction = grbt_y_predict[index]
            if not prediction:
                sentence = X_test[index]
                pred_sentiment = '\N{grinning face}' if prediction else '\N{pouting face}'
                truth_sentiment = '\N{grinning face}' if truth_val else '\N{pouting face}'
                print('P:', pred_sentiment, 'A:',
                      truth_sentiment, X_test[index])

    # Try a custom review

    # Get input from user
    custom_input_review = input('Enter a one sentence review\n')

    # Print input from the user
    print()
    print('One sentence review:')
    print(custom_input_review)

    # Predict sentiment value of user's input
    # Transform input into data we have trained with
    cust_transformed = X_train_vect.transform([custom_input_review])
    cust_predict = grbt_best.predict(
        cust_transformed)                # Make prediction

    # Print prediction result
    NEG_SENTIMENT, POS_SENTIMENT = getSentimmentEmojiFaces()
    pred_emoji = POS_SENTIMENT if cust_predict[0] else NEG_SENTIMENT

    print()
    print('Learning Model sentiment prediction:', pred_emoji)
