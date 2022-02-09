# Sentiment Prediction

## Description
A python application that predicts a sentiment value from a user's input.
A user's input consists of one sentence.
The sentiment value in this application is either a good review or a bad review. The sentiment prediction model cannot
predict neutral reviews. 

## Screenshots

## Installating From Source
1. Clone this Github repository into your desired system directory
2. Install [python](https://www.python.org/downloads/) if not installed
    * To check if a program is installed run `which python3` in your terminal
3. Install the pip list packages below if you do not have them installed
    * Install by running `pip install <package_name>` in the terminal

### Dependencies
* sklearn
* panda 
* matplotlib

## Running the Program

### From Source
1. Go to the recently cloned repository location 
2. Go the the sentiment_pred directory 
3. Run the sentiment_pred.py file
    * `python3 sentiment_pred.py`

#### Directory Tree
```
SentimentPrediction/
    ├── README.md
    ├── python_notebook/
    │   └── SimplifiedSentimentAnalysis.ipynb
    ├── screenshots/
    │   ├── dataOutputExample.png
    │   ├── doNotRunSection.png
    │   └── pred.png
    └── sentiment_pred/
        ├── /Training_Data
        │   ├── README.md
        │   ├── amazon_cells_labelled.txt
        │   ├── imdb_labelled.txt
        │   └── yelp_labelled.txt
        └── sentiment_pred.py
```

#### Note
The display learning results is turned off by default. To get the same output as the Google Colab set the variable
`display_data = False` to `True`. The variable is the first variable defined in the main function. 

### Google Colab
1. Click on the link [here](https://colab.research.google.com/drive/1WS92wYYLdjqsyPtaH123dIpxUtsxVHNa?usp=sharing)
to take you to the Project's Google Colab or click the following [link](https://colab.research.google.com/drive/1WS92wYYLdjqsyPtaH123dIpxUtsxVHNa?authuser=1#scrollTo=2rdxQ5cksgzA&line=4&uniqifier=1) 
to go straight to the custom sentiment prediction
2. Click play on any code except in the *Mount Data for Extraction* section since that will remove the files to train
the learning model in order to run the program.
3. The last section named; *Enter Your Own Review* is where you can test the learning model with your own sentence review

### Other Python Notebook Program
* You could run the program in a python notebook such as Jupyter Notebook
1. Upload or open the file *SimplifiedSentimentAnalysis.ipynb* from the repository 
    * `SentimentPrediction/python_notebook/SimplifiedSentimentAnalysis.ipynb` file path
2. Comment out or remove the section *Mount Drive*
3. In section *Functions Defined Step:1...* and in the function process_data change the directory where the files are
located for the training data instead of using gdrive.
    * `data_files = glob.glob("/content/gdrive/My Drive/Sentiment Analysis Data/*.txt")`

## Sources

### Sentiment Analysis
Sentiment Prediction is based off an analysis I done in the past to find the best learning model and its best 
parameters. The learning model is trained to predict the sentiment value of a one sentence review with a high successful
rate. To see more of this sentiment analysis click on the Github repository or Google Colab below.
* [Sentiment Analysis Github Repository](https://github.com/AmielCyber/Sentiment-Analysis)
* [Sentiment Analysis Google Colab](https://colab.research.google.com/drive/1AnsFgIXoibD4XET9OAR2HvEqMX4LLxjx?usp=sharing)

### Methods and Reference Used
Machine Learning methods and refenerence used from the following book:

An Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido (O’Reilly). Copyright 2017 Sarah Guido and Andreas Müller, 978-1-449-36941-5.

### Google Colab Link for Sentiment Prediction
[Sentiment Prediction Google Colab](https://colab.research.google.com/drive/1WS92wYYLdjqsyPtaH123dIpxUtsxVHNa?usp=sharing)

