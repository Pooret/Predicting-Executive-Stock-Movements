#!/usr/bin/env python
# coding: utf-8

# In[4]:

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# For machine learning
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support as score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve
import warnings
import plotly.figure_factory as ff
import random
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
#!/usr/bin/env python
# coding: utf-8

# Data Wrangling
import twint
import yfinance as yf
from datetime import datetime as dt
from datetime import timedelta
from datetime import date

# EDA
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import spacy
import re
import html
from bs4 import BeautifulSoup
import unicodedata
from wordcloud import WordCloud
get_ipython().run_line_magic('matplotlib', 'inline')
from textblob import TextBlob, Blobber
from spacy import displacy
from IPython.display import display_html
from itertools import chain,cycle
import scipy.stats as stats
from scipy.stats import spearmanr, pearsonr
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.stats import ttest_ind
from nltk.tokenize import sent_tokenize

# For text preprocessing
import spacy
from nltk.tokenize import sent_tokenize
from textblob.sentiments import NaiveBayesAnalyzer
import emoji
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import feature_selection
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
import twint
from bs4 import BeautifulSoup
import re
import yfinance as yf
from nltk.tokenize import WordPunctTokenizer
from datetime import datetime as dt
from datetime import timedelta
import demoji
from datetime import date
import contractions
import wikipedia
from sklearn.pipeline import make_pipeline
import contractions
nlp = spacy.load('en_core_web_lg')
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Training/Testing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder

# Performance Measures
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, make_scorer

# Classifers
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# Interpretability
from sklearn.inspection import permutation_importance
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from lime import lime_tabular
 



def get_text_features(df, text_col):
    
    def average_word_length(text):
        words = text.split()
        word_length = 0
        for word in words:
            word_length += len(word)
        return word_length / len(words)
    
    pattern_email = r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)'
    
    df['word counts'] = df[text_col].apply(lambda x: len(str(x).split())) # word counts
    df['character counts'] = df[text_col].apply(lambda x: len(x)) # character counts
    df['average word length'] = df[text_col].apply(lambda x: average_word_length(x)) #avg word length
    df['stop word counts'] = df[text_col].apply(lambda x: len([t for t in x.split() if t in STOP_WORDS])) #stopword
    #df['text length'] = df['cleaned'].apply(lambda x: len(x))
    #df['mention counts'] = df['cleaned'].apply(lambda x: len([t for t in x.split() if t.startswith('@')]))
    #df['url counts'] = df['cleaned'].apply(lambda x: len([t for t in x.split() if t.startswith('http')]))
    df['number counts'] = df[text_col].apply(lambda x: len([t for t in x.split() if t.isdigit()])) # numbers
    df['upper counts'] = df[text_col].apply(lambda x: len([t for t in x.split() if t.isupper() and len(x) > 3]))
    df['email counts'] = df[text_col].apply(lambda x: len(re.findall(pattern_email, x))) #number of emails


def clean_text(text_col, preserve_syntax=False, remove_hashtags=True, stop_words=STOP_WORDS):
       
    def remove_accents(text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def make_to_base(text):
        x_list = []
        doc = nlp(text)
    
        for token in doc:
            lemma = str(token.lemma_)
            x_list.append(lemma)
        return " ".join(x_list)

    def remove_junk_words(text_col):
        all_text = " ".join(text_col)
        freq_words = pd.Series(all_text.split()).value_counts()
        words = all_text.split()
        junk_words = [word for word in words if len(word) <= 2]
        text_col = " ".join([t for t in text_col.split() if t not in junk_words])
        rare = freq_words[freq_words.values == 1]
        text_col = " ".join([t for t in text_col.split() if t not in rare])
        
        return text_col
        
    pattern_email = r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)'    

    # See about adding a period if links, mentions or hashtags are followed by a word that is capitalized.
    text = re.sub(r"(https?|ftp)\S+", '.', text_col) # Remove links
    text = " ".join([contractions.fix(word) for word in text.split()])
    text = BeautifulSoup(text, 'lxml').get_text() # Remove HTML tags
    text = html.unescape(text) # Remove HTML encoding
    text = remove_accents(text) # Remove accented characters
    text = re.sub(pattern_email, '', text)
    text = re.sub(r"@\S+", '', text) # Remove @mentions (period if capital followed?)
    if remove_hashtags == True:
        text = re.sub(r"#\S+", '', text) # Remove #hashtags
    else:
        text = re.sub(r"#", '', text) #  #hashtags
    text = " ".join(text.split()) # Remove redundant white space
    text = re.sub(r'\.+', ".", text)
    text = re.sub(r'\s([?.!"](?:\s|$))', r'\1', text) 
     #https://stackoverflow.com/questions/18878936/how-to-strip-whitespace-from-before-but-not-after-punctuation-in-python
    text = re.sub(r'^\.?', '', text)
    text = re.sub('[^A-Z a-z 0-9 .?!,]+', '', text) # Remove special characters
    text = re.sub(r'\.{2,3}', '.', text)
    text = re.sub(r'(\. \. )', '.', text)
    text = re.sub(r'\.\s\.', '.', text)
    text = re.sub(r'!\.', '!', text)
    text = re.sub(r'[.!]{2,3}', '.', text)
    text = " ".join(text.split()) # Remove redundant white space
    if preserve_syntax == True:
        return text
    else:
        text = text.lower() # Normalize capitalization
        text = re.sub('[0-9]+', '', text) # Remove numbers
        text = re.sub('[^a-z ]+', '', text)# Remove all special characters
        text = " ".join([t for t in text.split() if t not in STOP_WORDS]) #stopwords
        text = remove_junk_words(text) # Remove short/rare words
        text = " ".join(text.split()) # Remove redundant white space
        text = make_to_base(text) # Lemmatize
        text = remove_junk_words(text)
        return text
    
    
def make_wordcloud(text):
    from wordcloud import WordCloud
    all_text = " ".join(text)
    wc = WordCloud(width = 1000, height=800).generate(all_text)
    plt.imshow(wc)
    plt.axis('off')
    plt.show()


def combine_tweets(df):
    """
    takes in a twint dataframe and returns a dataframe with each row a combination of
    tweets from that day.
    """
    username = df.username
    collected_tweets = {}
    #df['time'] = pd.to_datetime(df['time'])
    #df['number of tweets'] = 1

    
    # If tweet is earlier than 9:30, it applies to that price (opening) on the same date. 

    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.date

    for i in range(len(df)):
        if df['time'].iloc[i] < dt.strptime('09:30:00', '%H:%M:%S').time():
            df['time'].iloc[i] = dt.strptime('09:30:00', '%H:%M:%S').time()
        
    # It tweet is after 9:30, but before 16:00 (closing), it applies to the following price on the same date.
    
        if (df['time'].iloc[i] > dt.strptime('09:30:00', '%H:%M:%S').time()) and (df['time'].iloc[i] < dt.strptime('16:00:00', '%H:%M:%S').time()):
            df['time'].iloc[i] = dt.strptime('16:00:00', '%H:%M:%S').time()
            
    # If tweet is after 16:00, apply it to the next opening date.
    #for i in range(len(df)):
        if df['time'].iloc[i] > dt.strptime('16:00:00', '%H:%M:%S').time():
            df['date'].iloc[i] = df['date'].iloc[i] + timedelta(days=1)
            df['time'].iloc[i] = dt.strptime('09:30:00', '%H:%M:%S').time()
            
    # Combine dates and times
    df['date'] = df['date'].astype(str)
    df['time'] = df['time'].astype(str)
    df['date'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    
    tweet = ""
    df['number of tweets'] = 1
    to_merge = df.groupby('date').sum()
    #to_merge['date'] = pd.to_datetime(to_merge['date'], format='%Y-%M-%d').dt.date
    # date is dictionary key
    collected_tweets[df['date'].iloc[0]] = tweet
   
    for i in range(len(df.index)):
        current_date = df['date'].iloc[i]
        if current_date in collected_tweets:
            collected_tweets[current_date] += " " + str(df['tweet'].iloc[i])
        else:
            collected_tweets[current_date] = str(df['tweet'].iloc[i])
            
    df = pd.DataFrame.from_dict(collected_tweets, orient='index', columns = ['tweet'])
    df.reset_index(inplace=True)
    df = df.rename(columns={'index':'date'})
    df['username'] = username
    df_merged = pd.merge(df, to_merge.reset_index(), on='date')
    
    return df_merged


def organize_stocks(stock):

        # Instatiate Open and Close
        stock_open = stock[['date','open']]
        stock_close = stock[['date','close']]

        # Convert dates to datetime objects
        stock_open['date'] = pd.to_datetime(stock_open['date'])
        stock_close['date'] = pd.to_datetime(stock_close['date'])

        # Convert datetimes into datetime string format
        stock_open['date'] = stock_open['date'].dt.strftime('%Y-%m-%d 09:30:00')
        stock_close['date'] = stock_close['date'].dt.strftime('%Y-%m-%d 16:00:00')

        # Convert strings back into datetime objects
        stock_open['date'] = pd.to_datetime(stock_open['date'])
        stock_close['date'] = pd.to_datetime(stock_close['date'])

        # Get earliest and latest stock price dates to create a date index
        stock_open['price'] = stock_open['open']
        stock_open.drop('open', axis=1, inplace=True)

        stock_close['price'] = stock_close['close']
        stock_close.drop('close', axis=1, inplace=True)

        start_date_open = dt.strftime(stock_open.reset_index().date.min(), '%Y-%m-%d %H:%M:%S')
        end_date_open = dt.strftime(stock_open.reset_index().date.max(), '%Y-%m-%d %H:%M:%S')

        start_date_close = dt.strftime(stock_close.reset_index().date.min(), '%Y-%m-%d %H:%M:%S')
        end_date_close = dt.strftime(stock_close.reset_index().date.max(), '%Y-%m-%d %H:%M:%S')

        date_indx_open = pd.date_range(start_date_open, end_date_open).tolist()
        date_indx_close = pd.date_range(start_date_close, end_date_close).tolist()
        date_indx_open = pd.Series(date_indx_open, name='date')
        date_indx_close = pd.Series(date_indx_close, name='date')

        # Merge date index onto stock dataframes
        stock_open = pd.merge(date_indx_open, stock_open, how='left')
        stock_close = pd.merge(date_indx_close, stock_close, how='left')

        # Interpolate missing values
        stock_open['price'].interpolate(method='linear', inplace=True)
        stock_close['price'].interpolate(method='linear', inplace=True)

        # Reset index and join open and close dataframes together
        stock_open.set_index('date', inplace=True)
        stock_close.set_index('date', inplace=True)

        stock = pd.concat([stock_open, stock_close])
        
        stock.sort_index(inplace=True)
        
        return stock
    
def plot_univariate_dist(df, feature, color, plot_mean = True, plot_std = False, std_lab=False, kde = True, stat='count', truncate=True, auto_bins=False, spearmanr=False):

    mean = df[feature].mean()
    std = df[feature].std()
    bins = round(np.sqrt(len(df)))
    max_height = test = pd.cut(df[feature], bins = bins).value_counts()[0]
    min_val = df[feature].min()
    
    if kde == True:
        sns.histplot(df[feature], bins=bins, color=color, kde=True, edgecolor="black", line_kws = {'lw':'4'}, stat=stat, alpha=0.55)
    else:
        sns.histplot(df[feature], bins=bins, color=color, stat=stat, alpha=0.5, edgecolor='black')
    
    if plot_mean == True:
        plt.axvline(x=mean, linestyle = '--',color='k', alpha=0.5)
    if plot_std == True:    
        plt.axvline(x= std + mean, linestyle='-.', color = 'r', alpha=0.5)
        plt.axvline(x= 2*std + mean, linestyle='--', color = 'r', alpha=0.5)
        
        plt.axvline(x= -std + mean, linestyle = '-.', color='r', alpha=0.5)
        plt.axvline(x= -2*std + mean, linestyle = '--', color='r', alpha=0.5)
    if std_lab == True:
        plt.text(std + mean, max_height, '+ 1 SD', fontsize=20)
        plt.text(2*std + mean, max_height, '+ 2 SD', fontsize=20)
       
        plt.text(-2*std + mean, max_height, '- 2 SD', fontsize=20)
        plt.text(-std + mean, max_height, '- 1 SD', fontsize=20)
    if truncate == True:
        if min_val < 0:
            plt.xlim(-3.75*std, 3.75*std)
        else:
            plt.xlim(-1*std, 3.75*std)
        
    plt.ylim(0, max_height + (max_height * 0.5))
    if spearmanr == True:
        x = df['percent change']
        #x.replace('rise', 2, inplace=True)
        #x.replace('no change', 1, inplace=True)
        #x.replace('fall', 0, inplace=True)
        y = df[feature]
        r, p = stats.spearmanr(x, y)
        plt.text(std*0.5 + mean, max_height + max_height*.1, 'Spearman_r: {:.3f}, p: {:.3f}'.format(r,p), fontsize=12)
        
def plot_bivariate_dist(df, feature):

    fig, ax = plt.subplots(3,1, sharex=True)
    sns.boxplot(df[df['bins'] == 'drop'][feature], color='red', ax=ax[0])
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Drop')
    sns.boxplot(df[df['bins'] == 'no change'][feature], color='orange', ax=ax[1])
    ax[1].set_xlabel('')
    ax[1].set_ylabel('No Change')
    sns.boxplot(df[df['bins'] == 'rise'][feature], color='green', ax=ax[2])
    ax[2].set_ylabel('Rise')
    
def data_table(df, feature):
    
    std = df[feature].std()

    data_table_indx = ['count', 'null_vals', 'btwn_0_1_std', 'btwn_1_2_std', 'over_2_std']
    data_table = pd.DataFrame(index = data_table_indx, columns=['counts', 'pct_total'])

    data_table['counts']['count'] = len(df[feature])
    data_table['pct_total']['count'] = round(data_table['counts']['count'] / len(df), 3)

    null_vals = len(df[feature][df[feature] == 0])
    data_table['counts']['null_vals'] = null_vals
    data_table['pct_total']['null_vals'] = round(null_vals / data_table['counts']['count'], 3)

    witin = (df[feature] >= -std) & (df[feature] <= std)
    sum_within = len(df[feature][witin])
    data_table['counts']['btwn_0_1_std'] = sum_within
    data_table['pct_total']['btwn_0_1_std'] = round(sum_within / data_table['counts']['count'], 3)

    within_2 = (df[feature] >= -2*std) & (df[feature] <= 2*std)
    sum_within_2 = len(df[feature][within_2]) - sum_within
    data_table['counts']['btwn_1_2_std'] = sum_within_2
    data_table['pct_total']['btwn_1_2_std'] = round(sum_within_2 / data_table['counts']['count'], 3)

    over_2 = len(df[feature]) - (sum_within_2 + sum_within)
    data_table['counts']['over_2_std'] = over_2
    data_table['pct_total']['over_2_std'] = round(over_2 / data_table['counts']['count'], 3)
    assert data_table['counts']['count'] == data_table['counts'][['btwn_0_1_std', 'btwn_1_2_std', 'over_2_std']].sum()
    
    return data_table

def display_side_by_side(*args,titles=cycle([''])):
    """https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side"""
    html_str=''
    for df,title in zip(args, chain(titles,cycle(['</br>'])) ):
        html_str+='<th style="text-align:center"><td style="vertical-align:top">'
        html_str+=f'<h2>{title}</h2>'
        html_str+=df.to_html().replace('table','table style="display:inline"')
        html_str+='</td></th>'
    display_html(html_str,raw=True)
  
def emoji_counts(text):
    return(emoji.emoji_count(text)) #emoji.emoji_lis(text))

def extract_text_features(df, text):
    
    def num_sentences(text):
        sentences = sent_tokenize(text)
        num_sentences = len(sentences)
        return num_sentences
    
    def num_words(text):
        cleaned = text.lower()
        cleaned = re.sub(r'[^A-Za-z ]+', '', cleaned)
        return len(cleaned.split())

    def avg_words_per_sentence(text):
        sent_length = 0 
        sentences = sent_tokenize(text)
        for sentence in sentences:
            sent_length += len(sentence.split())
        return sent_length / len(sentences)

    def noun_counts(text):
        doc = nlp(text)
        pos_counts = 0
        tokens = []
        for token in doc:
            if token.pos_ == 'NOUN':
                pos_counts += 1
                tokens.append(token.text)
        return pos_counts, tokens

    def verb_counts(text):
        doc = nlp(text)
        pos_counts = 0
        tokens = []
        for token in doc:
            if token.pos_ == 'VERB':
                pos_counts += 1
                tokens.append(token.text)
        return pos_counts, tokens

    def stopword_counts(text):
        doc = nlp(text)
        stop_counts = 0
        stop_words = []
        for token in doc:
            if token.is_stop:
                stop_counts += 1
                stop_words.append(token.text)
        return stop_counts, stop_words

    def org_counts(text):
        doc = nlp(text)
        org_counts = 0
        orgs = []
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                org_counts += 1
                orgs.append(ent)
        return org_counts, orgs
    
    df['number of sentences'] = df[text].apply(lambda x: num_sentences(x))
    df['number of words'] = df[text].apply(lambda x: num_words(x))
    df['avg words per sentence'] = df[text].apply(lambda x: avg_words_per_sentence(x))
    count = df[text].apply(lambda x: noun_counts(x))
    df['number of nouns'] = count[0][0]
    count = df[text].apply(lambda x: verb_counts(x))
    df['number of verbs'] = count[0][0]
    count = df[text].apply(lambda x: stopword_counts(x))
    df['number of stopwords'] = count[0][0]
    count = df[text].apply(lambda x: org_counts(x))
    df['number of organizations'] = count[0][0]
    
    return df

def text_to_matrix(df, kind='tfidf', ngram_range=(1,1), min_df=1, feature_names=False):
    """ 
    Returns sparse matrix of passed Series or Dataframe. If feature_names=True, returns matrix, feature_names.
    """
    if kind == 'tfidf':
        vect = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range)
        
    if kind == 'count':   
        vect = CountVectorizer(min_df=min_df, ngram_range=ngram_range)
    
    matrix = vect.fit_transform(df)
    
    if feature_names:
        return matrix, vect.get_feature_names()
    else:
        return matrix     
    
def frequency_table(text):
    ents_counts = {}
    doc = nlp(text)
    for ent in doc.ents:
        key = str(ent)
        if key in ents_counts.keys():
            ents_counts[key][0] += 1
        else:   
            ents_counts[key] = [1, str(ent.label_)]
    df = pd.DataFrame.from_dict(ents_counts, orient='index', columns=['count', 'label'])
    return df

def plot_freq_table(table, limit=25, user=None):
    if user:
        user = user
    else:
        user = ''
    table = table.sort_values(by='count', ascending=False)
    sns.set()
    #plt.figure(figsize=(18,10))
    table = table[:limit]
    sns.barplot(table['count'].values, table.index, hue=table['label'].values, dodge=False)
    plt.title('{} Frequency Plot of Word Entities'.format(user))
    plt.xlabel('Counts')
    plt.ylabel('Entity')

def reduce_dimensions(df, n_components=None, method='pca', svd_solver='auto', elbow_plot=False):
    
    """n_components is either an int for number of dimensions to reduce to,
    or a float between 0-1 for explained variance to keep, or 'mle' for Maximum Likelihood Estimation.
    Default is None to keep all components. Returns reduced array, model, cumsum, and number of components."""
    
    decomp = PCA(n_components = n_components, svd_solver=svd_solver)
    reduced = decomp.fit_transform(df)
    cumsum = np.cumsum(decomp.explained_variance_ratio_)
    if type(n_components) == float:
        number_components = np.argmax(cumsum >= n_components) + 1
    else:
        number_components = n_components
    if elbow_plot:
        cumsum = np.cumsum(decomp.explained_variance_ratio_)
        plt.plot(cumsum)
        plt.title('Explained Variance to Dimensions Plot')
        plt.xlabel('Dimensions')
        plt.ylabel('Explained Variance')

    return reduced, decomp, cumsum, number_components

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Hands-on Machine Learning with Scikit-Learn, Keras & TensofFlow
    """
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.legend()
    plt.xlabel('thresholds')
    plt.ylabel('score')
    #plt.hlines()
    
def plot_learning_curves(model, X, y):
    """Hands-on Machine Learning with Scikit-Learn, Keras & TensofFlow"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_errors, test_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_test_predict = model.predict(X_test)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        tes_errors.append(mean_squared_error(y_test, y_test_predict))
    plt.plot(np.sqrt(train_errors), 'r-+', linewidth = 2, label='train')
    plt.plot(np.sqrt(test_errors), 'b-', linewidth = 3, label='test')
    
def full_report(clf, X_train, X_test, y_train, y_test, pos_label):
    
    y_pred = clf.predict(X_test)
    scores = clf.score(X_test, y_test)
    cv_result = cross_validate(clf, X_train, y_train, cv=5)
    scores = cv_result["test_score"]
    print("The mean cross-validation accuracy is: "
          f"{scores.mean():.3f} +/- {scores.std():.3f}")
    print('Precision: {:.3f}'.format(precision_score(y_test, y_pred, pos_label=pos_label)))
    print('Recall: {:.3f}'.format(recall_score(y_test, y_pred, pos_label=pos_label)))
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced accuracy: {balanced_accuracy:.3f}")
    print(classification_report(y_test, y_pred))

    sns.set()
    plot_precision_recall_curve(clf, X_test, y_test, pos_label=pos_label)
    plt.show()

    plot_roc_curve(clf, X_test, y_test, pos_label=pos_label)
    from sklearn.dummy import DummyClassifier
    dummy_classifier = DummyClassifier(strategy="most_frequent")
    dummy_classifier.fit(X_train, y_train)
    y_score = dummy_classifier.predict_proba(X_test)
    fpr, tpr, thresh = metrics.roc_curve(y_test, y_score[:,0], pos_label=pos_label)
    auc = metrics.roc_auc_score(y_test, y_score[:,0])
    plt.plot(fpr,tpr,label="data 2, auc="+str(auc), linestyle='--')
    plt.show()

    conf_mat = plot_confusion_matrix(clf, X_test, y_test, cmap='Blues')
    plt.grid(False)
    plt.show()
    
def extract_key_sentences(text, key_word):
    """Uses nltk's sent_tokenize method, so text data should contain hard stops (e.g. punctuation)"""
    text = text.lower()
    sentences = sent_tokenize(text)
    key_sentences = []
    for sentence in sentences:
        if key_word in sentence:
            key_sentences.append(sentence)
    return " ".join(key_sentences)

def cross_val(X, y, user, clf_list, target_name, pos_label):
    all_scores = pd.DataFrame()
    for clf in clf_list:
        scoring = {'balanced_accuracy': make_scorer(balanced_accuracy_score),
                   'precision': make_scorer(precision_score, average='weighted'),
                   'recall': make_scorer(recall_score, average='weighted'),
                   'f1': make_scorer(f1_score, average='weighted'),
                   'roc_auc': make_scorer(roc_auc_score, needs_proba=True, average='weighted')
                   }
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        scores = cross_validate(clf, X, y, scoring=scoring, error_score='raise')
        scores_df = pd.DataFrame(scores)
        models_df = pd.DataFrame(index=[0])
        models_df['username'] = user
        models_df['model'] = clf
        for col in scores_df.describe():
            mean = scores_df.describe()[col]['mean']
            std = scores_df.describe()[col]['std']
            models_df['{}_mean'.format(col)] = mean
            models_df['{}_std'.format(col)] = std
        all_scores = all_scores.append(models_df)
        all_scores['n_samples'] = len(y)
        all_scores['n_pos_label'] = len(y[y[target_name] == pos_label])
    return all_scores

def most_predictive_words(text, target, min_df=0):
    """http://localhost:8889/lab/tree/Mini_Project_Naive_Bayes.ipynb"""
    
    vectorizer = CountVectorizer(min_df=min_df)
    X = vectorizer.fit_transform(text)
    y = target
    clf = MultinomialNB().fit(X, y)

    words = np.array(vectorizer.get_feature_names())
    x = np.eye(X.shape[1])
    for i in range(len(clf.classes_)):
        probs = clf.predict_log_proba(x)[:, i]
        ind = np.argsort(probs)

        good_words = words[ind[:10]]
        bad_words = words[ind[-10:]]

        good_prob = probs[ind[:10]]
        bad_prob = probs[ind[-10:]]

        print("\n Most Predictive Words for bin: {}\n     ".format(clf.classes_[i]))
        for w, p in zip(good_words, good_prob):
            print("{:>20}".format(w), "{:.2f}".format(1 - np.exp(p)))

        print("\n Least Predictive Words for bin: {}\n     ".format(clf.classes_[i]))
        for w, p in zip(bad_words, bad_prob):
            print("{:>20}".format(w), "{:.2f}".format(1 - np.exp(p)))
    print('\n')

def most_important_features(model, X_val, y_val, feature_list):

    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X_val, y_val,
                                n_repeats=30,
                                random_state=0)
    print("Important Features:\n")
    for i in r.importances_mean.argsort()[::-1]:
         if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print("{:>25}".format(feature_list[i]),
                   f"{r.importances_mean[i]:.3f}"
                   f" +/- {r.importances_std[i]:.3f}")


# Functions

def emoji_counts(text):
    return(emoji.emoji_count(text)) #emoji.emoji_lis(text))

def extract_text_features(text):
    
    df = pd.DataFrame()
    df['text'] = text
    
def num_sentences(text):
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    return num_sentences

def num_words(text):
    cleaned = text.lower()
    cleaned = re.sub(r'[^A-Za-z ]+', '', cleaned)
    return len(cleaned.split())

def avg_words_per_sentence(text):
    sent_length = 0 
    sentences = sent_tokenize(text)
    for sentence in sentences:
        sent_length += len(sentence.split())
    return sent_length / len(sentences)

def noun_counts(text):
    doc = nlp(text)
    pos_counts = 0
    tokens = []
    for token in doc:
        if token.pos_ == 'NOUN':
            pos_counts += 1
            tokens.append(token.text)
    return pos_counts, tokens

def verb_counts(text):
    doc = nlp(text)
    pos_counts = 0
    tokens = []
    for token in doc:
        if token.pos_ == 'VERB':
            pos_counts += 1
            tokens.append(token.text)
    return pos_counts, tokens

def stopword_counts(text):
    doc = nlp(text)
    stop_counts = 0
    stop_words = []
    for token in doc:
        if token.is_stop:
            stop_counts += 1
            stop_words.append(token.text)
    return stop_counts, stop_words

def org_counts(text):
    doc = nlp(text)
    org_counts = 0
    orgs = []
    for ent in doc.ents:
        if ent.label_ == 'ORG':
            org_counts += 1
            orgs.append(ent)
    return org_counts, orgs

def text_to_matrix(df, kind='tfidf', ngram_range=(1,1), min_df=1, feature_names=False):
    """ 
    Returns sparse matrix of passed Series or Dataframe. If feature_names=True, returns matrix, feature_names.
    """
    if kind == 'tfidf':
        vect = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range)
        
    if kind == 'count':   
        vect = CountVectorizer(min_df=min_df, ngram_range=ngram_range)
    
    matrix = vect.fit_transform(df)
    
    if feature_names:
        return matrix, vect.get_feature_names()
    else:
        return matrix     
    
def frequency_table(text):
    ents_counts = {}
    doc = nlp(text)
    for ent in doc.ents:
        key = str(ent)
        if key in ents_counts.keys():
            ents_counts[key][0] += 1
        else:   
            ents_counts[key] = [1, str(ent.label_)]
    df = pd.DataFrame.from_dict(ents_counts, orient='index', columns=['count', 'label'])
    return df

def plot_freq_table(table, limit=25, user=None):
    if user:
        user = user
    else:
        user = ''
    table = table.sort_values(by='count', ascending=False)
    sns.set()
    #plt.figure(figsize=(18,10))
    table = table[:limit]
    sns.barplot(table['count'].values, table.index, hue=table['label'].values, dodge=False)
    plt.title('{} Frequency Plot of Word Entities'.format(user))
    plt.xlabel('Counts')
    plt.ylabel('Entity')

def reduce_dimensions(df, n_components=None, method='pca', svd_solver='auto', elbow_plot=False):
    
    """n_components is either an int for number of dimensions to reduce to,
    or a float between 0-1 for explained variance to keep, or 'mle' for Maximum Likelihood Estimation.
    Default is None to keep all components. Returns reduced array, model, cumsum, and number of components."""
    
    decomp = PCA(n_components = n_components, svd_solver=svd_solver)
    reduced = decomp.fit_transform(df)
    cumsum = np.cumsum(decomp.explained_variance_ratio_)
    if type(n_components) == float:
        number_components = np.argmax(cumsum >= n_components) + 1
    else:
        number_components = n_components
    if elbow_plot:
        cumsum = np.cumsum(decomp.explained_variance_ratio_)
        plt.plot(cumsum)
        plt.title('Explained Variance to Dimensions Plot')
        plt.xlabel('Dimensions')
        plt.ylabel('Explained Variance')

    return reduced, decomp, cumsum, number_components

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Hands-on Machine Learning with Scikit-Learn, Keras & TensofFlow
    """
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.legend()
    plt.xlabel('thresholds')
    plt.ylabel('score')
    #plt.hlines()
    
def plot_learning_curves(model, X, y):
    """Hands-on Machine Learning with Scikit-Learn, Keras & TensofFlow"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_errors, test_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_test_predict = model.predict(X_test)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        tes_errors.append(mean_squared_error(y_test, y_test_predict))
    plt.plot(np.sqrt(train_errors), 'r-+', linewidth = 2, label='train')
    plt.plot(np.sqrt(test_errors), 'b-', linewidth = 3, label='test')
    
def full_report(clf, X_train, X_test, y_train, y_test, show_metrics=False, class_label=1):
    
    y_pred = clf.predict(X_test)
    scores = clf.score(X_test, y_test)
    cv_result = cross_validate(clf, X_train, y_train, cv=5)
    scores = cv_result["test_score"]
    print("The mean cross-validation accuracy is: "
          f"{scores.mean():.3f} +/- {scores.std():.3f}")
    print('Precision: {:.3f}'.format(precision_score(y_test, y_pred)))
    print('Recall: {:.3f}'.format(recall_score(y_test, y_pred)))
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced accuracy: {balanced_accuracy:.3f}")
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, show_metrics=show_metrics)

    sns.set()
    plot_precision_recall_curve(clf, X_test, y_test)
    plt.show()

    plot_roc_curve(clf, X_test, y_test)
    from sklearn.dummy import DummyClassifier
    dummy_classifier = DummyClassifier(strategy="most_frequent")
    dummy_classifier.fit(X_train, y_train)
    y_score = dummy_classifier.predict_proba(X_test)
    fpr, tpr, thresh = metrics.roc_curve(y_test, y_score[:,class_label])
    auc = metrics.roc_auc_score(y_test, y_score[:,class_label])
    plt.plot(fpr,tpr,label="data 2, auc="+str(auc), linestyle='--')
    plt.show()
    
    
def extract_key_sentences(text, key_word):
    """Uses nltk's sent_tokenize method, so text data should contain hard stops (e.g. punctuation)"""
    text = text.lower()
    sentences = sent_tokenize(text)
    key_sentences = []
    for sentence in sentences:
        if key_word in sentence:
            key_sentences.append(sentence)
    return " ".join(key_sentences)

def cross_val(X, y, user, clf_list, target_name):
    all_scores = pd.DataFrame()
    for clf in clf_list:
        scoring = {'balanced_accuracy': make_scorer(balanced_accuracy_score),
                   'precision': make_scorer(precision_score),
                   'recall': make_scorer(recall_score),
                   'f1': make_scorer(f1_score),
                   'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
                   }
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        scores = cross_validate(clf, X, y, scoring=scoring, error_score='raise')
        scores_df = pd.DataFrame(scores)
        models_df = pd.DataFrame(index=[0])
        models_df['username'] = user
        models_df['model'] = clf
        for col in scores_df.describe():
            mean = scores_df.describe()[col]['mean']
            std = scores_df.describe()[col]['std']
            models_df['{}_mean'.format(col)] = mean
            models_df['{}_std'.format(col)] = std
        all_scores = all_scores.append(models_df)
        all_scores['n_samples'] = len(y)
        all_scores['n_pos_label'] = len(y[y[target_name] == 1]) 
    return all_scores

def most_predictive_words(clf, text, target, min_df=0):
    """http://localhost:8889/lab/tree/Mini_Project_Naive_Bayes.ipynb"""
    
    vectorizer = CountVectorizer(min_df=min_df)
    X = vectorizer.fit_transform(text)
    y = target
    clf.fit(X, y)

    words = np.array(vectorizer.get_feature_names())
    x = np.eye(X.shape[1])
    for i in range(len(clf.classes_)):
        probs = clf.predict_proba(x)[:, i]
        ind = np.argsort(probs)

        good_words = words[ind[-10:]]

        good_prob = probs[ind[-10:]]

        print("\n Most Predictive Words for bin: {}\n     ".format(clf.classes_[i]))
        for w, p in zip(good_words, good_prob):
            print("{:>20}".format(w), "{:.2f}".format(p))
    print('\n')

def most_important_features(model, X_val, y_val, feature_list):

    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X_val, y_val,
                                n_repeats=30,
                                random_state=0)
    print("Important Features:\n")
    for i in r.importances_mean.argsort()[::-1]:
         if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print("{:>25}".format(feature_list[i]),
                   f"{r.importances_mean[i]:.3f}"
                   f" +/- {r.importances_std[i]:.3f}")
            
def cross_validation_scores(original_df, target_name, numerical_feat, text_name, clf_list):

    scores = pd.DataFrame()
    for user in list(original_df['username'].unique()):
        df = original_df[original_df['username'] == user]
        y = df[[target_name]]
        num_data = df[numerical_feat]
        text_data = df[text_name]
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(num_data)
        scaled_numerical_data = scaler.transform(num_data)

        vectorizer = TfidfVectorizer()
        vectorizer.fit(text_data)
        sparse_text_matrix = vectorizer.transform(text_data)

        X_num = pd.DataFrame(sparse_text_matrix.toarray(), columns=vectorizer.get_feature_names())
        X_text = pd.DataFrame(scaled_numerical_data, columns=num_data.columns)
        X = pd.concat([X_num, X_text], axis=1)

        # Separate into Training/Testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y)


        scores = scores.append(cross_val(X_train, y_train, user, clf_list, target_name=target_name),ignore_index=True)
        
    return scores

def examine_model(original_df, scores, scorer, target_name, numerical_feat, pos=0, class_label=1, reduce=False):
    
    scorer_mean = scorer + "_mean"
    scorer_std = scorer + "_std"
    mean = scores.sort_values(by=scorer_mean, ascending=False).reset_index(drop=True)[scorer_mean][pos]
    std = scores.sort_values(by=scorer_mean, ascending=False).reset_index(drop=True)[scorer_std][pos]
    clf = scores.sort_values(by=scorer_mean, ascending=False).reset_index(drop=True)['model'][pos]
    user = scores.sort_values(by=scorer_mean, ascending=False).reset_index(drop=True)['username'][pos]

    target_name = 'classification'
    text_name = 'cleaned'
    df = original_df[original_df['username'] == user].copy(deep=True)
    df = df[df[target_name].notnull()]
    y = df[[target_name]]
    num_data = df[numerical_feat]
    text_data = df[text_name]

    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(num_data)
    scaled_numerical_data = scaler.transform(num_data)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(text_data)
    sparse_text_matrix = vectorizer.transform(text_data)
    
    if reduce:
        pca = PCA(n_components='mle')
        reduced_numerical_data = pca.fit_transform(scaled_numerical_data)
        
        components = pd.DataFrame(pca.components_, columns=numerical_feat)
        numerical_feat = components.T[0][np.abs(components.T[0]) > 0.01].index
        
        num_data = df[numerical_feat]
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(num_data)
        scaled_numerical_data = scaler.transform(num_data)
        
        pca = PCA(n_components='mle')
        reduced_numerical_data = pca.fit_transform(scaled_numerical_data)
        
        n_components = sparse_text_matrix.shape[1] - 1
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(sparse_text_matrix)
        reduced_text_matrix = svd.transform(sparse_text_matrix)

        threshold = 0.95
        cumsum = np.cumsum(svd.explained_variance_ratio_)
        text_dimensions = np.where(cumsum > threshold)[0][0]
        
        svd = TruncatedSVD(n_components=text_dimensions)
        reduced_text_matrix = svd.fit_transform(sparse_text_matrix)
        
        X_text = pd.DataFrame(reduced_text_matrix)
        X_num = pd.DataFrame(reduced_numerical_data)
    else:
        X_text = pd.DataFrame(sparse_text_matrix.toarray(), columns=vectorizer.get_feature_names())
        X_num = pd.DataFrame(scaled_numerical_data, columns=num_data.columns)

    X = pd.concat([X_num, X_text], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y)
    clf.fit(X_train, y_train)
    print(user)
    print('{}: {:.3f}'.format(scorer_mean.capitalize(), mean))
    print('{}: {:.3f}'.format(scorer_std.capitalize(), std))
    print('Range of {}: {:.3f} -- {:.3f}'.format(scorer_mean, mean-std, mean+std))
    print(clf)
    full_report(clf, X_train, X_test, y_train, y_test)
    y_proba = clf.predict_proba(X_test)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba[:,class_label])
    plt.figure(figsize=[8,5])
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    return X_train, X_test, y_train, y_test
   


def plot_confusion_matrix(y_true, y_pred, show_metrics=True, beta=2):
    mat = confusion_matrix(y_true, y_pred)
    #https://vitalflux.com/python-draw-confusion-matrix-matplotlib/
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(mat, cmap=plt.cm.Blues, alpha=0.7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(x=j, y=i,s=mat[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.grid(False)
    plt.show()
    
    TP = mat[1][1]
    TN = mat[0][0]
    FP = mat[0][1]
    FN = mat[1][0]
    total = TP + TN + FP
    
    if show_metrics == True:
        # Metrics
        acc = TP + TN / mat.sum()
        err_rate = 1 - acc
        tpr = TP / (TP + FN)
        fpr = FP / (TN + FP)
        tnr = TN / (TN + FP)
        fnr = FN / (TP + FN)
        prec = TP / (TP + FP)
        f1 = (2*(prec*tpr)) / (prec + tpr)
        f2_beta = (1 + beta**2) * ((prec*tpr) / ((beta**2 * prec) + tpr))
        
        
        print("  Confusion Matrix Metrics")
        print("===========================")
        print('  How often is the classifier correct?')
        print('  Accuracy: {:.3f}%'.format(acc))
        print()
        print('  How often is it wrong?')
        print('  Error Rate: {:.3f}%'.format(err_rate))
        print()
        print("  When it's actually yes, how often does it predict yes?")
        print('  True Positive Rate (Recall): {:.3f}%'.format(tpr*100))
        print()
        print("  When it predicts yes, how often is it correct?")
        print("  Precision: {:.3f}%".format(prec*100))
        print()
        print("  When it's actually no, how often does it predict yes?")
        print('  False Positive Rate: {:.3f}%'.format(fpr*100))
        print()
        print("  When it's actually no, how often does it predict no?")
        print("  True Negative Rate (Specificity): {:.3f}%".format(tnr*100))
        print("===========================")
        print("  F1 Score: {:.3f}".format(f1))
        print('  Weighted F1 Score (beta = {}): {:.3f}'.format(beta, f2_beta))
        print('  G-mean: {:.3f}'.format(np.sqrt(tpr*prec)))

    else:
        return
    
def text_prob_visual(clf, train_data, test_data, class_names, text_idx, num_features):
    idx = text_idx
    vectorizer = TfidfVectorizer(lowercase=False)
    train_vectors = vectorizer.fit_transform(train_data['cleaned'])
    test_vectors = vectorizer.transform(test_data['cleaned'])
    y_train= train_data['classification']
    y_test = test_data['classification']
    clf.fit(train_vectors, y_train)
    pred = clf.predict(test_vectors)
    f1_score(y_test, pred, average='binary')
    c = make_pipeline(vectorizer, clf)
    print(c.predict_proba([test_data['cleaned'].iloc[0]]))
    #class_names = ['no change', 'change']
    explainer = LimeTextExplainer(class_names=class_names)
    
    exp = explainer.explain_instance(test_data['cleaned'][idx], c.predict_proba, num_features=num_features)
    print('Document id: %d' % idx)
    print('Percent Change: {:.3f}'.format(test_data['percent change'][idx]))
    print('Probability(Change) = {:.3f}'.format(c.predict_proba([test_data['text'][idx]])[0,1]))
    print('True class: %s' % class_names[y_test[idx]])
    exp.show_in_notebook(text=True)

def chi2_sig(df, category_names, numerical_feat, text_col, p_value_limit=0.95): 
    #https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794
    print('Statistically Significant Words Per Bin')
    print('P_val = {}'.format(p_value_limit))
    print("========================================")
    print("")
    cat = (df[category_names].unique)
    y = df[category_names]
    text = df[text_col]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(text)
    X_train = vectorizer.transform(text)
    X_names = vectorizer.get_feature_names()
    dtf_features = pd.DataFrame()
    for cat in np.unique(y):
        chi2, p = feature_selection.chi2(X_train, y==cat)
        dtf_features = dtf_features.append(pd.DataFrame(
                       {"feature":X_names, "score":1-p, "y":cat}))
        dtf_features = dtf_features.sort_values(["y","score"], 
                        ascending=[True,False])
        dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
    X_names = dtf_features["feature"].unique().tolist()
    for cat in np.unique(y):
       print("# {}:".format(cat))
       print("  . selected features:",
             len(dtf_features[dtf_features["y"]==cat]))
       print("  . top features:", ",".join(
    dtf_features[dtf_features["y"]==cat]["feature"].values[:10]))
       print(" ")
    print("========================================")   

        
    # Numerical Features
    print('Statistically Significant Features Per Bin')
    print('P_val = {}'.format(p_value_limit))
    print("========================================")
    print("")
    cat = (df[category_names].unique)
    cat = (df[category_names].unique)
    y = df[category_names]
    features = df[numerical_feat]
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(features)
    X_train = scaler.transform(features)

    X_names = numerical_feat
    dtf_features = pd.DataFrame()
    for cat in np.unique(y):
        chi2, p = feature_selection.chi2(X_train, y==cat)
        dtf_features = dtf_features.append(pd.DataFrame(
                       {"feature":X_names, "score":1-p, "y":cat}))
        dtf_features = dtf_features.sort_values(["y","score"], 
                        ascending=[True,False])
        dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
    X_names = dtf_features["feature"].unique().tolist()
    for cat in np.unique(y):
       print("# {}:".format(cat))
       print("  . selected features:",
             len(dtf_features[dtf_features["y"]==cat]))
       print("  . top features:", ",".join(
    dtf_features[dtf_features["y"]==cat]["feature"].values[:10]))
       print(" ")
    print("========================================")

def feat_prob_visual(clf, train_data, test_data, class_names, features,
                     idx_id, num_features):

    
    indx_l = list(test_data.index)
    idx = indx_l.index(idx_id)
    
    X_train = train_data[features]
    X_test = test_data[features]
    y_train = train_data['classification']
    y_test = test_data['classification']
    #clf = lgr_cv.best_estimator_
    #clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    

    lime_explainer = lime_tabular.LimeTabularExplainer(
        X_train.values,
        training_labels=y_train.values,
        feature_names=X_train.columns.tolist(),
        feature_selection="lasso_path",
        class_names=class_names,
        discretize_continuous=True,
        discretizer="entropy",
    )

    exp = lime_explainer.explain_instance(X_test.iloc[idx], clf.predict_proba, num_features=num_features)

    exp.show_in_notebook(show_table=True)

def text_prob_visual(clf, train_data, test_data, class_names, idx_id, num_features):
    indx_l = list(test_data.index)
    idx = indx_l.index(idx_id)
    y_train= train_data['classification']
    y_test = test_data['classification']
    vectorizer = TfidfVectorizer(lowercase=False)
    train_vectors = vectorizer.fit_transform(train_data['cleaned'])
    test_vectors = vectorizer.transform(test_data['cleaned'])
    clf.fit(train_vectors, y_train)
    pred = clf.predict(test_vectors)
    f1_score(y_test, pred, average='binary')
    c = make_pipeline(vectorizer, clf)
    #print(c.predict_proba([test_data['cleaned'].iloc[0]]))
    #class_names = ['no change', 'change']
    explainer = LimeTextExplainer(class_names=class_names)
    
    exp = explainer.explain_instance(test_data['cleaned'].iloc[idx], c.predict_proba, num_features=num_features)
    print('Document id: %d' % idx_id)
    print('Percent Change: {:.3f}'.format(test_data['percent change'].iloc[idx]))
    print('Probability(Change) = {:.3f}'.format(c.predict_proba([test_data['text'].iloc[idx]])[0,1]))
    print('True class: %s' % class_names[y_test.iloc[idx]])
    exp.show_in_notebook(text=True)
    
def interpret_instance(clf, train_data, test_data, class_names, idx, features, num_features):
    
    
    text_prob_visual(clf, train_data, test_data, class_names, idx, num_features)
    feat_prob_visual(clf, train_data, test_data, class_names, features,
                     idx, num_features)

def elbow_plot(cumsum, threshold):
    plt.plot(cumsum)
    plt.title('Explained Variance to Dimensions Plot')
    plt.xlabel('Dimensions')
    plt.ylabel('Explained Variance')
    dim_threshold = np.where(cumsum >= threshold)[0][0]
    threshold = cumsum[dim_threshold]
    plt.hlines(y = threshold, xmin=0, xmax=dim_threshold, colors = 'red', linestyles='--')
    plt.vlines(x = dim_threshold, ymin=cumsum[0], ymax=threshold, colors = 'red', linestyles='--')
    plt.scatter(x=dim_threshold, y=threshold)
    plt.annotate('{:.2f}'.format(threshold), (dim_threshold, threshold))
    plt.show()
   
def get_index(text_df, target_word):
    for i in range(len(text_df)):
        tweet = text_df.iloc[i]
        if target_word in tweet.split():
            i = text_df.index[i]
            return i
      
def plot_threshold(precisions, recalls, thresholds, clf, recall_thresh=0.5):
    thresh_ind = np.where(recalls > recall_thresh)[0][-1]
    rec_thresh = recalls[thresh_ind]
    prec_thresh = precisions[thresh_ind]
    thresh = thresholds[thresh_ind]
    print('Treshold: {:.3f}'.format(thresh))
    plt.figure(figsize=(7,5))
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.vlines(x=thresh, ymin = 0.0, ymax = rec_thresh, colors='red', linestyles='--', alpha=0.5)
    plt.hlines(y=rec_thresh, xmin=thresholds.min(), xmax=thresh, colors='red', linestyles='--', alpha=0.5)
    plt.hlines(y=prec_thresh, xmin=thresholds.min(), xmax=thresh, colors='red', linestyles='--', alpha=0.5)
    plt.scatter(x=thresh, y=rec_thresh)
    plt.scatter(x=thresh, y=prec_thresh)
    plt.show()
    return thresh


def make_xy(df, target_name, numerical_feat, text_name, min_df=1, ngram_range=(1,1), decomp=False):
    
    
    y = df[[target_name]]
    num_data = df[numerical_feat]
    text_data = df[text_name]
    
    # Vectorize
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)
    vectorizer.fit(text_data)
    sparse_text_matrix = vectorizer.transform(text_data)
    
    # Reduce
    n_components = sparse_text_matrix.shape[1] - 1
    n_text_features = sparse_text_matrix.shape[1]
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(sparse_text_matrix)
    reduced_text_matrix = svd.transform(sparse_text_matrix)

    threshold = 0.95
    cumsum = np.cumsum(svd.explained_variance_ratio_)
    text_dimensions = np.where(cumsum > threshold)[0][0]
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_numerical_data = scaler.fit_transform(num_data)

    pca = PCA(n_components='mle')
    reduced_numerical_data = pca.fit(scaled_numerical_data).transform(scaled_numerical_data)

    threshold = 0.95
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    numfeat_dimensions = np.where(cumsum > threshold)[0][0]
    
    components = pd.DataFrame(pca.components_, columns=numerical_feat)
    components['exp_var_ratio'] = pca.explained_variance_ratio_
    components_df = components[:numfeat_dimensions]
    components_df.T.sort_values(by=0, ascending=False)
    components_df = components[:numfeat_dimensions]
    numerical_feat = components.T[0][np.abs(components.T[0]) > 0.1].drop('exp_var_ratio').index
    
    num_data = df[numerical_feat]
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_numerical_data = scaler.fit_transform(num_data)

    pca = PCA(n_components='mle')
    reduced_numerical_data = pca.fit(scaled_numerical_data).transform(scaled_numerical_data)

    threshold = 0.95
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    numfeat_dimensions = np.where(cumsum > threshold)[0][0]
    
    pca = PCA(n_components=numfeat_dimensions)
    reduced_numerical_data = pca.fit(scaled_numerical_data).transform(scaled_numerical_data)

    n_text_features = sparse_text_matrix.shape[1]

    svd = TruncatedSVD(n_components=text_dimensions)
    svd.fit(sparse_text_matrix)
    reduced_text_matrix = svd.transform(sparse_text_matrix)

    X_num = pd.DataFrame(reduced_numerical_data)
    X_text = pd.DataFrame(reduced_text_matrix)
    X = pd.concat([X_num, X_text], axis=1)
    
    return X, y

# default threshold is 0.5
def plot_threshold(precisions, recalls, thresholds, clf, recall_thresh=0.5):
    thresh_ind = np.where(recalls > recall_thresh)[0][-1]
    rec_thresh = recalls[thresh_ind]
    prec_thresh = precisions[thresh_ind]
    thresh = thresholds[thresh_ind]
    print('Treshold: {:.3f}'.format(thresh))
    plt.figure(figsize=(7,5))
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.vlines(x=thresh, ymin = 0.0, ymax = rec_thresh, colors='red', linestyles='--', alpha=0.5)
    plt.hlines(y=rec_thresh, xmin=thresholds.min(), xmax=thresh, colors='red', linestyles='--', alpha=0.5)
    plt.hlines(y=prec_thresh, xmin=thresholds.min(), xmax=thresh, colors='red', linestyles='--', alpha=0.5)
    plt.scatter(x=thresh, y=rec_thresh)
    plt.scatter(x=thresh, y=prec_thresh)
    plt.show()
    return thresh