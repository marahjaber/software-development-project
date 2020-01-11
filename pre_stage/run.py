#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import numpy as np
import tensorflow as tf
import pandas as pd
import re
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation, Dense, Dropout
from keras.optimizers import Adam
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression

# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier

# Performance metric
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score

#import nltk
from sklearn.metrics import classification_report, accuracy_score

import pickle
#from sklearn.metrics import multilabel_confusion_matrix

from sklearn.model_selection import KFold, cross_val_score

def copy_data(src_file_path, dst_file_path):
    if not os.path.exists(dst_file_path):
        os.mkdir(dst_file_path)

    for logfile in glob.glob(src_file_path + '/*'):
        if os.stat(logfile)[6] >= 1000:
            logfile_name = logfile.split('/')[-1]
            shutil.copyfile(logfile, dst_file_path + '/' + logfile_name)


def read_data(logfile_path):
    cols = ['filename','text', 'labels']
    log_collection = pd.DataFrame(columns = cols)
    logs = pd.DataFrame()
    
    logfiles = glob.glob(logfile_path + '/*')
    for logfile in logfiles:
        if os.stat(logfile)[6] < 1000:
            continue
        logs = pd.read_csv(logfile, sep='\n', header=None, names=['data'])
        
        with open(logfile, 'r') as file:
            data = file.read().replace('\n', '')

        log_data, subject, body = {}, "", ""
        raw_data = open(logfile)
        
        #remove email headers
        for record in raw_data:
            if record.startswith("Subject:"):
                subject = record.split("Subject:")[1].strip()
            elif not record.startswith("=?utf") and not record.startswith("X-Microsoft") and not record.startswith("To")  and not record.startswith("X-Authentication-Warning:") and not record.startswith("From") and not record.startswith("Received") and not record.startswith("Sender") and not record.startswith("Content-Type") and not record.startswith("User-Agent") and not record.startswith("Message-ID")  and not record.startswith("X-DurhamAcUk-MailScanner") and not record.startswith("X-") and not record.startswith("Return-Path") and not record.startswith("Precedence") and not record.startswith("Content-Transfer-Encoding") and not record.startswith("Date") :
                body += record         
            else:
                continue
        log_data[subject] = body

        #label log file
        labels = label_file(logfile)
        
        log_collection = log_collection.append({'filename':logfile.split('/')[-1],'text': body, 'labels':labels},ignore_index=True)
        
   
    #clean text
    log_collection['clean_text'] = log_collection['text'].apply(lambda x: clean_text(x))
    
    # Remove empty lines
    log_collection = log_collection.dropna()

    # Reset the index
    log_collection = log_collection.reset_index(drop=True)

    return log_collection


def label_file(input_file):
    fp = input_file
    raw_data = open(fp)
    labels = set()
    for line in raw_data:
        line = line.lower()
        if line.find('disk filling') != -1:
            labels.add('stage2')
            
        elif line.find('swap usage') != -1:
            labels.add('stage2')
            
        elif line.find('update failed') != -1:
            labels.add('stage2')
            
        elif line.find('error') != -1:
           labels.add('stage1')
      
        elif line.find('errors') != -1:
           labels.add('stage1')

        elif line.find('failure') != -1:
           labels.add('stage1')
        
        elif line.find('fail') != -1:
           labels.add('stage1')

        elif line.find('failed') != -1:
           labels.add('stage1')
  
        elif line.find('disconnecting') != -1:
           labels.add('stage1')

        elif line.find('disconnected') != -1:
           labels.add('stage1')
       
        elif line.find('disconnect:') != -1:
           labels.add('stage1')

        elif line.find('illegal') != -1:
           labels.add('stage1')
 
        elif line.find('unmatched') != -1:
           labels.add('stage1') 
      
        elif (line.find('warning') != -1 and line.find("x-authentication-warning") == -1 and line.find('warning. disk filling up.') == -1):
           labels.add('stage1')

    return labels

# function for text cleaning 
def clean_text(text):
    # remove backslash-apostrophe
    text = re.sub("\'", "", text) 
    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    # remove whitespaces 
    text = ' '.join(text.split()) 
    # convert text to lowercase 
    text = text.lower()     
    #remove stop words
    text = remove_stopwords(text)
    return text


# function to remove stopwords
def remove_stopwords(text):
    #define custom stop words for our logs
    stop_words = "qb qc qd qe wv ya yp ym qh qi qu zl za jan feb mar apr may jun jul aug sep oct nov dec ql yh ve ms smtp esmtp ut with id BST bst utf iso tue dc he dev cosma uk durham dur ac nb utf iso zz sat us ascii jf jd iu it is org ib tue np tue qm ip eu sat fri mon thu wed owner mon tue wed thu fri sat sun with id oct ib eu"
    
    #stop_words = "jan feb mar apr may jun jul aug sep oct nov dec smtp esmtp ut with id BST bst utf iso tue dev cosma uk durham dur ac sat us ascii org ipeu sat fri mon thu wed owner mon tue wed thu fri sat sun with id oct eu"

    stop_words = stop_words.split(" ")
    without_stop_words = [w for w in text.split() if not w in stop_words]
    return ' '.join(without_stop_words)


def prepare_data(xtrain,xval):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2),max_df=0.9,min_df=3)#,token_pattern=r'(\S+)' )
    tfidf_vectorizer.fit(xtrain)
    xtrain_tfidf = tfidf_vectorizer.transform(xtrain)
    xval_tfidf = tfidf_vectorizer.transform(xval)
    tfidf_vocab = tfidf_vectorizer.vocabulary_
    return (xtrain_tfidf, xval_tfidf,tfidf_vocab)

def train(X_train,y_train):
    lr = LogisticRegression(C=30, penalty='l1', dual=False, solver='liblinear')
    model = OneVsRestClassifier(lr)
    model.fit(X_train, y_train)

    kf = KFold(n_splits=10, random_state = 42, shuffle = True)
    scores = cross_val_score(model, X_train, y_train, cv = kf)
    
    print('\nCross-validation scores:', scores)
    print('Cross-validation accuracy: {:.4f} (+/- {:.4f})\n'.format(scores.mean(), scores.std() * 2))
    return model

def report(actual, predictions):       
    print ("\n\033[1m Performance Report \033[0m\033[50m\n")
    print('f1 score: ',str(round(f1_score(actual, predictions, average="micro"),3)))

    #print (multilabel_confusion_matrix(np.array(actual), np.array(predictions)))
    print (classification_report(actual, predictions, target_names=multilabel_binarizer.classes_))
    print ('Accuracy: ' + str(round(accuracy_score(actual, predictions), 3)))


# Main Code
source_data_dir = '../train_data'
test_data_dir = '../test_data'
data_dir = 'copied'

print ('read data start')
if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
copy_data(source_data_dir, data_dir)
log_collection = read_data(data_dir)
test_data = read_data(test_data_dir)
print ('read data end')
print ('prepare data start')
multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(log_collection['labels'])
print ('train test split start')
y_train = multilabel_binarizer.transform(log_collection['labels'])
y_test =  multilabel_binarizer.transform(test_data['labels'])
X_train = log_collection['clean_text']
X_test = test_data['clean_text']
id_train = np.arange(len(log_collection['clean_text']))
id_test = np.arange(len(test_data['clean_text']))
X_train,X_test,tfidf_vocab = prepare_data(X_train,X_test)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}
print ('train test split end')
print ('prepare data end')

# Training
print ('train model start')
model = train(
    X_train,
    y_train)
print ('train model end')

#save model
model_file_path = 'pre_stage_classifier.pkl'
pickle.dump(model, open(model_file_path, 'wb'))

print("\ntrainded model saved to", model_file_path)

# Prediction
print("\nprediction")
actual_labels = multilabel_binarizer.inverse_transform(y_test)
y_pred = model.predict(X_test)
t = 0.4
y_pred_prob = model.predict_proba(X_test)
y_pred_new = (y_pred_prob >= t).astype(int)

predicted_labels = multilabel_binarizer.inverse_transform(y_pred_new)

#output
#copy log to output/stage1 or stage2 output folder
output_dir = "output"
stage1_dir = output_dir+"/"+"stage1"
stage2_dir = output_dir+"/"+"stage2"

if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

if not os.path.exists(output_dir):
        os.mkdir(output_dir)

if not os.path.exists(stage1_dir):
        os.mkdir(stage1_dir)

if not os.path.exists(stage2_dir):
        os.mkdir(stage2_dir)

index = 0
for id_ in id_test:
  filename = test_data['filename'][id_]
  if 'stage1' in predicted_labels[index]:
    shutil.copy( test_data_dir + '/' + filename, stage1_dir)
  if 'stage2' in predicted_labels[index]:
    shutil.copy( test_data_dir + '/' + filename, stage2_dir)
  index+=1

print("output files copied to output/stage1, output/stage2 folders")

#copy predicted wrong logs to output/debug/wrong folder
debug_dir = output_dir + "/" + "debug"
debug_wrong_dir = debug_dir + "/" + 'wrong'

if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)

if not os.path.exists(debug_dir):
        os.mkdir(debug_dir)

if not os.path.exists(debug_wrong_dir):
        os.mkdir(debug_wrong_dir)

for id_ in range(len(id_test)):
  filename = test_data['filename'][id_]  
  if(not(np.array_equal(np.array(predicted_labels[id_]),np.array(actual_labels[id_])))):
    shutil.copy( test_data_dir + '/' + filename, debug_wrong_dir)

print("\npredicted wrong copied to output/debug/wrong folder")

#reporting
print("\nreporting\n")
for id_ in range(len(id_test)):
  filename = test_data['filename'][id_]
  if(not(np.array_equal(np.array(predicted_labels[id_]),np.array(actual_labels[id_])))):
    print("Predicted wrong: log file name: ",filename, ",Prediction: ",np.array(predicted_labels[id_]), ",Actual: ", np.array(actual_labels[id_]) )
   
report(y_test, y_pred)

def print_words_for_tag(classifier, tag, tags_classes, index_to_words):
    """
        classifier: trained classifier
        tag: particular Label
        tags_classes: a list of Labels names from MultiLabelBinarizer
        index_to_words: index_to_words transformation
    """
    print('Label:\t{}'.format(tag))
    
    tag_n = np.where(tags_classes==tag)[0][0]
    
    model = classifier.estimators_[tag_n]
    top_positive_words = [index_to_words[x] for x in model.coef_.argsort().tolist()[0][-8:]]
    top_negative_words = [index_to_words[x] for x in model.coef_.argsort().tolist()[0][:8]]
    
    print('Top positive words:\t{}'.format(', '.join(top_positive_words)))
    print('Top negative words:\t{}\n'.format(', '.join(top_negative_words)))

print("\n")
print_words_for_tag(model, 'stage1', multilabel_binarizer.classes_, tfidf_reversed_vocab)
print_words_for_tag(model, 'stage2', multilabel_binarizer.classes_, tfidf_reversed_vocab)			
