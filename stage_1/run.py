import os
import glob
import shutil
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
from string import digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer

def insert_label(line):
    d = line.lower()
    found = "pass"
    if   ("unmatched" in d) or ("mismatch" in d):
        found = "unmatched"
    elif ("error" in d) and (not("corrected error" in d)) and (not("0 error" in d)) and (not("No known data errors" in d)) and (not("/error." in d)) and (not(("http" in d) and ("error" in d))):
        found = "error"
    elif ("failure" in d) and (not("0 failure" in d)):
        found = "failure"
    elif ("fail" in d) and (not("failed 0" in d)) and (not("fail action=none" in d)):
        found = "fail"
    elif ("warning" in d) and (not("warning     : 0" in d)):
        found = "warning"
    elif "disconnect" in line.lower():
        found = "disconnect"
    elif "illegal" in line.lower():
        found = "illegal"
    return found

def prepare_data(train_file_path, test_file_path, artificial_log_path, artificial_log_en):
    
    train_with_file_name = []
    logfiles = glob.glob(train_file_path)
    for logfile in logfiles:
        line_number = 0
        fp = open(logfile, 'r')
        for line in fp.readlines():
              train_with_file_name.append(line.rstrip()+"logfilenameis"+logfile+",line_number="+str(line_number))
              line_number = line_number + 1
        fp.close()

    test_with_file_name = []
    logfiles = glob.glob(test_file_path)
    for logfile in logfiles:
        line_number = 0
        fp = open(logfile, 'r')
        for line in fp.readlines():
              test_with_file_name.append(line.rstrip()+"logfilenameis"+logfile+",line_number="+str(line_number))
              line_number = line_number + 1
        fp.close()

    ai_with_file_name = [] 
    logfiles = glob.glob(artificial_log_path)
    if(artificial_log_en == 1):
        for logfile in logfiles:
            line_number = 0
            fp = open(logfile, 'r')
            count = 2
            if(logfile == "./artificial_log/gen_msg.000"):
                count = 500
            for line in fp.readlines():
                temp = line.rstrip();
                for i in range(count):
                    ai_with_file_name.append(temp+"logfilenameis"+logfile+",line_number="+str(line_number))
                line_number = line_number + 1
            fp.close()
 
    print("ai log length:", len(ai_with_file_name))
    print("train log length:", len(train_with_file_name))
    print("test log length:", len(test_with_file_name))
    test_ratio = 0.8 * len(test_with_file_name) / (len(train_with_file_name)+len(test_with_file_name)+len(ai_with_file_name))
    np.random.shuffle(ai_with_file_name)
    np.random.shuffle(train_with_file_name)
    np.random.shuffle(test_with_file_name)
    text_with_file_name = ai_with_file_name + train_with_file_name + test_with_file_name

    text = []
    labels = []
    file_name = []
    for i in range(len(text_with_file_name)):
        temp_string = text_with_file_name[i].split("logfilenameis", 2)
        text.append(temp_string[0])
        file_name.append(temp_string[1])
        labels.append(insert_label(text[i]))
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    X = tokenizer.texts_to_matrix(text, mode='tfidf')
    
    encoder = LabelBinarizer()
    encoder.fit(labels)
    y = encoder.transform(labels)

    return X, y, labels, text, file_name, test_ratio

def build_nn(input_size, hidden_size, num_classes, dropout):
    nn = Sequential()
    nn.add(Dense(hidden_size, input_shape=(input_size,)))
    nn.add(Activation('relu'))
    nn.add(Dropout(dropout))
    nn.add(Dense(num_classes))
    nn.add(Activation(tf.nn.softmax))
    nn.summary()
    
    return nn

def train(X_train, y_train, criterion, optimiser, batch_size, num_epochs):
    network.compile(loss=criterion,
                  optimizer=optimiser,
                  metrics=['accuracy'])

    history = network.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        verbose=1,
                        validation_split=0.2)
    
    return network

def report(actual, predictions):
    print("\033[1m Performance Report \033[0m\033[50m\n")
    
    actual = np.array(actual)
    
    print(confusion_matrix(actual, predictions))
    print
    print(classification_report(actual, predictions))
    print("Accuracy: " + str(round(accuracy_score(actual, predictions),2)))
    print
    
    #stage 1-2
def func_stopwords(text, standard_stop):

    for item in standard_stop:
        if item in text:
            text = text.replace(item, '')

    remove_digits = str.maketrans('','',digits)
    text = text.translate(remove_digits)

    return text

def add_priority(text, priority_list):

    raw = copy.deepcopy(text)
    for keyword in priority_list:
        for line in raw:
            if keyword in line:
                for i in range(len(text)):
                    text.append(keyword)
                break
    return text

def read_logs(logfile_path):

    #read stopwords
    stopwords = []
    if(os.path.exists('./Config/stopwords.txt')):
        fp = open('./Config/stopwords.txt', 'r')
        for line in fp.readlines():
            stopwords.append(line.rstrip())
        fp.close()

    #read input abnormal event from stage 1-1 
    text = []
    fp = open(logfile, 'r')
    for line in fp.readlines():
        string_split = line.rstrip().split("logfilenameis:", 2)
        raw_file_name = string_split[1].split(",", 2)
        raw_line_number = raw_file_name[1].split("=", 2)
        raw_fp = open(raw_file_name[0])
        raw_text = raw_fp.readlines()
        join = ''
        for index in range(max(int(raw_line_number[1])-5,0), min(int(raw_line_number[1])+5, len(raw_text))):
            join = join + raw_text[index].rstrip() + "\n"
        text.append(func_stopwords(join, stopwords))
    fp.close()

    return text

def read_priority_list():

    #read priority list from user
    if(os.path.exists("./Config/priority_list")):
        priority_list = []
        fp = open("./Config/priority_list", 'r')
        for line in fp.readlines():
            priority_list.append(line.rstrip())
        fp.close()

    return priority_list

def func_tfidf(output_file, text):

    vectorizer = TfidfVectorizer(stop_words='english',
                                 max_features=10,
                                 min_df=1,
                                 analyzer='word')
    X = vectorizer.fit_transform(text)
    feature_names = vectorizer.get_feature_names()

    total_tf_idf = X.sum(axis = 1)

    dense = X.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    df.to_csv(output_file)

    return feature_names

#MAIN Code Start
stage = 7
disable_stage_1_1 = 0
disable_stage_1_2 = 0
train_file_path = "../train_data/*"
test_file_path = "../pre_stage/output/stage1/*"

artificial_log_en = 1
artificial_log_path = "./artificial_log"
artificial_log_file = artificial_log_path+"/gen_msg.*"

if((stage > 0) and (disable_stage_1_1 == 0)):
    print("Stage_1_1: prepare data start")
    X, y, labels, original_message, file_name, test_ratio = prepare_data(train_file_path, test_file_path, artificial_log_file, artificial_log_en)

if((stage > 1) and (disable_stage_1_1 == 0)):
    print("Stage_1_1: train test split start")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False, stratify=None)

if((stage > 2) and (disable_stage_1_1 == 0)):
    print("Stage_1_1: build model start")
    # Hyperparameters
    input_size = X_train.shape[1] # this is the vocab size
    hidden_size = 512
    num_classes = y_train.shape[1]
    dropout = 0.3

    #num_epochs = 5
    num_epochs = 2
    batch_size = 32#maybe we can try other number, ex:64
    learning_rate = 0.0005

    network = build_nn(input_size, hidden_size, num_classes, dropout)
    criterion = 'categorical_crossentropy'
    optimiser = Adam(lr=learning_rate)

if((stage > 3) and (disable_stage_1_1 == 0)):
    print("Stage_1_1: train model start")
    model = train(X_train, y_train, criterion, optimiser, batch_size, num_epochs)

if((stage > 4) and (disable_stage_1_1 == 0)):
    print("Stage_1_1: predict start")
    file_types = np.unique(labels)
    predictions = model.predict(np.array(X_test))
    predicted_labels = [ file_types[np.argmax(p)] for p in predictions]
    actual_labels = [ file_types[np.argmax(y)] for y in y_test]    
    # Reporting
    report(actual_labels, predicted_labels)

if((stage > 5) and (disable_stage_1_1 == 0)):
    print("Stage_1_1: output start")
    if not(os.path.exists("./Stage_1_1_Result")):
        os.mkdir("./Stage_1_1_Result")
    if not(os.path.exists("./Debug")):
        os.mkdir("./Debug")

    output_category = file_types.tolist()
    output_category.append('predicted_wrong')
    output_category = np.array(output_category)

    fps = {path: open(path, 'w') for path in output_category}
    for index_test in range(len(X_test)):
        if not("gen_msg" in str(file_name[len(X_train)+index_test])):
            #fps[predicted_labels[index_test]].writelines(str(predicted_labels[index_test]))
            #fps[predicted_labels[index_test]].writelines("\t")
            #fps[predicted_labels[index_test]].writelines(str(actual_labels[index_test]))
            #fps[predicted_labels[index_test]].writelines("\t")
            fps[predicted_labels[index_test]].writelines(str(original_message[len(X_train)+index_test]))
            fps[predicted_labels[index_test]].writelines("\tlogfilenameis:")
            fps[predicted_labels[index_test]].writelines(str(file_name[len(X_train)+index_test]))
            fps[predicted_labels[index_test]].writelines("\n")
            if(not(np.array_equal(predicted_labels[index_test], actual_labels[index_test]))):
                fps['predicted_wrong'].writelines(str(predicted_labels[index_test]))
                fps['predicted_wrong'].writelines("\t")
                fps['predicted_wrong'].writelines(str(actual_labels[index_test]))
                fps['predicted_wrong'].writelines("\t")
                fps['predicted_wrong'].writelines(str(original_message[len(X_train)+index_test]))
                fps['predicted_wrong'].writelines("\t")
                fps['predicted_wrong'].writelines(str(file_name[len(X_train)+index_test]))
                fps['predicted_wrong'].writelines("\n")
    
    for index_label in output_category:
        fps[index_label].close()
        if(("pass" in str(index_label)) or ("predicted_wrong" in str(index_label))):
            shutil.move(index_label, "./Debug/"+index_label)
        else:
            shutil.move(index_label, "./Stage_1_1_Result/"+index_label)

if((stage > 6) and (disable_stage_1_2 == 0)):
    print("Stage_1_2: TFIDF start")
    if not(os.path.exists("./Stage_1_2_Result")):
        os.mkdir("./Stage_1_2_Result")

    priority_list = read_priority_list()

    out_fp = open("./Stage_1_2_Result/keyword_list.txt", 'w')

    logfiles = glob.glob("./Stage_1_1_Result/*")
    for logfile in logfiles:
        string_split = logfile.split("/")
        text = read_logs(logfile)
        if(len(text) == 0):
            continue
        text = add_priority(text, priority_list)
        feature_names = func_tfidf("./Stage_1_2_Result/verbose_"+string_split[-1], text)
        print("keywords in", string_split[-1], ":", feature_names)

        out_fp.writelines("keywords in ")
        out_fp.writelines(string_split[-1])
        out_fp.writelines(":\n")
        for i in range(len(feature_names)):
            out_fp.writelines(feature_names[i])
            out_fp.writelines(" ")
        out_fp.writelines("\n\n")

    out_fp.close()

