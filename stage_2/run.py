import os
import numpy as np
from dateutil.parser import parse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn
import matplotlib.dates as mdate
import pandas as pd
import datetime as dt

def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])
        
        
def predict_and_plot(dates_len, dates, percentages, error_type_str):
    if len(percentages) <= 1:
        return 0

    elif len(percentages) < 20:
        dates = dates[::-1]
        percentages = percentages[::-1]
        dates_num = pd.to_datetime(dates)
        dates_num = dates_num.map(dt.datetime.toordinal)
        dates = np.array(dates).reshape(-1, 1)
        dates_num = np.array(dates_num).reshape(-1, 1)
        percentages = np.array(percentages).reshape(-1, 1)
        lr = LinearRegression().fit(dates_num, percentages)
        
        plt.scatter(x_train, y_train, color='b', label='train data')
        plt.scatter(x_test, y_test, color='r', label='test data')
        y_test_pred = lr.predict(dates_num)
        plt.plot(dates, y_test_pred, color='black', label='best line')
        plt.xlabel("dates")
        plt.ylabel("percentages")
        plt.legend(loc=2)

    else:
        dates = dates[::-1]
        percentages = percentages[::-1]
          
        dates_num = pd.to_datetime(dates)
        dates_num = dates_num.map(dt.datetime.toordinal)
    
        dates = np.array(dates).reshape(-1, 1)
        dates_num = np.array(dates_num).reshape(-1, 1)
    
    
        percentages = np.array(percentages).reshape(-1, 1)
    
        x_train = dates[ : -11]
        x_train_num = dates_num[ : -11]
        y_train = percentages[ : -11]
        lr = LinearRegression().fit(x_train_num, y_train)
    
        x_test = dates[-10 : ]
        x_test_num = dates_num[-10 : ]
        y_test = percentages[-10 : ]
    
        plt.scatter(x_train, y_train, color='b', label='train data')
        plt.scatter(x_test, y_test, color='r', label='test data')
        y_test_pred = lr.predict(dates_num)
        plt.plot(dates, y_test_pred, color='black', label='best line')
    
        plt.xlabel("dates")
        plt.ylabel(error_type_str + "percentages")
        plt.legend(loc=2)

        
# main code
print("stage 2 start")
df_percentages = []
su_percentages = []
df_dates = []
su_dates = []
uf_dates = []

Str1 = " 91% Used. Warning. Disk Filling up."
Str2 = "SWAP WARNING - 15% free (1194 MB out of 8095 MB)"

path = "../pre_stage/output/stage2/" 
files= os.listdir(path) 
print("start reading data from ", path)
for file in files: 
    if not os.path.isdir(file): 
          f = open(path+"/"+file); 
          first_line = f.readline()
          date = parse(first_line, fuzzy = True)
          iter_f = iter(f); 
          for line in iter_f:
              df_ratio = levenshtein_ratio_and_distance(Str1,line,ratio_calc = True)
              su_ratio = levenshtein_ratio_and_distance(Str2,line,ratio_calc = True)
              if df_ratio > 0.7:
                  df_str_list = line.split(" ")
                  df_float = float(df_str_list[3].rstrip('%'))/100
                  df_percentages.append(df_float)
                  df_dates.append(date)
              elif df_ratio > 0.7:
                  su_str_list = line.split(" ")
                  df_float = float(su_str_list[3].rstrip('%'))/100
                  df_percentages.append(df_float)
                  df_dates.append(date)
              else:
                  uf_dates.append(date)

print("finish reading")
print("creating plot")
plot_name = "stage2.pdf"                  
predict_and_plot(len(df_dates), df_dates, df_percentages, "disk filling up")
predict_and_plot(len(su_dates), su_dates, su_percentages, "swap usage")
plt.savefig(plot_name)
print("plot saved to",plot_name)
print("finish stage 2")
