import pandas as pd
import numpy as np
import csv
import glob, os
from itertools import groupby
from scipy import stats
import random


#filtering out required columns 
df=pd.read_csv("stock07_13.csv")
keep_col = ['datadate','conm','prccd']
new_df = df[keep_col]
new_df.to_csv("stockDataFiltered.csv", index=False)


#Adding two extra rows after each firm for calculation correctness
firmx = []
with open('stockDataFiltered.csv') as f:
    reader = csv.reader(f)
    with open('stockDataMergedTemp.csv', 'w') as g:
        writer = csv.writer(g)
        next(reader, None)
        writer.writerow(['date', 'firm', 'closingPrice'])
        for row in reader:
        	if row[1] not in firmx:
        		firmx.append(row[1])
        		new_row =  [ row[1]]
        		writer.writerow(new_row)
        		writer.writerow(new_row)
        	writer.writerow(row)


#diff calculation & percentage diff calculation 
df = pd.read_csv('stockDataMergedTemp.csv')
#df.drop(df.head(2).index, inplace=True)
df['diff'] = df['closingPrice'].shift(2) - df['closingPrice'].shift(-2)
df['percentage_diff'] =  ( df['diff']/ df['closingPrice'].shift(-2) ) * 100
df.to_csv('stockDataReorderedWithDiff.csv', index=False, sep=',')


#drop columns with empty cells 
df = pd.read_csv('stockDataReorderedWithDiff.csv')
df.dropna(axis=0, how='any').to_csv('stockDataReorderedFinal.csv', index=False, sep=',')




#Dropping columns with 0 diff 
with open('stockDataReorderedFinal.csv') as inp, open('stockDataFinalEnhanced.csv', 'w') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        #if (row[3] != "0.0" and row[4] != 'percentage_diff' and float(row[4]) > -1 and float(row[4]) < 1) :
        if (row[3] != "0.0" and row[4] != 'percentage_diff' and float(row[4]) > -100 and float(row[4]) < 100) :
                writer.writerow(row)




with open('stockDataFinalEnhanced.csv') as f:
    reader = csv.reader(f)
    with open('stockDataFinal.csv', 'w') as g:
        writer = csv.writer(g)
        next(reader, None)
        writer.writerow(['date', 'firm', 'closingPrice', 'diff', 'percentage_diff' ])
        for row in reader:
            writer.writerow(row)



#Filtering out companies in the 4 years 
firm = []
year = []
count = 0;
with open('stockDataFinal.csv') as f:
    reader = csv.reader(f)
    with open('FinalCompanies.csv', 'w') as g:
        writer = csv.writer(g)
        next(reader, None)
        for row in reader:
            if row[1] not in firm:
                count = count + 1 
                y = (row[0].split("/"))[2]
                if y not in year:
                    year.append(y)
                if len(year) == 7:
                    firm.append(row[1])
                    new_row = [row[1]]
                    writer.writerow(new_row)
        print("Total firms in data: ",count)
        print("Total firms in all 7 years in data:", len(firm))



#max min value calculations
df = pd.read_csv('stockDataFinal.csv')
#df.drop(df.head(2).index, inplace=True)
print("Percentage Diff Statistics: ")
print ( "Max:", max(df['percentage_diff']) )
print ( "Min:", min(df['percentage_diff']) )
print ( "Median:", np.median(df['percentage_diff']) )
print ( "Mean:", np.mean(df['percentage_diff']) )
#print ( "Mode:", stats.mode(df['percentage_diff']) )
print ("********************************************")
print("Absolute Diff Statistics: ")
print ( "Max:", max(df['diff']) )
print ( "Min:", min(df['diff']) )
print ( "Median:", np.median(df['diff']) )
print ( "Mean:", np.mean(df['diff']) )
#print ( "Mode:", stats.mode(df['diff']) )




#Randomly choosing 1000 companies' data depending on the above filtered companies 
random.shuffle(firm)
firm = firm[0:701]
with open('stockDataFinal.csv') as f:
    reader = csv.reader(f)
    with open('stockData07to13_percdiff_100.csv', 'w') as g:
        writer = csv.writer(g)
        next(reader, None)
        for row in reader:
            if row[1] in firm:
                writer.writerow(row)




"""
#TODO : Debug | Code for increasing desirable rows
df = pd.read_csv('stockDataReorderedWithDiff.csv')
df['diff2'] = np.where(df['diff'].isnull, df['closingPrice'].shift(-1) - df['closingPrice'].shift(1), " ")
df.to_csv('stockDataReorderedWithDiff2.csv', index=False, sep=',')

"""





