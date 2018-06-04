
# coding: utf-8

# In[ ]:

import multiprocessing
import gensim
import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join
#nltk.download()
import string 
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
import pickle 
import pandas as pd 
import os

from sklearn.manifold import TSNE
import re
import matplotlib.pyplot as plt
from gensim.models.deprecated.doc2vec import LabeledSentence

# In[ ]:





#reading documents

data=[]
docLabels=[]
tot=0
dub=0
for i in range(2001,2014):

    pathname = os.path.join("/data/WorkData/firmEmbeddings/CaseData/Filtered_1", str(i))
    files=os.listdir(pathname)
    print(pathname)
    print(len(files))
    tot=tot+len(files)
    for file in files:
        file1=file.split("_")[0]
        caseid=file1
        if caseid not in docLabels:
            
            try:
                #print(pathname+"/"+file)
                data.append(open(pathname+"/"+file).read())
                docLabels.append(caseid)
            except:
                print("error")
                print(file)
                print(i)
                continue
            


        else:
            dub=dub+1
            
            
print(len(docLabels))
print(len(data))
print(tot)
print(dub)
dd=docLabels[0]
# In[ ]:





def clean(text):

    text=text.lower()
    table = str.maketrans("", "", string.punctuation)
    wnl = nltk.WordNetLemmatizer()
    
    
    text=text.translate(table)
    #print("without punctuation ")
    #print(text)
    token = nltk.word_tokenize(text)
    #print("tokenized")
    #print(token)
    

    stop_words = set(stopwords.words('english'))
    filtered_sentence = []

    for w in token:
        if w not in stop_words:
            filtered_sentence.append(w)
   
    token=filtered_sentence
    #print("without stop words")
    #print(token)
    token=[wnl.lemmatize(t) for t in token]
    #print("after lemma")
    #print(token)
    return token

new_data=[]
for d in data:
    d=clean(d)
    new_data.append(d)
data=new_data

print("data cleaned")
# In[ ]:
#LabeledSentece = gensim.models.deprecated.doc2vec.LabeledSentence

#create iterator of data and doclabels
class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield LabeledSentence(doc,[self.labels_list[idx]])


# In[ ]:



it = LabeledLineSentence(data, docLabels)
print("iterator created")

# In[ ]:
cores = multiprocessing.cpu_count()
print("cores - ",cores)

model = gensim.models.Doc2Vec(dm=1, dm_mean=1, size=100, negative=6, hs=0, window=5, min_count=5, alpha=0.025, min_alpha=0.025, workers=cores)
model.build_vocab(it)

#training 
for epoch in range(100):
    print('iteration '+str(epoch+1))
    model.train(it,total_examples=model.corpus_count,epochs=model.epochs)
    model.alpha -= 0.0002
    print(model.alpha)
    model.min_alpha = model.alpha

model.save('/data/WorkData/firmEmbeddings/CaseData/doc2vec_2.model')
print("model saved")


# In[ ]:


d2v_model = gensim.models.doc2vec.Doc2Vec.load('/data/WorkData/firmEmbeddings/CaseData/doc2vec_2.model')


# In[ ]:


#finding similar docs
sims = d2v_model.docvecs.most_similar(dd)
print((sims))

