README for CASE DATA

Run the files in following order

1.  filterCases.ipynb - Filters cases from sentences folder to get cases for category 6 and 7. 
It uses bb2topic.pkl, bb2genis.pkl, caseid_date.csv. This generates new folder Filtered_1 and 
 the files - filtered.pkl, casedata.pkl. The folder Filtered_1 contains all cases belonging to
 category 6 and 7.

2. ngramdataGenerate.ipynb - Filters bigram pickle files to get cases for category 6 and 7 . 
It uses casedata.pkl and [20180208]build_vocab_lemma_pos/phrased/ and creates new folder 
PickleFiles. The folder PickleFiles contains all cases belonging to
 category 6 and 7.

3. bigram.ipynb- It creates final ngramdata.pkl. The code uses id2gram.pkl, casedata.pkl, df-tf.pkl 
and files from PickleFiles folder to generate the data.

4. doc2vec.py- Uses text from Filtered_1 and runs doc2vec algorithm on filtered cases and generate 
doc2vec_2.model

5. modeltodata.ipynb - Uses casedata.pkl and doc2vec_2.model. It maps model vectors to case meta 
data and creates visualization of docvectors. The code produces following files 
docvector.pkl, traindocvector.pkl, testdocvector.pkl, validationdocvector.pkl

