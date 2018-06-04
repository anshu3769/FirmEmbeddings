
We have run our scripts on the GPU server 13.90.81.78. The data is present on the same server.


# Raw data for cases and stock price change can be found at path-
    # Main Directory 
        /data/WorkData/firmEmbeddings/
        
    # Case Data is inside the directory
        CaseData/
        
    # Stock Data is inside the directory
        StockData/
        
# The data after processing and joining can be found at path - 
    # Main Directory 
        /data/WorkData/firmEmbeddings/Models/
        
    # Random Forest for Stock Prediction Data
         StockPredictionUsingRandomForest/
         
    # Neural Network for Stock Prediction Data
        StockPredictionUsingNeuralNetwork/Data_stock_all/
        
    # Neural Network for Firm Embeddings Data
        FirmEmbeddings/
        
        
# To install the packages for running all the scripts execute the command-
    chmod 755 requirements.sh
    sh requirements.sh

# Go to the python shell and execute the command for downloading punkt -
    python3
    >> nltk.download('punkt')
  
  
# Scripts to process the raw case data-
    
    These files are present in CaseData folder. Run the files in following order-
    The data generated from these scripts will be used with stock data in creating final data for training 
    the models. This data can be found in /data/WorkData/firmEmbeddings/CaseData/ folder present on the server.
      
    1.  filterCases.ipynb - Filters cases from sentences folder to get cases for category 6 and 7. It uses 
    bb2topic.pkl, bb2genis.pkl, caseid_date.csv. This generates new folder Filtered_1 and the files 
    -filtered.pkl, casedata.pkl. The Filtered_1 contains all cases belonging to category 6 and 7.

    2. ngramdataGenerate.ipynb - Filters bigram pickle files to get cases for category 6 and 7 . It uses 
    casedata.pkl and [20180208]build_vocab_lemma_pos/phrased/ and creates new folder PickleFiles. The PickleFiles contains all cases belonging to category 6 and 7.
    
    3. bigram.ipynb- It creates final ngramdata.pkl. The code uses id2gram.pkl, casedata.pkl, df-tf.pkl 
    and files from PickleFiles folder to generate data. 

    4. doc2vec.py- Uses text from Filtered_1 and runs doc2vec algorithm on filtered cases and generate 
    doc2vec_2.model

    5. modeltodata.ipynb - Uses casedata.pkl and doc2vec_2.model. It maps model vectors to case meta 
    data and creates visualization of docvectors. The code produces following files docvector.pkl, 
    traindocvector.pkl, testdocvector.pkl, validationdocvector.pkl


# Script to process the raw Stock Data - 
    Run the script filterCompanies.py present in path StockData to process the stock data
    python3 filterCompanies.py


# Script to join the two data sets - 
    These files are present in JoiningDataPrep folder
    
    1. StockAndCaseDataJoined - joins case and stock data. This script uses stockData07to13_logdiff_5_0.1.csv 
    and following docvector files - traindocvector.pkl, testdocvector.pkl, validationdocvector.pkl. 
    And produces following files - training_data_CaseCompanyStockChange.pkl, 
    testing_data_CaseCompanyStockChange.pkl, validation_data_CaseCompanyStockChange.pkl
    
    2. ProcessJoinedDataForNN.ipynb - processes data for final run and creates val_data_final.pkl, 
    train_data_final.pkl, test_data_final.pkl
    
    3. Finaldata_stockPred.ipynb - produces final data for all cases and category 
    6 and 7 for stock prediction 
    
    4. Finaldata_firmEmbed.ipynb - produces final data for all cases and category 6 and 7 
    for firm embeddings and uses Company_meta.pkl
    
    5. RankCompany.ipynb - used to create Company_meta_rank.pkl
    
    After running all these scripts, the data for all the models will be copied in their respective
    paths mentioned above.
 
# Script to generate models for stock prediction and firm embeddings -

    #Change file permissios to run the script
    chmod 755 RunAllmodels.sh
    
    # Run the following command to execute the script -
    sh RunAllmodels.sh
    
    This script contains three scripts. Path locations for the scripts on github are - 
    1. RunRandomForest.py is present in the directory Random_Forest/
    2. FirmEmbeddingsModel.py is present in the directory FirmEmbeddings/ 
    3. NeuralNetworkRun_3layers.py is present in the directory StockPrediction/
    
    The script RunRandomForest.py will generate the Random Forest model and it will also plot the 
    graph for actual vs predicted change in stock price.
    
    
    The predictions on test data after running the NeuralNetworkRun_3layers.py script are saved in 
    predictions.txt in the same path in which data is present. The file predictions.txt along 
    with actual.txt (which is also present in the same path as predictions.txt) will be used by the 
    notebook StockPrediction/ScatterPlotPredictedvsActual.ipynb in plotting the actual/predicted 
    stock price change. The notebook contains the absolute path for these files. 
    Thus the notebook can also be run from anywhere on the GPU server.
    
    
    The firm embeddings matrix after running the script FirmEmbeddingsModel.py saves the matrix 
    in the same path in which data is present. This matrix will be used by 
    FirmEmbeddings/VisualizeFirmsEmbeddings.ipynb to visualize the embeddings. This notebook contains 
    the Tsne plots for category 6, 7 and combines cases. It also contains the embeddings visualization 
    against industries of the firms, ranking of the firms, states in which they lie. The 
    notebook  also contains the cosine similarity plots for the two categories - Finance 
    and Manufacturing. 

