ReadMe

Team: "Sabya"

Dataset:
"File name" - 'sarcasm_v2.csv' the actual dataset
Specifications : It has 4693 rows and 5 columns (But this not the actual thing we have just taken the text in the responses column for each row and created individual text files. All those are together put in the folder name called "databin". This inturn contains two folders namely sarc and nonsarc which has the individual text files within them"

Mandatory the databin , sarc and nonsarc folders shouldnt have any other files within them should be extracted as attached. the code fails to run if there are any other files other than the text files for responses.

Code: 'Sarcasm_detection.ipynb'

Steps to be followed:

1. The following packages with versions should be installed using condas which is what we have done

jupyter                   1.0.0
nltk                      3.2.4
numpy                     1.13.3 
python			  3.6
Keras                     2.0.9
sklearn                   0.19.1 
textblob                  0.13.0 
graphviz                  0.8.1


2. The zipped datafolder needs to be extracted and we get a folder named databin and also extract the code file and we get the ipynb file for code. The databin folder and the 'Sarcasm_detection.ipynb' should be in the same directory.
3.In jupyter notebook run the code cell by cell.
Few cells might take more time specially the cells for running the neural network code as it has to run for 3 epochs. Wait till each cell finishes the execution completely to avoid consistency issues.

4.The code will take in the data from folder and it builds several classification and regression models. Final plots for the comparative accuracies and fscores gets saved in the directory.

Note:
Might take certain time for all the models including the RNN model
Comments present in the source code provide step-by-step explanation of each step. It also has some comments (as solutions) if there are any possible errors that might occur due to version compatibility.
