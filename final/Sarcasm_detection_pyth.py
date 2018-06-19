
# coding: utf-8

# In[20]:

import re
from collections import Counter
import pickle
import numpy as np
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from scipy.sparse import csr_matrix, hstack
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
import random
import math
from sklearn.metrics import precision_recall_curve
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import neighbors 
from sklearn.metrics import f1_score
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Convolution1D, Flatten
from keras.utils import to_categorical


# In[21]:

def remove_url(text):
    pattern = "((http|ftp|https):\/\/)?([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w.,@?^=%&:\/~+#-])?"
    return re.sub(pattern, "", text)

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    tokens = nltk.word_tokenize(text)
    removed_stopwords = " ".join([w for w in tokens if not w in stop_words])
    return removed_stopwords

def is_too_short(text):
    tokens = nltk.word_tokenize(text)
    return len(tokens) <= 3


# In[22]:

def divide_text(text, n):
    tokens = nltk.word_tokenize(text)
    spilting_length = len(tokens) / n
    out = []
    x = 0
    for i in range(n):
        str_list = tokens[x:int(x+spilting_length)]
        string = " ".join(str_list)
        out.append(string)
        x = int(x+spilting_length)
    return out

def get_sentiment(arr):
    n = len(arr)
    polar = []
    for i in range(n):
        polar.append(TextBlob(arr[i]).sentiment.polarity)
    return polar

def find_sentiment(arr):
    n = len(arr)
    #Returns new array of same dimension without initialisation
    out = np.empty((len(arr), 6))
    for i in range(len(arr)):
        uni_polarity = TextBlob(arr[i]).sentiment.polarity
        bigrams_list = divide_text(arr[i], 2)
        bi_polarity = get_sentiment(bigrams_list)
        trigram_list = divide_text(arr[i], 3)
        tri_polarity = get_sentiment(trigram_list)
        out[i] = [uni_polarity, bi_polarity[0], bi_polarity[1], tri_polarity[0], tri_polarity[1], tri_polarity[2]]
    return out


# In[23]:

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


# In[24]:

def count_features(text):
    def count_apost(text):
            return text.count('!')
    def count_qn(text):
            return text.count('?')
    def count_capitals(text):
            count=0
            tokens=nltk.word_tokenize(text)
            for each_word in tokens:
                if each_word[0].isupper():
                    count+=1
            return count
    def data_len(text):
            return len(text)
    def count_quotes(text):
            return text.count('\"')
    def count_emoji(text):
        emoji_list = [":p", ":)", ";)", "emoticonX"]
        emoji_count = [text.count(e) for e in emoji_list]
        return sum(emoji_count)
    
    ap = count_apost(text)
    qn = count_qn(text)
    cap = count_capitals(text)
    l = data_len(text)
    quotes = count_quotes(text)
    #emo = count_emoji(text)
    return np.array([ ap, qn , cap , l , quotes])


# In[25]:

def pos_tag_finder(text):
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    counts = Counter(tag for word,tag in tags)
    total = sum(counts.values())
    return dict((word, float(count)/total) for word,count in counts.items())

def get_pos_features(arr):
    out = np.array([])
    for i in range(len(arr)):
        pos_tags = pos_tag_finder(arr[i])
        out = np.append(out, pos_tags)
    return out


# In[26]:

#Load dataset
dataset = load_files('container/', encoding="utf8", decode_error="replace")


# In[27]:

X = np.array([])
y = np.array([])
for i in range(len(dataset.data)):
    if not is_too_short(dataset.data[i]):
        noisless_text = remove_url(str(dataset.data[i]))
        noisless_text = remove_stopwords(noisless_text)
        #noisless_text = TextBlob(noisless_text).correct()
        X = np.append(X, noisless_text)
        if dataset.target[i] == 0:
            y = np.append(y, 'notsarc')
        else:
            y = np.append(y, 'sarc')


# In[28]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#Getting features of length of text ,count of capitals, apostrophe, question marks, quotes, abbreviation
feat_count_train=np.array([])
f=np.array([])
feat_count_test = np.array([])
for x in X_train:
    f = count_features(x)
    feat_count_train = np.append(feat_count_train ,f )
feat_count_train = feat_count_train.reshape(len(X_train) , 5)
for x in X_test:
    f = count_features(x)
    feat_count_test = np.append(feat_count_test ,f )
feat_count_test = feat_count_test.reshape(len(X_test) , 5)


# In[29]:

#Returns unigram, bi gram and trigram polarity
sentiment_train = csr_matrix(find_sentiment(X_train))
sentiment_test = csr_matrix(find_sentiment(X_test))


# In[30]:

#Stem reduces size of dictionary by converting words to their root form
stemmer = PorterStemmer()


# In[31]:

#Get the tfidf matrix(count of words per sentence)
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', ngram_range=(1, 5), max_features=1939)
tfs_train = tfidf.fit_transform(X_train)
tfs_test = tfidf.transform(X_test)


# In[32]:

#scikit-learn estimators expect numerical features, we convert the categorical and boolean features using one-hot encoding
#In order to perform one-hot encoding, we need to give the entire data as an input and cannot perform one-hot encoding per observation. 
vec = DictVectorizer()

pos_train = vec.fit_transform(get_pos_features(X_train))
pos_test = vec.transform(get_pos_features(X_test))


# In[18]:

#To split the text into topics and use these topics as features, (some topics are more likely to be sarcastic)
lda = LatentDirichletAllocation(n_components=10, learning_method='online')

topic_train = lda.fit_transform(tfs_train)
topic_test = lda.transform(tfs_test)


# In[19]:

#COMBINE ALL THE ABOVE FEATURES
final_train = hstack([sentiment_train, tfs_train, pos_train, topic_train, feat_count_train])
final_test = hstack([sentiment_test, tfs_test, pos_test, topic_test, feat_count_test])
#print(final_train)
print(final_train.shape)
accuracies = []
f_scores = []


# In[70]:

#Naive Bayes
#1. Bernoulli
from sklearn.naive_bayes import GaussianNB, BernoulliNB
Bnb_clf = BernoulliNB()
Bnb_clf = Bnb_clf.fit(final_train.toarray(), y_train)
predict = Bnb_clf.predict(final_test.toarray())
print(accuracy_score(y_test, predict))
print(classification_report(y_test, predict))
accuracies.append(accuracy_score(y_test, predict))
f_scores.append(f1_score(y_test, predict,average = "macro"))


# In[71]:

#Naive Bayes
#2. Gaussian
gnb_clf = GaussianNB()
gnb_clf = gnb_clf.fit(final_train.toarray(), y_train)
predict = gnb_clf.predict(final_test.toarray())
print(accuracy_score(y_test, predict))
print(classification_report(y_test, predict))
accuracies.append(accuracy_score(y_test, predict))
f_scores.append(f1_score(y_test, predict,average = "macro"))


# In[72]:

#Logistic Regression

logistic_clf = LogisticRegression()
logistic_clf = logistic_clf.fit(final_train, y_train)
predict = logistic_clf.predict(final_test)
print(accuracy_score(y_test, predict))
print(classification_report(y_test, predict))
accuracies.append(accuracy_score(y_test, predict))
f_scores.append(f1_score(y_test, predict,average = "macro"))


# In[ ]:




# In[73]:

#SVM WITH RBF KERNEL
svm_clf = SVC(C=4,gamma=1.3)
svm_clf = svm_clf.fit(final_train, y_train)
predict = svm_clf.predict(final_test)
print(accuracy_score(y_test, predict))
print(classification_report(y_test, predict))
accuracies.append(accuracy_score(y_test, predict))
f_scores.append(f1_score(y_test, predict,average = "macro"))


# In[74]:

#lINEAR SVM
linear_svm_clf = LinearSVC(C=0.1)
linear_svm_clf = linear_svm_clf.fit(final_train, y_train)
predict = linear_svm_clf.predict(final_test)
print(accuracy_score(y_test, predict))
print(classification_report(y_test, predict))
accuracies.append(accuracy_score(y_test, predict))
f_scores.append(f1_score(y_test, predict,average = "macro"))


# In[75]:

#Decision Tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(final_train, y_train)
predict = clf.predict(final_test)
print(accuracy_score(y_test, predict))
print(classification_report(y_test, predict))
accuracies.append(accuracy_score(y_test, predict))
f_scores.append(f1_score(y_test, predict,average = "macro"))


# In[ ]:




# In[76]:

#PLOT
import graphviz 
with open("sarcasm_dt.txt", "w") as f:
    f = tree.export_graphviz(clf, out_file=f, max_depth =3)
#Use the text file to plot the tree on http://webgraphviz.com/


# In[77]:

rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf = rf_clf.fit(final_train, y_train)
predict = rf_clf.predict(final_test)
print(accuracy_score(y_test, predict))
print(classification_report(y_test, predict))
accuracies.append(accuracy_score(y_test, predict))
f_scores.append(f1_score(y_test, predict,average = "macro"))


# In[80]:

#KNN (N=21 GAVE BEST ACCURACY)
n_neighbours = 21
for weights in ['uniform', 'distance']:
    clf= neighbors.KNeighborsClassifier(n_neighbours, weights = weights)
    clf.fit(final_train , y_train)
    predict = clf.predict(final_test)
    print(weights)
    print(accuracy_score(y_test, predict))
    print(classification_report(y_test, predict))
    accuracies.append(accuracy_score(y_test, predict))
    f_scores.append(f1_score(y_test, predict,average = "macro"))


# In[83]:

#Recurrent neural net
#Training
xlist = list(X_train)
#print(xlist)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(xlist)
print(len(tokenizer.word_index))
sequences = tokenizer.texts_to_sequences(xlist)
#print(sequences)
l = len(max(sequences,key = lambda  x : len(x)))
print(l)
padded_sequences = pad_sequences(sequences, maxlen = 1000) #padded_sequencies is the tokenized and padded data
#padded_sequences


# In[84]:

model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, 128, input_length=1000)) #maxlen of tokenizerwordindex
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2)) #128 depends on no of words in a row 
model.add(Dense(2, activation='sigmoid')) #2 because of one hot enc
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[85]:

from keras.utils import to_categorical

y_train_new = []
y_test_new = []
for x in y_train:
    if x == 'sarc':
        y_train_new.append(1)
    else:
        y_train_new.append(0)
for x in y_test:
    if x == 'sarc':
        y_test_new.append(1)
    else:
        y_test_new.append(0)
        
                
#print(y_train_new)
y_train_new = to_categorical(y_train_new, num_classes = 2)
y_test_new = to_categorical(y_test_new,  num_classes = 2) 
#print(y_train_new)


# In[86]:

model.fit(padded_sequences, y_train_new, validation_split=0.2, epochs=3)


# In[87]:

#Testing
xlist_test = list(X_test)
#print(xlist)


#tokenizer = Tokenizer()
#tokenizer.fit_on_texts(xlist_test)
print(len(tokenizer.word_index))
sequences = tokenizer.texts_to_sequences(xlist_test)
#print(sequences)
l_test = len(max(sequences,key = lambda  x : len(x)))
print(l_test)
padded_sequences_test = pad_sequences(sequences, maxlen = 1000) #padded_sequencies is the tokenized and padded data
#padded_sequences


# In[89]:

scores = model.evaluate(padded_sequences_test,y_test_new,verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
#y_pred = model.predict()
accuracies.append(scores[1]*100)


# In[97]:

objects = ('NB Bernoulli', 'NB Gaussian', 'Logistic regression',  'SVM RBF', 'SVM Linear','Decision trees', 'Random Forests', 'kNN', 'RNN')
y_pos = np.arange(len(objects))
performance = accuracies[0:9]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation=90)

plt.ylabel('Classification Model')
plt.title('Comparison of accuracies')
 
plt.show()


# In[95]:

objects = ('NB Bernoulli', 'NB Gaussian', 'Logistic regression' 'Decision trees', 'Random Forests', 'SVM RBF', 'SVM Linear', 'kNN', 'RNN')
print(len(objects))
print(len(accuracies[0:8]))


# In[1]:

objects = ('NB Bernoulli', 'NB Gaussian', 'Logistic regression',  'SVM RBF', 'SVM Linear','Decision trees', 'Random Forests', 'kNN')
y_pos = np.arange(len(objects))
performance_fscores = f_scores[0:8]
 
plt.bar(y_pos, performance_fscores, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation=90)

plt.ylabel('Classification Model')
plt.title('Comparison of accuracies')
 
plt.show()


# In[ ]:



