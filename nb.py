import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import sklearn
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import average_precision_score

from preprocessSentences import get_test_data, get_vocab, bow_to_tfidf
from preprocessSentences import cross_val
import matplotlib.pyplot as plt


def get_err(pred, y_test, n = None):
	if n is None:
		err = np.sum(np.abs((pred - y_test))).astype(float)/len(y_test)
	else:
		err = np.sum(np.abs((pred - y_test))).astype(float)/n
	return err





############################ START HERE #################################
bow = np.genfromtxt('out_bag_of_words_3.csv',delimiter=',')
bow_bigram = np.genfromtxt('out_bag_of_words_bigrams1.csv',delimiter=',')
classes = np.loadtxt('out_classes_3.txt')

bow_tfidf = bow_to_tfidf(bow)


#X_train = bow_bigram
#y_train = np.array(classes)
#X_test,y_test = get_test_data_bigram()

X_train = bow
y_train = np.array(classes)
X_test,y_test = get_test_data()


def flatten(f):
	a = np.zeros(len(f))
	for i in range(len(f)):
		a[i] = f[i]
	return a

feature_selection = False
tfidf = False


if tfidf:
	X_train = bow_to_tfidf(X_train)
	X_test = bow_to_tfidf(X_test)


pos_inds = (y_train == 1)
neg_inds = (y_train == 0)

MI = np.zeros(X_train.shape[1])
PD = np.zeros(X_train.shape[1])
CHI = sklearn.feature_selection.chi2(X_train,y_train)[0]
for f in range(X_train.shape[1]):
	feature_pred = flatten(X_train[:,f])
	MI[f] = sklearn.metrics.mutual_info_score(y_train,feature_pred)
	pdf = np.sum(X_train[pos_inds,f])
	ndf = np.sum(X_train[neg_inds,f])
	PD[f]  = np.abs(pdf-ndf)/(pdf+ndf)



if feature_selection:
	n = 400
	imp_inds = np.argsort(MI)
	#imp_inds = np.argsort(PD)
	#imp_inds = np.argsort(CHI)
	X_train = X_train[:,imp_inds[-n:]]
	X_test = X_test[:, imp_inds[-n:]]



nb_c = naive_bayes.MultinomialNB()
nb_c.fit(X_train,y_train)

svm = SVC(C=1.0, kernel='linear',probability=True)
svm.fit(X_train,y_train)

lr_c = LogisticRegression(penalty='l2',C=1.0)
lr_c.fit(X_train,y_train)

### Logistic Regression important variables


beta = lr_c.coef_
important_inds = np.argsort(beta)
top_five = important_inds[0,-10:]
bottom_five = important_inds[0,:10]

vocab = get_vocab()
vocab[top_five]
vocab[bottom_five]


nb_probs = nb_c.predict_proba(X_test)
lr_probs = lr_c.predict_proba(X_test)
svm_probs = svm.predict_proba(X_test)
nb_probs_max = np.max(nb_probs,axis=1)
lr_probs_max = np.max(lr_probs,axis=1)
svm_probs_max = np.max(svm_probs,axis=1)
nb_probs = nb_probs[:,1]
lr_probs = lr_probs[:,1]
svm_probs = svm_probs[:,1]


nb_pred = nb_c.predict(X_test)
lr_pred = lr_c.predict(X_test)
svm_pred = svm.predict(X_test)

nb_high_conf = np.arange(len(X_test))[nb_probs > .9]
lr_high_conf = np.arange(len(X_test))[lr_probs > .9]
svm_high_conf = np.arange(len(X_test))[svm_probs > .9]

print get_err(nb_pred[nb_high_conf],y_test[nb_high_conf])
print get_err(lr_pred[lr_high_conf],y_test[lr_high_conf])
print get_err(svm_pred[svm_high_conf],y_test[svm_high_conf])


intersect = nb_pred * lr_pred * svm_pred
intersect9 = (nb_probs > .9) & (lr_probs > .9) & (svm_probs > .9)
intersect8 = (nb_probs > .8) & (lr_probs > .8) & (svm_probs > .8)

err = np.sum(np.abs(intersect9*(nb_pred - y_test)))/len(y_test)
print err

##### prints examples of easy sentences

docs = get_test_docs()

int_docs = []
for i in np.arange(len(docs))[intersect9.astype(bool)]:
	print docs[i]

easiest = nb_probs + lr_probs + svm_probs
easiest_p = np.argsort(easiest)[-3:]
easiest_n = np.argsort(easiest)[:3]

hardest = nb_probs_max + lr_probs_max + svm_probs_max
hardest_p = np.argsort(hardest)[:6]

for i in easiest_p:
	print docs[i]

for i in easiest_n:
	print docs[i]

for i in hardest_p:
	print docs[i]

#### LOOK at examples that are hard/easy for all 3 classifiers

#####

#### correlation coefficients
print np.corrcoef(nb_pred,lr_pred)[0,1]
print np.corrcoef(svm_pred,lr_pred)[0,1]
print np.corrcoef(nb_pred,svm_pred)[0,1]
#####

### Ensemble Classifier improves upon svm and nb but not lr
pred_ens = (nb_pred + lr_pred + svm_pred) > 1
probs_ens = (nb_probs + lr_probs + svm_probs) > 1.5
get_err(pred_ens,y_test)
get_err(probs_ens,y_test)
###


## Custom ensemble...Doesn't improve
pred = np.max(np.c_[nb_probs,lr_probs,svm_probs],axis=1) > .65
get_err(pred,y_test)
###


### Histogram of probabilities
### NB is very skewed to the right
### LR is pretty uniformly distributed
### SVM is also pretty uniformly distributed
### ODD! NB is the most accurate in its confidence rating as well
nb_probs = nb_c.predict_proba(X_test)
lr_probs = lr_c.predict_proba(X_test)
svm_probs = svm.predict_proba(X_test)
nb_probs = np.max(nb_probs,axis=1)
lr_probs = np.max(lr_probs,axis=1)
svm_probs = np.max(svm_probs,axis=1)

plt.figure()
plt.title('SVM prediction probability')
plt.hist(svm_probs, bins='auto',color='blue')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.xlim([.5,1])
plt.show()

plt.figure()
plt.title('Logistic Regression prediction probability')
plt.hist(lr_probs, bins='auto',color='green')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.xlim([.5,1])
plt.show()

plt.figure()
plt.title('Naive Bayes prediction probability')
plt.hist(nb_probs, bins='auto',color='darkorange')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.xlim([.5,1])
plt.show()

###


### Compare the performance to confidence rating
### Can use this to talk about how trustworthy each 
### classifiers prediction rating is
ratings = np.arange(.5,1.1,.1)
nb_err = np.zeros([5,2])
lr_err = np.zeros([5,2])
svm_err = np.zeros([5,2])

for i in xrange(5):
	pl = ratings[i]
	ph = ratings[i+1]
	nb_inds = (nb_probs<=ph) & (nb_probs > pl)
	lr_inds = (lr_probs<=ph) & (lr_probs > pl)
	svm_inds = (svm_probs<=ph) & (svm_probs > pl)
	nb_err[i,0] = 1-get_err(nb_inds*nb_pred,nb_inds*y_test, n=nb_inds.sum()) 
	lr_err[i,0] = 1-get_err(lr_inds*lr_pred,lr_inds*y_test,n=lr_inds.sum()) 
	svm_err[i,0] = 1-get_err(svm_inds*svm_pred,svm_inds*y_test,n=svm_inds.sum()) 

	nb_err[i,1] = np.mean(nb_probs[nb_inds])
	lr_err[i,1] = np.mean(lr_probs[lr_inds])
	svm_err[i,1] = np.mean(svm_probs[svm_inds])



### ROC curves###
### Uses sklearn tutorial on how to create roc curve


nb_probs = nb_c.predict_proba(X_test)[:,1]
lr_probs = lr_c.predict_proba(X_test)[:,1]
svm_probs = svm.predict_proba(X_test)[:,1]
ens_probs = (nb_probs+lr_probs+svm_probs)/3


plt.figure()
lw = 2
fpr, tpr, _ = roc_curve(y_test,nb_probs)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,color='darkorange',lw=lw,label='Naive Bayes (area = %0.2f)' % roc_auc)
fpr, tpr, _ = roc_curve(y_test,svm_probs)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,color='blue',lw=lw,label='SVM (area = %0.2f)' % roc_auc)
fpr, tpr, _ = roc_curve(y_test,lr_probs)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,color='green',lw=lw,label='Logistic Regression (area = %0.2f)' % roc_auc)
fpr, tpr, _ = roc_curve(y_test,ens_probs)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,color='red',lw=lw,label='Ensemble (area = %0.2f)' % roc_auc)


plt.plot([0,1],[0,1],lw=lw,linestyle='--')
plt.xlim([0.0,.0])
plt.ylim([0.0,1.05])
plt.xlim([0.0,1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()


### Precision Recall
### Uses sklearn tutorial on how to create precision recal curve

nb_probs = nb_c.predict_proba(X_test)[:,1]
lr_probs = lr_c.predict_proba(X_test)[:,1]
svm_probs = svm.predict_proba(X_test)[:,1]
ens_probs = (nb_probs+lr_probs+svm_probs)/3


lw = 2
plt.clf()

precision, recall, _ = precision_recall_curve(y_test,nb_probs)
average_precision = average_precision_score(y_test,nb_probs)
plt.plot(recall, precision, lw=lw, color='darkorange',
         label='Naive Bayes (area = %0.2f)' % average_precision)
precision, recall, _ = precision_recall_curve(y_test,svm_probs)
average_precision = average_precision_score(y_test,nb_probs)
plt.plot(recall, precision, lw=lw, color='blue',
         label='SVM (area = %0.2f)' % average_precision)
precision, recall, _ = precision_recall_curve(y_test,lr_probs)
average_precision = average_precision_score(y_test,lr_probs)
plt.plot(recall, precision, lw=lw, color='green',
         label='Logistic Regression (area = %0.2f)' % average_precision)
precision, recall, _ = precision_recall_curve(y_test,ens_probs)
average_precision = average_precision_score(y_test,ens_probs)
plt.plot(recall, precision, lw=lw, color='red',
         label='Ensemble(area = %0.2f)' % average_precision)


plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall')
plt.legend(loc="lower left")
plt.show()



### Cross validation

def pnr(y_pred,y_test):
	tp = np.sum(y_pred*y_test)
	fp = np.sum(y_pred*(np.abs(1-y_test)))
	fn = np.sum(np.abs(1-y_pred)*y_test)

	return float(tp)/(tp+fp), float(tp)/(tp+fn)

bow = np.genfromtxt('out_bag_of_words_3.csv',delimiter=',')
bow_bigram = np.genfromtxt('out_bag_of_words_bigrams1.csv',delimiter=',')
classes = np.loadtxt('out_classes_3.txt')

bow_tfidf = bow_to_tfidf(bow)

X_train = bow
y_train = np.array(classes)
X_test,y_test = get_test_data()

X = np.r_[X_train, X_test]
y = np.r_[y_train, y_test]



k = 5
Train, Test = cross_val(k)

nb_err = np.zeros(k)
svm_err = np.zeros(k)
lr_err = np.zeros(k)
nb_err_train = np.zeros(k)
svm_err_train = np.zeros(k)
lr_err_train = np.zeros(k)
ens_err = np.zeros(k)
precision = np.zeros(k)
recall = np.zeros(k)

ens_err_train = np.zeros(k)
precision_train = np.zeros(k)
recall_train = np.zeros(k)

ens_err_fs = np.zeros(k)
precision_fs = np.zeros(k)
recall_fs = np.zeros(k)

ens_err_train_fs = np.zeros(k)
precision_train_fs = np.zeros(k)
recall_train_fs = np.zeros(k)


def get_MI(X_train):
	MI = np.zeros(X_train.shape[1])
	for f in range(X_train.shape[1]):
		feature_pred = flatten(X_train[:,f])
		MI[f] = sklearn.metrics.mutual_info_score(y_train,feature_pred)
	return MI


## Do the same with feature selection

k = 5
Train, Test = cross_val(k)
nb_err = np.zeros(k)
svm_err = np.zeros(k)
lr_err = np.zeros(k)
nb_err_train = np.zeros(k)
svm_err_train = np.zeros(k)
lr_err_train = np.zeros(k)
ens_err = np.zeros(k)
ens_err_prob = np.zeros(k)
for i in xrange(k):
	print i
	X_test, y_test = Test[i]
	X_train, y_train = Train[i]
	y_train = y_train.astype(int)
	y_test = y_test.astype(int)

	pos_inds = (y_train == 1)
	neg_inds = (y_train == 0)

	MI = np.zeros(X_train.shape[1])
	PD = np.zeros(X_train.shape[1])
	CHI = sklearn.feature_selection.chi2(X_train,y_train)[0]
	for f in range(X_train.shape[1]):
		feature_pred = flatten(X_train[:,f])
		MI[f] = sklearn.metrics.mutual_info_score(y_train,feature_pred)
		pdf = np.sum(X_train[pos_inds,f])
		ndf = np.sum(X_train[neg_inds,f])
		PD[f]  = np.abs(pdf-ndf)/(pdf+ndf)

	n = 200
	imp_inds = np.argsort(MI)
	#imp_inds = np.argsort(PD)
	#imp_inds = np.argsort(CHI)
	X_train = X_train[:,imp_inds[-n:]]
	X_test = X_test[:, imp_inds[-n:]]

	nb_c = naive_bayes.MultinomialNB()
	nb_c.fit(X_train,y_train)
	nb_err_train[i] = 1 - get_err(nb_c.predict(X_train).astype(int),y_train)
	nb_err[i] = 1-get_err(nb_c.predict(X_test),y_test)

	svm = SVC(C=1.0, kernel='linear',probability=True)
	svm.fit(X_train,y_train)
	svm_err_train[i] = 1 - get_err(svm.predict(X_train),y_train)	
	svm_err[i] = 1-get_err(svm.predict(X_test),y_test)

	lr_c = LogisticRegression(penalty='l2',C=1.0)
	lr_c.fit(X_train,y_train)
	lr_err_train[i] = 1 - get_err(lr_c.predict(X_train),y_train)
	lr_err[i] = 1-get_err(lr_c.predict(X_test),y_test)

	ens_err[i] = 1-get_err(pred_ens,y_test)
	ens_err_prob[i] = 1 - get_err(probs_ens,y_test)




