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

def flatten(f):
	a = np.zeros(len(f))
	for i in range(len(f)):
		a[i] = f[i]
	return a
def pnr(y_pred,y_test):
	tp = np.sum(y_pred*y_test)
	fp = np.sum(y_pred*(np.abs(1-y_test)))
	fn = np.sum(np.abs(1-y_pred)*y_test)

	return float(tp)/(tp+fp), float(tp)/(tp+fn)



def compute_stats():
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

	for i in xrange(k):

		X_test, y_test = Test[i]
		X_train, y_train = Train[i]
		y_train = y_train.astype(int)
		y_test = y_test.astype(int)
		print i
		nb_c = naive_bayes.MultinomialNB()
		nb_c.fit(X_train,y_train.astype(int))
		nb_pred = nb_c.predict(X_test)
		nb_pred_train = nb_c.predict(X_train)
		nb_err_train[i] = 1 - get_err(nb_c.predict(X_train),y_train)
		nb_err[i] = 1-get_err(nb_pred,y_test)
		svm = SVC(C=1.0, kernel='linear',probability=True)
		svm.fit(X_train,y_train)
		svm_pred = svm.predict(X_test)
		svm_pred_train = svm.predict(X_train)
		svm_err_train[i] = 1 - get_err(svm.predict(X_train),y_train)
		svm_err[i] = 1-get_err(svm_pred,y_test)
		lr_c = LogisticRegression(penalty='l2',C=1.0)
		lr_c.fit(X_train,y_train)
		lr_pred = lr_c.predict(X_test)
		lr_pred_train = lr_c.predict(X_train)
		lr_err_train[i] = 1 - get_err(lr_c.predict(X_train),y_train)
		lr_err[i] = 1-get_err(lr_pred,y_test)
		pred_ens = (nb_pred + lr_pred + svm_pred) > 1
		pred_ens_train = (nb_pred_train + lr_pred_train + svm_pred_train) > 1
		pre, rec = pnr(pred_ens,y_test)
		ens_err[i] = 1-get_err(pred_ens,y_test)
		precision[i] = pre
		recall[i] = rec
		pre_train,rec_train = pnr(pred_ens_train,y_train)
		ens_err_train[i] = 1 - get_err(pred_ens_train,y_train)
		precision_train[i] = pre_train
		recall_train[i] = rec_train
		pos_inds = (y_train == 1)
		neg_inds = (y_train == 0)
		MI = get_MI(X_train,y_train)
		n = 200
		imp_inds = np.argsort(MI)
		X_train = X_train[:,imp_inds[-n:]]
		X_test = X_test[:, imp_inds[-n:]]
		nb_c = naive_bayes.MultinomialNB()
		nb_c.fit(X_train,y_train.astype(int))
		nb_pred = nb_c.predict(X_test)
		nb_pred_train = nb_c.predict(X_train)
		svm = SVC(C=1.0, kernel='linear',probability=True)
		svm.fit(X_train,y_train)
		svm_pred = svm.predict(X_test)
		svm_pred_train = svm.predict(X_train)
		lr_c = LogisticRegression(penalty='l2',C=1.0)
		lr_c.fit(X_train,y_train)
		lr_pred = lr_c.predict(X_test)
		lr_pred_train = lr_c.predict(X_train)
		pred_ens = (nb_pred + lr_pred + svm_pred) > 1
		pred_ens_train = (nb_pred_train + lr_pred_train + svm_pred_train) > 1
		pre, rec = pnr(pred_ens,y_test)
		ens_err_fs[i] = 1-get_err(pred_ens,y_test)
		precision_fs[i] = pre
		recall_fs[i] = rec
		pre_train,rec_train = pnr(pred_ens_train,y_train)
		ens_err_train_fs[i] = 1 - get_err(pred_ens_train,y_train)
		precision_train_fs[i] = pre_train
		recall_train_fs[i] = rec_train

	return np.c_[precision,recall,precision_fs,recall_fs,ens_err,ens_err_fs,ens_err_train,ens_err_train_fs]


def get_MI(X_train,y_train):
	MI = np.zeros(X_train.shape[1])
	for f in range(X_train.shape[1]):
		feature_pred = flatten(X_train[:,f])
		MI[f] = sklearn.metrics.mutual_info_score(y_train,feature_pred)
	return MI
