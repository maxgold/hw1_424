import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, isdir, join
import numpy
import numpy as np
import re
import sys
import getopt
import codecs
import time
import os
import csv
from sklearn.feature_extraction.text import TfidfTransformer


chars = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '!', '.', ';', 
'?', '*', '\\', '\/', '~', '_','|','=','+','^',':','\"','\'','@','-']

def stem(word):
   regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
   stem, suffix = re.findall(regexp, word)[0]
   return stem

def unique(a):
   """ return the list with duplicate elements removed """
   return list(set(a))

def intersect(a, b):
   """ return the intersection of two lists """
   return list(set(a) & set(b))

def union(a, b):
   """ return the union of two lists """
   return list(set(a) | set(b))

def get_files(mypath):
   return [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

def get_dirs(mypath):
   return [ f for f in listdir(mypath) if isdir(join(mypath,f)) ]

# Reading a bag of words file back into python. The number and order
# of sentences should be the same as in the *samples_class* file.
def read_bagofwords_dat(myfile):
  bagofwords = numpy.genfromtxt('myfile.csv',delimiter=',')
  return bagofwords

def tokenize_corpus(path, train=True):

  porter = nltk.PorterStemmer() # also lancaster stemmer
  wnl = nltk.WordNetLemmatizer()
  stopWords = stopwords.words("english")
  classes = []
  samples = []
  docs = []
  if train == True:
    words = {}
  f = open(path, 'r')
  lines = f.readlines()

  for line in lines:
    classes.append(line.rsplit()[-1])
    samples.append(line.rsplit()[0])
    raw = line.decode('latin1')
    raw = ' '.join(raw.rsplit()[1:-1])
    # remove noisy characters; tokenize
    raw = re.sub('[%s]' % ''.join(chars), ' ', raw)
    tokens = word_tokenize(raw)
    tokens = [w.lower() for w in tokens]
    tokens = [w for w in tokens if w not in stopWords]
    tokens = [wnl.lemmatize(t) for t in tokens]
    tokens = [porter.stem(t) for t in tokens]   
    if train == True:
     for t in tokens: 
         try:
             words[t] = words[t]+1
         except:
             words[t] = 1
    docs.append(tokens)

  if train == True:
     return(docs, classes, samples, words)
  else:
     return(docs, classes, samples)


def wordcount_filter(words, num=5):
   keepset = []
   for k in words.keys():
       if(words[k] > num):
           keepset.append(k)
   print "Vocab length:", len(keepset)
   return(sorted(set(keepset)))

def bigram_filter(bigrams, num=5):
   keepset = []
   for k in bigrams.keys():
       if(bigrams[k] > num):
           keepset.append(k)
   print "Bigram vocab length:", len(keepset)
   return(sorted(set(keepset)))



def find_wordcounts(docs, vocab):
   bagofwords = numpy.zeros(shape=(len(docs),len(vocab)), dtype=numpy.uint8)
   vocabIndex={}
   for i in range(len(vocab)):
      vocabIndex[vocab[i]]=i

   for i in range(len(docs)):
       doc = docs[i]

       for t in doc:
          index_t=vocabIndex.get(t)
          if index_t>=0:
             bagofwords[i,index_t]=bagofwords[i,index_t]+1

   print "Finished find_wordcounts for:", len(docs), "docs"
   return(bagofwords)


def find_bigramcounts(docs,bigram_vocab):
  bagofwords = np.zeros(shape=(len(docs),len(bigram_vocab)),dtype=np.uint8)
  bigramIndex = {}
  for i in range(len(bigram_vocab)):
    bigramIndex[bigram_vocab[i]] = i

  for i in range(len(docs)):
    doc = docs[i]

    for t in xrange(len(doc)-1):
        bg = doc[t]+doc[t+1]  
        index_t = bigramIndex.get(bg)
        if index_t>=0:
          bagofwords[i,index_t] += 1

  return bagofwords



def get_test_data():
  path = '.'
  testtxt = 'test.txt'
  word_count_threshold = 3
  vocabf_test = 'out_vocab_3.txt'
  vocabfile = open('./out_vocab_3.txt', 'r')
  vocab_test = [line.rstrip('\n') for line in vocabfile]
  vocabfile.close()

  (docs, classes, samples, words) = tokenize_corpus(testtxt, train=True)
  X_test = find_wordcounts(docs, vocab_test)
  y_test = np.array(classes)

  return X_test,np.array(y_test.astype(int))

def get_test_data_bigram():
  path = '.'
  testtxt = 'test.txt'
  word_count_threshold = 3
  vocabf_test = 'out_vocab_3.txt'
  bigram_vocab = 'out_bigram_vocab_3.txt'
  vocabfile = open('./out_vocab_3.txt', 'r')
  vocab_test = [line.rstrip('\n') for line in vocabfile]
  vocabfile.close()

  bg_vocabfile = open('./out_bigram_vocab_3.txt')
  bg_vocab_test = [line.rstrip('\n') for line in bg_vocabfile]
  bg_vocabfile.close()

  (docs, classes, samples, words) = tokenize_corpus(testtxt, train=True)
  bow = find_wordcounts(docs, vocab_test)
  bow_bg = find_bigramcounts(docs, bg_vocab_test)
  X_test = np.c_[bow,bow_bg]
  y_test = np.array(classes)

  return X_test,np.array(y_test.astype(int))


def get_vocab():
  path = '.'
  testtxt = 'test.txt'
  word_count_threshold = 3
  vocabf_test = 'out_vocab_3.txt'
  vocabfile = open('./out_vocab_3.txt', 'r')
  vocab_test = [line.rstrip('\n') for line in vocabfile]
  vocabfile.close()

  return np.array(vocab_test)

def bow_to_tfidf(bow):
  tfidf_transformer = TfidfTransformer()
  bow_tfidf = tfidf_transformer.fit_transform(bow)
  return bow_tfidf.todense()


def get_train_docs():
  f = open('./train.txt', 'r')
  lines = f.readlines()
  f.close()
  return lines

def get_test_docs():
  f = open('./test.txt', 'r')
  lines = f.readlines()
  f.close()
  return lines


def tokenize_corpus_bigram(path, train=True):

  porter = nltk.PorterStemmer() # also lancaster stemmer
  wnl = nltk.WordNetLemmatizer()
  stopWords = stopwords.words("english")
  classes = []
  samples = []
  docs = []
  if train == True:
    words = {}
    bigrams = {}
  f = open(path, 'r')
  lines = f.readlines()

  for line in lines:
    classes.append(line.rsplit()[-1])
    samples.append(line.rsplit()[0])
    raw = line.decode('latin1')
    raw = ' '.join(raw.rsplit()[1:-1])
    # remove noisy characters; tokenize
    raw = re.sub('[%s]' % ''.join(chars), ' ', raw)
    tokens = word_tokenize(raw)
    tokens = [w.lower() for w in tokens]
    tokens = [w for w in tokens if w not in stopWords]
    tokens = [wnl.lemmatize(t) for t in tokens]
    tokens = [porter.stem(t) for t in tokens]   
    if train == True:
      for t in tokens: 
        try:
          words[t] = words[t]+1
        except:
          words[t] = 1
      
      for t in xrange(len(tokens)-1):
        bg = tokens[t]+tokens[t+1]  
        try:
          bigrams[bg] = bigrams[bg]+1
        except:
          bigrams[bg] = 1
      
    docs.append(tokens)

  if train == True:
     return(docs, classes, samples, bigrams, words)
  else:
     return(docs, classes, samples)



def cross_val(k, word_count_threshold = 3):
  trainpath = './train.txt'
  testpath = './test.txt'
  porter = nltk.PorterStemmer() # also lancaster stemmer
  wnl = nltk.WordNetLemmatizer()
  stopWords = stopwords.words("english")
  f1 = open(trainpath, 'r')
  f2 = open(testpath)
  lines1 = f1.readlines()
  lines2 = f2.readlines()

  lines = lines1 + lines2

  T = len(lines)
  sz = T/k

  Train = {}
  Test = {}

  for i in xrange(k):
    test_inds = np.random.choice(np.arange(T),sz,replace=False)
    train_inds = np.setdiff1d(np.arange(T),test_inds)

    classes, test_classes = [], []
    samples, test_samples = [], []
    docs, test_docs = [], []
    words = {}

    count = 0

    for line in lines:
      if count not in test_inds:
        classes.append(line.rsplit()[-1])
        samples.append(line.rsplit()[0])
        raw = line.decode('latin1')
        raw = ' '.join(raw.rsplit()[1:-1])
        # remove noisy characters; tokenize
        raw = re.sub('[%s]' % ''.join(chars), ' ', raw)
        tokens = word_tokenize(raw)
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if w not in stopWords]
        tokens = [wnl.lemmatize(t) for t in tokens]
        tokens = [porter.stem(t) for t in tokens]   
        for t in tokens: 
          try:
            words[t] = words[t]+1
          except:
            words[t] = 1
        docs.append(tokens)
      else:
        test_classes.append(line.rsplit()[-1])
        test_samples.append(line.rsplit()[0])
        raw = line.decode('latin1')
        raw = ' '.join(raw.rsplit()[1:-1])
        # remove noisy characters; tokenize
        raw = re.sub('[%s]' % ''.join(chars), ' ', raw)
        tokens = word_tokenize(raw)
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if w not in stopWords]
        tokens = [wnl.lemmatize(t) for t in tokens]
        tokens = [porter.stem(t) for t in tokens]   

        test_docs.append(tokens)
      count += 1

    vocab = wordcount_filter(words, num=word_count_threshold)
    bow = find_wordcounts(docs, vocab)
    bow_test = find_wordcounts(test_docs,vocab)
    Train[i] = (bow,np.array(classes))
    Test[i] = (bow_test,np.array(test_classes))

  return Train, Test



def main(argv):

  path = '.'
  outputf = 'out'
  vocabf = ''

  word_count_threshold = 3
  bigram_count_threshold = 1
  (docs, classes, samples, bigrams, words) = tokenize_corpus_bigram(traintxt, train=True)
  vocab = wordcount_filter(words, num=word_count_threshold)
  bigram_vocab = bigram_filter(bigrams,num=bigram_count_threshold)
  # Write new vocab file
  vocabf = outputf+"_vocab_"+str(word_count_threshold)+".txt"
  outfile = codecs.open(path+"/"+vocabf, 'w',"utf-8-sig")
  outfile.write("\n".join(vocab))
  outfile.close()

  bigram_vocabf = outputf+"_bigram_vocab_"+str(word_count_threshold)+".txt"
  outfile = codecs.open(path+"/"+bigram_vocabf, 'w',"utf-8-sig")
  outfile.write("\n".join(bigram_vocab))
  outfile.close()
  
  bow = find_wordcounts(docs, vocab)
  bow_bigrams = find_bigramcounts(docs,bigram_vocab)

  bow_f = np.c_[bow,bow_bigrams]

  # Write bow file
  with open(path+"/"+outputf+"_bag_of_words_bigrams"+str(bigram_count_threshold)+".csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(bow_f)

  # Write classes
  outfile= open(path+"/"+outputf+"_classes_"+str(word_count_threshold)+".txt", 'w')
  outfile.write("\n".join(classes))
  outfile.close()

  # Write samples
  outfile= open(path+"/"+outputf+"_samples_class_"+str(word_count_threshold)+".txt", 'w')
  outfile.write("\n".join(samples))
  outfile.close()

  
  # start_time = time.time()

  # path = '.'
  # outputf = 'out'
  # vocabf = ''

  # try:
  #  opts, args = getopt.getopt(argv,"p:o:v:",["path=","ofile=","vocabfile="])
  # except getopt.GetoptError:
  #   print 'Usage: \n python preprocessSentences.py -p <path> -o <outputfile> -v <vocabulary>'
  #   sys.exit(2)
  # for opt, arg in opts:
  #   if opt == '-h':
  #     print 'Usage: \n python preprocessSentences.py -p <path> -o <outputfile> -v <vocabulary>'
  #     sys.exit()
  #   elif opt in ("-p", "--path"):
  #     path = arg
  #   elif opt in ("-o", "--ofile"):
  #     outputf = arg
  #   elif opt in ("-v", "--vocabfile"):
  #     vocabf = arg

  # traintxt = path+"/train.txt"
  # print 'Path:', path
  # print 'Training data:', traintxt

  # # Tokenize training data (if training vocab doesn't already exist):
  # if (not vocabf):
  #   word_count_threshold = 3
  #   (docs, classes, samples, words) = tokenize_corpus(traintxt, train=True)
  #   vocab = wordcount_filter(words, num=word_count_threshold)
  #   # Write new vocab file
  #   vocabf = outputf+"_vocab_"+str(word_count_threshold)+".txt"
  #   outfile = codecs.open(path+"/"+vocabf, 'w',"utf-8-sig")
  #   outfile.write("\n".join(vocab))
  #   outfile.close()
  # else:
  #   word_count_threshold = 0
  #   (docs, classes, samples) = tokenize_corpus(traintxt, train=False)
  #   vocabfile = open(path+"/"+vocabf, 'r')
  #   vocab = [line.rstrip('\n') for line in vocabfile]
  #   vocabfile.close()

  # print 'Vocabulary file:', path+"/"+vocabf 

  # # Get bag of words:
  # bow = find_wordcounts(docs, vocab)
  # # Check: sum over docs to check if any zero word counts
  # print "Doc with smallest number of words in vocab has:", min(numpy.sum(bow, axis=1))

  # # Write bow file
  # with open(path+"/"+outputf+"_bag_of_words_"+str(word_count_threshold)+".csv", "wb") as f:
  #   writer = csv.writer(f)
  #   writer.writerows(bow)

  # # Write classes
  # outfile= open(path+"/"+outputf+"_classes_"+str(word_count_threshold)+".txt", 'w')
  # outfile.write("\n".join(classes))
  # outfile.close()

  # # Write samples
  # outfile= open(path+"/"+outputf+"_samples_class_"+str(word_count_threshold)+".txt", 'w')
  # outfile.write("\n".join(samples))
  # outfile.close()

  # print 'Output files:', path+"/"+outputf+"*"

  # # Runtime
  # print 'Runtime:', str(time.time() - start_time)

if __name__ == "__main__":
  main(sys.argv[1:])

 
