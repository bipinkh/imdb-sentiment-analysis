from os import listdir
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import string

# uncomment this while submitting to vocareum
# train_path = "../resource/lib/publicdata/aclImdb/train/"
# test_path = "../resource/lib/publicdata/imdb_te.csv"

#comment this while submitting to vocareum
train_path = "/data/train"
test_path = "/data/imdb_te.csv"

#let's load stop words
stopwordsFile = open("stopwords.en.txt", "r", encoding="utf8")
stopwords = stopwordsFile.read()
stopwordsFile.close()
stopwords = stopwords.split("\n")


def core(string_set):
    words = string_set
    words = words.replace("<br />", " ")
    words = words.rstrip()
    replace = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    words = words.translate(replace)
    words = words.lower()
    words = words.split()
    words = [word for word in words if word not in stopwords]
    words = ' '.join(words)
    return words


def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    file = open(name, "w", encoding='utf8')
    file.write("row_number,text,polarity\n")

    count = 0
    for text in listdir(inpath + "pos"):
        fileInput = open(inpath + "pos/" + text, "r", encoding='utf8')
        text = core(fileInput.read())
        file.write(str(count) + "," + text + ",1" + "\n")
        count += 1
        fileInput.close()

    for text in listdir(inpath + "neg"):
        fileInput = open(inpath + "neg/" + text, "r", encoding='utf8')
        text = core(fileInput.read())
        file.write(str(count) + "," + text + ",0" + "\n")
        count += 1
        fileInput.close()

def unigram():

    countVector = CountVectorizer(stop_words=stopwords)
    transformedTrainData = countVector.fit_transform(training['text'])
    classifier = SGDClassifier(loss="hinge", penalty="l1")
    classifier.fit(transformedTrainData, training['polarity'])

    # Output classification for test set
    transformedTestData = countVector.transform(test_set['text'])
    results = classifier.predict(transformedTestData)
    with open("unigram.output.txt", "w") as f:
        for result in results:
            f.write(str(result) + "\n")


def unigramtdidf():

    countVector = TfidfVectorizer(stop_words=stopwords)
    transformedTrainData = countVector.fit_transform(training['text'])
    classifier = SGDClassifier(loss="hinge", penalty="l1")
    classifier.fit(transformedTrainData, training['polarity'])

    # Output classification for test set
    transformedTestData = countVector.transform(test_set['text'])
    results = classifier.predict(transformedTestData)
    with open("unigramtfidf.output.txt", "w") as f:
        for result in results:
            f.write(str(result) + "\n")

def bigram():

    countVector = CountVectorizer(stop_words=stopwords, ngram_range=(1, 2))
    transformedTrainData = countVector.fit_transform(training['text'])
    classifier = SGDClassifier(loss="hinge", penalty="l1")
    classifier.fit(transformedTrainData, training['polarity'])

    # Output classification for test set
    transformedTestData = countVector.transform(test_set['text'])
    results = classifier.predict(transformedTestData)
    with open("bigram.output.txt", "w") as f:
        for result in results:
            f.write(str(result) + "\n")


def bigramtdidf():

    countVector = TfidfVectorizer(stop_words=stopwords, ngram_range=(1, 2))
    transformedTrainData = countVector.fit_transform(training['text'])
    classifier = SGDClassifier(loss="hinge", penalty="l1")
    classifier.fit(transformedTrainData, training['polarity'])

    # Output classification for test set
    transformedTestData = countVector.transform(test_set['text'])
    results = classifier.predict(transformedTestData)
    with open("bigramtfidf.output.txt", "w") as f:
        for result in results:
            f.write(str(result) + "\n")


if "__main__" == __name__:

    imdb_data_preprocess(train_path)

    training = pd.read_csv("imdb_tr.csv")
    test_set = pd.read_csv(test_path, encoding="ISO-8859-1")
    test_set['text'] = test_set['text'].apply(core)

    unigram()
    unigramtdidf()
    bigram()
    bigramtdidf()





