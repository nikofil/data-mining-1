#!/usr/bin/python
import argparse

import pandas
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import  BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':

    # arguments for input/output files
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', help='Input train CSV file',
                        default='../datasets/train_set.csv')
    parser.add_argument('-i', '--test', help='Input test CSV file',
                        default='../datasets/test_set.csv')
    parser.add_argument('-o', '--output', help='Output CSV file',
                        default='classification.csv')
    parser.add_argument('-s', '--stopwords', help='Stopwords file',
                        default='../stop.txt')
    args = parser.parse_args()

    try:
        df = pandas.read_csv(args.train, sep='\t')

        traintexts = [line['Content'] for _, line in df.iterrows()]
        categs = [line['Category'] for _, line in df.iterrows()]
        labelenc = LabelEncoder()
        cat_encoded = labelenc.fit_transform(categs)
        cat_names = list(set(categs))

        print 'Train set:', args.train
        print 'Test set:', args.test
        print 'Stopwords:', args.stopwords

        # read the stopwords from the file
        with open(args.stopwords) as fstop:
            stopwords = [l.strip() for l in fstop.readlines()]

        vectorizer = TfidfVectorizer(stop_words=stopwords, sublinear_tf=True, use_idf=True)
        svd = TruncatedSVD(n_components=20)

        print('')
        print('Transforming input')
        # vectorize and transform
        points = vectorizer.fit_transform(traintexts)
        points_lsa = svd.fit_transform(points)

        nb = MultinomialNB()

        scores = cross_val_score(nb, points, cat_encoded, cv=10)
        print scores.mean(), scores.std()

        nb = BernoulliNB()

        scores = cross_val_score(nb, points, cat_encoded, cv=10)
        print scores.mean(), scores.std()

        nb = KNeighborsClassifier()

        scores = cross_val_score(nb, points_lsa, cat_encoded, cv=10)
        print scores.mean(), scores.std()

        nb = RandomForestClassifier()

        scores = cross_val_score(nb, points_lsa, cat_encoded, cv=10)
        print scores.mean(), scores.std()

        nb = LinearSVC()

        scores = cross_val_score(nb, points_lsa, cat_encoded, cv=10)
        print scores.mean(), scores.std()

        # write output to csv
        # print('')
        # print('Writing output')
        # df = pandas.DataFrame(out_data, index=['Cluster'+str(i+1) for i in xrange(5)])
        # df.to_csv(args.output, sep='\t')

        print('Finished!')

    except IOError as e:
        print 'Error:', str(e)
        exit(1)
