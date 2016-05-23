#!/usr/bin/python
import argparse
import os
import pandas
import operator

from sklearn import grid_search
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve

if __name__ == '__main__':

    # arguments for input/output files
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', help='Input train CSV file',
                        default='../datasets/train_set.csv')
    parser.add_argument('-i', '--test', help='Input test CSV file',
                        default='../datasets/test_set.csv')
    parser.add_argument('-s', '--stopwords', help='Stopwords file',
                        default='../stop.txt')
    args = parser.parse_args()

    try:
        traindf = pandas.read_csv(args.train, sep='\t')
        testdf = pandas.read_csv(args.test, sep='\t')

        traintexts = [(line['Title'], line['Content'])
                      for _, line in traindf.iterrows()]
        categs = [line['Category'] for _, line in traindf.iterrows()]
        labelenc = LabelEncoder()
        cat_encoded = labelenc.fit_transform(categs)
        cat_names = list(set(categs))

        testtexts = [(line['Title'], line['Content'])
                      for _, line in testdf.iterrows()]
        testids = [line['Id'] for _, line in testdf.iterrows()]

        print 'Train set:', args.train
        print 'Test set:', args.test
        print 'Stopwords:', args.stopwords

        # read the stopwords from the file
        with open(args.stopwords) as fstop:
            stopwords = [l.strip() for l in fstop.readlines()]

        title_vectorizer = TfidfVectorizer(preprocessor=operator.itemgetter(0),
                                           stop_words=stopwords,
                                           sublinear_tf=True, use_idf=True)

        text_vectorizer = TfidfVectorizer(preprocessor=operator.itemgetter(1),
                                           stop_words=stopwords,
                                           sublinear_tf=True, use_idf=True)

        union = FeatureUnion([('title', title_vectorizer),
                              ('text', text_vectorizer)])

        svd = TruncatedSVD(n_components=50)

        print('')
        print('Transforming input')
        # vectorize and transform
        points = union.fit_transform(traintexts)
        points_lsa = svd.fit_transform(points)

        testpoints = union.transform(testtexts)
        testpoints_lsa = svd.transform(testpoints)

        evaluationresult = {'Statistic Measure': ['Accuracy', 'ROC']}

        for clf, name, use_lsa in [
            (BernoulliNB(alpha=0.001), 'BinomialNB', False),
            (MultinomialNB(alpha=0.001), 'MultinomialNB', False),
            (KNeighborsClassifier(n_neighbors=5), 'KNN', True),
            (RandomForestClassifier(n_estimators=10), 'RandomForest', True),
            (SVC(C=1293), 'SVC', True)
        ]:
            points_used = points_lsa if use_lsa else points
            scores = cross_val_score(clf,
                                     points_used,
                                     cat_encoded,
                                     cv=10)
            print '{} accuracy: {}, deviation: {}'.format(
                name,
                str(scores.mean())[:5],
                str(scores.std())[:5])

            evaluationresult.update({name: [scores.mean(), 1]})

            out_dir = name + '/'
            if not os.path.exists(out_dir):
                os.mkdir(out_dir, 0755)

            clf.fit(points_used, cat_encoded)
            test_predics = labelenc.inverse_transform(clf.predict(
                testpoints_lsa if use_lsa else testpoints)).tolist()

            traindf = pandas.DataFrame({'Id': testids,
                                        'Predicted Category': test_predics})
            traindf.to_csv(out_dir + 'testSet_categories.csv', sep='\t')

        print('')
        print('Writing output')
        evaldf = pandas.DataFrame(evaluationresult)
        evaldf.to_csv('EvaluationMetric_10fold.csv', sep='\t')

        print('Finished!')

    except IOError as e:
        print 'Error:', str(e)
        exit(1)
