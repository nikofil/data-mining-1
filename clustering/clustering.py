#!/usr/bin/python
import argparse
import itertools
import operator
from math import acos, sqrt
from random import random

import pandas
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

if __name__ == '__main__':

    # cosine similarity
    def cos_sim(v1, v2):
        dot = sum(map(operator.mul, v1, v2))
        s1 = sum([x * x for x in v1])
        s2 = sum([x * x for x in v2])
        denom = sqrt(s1 * s2)
        return dot / denom if denom != 0 else 1

    def k_means(points, n=5, threshold=0.005, comp=cos_sim):
        # find the dimension of the points
        dim = len(points[0])
        # start with random means
        means = [[random() * 2 - 1 for i in xrange(dim)] for k in xrange(n)]
        # points don't belong to a cluster yet
        clust = [-1] * len(points)

        changing = True
        while changing:
            for i, p in enumerate(points):
                # find the distance of current point to each cluster
                dists = [comp(p, mean) for mean in means]
                # assign point to closest cluster (largest cos)
                clust[i] = max(enumerate(dists), key=operator.itemgetter(1))[0]

            # add all the points' positions for each cluster
            mean_sums = [[0] * dim for i in xrange(n)]
            mean_counts = [0] * n
            for i, cl in enumerate(clust):
                mean_sums[cl] = map(operator.add, mean_sums[cl], points[i])
                mean_counts[cl] += 1

            # average the positions to find the new means
            # if a cluster has no points, move it to a random position
            new_means = [map(lambda s: s / cnt, summ) if cnt > 0
                         else [random() * 2 - 1 for i in xrange(dim)]
                         for summ, cnt in zip(mean_sums, mean_counts)]
            # how much the means moved (squared distance to previous position)
            diffs = map(lambda x, y: sum([z * z for
                                          z in map(operator.sub, x, y)]), means, new_means)
            means = new_means

            # if none of the means changed more than the threshold, return
            if max(diffs) < threshold:
                changing = False
        return clust


    # arguments for input/output files
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input CSV file',
                        default='../datasets/train_set.csv')
    parser.add_argument('-o', '--output', help='Output CSV file',
                        default='clustering_KMeans.csv')
    parser.add_argument('-s', '--stopwords', help='Stopwords file',
                        default='../stop.txt')
    args = parser.parse_args()

    try:
        df = pandas.read_csv(args.input, sep='\t')

        texts = [line['Content'] for _, line in df.iterrows()]
        categs = [line['Category'] for _, line in df.iterrows()]
        cat_names = list(set(categs))

        print 'Input file:', args.input
        print 'Output file:', args.output
        print 'Stopwords:', args.stopwords

        # read the stopwords from the file
        with open(args.stopwords) as fstop:
            stopwords = [l.strip() for l in fstop.readlines()]

        vectorizer = CountVectorizer(stop_words=stopwords)
        transformer = TfidfTransformer()
        svd = TruncatedSVD(n_components=100)

        # vectorize, transform and perform SVD
        pipeline = Pipeline([
            ('vect', vectorizer),
            ('tran', transformer),
            ('svd', svd)
        ])

        print('')
        print('Transforming input')
        points = pipeline.fit_transform(texts)

        # assign a cluster to each point
        print('Running k-means')
        res = k_means(points)

        out_data = {cat: [] for cat in cat_names}

        # group points by their assigned cluster
        print('Gathering results')
        print('')
        for idx, group in itertools.groupby(
                sorted(enumerate(res), key=operator.itemgetter(1)),
                operator.itemgetter(1)):
            # group_points is an array points in the current cluster
            group_points = map(operator.itemgetter(0), group)

            # get the category each point in the cluster really belongs to
            real_categories = [categs[p] for p in group_points]

            # count how many points there are for each category
            cat_counts = [(name, real_categories.count(name))
                          for name in cat_names]

            # calculate the percentages of points in this cluster
            # belonging to each category
            cat_pcts = [(name, str(float(cnt) / len(real_categories))[:5])
                        for name, cnt in cat_counts]

            print('Cluster {}: ({})'.format(idx + 1, ', '.join(
                [name + ': ' + perc for (name, perc) in cat_pcts])))

            # append results to output data
            for cat_name, pct in cat_pcts:
                out_data[cat_name].append(pct)

        # write output to csv
        print('')
        print('Writing output')
        df = pandas.DataFrame(out_data, index=['Cluster'+str(i+1) for i in xrange(5)])
        df.to_csv(args.output, sep='\t')

        print('Finished!')

    except IOError as e:
        print 'Error:', str(e)
        exit(1)
