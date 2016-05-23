#!/usr/bin/python
from wordcloud import WordCloud, STOPWORDS
import os, pandas, argparse

if __name__ == '__main__':

    # arguments for input/output files
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input CSV file',
                        default='../datasets/train_set.csv')
    parser.add_argument('-o', '--output', help='Output directory',
                        default='generated_wordclouds/')
    parser.add_argument('-s', '--stopwords', help='Stopwords file',
                        default='../stop.txt')
    parser.add_argument('-d', '--display', help='Display created wordclouds',
                        action='store_true', default=False)
    args = parser.parse_args()

    try:
        df = pandas.read_csv(args.input, sep='\t')

        dir_out = args.output
        if dir_out[-1] != '/':
            dir_out += '/'

        # make the output dir if it doesn't exist
        if not os.path.exists(dir_out):
            os.mkdir(dir_out, 0755)

        print 'Input file:', args.input
        print 'Output directory:', dir_out
        print 'Stopwords:', args.stopwords
        
        # words to ignore
        stopwords = STOPWORDS.copy()
        with open(args.stopwords) as stopw_f:
            for word in stopw_f:
                stopwords.add(word.rstrip())

    except IOError as e:
        print 'Error:', str(e)
        exit(1)

    categories = dict()

    # gather all the content in one string for each category
    for index, row in df.iterrows():
        cat = row['Category']
        if cat not in categories:
            categories[cat] = ''
        categories[cat] += row['Content'] + ' '

    for (cat, texts) in categories.iteritems():
        path_out = dir_out + cat + '.png'
        print 'Generating wordcloud for category', cat, 'at', path_out
        wc = WordCloud(stopwords=stopwords).generate(texts)
        if args.display:
            wc.to_image().show()
        wc.to_file(path_out)
