from wordcloud import WordCloud
import pandas

if __name__ == '__main__':
    df = pandas.read_csv('../datasets/train_set.csv', sep='\t')

    categories = dict()

    for index, row in df.iterrows():
        cat = row['Category']
        if cat not in categories:
            categories[cat] = ''
        categories[cat] += row['Content'] + ' '

    for (cat, texts) in categories.iteritems():
        print cat, texts
        wc = WordCloud().generate(texts)
        wc.to_file('generated_wordclouds/' + cat)