import sys
sys.path.append('/Users/lucfaessler/Documents/Studium/Bachelor/Wirtschaftsinformatik/Semester 8 - SS25/Practical Course NLP/nlp_practical_2025_sEXism/code')

from preprocessing.normalization import *

def simple_preprocessing(text):
    text = lowercase(text)
    text = remove_mentions(text)
    text = remove_urls(text)
    text = remove_hashtags(text)
    return text.strip()


def preprocess_dataframe(df, column_name):
    df[column_name] = df[column_name].apply(simple_preprocessing)
    print(df[:10])
    return df