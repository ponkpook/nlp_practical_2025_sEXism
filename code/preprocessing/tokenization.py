import pandas as pd
import re
import nltk
from nltk.tokenize import TweetTokenizer

# Ensure NLTK punkt is available
nltk.download('punkt')

def whitespace_tokenize(texts: pd.Series) -> pd.Series:
    return texts.apply(lambda x: x.split())

def regex_word_tokenize(texts: pd.Series) -> pd.Series:
    pattern = re.compile(r'\b\w+\b')
    return texts.apply(lambda x: pattern.findall(x))

def nltk_word_tokenize(texts: pd.Series) -> pd.Series:
    return texts.apply(nltk.word_tokenize)

def char_tokenize(texts: pd.Series) -> pd.Series:
    return texts.apply(list)

def char_ngram_tokenize(texts: pd.Series, n: int = 3) -> pd.Series:
    def ngrams(text):
        text = text.replace(" ", "_")  # replace space to preserve word boundaries
        return [text[i:i+n] for i in range(len(text) - n + 1)]
    return texts.apply(ngrams)

def tweet_tokenize(texts: pd.Series) -> pd.Series:
    tokenizer = TweetTokenizer()
    return texts.apply(lambda x: tokenizer.tokenize(x))