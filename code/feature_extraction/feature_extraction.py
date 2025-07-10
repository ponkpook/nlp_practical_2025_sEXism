import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#import gensim.downloader as api
#import gensim
import nltk
from . import register_vectorizer

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

class Vectorizers:
    @staticmethod
    def count_vectorize(texts: pd.Series, ngram_range=(1,1)):
        vectorizer = CountVectorizer(ngram_range=ngram_range)
        return vectorizer.fit_transform(texts), vectorizer

    @staticmethod
    def tfidf_vectorize(texts: pd.Series, ngram_range=(1,1), pos_filter=None):
        if pos_filter is None:
            vectorizer = TfidfVectorizer(ngram_range=ngram_range)
            return vectorizer.fit_transform(texts), vectorizer
        
        def pos_filter_text(text):
            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)
            filtered = [word for word, pos in tagged if any(pos.startswith(tag) for tag in pos_filter)]
            return " ".join(filtered)
        
        filtered_texts = texts.apply(pos_filter_text)
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        return vectorizer.fit_transform(filtered_texts), vectorizer


@register_vectorizer("count")
class CountVectorizerWrapper:
    def __init__(self, ngram_range=(1, 1)):
        self.vectorizer = CountVectorizer(ngram_range=ngram_range)

    def fit(self, texts: pd.Series):
        self.vectorizer.fit(texts)

    def transform(self, texts: pd.Series):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: pd.Series):
        return self.vectorizer.fit_transform(texts)


@register_vectorizer("tfidf")
class TfidfVectorizerWrapper:
    def __init__(self, ngram_range=(1, 1), pos_filter=None):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        self.pos_filter = pos_filter

    def _pos_filter_text(self, text):
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        filtered = [word for word, pos in tagged if any(pos.startswith(tag) for tag in self.pos_filter)]
        return " ".join(filtered)

    def _maybe_filter(self, texts: pd.Series):
        if self.pos_filter is None:
            return texts
        return texts.apply(self._pos_filter_text)

    def fit(self, texts: pd.Series):
        self.vectorizer.fit(self._maybe_filter(texts))

    def transform(self, texts: pd.Series):
        return self.vectorizer.transform(self._maybe_filter(texts))

    def fit_transform(self, texts: pd.Series):
        return self.vectorizer.fit_transform(self._maybe_filter(texts))


@staticmethod
def word2vec_vectorize(texts: pd.Series, model_name='word2vec-google-news-300') -> pd.DataFrame:
    #model = api.load(model_name)
    
    #def document_vector(tokens):
    #    valid_tokens = [t for t in tokens if t in model]
    #    if len(valid_tokens) == 0:
    #        return [0]*model.vector_size
    #    return list(pd.np.mean([model[t] for t in valid_tokens], axis=0))
    
    #vectors = texts.apply(document_vector)
    #return pd.DataFrame(vectors.tolist())
    return None


def glove_vectorize(texts: pd.Series, model_name='glove-wiki-gigaword-100') -> pd.DataFrame:
    #model = api.load(model_name)
    
    #def document_vector(tokens):
    #    valid_tokens = [t for t in tokens if t in model]
    #    if len(valid_tokens) == 0:
    #        return [0]*model.vector_size
    #    return list(pd.np.mean([model[t] for t in valid_tokens], axis=0))
    
    #vectors = texts.apply(document_vector)
    #return pd.DataFrame(vectors.tolist())
    return None
