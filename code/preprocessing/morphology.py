import pandas as pd
import nltk
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
import spacy

# Download required NLTK resources (ensure this runs once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# Initialize tools
_porter = PorterStemmer()
_snowball = SnowballStemmer('english')
_lancaster = LancasterStemmer()
_wnl = WordNetLemmatizer()
_spacy_nlp = spacy.load("en_core_web_sm")


class Morphology:

    @staticmethod
    def porter_stem(texts: pd.Series) -> pd.Series:
        return texts.apply(lambda tokens: [_porter.stem(t) for t in tokens])

    @staticmethod
    def snowball_stem(texts: pd.Series) -> pd.Series:
        return texts.apply(lambda tokens: [_snowball.stem(t) for t in tokens])

    @staticmethod
    def lancaster_stem(texts: pd.Series) -> pd.Series:
        return texts.apply(lambda tokens: [_lancaster.stem(t) for t in tokens])

    @staticmethod
    def wordnet_lemmatize(texts: pd.Series) -> pd.Series:
        def _nltk_pos_tag_to_wordnet(nltk_tag):
            if nltk_tag.startswith('J'):
                return wordnet.ADJ
            elif nltk_tag.startswith('V'):
                return wordnet.VERB
            elif nltk_tag.startswith('N'):
                return wordnet.NOUN
            elif nltk_tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN

        def lemmatize_tokens(tokens):
            pos_tags = nltk.pos_tag(tokens)
            return [_wnl.lemmatize(t, _nltk_pos_tag_to_wordnet(tag)) for t, tag in pos_tags]

        return texts.apply(lemmatize_tokens)

    @staticmethod
    def spacy_lemmatize(texts: pd.Series) -> pd.Series:
        return texts.apply(lambda tokens: [token.lemma_ for token in _spacy_nlp(" ".join(tokens))])
