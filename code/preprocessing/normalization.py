import pandas as pd
import re
import unicodedata
import emoji
import wordsegment

def lowercase_df(df: pd.DataFrame, params) -> pd.DataFrame:
    text_col = params['columns']['text']
    df[text_col] = df[text_col].str.lower()
    return df

def lowercase(texts: pd.Series) -> pd.Series:
    return texts.str.lower()

def remove_punctuation(texts: pd.Series, keep_sentiment_symbols=False) -> pd.Series:
    if keep_sentiment_symbols:
        return texts.str.replace(r"[^\w\s!?]", "", regex=True)
    else:
        return texts.str.replace(r"[^\w\s]", "", regex=True)

def normalize_whitespace(texts: pd.Series) -> pd.Series:
    return texts.str.replace(r"\s+", " ", regex=True).str.strip()

def remove_accents(texts: pd.Series) -> pd.Series:
    def _remove_accents(text):
        if not isinstance(text, str):
            return text
        return ''.join(
            c for c in unicodedata.normalize('NFKD', text)
            if not unicodedata.combining(c)
        )
    return texts.apply(_remove_accents)

def replace_numbers(texts: pd.Series, token: str = "<NUM>") -> pd.Series:
    return texts.str.replace(r"\d+", token, regex=True)

def normalize_unicode(texts: pd.Series) -> pd.Series:
    return texts.apply(lambda x: unicodedata.normalize('NFKC', x) if isinstance(x, str) else x)

def expand_contractions(texts: pd.Series) -> pd.Series:
    try:
        import contractions
    except ImportError:
        raise ImportError("Please install the contractions library: pip install contractions")

    return texts.apply(lambda x: contractions.fix(x) if isinstance(x, str) else x)

def remove_urls(texts: pd.Series) -> pd.Series:
    return texts.str.replace(r'https?://\S+', '', regex=True)

def remove_mentions(texts: pd.Series) -> pd.Series:
    mention_pattern = re.compile(r'@[A-Za-z0-9_]+')
    return texts.str.replace(mention_pattern, '', regex=True)

def convert_emojis_to_text(texts: pd.Series) -> pd.Series:
    return texts.apply(emoji.demojize)

def reduce_repeated_characters(texts: pd.Series) -> pd.Series:
    repeat_pattern = re.compile(r'(.)\1{2,}')
    return texts.str.replace(repeat_pattern, r'\1', regex=True)

def append_negation_suffix(texts: pd.Series) -> pd.Series:
    negation_pattern = re.compile(r"\b(?:not|no|never|n't)\b", flags=re.IGNORECASE)

    def _negate_text(text):
        if not isinstance(text, str):
            return text
        
        tokens = text.split()
        negating = False
        result = []

        for token in tokens:
            if negation_pattern.match(token):
                negating = True
                result.append(token)
            elif negating:
                if re.search(r'[.!?,;:]$', token):
                    result.append(f"{token.rstrip('.!?,;:')}_NOT{token[-1]}")
                    negating = False
                else:
                    result.append(f"{token}_NOT")
            else:
                result.append(token)

            if re.search(r'[.!?,;:]$', token):
                negating = False  # reset at punctuation

        return ' '.join(result)
    
    return texts.apply(_negate_text)

def split_hashtags(texts: pd.Series) -> pd.Series:
    def segment_hashtag(text):
        # Find all hashtags in the text
        hashtags = re.findall(r'#(\w+)', text)
        for tag in hashtags:
            # Segment the hashtag word and replace the original tag
            segmented_words = ' '.join(wordsegment.segment(tag))
            text = text.replace(f'#{tag}', segmented_words)
        return text

    return texts.apply(segment_hashtag)

def remove_hashtag_symbol(texts: pd.Series) -> pd.Series:
    return texts.str.replace(r"#(\w+)", r"\1", regex=True)

def split_hashtags(texts: pd.Series) -> pd.Series:
    from wordsegment import load, segment
    load()
    hashtag_pattern = re.compile(r"#(\w+)")
    def _split(text):
        if not isinstance(text, str): return text
        return hashtag_pattern.sub(lambda m: ' '.join(segment(m.group(1))), text)
    return texts.apply(_split)

def remove_hashtag_symbol(texts: pd.Series) -> pd.Series:
    return texts.str.replace(r"#(\w+)", r"\1", regex=True)


def lowercase(text):
    if isinstance(text, str):
        return text.lower()
    return text

def remove_mentions(text):
    if isinstance(text, str):
        return re.sub(r'@\w+', '', text)
    return text

def remove_urls(text):
    if isinstance(text, str):
        return re.sub(r"http\S+|www\S+|https\S+", '', text)
    return text

def remove_hashtags(text):
    if isinstance(text, str):
        return re.sub(r"#\w+", '', text)  # removes '#' but keeps the word
    return text

