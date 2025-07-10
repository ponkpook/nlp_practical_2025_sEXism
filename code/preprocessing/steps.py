# steps.py
import re
import pandas as pd

def strip_urls(texts: pd.Series) -> pd.Series:
    return texts.str.replace(r'https?://\S+', '', regex=True)

def lowercase(texts: pd.Series) -> pd.Series:
    return texts.str.lower()

def word_tokenizer(texts: pd.Series) -> pd.Series:
    return texts.str.split()