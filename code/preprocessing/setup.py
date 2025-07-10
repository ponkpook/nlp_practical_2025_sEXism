# your_project/core/preprocessing.py

from functools import partial
from registry import register_preprocessing

# --- Import all your real implementation functions ---
from preprocessing.normalization import (
    lowercase,
    remove_punctuation,
    normalize_whitespace,
    remove_accents,
    replace_numbers,
    expand_contractions,
    append_negation_suffix,
    remove_urls,
    remove_mentions,
    convert_emojis_to_text,
    split_hashtags,
    remove_hashtag_symbol
)

# --- Register each function as a preprocessing step ---

# For functions with no parameters
@register_preprocessing("lowercase")
def build_lowercase_step(params):
    return lowercase

@register_preprocessing("normalize_whitespace")
def build_normalize_whitespace_step(params):
    return normalize_whitespace

@register_preprocessing("remove_accents")
def build_remove_accents_step(params):
    return remove_accents

@register_preprocessing("expand_contractions")
def build_expand_contractions_step(params):
    return expand_contractions

@register_preprocessing("append_negation_suffix")
def build_append_negation_suffix_step(params):
    return append_negation_suffix
    
@register_preprocessing("remove_urls")
def build_remove_urls_step(params):
    return remove_urls

@register_preprocessing("remove_mentions")
def build_remove_mentions_step(params):
    return remove_mentions

@register_preprocessing("convert_emojis_to_text")
def build_convert_emojis_to_text_step(params):
    return convert_emojis_to_text

# For functions that take parameters from the YAML config
@register_preprocessing("remove_punctuation")
def build_remove_punctuation_step(params):
    return partial(remove_punctuation, **params)

@register_preprocessing("replace_numbers")
def build_replace_numbers_step(params):
    return partial(replace_numbers, **params)

# For steps with different "modes" (more advanced)
@register_preprocessing("handle_hashtags")
def build_handle_hashtags_step(params):
    mode = params.get("mode")
    if mode == "split":
        return split_hashtags
    elif mode == "remove_symbol":
        return remove_hashtag_symbol
    else:
        raise ValueError(f"Unsupported hashtag mode: '{mode}'. Choose 'split' or 'remove_symbol'.")
    


from preprocessing.tokenization import (
    whitespace_tokenize,
    regex_word_tokenize,
    nltk_word_tokenize,
    char_tokenize,
    char_ngram_tokenize,
    tweet_tokenize
)

@register_preprocessing("whitespace")
def build_whitespace_tokenize_step(params):
    return whitespace_tokenize

@register_preprocessing("regex")
def build_regex_word_tokenize_step(params):
    return regex_word_tokenize

@register_preprocessing("word")
def build_nltk_word_tokenize_step(params):
    return nltk_word_tokenize

@register_preprocessing("char")
def build_char_tokenize_step(params):
    return char_tokenize

@register_preprocessing("char_ngram")
def build_char_ngram_tokenize_step(params):
    return char_ngram_tokenize

@register_preprocessing("tweet")
def build_tweet_tokenize_step(params):
    return tweet_tokenize