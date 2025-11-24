import re
import spacy
import itertools
import string
import pandas as pd
import nltk
import numpy as np
from typing import List, Union, Iterator
from spacy.tokens import Doc, Span, Token
from spacy.symbols import NOUN, PROPN, PRON
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
from spacy.tokenizer import Tokenizer
from spacy.language import Language
from nltk.corpus import stopwords as nltk_stopwords
from gensim.parsing.preprocessing import STOPWORDS as gensim_stopwords
from spacy.lang import char_classes
from spacy.matcher import Matcher
from spacy.util import filter_spans
from spacy.symbols import ORTH
from scispacy.consts import ABBREVIATIONS

nltk.download('stopwords')

# Load the English model from spaCy
nlp = spacy.load('en_core_sci_lg')
nlp.max_length = 10000000   # Increase maximum length for large documents

# Load the stopwords CSV
cleaning = pd.read_csv('/data/home/Zhuqian_He/originality/Data/stopwords.csv')
# cleaning = pd.read_csv('https://zenodo.org/records/13869486/files/stopwords.csv?download=1')

# Extract stopwords and removals from the file
stopwords = set(cleaning[cleaning['Type'] == 1]['Word'].unique())
removals = set(cleaning[cleaning['Type'] == 2]['Word'].unique())

# Customize stopwords further
stopwords.remove('anti')
stopwords = {s for s in stopwords if len(s) > 1}

# Combine stopwords with spaCy, Gensim, and NLTK
stopwords_spacy = nlp.Defaults.stop_words
stopwords_gensim = list(gensim_stopwords)
stopwords_nltk = nltk_stopwords.words('english')

# Combine all stopwords
combined_stopwords = set(itertools.chain(stopwords_spacy, stopwords_gensim, stopwords_nltk))
combined_stopwords.update(stopwords)

# Add all stopwords to spaCy's stopword list
for word in combined_stopwords:
    nlp.Defaults.stop_words.add(word)

# Extended punctuation for removal
# Be careful to keep hyphens '-' (not remove them)
extended_punctuation = r""".—!–"#$%&'()*+,./:;<=>?@[\]^_`{|}~¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿‽⁇⁈⁉‽⁇⁈⁉。。、、，、．．・・：：；；！！？？｡｡｢｢｣｣､､･･"""

def remove_new_lines(text: str) -> str:
    """
    Preprocess to remove newlines that split words. This function
       is intended to be called on a raw string before it is passed through a
       spaCy pipeline

    @param text: a string of text to be processed
    """
    text = text.replace("-\n\n", "")
    text = text.replace("- \n\n", "")
    text = text.replace("-\n", "")
    text = text.replace("- \n", "")
    return text

def combined_rule_prefixes() -> List[str]:
    """
    Helper function that returns the prefix pattern for the tokenizer.
    It is a helper function to accommodate spaCy tests that only test
    prefixes.
    """
    # Add lookahead assertions for brackets (may not work properly for unbalanced brackets)
    prefix_punct = char_classes.PUNCT.replace("|", " ")
    prefix_punct = prefix_punct.replace(r"\(", r"\((?![^\(\s]+\)\S+)")
    prefix_punct = prefix_punct.replace(r"\[", r"\[(?![^\[\s]+\]\S+)")
    prefix_punct = prefix_punct.replace(r"\{", r"\{(?![^\{\s]+\}\S+)")

    prefixes = (
        ["§", "%", "=", r"\+"]
        + char_classes.split_chars(prefix_punct)
        + char_classes.LIST_ELLIPSES
        + char_classes.LIST_QUOTES
        + char_classes.LIST_CURRENCY
        + char_classes.LIST_ICONS
    )
    return prefixes

def contains_digit(text: str) -> bool:
    """Check if a string contains any digits"""
    return any(char.isdigit() for char in text)

def split_special_chars(token: str) -> list:
    """Split slashes and hyphens"""
    subwords = []
    if '/' in token:
        parts = token.split('/')
        for part in parts:
            if '-' in part:
                subparts = part.split('-')
                subwords.extend(subparts)
            else:
                subwords.append(part)
    else:
        if '-' in token:
            subwords = token.split('-')
        else:
            subwords = [token]
    return [s.strip() for s in subwords if s.strip() and not all(c in string.punctuation for c in s.strip())]

def clean_token(token: str) -> str:
    """Cleaning logic for individual tokens (uses global variables directly)"""
    global stopwords, extended_punctuation
    if contains_digit(token):
        return ""
    if token.startswith('-') or token.endswith('-'):
        return ""
    token_clean = ''.join([c for c in token if c not in extended_punctuation])
    token_clean = token_clean.replace('\\', '').lower()
    if '--' in token_clean:
        return ""
    if len(token_clean) <= 1:
        return ""
    if token_clean in stopwords:
        return ""
    try:
        token_lemmatized = nlp(token_clean)[0].lemma_
        if token_lemmatized in stopwords or contains_digit(token_lemmatized):
            return ""
        return token_lemmatized
    except:
        return ""

def combined_rule_tokenizer(nlp: Language) -> Tokenizer:
    """
    Creates a custom tokenizer on top of spaCy's default tokenizer. The
    intended use of this function is to replace the tokenizer in a spaCy
    pipeline like so:

         nlp = spacy.load("some_spacy_model")
         nlp.tokenizer = combined_rule_tokenizer(nlp)

    @param nlp: a loaded spaCy model
    """
    
    # Remove the first hyphen to prevent tokenization of normal hyphens
    hyphens = char_classes.HYPHENS.replace("-|", "", 1)

    infixes = (
        char_classes.LIST_ELLIPSES
        + char_classes.LIST_ICONS
        + [
            r"×",   # Added this special '×' character to tokenize it separately
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}])\.(?=[{au}])".format(al=char_classes.ALPHA_LOWER, au=char_classes.ALPHA_UPPER),
            r"(?<=[{a}])\.(?=[{a}])".format(a=char_classes.ALPHA),
            r"(?<=[0-9])\.(?=[{a}])".format(a=char_classes.ALPHA),  
            r"(?<=[0-9)])\.(?=[{a}])".format(a=char_classes.ALPHA),  
            r"(?<=[{a}])\,(?=[{a}])".format(a=char_classes.ALPHA),
            r'(?<=[{a}"])[:<>=](?=[{a}])'.format(a=char_classes.ALPHA), # Removed '/' to prevent tokenization of slashes
            r"(?<=[0-9])(?=[a-zA-Z])",  # Split between digit and letter
            r"(?<=[a-zA-Z])(?=[0-9])",  # Split between letter and digit
            r"(?<=[0-9])(?=[/])",       # Split between digit and slash
            r"(?<=[/)(])(?=[0-9])",     # Split between slash/bracket and digit
        ]
    )

    prefixes = combined_rule_prefixes()
    
    # Add the right single quotation mark
    quotes = char_classes.LIST_QUOTES.copy() + ["’"]
    
    # Add lookbehind assertions for brackets (may not work properly for unbalanced brackets)
    suffix_punct = char_classes.PUNCT.replace("|", " ")

    suffixes = (
        char_classes.split_chars(suffix_punct)
        + char_classes.LIST_ELLIPSES
        + quotes
        + char_classes.LIST_ICONS
        + ["'s", "'S", "’s", "’S", "’s", "’S"]
        + [
            r"(?<=[0-9])\+",
            r"(?<=°[FfCcKk])\.",
            r"(?<=[0-9])(?:{})".format(char_classes.CURRENCY),
            # This is another place where we used a variable-width lookbehind
            # So now things like 'H3g' will be tokenized as ['H3', 'g']
            # Previously the lookbehind was (^[0-9]+)
            r"(?<=[0-9])(?:{u})".format(u=char_classes.UNITS),
            r"(?<=[0-9{}{}(?:{})])\.".format(
                char_classes.ALPHA_LOWER, r"%²\-\)\]\+", "|".join(quotes)
            ),
            # Add |\d to split off the period of a sentence that ends with 1D.
            r"(?<=[{a}|\d][{a}])\.".format(a=char_classes.ALPHA_UPPER),
        ]
    )

    infix_re = compile_infix_regex(infixes)
    prefix_re = compile_prefix_regex(prefixes)
    suffix_re = compile_suffix_regex(suffixes)

    # Update exclusions to include these abbreviations so the period is not split off
    exclusions = {abbreviation: [{ORTH: abbreviation}] for abbreviation in ABBREVIATIONS}
    tokenizer_exceptions = nlp.Defaults.tokenizer_exceptions.copy()
    tokenizer_exceptions.update(exclusions)

    tokenizer = Tokenizer(
        nlp.vocab,
        tokenizer_exceptions,
        prefix_search=prefix_re.search,
        suffix_search=suffix_re.search,
        infix_finditer=infix_re.finditer,
        token_match=nlp.tokenizer.token_match, # type: ignore
    )
    return tokenizer


nlp.tokenizer = combined_rule_tokenizer(nlp)

# Create the Matcher object
matcher = Matcher(nlp.vocab)
patterns = [
    # Pattern for a gerund verb followed by optional non-stopword adjectives and one or more nouns or proper nouns
    [{"POS": "VERB", "TAG": "VBG"},
     {"POS": "ADJ", "IS_STOP": False, "OP": "*"},
     {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}],
    # Pattern for one or more adjectives (excluding stopwords) followed by one or more nouns or proper nouns
    [{"POS": "ADJ", "IS_STOP": False, "OP": "*"}, 
     {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}],
    # Pattern for one or more nouns (excluding stopwords) including proper nouns
    [{"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}]
]

# Add patterns with the same ID as we process them uniformly
matcher.add("COMBINED_PATTERN", patterns)

def filter_mathml_and_invalid(text):
    """Filter out invalid characters and MathML tags"""
    text = re.sub(r'\b(msqrt|mn|math|mrow|mfrac|mi|mo)\b', '', text)
    text = re.sub(r'\bhttp\w+|www\w+\b', '', text)
    text = re.sub(r'\b[a-z]{2,4}\b(?![a-z0-9-])', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Extract and format words
def extract_words(doc: Doc) -> str:
    """
    @param doc: A spaCy Doc object
    @return: A string of cleaned, filtered words
    """
    global stopwords, extended_punctuation
    filtered_words = []
    for word in doc:
        token = word.text
        subwords = split_special_chars(token)
        for sub in subwords:
            clean_sub = clean_token(sub)  
            if clean_sub:
                filtered_words.append(clean_sub)
    filtered_words = list(set(filtered_words))  # Remove duplicates
    return ' '.join(filtered_words)

# Extract and format noun phrases
def extract_noun_phrases(doc: Doc) -> str:
    global stopwords, extended_punctuation, removals
    matches = matcher(doc)
    spans = filter_spans([doc[start:end] for _, start, end in matches])
    filtered_phrases = []
    for span in spans:
        lemmatized_phrase = []
        for token in span:
            subwords = split_special_chars(token.text)
            for sub in subwords:
                clean_sub = clean_token(sub)  
                if clean_sub and clean_sub not in removals:  
                    lemmatized_phrase.append(clean_sub)

        if not lemmatized_phrase or all(w in removals for w in lemmatized_phrase):
            continue

        # Remove leading and trailing stopwords
        lemmatized_phrase = [w for i, w in enumerate(lemmatized_phrase) if not (w in stopwords and all(x in stopwords for x in lemmatized_phrase[:i]))]
        lemmatized_phrase = [w for i, w in enumerate(lemmatized_phrase) if not (w in stopwords and all(x in stopwords for x in lemmatized_phrase[i:]))]

        # Filter out empty elements
        lemmatized_phrase = [w for w in lemmatized_phrase if w]
        if lemmatized_phrase:
            filtered_phrases.append(lemmatized_phrase)

    return ' '.join(['_'.join(phrase) for phrase in filtered_phrases if phrase])

# Process text and return cleaned string
def process_text(text: str, chunk: str) -> str:
    """Uniformly process text and return cleaned words or noun phrases"""
    clean_text = filter_mathml_and_invalid(text)
    doc = nlp(clean_text)
    if chunk == 'words':
        try:
            result = extract_words(doc)  
            return result if result.strip() else np.nan
        except Exception:
            return np.nan
    if chunk == 'phrases':
        try:
            result = extract_noun_phrases(doc)  
            return result if result.strip() else np.nan
        except Exception:
            return np.nan
