# Paper 1 — Porter Stemming Algorithm

## Citation

Porter, M. F. (1980). **An algorithm for suffix stripping**. *Program*, 14(3), 130–137.  
https://doi.org/10.1108/eb046814  
(Also widely mirrored; NLTK implements this public-domain algorithm.)

## Problem

English IR systems face **morphological variation**: *treat / treating / treatment* should often match the same concept. Without normalization, inverted indexes fragment related terms and hurt recall.

## Method

The **Porter stemmer** applies ordered suffix-stripping rules in steps (e.g. plurals, -ed/-ing, -ization → -ize). It is rule-based (not a dictionary lemmatizer), fast, and language-specific to English.

## Results / impact

Became a default baseline in classical IR for decades. Improves recall on many collections; can over-stem (e.g. *university/universe*) and hurt precision slightly — a known trade-off discussed in IR courses.

## Link to our system

Our pipeline uses NLTK’s `PorterStemmer` on both documents and queries before indexing / matching, so Boolean and TF-IDF operate on stemmed terms (e.g. *treatment* → *treatment*/*treat* depending on rules). This directly implements the “public-domain stemming” requirement of Project #1.
