"""Page 1 (≈1 min): dataset, vocabulary, preprocessing steps and our Porter stemmer."""
import streamlit as st

from src.hw2_answers import part6_answer
from ui.hw2 import common, corpus as corpus_sections, porter, preprocessing


def render(corpus, result):
    with common.card('Corpus and vocabulary (Dataset, Part II)', 'One PubMed subject, English abstracts only.'):
        corpus_sections.dataset_metrics(corpus, result)
        corpus_sections.vocabulary_overview(result)
    with common.card('Steps 1–5 on one sentence (Part I)', 'Edit the sentence to show any case: capitals, punctuation, stopwords, suffixes.'):
        preprocessing.sandbox()
    with common.card('Our Porter stemmer, rule by rule', 'Original 1980 rules implemented in src/porter.py; each changed step names the rule that fired.'):
        porter.word_steps()
        porter.agreement(corpus, result)

    common.more()
    with st.expander('Source query and scope'):
        corpus_sections.source(corpus)
    with st.expander('Vocabulary table with hapax and Zipf k'):
        corpus_sections.vocabulary_table(result)
    with st.expander('Browse the 1,000 documents'):
        corpus_sections.browse(corpus)
    with st.expander('Pipeline steps and condition definitions'):
        preprocessing.definitions()
    with st.expander('Trace a real abstract through A–D'):
        preprocessing.abstract_trace(corpus)
    with st.expander('Effect of A–D: top terms and curves (Part VI)'):
        preprocessing.condition_effect(result)
    with st.expander('Porter: all eight steps for a word'):
        porter.word_steps(full=True)
    with st.expander('Porter: before (C) vs after (D) stemming'):
        porter.before_after(result)
    with st.expander('Porter: merged word forms'):
        porter.merged_groups(result)
    with st.expander('Porter: stems of one abstract'):
        porter.abstract_stems(corpus)
    with st.expander('Manage corpora · fetch another PubMed query or archive one'):
        corpus_sections.manager()

    common.more('Answers', '')
    common.answers('Part VI · How does preprocessing affect the Zipf distribution?', part6_answer(result))
