"""Explain the course pipeline and demonstrate rule-based EOS behavior."""

import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eos import split_sentences
from src.stats import compute_stats
from ui.brand import footer, inject_css, intro, masthead, preprocessing_table, reading_text, sidebar_nav


def main() -> None:
    st.set_page_config(page_title="Helix · Methods & EOS demo", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
    inject_css()
    sidebar_nav()
    masthead("03 / Methods & demonstration")
    intro("Explain what the system does", "Visible steps. Inspectable results.",
          "Trace a piece of text through preprocessing and sentence detection, then use the same concepts to explain retrieval results.")
    system, processing, eos, demo = st.tabs(["System description", "Preprocessing", "EOS playground", "Demo guide"])
    st.page_link("pages/10_HW2_Overview.py", label="Homework 2: Zipf, Porter, Word2Vec & edit distance →")

    with system:
        st.markdown("### From source XML to ranked documents")
        st.markdown('<div class="hx-flow"><span>XML parser</span><span>Preprocessing</span><span>Inverted index</span><span>AND / OR / PHRASE</span><span>BM25 ranking</span><span>Results & document analysis</span></div>', unsafe_allow_html=True)
        st.dataframe([
            {"Component": "Input / parsing", "Behavior": "PMC XML: title, abstract, complete body, journal, year, PMID / PMCID. PubMed records can supplement the corpus with title/abstract only."},
            {"Component": "Preprocessing", "Behavior": "Lowercase → alphanumeric/hyphenated tokens → optional English stopword removal → optional NLTK Porter stemming."},
            {"Component": "Index", "Behavior": "Each term maps to documents, term frequency, and positions in the processed token sequence."},
            {"Component": "Retrieval", "Behavior": "AND, OR, and PHRASE select eligible documents. BM25 then ranks every eligible document with k1 = 1.2 and b = 0.75."},
            {"Component": "Analysis", "Behavior": "Character, token, vocabulary, and sentence counts from the parsed text; EOS uses punctuation and abbreviation rules."},
        ], hide_index=True, use_container_width=True)
        a, b = st.columns(2)
        with a:
            st.markdown("#### Candidate selection")
            st.write("AND requires all processed query terms. OR requires at least one. PHRASE requires the stemmed terms to occur consecutively in one paragraph. These modes select eligible documents; they do not produce the displayed ranking score.")
        with b:
            st.markdown("#### BM25 ranking")
            st.latex(r"\operatorname{BM25}(q,d)=\sum_{t\in q}\operatorname{IDF}(t)\frac{tf_{t,d}(k_1+1)}{tf_{t,d}+k_1\left(1-b+b\frac{|d|}{\overline{dl}}\right)}")
            st.latex(r"\operatorname{IDF}(t)=\ln\left(1+\frac{N-df(t)+0.5}{df(t)+0.5}\right),\qquad k_1=1.2,\ b=0.75")
        st.caption("BM25 rewards rare query terms and repeated evidence, while limiting the benefit of repetition and normalizing for document length. Scores compare ranking within the same query only.")

    with processing:
        text = st.text_area("Text to preprocess", "The patients with type 2 diabetes were receiving insulin treatments.", height=120)
        a, b = st.columns(2)
        stem = a.checkbox("Apply Porter stemming", value=True)
        stop = b.checkbox("Remove stopwords", value=True)
        try:
            preprocessing_table(text, remove_stopwords=stop, stem=stem)
        except LookupError as exc:
            st.warning(str(exc))
        st.caption("This playground does not change your index. Choose the real collection settings on the XML collection page.")
        st.info("Numbers and hyphenated terms are retained. Stemming groups some word forms; it does not add biomedical synonyms or understand meaning.")

    with eos:
        sample = st.text_area("Text to split into sentences",
            'Dr. Smith and J. Lee studied type 2 diabetes. The mean was 3.14. They reported "Improvement!" p53 expression changed.',
            height=150)
        stats = compute_stats(sample)
        for col, label, number in zip(st.columns(4), ["Characters", "Words", "Sentences (EOS)", "Unique words"],
                                     [stats.num_characters, stats.num_words, stats.num_sentences, stats.vocabulary_size]):
            col.metric(label, number)
        st.caption("Characters include whitespace. Words are lowercase alphanumeric/hyphenated tokens; unique words are counted before stemming.")
        sentences = split_sentences(sample)
        if sentences:
            st.dataframe([{"Sentence": i, "Text": value} for i, value in enumerate(sentences, 1)],
                         hide_index=True, use_container_width=True)
        else:
            st.info("Enter some text to inspect its sentence boundaries.")
        with st.expander("Rules and limitations", expanded=True):
            st.write("A period, question mark, or exclamation mark followed by whitespace/end marks a candidate boundary. Closing quotes and brackets stay with the sentence. Known abbreviations (Dr., Fig., et al., e.g.) and single-letter initials are protected. A decimal point inside a number is not a sentence boundary.")
            st.caption("This is a heuristic, not a gold-standard sentence segmenter. Ambiguous abbreviations, formulas, and unusual scientific notation can still produce mistakes.")

    with demo:
        st.markdown("### Demonstration sequence")
        st.dataframe([
            {"Step": "1. Dataset", "Show": "Import PMC XML. Inspect full-text availability, document counts, and document lengths."},
            {"Step": "2. Preprocessing", "Show": "Explain tokenization, stopword removal, and Porter stemming. Build the index."},
            {"Step": "3. Keyword retrieval", "Show": "Enter a query. Compare AND, OR, and PHRASE candidate selection; inspect BM25-ranked matches, scores, and retrieval time."},
            {"Step": "4. Document analysis", "Show": "Open a result. Read its complete parsed text; inspect characters, words, sentence counts, and vocabulary."},
            {"Step": "5. Smart EOS", "Show": "Inspect the sentence list and try initials, abbreviations, and decimals in the playground."},
        ], hide_index=True, use_container_width=True)
        st.caption("The homework specifies XML-based retrieval, basic matching, preprocessing, document statistics, and smart EOS. It does not prescribe a UI style, corpus size, BM25 specifically, or a two-paper presentation.")
        source = ROOT / "homework-1-26-20260908.pdf"
        if source.exists():
            st.download_button("Download the homework brief", source.read_bytes(), file_name=source.name, mime="application/pdf")
        st.page_link("pages/1_Search.py", label="Start the retrieval demo →")
    footer()


if __name__ == "__main__":
    main()
