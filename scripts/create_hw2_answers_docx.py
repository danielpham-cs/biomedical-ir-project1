"""About 1.5 pages: concept-level answers to every question of the detailed Project 2 brief.

The professor checks understanding of the theory, so the sheet explains concepts and
reasons and quotes only the few numbers a question asks for (e.g. the exponent).
Full tables stay in the demo system and output/project2_glp1. Output: output/HW2_answers.docx
"""
import re
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/HW2_answers.docx'
TEAL = RGBColor(0x08, 0x7F, 0x78)
INK = RGBColor(0x15, 0x2E, 0x3A)
GREY = RGBColor(0x56, 0x6B, 0x76)

doc = Document()
section = doc.sections[0]
section.page_width, section.page_height = Cm(21.0), Cm(29.7)
section.left_margin = section.right_margin = Cm(1.8)
section.top_margin = section.bottom_margin = Cm(1.5)
normal = doc.styles['Normal']
normal.font.name = 'Calibri'
normal.element.rPr.rFonts.set(qn('w:eastAsia'), 'Calibri')
normal.font.size = Pt(10)
normal.font.color.rgb = INK
normal.paragraph_format.space_after = Pt(2)
normal.paragraph_format.space_before = Pt(0)
normal.paragraph_format.line_spacing = 1.0


def runs(paragraph, text, size=None, color=None):
    """Write text with **bold** and *italic* markers as separate runs."""
    for part in re.split(r'(\*\*.+?\*\*|\*.+?\*)', text):
        if not part:
            continue
        bold, italic = part.startswith('**'), part.startswith('*') and not part.startswith('**')
        run = paragraph.add_run(part.strip('*') if (bold or italic) else part)
        run.bold, run.italic = bold, italic
        if size:
            run.font.size = Pt(size)
        if color:
            run.font.color.rgb = color
    return paragraph


def para(text, size=None, color=None, after=2):
    p = runs(doc.add_paragraph(), text, size, color)
    p.paragraph_format.space_after = Pt(after)
    return p


def items(*texts):
    """One paragraph per question, with a hanging indent so the bold labels line up."""
    for text in texts:
        p = para(text)
        p.paragraph_format.left_indent = Cm(0.35)
        p.paragraph_format.first_line_indent = Cm(-0.35)


def heading(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True  # never leave a heading alone at the bottom of a page
    run = p.add_run(text)
    run.bold, run.font.size, run.font.color.rgb = True, Pt(10.5), TEAL
    border = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    for key, value in (('val', 'single'), ('sz', '4'), ('space', '1'), ('color', 'B9CDD2')):
        bottom.set(qn(f'w:{key}'), value)
    border.append(bottom)
    p._p.get_or_add_pPr().append(border)


# ---------------------------------------------------------------- title
title = doc.add_paragraph()
title.paragraph_format.space_after = Pt(0)
run = title.add_run("Project #2 — Zipf's Law in a scientific collection: answers to the detailed brief")
run.bold, run.font.size, run.font.color.rgb = True, Pt(13), INK
para('Biomedical Information Retrieval · Name / ID: ____________________ · Collection: about 1,000 English PubMed abstracts on '
     'GLP-1, one unique PMID each. All tables and plots are shown in the demo system.', size=8.5, color=GREY, after=4)

# ---------------------------------------------------------------- I, II
heading('Parts I–II — Text preprocessing and the vocabulary')
items('**Pipeline.** *Tokenization* splits text into tokens; *case folding* maps Data/DATA/data to one term; *punctuation handling* merges '
      '"data," and "data." but keeps internal marks (GLP-1); *stopwords* (the, of, and) are kept in one version and removed in another; '
      '*stemming* (our own Porter algorithm) cuts suffixes, while *lemmatization* would map words to dictionary forms.',
      '**Vocabulary.** *Tokens* are all occurrences, *unique terms* are the dictionary entries, and average tokens per document is document '
      'length. Each step redefines what a "term" is, so all these numbers change.')

# ---------------------------------------------------------------- III, IV
heading('Parts III–IV — Collection frequency and document frequency')
items('**CF(t)** is the total number of occurrences of t in the whole collection; ranking terms by CF gives the rank–frequency list used '
      'for Zipf. **DF(t)** is the number of documents that contain t at least once. Always DF ≤ CF and DF ≤ N.')

# ---------------------------------------------------------------- V
heading("Part V — Zipf distribution: rank–frequency and log-log regression")
para("Zipf's law states f(r) ∝ 1/r^k (ideally k ≈ 1); taking logarithms gives log f = log C − k·log r, a straight line. Experiment 1 "
     '(linear axes) shows a sharp L-shaped curve; Experiment 2 (log-log) straightens it, and linear regression log(CF) = a − b·log(r) gives '
     'the slope, intercept, exponent k = b, R² and RMSE.')
items("**Q1 Does the collection follow Zipf's Law?** Approximately: the log-log plot is close to a straight line (few very frequent terms, "
      'many rare ones), but the top bends away and the tail forms steps, because many rare terms share the same counts (1, 2, 3 …).',
      '**Q2 What is the estimated Zipf exponent?** The negative slope, about 1.25 here, steeper than the ideal 1; it depends on preprocessing '
      'and on the ranks fitted.',
      '**Q3 Is R² sufficient?** No. Sorted ranks make any decreasing curve score a high R²; the points are not independent; the many tied tail '
      'points dominate the fit; and R² does not compare against other distributions (e.g. log-normal). It describes a trend, it is not a test.',
      '**Q4 Which portion fits best?** The high-frequency head, with k near 1. The middle is steeper, and the low-frequency tail fits worst '
      'because its tiny integer counts form flat steps.')

# ---------------------------------------------------------------- VI
heading('Part VI — How does preprocessing affect the Zipf distribution?')
items('**1 Vocabulary size.** Punctuation removal and stemming merge variants and shrink the vocabulary; stopword removal deletes only a '
      'short list of types.',
      '**2 High-frequency terms.** With stopwords the head is function words (and, the, of); without them it shows the subject (glp-1, '
      'receptor, patients), and many tokens disappear.',
      '**3 Zipf exponent.** Each step changes k: merging variants shortens the tail and steepens the slope; removing stopwords flattens the head.',
      '**4 Shape.** The heavy tail survives every condition; stopwords reshape the head, punctuation and stemming the tail.')

# ---------------------------------------------------------------- VII, VIII
heading('Parts VII–VIII — CF vs DF, and the connection to TF-IDF')
items('**VII Q1 Why can a term have high CF but relatively low DF?** CF counts every occurrence, DF each document once; a term repeated '
      'inside a few documents (e.g. an abbreviation reused in one paper) gains CF without spreading.',
      '**VII Q2 Which measure shows whether a term is widely distributed?** DF (or DF/N); CF cannot tell one document from many.',
      '**VII Q3 Why are CF and DF both important?** CF reflects volume (tokens, positional postings); DF is the posting-list length and the '
      'basis of IDF, i.e. how well a term separates documents.',
      '**VIII Why do very common terms get lower IDF?** idf = log(N/df): as df approaches N the ratio approaches 1 and its logarithm 0, so a '
      'word found everywhere gets almost no weight, while a word in one document gets the maximum log N.',
      "**VIII Zipf and TF-IDF.** Zipf's frequent head terms are also high-DF terms, so IDF moves weight from the head to rarer, more specific "
      "terms; TF-IDF counteracts the skew Zipf describes. IDF must come from DF, not from a term's rank.")

# ---------------------------------------------------------------- IX
essay = [
    "Zipf's law matters to an IR system because it describes how unevenly both the workload and the information are spread over a "
    "vocabulary: a small head of terms produces most of the occurrences, while a long tail of rare terms makes up most of the vocabulary.",
    "**Stopword removal.** The head consists of function words that occur in nearly every document. A term that appears in almost all "
    "documents cannot separate relevant from non-relevant ones, so its discriminative value is close to zero, while it costs a large share "
    "of the processing and index space. Removing or down-weighting such words therefore saves much space for little loss. The price is that "
    "phrase and negation queries (\"not effective\", \"to be or not to be\") need the original text or a positional index.",
    "**Inverted index.** Zipf shapes every part of the index. The dictionary is dominated by the tail: thousands of terms that occur once "
    "each need an entry but have a single posting. The postings are dominated by the head: a few terms have lists that cover almost every "
    "document and hold most of the positions. Index storage therefore grows with the head's long posting lists and the tail's large "
    "dictionary. Compression exploits exactly this skew: long, dense posting lists compress well with gap (delta) encoding and "
    "variable-byte or Elias codes, and front coding or blocking reduces a dictionary of many rare strings.",
    "**TF-IDF.** Ranking by raw term frequency would reward the head terms that every document shares. IDF gives those terms weights near "
    "zero and gives rare, specific terms the largest weights, which is why high-frequency terms have low discriminative power. Zipf "
    "explains why such a correction is needed for almost every query.",
    "In short, Zipf tells the designer where the volume is (the head) and where the vocabulary is (the tail), and an IR system treats the "
    "two differently: it filters or down-weights the head, compresses its long lists, and keeps the tail cheap to store but highly weighted.",
]
essay_words = sum(len(re.findall(r'\S+', text.replace('**', ''))) for text in essay)
assert 300 <= essay_words <= 500, essay_words  # the brief asks for 300–500 words
heading(f"Part IX — Why does Zipf's Law matter to an Information Retrieval system? ({essay_words} words)")
for text in essay:
    para(text)

# ---------------------------------------------------------------- optional, final, RQs
heading('Optional challenge — Does domain specificity affect the Zipf distribution?')
para('A medical collection (GLP-1, PubMed) and a computer-science collection (500 arXiv abstracts from ten subfields) of equal size keep the same heavy-tailed, near-linear log-log shape with similar exponents, '
     'but share little vocabulary and none of their top content terms: domain changes which terms form the head, not the form of the law.')

heading('Final challenge — Why is an approximate law still useful for IR?')
para('Zipf is not exact: k is not 1, it shifts with preprocessing, and both ends of the curve bend. Yet IR design depends on the shape, not '
     'the exact exponent: in every condition and domain a few terms dominate the tokens and a long tail dominates the vocabulary. That robust '
     'skew is enough to justify stopword handling, index compression and DF-based weighting, which are then verified on the real system.')

heading('Research questions RQ1–RQ5')
items('**RQ1 Does the distribution follow Zipf?** Approximately; near-linear in log-log, not exact.',
      '**RQ2 Empirical exponent?** About 1.25, depending on preprocessing and the fitted range.',
      '**RQ3 Effect of preprocessing?** Stopwords reshape the head; punctuation handling and stemming shrink the tail; k changes.',
      '**RQ4 CF vs DF?** CF ≥ DF; CF measures volume, DF spread; only DF determines IDF.',
      '**RQ5 Why does Zipf matter to IR?** It explains stopword removal, index size and compression, and IDF weighting.')

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(OUT, f'(essay {essay_words} words)')
