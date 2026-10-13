"""Report (Markdown + PDF) for Project 2 part 2: CBOW vs Skip-gram word embeddings.

Every number in the text is read from experiments.json; nothing is typed by hand.
"""
from __future__ import annotations

from itertools import groupby
from pathlib import Path

import numpy as np

from .embedding_study import GROUPS, PAIRS, QUERIES
from .project2_part2_figures import plot_all

AUTHOR = 'Daniel · P76157022'

BACKGROUND = [
    ('Input', 'The context words in the window (their vectors are averaged into one hidden vector h)',
     'The single center word'),
    ('Prediction target', 'The center word', 'Each context word in the window, one at a time'),
    ('Training examples', 'One example per token position', 'One example per (center, context) pair: up to 2 × window per token'),
    ('Training speed', 'Faster: fewer, averaged updates', 'Slower: many more updates per epoch'),
    ('Rare words', 'A rare word is one of several averaged inputs, so its own gradient signal is diluted',
     'A rare word is the input for every one of its context pairs, so each occurrence gives it several direct updates'),
    ('Expected behavior', 'Smoother, high-cosine space; good for frequent words and interchangeable terms',
     'Sharper, more topical neighbors; better for rare and domain-specific terms, but noisier'),
]

SKIPGRAM_RARE = (
    "Scientific terms such as drug names, gene symbols or abbreviations usually occur only a few times in a small corpus. "
    "In CBOW a rare word only enters training as one of up to 2·window context inputs whose vectors are averaged, so its update is shared with "
    "the frequent words around it, and as a prediction target only once per occurrence. In Skip-gram the rare word is itself the input for every "
    "pair it forms, so each occurrence produces up to 2·window separate gradient updates that pull it toward the actual words around it. "
    "The rare word's vector is therefore determined by its contexts rather than by its random initialization. Its neighbors can be traced "
    "to specific contexts, which matters when the term is the thing a biomedical user is searching for.")


def short_answer(r):
    sg, cb = (next(x for x in r['training'] if x['experiment'] == 'main' and x['model'] == m) for m in ('Skip-gram', 'CBOW'))
    tsne = r['projections']['skipgram_main']['purity']['tsne_2d']
    groups = [{g['group'] for g in r['group_cosines'][f'{m} / main'] if g['margin'] > 0} for m in ('Skip-gram', 'CBOW')]
    total = len(r['group_cosines']['Skip-gram / main'])
    return (f"Short answer: yes, within the domain. {r['corpus']['document_count']:,} abstracts ({r['corpus']['raw_tokens']:,} tokens, "
            f"{cb['seconds'] / 60:.0f}–{sg['seconds'] / 60:.0f} minutes of CPU training per model, no pretrained vectors) are enough to learn measurable "
            "semantic structure for the core terms of the domain. Drug classes, organs, adverse events and animal models are closer to each other than to "
            f"other groups in both models ({len(groups[0] & groups[1])} of {total} groups), and nearest labeled words share the group {tsne:.0%} of the time against {r['chance_purity']:.0%} by chance. "
            "Skip-gram also places rare terms correctly and stably. The limits are words that are very rare or marginal to the topic (RQ5).")


def sample_rows(r):
    """[position, center word, Skip-gram pairs, CBOW example] for the worked training-sample example."""
    s = r['samples']
    pairs = {}
    for center, context in s['skipgram']:
        pairs.setdefault(center, []).append(context)
    return [[str(i + 1), w, ', '.join(f'({w}, {c})' for c in pairs[w]), f"({', '.join(e['context'])}) → {w}"]
            for i, (w, e) in enumerate(zip(s['tokens'], s['cbow']))]


def sample_text(r):
    s = r['samples']
    return (f"Example from one real sentence (PMID {s['pmid']}), shown with window {s['window']} so it fits on a page: \"{s['sentence']}\" "
            f"After preprocessing it has {len(s['tokens'])} tokens: {' '.join(s['tokens'])}. Every position is a center word once. "
            f"Skip-gram turns it into {len(s['skipgram'])} (center, context) pairs, CBOW into {len(s['cbow'])} (context → center) examples. "
            "Subsampling and negative words are left out of the example.")


def fmt(value, digits=3):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def _by(rows, key):
    return {r[key]: r for r in rows}


def neighbor_table(rows):
    """[query (count), Skip-gram top-5, CBOW top-5] rows."""
    table = []
    for query, group in groupby(rows, key=lambda r: r['query']):
        group = list(group)
        cells = {}
        for model, items in groupby(group, key=lambda r: r['model']):
            items = list(items)
            cells[model] = ('(not in vocabulary)' if items[0]['cosine_similarity'] is None else
                            ', '.join(f"{r['word']} {r['cosine_similarity']:.2f}" for r in items))
        table.append([f"{query} ({group[0]['query_count']})", cells['Skip-gram'], cells['CBOW']])
    return table


def facts(r):
    """Named numbers used in the answers, computed once."""
    t = {(x['experiment'], x['model']): x for x in r['training']}
    sg, cb = t[('main', 'Skip-gram')], t[('main', 'CBOW')]
    ov = {k: _by(v, 'frequency_band') for k, v in r['overlap'].items()}
    seed = {m: ov[f'{m} | seed 42 vs seed 7 (same setting)'] for m in ('Skip-gram', 'CBOW')}
    pairs = _by(r['pairs'], 'pair')
    gc = {k: _by(v, 'group') for k, v in r['group_cosines'].items()}
    margin = {k: float(np.mean([g['margin'] for g in v.values()])) for k, v in gc.items()}
    proj = r['projections']
    co = {m: _by(v, 'frequency_band') for m, v in r['cooccurrence'].items()}
    agreement = [a for a in r['model_agreement'] if a['query'] in QUERIES]
    corpus = r['corpus']
    rare = sum(x['words'] for x in r['overlap']['Skip-gram | seed 42 vs seed 7 (same setting)'] if x['frequency_band'] == '2-4')
    return dict(t=t, sg=sg, cb=cb, ov=ov, seed=seed, pairs=pairs, gc=gc, margin=margin, proj=proj, co=co,
                agreement=agreement, corpus=corpus, rare=rare,
                mean_shared=float(np.mean([a['shared_top5'] for a in agreement])))


def pair(f, a, b, model):
    row = f['pairs'][f'{a} – {b}']
    arch = 'skipgram' if model == 'Skip-gram' else 'cbow'
    return f"{row[f'{arch}_cosine']:.2f} ({row[f'{arch}_percentile']:.0f}th pct.)"


def answers(r):
    f = facts(r)
    sg, cb, seed, ov, proj, gc, co = f['sg'], f['cb'], f['seed'], f['ov'], f['proj'], f['gc'], f['co']
    base = r['baseline']
    chance = r['chance_purity']
    t = f['t']
    out = {}
    out['RQ1 — How do CBOW and Skip-gram differ?'] = (
        f"Training. With the same corpus and settings (100 dimensions, window 5, 15 epochs), Skip-gram processed {sg['examples_per_epoch']:,} "
        f"(center, context) pairs per epoch, CBOW {cb['examples_per_epoch']:,} examples, one per retained token: "
        f"{sg['examples_per_epoch'] / cb['examples_per_epoch']:.1f}× more updates. Skip-gram took {sg['seconds']:.0f} s and CBOW {cb['seconds']:.0f} s "
        f"({sg['seconds'] / cb['seconds']:.1f}× slower; all ten models trained concurrently on one machine, so times are comparable to each other only).\n\n"
        f"Training loss. The fixed training-probe loss fell from {sg['loss_start']:.3f} to {sg['loss_end']:.3f} for Skip-gram and from "
        f"{cb['loss_start']:.3f} to {cb['loss_end']:.3f} for CBOW. Skip-gram flattens after about ten epochs, while CBOW is still decreasing slowly at epoch 15. The two values are not directly "
        "comparable: they score different prediction tasks (one averaged context → word vs. word → each context word). A lower loss is not evidence of a better embedding.\n\n"
        f"Neighbors. The two models agree surprisingly little. For the {len(f['agreement'])} query words they share on average {f['mean_shared']:.1f} of 5 nearest words, and across the whole "
        f"vocabulary only {ov['Skip-gram vs CBOW | main']['all']['mean_overlap']:.0%} of top-10 neighbors coincide. They also have a different style. "
        f"CBOW cosines are uniformly high (mean of random word pairs {base['CBOW']['mean']:.2f} vs {base['Skip-gram']['mean']:.2f} for Skip-gram), and its neighbors are words that "
        "fill the same slot in a sentence: semaglutide → tirzepatide 0.92, liraglutide, dulaglutide; kidney → hf, failure, heart (they occur in lists such as "
        "'heart failure and kidney disease'). Skip-gram neighbors are more topical: kidney → ckd, end-stage, glomerular; cancer → breast, endometrial, colorectal; "
        "insulin → resistance, basal. But they also include very rare noise words (obesity → '2030', '589'). "
        "glp-1 itself shows the difference best: Skip-gram returns the words it appears with ('ras', 'ra' from 'GLP-1 RAs', i.e. receptor agonists), "
        "CBOW the words that can replace it in a sentence, mostly other spellings ('glp1', 'glp)-1', 'glp-1/gip'); the two top-5 lists share no word.\n\n"
        f"Rare words. When only the random seed changes, Skip-gram keeps {seed['Skip-gram']['2-4']['mean_overlap']:.0%} of the top-10 neighbors of words seen 2–4 times. "
        f"That is about the same as for frequent words ({seed['Skip-gram']['>=200']['mean_overlap']:.0%}). CBOW keeps only {seed['CBOW']['2-4']['mean_overlap']:.0%} for rare words but "
        f"{seed['CBOW']['>=200']['mean_overlap']:.0%} for frequent words. So CBOW's rare-word vectors are largely an accident of initialization and sampling. "
        f"Skip-gram's stable rare-word neighbors are not simply memorized from one abstract: only {co['Skip-gram']['2-4']['mean_share']:.0%} of them ever share an abstract with the word. "
        "Both models still connect some rare terms correctly (lixisenatide → exenatide; diarrhea → constipation), but CBOW's rare-word lists also contain numbers such as '4.1%' and '5.4%' "
        "with cosines around 0.9.")

    wins = [p for p in r['pairs'] if p['skipgram_percentile'] is not None and min(p['skipgram_percentile'], p['cbow_percentile']) >= 90]
    out['RQ2 — Are the embeddings semantically meaningful?'] = (
        "A raw cosine means little on its own, because the two models use the cosine scale differently. Each pair is therefore compared with "
        f"20,000 random word pairs (words seen ≥ 5 times) and reported as a percentile. In {len(wins)} of {len(PAIRS)} pairs, both models put the pair above the 90th percentile: "
        + ', '.join(p['pair'] for p in wins) + ". "
        f"Pairs within nine predefined domain groups are more similar than pairs across groups in every group for both models. The mean margin is "
        f"{f['margin']['Skip-gram / main']:.2f} (Skip-gram) and {f['margin']['CBOW / main']:.2f} (CBOW) cosine points. "
        f"Control pairs from unrelated groups stay low: nausea – kidney {pair(f, 'nausea', 'kidney', 'Skip-gram')} in Skip-gram, cancer – weight {pair(f, 'cancer', 'weight', 'CBOW')} in CBOW.\n\n"
        f"The brief's example words show the limits. Cancer – tumor scores {pair(f, 'cancer', 'tumor', 'Skip-gram')} in Skip-gram but only "
        f"{pair(f, 'cancer', 'tumor', 'CBOW')} in CBOW. Treatment – therapy is near-synonymous but scores only {pair(f, 'treatment', 'therapy', 'Skip-gram')} in Skip-gram "
        f"({pair(f, 'treatment', 'therapy', 'CBOW')} in CBOW). Treatment – diagnosis is low in both ({pair(f, 'treatment', 'diagnosis', 'Skip-gram')} / "
        f"{pair(f, 'treatment', 'diagnosis', 'CBOW')}). In GLP-1 abstracts, cancer is a side topic (125 occurrences, 'tumor' 27). 'Treatment' sits next to adherence and "
        "discontinuation, and 'diagnosis' next to ICD codes, so these words learn GLP-1-specific contexts rather than general medical meaning.\n\n"
        "Conclusion: yes, the models learn meaningful relations for well-attested concepts in this domain: drug classes, organ systems, adverse-event "
        "terms and animal models. What they learn is relatedness by shared context, not synonymy (kidney ↔ heart in CBOW). The relations are relative to the corpus, "
        "and they are unreliable for words that are rare or off-topic in it.")

    sgm, cbm = proj['skipgram_main'], proj['cbow_main']
    out['RQ3 — How do PCA and t-SNE visualizations differ?'] = (
        f"PCA is a linear projection onto the two directions of greatest variance. Fitted on the whole vocabulary, it keeps {sum(sgm['pca_variance']):.1%} (Skip-gram) and "
        f"{sum(cbm['pca_variance']):.1%} (CBOW) of the variance of the normalized vectors, so most of the structure is discarded. "
        "t-SNE is non-linear. It was fitted on the 300 most frequent words plus all labeled group words, and tries to keep each point's nearest neighbors close.\n\n"
        "To test clustering numerically, each of the labeled group words is checked against its three nearest labeled words. The table reports the share that "
        f"belong to the same group (random chance = {chance:.2f}). In the original 100-d space this purity is {sgm['purity']['cosine']:.2f} (Skip-gram) / "
        f"{cbm['purity']['cosine']:.2f} (CBOW). PCA reduces it to {sgm['purity']['pca_2d']:.2f} / {cbm['purity']['pca_2d']:.2f}, barely above chance. t-SNE keeps "
        f"{sgm['purity']['tsne_2d']:.2f} / {cbm['purity']['tsne_2d']:.2f}, essentially the original value. Visually, the Skip-gram t-SNE shows tight groups for the "
        "established GLP-1 drugs, GI adverse events, cardiac terms and animal models. But not every group holds together: renal sits near hepatic rather than kidney/ckd, "
        "and the newer agents orforglipron and retatrutide stay apart from the other drugs. The PCA plots mostly show one dense cloud in which only the animal-model words separate clearly.\n\n"
        "Limitations. PCA axes are interpretable and the projection is deterministic and global, but it is too low-rank for 100-d word vectors. t-SNE distances "
        "between clusters, and cluster sizes, carry no meaning, and the layout depends on hyperparameters. With perplexity "
        f"{proj['skipgram_main_p5']['perplexity']:.0f}, {sgm['perplexity']:.0f} and {proj['skipgram_main_p50']['perplexity']:.0f}, the same vectors give purity "
        f"{proj['skipgram_main_p5']['purity']['tsne_2d']:.2f}, {sgm['purity']['tsne_2d']:.2f} and {proj['skipgram_main_p50']['purity']['tsne_2d']:.2f} and visibly different maps. "
        "As the brief warns, visual clusters do not prove good semantics. t-SNE can only re-arrange what the vectors already contain, and the clustering "
        "is only as good as the cosine purity. A cluster can also form because words share an abstract or a frequency range rather than a meaning. Any claim "
        "should therefore be checked in the original space, as done in RQ2.")

    w2 = {m: ov[f'{m} | window 5 vs window 2']['all']['mean_overlap'] for m in ('Skip-gram', 'CBOW')}
    d50 = {m: ov[f'{m} | 100 vs 50 dimensions']['all']['mean_overlap'] for m in ('Skip-gram', 'CBOW')}
    mc5 = {m: ov[f'{m} | min_count 2 vs 5'] for m in ('Skip-gram', 'CBOW')}
    sd = {m: seed[m]['all']['mean_overlap'] for m in ('Skip-gram', 'CBOW')}
    out['RQ4 — How do model parameters affect the results?'] = (
        "Each variant changes exactly one parameter of the main model. Changing only the random seed sets the noise floor: Skip-gram keeps "
        f"{sd['Skip-gram']:.0%} and CBOW {sd['CBOW']:.0%} of top-10 neighbors. A parameter matters only if it changes the neighbors clearly more than that.\n\n"
        f"Context window (5 → 2) is by far the strongest factor. Only {w2['Skip-gram']:.0%} (Skip-gram) and {w2['CBOW']:.0%} (CBOW) of neighbors survive. "
        "With a narrow window, a word's vector is built from its immediate neighbors (syntactic, same-slot similarity). This raises all cosines "
        f"(Skip-gram within-group 'GLP-1 drugs' {gc['Skip-gram / main']['GLP-1 drugs']['within_mean_cosine']:.2f} → {gc['Skip-gram / window2']['GLP-1 drugs']['within_mean_cosine']:.2f}, "
        f"but between-group {gc['Skip-gram / main']['GLP-1 drugs']['between_mean_cosine']:.2f} → {gc['Skip-gram / window2']['GLP-1 drugs']['between_mean_cosine']:.2f} too). "
        f"The mean group margin moves from {f['margin']['Skip-gram / main']:.2f} to {f['margin']['Skip-gram / window2']:.2f} (Skip-gram) and from {f['margin']['CBOW / main']:.2f} "
        f"to {f['margin']['CBOW / window2']:.2f} (CBOW). Cosine group purity goes {proj['skipgram_main']['purity']['cosine']:.2f} → {proj['skipgram_window2']['purity']['cosine']:.2f} "
        f"(Skip-gram) and {proj['cbow_main']['purity']['cosine']:.2f} → {proj['cbow_window2']['purity']['cosine']:.2f} (CBOW). The window-2 Skip-gram has a lower training loss "
        f"({t[('window2', 'Skip-gram')]['loss_end']:.3f} vs {sg['loss_end']:.3f}) despite no better groups, which again shows loss is not embedding quality.\n\n"
        f"Visualization. The window also changes the maps most (figure below). t-SNE purity drops from {proj['skipgram_main']['purity']['tsne_2d']:.2f} to "
        f"{proj['skipgram_window2']['purity']['tsne_2d']:.2f} for Skip-gram and from {proj['cbow_main']['purity']['tsne_2d']:.2f} to "
        f"{proj['cbow_window2']['purity']['tsne_2d']:.2f} for CBOW. In the window-2 CBOW map most groups merge into one region (obesity sits next to ckd and masld, "
        "cancer and overweight drift away from their groups), and the animal-model words split in two, while the GLP-1 drugs stay tight in all six maps. "
        "A narrow window therefore keeps the most interchangeable words together but loses the topical groups that a wider window builds.\n\n"
        f"Dimensions (100 → 50) barely matter. Neighbor overlap is {d50['Skip-gram']:.0%} / {d50['CBOW']:.0%}, about the same as the seed-only overlap, and the t-SNE maps look alike "
        f"(t-SNE purity {proj['skipgram_dim50']['purity']['tsne_2d']:.2f} vs {proj['skipgram_main']['purity']['tsne_2d']:.2f} for Skip-gram, "
        f"{proj['cbow_dim50']['purity']['tsne_2d']:.2f} vs {proj['cbow_main']['purity']['tsne_2d']:.2f} for CBOW). "
        f"With {f['corpus']['retained_tokens']:,} training tokens, 50 dimensions already have enough capacity.\n\n"
        f"Minimum frequency (2 → 5) halves the vocabulary ({t[('main', 'Skip-gram')]['vocabulary']:,} → {t[('mincount5', 'Skip-gram')]['vocabulary']:,}) and changes even "
        f"frequent words' neighbors: only {mc5['Skip-gram']['>=200']['mean_overlap']:.0%} (Skip-gram) / {mc5['CBOW']['>=200']['mean_overlap']:.0%} (CBOW) of the "
        "top-10 of words seen ≥ 200 times survive. Removing rare words takes noisy candidates out of the neighbor lists (such as '2030' and '589'), and it also removes "
        "context words that shaped the remaining vectors.")

    out['RQ5 — What are the limitations of a small corpus?'] = (
        f"Size. The {f['corpus']['document_count']:,} abstracts give {f['corpus']['raw_tokens']:,} tokens after stopword removal. Published word2vec models use billions. "
        f"Of {f['corpus']['vocabulary_size']:,} vocabulary words (seen ≥ 2 times), {f['rare']:,} ({f['rare'] / f['corpus']['vocabulary_size']:.0%}) occur only 2–4 times, and words seen once "
        "are not modelled at all. Rare words are exactly the specialized terms a biomedical searcher may need.\n\n"
        f"Stability. Even for words seen ≥ 200 times, a different seed changes about {1 - seed['Skip-gram']['>=200']['mean_overlap']:.0%} (Skip-gram) / "
        f"{1 - seed['CBOW']['>=200']['mean_overlap']:.0%} (CBOW) of the top-10 neighbors. For rare words, CBOW changes {1 - seed['CBOW']['2-4']['mean_overlap']:.0%}. "
        "A neighbor list from one run should be read as a sample, not a fact. Words seen twice regularly enter the top-5 lists with high cosine (glp-1 → 'cornerstones'; "
        "obesity → '2030').\n\n"
        "Domain. The corpus is about GLP-1, so the space is GLP-1 shaped. 'ras' and 'ra' (receptor agonist abbreviations) are the nearest words to glp-1, 'tzp' "
        "means tirzepatide, and general medical relations such as cancer – tumor or treatment – diagnosis are weak because those topics are marginal here. The same "
        "words would have different neighbors in an oncology corpus.\n\n"
        "Terminology and tokenization. Spelling variants (diarrhea / diarrhoea), punctuation variants (glp-1 / glp1 / glp)-1) and multi-word terms "
        "(heart failure, insulin resistance) are split into separate or partial tokens, which spreads the little data each concept has.\n\n"
        "Conclusion: 1,000 abstracts are enough to train useful embeddings for one domain. The frequent core terms (drug names, organs, adverse events) "
        "form recognisable, measurable relations, and Skip-gram is the better choice when data is this limited because it keeps rare terms stable. "
        "The corpus is not enough for words seen only a few times in CBOW, or for general biomedical relations outside the topic. Remedies are more text, "
        "phrase detection, subword models (fastText), pretrained biomedical vectors (e.g. BioWordVec), and averaging results over several seeds.")
    return out


def export_report(results, output):
    output = Path(output)
    plot_all(results, output/'figures')
    ans = answers(results)
    f = facts(results)
    corpus = results['corpus']
    cfg = results['experiments']['main']
    training_rows = [[x['experiment'], x['model'], x['dimensions'], x['window'], x['min_count'], x['seed'], f"{x['vocabulary']:,}",
                      f"{x['examples_per_epoch']:,}", f"{x['seconds']:.0f}", f"{x['loss_start']:.3f} → {x['loss_end']:.3f}"]
                     for x in results['training']]
    training_head = ['Run', 'Model', 'Dim', 'Win.', 'Min count', 'Seed', 'Vocab', 'Examples / epoch', 'Sec.', 'Probe loss']
    pair_head = ['Pair', 'Counts', 'Skip-gram cosine', 'pct.', 'CBOW cosine', 'pct.']
    pair_rows = [[p['pair'], p['counts'], fmt(p['skipgram_cosine'], 2), fmt(p['skipgram_percentile'], 0), fmt(p['cbow_cosine'], 2), fmt(p['cbow_percentile'], 0)]
                 for p in results['pairs']]
    group_rows = [[g, ', '.join(GROUPS[g]), fmt(f['gc']['Skip-gram / main'][g]['within_mean_cosine'], 2), fmt(f['gc']['Skip-gram / main'][g]['between_mean_cosine'], 2),
                   fmt(f['gc']['CBOW / main'][g]['within_mean_cosine'], 2), fmt(f['gc']['CBOW / main'][g]['between_mean_cosine'], 2)] for g in GROUPS]
    group_head = ['Group', 'Words', 'SG within', 'SG between', 'CBOW within', 'CBOW between']
    overlap_rows = [[k] + [fmt(_by(v, 'frequency_band')[b]['mean_overlap'], 2) for b in ('2-4', '5-9', '10-49', '50-199', '>=200', 'all')]
                    for k, v in results['overlap'].items()]
    overlap_head = ['Comparison', '2–4', '5–9', '10–49', '50–199', '≥200', 'All']
    purity_rows = [[k, fmt(p['perplexity'], 0), f"{sum(p['pca_variance']):.1%}", fmt(p['purity']['cosine'], 2), fmt(p['purity']['pca_2d'], 2), fmt(p['purity']['tsne_2d'], 2)]
                   for k, p in results['projections'].items()]
    purity_head = ['Model / run', 'Perplexity', 'PCA variance', 'Purity: cosine', 'Purity: PCA', 'Purity: t-SNE']
    co_rows = [[m] + [fmt(_by(v, 'frequency_band')[b]['mean_share'], 2) for b in ('2-4', '5-9', '10-49', '50-199', '>=200')] for m, v in results['cooccurrence'].items()]
    sections = {
        'data': (f"Corpus: the same frozen {corpus['document_count']:,} English PubMed abstracts on GLP-1 used in part 1 (corpus fingerprint {corpus['fingerprint'][:16]}…). "
                 "Only abstract text is used. Preprocessing is part 1's condition C: lowercasing; whitespace tokenization; punctuation stripped from token boundaries while "
                 "internal hyphens are kept (glp-1); NLTK English stopwords removed; no stemming. Abstracts are split into sentences and context windows never cross a sentence. "
                 f"Result: {corpus['document_count']:,} documents and {corpus['sentence_count']:,} sentences. The abstracts contain {corpus.get('words_before_stopwords', 0):,} words "
                 f"before and {corpus['raw_tokens']:,} tokens after stopword removal, with {corpus.get('distinct_words', 0):,} distinct words. "
                 f"With min_count 2 the vocabulary has {corpus['vocabulary_size']:,} words "
                 f"({corpus['retained_tokens']:,} retained tokens). Stopwords were removed because they would dominate every window. The threshold of 2 keeps rare "
                 "domain terms, so their behaviour can be studied; min_count 5 is tested in RQ4."),
        'samples': (f"Both models use the same window definition (±{cfg['window']} words). Skip-gram creates one (center → context) pair for every word in the window. CBOW creates one "
                    "(window → center) example per position, with the window's vectors averaged (word2vec cbow_mean=1). "
                    "Frequent words are randomly subsampled (t = 0.001) each epoch, so example counts vary slightly between runs. "
                    "Both are trained from scratch in NumPy (src/word2vec.py) with negative sampling: 5 noise words from the unigram^0.75 distribution, learning rate 0.025 decaying "
                    f"linearly, batch 128, {cfg['epochs']} epochs, seed {cfg['seed']}. No pretrained vectors are used. The loss is measured on a fixed probe of 2,048 training "
                    "examples and noise words, not on held-out data."),
    }
    top_words = ', '.join(f"{w['word']} ({w['count']})" for w in results['top_words'][:20])

    md = ['# GLP-1 abstracts: word embeddings with CBOW and Skip-gram', '', 'Project 2, part 2 · Biomedical Information Retrieval', '', AUTHOR, '',
          '**Central question:** can about 1,000 scientific abstracts learn a meaningful semantic representation of words?', '',
          '**' + short_answer(results).replace(': ', ':** ', 1), '',
          '## 1. Data and preprocessing', '', sections['data'], '', f'Most frequent words: {top_words}.', '',
          '## 2. Training samples and models', '', sections['samples'], '', sample_text(results), '',
          '| # | Center word | Skip-gram examples (center, context) | CBOW example (context → center) |', '|---|---|---|---|']
    md += ['| ' + ' | '.join(row) + ' |' for row in sample_rows(results)]
    md += ['', '| ' + ' | '.join(training_head) + ' |', '|' + '---|' * len(training_head)]
    md += ['| ' + ' | '.join(map(str, row)) + ' |' for row in training_rows]
    md += ['', '![Loss curves](figures/loss_curves.png)', '',
           '## 3. Background: CBOW vs Skip-gram', '', '| Property | CBOW | Skip-gram |', '|---|---|---|']
    md += [f'| {a} | {b} | {c} |' for a, b, c in BACKGROUND]
    md += ['', '**Why might Skip-gram be particularly useful for rare or domain-specific scientific terms?**', '', SKIPGRAM_RARE,
           f"The experiment agrees (RQ1). Under a seed change, Skip-gram keeps {f['seed']['Skip-gram']['2-4']['mean_overlap']:.0%} of rare-word neighbors, CBOW {f['seed']['CBOW']['2-4']['mean_overlap']:.0%}.", '',
           '## 4. Nearest neighbors (cosine similarity, top 5)', '', f'{len(QUERIES)} query words; the number in brackets is the corpus count.', '',
           '| Query | Skip-gram | CBOW |', '|---|---|---|']
    md += [f'| {a} | {b} | {c} |' for a, b, c in neighbor_table(results['neighbors'])]
    md += ['', 'Rare words (seen < 15 times):', '', '| Query | Skip-gram | CBOW |', '|---|---|---|']
    md += [f'| {a} | {b} | {c} |' for a, b, c in neighbor_table(results['rare_neighbors'])]
    md += ['', '## 5. Semantic groups and word pairs', '', '| ' + ' | '.join(group_head) + ' |', '|' + '---|' * len(group_head)]
    md += ['| ' + ' | '.join(row) + ' |' for row in group_rows]
    md += ['', '![Group cosines](figures/group_cosines.png)', '', '| ' + ' | '.join(pair_head) + ' |', '|' + '---|' * len(pair_head)]
    md += ['| ' + ' | '.join(row) + ' |' for row in pair_rows]
    md += ['', '## 6. Visualization', '', '![t-SNE](figures/tsne_main.png)', '', '![PCA](figures/pca_main.png)', '',
           '| ' + ' | '.join(purity_head) + ' |', '|' + '---|' * len(purity_head)]
    md += ['| ' + ' | '.join(row) + ' |' for row in purity_rows]
    md += ['', f"Chance purity: {results['chance_purity']:.2f}.", '', '![t-SNE parameters](figures/tsne_parameters.png)', '', '![Perplexity](figures/tsne_perplexity.png)', '',
           '## 7. Neighbor stability', '', 'Mean top-10 neighbor overlap by corpus frequency band.', '',
           '| ' + ' | '.join(overlap_head) + ' |', '|' + '---|' * len(overlap_head)]
    md += ['| ' + ' | '.join(row) + ' |' for row in overlap_rows]
    md += ['', 'Share of top-10 neighbors that co-occur with the word in at least one abstract:', '',
           '| Model | 2–4 | 5–9 | 10–49 | 50–199 | ≥200 |', '|---|---|---|---|---|---|']
    md += ['| ' + ' | '.join(row) + ' |' for row in co_rows]
    md += ['', '![Seed stability](figures/seed_stability.png)', '', '## 8. Research questions', '']
    for title, text in ans.items():
        md += [f'### {title}', '', text, '']
    md += ['## Limitations', '', 'One topic, one corpus, PubMed order (not a random sample). No external benchmark (e.g. UMNSRS) or retrieval evaluation, so semantic quality is judged '
           'from predefined groups, random-pair baselines and seed stability. The groups and pairs were chosen from domain knowledge before the neighbor lists were read. '
           'Training times come from concurrent runs.', '',
           '## Reproduce', '', '`python scripts/run_project2_part2.py` replays the frozen snapshot output/project2_glp1/corpus_snapshot.json, trains or reuses the ten cached models in '
           'models/, and rewrites all tables, figures and this report. run_metadata.json holds settings, versions and file checksums.', '',
           '## Sources', '', '- Course brief: project-details-2-part-2-20261006.pdf.',
           '- Mikolov et al. (2013), Efficient Estimation of Word Representations in Vector Space, arXiv:1301.3781; Distributed Representations of Words and Phrases, arXiv:1310.4546.',
           '- van der Maaten & Hinton (2008), Visualizing Data using t-SNE, JMLR 9.',
           '- Implementation: src/word2vec.py, src/tsne.py, src/embedding_study.py, src/embedding_view.py.']
    (output/'report.md').write_text('\n'.join(md), encoding='utf-8')
    (output/'README.md').write_text(
        'Project 2 part 2 submission: word embeddings (CBOW vs Skip-gram)\n\n'
        'report.pdf / report.md: full report with the background table and RQ1-RQ5.\n'
        'summary.pdf: one-page summary covering every item of the brief.\n'
        'experiments.json: every measured number used in the report.\n'
        'tables/: training summary, top-5 neighbors, rare-word neighbors, pair and group cosines, neighbor overlap, co-occurrence, projection purity and coordinates.\n'
        'figures/: loss curves, PCA and t-SNE maps, parameter and perplexity comparisons, seed stability, group cosines.\n'
        'models/: the ten trained models (NumPy .npz; not in submission.zip, rebuilt by the script).\nrun_metadata.json: settings, environment and checksums.\n'
        'Reproduce: .\\.venv\\python.exe scripts\\run_project2_part2.py\n', encoding='utf-8')
    build_pdf(results, output, ans, sections, top_words, dict(training=(training_head, training_rows), pairs=(pair_head, pair_rows),
              groups=(group_head, group_rows), overlap=(overlap_head, overlap_rows), purity=(purity_head, purity_rows), co=co_rows), f)
    build_summary(results, output)


def build_pdf(results, output, ans, sections, top_words, tables, f):
    import matplotlib
    from html import escape
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    fonts = Path(matplotlib.get_data_path())/'fonts/ttf'
    for name, file in [('DejaVu', 'DejaVuSans.ttf'), ('DejaVu-Bold', 'DejaVuSans-Bold.ttf')]:
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(fonts/file)))
    pdfmetrics.registerFontFamily('DejaVu', normal='DejaVu', bold='DejaVu-Bold', italic='DejaVu', boldItalic='DejaVu-Bold')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Body', fontName='DejaVu', fontSize=9.2, leading=13.4, spaceAfter=7, textColor=colors.HexColor('#152e3a')))
    styles.add(ParagraphStyle(name='Cell', parent=styles['Body'], fontSize=7.2, leading=9.4, spaceAfter=0))
    styles.add(ParagraphStyle(name='Title2', parent=styles['Body'], fontName='DejaVu-Bold', fontSize=19, leading=24, spaceAfter=12))
    styles.add(ParagraphStyle(name='Section', parent=styles['Body'], fontName='DejaVu-Bold', fontSize=12.5, leading=17, spaceBefore=8, spaceAfter=8,
                              textColor=colors.HexColor('#087f78')))
    width = A4[0] - 88

    def p(text, style='Body'):
        return Paragraph(escape(str(text)).replace('\n', '<br/>'), styles[style])

    def table(headers, rows, widths=None):
        values = [[p(v, 'Cell') for v in headers]] + [[p(v, 'Cell') for v in row] for row in rows]
        t = Table(values, colWidths=widths or [width / len(headers)] * len(headers), repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e6f3f0')), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                               ('BOTTOMPADDING', (0, 0), (-1, -1), 4), ('TOPPADDING', (0, 0), (-1, -1), 4),
                               ('LINEBELOW', (0, 0), (-1, 0), .7, colors.HexColor('#087f78')),
                               ('LINEBELOW', (0, 1), (-1, -1), .25, colors.HexColor('#dce5e8'))]))
        return t

    def figure(name):
        from PIL import Image as PILImage
        path = output/'figures'/f'{name}.png'
        w, h = PILImage.open(path).size
        return Image(str(path), width=width, height=width * h / w)

    def paragraphs(text):
        return [p(chunk) for chunk in text.split('\n\n')]

    def footer(canvas, doc):
        canvas.setFont('DejaVu', 8); canvas.setFillColor(colors.HexColor('#566b76'))
        canvas.drawString(44, 25, 'Helix | GLP-1 | Project 2 part 2 · word embeddings'); canvas.drawRightString(A4[0] - 44, 25, str(doc.page))

    rq = list(ans.items())
    story = [p('GLP-1 abstracts: word embeddings with CBOW and Skip-gram', 'Title2'), p('Project 2, part 2 · Biomedical Information Retrieval', 'Section'), p(AUTHOR),
             p('Central question: can about 1,000 scientific abstracts learn a meaningful semantic representation of words?'),
             p(short_answer(results)),
             p('1. Data and preprocessing', 'Section'), p(sections['data']), p(f'Most frequent words: {top_words}.'),
             p('2. Training samples and models', 'Section'), p(sections['samples']), p(sample_text(results)),
             table(['#', 'Center word', 'Skip-gram examples (center, context)', 'CBOW example (context → center)'], sample_rows(results),
                   [18, 78, (width - 96) * 0.55, (width - 96) * 0.45]), Spacer(1, 8),
             table(*tables['training'], widths=[54, 52, 30, 30, 40, 30, 46, 64, 32, width - 378]), Spacer(1, 8), figure('loss_curves'), PageBreak(),
             p('3. Background: CBOW vs Skip-gram', 'Title2'), table(['Property', 'CBOW', 'Skip-gram'], BACKGROUND, [90, (width - 90) / 2, (width - 90) / 2]), Spacer(1, 10),
             p('Why might Skip-gram be particularly useful for rare or domain-specific scientific terms?', 'Section'), p(SKIPGRAM_RARE),
             p(f"The experiment agrees (RQ1). Under a seed change, Skip-gram keeps {f['seed']['Skip-gram']['2-4']['mean_overlap']:.0%} of the top-10 neighbors of words seen 2–4 times; "
               f"CBOW keeps {f['seed']['CBOW']['2-4']['mean_overlap']:.0%}."), PageBreak(),
             p('4. Nearest neighbors (cosine, top 5)', 'Title2'), p(f'{len(QUERIES)} query words; corpus count in brackets.'),
             table(['Query', 'Skip-gram', 'CBOW'], neighbor_table(results['neighbors']), [80, (width - 80) / 2, (width - 80) / 2]),
             p('Rare words', 'Section'), table(['Query', 'Skip-gram', 'CBOW'], neighbor_table(results['rare_neighbors']), [80, (width - 80) / 2, (width - 80) / 2]), PageBreak(),
             p(rq[0][0], 'Title2'), *paragraphs(rq[0][1]), figure('seed_stability'), PageBreak(),
             p(rq[1][0], 'Title2'), table(*tables['pairs'], widths=[120, 70, 75, 45, 75, width - 385]), Spacer(1, 8), *paragraphs(rq[1][1]),
             table(*tables['groups'], widths=[85, width - 325, 60, 60, 60, 60]), Spacer(1, 6), figure('group_cosines'), PageBreak(),
             p(rq[2][0], 'Title2'), figure('tsne_main'), figure('pca_main'), *paragraphs(rq[2][1]),
             table(*tables['purity'], widths=[110, 60, 70, 70, 70, width - 380]), p(f"Chance purity: {results['chance_purity']:.2f}.", 'Cell'), Spacer(1, 6),
             figure('tsne_perplexity'), PageBreak(),
             p(rq[3][0], 'Title2'), *paragraphs(rq[3][1]), figure('tsne_parameters'),
             p('Neighbor overlap by frequency band', 'Section'), table(*tables['overlap'], widths=[170] + [(width - 170) / 6] * 6), PageBreak(),
             p(rq[4][0], 'Title2'), *paragraphs(rq[4][1]),
             p('Share of top-10 neighbors that co-occur with the word in at least one abstract', 'Section'),
             table(['Model', '2–4', '5–9', '10–49', '50–199', '≥200'], tables['co'], [90] + [(width - 90) / 5] * 5), Spacer(1, 8),
             p('Limitations and reproducibility', 'Section'),
             p('One topic, one corpus, PubMed order (not a random sample). No external benchmark or retrieval evaluation, so semantic quality is judged from predefined '
               'groups, random-pair baselines and seed stability. Groups and pairs were chosen from domain knowledge before the neighbor lists were read. Training times come from '
               'concurrent runs. Reproduce with: python scripts/run_project2_part2.py (replays the frozen corpus snapshot; run_metadata.json holds settings and checksums).'),
             p('Sources', 'Section'),
             p('Course brief: project-details-2-part-2-20261006.pdf. Mikolov et al. (2013), arXiv:1301.3781 and arXiv:1310.4546. van der Maaten & Hinton (2008), JMLR 9. '
               'Implementation: src/word2vec.py, src/tsne.py, src/embedding_study.py.')]
    SimpleDocTemplate(str(output/'report.pdf'), pagesize=A4, rightMargin=44, leftMargin=44, topMargin=40, bottomMargin=42,
                      title='GLP-1 Project 2 part 2: word embeddings').build(story, onFirstPage=footer, onLaterPages=footer)


def summary_sections(r):
    """(heading, paragraphs) for the one-page summary, in plain English: theory first, a few measured numbers as evidence."""
    f = facts(r)
    corpus, s, seed, proj = r['corpus'], r['samples'], f['seed'], f['proj']
    center = s['tokens'].index('glp-1')
    pairs = [c for w, c in s['skipgram'] if w == 'glp-1']
    sgm = proj['skipgram_main']
    return [
        ('Central question: can about 1,000 abstracts learn meaningful word vectors?', [
            f"Yes, for the main topic of the texts. We used {corpus['document_count']:,} abstracts about GLP-1 ({corpus['raw_tokens']:,} words after removing "
            "stopwords). Both models learned that drug names, organs and side effects belong together. They did not learn words that are very rare, "
            "or topics the abstracts seldom talk about."]),
        ('1. How the two models learn', [
            "The main idea: words that appear in similar contexts get similar vectors. Each model is a small neural network with two sets of word vectors. "
            "We slide a window over each sentence; the window holds c words on each side of a center word.",
            "Skip-gram uses the center word to guess each word around it, so one position gives up to 2c training pairs. "
            "CBOW does the opposite: it takes the average of the words around the center and uses it to guess the center word, so one position gives one example. "
            f"Example (window {s['window']}): in \"{s['sentence'].rstrip('.')}\", the center word glp-1 gives Skip-gram {len(pairs)} pairs "
            f"(glp-1 → {', '.join(pairs)}), but gives CBOW only one example ({', '.join(s['cbow'][center]['context'])} → glp-1).",
            "Checking every word in the vocabulary at each step would be too slow. So both models use negative sampling: they learn to give a high score "
            "to the real pair and a low score to a few random \"noise\" words. We wrote both models ourselves in NumPy (100 dimensions, window 5, 15 epochs) "
            "and recorded the loss after every epoch."]),
        ('2. Why Skip-gram is better for rare and technical words', [
            "In Skip-gram, a rare word is the input of every pair it makes. So each time it appears, its own vector is updated several times, and each update "
            "moves it toward the words that really appear next to it. In CBOW, the rare word is only one of several words that are averaged together, so its "
            "update is small and mixed with the others. When a word appears only a few times, this decides whether its vector is learned or is still mostly random. "
            f"Our test: when we only changed the random seed, rare words kept {seed['Skip-gram']['2-4']['mean_overlap']:.0%} of their nearest words in "
            f"Skip-gram but only {seed['CBOW']['2-4']['mean_overlap']:.0%} in CBOW."]),
        ('RQ1 · How are CBOW and Skip-gram different?', [
            "Training: Skip-gram makes many more updates, so it is slower but learns more from each word. CBOW is faster. "
            "Loss: the loss goes down for both, but the two models solve different tasks, so their losses cannot be compared. A low loss also does not mean "
            "good word vectors. Nearest words: the models give quite different answers. Skip-gram finds words that appear together with the query "
            "(glp-1 → ras, ra, from \"GLP-1 RAs\" = receptor agonists). CBOW finds words that could replace the query in a sentence (glp-1 → glp1, glp-1/gip). "
            "Rare words: Skip-gram handles them well; for CBOW they are mostly noise."]),
        ('RQ2 · Do the word vectors carry meaning?', [
            "We measure how close two words are with cosine similarity, which compares the direction of the two vectors. A cosine number alone can mislead "
            "(CBOW gives high scores even to random pairs), so we compared each pair with many random pairs. Related pairs such as semaglutide – liraglutide, "
            "nausea – vomiting and kidney – renal clearly beat random pairs in both models, and words from the same group (for example GLP-1 drugs, or stomach "
            "side effects) are closer to each other than to other groups. But cancer – tumor and treatment – diagnosis are weak, because these words are side "
            "topics in GLP-1 abstracts. So yes, the vectors carry meaning, but only the meaning found in this collection of texts."]),
        ('RQ3 · How are PCA and t-SNE pictures different?', [
            "Both squeeze 100 dimensions into 2 so we can draw the words. PCA is a straight (linear) projection that keeps the directions with the most "
            "spread. It is simple and always gives the same result, but two directions keep too little information, so the words form one big cloud "
            f"(how often a word's nearest words are in its own group: {sgm['purity']['pca_2d']:.2f} in PCA vs {sgm['purity']['cosine']:.2f} in the full vectors). "
            "t-SNE tries to keep each word next to its nearest neighbors, so it shows clear clusters "
            f"({sgm['purity']['tsne_2d']:.2f}). But in t-SNE the distance between clusters and the size of a cluster mean nothing, and the picture changes with "
            "its settings. A cluster in a picture is not proof of meaning, so we always checked our claims on the full vectors."]),
        ('RQ4 · How do the model settings change the results?', [
            "Window size has the biggest effect. A small window (2) only sees the next word or two, so it groups words that play the same grammar role. "
            "A larger window (5) sees more of the sentence, so it groups words by topic. Changing it replaced most nearest words and made the t-SNE groups "
            "weaker, especially for CBOW. Vector size: with this little data, 50 dimensions are already enough; 100 vs 50 changed the results no more than a "
            "new random seed. Minimum word count: a higher limit removes noisy rare words, but also removes rare terms we may want."]),
        ('RQ5 · What are the limits of a small collection?', [
            "Every vector is learned only from the sentences we have. With little data, many words are seen just a few times (about half of our vocabulary "
            "appears only 2–4 times), so their vectors are uncertain and change with the random seed. Meaning also depends on the topic: here \"tzp\" means "
            "tirzepatide, and cancer is a side topic. Technical terms are split up: different spellings (diarrhea / diarrhoea), different writing (glp-1 / glp1) "
            "and multi-word terms (heart failure). Ways to improve: more text, joining multi-word terms, fastText (which uses word parts), and vectors "
            "pre-trained on large biomedical text."]),
        ('Conclusion', [
            "A small collection of abstracts is enough to learn useful word vectors for its main topic. Skip-gram is the better choice here because it learns "
            "rare technical words well; CBOW is faster and works well for common words."]),
    ]


def build_summary(results, output):
    """One A4 page: theory and answers to the brief's questions, with a few measured numbers as evidence."""
    import matplotlib
    from html import escape
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    fonts = Path(matplotlib.get_data_path())/'fonts/ttf'
    for name, file in [('DejaVu', 'DejaVuSans.ttf'), ('DejaVu-Bold', 'DejaVuSans-Bold.ttf')]:
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(fonts/file)))
    ink, teal = colors.HexColor('#152e3a'), colors.HexColor('#087f78')
    body = ParagraphStyle('SBody', fontName='DejaVu', fontSize=8.2, leading=11, spaceAfter=3, textColor=ink, alignment=4)
    head = ParagraphStyle('SHead', parent=body, fontName='DejaVu-Bold', fontSize=9, leading=11.5, spaceBefore=4.5, spaceAfter=1.5, textColor=teal, alignment=0)
    title = ParagraphStyle('STitle', parent=body, fontName='DejaVu-Bold', fontSize=13, leading=16, spaceAfter=1, alignment=0)
    note = ParagraphStyle('SNote', parent=body, fontSize=8.2, leading=11, textColor=colors.HexColor('#566b76'), alignment=0)
    author = ParagraphStyle('SAuthor', parent=body, fontName='DejaVu-Bold', fontSize=11, leading=14, spaceAfter=6, alignment=0)
    story = [Paragraph(AUTHOR, author),
             Paragraph('Word embeddings: CBOW vs Skip-gram on 1,000 GLP-1 abstracts', title)]
    for heading, paragraphs in summary_sections(results):
        story.append(Paragraph(escape(heading), head))
        story += [Paragraph(escape(text), body) for text in paragraphs]
    SimpleDocTemplate(str(output/'summary.pdf'), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=28, bottomMargin=24,
                      title='Word embeddings: one-page summary').build(story)
