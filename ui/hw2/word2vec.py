"""Page 4 (≈45 s): CBOW and Skip-gram word embeddings and their comparison (Brief · Word2Vec)."""
from html import escape
from pathlib import Path

import streamlit as st

from src.eos import split_sentences
from src.word2vec import ARCHITECTURES, Word2VecConfig, context_window, model_key, pair_scores
from src.zipf_experiment import embedding_preprocessing, embedding_sentences, token_trace, tokens
from ui.hw2 import architecture_eval, common, training
from ui.hw2.embedding_map import embedding_map

ARCHITECTURE_LABELS = {'skipgram': 'Skip-gram', 'cbow': 'CBOW'}


def short(text, limit=110):
    return text if len(text) <= limit else text[:limit].rsplit(' ', 1)[0] + ' …'


def struck_sentence(sentence):
    """The original sentence with the words removed by condition C (stopwords) struck through."""
    out, cursor = [], 0
    for row in token_trace(sentence):
        out.append(escape(sentence[cursor:row['start']]))
        word = escape(sentence[row['start']:row['end']])
        out.append(word if row['C'] else f'<s>{word}</s>')
        cursor = row['end']
    out.append(escape(sentence[cursor:]))
    return ''.join(out)


def window_view(words, center, window):
    cells = []
    for i, word in enumerate(words):
        style = 'on' if i == center else ('' if abs(i - center) > window else 'ctx')
        cells.append(f'<span class="{style}">{escape(word)}</span>')
    return '<div class="p2-steps">' + ''.join(cells) + '</div>'


@st.fragment(run_every=2)
def training_progress():
    """Polls running jobs every 2 s; when one finishes, select its model and refresh the page."""
    for key, job in list(training.jobs().items()):
        if job['status'] == 'running':
            loss = f" · probe loss {job['loss']:.4f}" if job['loss'] is not None else ''
            st.progress(job['epoch'] / job['total'],
                        text=f"Training {ARCHITECTURE_LABELS[job['architecture']]} in the background: epoch {job['epoch']}/{job['total']}{loss}. "
                             'You can keep using the page.')
        elif job['status'] == 'done' and not job.get('announced'):
            job['announced'] = True
            st.session_state.p2_model_path = job['path']
            st.session_state.p2_select_trained_model = True
            st.rerun(scope='app')
        elif job['status'] == 'failed' and not job.get('announced'):
            job['announced'] = True
            st.error(f"Training failed: {job['error']}")


def model_candidates():
    """Published Skip-gram, the part 2 CBOW/Skip-gram pair, then models trained on this page."""
    default = common.selected_output() / 'skipgram.npz'
    candidates = [p for p in [default, *architecture_eval.main_models()] if p.exists()]
    return candidates + sorted(common.MODELS.glob('skipgram_*.npz'))


def load_model(corpus, prep):
    """Pick a saved model for this corpus (or train one); return it or None."""
    default = common.selected_output() / 'skipgram.npz'
    path = Path(st.session_state.get('p2_model_path', str(default)))
    with st.expander('Training settings / train another model'):
        with st.form('p2_train'):
            architecture = st.radio('Architecture', list(ARCHITECTURES), horizontal=True, format_func=ARCHITECTURE_LABELS.get,
                                    help='Skip-gram: each word predicts its window. CBOW: the averaged window predicts the word.')
            a, b, c = st.columns(3)
            dims = a.selectbox('Dimensions', [32, 64, 100, 128], index=1)
            window = b.number_input('Context window', 1, 15, 5)
            epochs = c.number_input('Epochs', 1, 100, 15)
            a, b, c = st.columns(3)
            minimum = a.number_input('Minimum count', 1, 100, 2)
            negatives = b.number_input('Negative samples', 1, 20, 5)
            seed = c.number_input('Seed', 0, 2147483647, 42)
            submitted = st.form_submit_button('Train in the background')
        if submitted:
            config = Word2VecConfig(dimensions=dims, window=int(window), epochs=int(epochs), min_count=int(minimum),
                                    negatives=int(negatives), seed=int(seed), architecture=architecture)
            key = model_key(config, corpus.fingerprint, prep)
            target = common.MODELS / f'skipgram_{key}.npz'
            if target.exists():
                st.session_state.p2_model_path = str(target)
                st.session_state.p2_select_trained_model = True
                st.info('A model with exactly these settings already exists; it is now selected.')
            else:
                training.start(key, embedding_sentences(corpus), config, corpus.metadata(), prep, target)
        training_progress()
        if path != default and st.button('Use published model'):
            st.session_state.pop('p2_model_path', None)
            st.session_state.p2_select_trained_model = True
            st.rerun()
    catalog = common.model_catalog(tuple(common.file_stamp(p) for p in model_candidates()), corpus.fingerprint)
    if catalog:
        options = list(catalog)
        preferred = str(path) if str(path) in options else options[0]
        if st.session_state.pop('p2_select_trained_model', False):
            st.session_state.pop('p2_model_selector', None)
        if st.session_state.get('p2_model_selector') not in options:
            st.session_state.p2_model_selector = preferred
        path = Path(st.selectbox('Saved models for this corpus', options, key='p2_model_selector', format_func=lambda p: catalog[p]))
    if not path.exists():
        st.info('No trained model is available. Train explicitly above, or run scripts/run_project2.py.')
        return None
    try:
        model = common.saved_model(str(path), common.file_stamp(path))
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Cannot load this model: {exc}. Train a new model to recover.')
        return None
    if model.metadata.get('model_key') != model_key(Word2VecConfig.from_dict(model.metadata['config']), corpus.fingerprint, prep):
        st.warning('Saved model belongs to a different corpus or pipeline. Train a compatible model above.')
        return None
    return model


def training_pairs(corpus):
    """Real sentence, movable center word, and the center → context pairs it produces."""
    doc = st.selectbox('1 · Abstract', corpus.documents, format_func=lambda d: f'{d.pmid} | {d.title}', key='p2_w2v_doc')
    # (original sentence, its condition C tokens) for every sentence that yields at least one pair
    sentences = [(s, words) for paragraph in doc.abstract.split('\n\n') for s in split_sentences(paragraph)
                 if len(words := tokens(s, 'C')) >= 2]
    if not sentences:
        st.info('This abstract has no sentence with two or more condition C tokens.')
        return None, []
    s_idx = st.selectbox('2 · Sentence of this abstract', range(len(sentences)), key=f'p2_w2v_sentence_{doc.pmid}',
                         format_func=lambda i: f'{i + 1}/{len(sentences)} · {short(sentences[i][0])}')
    original, words = sentences[s_idx]
    st.markdown(f'<div class="p2-original"><span>Original sentence</span>{struck_sentence(original)}</div>', unsafe_allow_html=True)
    b, c, d = st.columns([3, 1, 2])
    center = b.slider('3 · Center word (position)', 0, len(words) - 1, min(2, len(words) - 1), key=f'p2_w2v_center_{doc.pmid}_{s_idx}')
    window = c.number_input('Window c', 1, 10, 2, key='p2_w2v_window')
    view = d.radio('Training example as', list(ARCHITECTURE_LABELS), horizontal=True, format_func=ARCHITECTURE_LABELS.get, key='p2_w2v_view')
    st.markdown('<div class="p2-original"><span>Tokens used for training (condition C)</span></div>', unsafe_allow_html=True)
    st.markdown(window_view(words, center, window), unsafe_allow_html=True)
    pairs = context_window(words, center, window)
    contexts = [p['context'] for p in pairs]
    md = common.md_escape
    if view == 'skipgram':
        st.caption(f'Skip-gram: the green word predicts each orange word — {len(pairs)} examples: '
                   + ', '.join(f"{md(p['center'])} → {md(p['context'])}" for p in pairs)
                   + '. Each example raises the score σ(v·u) of the center and context vectors; '
                     'k noise words, drawn in proportion to frequency^0.75, have their score lowered.')
    else:
        st.caption(f'CBOW: the {len(contexts)} orange words are averaged into one vector h that predicts the green word — 1 example: '
                   f'mean({", ".join(md(w) for w in contexts)}) → {md(words[center])}. It raises the score σ(h·u) of the center word; '
                   'k noise words have their score lowered.')
    return words[center], contexts


def compare_architectures(corpus, prep, word):
    """Neighbours of one word under the Skip-gram and the CBOW model of this corpus, side by side."""
    # Prefer the part 2 pair: identical settings, only the architecture differs.
    pair = [p for p in architecture_eval.main_models() if p.exists()]
    candidates = pair + [p for p in model_candidates() if p not in pair]
    catalog = common.model_catalog(tuple(common.file_stamp(p) for p in candidates), corpus.fingerprint)
    chosen = {}
    for path in catalog:
        name = 'cbow' if catalog[path].startswith('CBOW') else 'skipgram'
        chosen.setdefault(name, path)
    if len(chosen) < 2:
        st.caption('Train a CBOW model (Model & training below) to compare it with Skip-gram here.')
        return
    columns = st.columns(2)
    for column, name in zip(columns, ('skipgram', 'cbow')):
        model = common.saved_model(chosen[name], common.file_stamp(Path(chosen[name])))
        with column:
            title, settings = catalog[chosen[name]].split(' · ', 1)
            st.markdown(f'**{title}** · {settings}')
            if word in model.words:
                st.dataframe([{'word': r['word'], 'cosine': round(r['cosine_similarity'], 3)} for r in model.most_similar(word, 8)],
                             hide_index=True, width='stretch')
            else:
                st.caption(f"'{word}' is not in this model's vocabulary.")


def render(corpus, result):
    prep = embedding_preprocessing('C')
    pairs_card = common.card('Training examples: a word and its window',
                             'Skip-gram predicts the window from the word; CBOW predicts the word from the window. '
                             'Condition C tokens (stopwords removed, no stemming); windows never cross sentences.')
    learned_card = common.card('What the trained vectors learned', 'Cosine similarity of learned center vectors; PCA of all vectors.')
    compare_card = common.card('Skip-gram vs CBOW on the same corpus', 'Same settings except the architecture; nearest words of the word explored above.')
    eval_card = common.card('Evaluation: CBOW vs Skip-gram',
                            'Ten models (5 settings × 2 architectures) trained from scratch on the 1,000 abstracts; '
                            'training cost, loss, semantic pairs and groups, neighbour stability and projection purity.')
    common.more('Model & training', 'NumPy implementation (src/word2vec.py); no pretrained vectors or Word2Vec library.')
    details = st.container()
    with details:
        model = load_model(corpus, prep)
    with pairs_card:
        center, contexts = training_pairs(corpus)
    if model is None:
        return
    with learned_card:
        left, right = st.columns([1, 2])
        with left:
            query = st.text_input('Explore a word', 'glp-1', key='p2_embedding_query')
            words_q = tokens(query, 'C')
            valid = len(words_q) == 1 and words_q[0] in model.words
            if valid:
                neighbors = model.most_similar(words_q[0], 10)
                st.dataframe([{'word': r['word'], 'cosine': round(r['cosine_similarity'], 3)} for r in neighbors], hide_index=True, width='stretch')
            else:
                st.info('Enter one word in the trained vocabulary, e.g. glp-1, semaglutide, receptor, insulin.')
        with right:
            embedding_map(model, words_q[0] if valid else None, model.metadata.get('model_key', 'model'))
    with compare_card:
        if words_q:
            compare_architectures(corpus, prep, words_q[0])
    with eval_card:
        architecture_eval.render(words_q[0] if valid else architecture_eval.EXAMPLE)
    with details:
        config = Word2VecConfig.from_dict(model.metadata['config'])
        cols = st.columns(5)
        cols[0].metric('Vocabulary', f'{len(model.words):,}')
        cols[1].metric('Dimensions', model.vectors.shape[1])
        cols[2].metric('Window', config.window)
        cols[3].metric('Negatives', config.negatives)
        cols[4].metric('Training pairs', f"{model.metadata.get('training_pairs', 0):,}")
        st.latex(r'\log\sigma(\mathbf{v}_{c}^{\top}\mathbf{u}_{o})+\sum_{i=1}^{k}\log\sigma(-\mathbf{v}_{c}^{\top}\mathbf{u}_{n_i})')
        common.chart({'data': {'values': model.metadata['history']}, 'mark': {'type': 'line', 'point': True}, 'height': 200, 'title': 'Probe loss per epoch',
                      'encoding': {'x': {'field': 'epoch', 'type': 'quantitative'}, 'y': {'field': 'probe_loss', 'type': 'quantitative', 'scale': {'zero': False}}}},
                     key='p2_w2v_loss')
        st.caption(model.metadata['loss_note'])
        if center in model.words:
            scored = pair_scores(model, center, contexts + [w for w in model.words[-3:] if w not in contexts])
            for row in scored:
                row['pair'] = 'context in window' if row['context'] in contexts else 'rare word (not in window)'
            st.markdown(f'**σ(v·u) for the demo pairs** (center "{center}")')
            st.dataframe(scored, hide_index=True, width='stretch')
        if valid:
            common.csv_download('Download nearest words', model.most_similar(words_q[0], 10), 'nearest_words.csv')
        with st.expander('Model configuration and corpus evidence'):
            st.json(model.metadata)
        st.download_button('Download trained model', model.to_bytes(), 'skipgram.npz', 'application/octet-stream')
