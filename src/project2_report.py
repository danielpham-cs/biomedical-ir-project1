"""Data-grounded figures, research answers and PDF exports for Project 2."""
from __future__ import annotations
import json
import math
import re
from html import escape
from pathlib import Path

import numpy as np

from .tables import rows_csv
from .embedding_view import pca_2d, projection_rows


def fmt(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def findings(result):
    conditions = result['conditions']
    b = conditions['B']
    fit = b['fits']['All']
    eligible = [(name, f) for name, f in b['fits'].items() if name != 'All' and f['r_squared'] is not None and f['points'] >= 2]
    best_name, best = min(eligible, key=lambda item: item[1]['rmse_log10']) if eligible else ('none', fit)
    terms = b['selected_terms']
    common = max(terms, key=lambda r: r['DF'])
    concentrated = max(terms, key=lambda r: r['CF_per_document'])
    a,c,d = [conditions[x]['stats'] for x in 'ACD']
    exponent_text = ', '.join(f"{k}: {v['fits']['All']['zipf_exponent']:.4f}" for k,v in conditions.items())
    return {
        'RQ1 — Does the corpus follow Zipf?':
            f"The rank-frequency curves have a high-frequency head and a long rare-term tail, with an approximately linear middle on log-log axes. "
            f"For condition B, the full-vocabulary fit has R²={fit['r_squared']:.4f} and log10 RMSE={fit['rmse_log10']:.4f}. "
            f"The exponent differs from the idealized value 1; the corpus is Zipf-like rather than an exact 1/r distribution. "
            f"Among predefined regions with nonconstant frequencies, {best_name} has the smallest log-space RMSE "
            f"({best['rmse_log10']:.4f}, R²={fmt(best['r_squared'])}). The complete regional table and residuals show where the approximation changes. "
            "R² alone is insufficient: it measures a descriptive OLS fit after a logarithmic transformation. Ranked observations are dependent, "
            "many tail frequencies are tied, and no competing distribution or formal power-law goodness-of-fit test is evaluated. Constant tails have undefined R², not proof of an exact law.",
        'RQ2 — What is the empirical exponent?':
            f"With log10(CF)=a+m*log10(rank), k=-m. Full-vocabulary estimates are {exponent_text}. "
            f"Condition B has slope {fit['slope']:.4f}, intercept {fit['intercept']:.4f}, and k={fit['zipf_exponent']:.4f}. "
            "The estimate depends on preprocessing and the fitted rank range. The 1/r reference is anchored to the most frequent term; it is not the fitted regression line.",
        'RQ3 — How does preprocessing affect the distribution?':
            f"A contains {a['tokens']:,} tokens and {a['vocabulary']:,} distinct terms. Boundary punctuation normalization reduces vocabulary "
            f"to {b['stats']['vocabulary']:,} in B, while retaining internal punctuation in biomedical identifiers such as GLP-1. "
            f"Stopword removal reduces tokens from {b['stats']['tokens']:,} to {c['tokens']:,}; vocabulary becomes {c['vocabulary']:,}. "
            f"Porter then reduces vocabulary to {d['vocabulary']:,} ({100*(1-d['vocabulary']/c['vocabulary']):.2f}% below C), with {d['tokens']:,} retained tokens. "
            "Punctuation merges surface variants, stopwords reshape the head, and stemming combines morphological forms throughout the curve. "
            "Neither a smaller vocabulary nor an exponent closer to one proves better retrieval. Inspect the Top 50 tables and merged forms for specific examples.",
        'RQ4 — What is the relationship between CF and DF?':
            f"CF counts occurrences; DF counts distinct documents. In condition B, '{common['term']}' occurs {common['CF']:,} times across "
            f"{common['DF']:,} abstracts, with IDF={common['IDF']:.4f}. In contrast, '{concentrated['term']}' has CF={concentrated['CF']}, "
            f"DF={concentrated['DF']} and CF/DF={concentrated['CF_per_document']:.2f}: repetition is concentrated rather than widespread. "
            "A paper can repeat a specialized abbreviation many times, raising CF while contributing only one to DF. DF/N directly describes document coverage. "
            "Both measures matter: CF captures repetition and processing volume; DF determines postings-list coverage and collection-level discrimination. "
            "IDF=ln(N/DF) is smaller for terms occurring in many documents. CF alone cannot determine IDF, because equal CF can have different DF.",
        'RQ5 — Why does Zipf matter to IR?': ir_essay(result),
        'Final challenge — Why is an approximate law useful?':
            f"The four conditions produce different exponents ({exponent_text}), yet all show strongly uneven term frequencies. "
            "That robust qualitative structure is useful even though no single exact law explains every rank. It motivates separate treatment of very common "
            "terms, compact storage for a large rare-term vocabulary, and weighting based on document coverage. These are engineering hypotheses supported by "
            "observed counts, not universal guarantees. This study measures vocabulary reduction and frequency structure; it does not measure compressed index "
            "bytes or retrieval relevance. A subsequent evaluation should compare latency, actual storage, and judged search quality before claiming an improvement. "
            "The useful lesson is to let the empirical curve guide design, then validate the design against the system's retrieval objective."
    }


def ir_essay(result):
    b,c,d = [result['conditions'][x]['stats'] for x in 'BCD']
    return (
        f"Zipf's law matters because an information retrieval system processes a highly uneven vocabulary. In this collection, condition B contains "
        f"{b['vocabulary']:,} distinct terms and {b['tokens']:,} occurrences. A small head contributes many occurrences, while numerous rare forms occupy "
        "the tail. This structure connects language statistics to indexing cost and term weighting. It is an empirical design guide rather than a rule "
        "that every biomedical collection must satisfy exactly.\n\n"
        f"Stopword removal changes the workload substantially: condition C retains {c['tokens']:,} tokens, compared with {b['tokens']:,} in B. "
        "Common grammatical words often occur in most documents, so their presence provides little evidence that one paper is more relevant than another. "
        "Removing them can shorten query processing and positional data. However, frequent words are not automatically useless: phrases, negation, and "
        "biomedical concepts may depend on them. The experiment demonstrates changed counts, not an improvement in retrieval effectiveness. Phrase search "
        "should therefore preserve the original text and use an explicitly documented matching policy.\n\n"
        "For an inverted index, each unique term needs a dictionary entry and a postings list. High-DF terms have long document lists; high-CF terms may "
        "also have many stored positions. The long tail creates many short lists and dictionary entries, even when each term has few occurrences. "
        f"Porter reduces vocabulary from {c['vocabulary']:,} to {d['vocabulary']:,} between C and D, suggesting a smaller dictionary, but merged lists may "
        "become longer. Actual storage also depends on document-ID gaps, position encoding and implementation overhead. Gap-based integer compression "
        "and compact dictionaries are relevant design options; this study does not claim measured compression savings.\n\n"
        "TF-IDF separates within-document repetition from collection-wide prevalence. Raw TF can reward a term repeated in a document, while "
        "IDF=ln(N/DF) discounts terms shared by many documents. A term found in every document has IDF zero under this formula. High CF alone does "
        "not imply low IDF: a specialized term can be repeated heavily in only a few abstracts. Comparing CF and DF reveals this distinction. "
        "Zipf's head helps explain why frequency alone is an inadequate ranking signal, but DF must be measured independently. Helix keeps BM25 "
        "ranking separate from this textbook-IDF analysis. Ultimately, frequency statistics justify concrete experiments on index size, query latency "
        "and relevance judgments; they cannot replace those evaluations."
    )


def plot_figures(result, model, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})
    colors = {'A':'#64748b','B':'#087f78','C':'#c07828','D':'#7654a3'}
    for mode in ('rank_frequency','loglog_fit','residuals'):
        fig, axes = plt.subplots(2,2,figsize=(10.5,7.3), constrained_layout=True)
        for (condition,data),ax in zip(result['conditions'].items(),axes.flat):
            x = np.array([r['rank'] for r in data['rows']])
            y = np.array([r['CF'] for r in data['rows']])
            fit=data['fits']['All']
            predicted=10**fit['intercept']*x**fit['slope']
            if mode=='residuals':
                ax.semilogx(x,np.log10(y)-np.log10(predicted),color=colors[condition],lw=1)
                ax.axhline(0,color='#64748b',ls='--',lw=.8)
                ax.set_ylabel('Observed - fitted log10(CF)')
            else:
                ax.plot(x,y,color=colors[condition],lw=1.3,label='Observed CF')
                if mode=='loglog_fit':
                    ax.set_xscale('log'); ax.set_yscale('log')
                    ax.plot(x,predicted,color='#c64f4f',ls='--',label=f"Fit k={fit['zipf_exponent']:.3f}")
                    ax.plot(x,y[0]/x,color='#8b8b8b',ls=':',label='1/r reference')
                ax.set_ylabel('Collection frequency (CF)'); ax.legend(fontsize=8)
            ax.set_title(f'Condition {condition} | {len(x):,} terms')
            ax.set_xlabel('Term rank'); ax.grid(alpha=.15)
        fig.savefig(directory/f'{mode}.png',dpi=180); fig.savefig(directory/f'{mode}.pdf'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8),constrained_layout=True)
    for condition,data in result['conditions'].items():
        ax.loglog([r['rank'] for r in data['rows']],[r['CF'] for r in data['rows']],color=colors[condition],label=condition,lw=1.3)
    ax.set(xlabel='Term rank (log scale)',ylabel='Collection frequency (log scale)',title=f"Same {result['corpus']['document_count']:,} abstracts, four preprocessing conditions")
    ax.legend(title='Condition'); ax.grid(alpha=.15); fig.savefig(directory/'conditions.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.6),constrained_layout=True)
    rows=result['conditions']['B']['selected_terms']
    ax.scatter([r['DF'] for r in rows],[r['CF'] for r in rows],color='#087f78')
    for term in ('and','glp-1','semaglutide','tirzepatide','cdr'):
        for row in rows:
            if row['term']==term: ax.annotate(term,(row['DF'],row['CF']),xytext=(5,5),textcoords='offset points',fontsize=8)
    ax.set(xscale='log',yscale='log',xlabel='Document frequency',ylabel='Collection frequency',title='Condition B | selected 20 terms')
    ax.grid(alpha=.15); fig.savefig(directory/'cf_df.png',dpi=180); plt.close(fig)
    history=model.metadata['history']
    fig,ax=plt.subplots(figsize=(8,3.8),constrained_layout=True)
    ax.plot([r['epoch'] for r in history],[r['probe_loss'] for r in history],color='#087f78',marker='o')
    ax.set(xlabel='Epoch (0 = initialization)',ylabel='Fixed training probe loss',title='Skip-gram training (not held-out accuracy)')
    ax.grid(alpha=.15); fig.savefig(directory/'training_loss.png',dpi=180); plt.close(fig)


def export_report(result, model, output):
    output=Path(output)
    answers=findings(result)
    word_count=len(re.findall(r"\S+",answers['RQ5 — Why does Zipf matter to IR?']))
    if not 300 <= word_count <= 500:
        raise ValueError(f'IR discussion has {word_count} words, expected 300–500.')
    plot_figures(result,model,output/'figures')
    coordinates,variance=pca_2d(model.vectors)
    (output/'tables/pca_coordinates.csv').write_text(rows_csv(projection_rows(model,coordinates,mode='frequent',limit=len(model.words))),encoding='utf-8-sig')
    examples=[w for w in ('glp-1','semaglutide','tirzepatide','receptor','insulin') if w in model.words]
    neighbors=[dict(query=word,**row) for word in examples for row in model.most_similar(word,10)]
    (output/'tables/neighbors.csv').write_text(rows_csv(neighbors),encoding='utf-8-sig')
    (output/'research_answers.json').write_text(json.dumps(answers,ensure_ascii=False,indent=2),encoding='utf-8')
    pca={'explained_variance_ratio':variance.tolist(),'model_key':model.metadata['model_key']}
    (output/'pca_metadata.json').write_text(json.dumps(pca,indent=2),encoding='utf-8')
    meta=result['corpus']
    methods=(f"Corpus: {meta['document_count']:,} English PubMed abstracts. Query: {meta['query']}. "
        "The saved selection follows PubMed's returned order, excluding previously archived papers; it is not a random sample. "
        "Each PMID is unique and every record has a nonempty English abstract. Only AbstractText content is analyzed, without titles, keywords, section labels or article bodies. "
        "Four cumulative conditions use exactly the same raw abstracts. A splits on whitespace and lowercases, keeping attached punctuation. "
        "B strips non-alphanumeric boundaries, preserves numeric trailing percent signs and internal punctuation, and normalizes Unicode hyphens/apostrophes. "
        "C removes the recorded NLTK English stopword list. D applies the local original Porter 1980 algorithm to ASCII alphabetic tokens. "
        "All vocabulary terms are ranked by decreasing CF, with alphabetic tie-breaking. Regression uses unweighted OLS of log10(CF) against log10(rank). "
        "Rank regions were fixed before interpreting the fits: first 10%, next 70%, last 20%, with ceiling boundaries. Textbook IDF uses natural logarithms. "
        "The corpus snapshot, settings, stopword list, source hashes, exported tables and run metadata support offline reproduction.")
    summary_rows=[dict(condition=c,**d['stats'],**d['fits']['All']) for c,d in result['conditions'].items()]
    lines=['# GLP-1 scientific abstracts: Project 2 report','',methods,'',f"Corpus fingerprint: `{meta['fingerprint']}`",'',
           '| Condition | Tokens | Vocabulary | Average tokens/doc | k | R² | RMSE (log10) |','|---|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {r['condition']} | {r['tokens']} | {r['vocabulary']} | {r['average_tokens_per_document']:.3f} | {fmt(r['zipf_exponent'])} | {fmt(r['r_squared'])} | {fmt(r['rmse_log10'])} |" for r in summary_rows]
    for name in ('conditions','rank_frequency','loglog_fit','residuals','cf_df'):
        lines += ['',f'![{name}](figures/{name}.png)','']
    for title,answer in answers.items(): lines += [f'## {title}','',answer,'']
    lines += ['## Word2Vec','',f"Skip-gram with negative sampling; condition C; {len(model.words):,} words × {model.vectors.shape[1]} dimensions. "
              f"Fixed training probe loss: {model.metadata['history'][0]['probe_loss']:.4f} → {model.metadata['history'][-1]['probe_loss']:.4f}. "
              f"PCA 2D retains {sum(variance):.2%} of normalized-vector variance. Training loss and cosine are not retrieval accuracy.",'',
              '```json',json.dumps(model.metadata['config'],indent=2),'```','',
              '![Training loss](figures/training_loss.png)','',
              '## Limitations','',
              'The single-topic, date-ordered sample is not representative of all biomedical language. Boundary-only punctuation handling retains some compound strings and numerical values. '
              'Stemming can merge distinct meanings. OLS diagnostics are descriptive, not a formal power-law test. No relevance labels, held-out embedding benchmark, '
              'retrieval-quality comparison or actual index-compression benchmark were collected.','',
              '## Reproduce','',r'Run `.\.venv\python.exe scripts\run_project2.py`. To replay frozen texts offline, add `--snapshot output/project2_glp1/corpus_snapshot.json`. All source XML remains local.','',
              '## Sources','',
              '- Course briefs: Homework-2-26-20260922.pdf and project-2-details-20260929.pdf.',
              '- Source corpus: PubMed XML records identified in corpus_snapshot.json and run_metadata.json.',
              '- Algorithm implementations: src/zipf_experiment.py, src/porter.py, src/word2vec.py.',
              '- The exact stopword list and all experiment choices are recorded in analysis.json.','',
              f'IR implications discussion length: {word_count} whitespace-delimited words.']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    (output/'README.md').write_text('Project 2 submission\n\nreport.pdf: full report, including Top 50 tables for A–D.\nexecutive_summary.pdf: one-page summary.\n'
        'analysis.json: complete statistics, settings and fits.\ncorpus_snapshot.json: frozen abstract texts and source hashes.\n'
        'tables/: all frequencies, selected terms, Porter groups, model neighbors and PCA.\nfigures/: plotted evidence.\n'
        'skipgram.npz and training.json: trained model/config/history.\nrun_metadata.json: checksums and environment.\n'
        'Reproduce the current manifest: .\\.venv\\python.exe scripts\\run_project2.py\n'
        'Or replay frozen texts: python scripts/run_project2.py --snapshot output/project2_glp1/corpus_snapshot.json\n',encoding='utf-8')
    build_pdfs(result,model,output,answers,methods,summary_rows)


def build_pdfs(result,model,output,answers,methods,summary_rows):
    import matplotlib
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
    fonts=Path(matplotlib.get_data_path())/'fonts/ttf'
    for name,file in [('DejaVu','DejaVuSans.ttf'),('DejaVu-Bold','DejaVuSans-Bold.ttf')]:
        if name not in pdfmetrics.getRegisteredFontNames(): pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVu-Bold',italic='DejaVu',boldItalic='DejaVu-Bold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Body',fontName='DejaVu',fontSize=9.3,leading=13.7,spaceAfter=8,textColor=colors.HexColor('#152e3a')))
    styles.add(ParagraphStyle(name='SmallCell',parent=styles['Body'],fontSize=7.5,leading=10,spaceAfter=0))
    styles.add(ParagraphStyle(name='Title2',parent=styles['Body'],fontName='DejaVu-Bold',fontSize=21,leading=26,spaceAfter=18))
    styles.add(ParagraphStyle(name='Section',parent=styles['Body'],fontName='DejaVu-Bold',fontSize=13,leading=18,spaceBefore=10,spaceAfter=10,textColor=colors.HexColor('#087f78')))
    width=A4[0]-88
    def p(text,style='Body'): return Paragraph(escape(str(text)).replace('\n','<br/>'),styles[style])
    def table(headers,rows,widths=None):
        values=[[p(v,'SmallCell') for v in headers]]+[[p(v,'SmallCell') for v in row] for row in rows]
        t=Table(values,colWidths=widths or [width/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6f3f0')),('VALIGN',(0,0),(-1,-1),'TOP'),
                              ('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),
                              ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#087f78')),
                              ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dce5e8'))]))
        return t
    def figure(name,ratio): return Image(str(output/'figures'/f'{name}.png'),width=width,height=width*ratio)
    def footer(canvas,doc):
        canvas.setFont('DejaVu',8); canvas.setFillColor(colors.HexColor('#566b76'))
        canvas.drawString(44,25,'Helix | GLP-1 | Project 2'); canvas.drawRightString(A4[0]-44,25,str(doc.page))
    story=[p('GLP-1 scientific abstracts','Title2'),p('Project 2 · Experimental report','Section'),p(methods),
           p('Corpus fingerprint','Section'),p(result['corpus']['fingerprint'],'SmallCell'),
           p('Evidence package','Section'),p('Tables and plots use one frozen corpus. All numeric claims refer to the recorded condition. The textbook IDF analysis is separate from BM25 retrieval.'),PageBreak(),
           p('Preprocessing and vocabulary','Title2'),
           table(['Stage','Tokens','Vocabulary','Mean tokens/doc','Hapax'],[[r['condition'],f"{r['tokens']:,}",f"{r['vocabulary']:,}",f"{r['average_tokens_per_document']:.3f}",f"{r['hapax']:,}"] for r in summary_rows]),
           Spacer(1,12),figure('conditions',.6),p('RQ3 — Preprocessing effects','Section'),p(answers['RQ3 — How does preprocessing affect the distribution?']),PageBreak(),
           p('Rank-frequency distributions','Title2'),figure('rank_frequency',7.3/10.5),p('Frequency is CF, summed over the collection. Ranks start at one; alphabetic order breaks ties. The linear view makes the dominance of the head visible.'),PageBreak(),
           p('Log-log regression','Title2'),figure('loglog_fit',7.3/10.5),
           table(['Stage','Slope','Intercept','k','R²','RMSE log10'],[[r['condition'],fmt(r['slope']),fmt(r['intercept']),fmt(r['zipf_exponent']),fmt(r['r_squared']),fmt(r['rmse_log10'])] for r in summary_rows]),
           Spacer(1,10),p(answers['RQ2 — What is the empirical exponent?']),PageBreak(),
           p('Regional fit and residuals','Title2')]
    regional=[]
    for c,data in result['conditions'].items():
        for name,f in data['fits'].items():
            if name!='All': regional.append([c,name,f"{f['rank_start']}–{f['rank_end']}",fmt(f['zipf_exponent']),fmt(f['r_squared']),fmt(f['rmse_log10'])])
    story += [table(['Stage','Region','Ranks','k','R²','RMSE'],regional,[35,130,95,70,85,width-415]),Spacer(1,10),p(answers['RQ1 — Does the corpus follow Zipf?']),PageBreak(),
              p('Residuals from the full fit','Title2'),figure('residuals',7.3/10.5),p('Discrete low-frequency ties create a structured tail. A constant-frequency region has undefined R²; a near-zero residual there is not evidence that the complete distribution is an exact power law.'),PageBreak(),
              p('CF, DF and textbook IDF','Title2'),p(f"Condition B; IDF = ln(N/DF), N = {result['corpus']['document_count']:,}. Selection: top 10 CF, available biomedical terms, then high CF/DF terms with CF ≥ 5."),
              table(['Term','CF','DF','CF/DF','IDF'],[[r['term'],r['CF'],r['DF'],fmt(r['CF_per_document'],2),fmt(r['IDF'])] for r in result['conditions']['B']['selected_terms']],[180,75,75,85,width-415]),
              Spacer(1,10),p(answers['RQ4 — What is the relationship between CF and DF?']),PageBreak(),
              p('CF versus document coverage','Title2'),figure('cf_df',4.6/8),p('Terms with similar CF may have different DF and therefore different IDF. Numeric tables and the interactive term-to-document view support inspection of actual repetition.'),PageBreak(),
              p('Information retrieval implications','Title2'),p(answers['RQ5 — Why does Zipf matter to IR?']),p('Final challenge','Section'),p(answers['Final challenge — Why is an approximate law useful?']),PageBreak(),
              p('Word2Vec: Skip-gram','Title2'),p(f"Condition C, {len(model.words):,} vocabulary words, {model.vectors.shape[1]} dimensions. Config: {json.dumps(model.metadata['config'])}"),
              figure('training_loss',3.8/8),p('Probe loss uses fixed training examples and negative samples, including initialization. It is not held-out accuracy. Neighbors measure context similarity, not guaranteed biomedical synonymy.'),
              table(['Query','Five nearest words'],[[w,', '.join(f"{r['word']} ({r['cosine_similarity']:.3f})" for r in model.most_similar(w,5))] for w in ('glp-1','semaglutide','tirzepatide') if w in model.words],[85,width-85]),PageBreak(),
              p('Limitations and reproducibility','Title2'),p('The sample is one English biomedical topic selected in PubMed order, not a random cross-domain corpus. Tokenization retains internal compounds and numerical strings. Stemming can merge distinct meanings. There are no relevance labels or external embedding benchmark, so no retrieval-accuracy or semantic-quality claim is made.'),
              p('Reproduce','Section'),p('Run .\\.venv\\python.exe scripts\\run_project2.py. Source abstracts and hashes are in corpus_snapshot.json; run_metadata.json records settings, environment and file checksums. The exact stopword list is in analysis.json. Full frequencies and selected terms are CSV exports.'),
              p('Sources','Section'),p('Course briefs: Homework-2-26-20260922.pdf; project-2-details-20260929.pdf. PubMed source records are identified by PMID in the corpus snapshot. Local implementation: src/zipf_experiment.py, src/porter.py and src/word2vec.py.')]
    for condition,data in result['conditions'].items():
        story += [PageBreak(),p(f'Appendix: Top 50 · condition {condition}','Title2'),
                  table(['Rank','Term','CF','DF','IDF'],[[r['rank'],r['term'],r['CF'],r['DF'],fmt(r['IDF'])] for r in data['rows'][:50]],[40,230,75,75,width-420])]
    SimpleDocTemplate(str(output/'report.pdf'),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=40,bottomMargin=42,title='GLP-1 Project 2 experimental report').build(story,onFirstPage=footer,onLaterPages=footer)
    b=result['conditions']['B']; a=result['conditions']['A']; c=result['conditions']['C']; d=result['conditions']['D']
    executive=[p('Executive summary','Title2'),p(f"What the {result['corpus']['document_count']:,} GLP-1 abstracts revealed",'Section'),
        p(f"The experiment uses a frozen set of {result['corpus']['document_count']:,} English PubMed abstracts. Titles and keywords are excluded; four cumulative preprocessing conditions share identical source texts. The values below are measured results, not textbook expectations."),
        table(['Condition','Vocabulary','Zipf k','R²','RMSE log10'],[[r['condition'],f"{r['vocabulary']:,}",fmt(r['zipf_exponent']),fmt(r['r_squared']),fmt(r['rmse_log10'])] for r in summary_rows]),
        p('1. Zipf is an approximation','Section'),p(f"Condition B is strongly linear on log-log axes (R²={b['fits']['All']['r_squared']:.4f}), but its estimated k={b['fits']['All']['zipf_exponent']:.4f} differs from 1. Head, middle and tied-frequency tail behave differently. R² does not establish a power law; regional fits and residuals reveal the limits of the approximation."),
        p('2. Preprocessing changes the empirical answer','Section'),p(f"Punctuation normalization reduces vocabulary from {a['stats']['vocabulary']:,} to {b['stats']['vocabulary']:,}. Stopword removal reduces tokens to {c['stats']['tokens']:,}; Porter reduces vocabulary further to {d['stats']['vocabulary']:,}. A single exponent without the preprocessing settings would hide these differences."),
        p('3. Frequency is not document coverage','Section'),p(answers['RQ4 — What is the relationship between CF and DF?'].split('A paper can')[0]),
        p('4. Implications and limits','Section'),p('The corpus supports explicit decisions about stopwords, dictionary size, postings and IDF weighting. It does not establish better retrieval accuracy or measured compression savings. The useful lesson beyond the textbook is the size and direction of preprocessing effects in this particular domain, and the need to inspect CF and DF separately. A fixed snapshot, public PMID list and exported settings make the conclusions reproducible.')]
    SimpleDocTemplate(str(output/'executive_summary.pdf'),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=36,bottomMargin=42,title='GLP-1 Project 2 executive summary').build(executive,onFirstPage=footer,onLaterPages=footer)
