"""Small, explicit English and Traditional-Chinese UI translation layer."""

from __future__ import annotations

import streamlit as st

LANGUAGES = {"English": "en", "台灣": "zh-TW"}

TEXT = {
    "en": {
        "language": "Language", "appearance": "Appearance", "dark_mode": "Dark mode",
        "overview": "Overview", "abstract_lookup": "Search and analyze abstract",
        "local_search": "Search local library", "advanced": "Advanced course pages",
        "collection": "XML import and index settings", "methods": "Methods and EOS playground",
        "settings": "Preferences",
        "hero_kicker": "LOCAL BIOMEDICAL RETRIEVAL", "hero_title": "Search evidence.<br><em>Inspect the text.</em>",
        "hero_body": "Retrieve PubMed abstracts by PMID or PMCID, inspect the exact statistics behind each record, and search a local collection with BM25.",
        "ready": "READY COLLECTION", "ready_title": "PubMed abstracts", "ready_body": "Structured XML, transparent ranking, and reproducible text rules.",
        "abstract_xml": "Abstract XML", "indexed_records": "Indexed records", "abstract_only": "Abstract-only", "vocabulary": "Vocabulary",
        "start_task": "Start a task", "lookup_title": "Look up one article", "lookup_body": "Paste a PMID or PMCID to retrieve its abstract, Keywords, word statistics, and rule-based EOS sentences.",
        "lookup_action": "Search PubMed abstract", "library_title": "Search the local library", "library_body": "Run AND, OR, or PHRASE retrieval across the downloaded abstract collection.",
        "library_action": "Open local search", "search_collection": "Search your collection", "keywords": "Keywords", "search_button": "Search local library",
        "search_hint": "AND and OR remove common English stopwords. PHRASE preserves them. All results use Porter stemming and BM25 ranking.",
        "how": "How Helix works", "how_caption": "The system design below maps directly to the retrieval, statistics, smart EOS, and visualization requirements of the project.",
        "technical": "Technical specification",
        "lookup_page_title": "PubMed abstract lookup and analysis", "lookup_page_heading": "Search an abstract. Analyze it immediately.",
        "lookup_page_body": "Enter an article ID to fetch its PubMed title and abstract. Helix then counts characters, words, terms, and EOS sentences without downloading or saving the article.",
        "pmid_or_pmcid": "PMID or PMCID", "search_abstract": "Search abstract", "lookup_hint": "Lookup requires an internet connection. Downloading and local storage are optional after results appear.",
        "abstract": "Abstract", "statistics": "Statistics", "sentences_eos": "Sentences and EOS", "characters": "Characters", "words": "Words", "sentences": "Sentences", "unique_words": "Unique words",
        "save_options": "Save or download options", "download_xml": "Download PubMed XML", "download_json": "Download abstract JSON",
        "search_page_title": "Local search and document analysis", "search_heading": "Search abstract, keywords, and body.", "search_body": "The title is excluded from matching. AND and OR remove common stopwords; PHRASE retains them. All modes apply Porter stemming and rank matching articles with BM25.",
        "query_mode": "Query mode", "search_terms": "Search terms", "no_saved": "No saved articles yet.", "did_you_mean": "Did you mean",
    },
    "zh-TW": {
        "language": "語言", "appearance": "外觀", "dark_mode": "深色模式",
        "overview": "首頁", "abstract_lookup": "搜尋與分析摘要",
        "local_search": "搜尋本機文獻庫", "advanced": "進階課程頁面",
        "collection": "XML 匯入與索引設定", "methods": "方法與 EOS 測試工具",
        "settings": "偏好設定",
        "hero_kicker": "本機生醫文獻檢索", "hero_title": "搜尋證據。<br><em>檢視文字。</em>",
        "hero_body": "以 PMID 或 PMCID 取得 PubMed 摘要，檢視精確文字統計，並以 BM25 搜尋本機文獻庫。",
        "ready": "已備妥的資料集", "ready_title": "PubMed 摘要", "ready_body": "結構化 XML、可檢視的排名結果，以及可重現的文字規則。",
        "abstract_xml": "摘要 XML", "indexed_records": "已建立索引的紀錄", "abstract_only": "僅有摘要", "vocabulary": "詞彙表",
        "start_task": "開始使用", "lookup_title": "查詢單篇文章", "lookup_body": "貼上 PMID 或 PMCID，即可取得摘要、關鍵字、字詞統計與規則式 EOS 句子。",
        "lookup_action": "搜尋 PubMed 摘要", "library_title": "搜尋本機文獻庫", "library_body": "在已下載的摘要資料集上執行 AND、OR 或 PHRASE 檢索。",
        "library_action": "開啟本機搜尋", "search_collection": "搜尋你的文獻庫", "keywords": "關鍵字", "search_button": "搜尋本機文獻庫",
        "search_hint": "AND 與 OR 會移除常見英文停用詞；PHRASE 保留停用詞。所有結果皆使用 Porter 詞幹化與 BM25 排名。",
        "how": "Helix 如何運作", "how_caption": "以下系統設計直接對應本專案的檢索、統計、智慧 EOS 與視覺化需求。",
        "technical": "技術規格",
        "lookup_page_title": "PubMed 摘要查詢與分析", "lookup_page_heading": "搜尋摘要，立即分析。",
        "lookup_page_body": "輸入文章 ID 以取得 PubMed 標題與摘要。Helix 隨即計算字元、詞彙、詞頻與 EOS 句子，不會自動下載或儲存文章。",
        "pmid_or_pmcid": "PMID 或 PMCID", "search_abstract": "搜尋摘要", "lookup_hint": "查詢需要網際網路。結果出現後，下載與本機儲存皆為選用功能。",
        "abstract": "摘要", "statistics": "統計資料", "sentences_eos": "句子與 EOS", "characters": "字元數", "words": "詞數", "sentences": "句數", "unique_words": "不重複詞數",
        "save_options": "儲存或下載選項", "download_xml": "下載 PubMed XML", "download_json": "下載摘要 JSON",
        "search_page_title": "本機搜尋與文件分析", "search_heading": "搜尋摘要、關鍵字與正文。", "search_body": "標題不會參與比對。AND 與 OR 移除常見英文停用詞；PHRASE 保留停用詞。所有模式都使用 Porter 詞幹化與 BM25 排名。",
        "query_mode": "查詢模式", "search_terms": "搜尋詞", "no_saved": "目前沒有已儲存的文章。", "did_you_mean": "你是不是想找",
    },
}

ALGORITHM_CARDS = {
    "en": [
        ("01 / INPUT", "XML parser", "Reads PMC JATS or PubMed XML. This workspace uses PubMed abstract XML, so article full text is optional."),
        ("02 / TEXT RULES", "Unified engine", "Uses the primary abstract plus Keywords. It normalizes whitespace and retains biomedical Unicode, decimals, percentages, hyphens, and apostrophes."),
        ("03 / RETRIEVAL", "Index and BM25", "An inverted index supports AND, OR, and PHRASE candidate selection. BM25 ranks every matching document."),
        ("04 / STATISTICS", "Transparent counts", "Shows normalized characters, no-space characters, words, unique words, term frequency, and average sentence length."),
        ("05 / EOS", "Rule-based sentences", "Protects decimal numbers, common abbreviations, initials, and closing quotation marks when detecting sentence boundaries."),
        ("06 / REVIEW", "Inspectable results", "Read structured abstracts, inspect BM25 scores and term counts by scope, and review every detected sentence."),
    ],
    "zh-TW": [
        ("01 / 輸入", "XML 解析器", "讀取 PMC JATS 或 PubMed XML。本工作區使用 PubMed 摘要 XML，因此不需要文章全文。"),
        ("02 / 文字規則", "統一文字引擎", "使用主要摘要與關鍵字，並標準化空白，同時保留生醫 Unicode 字元、小數、百分比、連字號與 apostrophe。"),
        ("03 / 檢索", "倒排索引與 BM25", "倒排索引支援 AND、OR 與 PHRASE 候選文章選取；BM25 會對每篇命中文章進行排名。"),
        ("04 / 統計", "可檢視的統計數值", "顯示標準化字元數、去除空白的字元數、詞數、不重複詞數、詞頻與平均句長。"),
        ("05 / EOS", "規則式句子偵測", "偵測句尾時會保護小數、常見縮寫、人名首字母與結尾引號。"),
        ("06 / 檢視", "可查核的結果", "可閱讀結構化摘要、檢視 BM25 分數與不同範圍的詞頻，並查看每個偵測到的句子。"),
    ],
}


def language() -> str:
    return st.session_state.get("ui_language", "en")


def t(key: str) -> str:
    return TEXT[language()].get(key, TEXT["en"].get(key, key))


def algorithm_cards() -> list[tuple[str, str, str]]:
    return ALGORITHM_CARDS[language()]


def preferences() -> None:
    """Render global language and theme controls in the sidebar."""
    current = language()
    labels = list(LANGUAGES)
    active_label = next(label for label, code in LANGUAGES.items() if code == current)
    widget_key = "_ui_language_widget"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = active_label
    label = st.selectbox(t("language"), labels, key=widget_key)
    selected = LANGUAGES[label]
    if selected != current:
        st.session_state.ui_language = selected
        st.rerun()
    # Widget state is page-scoped in Streamlit. Keep the actual preference in
    # a separate, non-widget key so it survives navigation between pages.
    widget_key = "_ui_dark_mode_widget"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = dark_mode()

    def save_theme() -> None:
        st.session_state.ui_dark_mode = bool(st.session_state[widget_key])

    st.toggle(t("dark_mode"), key=widget_key, on_change=save_theme)


def dark_mode() -> bool:
    return bool(st.session_state.get("ui_dark_mode", False))
