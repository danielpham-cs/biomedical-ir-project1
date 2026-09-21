"""Localized, report-ready explanations for the overview method cards."""

from __future__ import annotations


METHOD_DETAILS = {
    "en": [
        r"""### Purpose
Convert a PubMed or PMC XML file into one consistent document record for the rest of Helix.

### Rules applied
- **PubMed XML:** merge every `AbstractText` in source order and append `KeywordList > Keyword` as a final `Keywords:` line.
- **PMC JATS XML:** select the main `article-meta` abstract, preferring an abstract without `abstract-type` or with `abstract-type="normal"`.
- Article titles, authors, journal information, DOI, dates, references and acknowledgements stay as metadata. They are excluded from search and abstract statistics.

### Output
The parser produces PMID, PMCID when available, title, normalized abstract, Keywords and optional body. PubMed records normally contain no body, which is enough for this abstract-focused project.""",
        r"""### Purpose
Make every statistic and search decision reproducible by applying the same deterministic text rules.

### Rules applied
- Consecutive whitespace, line breaks and tabs become one space.
- Biomedical Unicode stays unchanged, including `α`, `β` and `Δ`.
- `0.65` and `24.9%` remain one token; a numeric range such as `12–15%` becomes two tokens.
- Hyphens and apostrophes remain inside a token only when both neighbouring characters are letters or digits: `SARS-CoV-2`, `5-stage` and `Alzheimer's` are each one word.
- An em dash and ordinary punctuation separate words.

Statistics use the normalized abstract plus Keywords. Search can additionally use available body text, but never the title.""",
        r"""### Candidate selection
An **inverted index** maps each normalized stem to the documents and positions containing it. Helix selects candidates first, then ranks only valid matches.

- **AND:** every valid term must occur.
- **OR:** at least one valid term must occur.
- **PHRASE:** terms must occur consecutively in the same paragraph; stopwords are retained.
- AND and OR remove common English stopwords. All modes lowercase and apply Porter stemming.

### BM25 ranking

$$
\operatorname{BM25}(q,d)=\sum_{t\in q}\operatorname{idf}(t)\frac{f(t,d)(k_1+1)}{f(t,d)+k_1\left(1-b+b\frac{|d|}{\overline{|d|}}\right)}
$$

$f(t,d)$ is term frequency and $|d|$ is document length. Helix uses $k_1=1.2$ and $b=0.75$. The score compares documents within one query; it is not a percentage.""",
        r"""### Counting scope
Every measure uses the whitespace-normalized abstract plus its final Keywords line. The title and bibliographic metadata are excluded.

- **Characters:** every character after whitespace normalization, including spaces.
- **Characters without spaces:** the same text after spaces are removed.
- **Words:** tokens from the word rules. Stopwords are retained and no stemming is used.
- **Unique words:** distinct lowercased tokens.
- **Term frequency:** occurrences of each token.
- **Average sentence length:** word count divided by sentences produced by the EOS rules.

Showing all measures makes the result easy to reproduce and audit.""",
        r"""### Goal
Estimate sentence boundaries with explicit rules instead of relying on an external biomedical NLP model.

### Rule sequence
1. Process each non-empty paragraph separately; any remaining text counts as a sentence.
2. Treat `.`, `!`, `?`, `。`, `！` and `？` as possible endings only before whitespace or paragraph end.
3. Protect decimal points between digits, so `0.65` does not split.
4. Protect known abbreviations: `e.g.`, `i.e.`, `et al.`, `Dr.`, `Fig.` and `vs.`.
5. Protect initials and multi-letter abbreviations, then handle a closing quotation mark after a valid boundary.

The result is a transparent rule-based estimate, not a claim of perfect linguistic segmentation.""",
        r"""### Evidence shown to the user
The review layer makes the retrieval pipeline inspectable rather than presenting only a ranked title list.

- Results show title, PMID and PMCID when available. The title opens the source record on PubMed.
- Matching terms are highlighted in context.
- Each result displays a BM25 score and can reveal term counts in the abstract, body and complete searchable text.
- The article screen displays structured abstract text, all statistics and every sentence detected by the EOS rules.

This lets a reviewer trace the path from XML input to token rules, candidate selection, ranking and visible evidence.""",
    ],
    "zh-TW": [
        """### 目的
將 PubMed 或 PMC XML 轉成統一的文件記錄，供 Helix 後續進行統計和檢索。

### 擷取規則
- **PubMed XML：**依原本順序合併 `AbstractText`，再將 `KeywordList > Keyword` 加成最後的 `Keywords:` 行。
- **PMC JATS XML：**選取 `article-meta` 的主摘要，優先使用無 `abstract-type` 或 `abstract-type=\"normal\"` 的摘要。
- 標題、作者、期刊、DOI、日期、參考文獻只是 metadata，不納入檢索或摘要統計。

### 輸出
結構化紀錄包含 PMID、PMCID、標題、正規化摘要、Keywords 與選用全文。""",
        """### 目的
以同一套確定性文字規則計算與檢索，讓結果可重現。

### 規則
- 連續空白、換行和 Tab 合併為一個空白。
- 保留 biomedical Unicode，包含 `α`、`β` 和 `Δ`。
- `0.65` 與 `24.9%` 各是一個 token；`12–15%` 是兩個 token。
- 前後都是英文字母或數字時，hyphen 與 apostrophe 留在詞內；`SARS-CoV-2`、`5-stage` 和 `Alzheimer's` 各算一個 word。

統計使用摘要加 Keywords；檢索可加上已提供的 body，但標題不參與匹配。""",
        r"""### 候選文件選取
**Inverted index** 將每個正規化後的 stem 對應到出現它的文件和位置。系統先選候選文件，再排名。

- **AND：**所有有效查詢詞必須出現。
- **OR：**至少一個有效詞出現。
- **PHRASE：**詞必須在同一段落連續出現，並保留 stopwords。
- AND 與 OR 移除英文 stopwords；所有 mode 都使用 Porter stemming。

### BM25

$$
\operatorname{BM25}(q,d)=\sum_{t\in q}\operatorname{idf}(t)\frac{f(t,d)(k_1+1)}{f(t,d)+k_1\left(1-b+b\frac{|d|}{\overline{|d|}}\right)}
$$

$f(t,d)$ 是 term frequency，$|d|$ 是文件長度。Helix 使用 $k_1=1.2$、$b=0.75$；BM25 只能比較同一查詢中的排名。""",
        """### 統計範圍
所有統計都是正規化後的摘要加最後的 Keywords 行；標題與書目 metadata 不計入。

- **Characters：**包含空白的字元數。
- **Characters without spaces：**移除空白後的字元數。
- **Words：**依斷詞規則得到的 token；保留 stopwords，不使用 stemming。
- **Unique words：**小寫後的不重複 token 數。
- **Average sentence length：**依 EOS 規則切句後計算。

這些數值可重現，也可回到原文查證。""",
        """### 目標
不依賴外部 biomedical NLP model，而以明確規則估計句子邊界。

### 規則順序
1. 各個非空白段落分開處理；剩餘非空白文字仍算一句。
2. `.`, `!`, `?`, `。`, `！` 與 `？` 在後方為空白或段落結尾時才是可能句尾。
3. 保護數字之間的小數點，因此 `0.65` 不切句。
4. 保護 `e.g.`、`i.e.`、`et al.`、`Dr.`、`Fig.` 和 `vs.` 等縮寫。
5. 保護人名首字母與連續字母縮寫，然後處理句尾的引號。

這是可檢視的 rule-based estimate，不是完美的語言學切句。""",
        """### 使用者可檢視的證據
結果頁面不只顯示排名標題，而是呈現背後的證據。

- 結果顯示標題、PMID 與 PMCID；標題連到 PubMed 原始記錄。
- 查詢詞會在上下文中 highlight。
- 每筆結果顯示 BM25 score，並可查看 term 在 abstract、body 與全部可檢索文字中的次數。
- 摘要頁顯示結構化文字、統計與每個 EOS 句子。

評閱者可以追蹤 XML 輸入、斷詞、候選選取、排名與可見證據之間的關係。""",
    ],
}


def method_details(language: str) -> list[str]:
    """Return localized explanation copy in the same order as the overview cards."""
    return METHOD_DETAILS.get(language, METHOD_DETAILS["en"])


def method_detail_cta(language: str) -> str:
    return (
        "Click to inspect the method →"
        if language == "en"
        else "點擊查看規則與演算法細節 →"
    )
