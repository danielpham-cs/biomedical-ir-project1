# Kế hoạch đáp ứng Project 2 — yêu cầu chi tiết ngày 29/09/2026

> Implementation update (2026-09-30): code, seven demo screens, corpus/model and reports are complete. See [implementation and test results](project2_implementation.md). The design/audit below is historical. Browser color/layout verification remains manual.

Nguồn: `C:/Users/NCKU/Downloads/project-2-details-20260929.pdf` (8 trang), đối chiếu với `Homework-2-26-20260922.pdf` và mã nguồn hiện tại. Đây là kế hoạch triển khai, không phải báo cáo kết quả khoa học.

Kế hoạch màn hình, điều hướng, trạng thái, dark mode và kịch bản demo nằm trong [UI và demo plan](project2_ui_demo_plan.md).

Người dùng đã chọn corpus khoảng **1.000 bài GLP-1** thay cho đề xuất 300 bài ban đầu. Việc soft-delete và tải corpus đã được yêu cầu riêng; các mục phân tích mới dưới đây là công việc còn phải triển khai.

## 1. Kết luận đối chiếu

Đánh giá trước đây dựa trên đề ngắn không còn đủ cho bản chi tiết. Project có sẵn tokenizer, Porter, thống kê tần suất, đồ thị Zipf, Skip-gram, tìm kiếm, sửa chính tả và highlight. Tuy nhiên, bản mới yêu cầu một thí nghiệm có đối chứng và báo cáo nghiên cứu rõ hơn.

| Yêu cầu trong PDF | Hiện trạng khi kiểm tra | Công việc cần làm |
|---|---|---|
| Khoảng 1.000 scientific abstracts tiếng Anh, ID duy nhất; tr. 2 | Corpus thí nghiệm cũ: 100 bài diabetes; index 104 bài; SQLite 2 bản ghi | Chuyển sang 1.000 bài GLP-1, đồng bộ DB/XML/index/manifest |
| Tokenization, case folding, punctuation, giữ/bỏ stopwords, stemming/lemmatization; tr. 3 | Có pipeline nhưng tokenizer đã xử lý punctuation từ đầu | Tách pipeline thí nghiệm A–D; giữ Porter đã có |
| Documents, tokens, unique terms, average tokens/document; tr. 3 | Có số documents/tokens/vocabulary, thiếu bảng đầy đủ theo từng điều kiện | Xuất đủ 4 thống kê cho A/B/C/D |
| CF cho mọi term, Top 50; tr. 3 | `frequency` hiện là CF, UI chỉ hiển thị Top 30 | Đặt tên CF rõ ràng, hiển thị ít nhất 50 và tải toàn bộ bảng |
| DF cho mọi term; tr. 3 | Index có postings/DF cho pipeline tìm kiếm | Tính DF riêng trên đúng corpus và từng điều kiện thí nghiệm |
| Rank-frequency và log-log; tr. 3 | Có một biểu đồ chuyển đổi thang đo | Giữ tương tác, xuất hai hình riêng cho báo cáo |
| Slope, intercept, Zipf exponent, R², RMSE; tr. 3 | Chỉ trả slope và R²; intercept tính nội bộ rồi bỏ | Bổ sung đủ chỉ số và đường hồi quy thực nghiệm |
| Đánh giá vùng đầu/giữa/đuôi; tr. 4 | Chưa có | Fit theo vùng định trước, residual plot, nêu giới hạn của R² |
| So sánh A/B/C/D; tr. 4 | Chỉ trước/sau Porter và checkbox stopwords | Chạy bốn điều kiện trên cùng một snapshot |
| CF–DF cho ít nhất 20 terms; tr. 4–5 | Chưa có bảng thí nghiệm này | Bảng, scatter plot và trả lời ba câu hỏi |
| CF/DF/IDF cho ít nhất 10 terms; tr. 5 | Ranking dùng BM25 IDF, không phải công thức trong đề | Tính IDF = log(N/DF) riêng cho phân tích |
| 300–500 words về ý nghĩa đối với IR; tr. 5–6 | Chưa có bài viết theo yêu cầu | Viết sau khi có số liệu cuối cùng |
| Final Challenge; tr. 7 | Chưa có lập luận dựa trên bộ dữ liệu mới | Giải thích ích lợi của quy luật xấp xỉ bằng kết quả thực nghiệm |
| Executive Summary một trang; RQ1–RQ5; tr. 8 | Chưa có bản đúng yêu cầu mới | Tạo report đầy đủ và summary riêng một trang |
| So sánh hai lĩnh vực; tr. 7 | Chưa có | Tùy chọn, làm sau phần bắt buộc |

Bản chi tiết tập trung vào Zipf và IR, không nhắc lại Word2Vec/spelling/demo. Không có câu nào xác nhận hủy các yêu cầu trong đề trước; kế hoạch giữ các chức năng đó. Không cần bổ sung CBOW vì đã chọn Skip-gram; không cần thêm lemmatization vì đã có Porter.

## 2. Giai đoạn 1 — Corpus GLP-1 và soft-delete (P0)

Phần này thực hiện yêu cầu riêng của người dùng, không suy ra quyền sửa dữ liệu chỉ từ nội dung PDF.

**Đã thực hiện ngày 29/09/2026:** 1.000 bài GLP-1 active và 104 bài cũ soft-deleted; đã kiểm tra DB/XML/index/manifest khớp nhau. Snapshot: `data/archives/before_glp1_20260929T080819389728Z`. Corpus fingerprint: `9bb086392522fe357250861c37fd42e4a705e440126f7cf07439854d4c57e0bc`. Các giai đoạn 2–6 bên dưới vẫn là kế hoạch, chưa phải tính năng đã triển khai. Model và báo cáo diabetes cũ chưa được train/chạy lại.

- Thêm `articles.deleted_at`; mặc định mọi danh sách, corpus và lần rebuild chỉ dùng bài chưa bị ẩn.
- Đăng ký các bài cũ đang nằm trong XML/index vào SQLite trước khi ẩn. Không chỉ ẩn 2 hàng hiện có trong DB.
- Giữ nguyên XML/JSON và model cũ; snapshot manifest/index/database trước khi chuyển corpus. Có lệnh khôi phục.
- Tải PubMed theo GLP-1 và các biến thể tên đầy đủ, kèm `hasabstract[text]` và `english[lang]`.
- Kiểm tra từng bài có PMID duy nhất, abstract thực sự không rỗng và metadata ngôn ngữ tiếng Anh. Không coi keywords thay thế cho abstract.
- Tải theo batch, có retry theo cơ chế NCBI hiện có và cache để tiếp tục sau lỗi. Chỉ chuyển corpus khi đủ số bài hợp lệ.
- Loại các PMID đã được yêu cầu ẩn khỏi bộ tải mới để không vô tình phục hồi chúng.
- Ghi query, thời điểm tải, thứ tự PMID, số lượng và đường dẫn snapshot. Giữ cố định lựa chọn này cho cả A–D.

Mã liên quan: `scripts/download_glp1_corpus.py`, `src/library.py`, `src/index.py`, `src/parse_pmc.py`, `src/homework2.py`, các trang Collection/Search/Homework 2.

**Nghiệm thu:** 1.000 active articles trong SQLite, 1.000 PMID trong manifest và 1.000 bài trong index; mọi bài cũ có thể khôi phục; rebuild không làm bài ẩn xuất hiện lại; liên kết trực tiếp tới bài ẩn không mở nội dung như một bài active.

## 3. Giai đoạn 2 — Thiết kế thí nghiệm A/B/C/D (P0)

PDF liệt kê bốn điều kiện nhưng không quy định chính xác chúng độc lập hay tích lũy. Đề xuất dùng chuỗi tích lũy, ghi rõ trong báo cáo:

| Điều kiện | Xử lý | Stopwords | Porter |
|---|---|---|---|
| A | Tokenization đơn giản theo whitespace + lowercase; giữ punctuation gắn với token | Giữ | Không |
| B | A + chuẩn hóa dấu câu ở biên token; bỏ token chỉ gồm dấu câu | Giữ | Không |
| C | B + loại English stopwords | Bỏ | Không |
| D | C + Porter 1980 | Bỏ | Có |

- Đây là lựa chọn thiết kế đề xuất, không phải chi tiết được PDF quy định.
- Cần chứng minh A và B khác nhau bằng ví dụ như `data, data. data;`. Không được dùng tokenizer hiện tại đã bỏ punctuation cho cả A và B rồi đổi tên điều kiện.
- Công bố cách xử lý `GLP-1`, số thập phân, phần trăm, apostrophe và dấu gạch Unicode. Đề xuất giữ gạch nối nội bộ của định danh y sinh, chỉ bỏ punctuation ở biên; không biến `GLP-1` thành một nghĩa khác một cách âm thầm.
- Dùng **nội dung abstract** cho thí nghiệm chính. Hiện `Document.abstract` gộp thêm keywords; cần lấy abstract riêng từ XML hoặc tách thành trường riêng khi parse. Đây là lựa chọn nhằm bám sát mục tiêu scientific abstracts của đề; nếu khảo sát thêm abstract+keywords thì đặt tên thành thí nghiệm bổ sung.
- Không đổi pipeline/offset hiện có của retrieval chỉ để tạo A–D. Giữ module thí nghiệm riêng để tránh phá highlight và PHRASE.
- Mỗi điều kiện ghi đủ N, total tokens, unique terms, average tokens/document, hapax và Top 50 CF. N vẫn là số documents trong snapshot ngay cả khi một abstract hết token sau lọc.

Đề xuất mã: `src/zipf_experiment.py` mới; tái sử dụng `src/porter.py`; bổ sung abstraction lấy plain abstract trong `src/pubmed.py`/`src/homework2.py`.

**Nghiệm thu:** cùng danh sách PMID và raw-text fingerprint cho A–D; có trace từng bước; cùng input cho kết quả tái lập; test dấu câu và thuật ngữ GLP-1 có kết quả được khai báo trước.

## 4. Giai đoạn 3 — CF, DF và hồi quy Zipf (P0)

Với mỗi điều kiện:

- `TF(t,d)` là số lần term xuất hiện trong document d.
- `CF(t) = sum_d TF(t,d)`.
- `DF(t) = count_d(TF(t,d) > 0)`; một term lặp 20 lần trong một bài chỉ cộng 1 vào DF.
- Sort theo CF giảm dần, dùng thứ tự chữ làm tie-break; rank bắt đầu từ 1.
- Fit `y = a + m*x`, với `x=log10(rank)`, `y=log10(CF)`.
- Xuất slope `m`, intercept `a`, Zipf exponent `k=-m`, R² và RMSE **trên thang log10(CF)**. Không gộp RMSE raw-frequency với RMSE log.
- Phân biệt đường tham chiếu `1/r` hiện có với đường fit `10^a * r^m` từ dữ liệu.
- Xuất `rank_frequency.png`, `loglog_fit.png`, `residuals.png` và toàn bộ bảng CSV/JSON; hình cho báo cáo phải có nhãn trục/chú thích rõ.
- Đề xuất phân vùng rank được công bố trước: head 10% đầu, middle 10–80%, tail 20% cuối. Đây là lựa chọn phân tích, không phải ngưỡng trong PDF. Ghi cả rank đầu/cuối và số term từng vùng; không chọn lại ngưỡng chỉ để làm R² đẹp.
- Fit từng vùng và so sánh residual, slope, R², RMSE. Trường hợp tail toàn CF=1 hoặc không đủ biến thiên, hiển thị R² không xác định và giải thích, không báo là fit hoàn hảo.
- Kết luận có thể là "gần Zipf trong một khoảng rank". R² cao của log-log OLS không đủ chứng minh một power law chính xác; ties và phần đuôi rời rạc cũng cần được bàn luận. Không bắt buộc triển khai kiểm định power-law nâng cao để đáp ứng PDF.

**Nghiệm thu:** bộ dữ liệu nhân tạo `CF ~ C/r^k` khôi phục k trong sai số hợp lý; kiểm tra `sum(CF)=total_tokens`, `1<=DF<=N`, `DF<=CF`; xử lý từ vựng ít hoặc constant frequency an toàn.

## 5. Giai đoạn 4 — CF/DF/IDF và giao diện phân tích (P1)

- Dùng điều kiện B làm mặc định cho bảng minh họa CF/DF/IDF để còn thấy stopwords; cho phép chuyển điều kiện nhưng không trộn CF của B với DF của D.
- Chọn ít nhất 20 term: kết hợp term phổ biến, term GLP-1/y sinh và term có tần suất tập trung trong ít documents. Ghi cách chọn và chỉ sử dụng term thực sự tồn tại.
- Hiển thị bảng `term, CF, DF, DF/N, CF/DF` và scatter CF–DF.
- Tính `IDF=ln(N/DF)` cho ít nhất 10 term (có thể dùng luôn cả 20 term). PDF không quy định log base; ghi rõ dùng log tự nhiên.
- Ví dụ kiểm tra: term có DF=N phải có IDF=0; giảm DF trong cùng N phải làm IDF tăng.
- Giữ BM25 cho retrieval. `src/index.py` đang dùng IDF BM25; không tái sử dụng số đó dưới nhãn IDF của đề. `tfidf_search()` hiện chỉ là wrapper gọi BM25, không được mô tả như triển khai TF-IDF thực sự.
- Có thể thêm một ví dụ `TF*IDF` theo document để giải thích TF-IDF; đề không bắt buộc thay ranking BM25 bằng TF-IDF.
- Mở rộng Homework 2 thành các khu vực: Corpus, Preprocessing A–D, Zipf & regression, CF/DF/IDF, Word2Vec, Report & demo.
- Corpus vẫn có tìm theo title/PMID và liên kết đọc nội bộ. Kiểm tra light/dark bằng trình duyệt thật khi có kết nối; AppTest không chứng minh màu sắc hiển thị đúng.

**Nghiệm thu:** đủ Top 50; ít nhất 20 CF/DF và 10 IDF; mọi bảng có condition/corpus scope; CSV khớp số liệu UI; tìm kiếm/title links còn hoạt động.

## 6. Giai đoạn 5 — Chạy lại Word2Vec và demo trên corpus mới (P1)

- Giữ Skip-gram + negative sampling hiện có, train lại trên corpus GLP-1 và lưu fingerprint mới.
- Chọn trước preprocessing của Word2Vec và ghi rõ. Không bắt buộc train bốn model để so sánh A–D; yêu cầu bốn điều kiện đang nói về phân bố từ.
- Model diabetes cũ được lưu làm lịch sử và không dùng để diễn giải kết quả GLP-1.
- Chọn từ demo từ vocabulary mới; với minimum count, các từ OOV phải có thông báo rõ.
- Demo sửa chính tả, partial/fuzzy search, highlight và vị trí từ trên bài GLP-1; không dùng nguyên kết quả cũ về diabetes.

**Nghiệm thu:** model, corpus, bảng tần suất và UI ghi đúng topic và sample size; model save/load hoạt động; truy vấn demo có kết quả xác thực.

## 7. Giai đoạn 6 — Báo cáo và Executive Summary (P1)

Tạo kết quả từ số liệu chạy cuối cùng, không điền trước kết luận "tuân theo Zipf".

Final report cần có dataset/query/snapshot, A–D, statistics, Top 50, hai dạng plot, bảng hồi quy, phân tích head/middle/tail, CF/DF/IDF và limitations. Trả lời rõ:

1. **RQ1:** phân bố có gần Zipf không; dựa trên curve, fit và residual.
2. **RQ2:** exponent thực nghiệm là bao nhiêu, trong condition và vùng rank nào.
3. **RQ3:** preprocessing đổi vocabulary, high-frequency terms, exponent và shape ra sao.
4. **RQ4:** CF và DF khác nhau thế nào; high CF/low DF nói lên điều gì; DF phản ánh độ phủ documents.
5. **RQ5:** ý nghĩa với IR; thêm đoạn **300–500 từ tiếng Anh** về stopwords, dictionary/postings/storage/compression và TF-IDF.

Trả lời riêng Final Challenge: một quy luật kinh nghiệm xấp xỉ vẫn giúp đưa ra quyết định IR như thế nào, dùng chính số liệu trong corpus để lập luận.

Xuất **Executive Summary riêng đúng một trang**, trả lời điều học được từ phân tích 1.000 abstracts mà đọc sách không cho biết; gồm mức gần Zipf, exponent, tác động preprocessing và ý nghĩa IR. Summary không thay thế final report.

Đầu ra đề xuất: `output/project2_glp1/report.md` và bản PDF/DOCX khi chốt số liệu; `executive_summary.pdf` một trang; `figures/`; `tables/`; `run_metadata.json`; manifest và hướng dẫn tái lập. Model để riêng, có checksum/fingerprint liên kết.

## 8. Thứ tự triển khai và điểm chốt

1. Hoàn tất corpus + soft-delete, kiểm tra snapshot/restore.
2. Chốt phạm vi abstract-only và quy tắc A–D; thêm tests trước khi chạy toàn bộ.
3. Hoàn tất CF/DF/regression/IDF và chạy cả bốn điều kiện bằng một CLI.
4. Đưa kết quả lên UI, train model GLP-1, kiểm tra demo và theme.
5. Viết report từ số liệu đã đóng băng, render/kiểm tra summary một trang.
6. Chạy lại từ snapshot và kiểm tra toàn bộ đầu ra; chỉ sau đó cân nhắc so sánh hai lĩnh vực.

So sánh 500 medical với 500 computer-science abstracts là **optional** trong PDF, không chặn hoàn thành bài chính. Không ưu tiên thêm CBOW, t-SNE hoặc đổi hệ thống tìm kiếm khi các mục bắt buộc còn thiếu.

Kế hoạch được coi là hoàn tất khi mọi hàng bắt buộc trong bảng đối chiếu có một đầu ra kiểm chứng được và RQ1–RQ5 đều được trả lời bằng số liệu của cùng corpus GLP-1.
