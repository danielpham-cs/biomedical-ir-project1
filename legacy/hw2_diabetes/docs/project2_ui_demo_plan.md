# Kế hoạch UI và demo Project 2

> Implementation update (2026-09-30): code, seven demo screens, corpus/model and reports are complete. See [implementation and test results](project2_implementation.md). The design/audit below is historical. Browser color/layout verification remains manual.

Tài liệu này bổ sung `project2_detailed_plan.md`. Đây là thiết kế đề xuất; chưa phải toàn bộ giao diện đã được triển khai.

Mục tiêu: giảng viên có thể đi từ corpus đến kết luận nghiên cứu và kiểm tra các chức năng IR trong một luồng 10–12 phút. Mỗi màn hình trả lời một câu hỏi, có dữ liệu chứng minh và một hành động demo rõ ràng.

## 1. Cấu trúc điều hướng

Giữ Helix, màu xanh lá/teal và trang đọc bài nội bộ. Trong sidebar, đặt nhóm **Project 2** theo thứ tự:

1. Overview & corpus
2. Preprocessing
3. Zipf analysis
4. CF / DF / IDF
5. Word2Vec
6. Search & spelling
7. Report & demo

Các công cụ tải thêm PMID/XML và quản trị bài ẩn nằm trong **Manage collection**. Không để những công cụ quản trị làm gián đoạn luồng demo. So sánh hai domains là mục tùy chọn cuối cùng, chỉ xuất hiện khi đã có dữ liệu.

Không đặt mọi module nặng trong một trang `st.tabs` rồi tính lại tất cả khi người dùng tìm corpus. Với Streamlit hiện tại, nên tách trang hoặc chỉ render màn được chọn; cache theo corpus fingerprint + condition + model settings.

## 2. Thành phần dùng chung

- Thanh ngữ cảnh: **GLP-1 · 1.000 abstracts · corpus snapshot · condition đang chọn**. Chỉ hiện condition ở màn có liên quan.
- Mặc định cùng một corpus cho toàn bộ thí nghiệm. Các số liệu/model cũ phải ghi rõ tên corpus, không gắn nhãn GLP-1 cho báo cáo diabetes.
- Khu vực đầu màn: tiêu đề, một câu hỏi cần trả lời và 2–4 chỉ số quan trọng. Công thức/chi tiết thuật toán đặt trong expander, nhưng tên condition và phạm vi thống kê phải luôn nhìn thấy.
- Mỗi biểu đồ có tiêu đề, trục/đơn vị, legend, tooltip và nút tải. Bảng có sort/search khi nhiều dòng và xuất CSV.
- Trạng thái kết quả: `Not run`, `Running`, `Ready`, `Settings changed`, `Failed`. Hiển thị bằng chữ, không chỉ dùng màu.
- Chỉ tính lại/train khi có thao tác rõ ràng. Đổi số epochs/window không tự train; kết quả cũ được ghi là thuộc cấu hình trước đến khi chạy xong.
- Hiển thị thời gian thực tế từ lần chạy, không điền các con số demo giả.
- Điều hướng Back/Next giữ corpus, search filter, condition và theme; mở paper phải trở về đúng nơi người dùng đang xem.

## 3. Đặc tả từng màn

| Màn | Câu hỏi người xem cần trả lời | Nội dung chính | Thao tác demo | Bằng chứng xuất được |
|---|---|---|---|---|
| Overview & corpus | Dữ liệu đến từ đâu, có đúng 1.000 abstracts GLP-1 không? | Query, source PubMed, ngôn ngữ, số active/archived, missing/duplicate checks, bảng PMID/title/year, trạng thái các thí nghiệm | Search theo title hoặc PMID; mở bài đọc nội bộ; trở lại bảng | Manifest, thống kê corpus |
| Preprocessing | A/B/C/D khác nhau như thế nào? | Abstract gốc; trace token từng bước; bảng settings; N/tokens/vocabulary/average tokens cho bốn conditions | Chọn một paper; chuyển A→B→C→D; chỉ ra punctuation, stopword, từ được Porter gộp | Bảng so sánh, token trace |
| Zipf analysis | Corpus gần Zipf ở đâu, exponent bao nhiêu? | Rank-frequency và log-log; dữ liệu thực, đường 1/r và fitted line; bảng slope/intercept/k/R²/RMSE; head/middle/tail; residual | Chuyển condition; bật overlay A–D; chọn vùng rank; hover term; giải thích vùng lệch | Hình PNG/PDF, regression CSV, Top 50 và full-frequency CSV |
| CF / DF / IDF | Lặp nhiều và xuất hiện rộng khác nhau thế nào? | Ít nhất 20 terms; CF/DF/DF÷N/CF÷DF/IDF; CF–DF scatter; công thức IDF; ví dụ TF document-level | Chọn term high-CF/low-DF; mở danh sách paper chứa term; đối chiếu một stopword với một term chuyên ngành | CF–DF–IDF CSV, danh sách documents của term |
| Word2Vec | Skip-gram học được quan hệ ngữ cảnh gì? | Model/corpus/config; loss curve; neighbors cosine; PCA; giới hạn của small-domain corpus | Nhập một term trong vocabulary; xem neighbors; click điểm PCA; demo train nhỏ nếu đủ thời gian | Model NPZ, config/history, neighbors và PCA CSV |
| Search & spelling | Người dùng tìm được paper và thấy từ khớp như thế nào? | Search box; AND/OR/PHRASE; Standard/Partial/Fuzzy; gợi ý sửa lỗi; kết quả BM25; reader/highlight/locations | Gõ lỗi; xem và áp dụng gợi ý; mở bài; next/previous occurrence; quay lại kết quả | Match locations CSV, searchable text |
| Report & demo | Hệ thống đã chứng minh điều gì và nộp được gì? | RQ1–RQ5; kết luận theo số liệu; limitations; bài viết 300–500 từ; executive summary; checklist đầu ra | Mở kết quả của một RQ; tải report/summary; kiểm tra manifest/model cùng corpus | Report, summary một trang, bundle tables/figures/metadata |

### Overview & corpus

- Phần đầu chỉ cần nguồn, topic, active documents và trạng thái sẵn sàng; số bài ẩn đặt trong Manage collection hoặc expander.
- Bảng có các cột PMID, Title, Year; title màu chữ thông thường, hover/focus chuyển teal; liên kết vào reader trong Helix.
- Dùng phân trang 25/50 dòng nếu corpus 1.000 bài gây trang quá dài. Tìm kiếm áp dụng trên toàn corpus trước khi phân trang; luôn hiện `Showing X of 1,000`.
- Có empty state cho query không khớp; xóa query trả lại toàn bộ danh sách.
- Paper ẩn không xuất hiện trong bảng/search/term drill-down; link trực tiếp có thông báo archived và đường quay lại.
- Bảng quản lý Archived hiển thị riêng và hỗ trợ Restore. Restore phải nói rõ nó thay đổi thư viện active; không âm thầm làm thay đổi snapshot thí nghiệm đã đóng băng.

### Preprocessing

- Ưu tiên hai vùng: bên trái abstract gốc; bên phải các token sau bước đang chọn. Có thể xếp dọc khi cửa sổ hẹp.
- A/B/C/D có mô tả bằng chữ, không chỉ ký hiệu. Bảng định nghĩa conditions luôn có thể mở ngay trên màn.
- Phân biệt `removed`, `normalized`, `stemmed` bằng nhãn; không chỉ dùng màu.
- Chọn ví dụ thực sự có punctuation/stopwords và nhóm Porter khác nhau trong GLP-1 corpus. Không dùng ví dụ giả làm số liệu corpus.
- Chế độ `Try your own text` là sandbox tách biệt, có nhãn không thuộc thí nghiệm; nó không cập nhật N/CF/DF của corpus.

### Zipf analysis

- Mặc định một condition, hai plot cạnh nhau trên màn rộng; xếp dọc trên màn hẹp. So sánh A–D bằng toggle riêng để tránh tám đường chồng nhau từ đầu.
- Chỉ số k được đặt cạnh slope; ghi `k = -slope`. RMSE ghi rõ `log10(CF)`.
- Top 50 nằm dưới plot; toàn bộ vocabulary có tải CSV và bộ lọc. Frequency spectrum nằm trong phần bổ sung, không thay thế rank-frequency.
- High/middle/low-frequency views ghi khoảng rank thực, số points, condition và quy tắc chia vùng.
- Khi R² không xác định do dữ liệu hằng/ít điểm, hiện `N/A` với lý do. Không làm tròn thành 1.00 gây hiểu nhầm.
- Khối `Interpretation` trả lời: gần tuyến tính không, vùng nào tốt nhất, R² có đủ không. Nội dung được viết từ lần chạy cuối, không tự động khẳng định "Zipf holds" chỉ vì vượt một ngưỡng.

### CF / DF / IDF

- Mặc định condition B để còn stopwords; luôn hiện N và log base.
- Chọn một term sẽ hiển thị: CF toàn corpus, DF documents, histogram TF theo document hoặc bảng TF gọn, và link đọc paper có highlight term đó.
- Ví dụ giải thích cần chỉ ra term lặp nhiều trong ít papers và term phân bố trên nhiều papers. Chọn dựa trên kết quả thật, không giả định trước rằng term nào chắc chắn thỏa.
- Bảng IDF dùng công thức đề `ln(N/DF)`. BM25 score xuất hiện ở Search có nhãn riêng; không gọi BM25 IDF là cùng chỉ số.

### Word2Vec

- Màn mở đầu ưu tiên model đã train và kiểm chứng với corpus GLP-1; không bắt đầu bằng một form dài.
- `Training settings` nằm trong expander; có nút Train, progress epoch, loss và lỗi dễ hiểu.
- Lựa chọn `small training demo` phải ghi rõ sample size riêng. Model demo không thay thế model chính 1.000 abstracts nếu người dùng chưa chọn sử dụng nó.
- OOV: giải thích và gợi ý từ có thật trong vocabulary; không trả vector/nearest words giả.
- PCA có retained variance; giải thích khoảng cách 2D không đồng nhất với cosine trong không gian gốc.

### Search & reader

- Luồng chính: query → kết quả → đọc bài → vị trí từ khớp → Back giữ nguyên kết quả.
- Gợi ý sửa chính tả cần một thao tác áp dụng; cho phép xem cả kết quả query gốc. Fuzzy retrieval được mô tả riêng với correction.
- Khi mở bài từ Corpus, tab Abstract được chọn đầu tiên. Khi mở từ Search, ưu tiên Keyword locations hoặc ngữ cảnh khớp.
- Khi đến từ bảng CF/DF, reader nhận đúng term/condition hoặc nói rõ đang highlight surface forms tương ứng; không dùng query cũ còn trong session.
- Query mẫu được chọn sau khi chạy trên GLP-1 corpus; ví dụ ứng viên `glp-1 receptor`, `semaglutide`, lỗi `semaglutde`. Chỉ đưa lên nút demo nếu đã kiểm tra có kết quả/gợi ý hợp lý.

### Report & demo

- Checklist phải phân biệt "có code", "đã chạy trên corpus hiện tại", "đã có report"; không đánh dấu hoàn thành chỉ vì file báo cáo diabetes cũ tồn tại.
- Mỗi RQ dẫn tới plot/table dùng làm bằng chứng.
- Có `Demo mode` để dùng kết quả đã tính sẵn và checklist trình bày. Chế độ này không chỉnh số liệu hoặc giả lập thành công.
- Có `Download submission bundle`; chỉ chứa đầu ra của cùng một snapshot/condition specification, kèm README tái lập. Executive Summary là file riêng một trang.

## 4. Kịch bản demo 10–12 phút

| Thời gian dự kiến | Màn và hành động | Ý cần nói |
|---|---|---|
| 0:00–1:00 | Overview: query, 1.000 bài, mở một PMID | Nguồn, phạm vi, cách chọn và đóng băng corpus |
| 1:00–2:30 | Preprocessing: một abstract, A→D, bảng tổng hợp | Thay đổi token/vocabulary do từng bước |
| 2:30–5:00 | Zipf: hai plots, fitted line, k/R²/RMSE, ba vùng rank | Kết luận có bằng chứng; giới hạn của R² |
| 5:00–6:30 | CF/DF/IDF: chọn hai terms và mở documents | Lặp lại so với độ phủ; vì sao IDF khác nhau |
| 6:30–8:00 | Word2Vec: model, loss, neighbors/PCA | Center→context, cosine, giới hạn diễn giải |
| 8:00–10:00 | Search: typo/correction, Partial/Fuzzy, reader/highlight | Tìm kiếm thực tế và vị trí từ khớp |
| 10:00–12:00 | Report: RQ1–RQ5 và summary | Những điều học được từ dữ liệu, đầu ra và tái lập |

Đây là phân bổ thời gian đề xuất, không phải thời lượng PDF quy định. Chuẩn bị một tuyến demo ngắn hơn nếu giảng viên giới hạn thời gian; vẫn giữ Zipf/A–D/CF–DF–IDF là trọng tâm của bản chi tiết.

## 5. Trạng thái lỗi và tính nhất quán

| Tình huống | UI cần làm |
|---|---|
| Tải corpus chưa đủ / NCBI lỗi | Hiện số tải hợp lệ và nút tiếp tục; không chuyển thư viện active sang corpus dở dang |
| Chưa chạy A–D | Giữ màn corpus dùng được; các màn phân tích có nút chạy và mô tả đầu ra |
| Đổi corpus/preprocessing | Gắn nhãn kết quả cũ; không đặt số liệu/model cũ dưới topic mới |
| Chưa train model GLP-1 | Hiện trạng thái Not trained; không tự lấy model diabetes |
| Training thất bại | Giữ model hợp lệ trước đó, hiện lỗi và cấu hình thất bại |
| Không có search hit / corpus match | Empty state và cách thay query; không để bảng trắng không giải thích |
| Restore bài cũ | Cập nhật active library và cache/index; snapshot nghiên cứu chỉ đổi qua thao tác chọn/tạo corpus rõ ràng |
| Reload / back / mở bài | Giữ context hợp lý và có đường quay lại đúng màn nguồn |

## 6. Light/dark và kiểm thử giao diện

- Dùng một bộ theme tokens cho text, muted text, surface, elevated surface, border, accent; cả native Streamlit controls và HTML tables phải nhận cùng palette.
- Kiểm tra riêng input trống, có chữ, placeholder, focus, disabled, clear button, number-input step controls, select dropdown, table header/body/hover/visited links, scrollbars, charts/tooltips, reader và dialogs.
- Title thường là màu text; hover/focus mới teal; trạng thái đã click vẫn giữ màu text.
- Contrast mục tiêu ít nhất 4.5:1 cho chữ thông thường; focus phải nhìn thấy khi dùng bàn phím. Không coi "đổi color trong CSS" là đủ nếu foreground/background thực tế chưa được kiểm tra.
- Kiểm tra ở desktop 1440px, laptop 1280px và cửa sổ 768px; không tràn chữ bảng, không cắt nhãn biểu đồ, cột title được wrap hoặc có cách đọc đầy đủ.
- AppTest xác minh tương tác/state; kiểm tra bằng trình duyệt và screenshot xác minh màu/layout. Nếu công cụ browser không khả dụng, ghi rõ phần visual QA còn chờ, không báo đã nghiệm thu màu sắc.

## 7. Thứ tự triển khai UI

1. **UI-1:** Chốt sidebar, thanh ngữ cảnh, theme tokens và corpus table/reader/back behavior.
2. **UI-2:** Preprocessing A–D + Zipf metrics/plots, sau khi backend thống kê hoàn tất.
3. **UI-3:** CF/DF/IDF + term-to-document drill-down; tái sử dụng reader.
4. **UI-4:** Model readiness, Word2Vec workflow, search/correction demo queries trên dữ liệu thật.
5. **UI-5:** Report/RQ links, submission bundle, demo mode và kiểm thử toàn tuyến.

**Definition of done:** đi hết tuyến demo bằng dữ liệu đã đóng băng mà không cần sửa code, gọi mạng ngoài hoặc train lại model lớn; tất cả figures/tables có nguồn và tải được; state/context đúng khi qua lại; chức năng không lỗi và visual QA đạt ở cả light/dark.
