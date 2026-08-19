# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Nguyễn Đức Sơn — MSSV: 2A202601485
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 19/08/2026

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)
> Nêu ít nhất 1 tình huống cụ thể mà cơ chế Coreference Resolution phân giải sai hoặc gặp khó khăn. Hậu quả đối với Knowledge Graph là gì?

*Trả lời:*
- **Ví dụ đúng (đối chứng):** chunk chứa câu "Warren Buffett offered some insights into his most recent buys and sells" được resolve thành "...insights into Warren Buffett's most recent buys and sells" — antecedent xuất hiện rõ trong cùng chunk nên hệ thống resolve an toàn, đúng quy tắc conservative.
- **Ví dụ khó/giữ nguyên (chunk_id=`00bf56b1ff85b6f5aaa5::c0000`):** câu bắt đầu bằng "We are announcing 2019's annual Chief FOIA Officers' Council meeting..." — đại từ "We" không có antecedent rõ ràng trong cùng chunk (văn bản trích từ thông cáo báo chí, chủ ngữ ẩn ở phần bị cắt do chunking). Hệ thống đúng theo thiết kế: **không suy diễn**, giữ nguyên "We" và ghi vào `unresolved_mentions`.
- **Hậu quả nếu ép resolve sai:** nếu hệ thống "liều" gán "We" = tên một công ty cụ thể (hallucination), NER+RE bước sau sẽ tạo ra **False Edge** — ví dụ gán nhầm một quan hệ `PARTNERED_WITH` cho công ty sai — làm hỏng độ tin cậy toàn bộ nhánh đồ thị liên quan. Đây là lý do rule "chỉ resolve khi antecedent rõ ràng trong cùng chunk" quan trọng hơn việc tối đa hóa số lượng resolve.

---

### 2. Entity Resolution Threshold & Lexical Guard
> Ngưỡng cosine similarity bạn chọn? Trích 1 cặp thực thể similarity > 0.85 nhưng bị Lexical Guard chặn.

*Trả lời:*
- **Ngưỡng cosine similarity:** `threshold = 0.90` (vector ANN qua FAISS `IndexFlatIP`, embedding `all-MiniLM-L6-v2`)
- **Ngưỡng Lexical Guard:** `SequenceMatcher ratio >= 0.72` sau khi chuẩn hóa (bỏ suffix Inc/Corp/Ltd...)
- **Cặp thực thể bị Guard chặn (dữ liệu thật từ `entity_resolution_audit_df`):** `"The company"` vs `"company"` — cosine similarity = **0.9049** (>0.85, đủ điều kiện vector match) nhưng bị `REJECT_GUARD`.
- **Lý do chặn:** sau chuẩn hóa, `"the company"` và `"company"` có SequenceMatcher ratio = 0.70 (<0.72) do khác độ dài chuỗi đáng kể dù ngữ nghĩa gần nhau qua embedding. Đây thực chất là **artifact của coreference** ("the company" là cụm chỉ định chung, không phải tên riêng) — Lexical Guard đóng vai trò lưới an toàn thứ hai, ngăn hệ thống gộp nhầm các cụm từ generic có similarity cao về mặt vector nhưng không phải cùng một thực thể có tên riêng.
- **Lưu ý về scale:** với tập 132 triples (do giới hạn thời gian trích xuất), `entity_resolution_audit_df` chỉ có 1 dòng — thấp hơn ngưỡng khuyến nghị 10+ dòng của rubric. Ở quy mô lớn hơn (nhiều alias công ty hơn: "MSFT"/"Microsoft Corp"/"Microsoft Corporation"), số dòng audit sẽ tăng tương ứng.

---

### 3. Đồ thị & Super-node Mitigation
> Top 3 thực thể bậc cao nhất? Ưu/nhược điểm của việc lấy 50 cạnh mới nhất tại Super-node?

*Trả lời:*
- **Top 3 (dữ liệu thật sau ingest 170 nodes / 108 edges):**

| Hạng | Tên thực thể | Loại | Bậc (Degree) |
|------|--------------|------|---------------|
| 1 | Microsoft | Company | 5 |
| 1 | ServiceNow | Company | 5 |
| 3 | NVIDIA | Company | 3 |

- **Nhận xét quan trọng:** ở quy mô dữ liệu đã trích xuất trong buổi lab (132 triples, do giới hạn thời gian/API quota), **không có node nào vượt ngưỡng `SUPER_NODE_DEGREE=100`**, nên cơ chế Super-node Mitigation (cắt còn 50 cạnh mới nhất) **chưa từng được kích hoạt thực tế** trong lần chạy này — chỉ được kiểm chứng qua code path, không qua dữ liệu thật.
- **Ưu điểm của Temporal Mitigation (khi kích hoạt ở scale lớn):** với các super-node thật (vd: "Google", "Microsoft" ở quy mô hàng chục nghìn bài báo có thể có degree hàng nghìn), giữ lại 50 cạnh mới nhất giúp: (1) tránh bùng nổ context vượt `MAX_GRAPH_CONTEXT_CHARS`, (2) ưu tiên thông tin còn tính thời sự cho các câu hỏi dạng "hiện trạng gần đây".
- **Rủi ro:** câu hỏi dạng lịch sử ("Microsoft mua lại công ty nào năm 2010?") có thể bị cắt mất cạnh liên quan nếu cạnh đó cũ hơn 50 cạnh gần nhất — đánh đổi giữa độ mới (recency) và độ đầy đủ (completeness/recall) của câu trả lời.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge, 50 câu golden, model đánh giá `allam-2-7b` qua Groq):

| Nhóm câu hỏi | Metric | Flat RAG | GraphRAG | Nhận xét |
|---|---|---|---|---|
| cross-doc | Comprehensiveness | 3.09 | 1.96 | Flat RAG tốt hơn |
| cross-doc | Faithfulness | 2.91 | 1.59 | Flat RAG tốt hơn |
| cross-doc | Multi-hop reasoning | 3.09 | 1.59 | Flat RAG tốt hơn |
| multi-hop | Comprehensiveness | 2.74 | 1.87 | Flat RAG tốt hơn |
| multi-hop | Faithfulness | 2.48 | 1.74 | Flat RAG tốt hơn |
| multi-hop | Multi-hop reasoning | 2.87 | 1.78 | Flat RAG tốt hơn |
| factoid | Comprehensiveness | 1.80 | 2.20 | Gần nhau |
| Tất cả | Latency (s) trung bình | ~6.6 | ~8.2 | GraphRAG chậm hơn |
| Tất cả | Token usage trung bình | ~833 | ~1086 | GraphRAG tốn token hơn |

*(Bảng đầy đủ: `outputs/graphrag_vs_flatrag_summary.csv`; chi tiết từng câu: `outputs/graphrag_eval_results.csv`)*

**Kết quả thực nghiệm cho thấy Flat RAG thắng GraphRAG ở hầu hết tiêu chí** — trái ngược kỳ vọng lý thuyết. Nguyên nhân chính: KG chỉ có 108 edges (do `EXTRACTION_MAX_CHUNKS` giới hạn thời gian lab), quá thưa để graph traversal tìm đủ context so với Flat RAG có index phủ toàn bộ 2,105 chunks.

#### Phân tích 2 Ca lỗi Điển hình:

1. **Ca lỗi Flat RAG thất bại (GraphRAG thành công) — G5000-02:**
   - *Câu hỏi:* "Did the first two Aeris/Ericsson reports describe a completed acquisition or a planned transfer, and what later evidence changes the event state?"
   - *Flat RAG trả lời (SAI):* "The first two reports describe a **completed acquisition**..." — mâu thuẫn với reference answer.
   - *GraphRAG trả lời (ĐÚNG, khớp reference):* "The first two... reports describe a **planned transaction**... The event state changes with the third report..." — đúng theo `reference_answer`.
   - *Tại sao GraphRAG đúng còn Flat RAG sai:* GraphRAG có edge provenance kèm `published_date` cho phép phân biệt trình tự thời gian (planned → completed) rõ ràng hơn, trong khi Flat RAG lấy top-k chunk theo similarity thuần túy, dễ trộn lẫn 2 bài báo cùng chủ đề nhưng khác trạng thái sự kiện.
   - **Phát hiện quan trọng đi kèm:** mặc dù GraphRAG trả lời đúng, **LLM-as-a-Judge (`allam-2-7b`) lại chấm GraphRAG 1/1/1 và Flat RAG 4/4/4** — ngược hoàn toàn với chất lượng thực tế. Đây là bằng chứng cho thấy **model Judge nhỏ (7B) kém tin cậy**, một hạn chế cần nêu rõ khi dùng Groq free-tier với quota giới hạn buộc phải hạ cấp model.

2. **Ca lỗi GraphRAG thất bại — nhóm cross-doc nói chung:**
   - *Nguyên nhân:* GraphRAG thua toàn diện ở nhóm `cross-doc` (so sánh nhiều bài báo). Do `EXTRACTION_MAX_CHUNKS` giới hạn, nhiều chunk liên quan đến golden questions (21/51 chunk mục tiêu) không có relation nào được trích xuất — không phải vì extraction lỗi, mà vì blurb quá ngắn (trung bình ~204 ký tự) không đủ ngữ cảnh cho LLM tìm thấy quan hệ Company/Person/Technology rõ ràng.
   - *Đề xuất khắc phục:* (1) tăng `EXTRACTION_MAX_CHUNKS` để phủ nhiều bài hơn, (2) dùng trường text đầy đủ thay vì `description` (tóm tắt ngắn) nếu dataset có, (3) nới `ALLOWED_RELATIONS` để bắt thêm quan hệ tường thuật (`MENTIONED_WITH`, `REPORTED_ON`) thay vì chỉ 8 quan hệ nghiệp vụ chặt.

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent

*Trả lời:*
- **Đánh đổi Quality vs Cost vs Latency:** GraphRAG tốn thêm ~30% latency và ~30% token so với Flat RAG (do phải gọi thêm seed-extraction LLM + graph traversal + vẫn kèm vector context). Ở dữ liệu thưa như lần chạy này, chi phí thêm không đổi lại chất lượng tương xứng — nhưng về lý thuyết, khi KG đủ dày đặc, GraphRAG nên vượt trội ở multi-hop/cross-doc vì có thể nối thông tin qua các cạnh tường minh thay vì chỉ dựa vào similarity ngữ nghĩa.
- **Quyết định kỹ thuật đáng chú ý trong quá trình làm:** ban đầu dùng model `llama-3.3-70b-versatile` theo mặc định bài lab nhưng model đã bị Groq deprecate (lỗi 404). Phải đổi sang `openai/gpt-oss-120b`, và khi model này liên tục hết quota TPD (200k token/ngày) giữa chừng pipeline, phải luân phiên dùng thêm `openai/gpt-oss-20b`, `qwen/qwen3.6-27b`, `allam-2-7b` cho từng bước khác nhau — một quyết định thực dụng để hoàn thành pipeline trong thời gian giới hạn, nhưng đánh đổi bằng việc **không nhất quán một model duy nhất** xuyên suốt (ảnh hưởng đến khả năng so sánh công bằng giữa các bước).
- **Giải pháp khi scale lên 350MB (~100,000 bài báo):**
  - *Bottleneck đầu tiên:* chi phí NER+RE extraction qua LLM — ở scale nhỏ (439 chunks) đã tốn hàng giờ và nhiều lần chạm rate-limit; ở 100,000 bài sẽ cần hàng trăm nghìn lượt gọi LLM.
  - *Giải pháp:* (1) batch/async extraction với worker queue và nhiều API key luân phiên, (2) dùng model rẻ hơn cho pass đầu (candidate extraction) và model mạnh hơn chỉ để verify/dedupe, (3) HNSW/IVF index thay `IndexFlatIP` cho Entity Resolution khi số lượng entity lớn (tránh O(N²)), (4) partition dữ liệu theo thời gian/domain để xử lý song song và tránh 1 lần OOM.

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code
| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|------------------|------------------------|-----------------------------|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()`, `run_coref()` | Hoạt động đúng thiết kế: chỉ resolve khi antecedent rõ trong cùng chunk (ví dụ "his" → "Warren Buffett's"), giữ nguyên khi mơ hồ (ví dụ "We"). |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Lọc hiệu quả: loại bỏ mọi relation/type ngoài whitelist ngay trong `run_extraction()`, tránh graph bị nhiễu bởi các loại quan hệ tùy tiện do LLM tự bịa. |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | `UNWIND` batch 1000 giúp insert 170 nodes/108 edges gần như tức thời; sanity check xác nhận 0 edge thiếu provenance. |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, `UF` | Bắt đúng 1 ca REJECT_GUARD ("The company" vs "company") — chứng minh guard hoạt động, dù dataset nhỏ nên audit table còn mỏng. |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()`, `SUPER_NODE_DEGREE` | Code path đúng nhưng chưa được kích hoạt thực tế vì graph quá thưa (degree tối đa 5 << ngưỡng 100). |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()` | Chạy đủ 50/50 câu, nhưng phát hiện model Judge nhỏ (`allam-2-7b`, dùng do quota) cho điểm không nhất quán với chất lượng thực tế câu trả lời (case G5000-02). |

---

### 2. Quá trình Debugging & Bài học
- **Lỗi kỹ thuật phức tạp nhất gặp phải:** chuỗi lỗi liên hoàn về hạ tầng LLM: (1) model mặc định của bài lab (`llama-3.3-70b-versatile`) đã bị Groq gỡ bỏ hoàn toàn khỏi catalog, (2) model thay thế `gpt-oss-120b`/`gpt-oss-20b` liên tục chạm giới hạn 200,000 token/ngày giữa chừng pipeline (coreference + NER/RE tốn rất nhiều token do các model này sinh nhiều "reasoning token" ẩn), (3) một số model nhỏ hơn (`qwen`, model 20b) thỉnh thoảng trả JSON sai định dạng (`item` là string thay vì object) gây crash `AttributeError`/`TypeError` nếu không có defensive check.
- **Cách xử lý:** xây dựng cơ chế retry đa tầng — mỗi khi 1 model hết quota, chuyển sang model khác còn quota (được xác minh bằng lệnh test nhỏ trước khi retry hàng loạt); mọi bước tốn thời gian đều lưu checkpoint CSV ra đĩa ngay sau khi hoàn thành để không mất tiến độ nếu máy/kernel bị ngắt giữa chừng (thực tế đã xảy ra ít nhất 2 lần trong buổi làm).

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)
- **Tên đồ án / Dự án:** Trợ lý tra cứu tri thức nội bộ doanh nghiệp (Enterprise Knowledge Assistant) — hệ thống hỏi-đáp trên tài liệu, tin tức và hồ sơ quan hệ đối tác/khách hàng của công ty.
- **Đặc thù bài toán & Lý do chọn giải pháp:** Phần lớn câu hỏi thực tế của người dùng nội bộ là factoid đơn giản ("Hợp đồng X ký ngày nào?", "Ai phụ trách dự án Y?") — với nhóm này Flat RAG là đủ và rẻ hơn nhiều. Tuy nhiên có một tỷ lệ đáng kể câu hỏi dạng multi-hop/cross-doc thực sự cần thiết (ví dụ: "Đối tác nào từng làm việc với cả phòng A và phòng B, và hợp đồng đó còn hiệu lực không?") — những câu này Flat RAG dễ bỏ sót vì thông tin nằm rải rác ở nhiều tài liệu không tương đồng về mặt ngữ nghĩa bề mặt. Vì vậy lựa chọn kiến trúc **Hybrid RAG** (Flat + GraphRAG) giống mô hình trong lab, không dùng GraphRAG thuần vì chi phí vận hành/latency cao hơn không đáng cho phần lớn truy vấn đơn giản.
- **Cấu trúc Node & Relation dự kiến:**
  - Nodes: `Person` (nhân viên/đối tác), `Organization` (phòng ban/công ty đối tác), `Project`, `Document` (hợp đồng/báo cáo)
  - Relations: `WORKS_ON`, `MANAGES`, `PARTNERED_WITH`, `SIGNED`, `REFERENCES` — kèm provenance bắt buộc `source_document_id`, `date`, `evidence` giống pattern đã học trong lab.
- **Chiến lược xử lý Super-node & Entity Resolution:** Áp dụng đúng bài học từ lab: ngưỡng cosine ~0.90 cho vector match + lexical/rule guard riêng cho tên người/tổ chức tiếng Việt (xử lý dấu, viết tắt phòng ban) để tránh false merge; theo dõi degree distribution ngay từ đầu (các phòng ban lớn hoặc đối tác chiến lược dễ thành super-node thật) để bật cap 50 cạnh mới nhất kịp thời, tránh tình huống như trong lab là dữ liệu quá thưa nên cơ chế này chưa từng được kiểm chứng bằng dữ liệu thật.

---

## 🎯 TỰ ĐÁNH GIÁ
| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 4 | Hiểu rõ luồng end-to-end (coreference → extraction → entity resolution → bulk insert → hybrid retrieval → eval), giải thích được từng lựa chọn threshold/policy. |
| Khả năng kiểm soát AI Coding Agent | 4 | Không nhận toàn bộ output của Agent làm đúng — tự phát hiện và sửa nhiều bug hạ tầng (model deprecated, SSL, column mapping, JSON malformed), quyết định phạm vi dữ liệu phù hợp thời gian thực tế. |
| Chất lượng đồ thị tri thức xây dựng | 3 | Graph chạy đúng schema/provenance (0 lỗi), nhưng quy mô còn nhỏ (132 triples/108 edges) do giới hạn thời gian và quota API — chưa đủ dày để thấy rõ ưu thế GraphRAG hay kích hoạt Super-node Mitigation bằng dữ liệu thật. |
| Khả năng phân tích và debug hệ thống | 4 | Truy vết được nhiều lớp lỗi khác nhau (model 404, rate limit theo từng model riêng biệt, SSL do antivirus, JSON schema không nhất quán giữa các model) và có chiến lược retry/checkpoint để không mất tiến độ khi bị gián đoạn giữa chừng. |
