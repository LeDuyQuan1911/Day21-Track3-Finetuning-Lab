# Lab 21 — Báo cáo fine-tuning và đánh giá

**MSSV:** 2A202602731
**Họ tên:** chưa được cung cấp
**Ngày đo:** 08/10/2026
**Môi trường:** Google Colab, Tesla T4 14.6 GB, fp16
**Base model:** unsloth/Qwen3.5-4B

## 1. Lựa chọn và dữ liệu

Tôi dùng 250 ticket chăm sóc khách hàng tiếng Việt do repo sinh theo seed, gán bốn trường intent, urgency, product, sentiment. Tập train/validation là 225/25 mẫu (seed 42); target eval có 50 mẫu và regression có 15 câu hỏi phổ thông. Bốn trường có nhãn đóng, nên có thể chấm độ chính xác từng trường mà không cần LLM làm giám khảo. Tôi dùng model mặc định cho T4 để cùng một base phục vụ baseline trước train và mọi adapter sau train. Dữ liệu tổng hợp dùng các khuôn câu tương tự giữa train và eval, nên điểm target chưa chứng minh khả năng tổng quát sang ticket thật.

NB1 đo độ dài token: p95 = 98, tối đa = 101; tôi đặt max_length = 256 theo suggested_max_length trong results/token_stats.json. Cấu hình đúng gắn LoRA vào 12 loại linear thuộc text decoder, r = 16, alpha = 32, LR = 1e-4, batch hiệu dụng 16, hai epoch và 30 optimizer step. Colab dùng fp16 vì T4 không hỗ trợ bf16. Model có 24 lớp linear attention và 8 lớp full attention theo cấu hình in ở NB3.

## 2. Bằng chứng loss mask — NB1

results/mask_proof.json ghi 39/94 token được tính loss, supervised_fraction = 0.4149; answer_is_supervised = true và question_is_masked = true. Phần giải mã từ các vị trí labels khác -100 bắt đầu bằng token đóng think, tiếp theo là JSON nhãn và token kết thúc câu trả lời. System prompt và ticket người dùng đều mang nhãn -100. Trên toàn bộ 225 mẫu train đã tokenize, NB3 ghi 9.014/20.951 token được giám sát (43.0%). Vì vậy loss được đặt trên câu trả lời, không phải cả prompt. results/template_check.json xác nhận chat template giữ khối think. Đường assistant mask tự động của tokenizer trả về mask rỗng do template thiếu marker generation; pipeline train dùng mask tiền tokenize đã được NB1 chứng minh.

## 3. Baseline đóng băng trước train — NB2

NB2 chạy trên base chưa gắn adapter, đủ 50 target và 15 regression. results/baselines_frozen.json lưu dự đoán từng mẫu của (b), SHA prompt tối ưu 719e74d3b6232053 và checksum hai tập eval. Tôi không sửa prompt tối ưu sau khi đóng băng. Prompt (b) gồm chỉ dẫn JSON rõ ràng và few-shot, là đối thủ thật của fine-tune.

| Cấu hình | Target | Regression | Format | Latency ms/mẫu |
|---|---:|---:|---:|---:|
| (a) base + prompt đơn giản | 0.000 | 0.7911 | 0.000 | 3166.9 |
| (b) base + prompt tối ưu | 0.765 | 0.7911 | 1.000 | 994.2 |
| (c) LoRA đúng + prompt đơn giản | 0.965 | 0.6778 | 1.000 | 1338.9 |

(b) cao hơn (a) 0.765 điểm target; không có việc làm yếu prompt (b) để tạo chiến thắng giả cho (c). Latency là số đo greedy decode trên T4 trong lần chạy này, không phải cam kết hiệu năng phục vụ.

## 4. Ba đối chứng NB4, chấm lại bằng NB5

Mỗi run dùng cùng 225 mẫu, mask, batch và 30 optimizer step. attn_only chỉ thay vị trí gắn adapter: rank 283 làm 32.456.704 tham số học, so với 32.464.896 của correct (lệch khoảng 0.025%). wrong_lr chỉ giảm LR mười lần. qlora chỉ đổi base sang 4-bit và được chấm trên base 4-bit tương ứng.

| Run | Vị trí / lượng tử | r | Tham số học | LR | Loss train | Target NB5 | Format | VRAM đỉnh GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| correct | 12 text-linear, fp16 | 16 | 32.464.896 | 1e-4 | 0.6260 | 0.965 | 1.000 | 8.78 |
| attn_only | q,v, fp16 | 283 | 32.456.704 | 1e-4 | 0.5382 | 0.970 | 1.000 | 8.79 |
| wrong_lr | 12 text-linear, fp16 | 16 | 32.464.896 | 1e-5 | 1.5702 | 0.000 | 0.000 | 8.78 |
| qlora | 12 text-linear, 4-bit | 16 | 32.464.896 | 1e-4 | 0.7058 | 0.940 | 1.000 | 3.86 |

**4.1 — Rank và vị trí.** attn_only thắng nhẹ correct trên target: 0.970 so với 0.965, chênh 0.005, tương đương một trường đúng thêm trên 200 trường được chấm. Nó cũng có loss train thấp hơn, nên thứ tự hai thước đo ở cặp này trùng nhau. Kết quả không ủng hộ khẳng định rằng gắn vào mọi linear luôn tốt hơn trên bài triage hẹp này. Rank 283 đã cân bằng ngân sách tham số, nhưng một lần chạy với 50 ticket và một seed chưa tách được hiệu ứng vị trí khỏi biến thiên đo lường nhỏ như 0.005.

**4.2 — Learning rate.** wrong_lr dùng 1e-5 thay 1e-4, còn các biến kiểm soát và 30 step giữ nguyên. Ở mốc một epoch, loss log của nó là 1.606 trong khi correct là 0.1399; loss tổng hợp cuối là 1.5702 so với 0.6260. Chỉ nhìn loss có thể nói chung chung rằng cấu hình này học kém, nhưng điểm target 0.000 và format 0.000 cho thấy dưới cùng ngân sách step nó chưa học được cách trả JSON. Vì vậy LR là nguyên nhân thực nghiệm có bằng chứng mạnh hơn việc tăng rank ở đây.

**4.3 — QLoRA.** VRAM đỉnh từ 8.78 xuống 3.86 GB, tiết kiệm 4.92 GB, khoảng 56.0%. Đổi lại, target giảm từ 0.965 xuống 0.940, loss tổng hợp tăng từ 0.6260 lên 0.7058 và thời gian train tăng từ 398.1 lên 452.8 giây. Số đo ủng hộ việc ưu tiên fp16 khi T4 đủ bộ nhớ cho model này, nhưng không chứng minh QLoRA vô dụng: 0.940 vẫn cao hơn baseline (b) 0.765 và format vẫn 1.000. Nếu chỉ có khoảng 4 GB VRAM, đánh đổi này có thể hợp lý tùy yêu cầu ứng dụng.

## 5. Phán quyết và phân tích hồi quy

**Verdict: FAILED.** Target tăng +0.200 so với (b), nhưng regression giảm -0.1133, vượt ngưỡng cho phép 0.020. Format vẫn 1.000; latency fine-tune là 1338.9 ms/mẫu, chậm hơn (b) 344.7 ms/mẫu trong phép đo này. valid_trace_rate = 0.0 không phải bằng chứng reasoning collapse vì dataset không có trace để học và tác vụ yêu cầu JSON ngắn.

Kết quả cho thấy adapter thực sự học cách phân loại ticket: nó nâng độ chính xác từng trường từ 0.765 lên 0.965 khi chỉ dùng prompt ngắn. Tuy nhiên, cổng triển khai còn kiểm tra năng lực phổ thông. Mười lăm câu regression được chấm bằng keyword recall, và điểm giảm từ 0.7911 xuống 0.6778. Đó là suy giảm 0.1133, hơn năm lần mức cho phép 0.020. Tôi giữ nguyên verdict FAILED, không nới cổng, không đổi tập eval, và không chuyển sang so với baseline (a) yếu hơn. Bộ train chỉ gồm JSON triage có thể kéo hành vi chung của model về khuôn trả lời quá hẹp. Cần thử thêm 1–5% dữ liệu replay cho năng lực phổ thông rồi chấm lại cả hai tập. Tập regression nhỏ và metric keyword recall chưa thay thế được kiểm định người dùng thực tế; nó đủ để cảnh báo, chưa đủ để định lượng mọi dạng suy giảm.

## 6. Định tính trên 50 ticket target

results/qualitative.json chứa dự đoán của (b) và (c) cho từng ticket: fine-tune thắng 33, hòa 17, thua 0 theo độ chính xác bốn trường. Vì không có ca thua trên tập target, yêu cầu nêu hai ca thua không thể đáp ứng trung thực. Bảng dưới đây gồm ba ca thắng và hai ca hòa vẫn sai một trường; suy giảm của fine-tune xuất hiện ở regression, không phải ở các cặp target này.

| Chỉ số | Ticket rút gọn | Nhãn đúng / khác biệt | Điểm (b) → (c) | Nhận xét |
|---:|---|---|---:|---|
| 6 | Đổi size. Hỏi cho biết thôi. | doi_tra, thap | 0.50 → 1.00 | (b) đoán hoàn tiền, cao; (c) sửa cả intent và urgency. |
| 7 | Muốn đổi. Đã 3 ngày rồi. | doi_tra, trung_binh | 0.50 → 1.00 | (b) đoán vận chuyển, cao; (c) đúng cả hai. |
| 19 | Vỡ khi nhận. Ngay lập tức. | san_pham_loi, trung_tinh | 0.50 → 1.00 | (b) đoán hoàn tiền và cảm xúc tiêu cực; (c) đúng nhãn. |
| 3 | Chưa thấy tiền. Khi nào tiện. | urgency đúng là thap | 0.75 → 0.75 | Cả hai đoán trung_binh; fine-tune chưa sửa ca khó này. |
| 12 | Bị lỗi. Khi nào tiện. | urgency đúng là thap | 0.75 → 0.75 | Cả hai đoán trung_binh; lỗi còn lại ở sắc thái thời hạn. |

Hai ca hòa còn sai gợi ý rằng nhãn urgency từ cụm “khi nào tiện” chưa được học chắc. Đây là quan sát trên ví dụ, không đủ cơ sở để quy mọi lỗi còn lại cho một nguyên nhân duy nhất.

## 7. Kết luận và điều học được

Tôi chưa nên triển khai adapter correct như bản thay thế chung cho base model. Nó nâng điểm triage thêm 0.200 so với một baseline prompt tối ưu thật, và vẫn tạo JSON đúng định dạng trên toàn bộ 50 ticket. Nhưng điều kiện hồi quy của bài đã thất bại: giảm 0.1133 ở nhóm câu hỏi phổ thông, trong khi chỉ cho phép giảm 0.020. Một phiên bản dùng riêng cho triage, có định tuyến rõ ràng và kiểm định bổ sung, có thể đáng thử; phép đo hiện tại chưa cấp phép cho quyết định đó. Đối chứng wrong_lr cho thấy cùng dữ liệu và step, giảm LR mười lần khiến model không tạo được JSON hợp lệ. attn_only nhỉnh hơn correct 0.005 dù chỉ gắn q,v, nên không thể tuyên bố mọi linear là lựa chọn tốt nhất cho mọi tác vụ từ một lần chạy. QLoRA giảm hơn nửa VRAM nhưng mất 0.025 điểm target và tốn thêm thời gian. Bằng chứng mask ở NB1 là điều kiện đầu tiên để tin các phép so này: nếu câu hỏi cũng nằm trong loss, mọi bảng điểm phía sau sẽ khó diễn giải. Cuối cùng, dữ liệu tổng hợp dùng lại khuôn tạo câu giữa train và eval có thể làm điểm target cao hơn hiệu năng trên ticket thật; bước tiếp theo phải là tập kiểm thử độc lập hơn, thêm replay phổ thông, rồi đóng băng và chạy lại toàn bộ cổng đánh giá.

Ba điều cụ thể tôi học được:

1. Prompt tối ưu + few-shot đưa base từ 0.000 lên 0.765; chỉ so fine-tune với prompt đơn giản sẽ thổi phồng giá trị của train.
2. Cân bằng số tham số học ở attn_only cần rank 283 thay vì 16; sau cân bằng, nó đạt 0.970, cao hơn correct 0.965 trên tập này.
3. Điểm target tăng không bảo đảm an toàn triển khai: correct thắng 33 ticket, không thua ticket nào, nhưng vẫn thất bại cổng regression.

Nếu có thêm hai giờ, tôi sẽ bổ sung replay 1%, 3% và 5%, giữ nguyên 50 target, 15 regression và prompt (b), chạy lại nhiều seed, rồi đánh giá trên ticket thực chưa dùng cùng khuôn sinh.

## Phụ lục: trạng thái thưởng

**NB6 hoàn tất trên T4.** results/merge_check.json ghi target trước merge = 0.965, sau merge = 0.965, delta = 0.000 trên đủ 50 mẫu, đạt tolerance 0.010. results/hot_swap.json ghi một base fp16 đã lần lượt dùng ba adapter correct, attn_only và wrong_lr cho cùng một ticket. Hai adapter đầu trả JSON đúng; wrong_lr trả lời bằng văn xuôi, phù hợp với format 0.000 của đối chứng đó. Model merge đã được lưu trong Colab nhưng không đưa vào ZIP vì riêng trọng số đầy đủ gần 8 GB; ZIP chứa adapter correct theo định dạng nộp A. Tôi không khai báo đã làm dataset miền riêng, reasoning-trace collapse, quét rank hoặc HF Hub khi chưa có artefact tương ứng.
