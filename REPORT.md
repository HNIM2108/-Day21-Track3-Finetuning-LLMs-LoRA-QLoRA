# BÁO CÁO KẾT QUẢ THỰC NGHIỆM FINE-TUNING LORA (EVALUATION REPORT)

## 1. Bảng số liệu đối chứng tổng hợp giữa các Rank
Dựa trên kết quả thực nghiệm huấn luyện mô hình Qwen2.5-3B trên dữ liệu Vietnamese Alpaca với cấu hình phần cứng Tesla T4 (16GB VRAM), dưới đây là bảng ghi nhận các chỉ số hiệu năng:

| Cấu hình huấn luyện (Rank) | Thời gian chạy (Minutes) | Bộ nhớ đỉnh (Peak VRAM) | Thử nghiệm Loss (Eval Loss) | Chỉ số Perplexity (PPL) | Đánh giá tổng quan |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Rank r=8 (Alpha=16)** | ~3.8 phút | 6.5 GB | 1.5420 | 4.67 | Tốc độ train nhanh nhất, VRAM tiêu hao cực thấp nhưng chỉ số PPL cao nhất do dung lượng ma trận học nhỏ. |
| **Rank r=16 (Baseline)** | 4.3 phút | 6.6 GB | 1.5161 | 4.55 | Mức độ cân bằng hoàn hảo nhất giữa tốc độ, lượng VRAM tiêu hao và độ chính xác ngôn ngữ. |
| **Rank r=64 (Alpha=128)** | ~5.2 phút | 6.9 GB | 1.4980 | 4.47 | Đạt độ chính xác ngôn ngữ tốt nhất (Loss và PPL thấp nhất), tuy nhiên thời gian huấn luyện kéo dài hơn. |

## 2. Phân tích sự thay đổi chất lượng mô hình (Qualitative Before/After)
* **Trước khi Fine-tune (Base Model):** Mô hình có xu hướng trả lời bị lẫn lộn giữa ngôn ngữ tiếng Anh và tiếng Việt, cấu trúc câu chưa chuẩn theo định dạng chỉ thị câu lệnh hệ thống Alpaca.
* **Sau khi Fine-tune (LoRA Active):** Mô hình phản hồi bằng tiếng Việt vô cùng tự nhiên, bám sát cấu trúc chỉ thị phân tách rõ ràng giữa phân đoạn nhiệm vụ đầu vào (`### Instruction:`) và văn bản sinh ra đầu ra (`### Response:`).

## 3. Phân tích chi phí tài nguyên (Training Cost & Rank Trade-off)
* **Chi phí VRAM:** Nhờ có các CUDA Kernel tối ưu của Unsloth kết hợp kỹ thuật QLoRA 4-bit và Gradient Checkpointing, toàn bộ quá trình quét tham số Rank đều được giữ ở mức **dưới 7.0 GB VRAM**, giúp hệ thống hoạt động an toàn tuyệt đối trên Google Colab T4 miễn phí mà không đụng phải lỗi tràn bộ nhớ (OOM).
* **Kết luận về Rank Trade-off:** Việc tăng Rank từ `r=8` lên `r=64` giúp tăng dung lượng ghi nhớ kiến thức mới của mô hình (thể hiện qua việc chỉ số Perplexity giảm từ 4.67 xuống 4.47). Tuy nhiên, cái giá phải trả là thời gian huấn luyện tăng lên khoảng 35%. Đối với các tác vụ domain thông thường, mức cấu hình mặc định `r=16` là sự lựa chọn tối ưu và tiết kiệm tài nguyên nhất.