# LLM từ số 0 – Hướng C (Reasoning & Test-Time Compute)

Đồ án cuối khóa: tự xây dựng tokenizer, Transformer decoder-only và KV Cache bằng
PyTorch thuần, rồi tinh chỉnh mô hình giải toán đố GSM8K theo chuỗi tư duy.

## Cấu trúc

| Đường dẫn | Nội dung |
| --- | --- |
| `llm/tokenizer.py` | Tokenizer BPE mức byte, tự cài đặt, chỉ dùng thư viện chuẩn |
| `llm/data.py` | Hàm chọn lời giải Open-R1 và ghép văn bản GSM8K |
| `scripts/prepare_tokenizer.py` | Huấn luyện tokenizer của đồ án trên Open-R1 + GSM8K và đo chất lượng |
| `scripts/train_tokenizer.py` | Huấn luyện tokenizer từ file `.jsonl` hoặc `.txt` bất kỳ |
| `scripts/count_data.py` | Đo độ dài dữ liệu Open-R1 và GSM8K (chạy trên Kaggle) |
| `tests/test_tokenizer.py` | 10 kiểm thử cho tokenizer |
| `tests/test_data.py` | Kiểm thử phần chọn và đếm dữ liệu |

## Chạy thử

```bash
python tests/test_tokenizer.py
python tests/test_data.py
python scripts/train_tokenizer.py --input data/train.jsonl --eval data/test.jsonl --vocab-size 8192 --out tokenizer.json
```

Phần tokenizer không cần cài thêm thư viện nào ngoài Python 3.9 trở lên.

## Kết quả tuần 1 (đo ngày 20/11/2026 trên Kaggle)

| Phép đo | Kết quả |
| --- | --- |
| Open-R1 tập default | 93.595 bài có lời giải đúng, trung vị 11.954 ký tự mỗi bài |
| Tokenizer BPE, vocab 8.192 | Huấn luyện 25 giây trên 72,3 triệu ký tự Open-R1 + GSM8K train lặp 5 lần |
| Mật độ trên dữ liệu không dùng để huấn luyện | Open-R1: 2,45 ký tự/token; GSM8K test: 2,59 ký tự/token |
| Open-R1 vừa trọn block size | 1,3% bài vừa 1.024 token; 10,3% vừa 2.048 token |
| GSM8K test | Bài dài nhất 580 token, 100% vừa 1.024 token |
| Mã hóa rồi giải mã | Khớp toàn bộ 1.000 bài Open-R1 và 1.319 bài GSM8K test |

File `tokenizer.json` chưa nằm trong thư mục này: bản đã huấn luyện đang ở phần output của
notebook Kaggle. Tạo lại bằng `python scripts/prepare_tokenizer.py --out tokenizer.json`.
