"""
Tuần 1 - Đếm dữ liệu cho đồ án LLM từ số 0, Hướng C.

Mục đích: biết tập con Open-R1 gồm các lời giải NGẮN còn bao nhiêu mẫu,
và kiểm tra độ dài các bài GSM8K so với block size 1.024 token.

Cách chạy (cần Internet, không cần GPU):
  python scripts/count_data.py
Chạy thử nhanh: đặt MAX_ROWS = 2000. Chạy thật: để MAX_ROWS = None.

Kết quả đo ngày 20/11/2026 trên tập default: 93.595 bài dùng được, trung vị
11.954 ký tự; chỉ 4.108 bài (4,4%) dưới 4.000 ký tự.

Chưa có tokenizer nên độ dài đo bằng SỐ KÝ TỰ. Số token chỉ là ước lượng thô
(1 token ~ 3 đến 4 ký tự) và sẽ được đo lại khi tokenizer BPE viết xong.
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from llm.data import pick_generation

OPENR1_ID = "open-r1/OpenR1-Math-220k"
OPENR1_CONFIG = "default"          # tập 94k bài; "extended" là tập mở rộng
GSM8K_ID = "openai/gsm8k"
THRESHOLDS = [1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000]   # ký tự
SAVE_MAX_CHARS = 8000              # lưu rộng tay để sau này lọc lại theo token
OUT_PATH = "openr1_short.jsonl"
MAX_ROWS = None                    # None = quét toàn bộ

REQUIRED_COLS = ["problem", "generations", "correctness_math_verify"]


def percentile(sorted_vals, p):
    """Phân vị p (0-100) của một danh sách đã sắp xếp, lấy phần tử gần nhất."""
    if not sorted_vals:
        return 0
    k = round((len(sorted_vals) - 1) * p / 100)
    return sorted_vals[k]


def summarize(lengths, thresholds):
    """Với mỗi ngưỡng: số mẫu ngắn hơn ngưỡng và tổng số ký tự của các mẫu đó."""
    rows = []
    for t in thresholds:
        kept = [n for n in lengths if n <= t]
        rows.append({"nguong": t, "so_mau": len(kept), "tong_ky_tu": sum(kept)})
    return rows


def print_report(name, lengths, thresholds, total_rows):
    s = sorted(lengths)
    print(f"\n=== {name} ===")
    print(f"Số bài đã quét: {total_rows:,} | số bài dùng được: {len(s):,}")
    if not s:
        return
    print("Độ dài mỗi mẫu (ký tự): "
          + ", ".join(f"p{p}={percentile(s, p):,}" for p in (10, 25, 50, 75, 90))
          + f", max={s[-1]:,}")
    print(f"{'Ngưỡng ký tự':>13} | {'Số mẫu':>8} | {'Tỷ lệ':>6} | "
          f"{'Tổng ký tự':>13} | Token ước lượng (thô)")
    for r in summarize(s, thresholds):
        share = 100 * r["so_mau"] / len(s)
        lo, hi = r["tong_ky_tu"] / 4 / 1e6, r["tong_ky_tu"] / 3 / 1e6
        print(f"{r['nguong']:>13,} | {r['so_mau']:>8,} | {share:>5.1f}% | "
              f"{r['tong_ky_tu']:>13,} | {lo:.1f} đến {hi:.1f} triệu")


def scan_openr1(rows, out_path=OUT_PATH, max_rows=MAX_ROWS):
    """Quét Open-R1, trả về (danh sách độ dài, bộ đếm lý do loại, số bài đã quét)."""
    lengths, reasons, n = [], Counter(), 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            if n == 0:
                missing = [c for c in REQUIRED_COLS if c not in row]
                if missing:
                    sys.exit(f"Thiếu cột {missing}. Các cột hiện có: {list(row.keys())}")
            n += 1
            gen, why = pick_generation(row)
            if gen is None:
                reasons[why] += 1
            else:
                size = len(row["problem"]) + len(gen)
                lengths.append(size)
                if size <= SAVE_MAX_CHARS:
                    f.write(json.dumps({"problem": row["problem"], "generation": gen,
                                        "answer": row.get("answer"), "chars": size},
                                       ensure_ascii=False) + "\n")
            if n % 10000 == 0:
                print(f"  đã quét {n:,} bài, dùng được {len(lengths):,}")
            if max_rows and n >= max_rows:
                break
    return lengths, reasons, n


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Chưa có thư viện datasets. Chạy: pip install datasets")

    print(f"Đang quét {OPENR1_ID} ({OPENR1_CONFIG}), chế độ streaming...")
    stream = load_dataset(OPENR1_ID, OPENR1_CONFIG, split="train", streaming=True)
    lengths, reasons, n = scan_openr1(stream)
    print_report("Open-R1: bài toán + lời giải ngắn nhất đúng", lengths, THRESHOLDS, n)
    print("Số bài bị loại theo lý do:", dict(reasons))
    print(f"Đã lưu các mẫu <= {SAVE_MAX_CHARS:,} ký tự vào {OUT_PATH}")

    print(f"\nĐang tải {GSM8K_ID}...")
    gsm = load_dataset(GSM8K_ID, "main")
    for split in ("train", "test"):
        ls = [len(r["question"]) + len(r["answer"]) for r in gsm[split]]
        print_report(f"GSM8K {split}: câu hỏi + lời giải", ls, [1000, 2000, 3000, 4000], len(ls))


if __name__ == "__main__":
    main()
