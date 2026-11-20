"""Huấn luyện tokenizer của đồ án trên Open-R1 + GSM8K và đo chất lượng.

Cách chạy (cần Internet, không cần GPU, khoảng 5 phút trên Kaggle):
  python scripts/prepare_tokenizer.py --out tokenizer.json            # cấu hình đã chốt
  python scripts/prepare_tokenizer.py --gsm8k-repeat 1 5              # so sánh hai mức lặp

Kho huấn luyện tokenizer:
  - Open-R1 (tập default): lấy 1 bài trong mỗi `--openr1-every` bài, rải đều cả tập.
  - GSM8K train: lặp lại `--gsm8k-repeat` lần. GSM8K nhỏ hơn Open-R1 hàng trăm lần,
    lặp lại để từ vựng đời thường của GSM8K không bị từ vựng LaTeX lấn át.
Đo trên dữ liệu KHÔNG dùng để huấn luyện: các bài Open-R1 để riêng và GSM8K test.
Truyền nhiều giá trị cho --gsm8k-repeat để so sánh; mỗi giá trị lưu một file riêng.

Kết quả đo ngày 20/11/2026 (vocab 8.192, đo trên 1.000 bài Open-R1 để riêng và GSM8K test):
  lặp 1 lần: Open-R1 2,46 ký tự/token | GSM8K test 2,51 ký tự/token
  lặp 5 lần: Open-R1 2,45 ký tự/token | GSM8K test 2,59 ký tự/token   <- chọn mức này
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from llm.data import gsm8k_text, openr1_text, pick_generation
from llm.tokenizer import Tokenizer

OPENR1_ID, OPENR1_CONFIG, GSM8K_ID = "open-r1/OpenR1-Math-220k", "default", "openai/gsm8k"


def split_openr1(rows, every, max_heldout=1000):
    """Chia các bài Open-R1 dùng được: bài thứ 0, every, 2*every... để huấn luyện,
    bài ngay sau đó để đo."""
    train, held, k = [], [], 0
    for row in rows:
        gen, _ = pick_generation(row)
        if gen is None:
            continue
        if k % every == 0:
            train.append(openr1_text(row, gen))
        elif k % every == 1 and len(held) < max_heldout:
            held.append(openr1_text(row, gen))
        k += 1
    return train, held


def measure(tok, texts):
    """Đo một tokenizer trên một danh sách văn bản."""
    lens, n_chars, bad = [], 0, 0
    for t in texts:
        ids = tok.encode(t)
        bad += tok.decode(ids) != t
        lens.append(len(ids))
        n_chars += len(t)
    n = max(len(lens), 1)
    return {"so_van_ban": len(lens), "ky_tu_moi_token": n_chars / max(sum(lens), 1),
            "token_dai_nhat": max(lens, default=0),
            "vua_1024": 100 * sum(x <= 1024 for x in lens) / n,
            "vua_2048": 100 * sum(x <= 2048 for x in lens) / n, "sai_khu_hoi": bad}


def train_and_measure(openr1_train, gsm_train, openr1_held, gsm_test, repeat, vocab_size):
    t0 = time.time()
    tok = Tokenizer.train(openr1_train + gsm_train * repeat, vocab_size)
    return tok, time.time() - t0, measure(tok, openr1_held), measure(tok, gsm_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openr1-every", type=int, default=20)
    ap.add_argument("--gsm8k-repeat", type=int, nargs="+", default=[5])
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--out", default="tokenizer.json")
    args = ap.parse_args()

    from datasets import load_dataset
    print("Đang đọc Open-R1 (streaming)...")
    stream = load_dataset(OPENR1_ID, OPENR1_CONFIG, split="train", streaming=True)
    openr1_train, openr1_held = split_openr1(stream, args.openr1_every)
    gsm = load_dataset(GSM8K_ID, "main")
    gsm_train = [gsm8k_text(r) for r in gsm["train"]]
    gsm_test = [gsm8k_text(r) for r in gsm["test"]]
    print(f"Open-R1: {len(openr1_train):,} bài huấn luyện "
          f"({sum(map(len, openr1_train)) / 1e6:.1f} triệu ký tự), {len(openr1_held):,} bài để đo")
    print(f"GSM8K: {len(gsm_train):,} bài train ({sum(map(len, gsm_train)) / 1e6:.1f} triệu ký tự), "
          f"{len(gsm_test):,} bài test")

    for repeat in args.gsm8k_repeat:
        tok, secs, m_r1, m_gsm = train_and_measure(openr1_train, gsm_train, openr1_held,
                                                   gsm_test, repeat, args.vocab_size)
        out = args.out if len(args.gsm8k_repeat) == 1 else args.out.replace(".json", f"_r{repeat}.json")
        tok.save(out)
        print(f"\n=== GSM8K lặp {repeat} lần | vocab {tok.vocab_size:,} | huấn luyện {secs:.0f} giây | lưu {out} ===")
        for name, m in (("Open-R1 để riêng", m_r1), ("GSM8K test", m_gsm)):
            print(f"{name:>17}: {m['ky_tu_moi_token']:.2f} ký tự/token | dài nhất {m['token_dai_nhat']:,} token"
                  f" | vừa 1.024: {m['vua_1024']:.1f}% | vừa 2.048: {m['vua_2048']:.1f}%"
                  f" | sai khứ hồi: {m['sai_khu_hoi']}")


if __name__ == "__main__":
    main()
