"""Huấn luyện tokenizer BPE từ các file .jsonl hoặc .txt rồi lưu ra tokenizer.json.

Ví dụ:
  python scripts/train_tokenizer.py --input data/train.jsonl --eval data/test.jsonl \
      --vocab-size 8192 --out tokenizer.json

File .jsonl: mỗi dòng là một đối tượng JSON, mọi trường kiểu chuỗi được nối lại
thành một văn bản (GSM8K: question + answer; Open-R1: problem + generation).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm.tokenizer import Tokenizer


def iter_texts(paths, max_chars=None):
    """Đọc lần lượt từng văn bản, dừng khi đã đủ max_chars ký tự (nếu có đặt)."""
    seen = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if path.endswith(".jsonl"):
                    row = json.loads(line)
                    text = "\n".join(v for v in row.values() if isinstance(v, str))
                else:
                    text = line
                yield text
                seen += len(text)
                if max_chars and seen >= max_chars:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="file huấn luyện (.jsonl hoặc .txt)")
    ap.add_argument("--eval", nargs="*", default=[], help="file để đo, không dùng để huấn luyện")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--max-chars", type=int, default=None, help="giới hạn số ký tự dùng để huấn luyện")
    ap.add_argument("--out", default="tokenizer.json")
    args = ap.parse_args()

    t0 = time.time()
    tok = Tokenizer.train(iter_texts(args.input, args.max_chars), args.vocab_size)
    print(f"Huấn luyện xong trong {time.time() - t0:.1f} giây, vocab = {tok.vocab_size:,}")
    tok.save(args.out)
    print(f"Đã lưu {args.out}")

    eval_paths = args.eval or args.input
    label = "tập đo riêng" if args.eval else "chính tập huấn luyện"
    n_docs = n_chars = n_tokens = longest = bad = 0
    t0 = time.time()
    for text in iter_texts(eval_paths, None if args.eval else 5_000_000):
        ids = tok.encode(text)
        bad += tok.decode(ids) != text
        n_docs += 1
        n_chars += len(text)
        n_tokens += len(ids)
        longest = max(longest, len(ids))
    print(f"Đo trên {label}: {n_docs:,} văn bản, {n_chars:,} ký tự, {n_tokens:,} token")
    print(f"  {n_chars / max(n_tokens, 1):.2f} ký tự mỗi token | văn bản dài nhất {longest:,} token"
          f" | mã hóa {n_tokens / max(time.time() - t0, 1e-9):,.0f} token/giây")
    print(f"  Mã hóa rồi giải mã sai ở {bad} văn bản (phải bằng 0)")
    by_len = sorted(tok.vocab.items(), key=lambda kv: -len(kv[1]))[:15]
    print("  Token dài nhất:", [v.decode("utf-8", "replace") for _, v in by_len])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
