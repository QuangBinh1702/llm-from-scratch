"""Kiểm thử tokenizer. Chạy:  python tests/test_tokenizer.py   (hoặc pytest)"""
import os
import random
import sys
import tempfile
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm.tokenizer import CHUNK_RE, N_BYTES, SPECIAL_TOKENS, Tokenizer, train_merges

CORPUS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many in May.",
    "How many clips did Natalia sell altogether in April and May? 48/2 = 24, 48+24 = 72.",
    "<think>Let $x$ be the number. Then \\frac{x}{2} + 3 = 10, so x = 14.</think> Answer: 14",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting.",
    "Xin chào, đây là tiếng Việt có dấu: đồ án mô hình ngôn ngữ lớn từ số 0.",
    "The sum is \\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}, which gives 5050 when n = 100.",
] * 20

SAMPLES = [
    "", " ", "a", "  hai  dấu cách  ", "dòng 1\ndòng 2\r\n\tthụt lề", "12345", "x = 1234.56",
    "Tiếng Việt: Nguyễn Quang Bình, Đà Nẵng", "emoji 🙂🚀 và ký tự hiếm ½ ² ∑ ∫ 中文 العربية",
    "\\frac{a}{b} \\\\ \\( x^2_i \\) $$E=mc^2$$ a_b __init__", "<think>3 + 4 = 7</think> Answer: 7",
    "<bos><eos><pad>", "< think > <thin", "kết thúc bằng dấu \\", "\x00\x01 ký tự điều khiển",
]


def naive_train(chunk_freq, n_merges, first_new_id, min_frequency=2):
    """Bản BPE viết thẳng theo định nghĩa, chậm, dùng làm chuẩn đối chiếu."""
    words = {tuple(c.encode("utf-8")): f for c, f in chunk_freq.items()}
    merges = []
    for _ in range(n_merges):
        counts = Counter()
        for w, f in words.items():
            for p in zip(w, w[1:]):
                counts[p] += f
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))
        if counts[best] < min_frequency:
            break
        new_id = first_new_id + len(merges)
        merges.append(best)
        new_words = Counter()
        for w, f in words.items():
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                    out.append(new_id); i += 2
                else:
                    out.append(w[i]); i += 1
            new_words[tuple(out)] += f
        words = new_words
    return merges


def chunk_freq_of(texts):
    cf = Counter()
    for t in texts:
        cf.update(CHUNK_RE.findall(t))
    return cf


TOK = Tokenizer.train(CORPUS, 600)


def test_chunks_cover_all_text():
    for s in SAMPLES + CORPUS[:6]:
        assert "".join(CHUNK_RE.findall(s)) == s, repr(s)


def test_roundtrip():
    for s in SAMPLES + CORPUS[:6]:
        assert TOK.decode(TOK.encode(s)) == s, repr(s)
        assert TOK.decode(TOK.encode(s, allow_special=False)) == s, repr(s)


def test_roundtrip_random_unicode():
    rng = random.Random(0)
    alphabet = "abc XYZ 0123456789 \n\t.,;:!?\\{}$^_<>/|éấữđ🙂∑中"
    for _ in range(500):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        assert TOK.decode(TOK.encode(s)) == s, repr(s)
        assert TOK.decode(TOK.encode(s, allow_special=False)) == s, repr(s)


def test_each_digit_is_one_token():
    for s in ["12345", "x = 1234.56", "in 2026 there were 1319 items", "a1b22c333"]:
        ids = TOK.encode(s)
        pieces = [TOK.decode([i]) for i in ids]
        assert "".join(pieces) == s
        for piece in pieces:
            if any(ch.isdigit() for ch in piece):
                assert len(piece) == 1, (s, pieces)
    assert len(TOK.encode("9876543210")) == 10


def test_special_tokens():
    ids = TOK.encode("<think>3 + 4</think>")
    assert ids[0] == TOK.token_id("<think>") and ids[-1] == TOK.token_id("</think>")
    assert sum(i in TOK.id_to_special for i in ids) == 2
    # Không cho phép token đặc biệt: không ID đặc biệt nào được sinh ra.
    plain = TOK.encode("<think>3 + 4</think><eos>", allow_special=False)
    assert not any(i in TOK.id_to_special for i in plain)
    assert TOK.decode(plain) == "<think>3 + 4</think><eos>"
    wrapped = TOK.encode("hi", add_bos=True, add_eos=True)
    assert wrapped[0] == TOK.token_id("<bos>") and wrapped[-1] == TOK.token_id("<eos>")
    assert TOK.decode(wrapped, skip_special=True) == "hi"
    assert [TOK.token_id(t) for t in SPECIAL_TOKENS] == list(range(N_BYTES, N_BYTES + 5))


def test_ids_in_range_and_vocab_size():
    assert TOK.vocab_size <= 600
    for s in SAMPLES:
        assert all(0 <= i < TOK.vocab_size for i in TOK.encode(s))
    try:
        TOK.decode([TOK.vocab_size + 5])
        raise AssertionError("phải báo lỗi với ID ngoài vocab")
    except ValueError:
        pass


def test_training_is_deterministic():
    assert Tokenizer.train(CORPUS, 600).merges == Tokenizer.train(list(CORPUS), 600).merges


def test_fast_trainer_matches_naive_reference():
    cf = chunk_freq_of(CORPUS)
    first = N_BYTES + len(SPECIAL_TOKENS)
    for n in (1, 10, 100, 339, 2000):
        assert train_merges(cf, n, first) == naive_train(cf, n, first), n
    # Kho văn bản ngẫu nhiên: nhiều cặp hòa nhau, thử kỹ quy tắc phân xử.
    rng = random.Random(1)
    for trial in range(30):
        cf = Counter({"".join(rng.choice("abcd") for _ in range(rng.randint(1, 8))):
                      rng.randint(1, 5) for _ in range(40)})
        assert train_merges(cf, 60, first) == naive_train(cf, 60, first), trial


def test_merges_compress_and_never_cross_chunks():
    text = CORPUS[0]
    assert len(TOK.encode(text)) < len(text.encode("utf-8"))
    # Mã hóa cả câu phải bằng ghép kết quả mã hóa từng mẩu.
    joined = []
    for c in CHUNK_RE.findall(text):
        joined.extend(TOK.encode(c))
    assert joined == TOK.encode(text)


def test_save_and_load():
    path = os.path.join(tempfile.mkdtemp(), "tok.json")
    TOK.save(path)
    again = Tokenizer.load(path)
    assert again.merges == TOK.merges and again.vocab_size == TOK.vocab_size
    for s in SAMPLES:
        assert again.encode(s) == TOK.encode(s)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ĐẠT ", name)
    print(f"\nTất cả {len(tests)} kiểm thử đều đạt.")
