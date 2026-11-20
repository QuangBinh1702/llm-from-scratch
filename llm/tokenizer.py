"""Tokenizer BPE mức byte, tự cài đặt, chỉ dùng thư viện chuẩn của Python.

Ý tưởng:
  1. Văn bản được cắt thành các "mẩu" (chunk) bằng biểu thức chính quy: từ kèm
     dấu cách đứng trước, lệnh LaTeX (\\frac), cụm dấu câu, khoảng trắng.
     MỖI CHỮ SỐ là một mẩu riêng nên không bao giờ bị gộp với ký tự khác.
  2. Mỗi mẩu được đổi thành dãy byte UTF-8 (256 token gốc, không có ký tự lạ).
  3. Huấn luyện: lặp lại việc gộp cặp token kề nhau xuất hiện nhiều nhất thành
     một token mới, cho tới khi đủ kích thước vocab. Phép gộp không vượt qua
     ranh giới giữa hai mẩu.

Bố cục ID:  0..255 = byte | 256.. = token đặc biệt | sau đó = token do gộp.
"""
import heapq
import json
import re
from collections import Counter, defaultdict

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<think>", "</think>"]
N_BYTES = 256

# Thứ tự các nhánh có ý nghĩa: lệnh LaTeX | từ (kèm 1 dấu cách trước) | 1 chữ số
# | khoảng trắng (chừa lại dấu cách cuối cho từ đứng sau) | cụm dấu câu | dấu \ lẻ.
CHUNK_RE = re.compile(
    r"\\[A-Za-z]+| ?[^\W\d_]+|\d|\s+(?!\S)|\s+|(?:[^\s\w\\]|_)+|\\"
)


def _merge(ids, pair, new_id):
    """Thay mọi lần xuất hiện của cặp `pair` trong `ids` bằng `new_id`."""
    out, i, n = [], 0, len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_merges(chunk_freq, n_merges, first_new_id, min_frequency=2):
    """Học danh sách phép gộp từ bảng tần suất {mẩu (str): số lần xuất hiện}.

    Làm việc trên bảng tần suất thay vì văn bản thô, và chỉ cập nhật những mẩu
    có chứa cặp vừa gộp, nên nhanh hơn nhiều so với quét lại toàn bộ kho văn bản.
    Hòa nhau về tần suất thì chọn cặp có ID nhỏ hơn, để kết quả luôn lặp lại được.
    """
    words = [list(c.encode("utf-8")) for c in chunk_freq]
    freqs = list(chunk_freq.values())
    pair_count = defaultdict(int)
    pair_words = defaultdict(set)          # cặp -> chỉ số các mẩu có chứa cặp đó
    for i, w in enumerate(words):
        for p in zip(w, w[1:]):
            pair_count[p] += freqs[i]
            pair_words[p].add(i)
    heap = [(-c, p) for p, c in pair_count.items()]
    heapq.heapify(heap)

    merges = []
    while len(merges) < n_merges and heap:
        neg, pair = heapq.heappop(heap)
        if pair_count.get(pair, 0) != -neg:
            continue                        # mục cũ trong heap, tần suất đã đổi
        if -neg < min_frequency:
            break
        new_id = first_new_id + len(merges)
        merges.append(pair)
        changed = set()
        for i in pair_words.pop(pair):
            w, f = words[i], freqs[i]
            for p in zip(w, w[1:]):
                pair_count[p] -= f
                changed.add(p)
            w = words[i] = _merge(w, pair, new_id)
            for p in zip(w, w[1:]):
                pair_count[p] += f
                pair_words[p].add(i)
                changed.add(p)
        pair_count.pop(pair, None)
        for p in changed:
            c = pair_count.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))
            elif p in pair_count:
                del pair_count[p]
    return merges


class Tokenizer:
    def __init__(self, merges, special_tokens=SPECIAL_TOKENS):
        self.special_tokens = list(special_tokens)
        self.special_to_id = {t: N_BYTES + i for i, t in enumerate(self.special_tokens)}
        self.id_to_special = {i: t for t, i in self.special_to_id.items()}
        self.merges = [tuple(p) for p in merges]
        first = N_BYTES + len(self.special_tokens)
        self._pair_to_id = {p: first + i for i, p in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(N_BYTES)}
        for (a, b), new_id in self._pair_to_id.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        # Token đặc biệt dài hơn đứng trước để "</think>" không bị khớp nhầm.
        ordered = sorted(self.special_tokens, key=len, reverse=True)
        self._special_re = re.compile("(" + "|".join(map(re.escape, ordered)) + ")")
        self._cache = {}

    # ----- thuộc tính tiện dùng -----
    @property
    def vocab_size(self):
        return N_BYTES + len(self.special_tokens) + len(self.merges)

    def token_id(self, special_token):
        return self.special_to_id[special_token]

    # ----- huấn luyện -----
    @classmethod
    def train(cls, texts, vocab_size, special_tokens=SPECIAL_TOKENS, min_frequency=2):
        """Huấn luyện từ một dãy chuỗi văn bản (list hoặc generator)."""
        n_merges = vocab_size - N_BYTES - len(special_tokens)
        if n_merges < 0:
            raise ValueError("vocab_size nhỏ hơn 256 byte cộng số token đặc biệt")
        splitter = cls([], special_tokens)._special_re
        chunk_freq = Counter()
        for text in texts:
            for part in splitter.split(text):
                if part and part not in special_tokens:
                    chunk_freq.update(CHUNK_RE.findall(part))
        merges = train_merges(chunk_freq, n_merges, N_BYTES + len(special_tokens),
                              min_frequency)
        return cls(merges, special_tokens)

    # ----- mã hóa -----
    def _encode_chunk(self, chunk):
        ids = self._cache.get(chunk)
        if ids is None:
            ids = list(chunk.encode("utf-8"))
            pair_to_id = self._pair_to_id
            while len(ids) > 1:
                # Cặp được học sớm nhất (ID nhỏ nhất) phải được gộp trước.
                best = min(zip(ids, ids[1:]), key=lambda p: pair_to_id.get(p, 1 << 60))
                new_id = pair_to_id.get(best)
                if new_id is None:
                    break
                ids = _merge(ids, best, new_id)
            if len(self._cache) > 2_000_000:
                self._cache.clear()
            self._cache[chunk] = ids = tuple(ids)
        return ids

    def _encode_plain(self, text, out):
        for chunk in CHUNK_RE.findall(text):
            out.extend(self._encode_chunk(chunk))

    def encode(self, text, add_bos=False, add_eos=False, allow_special=True):
        """Đổi văn bản thành danh sách ID.

        allow_special=False: các chuỗi như "<think>" trong văn bản bị coi là chữ
        thường. Dùng cho văn bản do người dùng nhập để họ không chèn được token
        điều khiển vào prompt.
        """
        out = [self.special_to_id["<bos>"]] if add_bos else []
        if allow_special:
            for part in self._special_re.split(text):
                if part in self.special_to_id:
                    out.append(self.special_to_id[part])
                elif part:
                    self._encode_plain(part, out)
        else:
            self._encode_plain(text, out)
        if add_eos:
            out.append(self.special_to_id["<eos>"])
        return out

    # ----- giải mã -----
    def decode(self, ids, skip_special=False):
        parts = []
        for i in ids:
            if i in self.id_to_special:
                if not skip_special:
                    parts.append(self.id_to_special[i].encode("utf-8"))
            else:
                try:
                    parts.append(self.vocab[i])
                except KeyError:
                    raise ValueError(f"ID {i} nằm ngoài vocab (kích thước {self.vocab_size})")
        return b"".join(parts).decode("utf-8", errors="replace")

    # ----- lưu và nạp -----
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "special_tokens": self.special_tokens,
                       "chunk_pattern": CHUNK_RE.pattern,
                       "merges": [list(p) for p in self.merges]}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("chunk_pattern") != CHUNK_RE.pattern:
            raise ValueError("File tokenizer được tạo với quy tắc cắt mẩu khác bản code hiện tại")
        return cls(data["merges"], data["special_tokens"])
