import json, tempfile
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import count_data as c
from llm.data import pick_generation

T = "<think>abc</think> \\boxed{1}"
long_ok = "<think>" + "x" * 500 + "</think> 2"
rows = [
    # hai lời giải đúng -> chọn cái ngắn hơn
    {"problem": "p1", "generations": [long_ok, T], "is_reasoning_complete": [True, True],
     "correctness_math_verify": [True, True], "correctness_llama": None, "answer": "1"},
    # lời giải ngắn bị sai -> phải chọn cái dài
    {"problem": "p2", "generations": [T, long_ok], "is_reasoning_complete": [True, True],
     "correctness_math_verify": [False, True], "correctness_llama": None, "answer": "2"},
    # math_verify None nhưng llama True -> nhận
    {"problem": "p3", "generations": [T], "is_reasoning_complete": [True],
     "correctness_math_verify": [None], "correctness_llama": [True], "answer": "1"},
    # lời giải bị cắt -> loại
    {"problem": "p4", "generations": [T], "is_reasoning_complete": [False],
     "correctness_math_verify": [True], "correctness_llama": None, "answer": "1"},
    # không có lời giải đúng -> loại
    {"problem": "p5", "generations": [T], "is_reasoning_complete": [True],
     "correctness_math_verify": [False], "correctness_llama": [False], "answer": "1"},
    # đúng nhưng thiếu thẻ think -> loại
    {"problem": "p6", "generations": ["42"], "is_reasoning_complete": [True],
     "correctness_math_verify": [True], "correctness_llama": None, "answer": "42"},
    # không có generations
    {"problem": "p7", "generations": [], "is_reasoning_complete": [],
     "correctness_math_verify": [], "correctness_llama": None, "answer": "1"},
    # danh sách cờ ngắn hơn danh sách lời giải
    {"problem": "p8", "generations": [T, T], "is_reasoning_complete": [True],
     "correctness_math_verify": [True], "correctness_llama": None, "answer": "1"},
]
assert pick_generation(rows[0])[0] == T
assert pick_generation(rows[1])[0] == long_ok
assert pick_generation(rows[2])[0] == T
assert pick_generation(rows[3]) == (None, "khong_co_loi_giai_dung")
assert pick_generation(rows[4]) == (None, "khong_co_loi_giai_dung")
assert pick_generation(rows[5]) == (None, "thieu_the_think")
assert pick_generation(rows[6]) == (None, "khong_co_loi_giai")
assert pick_generation(rows[7])[0] == T

out = os.path.join(tempfile.mkdtemp(), "o.jsonl")
lengths, reasons, n = c.scan_openr1(iter(rows), out_path=out, max_rows=None)
assert n == 8 and len(lengths) == 4, (n, lengths)
assert sum(reasons.values()) == 4
saved = [json.loads(l) for l in open(out, encoding="utf-8")]
assert len(saved) == 4 and all(s["chars"] == len(s["problem"]) + len(s["generation"]) for s in saved)
s = c.summarize(lengths, [10, 100, 1000])
assert [r["so_mau"] for r in s] == [0, 3, 4], s
assert c.percentile([1, 2, 3, 4, 5], 50) == 3 and c.percentile([], 50) == 0
# max_rows dừng đúng chỗ
assert c.scan_openr1(iter(rows), out_path=out, max_rows=3)[2] == 3
c.print_report("thử", lengths, [10, 100, 1000], n)
print("TẤT CẢ KIỂM THỬ ĐẠT")
