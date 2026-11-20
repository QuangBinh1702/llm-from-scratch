"""Hàm chọn và ghép dữ liệu, dùng chung cho các script."""


def _flag(flags, i):
    """Lấy phần tử thứ i của một danh sách cờ, chịu được None và danh sách ngắn."""
    if flags is None or i >= len(flags):
        return None
    return flags[i]


def pick_generation(row):
    """Chọn lời giải NGẮN NHẤT trong số các lời giải đúng và đầy đủ của một bài Open-R1.

    Trả về (lời_giải, None) nếu chọn được, hoặc (None, lý_do_loại).
    """
    gens = row.get("generations") or []
    if not gens:
        return None, "khong_co_loi_giai"
    best = None
    seen_correct = False
    for i, g in enumerate(gens):
        if not g:
            continue
        if _flag(row.get("is_reasoning_complete"), i) is False:
            continue                                   # lời giải bị cắt giữa chừng
        ok = (_flag(row.get("correctness_math_verify"), i) is True
              or _flag(row.get("correctness_llama"), i) is True)
        if not ok:
            continue
        seen_correct = True
        if "<think>" not in g or "</think>" not in g:
            continue                                   # thiếu định dạng chuỗi tư duy
        if best is None or len(g) < len(best):
            best = g
    if best is not None:
        return best, None
    return None, ("thieu_the_think" if seen_correct else "khong_co_loi_giai_dung")


def openr1_text(row, generation):
    """Một văn bản Open-R1: đề bài, xuống dòng, lời giải (đã có sẵn <think>...</think>)."""
    return row["problem"].strip() + "\n" + generation.strip()


def gsm8k_text(row):
    """Một văn bản GSM8K thô: câu hỏi, xuống dòng, lời giải kèm đáp án."""
    return row["question"].strip() + "\n" + row["answer"].strip()
