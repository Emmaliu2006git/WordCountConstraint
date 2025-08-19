import re
from langdetect import detect
#1
def check_letter_frequency(text: str, v: dict) -> bool:
    # v: {"relation": "gte"|"eq", "letter": "a", "target": 80}
    count = text.count(v["letter"])
    if v["relation"] == "gte":
        return count >= int(v["target"])
    return count == int(v["target"])

#2
def check_keyword_frequency(text: str, v: dict) -> bool:
    # v: {"relation": "gte"|"eq", "phrase": "policy change", "target": 5}
    phrase = re.escape(v["phrase"])
    count = len(re.findall(rf"\b{phrase}\b", text))
    if v["relation"] == "gte":
        return count >= int(v["target"])
    return count == int(v["target"])

#3
def check_placeholder_count(text: str, v: dict) -> bool:
    # v: {"token": "[CITATION]", "target": 3}
    token = re.escape(v["token"])
    # match either plain [TOKEN] or bold **[TOKEN]**
    pattern = rf"(\*\*{token}\*\*|{token})"
    count = len(re.findall(pattern, text))
    return count == int(v["target"])

#4
def check_bullet_point_count(text: str, v: dict) -> bool:
    # v: {"target": 6}
    pattern = r"^\s*([-*•]|\d+[.)])\s+"
    lines = text.splitlines()
    matches = []

    for i, line in enumerate(lines):
        m = re.match(pattern, line)
        if m:
            matches.append((i, m.group(1)))  # (line_index, symbol)

    if len(matches) != int(v["target"]):
        return False

    # check consecutive line numbers
    line_indices = [idx for idx, _ in matches]
    if line_indices != list(range(line_indices[0], line_indices[0] + len(matches))):
        return False

    # normalize numbered lists
    symbols = ["num" if re.match(r"\d+[.)]", s) else s for _, s in matches]
    return all(sym == symbols[0] for sym in symbols)

#5
def check_required_keywords(text: str, v: dict) -> bool:
    # v: {"tokens": ["keyword1", "keyword2", ...]}
    for kw in v["tokens"]:
        if re.search(rf"\b{re.escape(kw)}\b", text) is None:
            return False
    return True

#6
def check_forbidden_words(text: str, v: dict) -> bool:
    # v: {"tokens": ["badword1", "badword2", ...]}
    for kw in v["tokens"]:
        if re.search(rf"\b{re.escape(kw)}\b", text):
            return False
    return True

#7
def check_must_mention_placeholders(text: str, v: dict) -> bool:
    # v: {"tokens": ["[CITATION]", "[DATA]"]}
    for ph in v["tokens"]:
        token = re.escape(ph)
        pattern = rf"(\*\*{token}\*\*|{token})"
        if re.search(pattern, text) is None:
            return False
    return True

#8
def check_paragraph_first_word(text: str, v: dict) -> bool:
    # v: {"options": ["However", "In conclusion", "A"]}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    options = [re.escape(opt) for opt in v["options"]]

    for p in paragraphs:
        if not any(re.match(rf"^{opt}\b", p) for opt in options):
            return False
    return True

#9
def check_section_progression(text: str, v: dict) -> bool:
    # v: {"section_number": 4}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return len(paragraphs) == int(v["section_number"])

#10
def check_list_structure(text: str, v: dict) -> bool:
    # v: {"items": 3, "subpoints_per_item": 2}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    n_items = int(v["items"])
    n_subs = int(v["subpoints_per_item"])

    # regex patterns
    item_pattern = re.compile(r"^\d+[.)]\s+")
    sub_pattern = re.compile(r"^[-*•]\s+")

    # scan through lines to find a candidate block
    for i in range(len(lines)):
        block_items = []
        j = i
        while j < len(lines) and item_pattern.match(lines[j]):
            block_items.append(j)
            j += 1
            # check subpoints for this item
            sub_count = 0
            while j < len(lines) and sub_pattern.match(lines[j]):
                sub_count += 1
                j += 1
            if sub_count != n_subs:
                break
        # check if a valid block was found
        if len(block_items) == n_items and all(
            len([1 for k in range(block_items[idx] + 1, block_items[idx + 1] if idx + 1 < len(block_items) else j) if sub_pattern.match(lines[k])]) == n_subs
            for idx in range(len(block_items))
        ):
            return True
    return False

#11
def check_multiple_responses(text: str, v: dict) -> bool:
    # v: {"target": 2}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return len(paragraphs) == int(v["target"])

#12
def check_title_placement(text: str, v: dict) -> bool:
    # v: {"case_rule": "uppercase" | "lowercase"}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    title = lines[0]

    if v["case_rule"] == "uppercase":
        return title.isupper()
    elif v["case_rule"] == "lowercase":
        return title.islower()
    return False

#13
def check_case_enforcement(text: str, v: dict) -> bool:
    # v: {"case_rule": "uppercase" | "lowercase"}
    if v["case_rule"] == "uppercase":
        return text.isupper()
    elif v["case_rule"] == "lowercase":
        return text.islower()
    return False

#14
def check_language_enforcement(text: str, v: dict) -> bool:
    # v: {"target": "English" | "Chinese" | "German" | "French" | "Japanese" | "Korean" | "Spanish"}
    try:
        lang = detect(text)
    except:
        return False

    mapping = {
        "English": "en",
        "Chinese": "zh-cn",
        "German": "de",
        "French": "fr",
        "Japanese": "ja",
        "Korean": "ko",
        "Spanish": "es",
    }

    target = v["target"]
    if target not in mapping:
        raise ValueError(f"Unsupported target language: {target}")
    return lang.startswith(mapping[target])

#15
def check_punctuation_restrictions(text: str, v: dict) -> bool:
    # v: {"tokens": ";"}
    forbidden = v["tokens"]
    return forbidden not in text

#16
def check_quotation_wrapping(text: str, v: dict) -> bool:
    # v: {"mark": "single" | "double"}
    if v["mark"] == "single":
        return "'" in text and '"' not in text
    elif v["mark"] == "double":
        return '"' in text and "'" not in text
    return False


