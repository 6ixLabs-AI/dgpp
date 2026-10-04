#!/usr/bin/env python3
"""gen_prompts.py — write the fixed prompt sets capture_refset.py replays (2026-10-03).

  prompts_short.jsonl   echo / completion / chat / tools rows, all under ~600 tokens
  prompts_long.jsonl    long-context retrieval: a filler document with planted facts, asked for
                        at ~8k / 32k / 128k / 235k prompt tokens

Deterministic: the same file comes out on every run (a fixed seed, no clock, no environment), so
the Atlas capture and a later DGPP capture are answers to byte-identical prompts.
"""
import json, random

ECHO = {
 "prose_en": "The lighthouse keeper climbed the spiral stairs every evening at dusk. He trimmed the "
   "wick, polished the great lens, and wound the clockwork that turned the lamp. On clear nights "
   "the beam reached ships twenty miles out; in fog he rang the bell by hand until morning. He "
   "had done this for thirty-one years, and he had never once let the light go dark.",
 "code_py": "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n"
   "    out, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n"
   "            out.append(a[i])\n            i += 1\n        else:\n            out.append(b[j])\n"
   "            j += 1\n    out.extend(a[i:])\n    out.extend(b[j:])\n    return out\n\n\n"
   "print(merge_sorted([1, 4, 9], [2, 3, 10, 11]))\n",
 "code_rust": "use std::collections::HashMap;\n\nfn word_count(text: &str) -> HashMap<String, usize> {\n"
   "    let mut counts = HashMap::new();\n    for word in text.split_whitespace() {\n"
   "        *counts.entry(word.to_lowercase()).or_insert(0) += 1;\n    }\n    counts\n}\n\n"
   "fn main() {\n    let counts = word_count(\"the quick brown fox jumps over the lazy dog the end\");\n"
   "    println!(\"{:?}\", counts.get(\"the\"));\n}\n",
 "math": "To solve 3x + 7 = 31, subtract 7 from both sides to get 3x = 24, then divide both sides by "
   "3 to get x = 8. Checking: 3 times 8 is 24, and 24 plus 7 is 31, so the solution is correct. "
   "The sum of the first ten positive integers is 55, and the sum of their squares is 385.",
 "json": "{\"order_id\": 48213, \"customer\": {\"name\": \"Dana Whitfield\", \"tier\": \"gold\"}, "
   "\"items\": [{\"sku\": \"KB-204\", \"qty\": 2, \"price\": 49.5}, {\"sku\": \"MS-110\", \"qty\": 1, "
   "\"price\": 24.0}], \"total\": 123.0, \"shipped\": false}",
 "zh": "春天来了，河边的柳树发出了新芽。孩子们放学以后在田野里放风筝，老人坐在门口晒太阳，"
   "谈论今年的收成。村子不大，只有几十户人家，但是每到傍晚，炊烟升起，整个山谷都显得格外安静。",
 "chat_raw": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
   "What is the boiling point of water at sea level in Celsius?<|im_end|>\n<|im_start|>assistant\n"
   "Water boils at 100 degrees Celsius at sea level.<|im_end|>\n",
 "tool_raw": "<|im_start|>user\nWhat's the weather in Toronto?<|im_end|>\n<|im_start|>assistant\n"
   "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Toronto\"}}\n</tool_call><|im_end|>\n"
   "<|im_start|>user\n<tool_response>\n{\"temp_c\": 14, \"sky\": \"overcast\"}\n</tool_response><|im_end|>\n"
   "<|im_start|>assistant\nIt is 14 degrees and overcast in Toronto right now.<|im_end|>\n",
}

COMPLETION = {
 "c_fib": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
 "c_planets": "The planets of the solar system, in order from the Sun, are Mercury,",
 "c_story": "Once upon a time, in a village at the edge of a great forest, there lived",
 "c_sql": "-- Return the ten customers with the highest total order value\nSELECT",
 "c_count": "1, 2, 3, 4, 5, 6, 7, 8,",
 "c_primes": "The first ten prime numbers are 2, 3, 5,",
}

CHAT = {
 "q_capital": "What is the capital of Australia? Answer in one sentence.",
 "q_sort": "Write a Python function that returns the second largest number in a list. Code only.",
 "q_explain": "Explain in three sentences why the sky is blue.",
 "q_arith": "What is 17 times 23? Show the working briefly.",
 "q_list": "List five fruits, one per line, nothing else.",
 "q_translate": "Translate into French: 'The train leaves at half past seven tomorrow morning.'",
 "q_json": "Return a JSON object with keys name and age for a 31-year-old called Priya. JSON only.",
 "q_logic": "Anna is taller than Ben. Ben is taller than Cara. Who is the shortest? One word.",
}

TOOLS = [
 {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
  "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                 "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}},
 {"type": "function", "function": {"name": "read_file", "description": "Read a file from disk",
  "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                 "max_lines": {"type": "integer"}}, "required": ["path"]}}},
 {"type": "function", "function": {"name": "run_sql", "description": "Run a read-only SQL query",
  "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]
TOOL_Q = {
 "t_weather": "What's the weather in Montreal in celsius?",
 "t_file": "Show me the first 20 lines of /etc/hosts.",
 "t_sql": "How many rows are in the orders table?",
 "t_none": "What is two plus two? Do not use a tool.",
}

# Long context: filler sentences built from word lists, with one planted "vault code" every N lines.
SUBJ = ["The surveyor", "A night clerk", "The ferry captain", "An apprentice", "The archivist",
        "A beekeeper", "The signalman", "A cartographer", "The quartermaster", "A glassblower"]
VERB = ["recorded", "repaired", "counted", "delivered", "measured", "catalogued", "inspected", "sealed"]
OBJ = ["the brass fittings", "nine crates of lamp oil", "the tide tables", "a ledger of arrivals",
       "the eastern stairwell", "forty lengths of rope", "the harbour bell", "a box of wax seals"]
WHEN = ["before dawn", "at midday", "during the storm", "on the third day", "after the market closed",
        "while the fog held", "at the turn of the tide", "by lantern light"]


def long_doc(target_tokens, seed):
    """~18.7 tokens a line (measured on Atlas /tokenize: 615 lines = 11,489 prompt tokens); five
    planted facts at fixed depths; returns (text, [(name, code)])."""
    rng = random.Random(seed)
    lines, facts = [], []
    n_lines = int(target_tokens / 18.7)
    names = ["amber", "cobalt", "garnet", "indigo", "juniper"]
    slots = {int(n_lines * f): nm for f, nm in zip((0.05, 0.27, 0.5, 0.73, 0.95), names)}
    for i in range(n_lines):
        if i in slots:
            code = f"{rng.randrange(1000, 9999)}-{rng.randrange(100, 999)}"
            facts.append((slots[i], code))
            lines.append(f"Entry {i}: the vault code for the {slots[i]} room is {code}.")
        else:
            lines.append(f"Entry {i}: {rng.choice(SUBJ)} {rng.choice(VERB)} {rng.choice(OBJ)} {rng.choice(WHEN)}.")
    return "\n".join(lines), facts


def main():
    short = []
    for k, v in ECHO.items():
        short.append({"id": "echo_" + k, "kind": "echo", "text": v})
    for k, v in COMPLETION.items():
        short.append({"id": k, "kind": "completion", "prompt": v, "max_tokens": 96})
    for k, v in CHAT.items():
        short.append({"id": k, "kind": "chat", "messages": [{"role": "user", "content": v}], "max_tokens": 200})
    for k, v in TOOL_Q.items():
        short.append({"id": k, "kind": "tools", "messages": [{"role": "user", "content": v}],
                      "tools": TOOLS, "max_tokens": 200})
    with open("prompts_short.jsonl", "w") as f:
        for r in short:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    longs = []
    # 235k is its own file: Atlas on one box (util 0.85) ran out of K/V blocks at ~193k tokens
    # (2026-10-03, "KV cache exhausted: no free blocks"), so the Atlas capture stops at 176k.
    sizes = (("8k", 8000), ("32k", 32000), ("128k", 128000), ("176k", 176000), ("235k", 235000))
    for label, toks in sizes:
        doc, facts = long_doc(toks, seed=toks)
        q = ("Below is a log. Read it, then answer the question after it.\n\n" + doc +
             "\n\nQuestion: list the vault code for each of these rooms, one per line, as "
             "'room: code': " + ", ".join(n for n, _ in facts) + ".")
        longs.append({"id": "long_" + label, "kind": "chat", "messages": [{"role": "user", "content": q}],
                      "max_tokens": 160, "expect": [c for _, c in facts]})
    with open("prompts_long.jsonl", "w") as f:
        for r in longs[:-1]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open("prompts_long_235k.jsonl", "w") as f:
        f.write(json.dumps(longs[-1], ensure_ascii=False) + "\n")
    print(f"prompts_short.jsonl: {len(short)} rows; prompts_long.jsonl: {len(longs) - 1} rows; "
          f"prompts_long_235k.jsonl: 1 row")


if __name__ == "__main__":
    main()
