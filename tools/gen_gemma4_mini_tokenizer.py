#!/usr/bin/env python3
"""Generates the miniature tokenizer of Gemma 4's shape and its goldens (2026-10-04):

  tests/data/gemma4_mini_tokenizer.json           a tokenizer.json with the release's exact pipeline
                                                  (normalizer, pre-tokenizer, decoder, post-processor,
                                                  BPE flags) over a vocabulary of a few hundred tokens
  tests/data/gemma4_mini_tokenizer_goldens.jsonl  what HF tokenizers makes of a case list with it

so the engine's SentencePiece-style BPE (src/text/tokenizer_spm.cpp) has a differential test that
needs no checkpoint (tests/unit/gemma4_tokenizer_test.cpp). The real tokenizer.json has its own
corpus (tools/gen_tokenizer_goldens.py --model nvidia/Gemma-4-31B-IT-NVFP4).

The vocabulary is built to exercise what the real one does: the 256 "<0xNN>" byte tokens (a
character outside the vocabulary falls back to them), runs of the space mark and of newlines that
are tokens of their own, added tokens inside the base id range with one a prefix of another, and
a merge list that makes the same token in more than one way ("aaa" from "aa" + "a" and from
"a" + "aa") in an order that makes HF's queue walk and a naive loop disagree if either is wrong.

Needs `tokenizers` (the golden source). Regenerating: python3 tools/gen_gemma4_mini_tokenizer.py
"""
import json
import os
import sys

SP = "▁"

ADDED = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<|turn>", "<turn|>", '<|"|>', "<|tool>", "<|tool_call>",
         "<tool_call|>"]

SINGLE = [SP, "\n", "\t"] + list("abcdefghijklmnopqrstuvwxyz") + list("ABCDE") + list("0123456789") + list(
    ".,:!?<>|_-{}\"'/=") + ["é", "你", "好", "ß"]

# In rank order. Every product becomes a vocabulary entry (once).
MERGES = [
    (SP, SP), (SP + SP, SP + SP), (SP + SP + SP + SP, SP + SP + SP + SP), (SP + SP, SP),
    ("\n", "\n"), ("\n\n", "\n"), ("\n\n", "\n\n"), ("\n", "\n\n"),
    ("t", "h"), ("th", "e"), (SP, "the"), ("i", "n"), (SP, "in"), ("e", "r"), ("h", "e"), ("he", "r"), ("r", "e"),
    (SP + "the", "re"), ("a", "a"), ("aa", "a"), ("a", "aa"), ("aa", "aa"), ("a", "b"), ("ab", "ab"), ("b", "a"),
    ("ab", "a"), ("i", "s"), (SP, "is"), (SP, "a"), ("o", "f"), (SP, "of"), ("u", "s"), ("us", "er"), ("m", "o"),
    ("d", "e"), ("mo", "de"), ("mode", "l"), ("1", "2"), ("12", "3"), ("2", "3"), (".", "."), ("..", "."),
    ("c", "a"), ("ca", "l"), ("cal", "l"), ("call", ":"), (SP, "你"), ("你", "好"), ("é", "e"), ("<", "|"),
    ("|", ">"), ("t", "u"), ("tu", "r"), ("tur", "n"),
]

CASES = [
    "", " ", "  ", "   ", "    ", "     ", " " * 8, " " * 9, " " * 17, "\n", "\n\n", "\n\n\n", "\n\n\n\n", "\n" * 5,
    "\n" * 9, " \n \n", "\t\t", "the", " the", "the the", " there", "there", "then", "in the there", " in in",
    "a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa", "a" * 16, "a" * 33, "ab", "abab", "ababab", "aba", "ababa",
    "abaab", "baab", "aab", "aabaab", "is a model of the user", "user\nmodel\n", "123", "1234", "12 23 123", "...",
    "....", ".....", "call:abc{a:1}", "你好", " 你好", "你 好", "étée", "ß", "é",
    # Characters outside the vocabulary: their UTF-8 bytes.
    "z", "Z", "ü", "€", "日本", "👍", "a\u0000b", "\u007f", "x y", "FGH", "+", "a+b",
    # A literal space mark is a space.
    SP, "a" + SP + "b", SP + SP,
    # Added tokens: standalone, adjacent, inline, prefix-of-another, and near misses.
    "<bos>", "<eos>", "<pad>", "<unk>", "<mask>", "<|turn>", "<turn|>", '<|"|>', "<|tool>", "<|tool_call>",
    "<tool_call|>", "<|tool><|tool_call>", "<|tool_call><|tool>", "<bos><|turn>user\nthe<turn|>\n<|turn>model\n",
    "a<|turn>b", "a <turn|> b", '<|"|>the<|"|>', "<|tool_call>call:ab{a:<|\"|>b<|\"|>}<tool_call|>",
    "<|turn", "turn|>", "<|tool_cal>", "<|tool_", "<| turn>", "<bos", "<<bos>>", "<|>", "<|\"|", "<|tur<|turn>",
    "<0x41>", "<0xE4>", "the <0x00> in",
    "the user is in the model there\n\n    aaaa abab 123 ... 你好 étée\n<|turn>model\n",
]


def build():
    vocab = {}
    for t in ADDED:
        vocab[t] = len(vocab)
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = len(vocab)
    for t in SINGLE:
        assert t not in vocab, t
        vocab[t] = len(vocab)
    for a, b in MERGES:
        assert a in vocab and b in vocab, (a, b)
        if a + b not in vocab:
            vocab[a + b] = len(vocab)
    added = [{"id": vocab[t], "content": t, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": True} for t in ADDED]
    return {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": SP},
        "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": False},
        "post_processor": {"type": "TemplateProcessing", "single": [{"Sequence": {"id": "A", "type_id": 0}}],
                           "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
                           "special_tokens": {}},
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": SP}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"}]},
        "model": {"type": "BPE", "dropout": None, "unk_token": "<unk>", "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": True, "byte_fallback": True, "ignore_merges": False,
                  "vocab": vocab, "merges": [[a, b] for a, b in MERGES]},
    }


def fnv1a64(data):
    h = 1469598103934665603
    for b in data:
        h = (h ^ b) * 1099511628211 & 0xFFFFFFFFFFFFFFFF
    return h


def main():
    import tokenizers
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "data")
    tok_path = os.path.join(root, "gemma4_mini_tokenizer.json")
    out_path = os.path.join(root, "gemma4_mini_tokenizer_goldens.jsonl")
    raw = json.dumps(build(), ensure_ascii=False, indent=0).encode()
    with open(tok_path, "wb") as f:
        f.write(raw)
    tok = tokenizers.Tokenizer.from_file(tok_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"model": "gemma4-mini", "tokenizers": tokenizers.__version__,
                            "revision_hash": f"{fnv1a64(raw):016x}", "cases": len(CASES)}) + "\n")
        for text in CASES:
            ids = tok.encode(text).ids
            decoded = tok.decode(ids, skip_special_tokens=False)
            if decoded != text.replace(SP, " "):
                sys.exit(f"HF verbatim round-trip failed for {text!r} -> {decoded!r}")
            f.write(json.dumps({"text": text, "ids": ids, "skipped": tok.decode(ids, skip_special_tokens=True)}) + "\n")
    print(f"wrote {tok_path} ({len(raw)} bytes, vocab {tok.get_vocab_size()}) and {out_path}: {len(CASES)} cases "
          f"(tokenizers {tokenizers.__version__})")


if __name__ == "__main__":
    main()
