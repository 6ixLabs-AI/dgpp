#include "text/tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "common/log.hpp"
#include "loaders/minijson.hpp"
#include "text/unicode_normalize.hpp"
#include "text/unicode_tables.hpp"

namespace dgpp::text {
namespace {

// The pinned Split patterns (tokenizer.json pre_tokenizer[0]). The scanner
// below hardcodes exactly these regexes; any other pattern is refused at
// load.
constexpr const char* kSplitPattern =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}"
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
// The Qwen3.8-Flash-Next pattern: letter runs admit marks, numbers split
// one per pretoken, marks stay out of the punctuation class.
constexpr const char* kSplitPatternQwen =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}"
    "| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
// The Qwen2 pattern (MiMo-V2.6-Flash, 2026-09-22): the GLM classes (letter
// runs without marks, marks in the punctuation class) with numbers one per
// pretoken as the Qwen3.8 pattern splits them.
constexpr const char* kSplitPatternQwen2 =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}"
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
// The DeepSeek-V4.1 pre-tokenizer: THREE Split stages (each Isolated,
// applied to the previous stage's pieces) then ByteLevel. Stage 1 cuts
// number runs into pieces of at most three; stage 2 isolates CJK runs
// (the three literal ranges); stage 3's alternatives, leftmost-first:
//   B1 [ASCII punct][A-Za-z]+           one ASCII punctuation/symbol char + ASCII letters
//   B2 [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+   optional prefix (whitespace, format
//                                      or control chars — numbers never
//                                      reach this stage) + a letter/mark run
//   B3  ?[\p{P}\p{S}]+[\r\n]*           optional SPACE + a punctuation/symbol
//                                      run + trailing CR/LFs
//   B4-B6 the whitespace alternatives of the GLM pattern (A5-A7)
// Text no alternative matches (format and control characters outside a
// letter run's prefix) stays a piece of its own: Split's Isolated
// behaviour keeps the gaps between matches.
constexpr const char* kSplitPatternDsv41Numbers = "\\p{N}{1,3}";
// The three literal CJK ranges (raw codepoints in the file, decoded by the JSON reader).
constexpr const char* kSplitPatternDsv41Cjk = "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+";
// CR and LF stand in the file as the control characters themselves.
constexpr const char* kSplitPatternDsv41Main =
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";
// The Mistral-Small-4 pattern (2026-10-04; tekken.json's config.pattern,
// carried verbatim into the checkpoint's tokenizer.json): words split on
// letter case — two word alternatives over the "upper" class
// [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] and the "lower" class
// [\p{Ll}\p{Lm}\p{Lo}\p{M}] — no contraction alternative, numbers one per
// pretoken, marks in the punctuation class, and '/' beside CR/LF in the
// punctuation run's tail. The cased-word scanner below spells out the
// alternatives.
constexpr const char* kSplitPatternTekken =
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+"
    "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*"
    "|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
// The MiniMax-M2.7 pattern (2026-10-04; OpenAI's o200k_base regex): the
// tekken pattern with an optional contraction closing either word
// alternative and number runs of at most three.
constexpr const char* kSplitPatternO200k =
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+"
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*"
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

[[noreturn]] void reject(const std::string& what) {
  throw std::runtime_error("glm_tokenizer: " + what);
}

// Null-safe field access for the shape validator: a missing field is a
// different tokenizer, not a default to guess.
const minijson::Value& field(const minijson::Value& v, const char* name,
                            const char* where) {
  const minijson::Value* f = v.find(name);
  if (!f) reject(std::string(where) + ": missing field '" + name + "'");
  return *f;
}

std::string_view want_string(const minijson::Value& v, const char* name,
                             const char* where) {
  const minijson::Value& f = field(v, name, where);
  if (!f.is_string()) reject(std::string(where) + ": '" + name + "' not a string");
  return f.as_string();
}

bool want_bool(const minijson::Value& v, const char* name, const char* where) {
  const minijson::Value& f = field(v, name, where);
  if (f.kind() != minijson::Value::Kind::Bool)
    reject(std::string(where) + ": '" + name + "' not a bool");
  return f.as_bool();
}

uint64_t fnv1a_bytes(const char* data, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) {
    h ^= static_cast<unsigned char>(data[i]);
    h *= 1099511628211ull;
  }
  return h;
}

std::string read_file(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) reject("cannot open " + path);
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::string buf(static_cast<size_t>(n), '\0');
  if (n > 0 && std::fread(buf.data(), 1, static_cast<size_t>(n), f) !=
                  static_cast<size_t>(n)) {
    std::fclose(f);
    reject("short read on " + path);
  }
  std::fclose(f);
  return buf;
}

// GPT-2's bytes_to_unicode: printable ASCII and the 0xA1-0xFF stretch map
// to themselves; every other byte maps to 256+n over the remaining
// codepoints. All mapped codepoints stay below 0x200.
void build_byte_level_maps(uint32_t* byte_to_cp, int32_t* cp_to_byte) {
  std::fill(cp_to_byte, cp_to_byte + 0x200, -1);
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    const bool direct = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                        (b >= 0xAE && b <= 0xFF);
    const uint32_t cp = direct ? static_cast<uint32_t>(b)
                               : static_cast<uint32_t>(256 + n++);
    byte_to_cp[b] = cp;
    cp_to_byte[cp] = b;
  }
}

// ---------------------------------------------------------------------------
// The Split scanner. Operates on CODEPOINTS (the regex is Unicode-class
// based) over one added-token-free segment. Successive matches emit the
// pretokens (the pattern covers every codepoint, so matches are
// gap-free; the defensive throw pins a scanner hole loudly instead of
// silently re-syncing).
//
// Alternation is leftmost-first (Rust regex semantics) — the scanner tries
// the alternatives in the pattern's order, with the greedy/backtracking
// outcomes each quantifier implies:
//   A1 (?i:'s|'t|'re|'ve|'m|'ll|'d)   apostrophe + case-insens form
//   A2 [^\r\n\p{L}\p{N}]?\p{L}+      optional non-CR/LF/letter/number
//                                    prefix + a letter run (also the
//                                    plain letter run when the prefix is
//                                    absent)
//   A3 \p{N}{1,3}                    up to 3 numbers
//   A4  ?[^\s\p{L}\p{N}]+[\r\n]*     optional literal SPACE + a punctuation
//                                    run + trailing CR/LFs
//   A5 \s*[\r\n]+                    whitespace through the run's last CR/LF
//   A6 \s+(?!\S)                     whitespace minus its last char when a
//                                    non-space follows (that char joins the
//                                    next match — the classic " a" space
//                                    prefix); the full run at end of text
//   A7 \s+                           the remaining whitespace run
// ---------------------------------------------------------------------------
// `qwen` selects the Qwen3.8 pattern's three differences (A2 admits marks
// in the run, A3 is one number, A4's class excludes marks).
class SplitScanner {
 public:
  // marks: the letter-run class admits \p{M} and the punctuation class
  // excludes it (the Qwen3.8 pattern); single_number: one \p{N} per
  // pretoken (Qwen3.8, Qwen2) instead of runs of up to three (GLM).
  SplitScanner(std::string_view text, std::vector<std::string_view>* out, bool marks, bool single_number)
      : text_(text), out_(out), marks_(marks), single_number_(single_number) {
    // Decode codepoint boundaries once. Strict UTF-8: a chat prompt
    // arrives as valid UTF-8; anything else is a caller bug, not a
    // tokenization ambiguity.
    size_t i = 0;
    while (i < text_.size()) {
      const int len = utf8_len(text_[i]);
      if (len < 1 || i + static_cast<size_t>(len) > text_.size())
        reject("input is not valid UTF-8");
      for (int k = 1; k < len; ++k) {
        const unsigned char cont =
            static_cast<unsigned char>(text_[i + k]);
        if ((cont & 0xC0) != 0x80)
          reject("input is not valid UTF-8 (bad continuation byte)");
      }
      uint32_t raw = 0;
      for (int k = 0; k < len; ++k)
        raw = (raw << 8) | static_cast<unsigned char>(text_[i + k]);
      cps_.push_back(decode_utf8(raw, len));
      starts_.push_back(i);
      i += static_cast<size_t>(len);
    }
    starts_.push_back(text_.size());
  }

  void run() {
    size_t i = 0;  // codepoint index
    while (i < cps_.size()) {
      const size_t end = match_at(i);
      if (end == i)
        reject("scanner hole at codepoint " + std::to_string(i));
      out_->push_back(text_.substr(starts_[i], starts_[end] - starts_[i]));
      i = end;
    }
  }

 private:
  static int utf8_len(char c) {
    const unsigned char b = static_cast<unsigned char>(c);
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return -1;
  }
  static uint32_t decode_utf8(uint32_t raw, int len) {
    switch (len) {
      case 1: return raw & 0x7F;
      case 2: return ((raw >> 8) & 0x1F) << 6 | (raw & 0x3F);
      case 3: return ((raw >> 16) & 0x0F) << 12 | ((raw >> 8) & 0x3F) << 6 |
                   (raw & 0x3F);
      default: return ((raw >> 24) & 0x07) << 18 | ((raw >> 16) & 0x3F) << 12 |
                    ((raw >> 8) & 0x3F) << 6 | (raw & 0x3F);
    }
  }
  static uint32_t lower_ascii(uint32_t cp) {
    return (cp >= 'A' && cp <= 'Z') ? cp + 32 : cp;
  }
  bool letter(size_t i) const { return unicode::is_letter(cps_[i]); }
  bool mark(size_t i) const { return marks_ && unicode::is_mark(cps_[i]); }
  // The letter-run class: \p{L} (GLM) or [\p{L}\p{M}] (Qwen).
  bool run_char(size_t i) const { return letter(i) || mark(i); }
  bool number(size_t i) const { return unicode::is_number(cps_[i]); }
  bool ws(size_t i) const { return unicode::is_white_space(cps_[i]); }
  bool punct(size_t i) const { return !ws(i) && !letter(i) && !number(i) && !mark(i); }
  bool eos(size_t i) const { return i >= cps_.size(); }
  uint32_t cp(size_t i) const { return eos(i) ? 0 : cps_[i]; }

  // A1: apostrophe + case-insensitive s|t|m|d (1) or re|ve|ll (2).
  bool match_contraction(size_t i, size_t* end) const {
    if (cp(i) != '\'') return false;
    const uint32_t a = lower_ascii(cp(i + 1));
    if (a == 's' || a == 't' || a == 'm' || a == 'd') {
      *end = i + 2;
      return true;
    }
    const uint32_t b = lower_ascii(cp(i + 2));
    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
        (a == 'l' && b == 'l')) {
      *end = i + 3;
      return true;
    }
    return false;
  }

  // A5: \s*[\r\n]+ — through the whitespace run's LAST CR/LF.
  bool match_newline_run(size_t i, size_t* end) const {
    size_t j = i;
    while (!eos(j) && ws(j)) ++j;
    size_t last_nl = i;
    for (size_t k = i; k < j; ++k)
      if (cps_[k] == '\r' || cps_[k] == '\n') last_nl = k + 1;
    if (last_nl == i) return false;
    *end = last_nl;
    return true;
  }

  // A6: \s+(?!\S) — the whole run at end of text; one char short when a
  // non-space follows (needs a run of >= 2 then).
  bool match_trailing_ws(size_t i, size_t* end) const {
    size_t j = i;
    while (!eos(j) && ws(j)) ++j;
    if (j == i) return false;
    if (eos(j)) {
      *end = j;
      return true;
    }
    if (j - i >= 2) {
      *end = j - 1;
      return true;
    }
    return false;
  }

  // Returns the codepoint index one PAST the match at i (never i).
  size_t match_at(size_t i) const {
    size_t end = 0;
    if (match_contraction(i, &end)) return end;

    // A2: optional prefix + letter run (Qwen: a letter-or-mark run).
    if (!eos(i) && run_char(i)) {
      size_t j = i + 1;
      while (!eos(j) && run_char(j)) ++j;
      return j;
    }
    if (!eos(i) && !letter(i) && !number(i) && cp(i) != '\r' && cp(i) != '\n' &&
        !eos(i + 1) && run_char(i + 1)) {
      size_t j = i + 2;
      while (!eos(j) && run_char(j)) ++j;
      return j;
    }

    // A3: 1..3 numbers (Qwen3.8, Qwen2: exactly one).
    if (!eos(i) && number(i)) {
      size_t j = i + 1;
      if (!single_number_)
        while (j < i + 3 && !eos(j) && number(j)) ++j;
      return j;
    }

    // A4: optional literal space + punct run + CR/LF tail.
    {
      size_t j = i;
      if (cp(i) == ' ' && !eos(i + 1) && punct(i + 1)) j = i + 1;
      if (!eos(j) && punct(j)) {
        while (!eos(j) && punct(j)) ++j;
        while (!eos(j) && (cps_[j] == '\r' || cps_[j] == '\n')) ++j;
        return j;
      }
    }

    // A5-A7: whitespace (order matters — newline runs first).
    if (match_newline_run(i, &end)) return end;
    if (match_trailing_ws(i, &end)) return end;
    size_t j = i;
    while (!eos(j) && ws(j)) ++j;
    return j;  // A7 (i is whitespace, so j > i)
  }

  std::string_view text_;
  std::vector<std::string_view>* out_;
  bool marks_ = false;
  bool single_number_ = false;
  std::vector<uint32_t> cps_;
  std::vector<size_t> starts_;  // size = cps+1 (end sentinel)
};

// The DeepSeek-V4.1 scanner (kSplitPatternDsv41*): the three stages over
// codepoints. Every stage's pieces bound the next stage's matches (HF's
// Sequence runs each pre-tokenizer on the splits the previous one left,
// so the lookahead of \s+(?!\S) sees a piece's end as the text's end).
class Dsv41Scanner {
 public:
  Dsv41Scanner(std::string_view text, std::vector<std::string_view>* out) : text_(text), out_(out) {
    size_t i = 0;
    while (i < text_.size()) {
      const int len = utf8_len(text_[i]);
      if (len < 1 || i + static_cast<size_t>(len) > text_.size()) reject("input is not valid UTF-8");
      for (int k = 1; k < len; ++k) {
        const unsigned char cont = static_cast<unsigned char>(text_[i + k]);
        if ((cont & 0xC0) != 0x80) reject("input is not valid UTF-8 (bad continuation byte)");
      }
      uint32_t raw = 0;
      for (int k = 0; k < len; ++k) raw = (raw << 8) | static_cast<unsigned char>(text_[i + k]);
      cps_.push_back(decode_utf8(raw, len));
      starts_.push_back(i);
      i += static_cast<size_t>(len);
    }
    starts_.push_back(text_.size());
  }

  void run() {
    // Stage 1: number runs in pieces of at most three; the spans between
    // go on to stage 2.
    const size_t n = cps_.size();
    size_t i = 0;
    while (i < n) {
      if (number(i)) {
        size_t j = i;
        while (j < n && j - i < 3 && number(j)) ++j;
        emit(i, j);
        i = j;
        continue;
      }
      size_t e = i;
      while (e < n && !number(e)) ++e;
      stage2(i, e);
      i = e;
    }
  }

 private:
  static int utf8_len(char c) {
    const unsigned char b = static_cast<unsigned char>(c);
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return -1;
  }
  static uint32_t decode_utf8(uint32_t raw, int len) {
    switch (len) {
      case 1: return raw & 0x7F;
      case 2: return ((raw >> 8) & 0x1F) << 6 | (raw & 0x3F);
      case 3: return ((raw >> 16) & 0x0F) << 12 | ((raw >> 8) & 0x3F) << 6 | (raw & 0x3F);
      default: return ((raw >> 24) & 0x07) << 18 | ((raw >> 16) & 0x3F) << 12 | ((raw >> 8) & 0x3F) << 6 | (raw & 0x3F);
    }
  }
  void emit(size_t a, size_t b) {
    if (b > a) out_->push_back(text_.substr(starts_[a], starts_[b] - starts_[a]));
  }
  bool number(size_t i) const { return unicode::is_number(cps_[i]); }
  bool cjk(size_t i) const {
    const uint32_t c = cps_[i];
    return (c >= 0x4E00 && c <= 0x9FA5) || (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF);
  }
  bool letter(size_t i) const { return unicode::is_letter(cps_[i]); }
  bool mark(size_t i) const { return unicode::is_mark(cps_[i]); }
  bool run_char(size_t i) const { return letter(i) || mark(i); }
  bool ps(size_t i) const { return unicode::is_punctuation(cps_[i]) || unicode::is_symbol(cps_[i]); }
  bool ws(size_t i) const { return unicode::is_white_space(cps_[i]); }
  bool crlf(size_t i) const { return cps_[i] == '\r' || cps_[i] == '\n'; }
  static bool ascii_punct(uint32_t c) {
    return (c >= 0x21 && c <= 0x2F) || (c >= 0x3A && c <= 0x40) || (c >= 0x5B && c <= 0x60) || (c >= 0x7B && c <= 0x7E);
  }
  static bool ascii_letter(uint32_t c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }

  // Stage 2 over [a, b): CJK runs are pieces; the spans between go on to
  // stage 3.
  void stage2(size_t a, size_t b) {
    size_t k = a;
    while (k < b) {
      size_t j = k;
      if (cjk(k)) {
        while (j < b && cjk(j)) ++j;
        emit(k, j);
      } else {
        while (j < b && !cjk(j)) ++j;
        stage3(k, j);
      }
      k = j;
    }
  }

  // Stage 3 over [a, b): the main pattern's matches, the gaps between
  // them pieces of their own.
  void stage3(size_t a, size_t b) {
    size_t k = a;
    size_t gap = b;  // b: no open gap
    while (k < b) {
      const size_t end = match3(k, b);
      if (end == k) {
        if (gap == b) gap = k;
        ++k;
        continue;
      }
      if (gap != b) {
        emit(gap, k);
        gap = b;
      }
      emit(k, end);
      k = end;
    }
    if (gap != b) emit(gap, b);
  }

  // The end of the leftmost-first match at k within [k, b), or k when no
  // alternative matches.
  size_t match3(size_t k, size_t b) const {
    // B1: one ASCII punctuation/symbol character + ASCII letters.
    if (ascii_punct(cps_[k]) && k + 1 < b && ascii_letter(cps_[k + 1])) {
      size_t j = k + 2;
      while (j < b && ascii_letter(cps_[j])) ++j;
      return j;
    }
    // B2: optional prefix + a letter/mark run.
    if (run_char(k)) {
      size_t j = k + 1;
      while (j < b && run_char(j)) ++j;
      return j;
    }
    if (!crlf(k) && !letter(k) && !ps(k) && k + 1 < b && run_char(k + 1)) {
      size_t j = k + 2;
      while (j < b && run_char(j)) ++j;
      return j;
    }
    // B3: optional literal space + a punctuation/symbol run + CR/LFs.
    {
      size_t j = k;
      if (cps_[k] == ' ' && k + 1 < b && ps(k + 1)) j = k + 1;
      if (j < b && ps(j)) {
        while (j < b && ps(j)) ++j;
        while (j < b && crlf(j)) ++j;
        return j;
      }
    }
    // B4: \s*[\r\n]+ — through the whitespace run's last CR/LF.
    {
      size_t j = k;
      while (j < b && ws(j)) ++j;
      size_t last_nl = k;
      for (size_t x = k; x < j; ++x)
        if (crlf(x)) last_nl = x + 1;
      if (last_nl > k) return last_nl;
      // B5: \s+(?!\S) — the whole run at the piece's end, one short before
      // a non-space (a run of >= 2 then); B6: \s+ — the run.
      if (j > k) {
        if (j == b) return j;
        if (j - k >= 2) return j - 1;
        return j;
      }
    }
    return k;
  }

  std::string_view text_;
  std::vector<std::string_view>* out_;
  std::vector<uint32_t> cps_;
  std::vector<size_t> starts_;
};

// ---------------------------------------------------------------------------
// The cased-word scanner (kSplitPatternTekken, kSplitPatternO200k): the
// two patterns whose word alternatives split on letter case. Over
// codepoints, like the scanners above. The classes:
//   P [^\r\n\p{L}\p{N}]               the optional word prefix: anything but
//                                     CR, LF, a letter or a number — so
//                                     whitespace, punctuation, controls and
//                                     MARKS all qualify
//   U [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]   "upper": a letter that is not Ll, or a mark
//   W [\p{Ll}\p{Lm}\p{Lo}\p{M}]         "lower": a letter that is not Lu/Lt, or a mark
// (caseless letters — Lm, Lo: CJK, Hangul, Arabic, Devanagari... — and
// marks sit in BOTH classes). The alternatives, leftmost-first:
//   C1 P?U*W+ c?                    a word ending in lower-class characters
//   C2 P?U+W* c?                    a word of upper-class characters
//   C3 \p{N} (tekken), \p{N}{1,3} (o200k)
//   C4  ?[^\s\p{L}\p{N}]+[\r\n/]*   optional literal SPACE + a punctuation
//                                   run (marks included) + trailing
//                                   CR/LF/'/' characters
//   C5-C7 the whitespace alternatives of the GLM pattern (A5-A7)
// c is the o200k pattern's (?i:'s|'t|'re|'ve|'m|'ll|'d): tried once where
// the word's letters end (the regex never backs into the letters to find
// one), under Unicode simple case folding — U+017F LONG S matches 's' (HF's
// Oniguruma does this; pinned by an exhaustive probe over every codepoint).
//
// The backtracking outcomes the scanner reproduces:
//   * C1's U* is greedy and runs through caseless letters and marks. When no
//     W character follows the run, the regex backs up to the run's LAST
//     character that is also W, which then is the whole W+: "ABカDE" matches
//     "ABカ", and a run of Lu/Lt alone fails C1 ("HTMLParser" is one piece —
//     U "HTMLP" then W "arser" — while "iPhone" is "i" | "Phone").
//   * The prefix is tried taken, then absent, and C1 is exhausted before C2
//     is tried at all. A mark is a legal prefix AND a legal run character, so
//     U+0301 before "ABC" fails C1 as a prefix, then matches C1 alone as its
//     own W+ — a piece of its own, although C2 would have taken all four.
//   * C2 is reached only when C1 failed, i.e. over a run of Lu/Lt alone: its
//     W* is then empty by construction, but the scanner runs it as written.
// Every codepoint starts a match (a letter C1 or C2, a mark C1, a number C3,
// whitespace C7, anything else C4), so the matches leave no text between
// them: the o200k file's Split(Removed, invert=true) — keep the matches,
// drop what lies between — produces exactly the pieces Isolated would, and
// the defensive throw below is the same loud pin as SplitScanner's.
// ---------------------------------------------------------------------------
class CasedSplitScanner {
 public:
  // contractions: a contraction may close a word (o200k); number_run: the
  // longest \p{N} run in one pretoken (tekken: 1, o200k: 3).
  CasedSplitScanner(std::string_view text, std::vector<std::string_view>* out, bool contractions,
                    size_t number_run)
      : text_(text), out_(out), contractions_(contractions), number_run_(number_run) {
    // Strict UTF-8, as SplitScanner; each codepoint's classes are looked up
    // once here (the general categories are disjoint, and no White_Space
    // codepoint is a letter, mark or number).
    size_t i = 0;
    while (i < text_.size()) {
      const unsigned char lead = static_cast<unsigned char>(text_[i]);
      const int len = utf8_len(lead);
      if (len < 1 || i + static_cast<size_t>(len) > text_.size())
        reject("input is not valid UTF-8");
      // The lead byte's payload bits, then six per continuation byte.
      uint32_t c = len == 1 ? lead : lead & (0xFFu >> (len + 1));
      for (int k = 1; k < len; ++k) {
        const unsigned char cont = static_cast<unsigned char>(text_[i + static_cast<size_t>(k)]);
        if ((cont & 0xC0) != 0x80) reject("input is not valid UTF-8 (bad continuation byte)");
        c = (c << 6) | (cont & 0x3Fu);
      }
      uint8_t cls = 0;
      if (unicode::is_letter(c)) {
        cls = kLetter;
        if (!unicode::is_lowercase_letter(c)) cls |= kUpper;
        if (!unicode::is_uppercase_or_titlecase_letter(c)) cls |= kLower;
      } else if (unicode::is_mark(c)) {
        cls = kUpper | kLower;
      } else if (unicode::is_number(c)) {
        cls = kNumber;
      } else if (unicode::is_white_space(c)) {
        cls = kSpace;
      }
      cps_.push_back(c);
      cls_.push_back(cls);
      starts_.push_back(i);
      i += static_cast<size_t>(len);
    }
    starts_.push_back(text_.size());
  }

  void run() {
    size_t i = 0;  // codepoint index
    while (i < cps_.size()) {
      const size_t end = match_at(i);
      if (end == i) reject("scanner hole at codepoint " + std::to_string(i));
      out_->push_back(text_.substr(starts_[i], starts_[end] - starts_[i]));
      i = end;
    }
  }

 private:
  enum : uint8_t { kLetter = 1, kUpper = 2, kLower = 4, kNumber = 8, kSpace = 16 };

  static int utf8_len(unsigned char b) {
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return -1;
  }
  bool eos(size_t i) const { return i >= cps_.size(); }
  uint32_t cp(size_t i) const { return eos(i) ? 0 : cps_[i]; }
  bool is(size_t i, uint8_t cls) const { return !eos(i) && (cls_[i] & cls) != 0; }
  bool upper(size_t i) const { return is(i, kUpper); }
  bool lower(size_t i) const { return is(i, kLower); }
  bool number(size_t i) const { return is(i, kNumber); }
  bool ws(size_t i) const { return is(i, kSpace); }
  bool crlf(size_t i) const { return cp(i) == '\r' || cp(i) == '\n'; }
  // P: not CR/LF, not a letter, not a number.
  bool prefix(size_t i) const { return !eos(i) && !is(i, kLetter | kNumber) && !crlf(i); }
  // C4's class: not whitespace, not a letter, not a number.
  bool punct(size_t i) const { return !eos(i) && !is(i, kSpace | kLetter | kNumber); }

  // U*W+ at s (C1 past its prefix): the greedy U run, then the greedy W run;
  // with no W run, back to the U run's last character that is also W.
  bool lower_word(size_t s, size_t* end) const {
    size_t a = s;
    while (upper(a)) ++a;
    size_t b = a;
    while (lower(b)) ++b;
    if (b > a) {
      *end = b;
      return true;
    }
    for (size_t k = a; k > s; --k)
      if (lower(k - 1)) {
        *end = k;
        return true;
      }
    return false;
  }

  // U+W* at s (C2 past its prefix).
  bool upper_word(size_t s, size_t* end) const {
    size_t a = s;
    while (upper(a)) ++a;
    if (a == s) return false;
    while (lower(a)) ++a;
    *end = a;
    return true;
  }

  // The optional contraction where a word's letters end at e: apostrophe +
  // s|t|m|d (1) or re|ve|ll (2), case-insensitive. Returns the match end.
  static uint32_t fold(uint32_t c) {
    if (c >= 'A' && c <= 'Z') return c + 32;
    return c == 0x17F ? uint32_t{'s'} : c;  // LATIN SMALL LETTER LONG S folds to 's'
  }
  size_t contraction(size_t e) const {
    if (!contractions_ || cp(e) != '\'') return e;
    const uint32_t a = fold(cp(e + 1));
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return e + 2;
    const uint32_t b = fold(cp(e + 2));
    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l'))
      return e + 3;
    return e;
  }

  // Returns the codepoint index one PAST the match at i (never i).
  size_t match_at(size_t i) const {
    size_t end = 0;
    // C1, then C2: each with the prefix taken first, then absent.
    if (prefix(i) && lower_word(i + 1, &end)) return contraction(end);
    if (lower_word(i, &end)) return contraction(end);
    if (prefix(i) && upper_word(i + 1, &end)) return contraction(end);
    if (upper_word(i, &end)) return contraction(end);

    // C3: one number (tekken) or 1..3 (o200k).
    if (number(i)) {
      size_t j = i + 1;
      while (j < i + number_run_ && number(j)) ++j;
      return j;
    }

    // C4: optional literal space + punct run + CR/LF/slash tail.
    {
      size_t j = i;
      if (cp(i) == ' ' && punct(i + 1)) j = i + 1;
      if (punct(j)) {
        while (punct(j)) ++j;
        while (crlf(j) || cp(j) == '/') ++j;
        return j;
      }
    }

    // C5: \s*[\r\n]+ — through the whitespace run's LAST CR/LF.
    size_t j = i;
    while (ws(j)) ++j;
    size_t last_nl = i;
    for (size_t k = i; k < j; ++k)
      if (crlf(k)) last_nl = k + 1;
    if (last_nl > i) return last_nl;
    // C6: \s+(?!\S) — the whole run at end of text, one short before a
    // non-space (a run of >= 2 then); C7: \s+ — the run.
    if (j > i && !eos(j) && j - i >= 2) return j - 1;
    return j;
  }

  std::string_view text_;
  std::vector<std::string_view>* out_;
  bool contractions_ = false;
  size_t number_run_ = 1;
  std::vector<uint32_t> cps_;
  std::vector<uint8_t> cls_;    // per codepoint: the k* class bits
  std::vector<size_t> starts_;  // size = cps+1 (end sentinel)
};

}  // namespace

// ---------------------------------------------------------------------------
// Load + validate.
// ---------------------------------------------------------------------------
Tokenizer Tokenizer::load(const std::string& path) {
  Tokenizer t;
  t.json_buf_ = read_file(path);
  t.revision_hash_ = fnv1a_bytes(t.json_buf_.data(), t.json_buf_.size());
  build_byte_level_maps(t.byte_to_cp_, t.cp_to_byte_);

  // The parse tree becomes a MEMBER: object keys (Member.key) are
  // tree-owned std::strings, so every view taken below stays valid for
  // the tokenizer's lifetime (json_buf_ is declared before doc_, and
  // both moves preserve their heap addresses).
  t.doc_ = minijson::parse(t.json_buf_);
  const minijson::Value& root = t.doc_.root;

  // The SentencePiece-style BPE shape (Gemma 4) shares nothing below but
  // the added-token trie: text/tokenizer_spm.cpp validates and loads it.
  if (is_spm_shape(root)) {
    load_spm(t);
    return t;
  }

  // --- normalizer: none, or NFC ----------------------------------------
  if (const minijson::Value* n = root.find("normalizer")) {
    if (!n->is_null()) {
      const minijson::Value* ty = n->find("type");
      if (ty && ty->is_string() && ty->as_string() == "NFC") {
        t.nfc_ = true;
      } else if (ty && ty->is_string() && ty->as_string() == "Sequence") {
        // DeepSeek-V4.1: an empty Sequence — no normalizer.
        const minijson::Value* list = n->find("normalizers");
        if (!list || !list->is_array() || !list->items().empty())
          reject("normalizer Sequence is not empty (only null, NFC and the empty Sequence are implemented)");
      } else {
        reject("normalizer is neither null, NFC nor the empty Sequence (the three implemented)");
      }
    }
  }

  // --- pre_tokenizer: the pinned Sequence ------------------------------
  const minijson::Value* pre = root.find("pre_tokenizer");
  if (!pre || !pre->is_object() ||
      want_string(*pre, "type", "pre_tokenizer") != "Sequence")
    reject("pre_tokenizer is not the pinned Sequence");
  const auto& pres = field(*pre, "pretokenizers", "pre_tokenizer").items();
  const size_t splits = pres.size() >= 1 ? pres.size() - 1 : 0;
  if ((splits != 1 && splits != 3) || want_string(pres[splits], "type", "pre_tokenizer[last]") != "ByteLevel")
    reject("pre_tokenizer is neither [Split, ByteLevel] nor [Split x 3, ByteLevel]");
  // Split behaviour: Isolated with invert=false — the matches and the text
  // between them are all pieces. The o200k file alone (MiniMax-M2.7) says
  // Removed with invert=true: the matches are the pieces and the text
  // between them is dropped — the same pieces for a pattern that leaves
  // nothing between its matches (the cased-word scanner's comment), and
  // accepted for that pattern only, below.
  bool removed_inverted = false;
  for (size_t k = 0; k < splits; ++k) {
    if (want_string(pres[k], "type", "pre_tokenizer[k]") != "Split") reject("pre_tokenizer stage is not a Split");
    const std::string_view behavior = want_string(pres[k], "behavior", "Split");
    const minijson::Value* inv = pres[k].find("invert");
    const bool invert = inv && inv->kind() == minijson::Value::Kind::Bool && inv->as_bool();
    if (splits == 1 && behavior == "Removed" && invert) {
      removed_inverted = true;
      continue;
    }
    if (behavior != "Isolated") reject("Split behavior is not Isolated");
    if (invert) reject("Split invert must be false");
  }
  const auto split_regex = [&](size_t k) {
    const minijson::Value& pattern = field(pres[k], "pattern", "Split");
    return want_string(pattern, "Regex", "Split.pattern");
  };
  if (splits == 1) {
    const std::string_view regex = split_regex(0);
    if (regex == kSplitPattern) t.pattern_ = 0;
    else if (regex == kSplitPatternQwen) t.pattern_ = 1;
    else if (regex == kSplitPatternQwen2) t.pattern_ = 3;
    else if (regex == kSplitPatternTekken) t.pattern_ = 4;
    else if (regex == kSplitPatternO200k) t.pattern_ = 5;
    else
      reject("Split pattern differs from the pinned regexes (the scanner "
             "hardcodes them — update the scanner or the checkpoint)");
    if (removed_inverted != (t.pattern_ == 5))
      reject("Split behavior/invert do not fit the pattern (the o200k regex is pinned as "
             "Removed with invert=true, every other regex as Isolated with invert=false)");
  } else {
    if (split_regex(0) != kSplitPatternDsv41Numbers || split_regex(1) != kSplitPatternDsv41Cjk ||
        split_regex(2) != kSplitPatternDsv41Main)
      reject("the three Split patterns differ from the pinned DeepSeek-V4.1 regexes (the scanner "
             "hardcodes them — update the scanner or the checkpoint)");
    t.pattern_ = 2;
  }
  if (want_bool(pres[splits], "use_regex", "ByteLevel"))
    reject("ByteLevel use_regex must be false (map-only)");
  if (want_bool(pres[splits], "add_prefix_space", "ByteLevel"))
    reject("ByteLevel add_prefix_space must be false");

  // --- decoder: ByteLevel (the reverse map) ----------------------------
  const minijson::Value* dec = root.find("decoder");
  if (!dec || !dec->is_object() ||
      want_string(*dec, "type", "decoder") != "ByteLevel")
    reject("decoder is not ByteLevel");

  // --- post_processor: offsets-only shapes -----------------------------
  // ByteLevel (GLM-5.3, Qwen) or a Sequence of ByteLevel processors
  // (GLM-4.7): both touch offsets only, never the ids.
  if (const minijson::Value* pp = root.find("post_processor")) {
    if (!pp->is_null()) {
      const minijson::Value* ty = pp->find("type");
      bool offsets_only = ty && ty->as_string() == "ByteLevel";
      if (ty && ty->as_string() == "Sequence") {
        const minijson::Value* procs = pp->find("processors");
        offsets_only = procs && procs->is_array();
        if (offsets_only)
          for (const auto& proc : procs->items()) {
            const minijson::Value* pt = proc.find("type");
            if (!pt || pt->as_string() != "ByteLevel") offsets_only = false;
          }
      }
      if (!offsets_only)
        reject("post_processor is neither null, ByteLevel nor a Sequence of "
               "ByteLevel (offsets-only) — special-token injection is not "
               "implemented at encode");
    }
  }

  // --- model: BPE with ignore_merges -----------------------------------
  const minijson::Value* model = root.find("model");
  if (!model || want_string(*model, "type", "model") != "BPE")
    reject("model.type is not BPE");
  // ignore_merges: absent (DeepSeek-V4.1) means false.
  if (const minijson::Value* im = model->find("ignore_merges"); im && !im->is_null())
    t.ignore_merges_ = want_bool(*model, "ignore_merges", "model");
  else
    t.ignore_merges_ = false;
  if (want_bool(*model, "byte_fallback", "model"))
    reject("model.byte_fallback must be false");
  if (const minijson::Value* v = model->find("unk_token"); v && !v->is_null())
    reject("model.unk_token must be null");
  // The affixes: null (GLM) or the empty string (Qwen) — both "none".
  for (const char* f : {"continuing_subword_prefix", "end_of_word_suffix"}) {
    const minijson::Value* v = model->find(f);
    if (v && !v->is_null() && !(v->is_string() && v->as_string().empty()))
      reject(std::string("model.") + f + " must be null or empty");
  }

  // --- vocab (dense id coverage validated below) -----------------------
  const auto& vocab = field(*model, "vocab", "model").members();
  t.vocab_.reserve(vocab.size());
  int64_t max_vocab_id = -1;
  for (const auto& [key, val] : vocab) {
    if (!val.is_number()) reject("vocab entry id is not a number");
    const int64_t id = val.as_int();
    if (!t.vocab_.emplace(key, id).second)
      reject("duplicate vocab token: " + key);
    max_vocab_id = std::max(max_vocab_id, id);
  }
  // The highest id encode can produce: the base vocabulary's here, raised
  // by the added tokens below. (Mistral-Small-4's added tokens are ids
  // 0..999 — its control tokens are also vocab keys, BELOW the BPE range —
  // so the last added token is not the highest id there.)
  t.max_id_ = max_vocab_id;

  // --- merges (rank = list order) ---------------------------------------
  const auto& merges = field(*model, "merges", "model").items();
  t.merge_rank_.reserve(merges.size());
  for (size_t r = 0; r < merges.size(); ++r) {
    // Two formats: ["first", "second"] (GLM's file) or "first second"
    // (Qwen's — one space; ByteLevel symbols never contain a space).
    std::string_view first, second;
    if (merges[r].is_string()) {
      const std::string_view m = merges[r].as_string();
      const size_t sp = m.find(' ');
      if (sp == std::string_view::npos || sp == 0 || sp + 1 >= m.size() ||
          m.find(' ', sp + 1) != std::string_view::npos)
        reject("malformed merge string at rank " + std::to_string(r));
      first = m.substr(0, sp);
      second = m.substr(sp + 1);
    } else {
      const auto& pair = merges[r].items();
      if (pair.size() != 2 || !pair[0].is_string() || !pair[1].is_string())
        reject("malformed merge entry at rank " + std::to_string(r));
      first = pair[0].as_string();
      second = pair[1].as_string();
    }
    if (!t.merge_rank_.emplace(MergeKey{first, second}, static_cast<uint32_t>(r)).second)
      reject("duplicate merge pair at rank " + std::to_string(r));
  }

  // --- added tokens: plain matching only -------------------------------
  const auto& added = field(root, "added_tokens", "root").items();
  const auto fresh_node = [] {
    TrieNode n;
    for (auto& c : n.child) c = -1;
    return n;
  };
  t.trie_.clear();
  t.trie_.push_back(fresh_node());  // root = trie_[0]
  t.added_tokens_.reserve(added.size());
  for (const auto& at : added) {
    AddedToken a;
    a.id = field(at, "id", "added_tokens").as_int();
    a.content = std::string(want_string(at, "content", "added_tokens"));
    a.special = want_bool(at, "special", "added_tokens");
    for (const char* f : {"single_word", "lstrip", "rstrip"}) {
      if (want_bool(at, f, "added_tokens"))
        reject("added token with " + std::string(f) +
               " set (plain matching only): " + a.content);
    }
    // The cased-word files (Mistral-Small-4, MiniMax-M2.7) are pinned with
    // every added token non-normalized: encode() cuts the added tokens out
    // of the RAW text and normalizes only what lies between them.
    if (t.pattern_ >= 4 && want_bool(at, "normalized", "added_tokens"))
      reject("added token with normalized set (added tokens match on the raw text only): " +
             a.content);
    t.added_tokens_.push_back(std::move(a));
    // Insert into the byte trie (leftmost-longest matching). Added
    // tokens are ASCII here, so a match can never start mid-codepoint
    // (UTF-8 continuation bytes are >= 0x80).
    int32_t node = 0;
    for (const char ch : t.added_tokens_.back().content) {
      const auto b = static_cast<unsigned char>(ch);
      int32_t next = t.trie_[static_cast<size_t>(node)].child[b];
      if (next < 0) {
        t.trie_.push_back(fresh_node());
        next = static_cast<int32_t>(t.trie_.size()) - 1;
        t.trie_[static_cast<size_t>(node)].child[b] = next;
      }
      node = next;
    }
    if (t.trie_[static_cast<size_t>(node)].id >= 0)
      reject("duplicate added token content: " + t.added_tokens_.back().content);
    t.trie_[static_cast<size_t>(node)].id = t.added_tokens_.back().id;
    t.max_id_ = std::max(t.max_id_, t.added_tokens_.back().id);
  }

  // --- reverse id map (decode path) ------------------------------------
  t.token_by_id_.assign(static_cast<size_t>(max_vocab_id) + 1, {});
  for (const auto& [key, val] : vocab) {
    const int64_t id = val.as_int();
    if (id < 0) reject("negative vocab id: " + key);
    auto& slot = t.token_by_id_[static_cast<size_t>(id)];
    if (!slot.empty()) reject("duplicate vocab id: " + std::to_string(id));
    slot = key;
  }
  for (size_t id = 0; id < t.token_by_id_.size(); ++id)
    if (t.token_by_id_[id].empty())
      reject("vocab ids are not dense (missing id " + std::to_string(id) +
             " — decode-by-id needs the dense base range)");

  const char* const pattern_names[] = {"glm", "qwen", "deepseek-v4.1", "qwen2", "tekken", "o200k"};
  DGPP_LOG_INFO(
      "tokenizer: loaded vocab {} merges {} added {} (revision 0x{:016x}; "
      "{} pattern, {}, ignore_merges {})",
      t.vocab_.size(), t.merge_rank_.size(), t.added_tokens_.size(),
      t.revision_hash_, pattern_names[t.pattern_], t.nfc_ ? "NFC" : "no normalizer",
      t.ignore_merges_);
  return t;
}

// ---------------------------------------------------------------------------
// Encode.
// ---------------------------------------------------------------------------
std::vector<int64_t> Tokenizer::encode(std::string_view text) const {
  std::vector<int64_t> ids;
  size_t seg_start = 0;
  size_t i = 0;
  while (i < text.size()) {
    // Leftmost-longest added-token walk at this byte.
    int32_t node = 0;
    int64_t best_id = -1;
    size_t best_len = 0;
    for (size_t k = i; k < text.size(); ++k) {
      const auto b = static_cast<unsigned char>(text[k]);
      node = trie_[static_cast<size_t>(node)].child[b];
      if (node < 0) break;
      if (trie_[static_cast<size_t>(node)].id >= 0) {
        best_id = trie_[static_cast<size_t>(node)].id;
        best_len = k - i + 1;
      }
    }
    if (best_id >= 0) {
      if (i > seg_start)
        encode_segment(text.substr(seg_start, i - seg_start), &ids);
      ids.push_back(best_id);
      i += best_len;
      seg_start = i;
    } else {
      ++i;
    }
  }
  if (seg_start < text.size())
    encode_segment(text.substr(seg_start), &ids);
  return ids;
}

void Tokenizer::encode_segment(std::string_view segment,
                                  std::vector<int64_t>* out) const {
  if (spm_) {
    encode_segment_spm(segment, out);
    return;
  }
  // The normalizer runs on each added-token-free segment (HF applies it
  // to the splits the added vocabulary leaves, the tokens themselves
  // being non-normalized).
  std::string normalized;
  if (nfc_ && !unicode::nfc_quick_yes(segment)) {
    normalized = unicode::nfc(segment);
    segment = normalized;
  }
  std::vector<std::string_view> pretokens;
  if (pattern_ == 2) {
    Dsv41Scanner scanner(segment, &pretokens);
    scanner.run();
  } else if (pattern_ == 4 || pattern_ == 5) {
    CasedSplitScanner scanner(segment, &pretokens, /*contractions=*/pattern_ == 5,
                              /*number_run=*/pattern_ == 5 ? 3 : 1);
    scanner.run();
  } else {
    SplitScanner scanner(segment, &pretokens, /*marks=*/pattern_ == 1, /*single_number=*/pattern_ == 1 || pattern_ == 3);
    scanner.run();
  }
  for (const std::string_view p : pretokens) {
    // ByteLevel map: each BYTE -> its alphabet codepoint (UTF-8; every
    // mapped codepoint is < 0x200 so <= 3 bytes).
    std::string mapped;
    mapped.reserve(p.size() * 3);
    for (size_t b = 0; b < p.size(); ++b) {
      const uint32_t cp = byte_to_cp_[static_cast<unsigned char>(p[b])];
      if (cp < 0x80) {
        mapped.push_back(static_cast<char>(cp));
      } else if (cp < 0x800) {
        mapped.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        mapped.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
      } else {
        mapped.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        mapped.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        mapped.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
      }
    }
    encode_word(mapped, out);
  }
}

void Tokenizer::encode_word(std::string_view mapped,
                               std::vector<int64_t>* out) const {
  // ignore_merges: a whole-word vocab hit skips the merge loop entirely
  // (GLM); with it off the merges always run (Qwen).
  if (ignore_merges_) {
    if (const auto it = vocab_.find(mapped); it != vocab_.end()) {
      out->push_back(it->second);
      return;
    }
  }
  // Initial symbols: one per mapped codepoint (1-3 UTF-8 bytes each).
  // `owned` holds every merged string for the WHOLE call — reserved once
  // so push_back can never reallocate (a realloc would dangle the
  // string_views in `symbols`; total merges <= initial count - 1).
  std::vector<std::string_view> symbols;
  std::vector<std::string> owned;
  symbols.reserve(mapped.size());
  for (size_t i = 0; i < mapped.size();) {
    const unsigned char b = static_cast<unsigned char>(mapped[i]);
    const size_t len = b < 0x80 ? 1 : (b & 0xE0) == 0xC0 ? 2 : 3;
    symbols.push_back(mapped.substr(i, len));
    i += len;
  }
  owned.reserve(symbols.size());

  // The merge loop: lowest rank first, ALL occurrences per pass,
  // left-to-right non-overlapping (HF 0.23's bpe semantics).
  for (;;) {
    uint32_t best_rank = UINT32_MAX;
    size_t best_i = 0;
    bool found = false;
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      const auto it = merge_rank_.find(MergeKey{symbols[i], symbols[i + 1]});
      if (it != merge_rank_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = i;
        found = true;
      }
    }
    if (!found) break;
    // Capture the pair BY VALUE before the rebuild (symbols[best_i+1] is
    // about to move).
    const std::string first(symbols[best_i]);
    const std::string second(symbols[best_i + 1]);
    std::vector<std::string_view> next;
    next.reserve(symbols.size());
    size_t i = 0;
    while (i < symbols.size()) {
      if (i + 1 < symbols.size() && symbols[i] == first &&
          symbols[i + 1] == second) {
        owned.push_back(first + second);
        next.push_back(owned.back());
        i += 2;
      } else {
        next.push_back(symbols[i]);
        ++i;
      }
    }
    symbols = std::move(next);
  }
  for (const std::string_view s : symbols) {
    const auto it = vocab_.find(s);
    if (it == vocab_.end())
      reject("BPE symbol missing from vocab (byte-level coverage broken): " +
             std::string(s));
    out->push_back(it->second);
  }
}

// ---------------------------------------------------------------------------
// Decode.
// ---------------------------------------------------------------------------
namespace {
// Verbatim decode of one id (added content or the ByteLevel-reversed
// vocab symbol). Throws on ids outside the vocab.
std::string decode_verbatim(const Tokenizer& t, int64_t id) {
  // Added tokens first (disjoint from the base range).
  for (const auto& a : t.added_tokens())
    if (a.id == id) return a.content;
  if (id < 0 || id >= static_cast<int64_t>(t.token_by_id_size()) ||
      t.token_by_id(static_cast<size_t>(id)).empty())
    throw std::runtime_error("decode: id outside the vocab: " +
                             std::to_string(id));
  const std::string_view sv = t.token_by_id(static_cast<size_t>(id));
  std::string out;
  for (size_t i = 0; i < sv.size();) {
    const unsigned char b = static_cast<unsigned char>(sv[i]);
    uint32_t cp;
    size_t len;
    if (b < 0x80) {
      cp = b;
      len = 1;
    } else if ((b & 0xE0) == 0xC0) {
      cp = ((b & 0x1F) << 6) |
           (static_cast<unsigned char>(sv[i + 1]) & 0x3F);
      len = 2;
    } else {
      cp = ((b & 0x0F) << 12) |
           ((static_cast<unsigned char>(sv[i + 1]) & 0x3F) << 6) |
           (static_cast<unsigned char>(sv[i + 2]) & 0x3F);
      len = 3;
    }
    if (cp >= 0x200 || t.cp_to_byte_at(cp) < 0)
      throw std::runtime_error(
          "decode: symbol codepoint outside the byte-level alphabet");
    out.push_back(static_cast<char>(t.cp_to_byte_at(cp)));
    i += len;
  }
  return out;
}
}  // namespace

std::string Tokenizer::decode(int64_t id, bool skip_special_tokens) const {
  if (skip_special_tokens) {
    for (const AddedToken& a : added_tokens_)
      if (a.id == id && a.special) return {};
  }
  if (spm_) return decode_verbatim_spm(id);
  return decode_verbatim(*this, id);
}

std::string Tokenizer::decode(const std::vector<int64_t>& ids,
                                 bool skip_special_tokens) const {
  std::string out;
  for (const int64_t id : ids) out += decode(id, skip_special_tokens);
  return out;
}

}  // namespace dgpp::text
