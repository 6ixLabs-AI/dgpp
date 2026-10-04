// The SentencePiece-style BPE shape of text::Tokenizer (Gemma 4, 2026-10-04;
// the shape is described in text/tokenizer.hpp). The reference is HF
// tokenizers' own code path for this tokenizer.json: normalizers::Replace,
// pre_tokenizers::Split, models::bpe (merge_word + Word::merge_all),
// decoders::{Replace, ByteFallback, Fuse}.
#include <algorithm>
#include <cstring>
#include <queue>
#include <stdexcept>
#include <utility>

#include "common/log.hpp"
#include "loaders/minijson.hpp"
#include "text/tokenizer.hpp"

namespace dgpp::text {
namespace {

// U+2581 LOWER ONE EIGHTH BLOCK, the SentencePiece space.
constexpr char kSpaceMark[] = "\xE2\x96\x81";

[[noreturn]] void reject(const std::string& what) {
  throw std::runtime_error("glm_tokenizer: " + what);
}

const minijson::Value& field(const minijson::Value& v, const char* name, const char* where) {
  const minijson::Value* f = v.find(name);
  if (!f) reject(std::string(where) + ": missing field '" + name + "'");
  return *f;
}

std::string_view want_string(const minijson::Value& v, const char* name, const char* where) {
  const minijson::Value& f = field(v, name, where);
  if (!f.is_string()) reject(std::string(where) + ": '" + name + "' not a string");
  return f.as_string();
}

bool want_bool(const minijson::Value& v, const char* name, const char* where) {
  const minijson::Value& f = field(v, name, where);
  if (!f.is_bool()) reject(std::string(where) + ": '" + name + "' not a bool");
  return f.as_bool();
}

// {"String": "<s>"} — the literal-pattern form of Replace / Split.
std::string_view literal_pattern(const minijson::Value& v, const char* where) {
  const minijson::Value& pattern = field(v, "pattern", where);
  if (!pattern.is_object() || pattern.find("String") == nullptr)
    reject(std::string(where) + ": pattern is not a literal String (a Regex is not implemented)");
  return want_string(pattern, "String", where);
}

// "<0xNN>" with two hex digits -> NN, else -1 (HF's ByteFallback test:
// length 6, the "<0x" prefix, the ">" suffix, a base-16 byte between).
int byte_token(std::string_view s) {
  if (s.size() != 6 || s.substr(0, 3) != "<0x" || s[5] != '>') return -1;
  int v = 0;
  for (const char c : s.substr(3, 2)) {
    int d;
    if (c >= '0' && c <= '9') d = c - '0';
    else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
    else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
    else return -1;
    v = v * 16 + d;
  }
  return v;
}

// The byte length of the UTF-8 sequence starting at s[i], or 1 when the
// bytes there are not a well-formed sequence (the lone byte then falls back
// to its "<0xNN>" token — HF never sees one: its input is a Rust string).
size_t utf8_len(std::string_view s, size_t i) {
  const unsigned char b = static_cast<unsigned char>(s[i]);
  size_t len = b < 0x80 ? 1 : (b & 0xE0) == 0xC0 ? 2 : (b & 0xF0) == 0xE0 ? 3 : (b & 0xF8) == 0xF0 ? 4 : 0;
  if (len == 0 || i + len > s.size()) return 1;
  for (size_t k = 1; k < len; ++k)
    if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) return 1;
  return len;
}

// HF's Word: a doubly linked list of symbols in a vector; a merged-away
// symbol keeps its slot with len 0.
struct Symbol {
  uint32_t c;
  int32_t prev;
  int32_t next;
  uint32_t len;
};

// HF's Merge ordering: the queue pops the lowest rank, then the leftmost
// position.
struct Merge {
  uint32_t pos;
  uint32_t rank;
  uint32_t new_id;
};
struct MergeLater {
  bool operator()(const Merge& a, const Merge& b) const {
    return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos;
  }
};

}  // namespace

bool Tokenizer::is_spm_shape(const minijson::Value& root) {
  const minijson::Value* n = root.find("normalizer");
  if (!n || !n->is_object()) return false;
  const minijson::Value* ty = n->find("type");
  return ty && ty->is_string() && ty->as_string() == "Replace";
}

void Tokenizer::load_spm(Tokenizer& t) {
  const minijson::Value& root = t.doc_.root;
  t.spm_ = true;
  t.pattern_ = 4;
  t.nfc_ = false;
  t.ignore_merges_ = false;

  // --- normalizer: Replace " " -> U+2581 --------------------------------
  const minijson::Value& norm = field(root, "normalizer", "root");
  if (literal_pattern(norm, "normalizer") != " " || want_string(norm, "content", "normalizer") != kSpaceMark)
    reject("normalizer Replace is not \" \" -> U+2581 (the SentencePiece space)");

  // --- pre_tokenizer: Split on " ", MergedWithPrevious -------------------
  // After the normalizer no space is left, so the Split never fires and a
  // segment is one word; any other pattern or behaviour would split.
  const minijson::Value* pre = root.find("pre_tokenizer");
  if (!pre || !pre->is_object() || want_string(*pre, "type", "pre_tokenizer") != "Split")
    reject("pre_tokenizer is not the Split of the SentencePiece-style shape");
  if (literal_pattern(*pre, "pre_tokenizer") != " ")
    reject("pre_tokenizer Split pattern is not the literal space");
  if (want_string(*pre, "behavior", "pre_tokenizer") != "MergedWithPrevious")
    reject("pre_tokenizer Split behavior is not MergedWithPrevious");
  if (want_bool(*pre, "invert", "pre_tokenizer")) reject("pre_tokenizer Split invert must be false");

  // --- decoder: Sequence[Replace U+2581 -> " ", ByteFallback, Fuse] -------
  const minijson::Value* dec = root.find("decoder");
  if (!dec || !dec->is_object() || want_string(*dec, "type", "decoder") != "Sequence")
    reject("decoder is not the Sequence of the SentencePiece-style shape");
  const auto& decs = field(*dec, "decoders", "decoder").items();
  if (decs.size() != 3 || want_string(decs[0], "type", "decoder[0]") != "Replace" ||
      literal_pattern(decs[0], "decoder[0]") != kSpaceMark || want_string(decs[0], "content", "decoder[0]") != " " ||
      want_string(decs[1], "type", "decoder[1]") != "ByteFallback" || want_string(decs[2], "type", "decoder[2]") != "Fuse")
    reject("decoder is not Sequence[Replace U+2581 -> \" \", ByteFallback, Fuse]");

  // --- post_processor: a TemplateProcessing that adds no token ------------
  if (const minijson::Value* pp = root.find("post_processor"); pp && !pp->is_null()) {
    if (want_string(*pp, "type", "post_processor") != "TemplateProcessing")
      reject("post_processor is neither null nor TemplateProcessing");
    const auto& single = field(*pp, "single", "post_processor").items();
    if (single.size() != 1 || single[0].find("Sequence") == nullptr)
      reject("post_processor.single is not the bare sequence — special-token injection is not implemented at encode");
    if (const minijson::Value* st = pp->find("special_tokens"); st && !st->is_null() && !st->members().empty())
      reject("post_processor.special_tokens is not empty — special-token injection is not implemented at encode");
  }

  // --- model: BPE over characters with byte fallback -----------------------
  const minijson::Value* model = root.find("model");
  if (!model || want_string(*model, "type", "model") != "BPE") reject("model.type is not BPE");
  if (!want_bool(*model, "byte_fallback", "model"))
    reject("model.byte_fallback must be true for the SentencePiece-style shape");
  if (const minijson::Value* im = model->find("ignore_merges"); im && !im->is_null() && want_bool(*model, "ignore_merges", "model"))
    reject("model.ignore_merges must be false for the SentencePiece-style shape");
  if (const minijson::Value* d = model->find("dropout"); d && !d->is_null()) reject("model.dropout must be null");
  for (const char* f : {"continuing_subword_prefix", "end_of_word_suffix"}) {
    const minijson::Value* v = model->find(f);
    if (v && !v->is_null() && !(v->is_string() && v->as_string().empty()))
      reject(std::string("model.") + f + " must be null or empty");
  }

  // --- vocab: dense ids, every byte token present ---------------------------
  const auto& vocab = field(*model, "vocab", "model").members();
  t.vocab_.reserve(vocab.size());
  int64_t max_vocab_id = -1;
  for (const auto& [key, val] : vocab) {
    if (!val.is_number()) reject("vocab entry id is not a number");
    const int64_t id = val.as_int();
    if (id < 0 || id > UINT32_MAX) reject("vocab id out of range: " + key);
    if (!t.vocab_.emplace(key, id).second) reject("duplicate vocab token: " + key);
    max_vocab_id = std::max(max_vocab_id, id);
  }
  t.token_by_id_.assign(static_cast<size_t>(max_vocab_id) + 1, {});
  t.spm_id_byte_.assign(static_cast<size_t>(max_vocab_id) + 1, -1);
  std::fill(std::begin(t.spm_byte_id_), std::end(t.spm_byte_id_), int64_t{-1});
  for (const auto& [key, val] : vocab) {
    const int64_t id = val.as_int();
    auto& slot = t.token_by_id_[static_cast<size_t>(id)];
    if (!slot.empty()) reject("duplicate vocab id: " + std::to_string(id));
    if (key.empty()) reject("empty vocab token at id " + std::to_string(id));
    slot = key;
    if (const int b = byte_token(key); b >= 0) {
      t.spm_id_byte_[static_cast<size_t>(id)] = static_cast<int16_t>(b);
      // The encoder writes the upper-case form (Rust's {:#04X}): that
      // spelling is the byte's token.
      static const char kHex[] = "0123456789ABCDEF";
      if (key[3] == kHex[b >> 4] && key[4] == kHex[b & 15]) t.spm_byte_id_[b] = id;
    }
  }
  for (size_t id = 0; id < t.token_by_id_.size(); ++id)
    if (t.token_by_id_[id].empty())
      reject("vocab ids are not dense (missing id " + std::to_string(id) + " — decode-by-id needs the dense base range)");
  // With all 256 byte tokens present every character encodes, and the unk
  // token (which HF would otherwise fuse) is unreachable.
  for (int b = 0; b < 256; ++b)
    if (t.spm_byte_id_[b] < 0)
      reject("byte fallback token missing for byte " + std::to_string(b) +
             " (the shape needs all 256 \"<0xNN>\" tokens: unk is not implemented)");

  // --- merges: (first id, second id) -> (rank, id of first + second) --------
  const auto& merges = field(*model, "merges", "model").items();
  t.spm_merges_.reserve(merges.size());
  std::string joined;
  for (size_t r = 0; r < merges.size(); ++r) {
    // The [first, second] form only: a symbol may hold a space-free
    // string of any characters, so the "first second" form is ambiguous.
    if (!merges[r].is_array()) reject("merge at rank " + std::to_string(r) + " is not a [first, second] pair");
    const auto& pair = merges[r].items();
    if (pair.size() != 2 || !pair[0].is_string() || !pair[1].is_string())
      reject("malformed merge entry at rank " + std::to_string(r));
    const std::string_view first = pair[0].as_string(), second = pair[1].as_string();
    const auto a = t.vocab_.find(first), b = t.vocab_.find(second);
    joined.assign(first);
    joined.append(second);
    const auto ab = t.vocab_.find(std::string_view(joined));
    if (a == t.vocab_.end() || b == t.vocab_.end() || ab == t.vocab_.end())
      reject("merge at rank " + std::to_string(r) + " names a token outside the vocabulary");
    const uint64_t key = (static_cast<uint64_t>(a->second) << 32) | static_cast<uint64_t>(b->second);
    // A pair listed twice: HF's map keeps the later entry. The pinned file
    // has none; a file that does is refused rather than guessed at.
    if (!t.spm_merges_.emplace(key, SpmMerge{static_cast<uint32_t>(r), static_cast<uint32_t>(ab->second)}).second)
      reject("duplicate merge pair at rank " + std::to_string(r));
  }

  // --- added tokens: plain matching, not normalized ---------------------------
  const auto& added = field(root, "added_tokens", "root").items();
  const auto fresh_node = [] {
    TrieNode n;
    for (auto& c : n.child) c = -1;
    return n;
  };
  t.trie_.clear();
  t.trie_.push_back(fresh_node());  // root = trie_[0]
  t.added_tokens_.reserve(added.size());
  t.max_id_ = max_vocab_id;
  for (const auto& at : added) {
    AddedToken a;
    a.id = field(at, "id", "added_tokens").as_int();
    a.content = std::string(want_string(at, "content", "added_tokens"));
    a.special = want_bool(at, "special", "added_tokens");
    for (const char* f : {"single_word", "lstrip", "rstrip"})
      if (want_bool(at, f, "added_tokens"))
        reject("added token with " + std::string(f) + " set (plain matching only): " + a.content);
    // A normalized added token is matched AFTER the normalizer (a space in
    // it would have become U+2581): not implemented.
    if (want_bool(at, "normalized", "added_tokens"))
      reject("added token with normalized set (matched after the normalizer — not implemented): " + a.content);
    if (a.content.empty()) reject("empty added token");
    // An added token inside the base range must be that id's vocab entry.
    if (a.id >= 0 && a.id <= max_vocab_id && t.token_by_id_[static_cast<size_t>(a.id)] != a.content)
      reject("added token '" + a.content + "' disagrees with the vocabulary at its id");
    t.added_tokens_.push_back(std::move(a));
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

  DGPP_LOG_INFO("tokenizer: loaded vocab {} merges {} added {} (revision 0x{:016x}; sentencepiece-style BPE, "
                "byte fallback)",
                t.vocab_.size(), t.spm_merges_.size(), t.added_tokens_.size(), t.revision_hash_);
}

void Tokenizer::encode_segment_spm(std::string_view segment, std::vector<int64_t>* out) const {
  if (segment.empty()) return;
  // The normalizer: every space becomes U+2581.
  std::string text;
  text.reserve(segment.size() + segment.size() / 4);
  for (const char c : segment) {
    if (c == ' ') text += kSpaceMark;
    else text.push_back(c);
  }
  // merge_word: one symbol per character — its vocabulary id, or the
  // "<0xNN>" tokens of its UTF-8 bytes.
  std::vector<Symbol> symbols;
  symbols.reserve(text.size());
  const auto push = [&](uint32_t id, uint32_t len) {
    const int32_t at = static_cast<int32_t>(symbols.size());
    if (at > 0) symbols.back().next = at;
    symbols.push_back(Symbol{id, at - 1, -1, len});
  };
  const std::string_view tv(text);
  for (size_t i = 0; i < tv.size();) {
    const size_t len = utf8_len(tv, i);
    if (const auto it = vocab_.find(tv.substr(i, len)); it != vocab_.end()) {
      push(static_cast<uint32_t>(it->second), static_cast<uint32_t>(len));
    } else {
      for (size_t k = 0; k < len; ++k)
        push(static_cast<uint32_t>(spm_byte_id_[static_cast<unsigned char>(tv[i + k])]), 1);
    }
    i += len;
  }
  // Word::merge_all: the queue starts with every adjacent pair that has a
  // merge; each pop merges ONE occurrence and queues the two pairs the new
  // symbol forms. An entry is stale when its left symbol is gone, is last,
  // or now pairs into another token than the entry recorded.
  const auto find_merge = [&](uint32_t a, uint32_t b) -> const SpmMerge* {
    const auto it = spm_merges_.find((static_cast<uint64_t>(a) << 32) | b);
    return it == spm_merges_.end() ? nullptr : &it->second;
  };
  std::priority_queue<Merge, std::vector<Merge>, MergeLater> queue;
  for (size_t i = 0; i + 1 < symbols.size(); ++i)
    if (const SpmMerge* m = find_merge(symbols[i].c, symbols[i + 1].c))
      queue.push(Merge{static_cast<uint32_t>(i), m->rank, m->new_id});
  const int32_t n = static_cast<int32_t>(symbols.size());
  while (!queue.empty()) {
    const Merge top = queue.top();
    queue.pop();
    Symbol& cur = symbols[top.pos];
    if (cur.len == 0 || cur.next == -1) continue;
    const int32_t next_pos = cur.next;
    const Symbol right = symbols[static_cast<size_t>(next_pos)];
    const SpmMerge* now = find_merge(cur.c, right.c);
    if (now == nullptr || now->new_id != top.new_id) continue;
    cur.c = top.new_id;
    cur.len += right.len;
    cur.next = right.next;
    symbols[static_cast<size_t>(next_pos)].len = 0;
    if (right.next > -1 && right.next < n) symbols[static_cast<size_t>(right.next)].prev = static_cast<int32_t>(top.pos);
    if (cur.prev >= 0)
      if (const SpmMerge* m = find_merge(symbols[static_cast<size_t>(cur.prev)].c, cur.c))
        queue.push(Merge{static_cast<uint32_t>(cur.prev), m->rank, m->new_id});
    if (cur.next >= 0 && cur.next < n)
      if (const SpmMerge* m = find_merge(cur.c, symbols[static_cast<size_t>(cur.next)].c))
        queue.push(Merge{top.pos, m->rank, m->new_id});
  }
  for (const Symbol& s : symbols)
    if (s.len != 0) out->push_back(static_cast<int64_t>(s.c));
}

std::string Tokenizer::decode_verbatim_spm(int64_t id) const {
  if (id < 0 || id >= static_cast<int64_t>(token_by_id_.size())) {
    // An added token past the base range (the pinned file has none).
    for (const AddedToken& a : added_tokens_)
      if (a.id == id) return a.content;
    throw std::runtime_error("decode: id outside the vocab: " + std::to_string(id));
  }
  // ByteFallback: a "<0xNN>" token is the byte itself. A run of them is the
  // UTF-8 the text had; the callers that stream join bytes across tokens
  // (HF's whole-sequence decode writes U+FFFD for a run that is not
  // UTF-8 — a per-token decode cannot know, and hands the bytes on).
  if (const int16_t b = spm_id_byte_[static_cast<size_t>(id)]; b >= 0) return std::string(1, static_cast<char>(b));
  // Replace: every U+2581 is a space.
  const std::string_view sv = token_by_id_[static_cast<size_t>(id)];
  std::string out;
  out.reserve(sv.size());
  for (size_t i = 0; i < sv.size();) {
    if (sv.compare(i, 3, kSpaceMark) == 0) {
      out.push_back(' ');
      i += 3;
    } else {
      out.push_back(sv[i]);
      ++i;
    }
  }
  return out;
}

}  // namespace dgpp::text
