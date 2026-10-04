// The SentencePiece-style BPE shape of text::Tokenizer (Gemma 4;
// src/text/tokenizer_spm.cpp) on a miniature tokenizer.json with the
// release's exact pipeline — no checkpoint needed: the differential goldens
// HF tokenizers wrote for it (tools/gen_gemma4_mini_tokenizer.py), the
// decode rules, and the shapes the loader refuses. The release's own
// tokenizer.json has its corpus in tests/host/tokenizer_test.cpp
// (tests/data/gemma4_tokenizer_goldens.jsonl).
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "text/tokenizer.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::string data_path(const char* name) { return std::string(DGPP_SOURCE_DIR) + "/tests/data/" + name; }

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

const dgpp::text::Tokenizer& mini() {
  static const dgpp::text::Tokenizer tok = dgpp::text::Tokenizer::load(data_path("gemma4_mini_tokenizer.json"));
  return tok;
}

std::string spaced(const std::string& text) {
  std::string out;
  for (size_t k = 0; k < text.size();) {
    if (text.compare(k, 3, "\xE2\x96\x81") == 0) {
      out.push_back(' ');
      k += 3;
    } else {
      out.push_back(text[k++]);
    }
  }
  return out;
}

// The miniature file with one anchor replaced, written beside the test's
// working directory; loading it must throw a message holding `needle`.
void refused(const std::string& from, const std::string& to, const char* needle) {
  std::string json = read_file(data_path("gemma4_mini_tokenizer.json"));
  const size_t at = json.find(from);
  if (at == std::string::npos) throw std::runtime_error("patch anchor missing: " + from);
  json.replace(at, from.size(), to);
  const std::string tmp =
      (std::filesystem::temp_directory_path() / ("gemma4_mini_bad_" + std::to_string(::getpid()) + ".json")).string();
  {
    std::ofstream f(tmp, std::ios::binary);
    f << json;
  }
  std::string msg;
  try {
    (void)dgpp::text::Tokenizer::load(tmp);
  } catch (const std::exception& e) {
    msg = e.what();
  }
  std::remove(tmp.c_str());
  require(msg.find(needle) != std::string::npos,
          std::string("expected a refusal holding '") + needle + "', got '" + msg + "'");
}

std::string ids_text(const std::vector<int64_t>& ids) {
  std::string s = "[";
  for (size_t i = 0; i < ids.size(); ++i) s += (i ? " " : "") + std::to_string(ids[i]);
  return s + "]";
}

}  // namespace

DGPP_TEST(gemma4_tokenizer_loads_the_sentencepiece_shape) {
  const dgpp::text::Tokenizer& tok = mini();
  require(tok.sentencepiece_bpe(), "shape");
  require(tok.added_tokens().size() == 11, "added tokens");
  // The added tokens sit inside the base range: the highest id is the vocabulary's last.
  require(tok.max_id() == static_cast<int64_t>(tok.token_by_id_size()) - 1, "max id");
  require(tok.token_by_id(2) == "<bos>" && tok.token_by_id(11) == "<0x00>" && tok.token_by_id(266) == "<0xFF>",
          "specials and byte tokens by id");
}

DGPP_TEST(gemma4_tokenizer_differential_goldens) {
  const dgpp::text::Tokenizer& tok = mini();
  std::istringstream lines(read_file(data_path("gemma4_mini_tokenizer_goldens.jsonl")));
  std::string line;
  require(static_cast<bool>(std::getline(lines, line)), "empty corpus");
  {
    const auto header = dgpp::minijson::parse(line);
    char got[32];
    std::snprintf(got, sizeof got, "%016llx", static_cast<unsigned long long>(tok.revision_hash()));
    require(header.root.at("revision_hash").as_string() == got,
            "the goldens are keyed to another gemma4_mini_tokenizer.json — regenerate both "
            "(tools/gen_gemma4_mini_tokenizer.py)");
  }
  size_t checked = 0;
  while (std::getline(lines, line)) {
    if (line.empty()) continue;
    const auto parsed = dgpp::minijson::parse(line);
    const std::string text(parsed.root.at("text").as_string());
    std::vector<int64_t> want;
    for (const auto& v : parsed.root.at("ids").items()) want.push_back(v.as_int());
    const std::vector<int64_t> got = tok.encode(text);
    require(got == want, "encode mismatch on '" + text + "': got " + ids_text(got) + ", want " + ids_text(want));
    // Verbatim: the text back, a literal U+2581 as the space it stands for.
    require(tok.decode(got, false) == spaced(text), "verbatim round trip on '" + text + "'");
    // HF's decode with the special tokens skipped.
    require(tok.decode(got, true) == std::string(parsed.root.at("skipped").as_string()),
            "skip-special decode on '" + text + "'");
    // One id at a time is the same bytes (the streaming path).
    std::string joined;
    for (const int64_t id : got) joined += tok.decode(id, false);
    require(joined == spaced(text), "per-id decode on '" + text + "'");
    ++checked;
  }
  require(checked == 105, "corpus size " + std::to_string(checked));
}

DGPP_TEST(gemma4_tokenizer_whitespace_and_fallback) {
  const dgpp::text::Tokenizer& tok = mini();
  const auto id = [&](const char* s) {
    const auto ids = tok.encode(s);
    require(ids.size() == 1, std::string("'") + s + "' is not one token");
    return ids[0];
  };
  // A space is the space mark; runs of it are single tokens; the decoder brings the spaces back.
  require(id(" ") == id("\xE2\x96\x81"), "space == U+2581");
  require(tok.decode(id("    "), false) == "    ", "a run of four");
  require(tok.token_by_id(static_cast<size_t>(id(" the"))) == "\xE2\x96\x81the", "the piece keeps its mark in the vocabulary");
  require(tok.decode(id(" the"), false) == " the", "and decodes with the space");
  // A segment is ONE word: a newline run merges across what other tokenizers call a boundary.
  require(tok.encode("\n\n\n\n").size() == 1, "four newlines are one token");
  // Outside the vocabulary: the UTF-8 bytes' tokens, ids 11 + byte in this file.
  require(tok.encode("\xC3\xBC") == std::vector<int64_t>({11 + 0xC3, 11 + 0xBC}), "u-umlaut falls back to two bytes");
  require(tok.encode("\xF0\x9F\x91\x8D") == std::vector<int64_t>({11 + 0xF0, 11 + 0x9F, 11 + 0x91, 11 + 0x8D}),
          "an emoji falls back to four bytes");
  require(tok.encode(std::string_view("\0", 1)) == std::vector<int64_t>({11}), "NUL");
  // A byte token decodes to its byte; a run of them is the character again.
  require(tok.decode(11 + 0xC3, false) == "\xC3" && tok.decode(std::vector<int64_t>{11 + 0xC3, 11 + 0xBC}, false) == "\xC3\xBC",
          "byte tokens decode to bytes");
  // Bytes that are not UTF-8 (never produced from valid text) fall back byte by byte and do not throw.
  require(tok.encode("\xFF\xC3") == std::vector<int64_t>({11 + 0xFF, 11 + 0xC3}), "malformed input falls back per byte");
  // The text "<0x41>" is ordinary characters, not the byte token.
  require(tok.encode("<0x41>").size() > 1, "the spelled-out byte token is not matched as one");
  // Out-of-range ids are refused.
  bool threw = false;
  try {
    (void)tok.decode(static_cast<int64_t>(tok.token_by_id_size()), false);
  } catch (const std::exception&) {
    threw = true;
  }
  require(threw, "an id past the vocabulary must throw");
}

DGPP_TEST(gemma4_tokenizer_added_tokens) {
  const dgpp::text::Tokenizer& tok = mini();
  // Leftmost-longest: <|tool_call> wins over its prefix <|tool>.
  require(tok.encode("<|tool_call>") == std::vector<int64_t>({9}) && tok.encode("<|tool>") == std::vector<int64_t>({8}),
          "prefix tokens");
  require(tok.encode("<|tool><|tool_call>") == std::vector<int64_t>({8, 9}), "adjacent");
  // The text around an added token is encoded on its own: no merge crosses it.
  const auto ids = tok.encode("a<|turn>a");
  require(ids.size() == 3 && ids[1] == 5 && ids[0] == ids[2], "segments around an added token");
  // Special tokens vanish from the skip-special decode and stay in the verbatim one.
  require(tok.decode(std::vector<int64_t>{2, 5}, true).empty(), "skipped");
  require(tok.decode(std::vector<int64_t>{2, 5}, false) == "<bos><|turn>", "verbatim");
}

DGPP_TEST(gemma4_tokenizer_refuses_other_shapes) {
  refused("\"content\": \"\xE2\x96\x81\"\n},\n\"pre_tokenizer\"", "\"content\": \"_\"\n},\n\"pre_tokenizer\"", "normalizer");
  refused("\"behavior\": \"MergedWithPrevious\"", "\"behavior\": \"Isolated\"", "MergedWithPrevious");
  refused("\"type\": \"ByteFallback\"", "\"type\": \"Strip\"", "decoder");
  refused("\"byte_fallback\": true", "\"byte_fallback\": false", "byte_fallback");
  refused("\"ignore_merges\": false", "\"ignore_merges\": true", "ignore_merges");
  refused("\"special_tokens\": {}", "\"special_tokens\": {\"<bos>\": {\"id\": \"<bos>\", \"ids\": [2], \"tokens\": [\"<bos>\"]}}",
          "special_tokens");
  // A missing byte token: unk would be reachable, and unk is not implemented.
  refused("\"<0x7F>\"", "\"<0x7F_>\"", "byte fallback token missing for byte 127");
  // A lower-case byte token is not what the encoder writes.
  refused("\"<0xAB>\"", "\"<0xab>\"", "byte fallback token missing for byte 171");
  // A merge naming a token outside the vocabulary.
  refused("[\n\"t\",\n\"h\"\n]", "[\n\"t\",\n\"q_\"\n]", "outside the vocabulary");
  // An added token matched after the normalizer.
  refused("\"normalized\": false", "\"normalized\": true", "normalized");
  refused("\"lstrip\": false", "\"lstrip\": true", "lstrip");
}
