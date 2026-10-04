// Gemma 4's chat template and tokenizer over the checkpoint's own files
// (host; skips with rc 2 when the checkpoint is not in the model cache).
// The two releases ship DIFFERENT templates over one tokenizer —
// nvidia/Gemma-4-31B-IT-NVFP4's folds OpenAI "tool" messages into the
// assistant turn as tool responses and renders reasoning beside calls;
// bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16's is the earlier one (a
// "tool" message is a turn of its own) — so each has its corpus
// (tests/data/gemma4_chat_template_goldens.jsonl and
// gemma4_26b_chat_template_goldens.jsonl) and this test runs once per corpus:
// the differential goldens of tools/gen_chat_template_goldens.py — jinja2
// under transformers' environment for the renders, HF tokenizers for the
// ids of those renders (tests/data/gemma4_chat_template_goldens.jsonl,
// keyed by the template's and the tokenizer's hashes). Every case is
// rendered through the interpreter (text/chat_template.hpp) and encoded
// with the SentencePiece-style BPE (text/tokenizer_spm.cpp); both must be
// byte-exact.
//
//   gemma4_chat_test [corpus.jsonl]     the model is the corpus header's
//                                        (DGPP_TP_REAL_MODEL overrides)
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common/log.hpp"

int g_argc = 0;
char** g_argv = nullptr;
#include "common/test.hpp"
#include "loaders/hf_cache.hpp"
#include "loaders/minijson.hpp"
#include "serve/frontend.hpp"
#include "text/chat_template.hpp"
#include "text/tokenizer.hpp"
#include "text/tool_grammar.hpp"
#include "text/tool_parser.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::string read_text_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string golden_path() {
  return g_argc > 1 ? g_argv[1] : "tests/data/gemma4_chat_template_goldens.jsonl";
}

std::string hex16(uint64_t v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
  return buf;
}

// The corpus (header + cases) and the checkpoint files it is keyed to.
struct World {
  std::vector<std::string> lines;
  std::string model;
  std::filesystem::path snap;
  dgpp::text::Tokenizer tok;
  dgpp::text::ChatTemplate tpl;
  // Whether this template folds role "tool" messages into the assistant's
  // turn by their tool_call_id (the 31B release's) or writes them as turns.
  bool folds_tool_messages = false;

  World(std::vector<std::string> corpus, std::string model_id, const std::string& dir)
      : lines(std::move(corpus)),
        model(std::move(model_id)),
        snap(dir),
        tok(dgpp::text::Tokenizer::load((snap / "tokenizer.json").string())),
        tpl(dgpp::text::ChatTemplate::load((snap / "chat_template.jinja").string())),
        folds_tool_messages(read_text_file((snap / "chat_template.jinja").string()).find("tool_call_id") !=
                            std::string::npos) {}
};

const World& world() {
  static const World w = [] {
    std::vector<std::string> lines;
    std::istringstream f(read_text_file(golden_path()));
    for (std::string line; std::getline(f, line);)
      if (!line.empty()) lines.push_back(line);
    require(!lines.empty(), "the golden corpus is empty");
    const auto header = dgpp::minijson::parse(lines[0]);
    std::string model(header.root.at("model").as_string());
    if (const char* env = std::getenv("DGPP_TP_REAL_MODEL"); env && *env) model = env;
    std::string err;
    const std::string snap = dgpp::hf::model_dir(model, &err);
    if (snap.empty()) {
      DGPP_LOG_WARN("gemma4_chat_test: model {} unavailable ({}); skipping", model, err);
      std::exit(2);
    }
    return World(std::move(lines), std::move(model), snap);
  }();
  return w;
}

}  // namespace

DGPP_TEST(gemma4_chat_template_differential_goldens) {
  const World& w = world();
  const auto header = dgpp::minijson::parse(w.lines[0]);
  // A corpus keyed to another template or tokenizer refuses rather than compares.
  require(header.root.at("template_hash").as_string() == hex16(w.tpl.source_hash()),
          "the goldens are keyed to another chat_template.jinja (" + hex16(w.tpl.source_hash()) +
              " here) — regenerate them (tools/gen_chat_template_goldens.py)");
  require(header.root.at("tokenizer_revision").as_string() == hex16(w.tok.revision_hash()),
          "the goldens are keyed to another tokenizer.json — regenerate them");
  require(w.tok.sentencepiece_bpe(), "the tokenizer is the SentencePiece-style shape");
  size_t checked = 0, ids_checked = 0;
  for (size_t i = 1; i < w.lines.size(); ++i) {
    const auto parsed = dgpp::minijson::parse(w.lines[i]);
    const std::string name(parsed.root.at("name").as_string());
    const std::string want(parsed.root.at("render").as_string());
    std::string got;
    try {
      got = w.tpl.render(dgpp::text::Value::from_minijson(parsed.root.at("kwargs")));
    } catch (const std::exception& e) {
      throw std::runtime_error("case " + name + ": the render threw: " + e.what());
    }
    if (got != want) {
      size_t at = 0;
      while (at < got.size() && at < want.size() && got[at] == want[at]) ++at;
      const size_t from = at > 40 ? at - 40 : 0;
      throw std::runtime_error("case " + name + ": render differs at byte " + std::to_string(at) + "\n  got : ..." +
                               got.substr(from, 120) + "\n  want: ..." + want.substr(from, 120));
    }
    if (const dgpp::minijson::Value* ids = parsed.root.find("ids")) {
      std::vector<int64_t> want_ids;
      for (const auto& v : ids->items()) want_ids.push_back(v.as_int());
      const std::vector<int64_t> got_ids = w.tok.encode(got);
      require(got_ids == want_ids, "case " + name + ": the render encodes to " + std::to_string(got_ids.size()) +
                                       " ids, the corpus has " + std::to_string(want_ids.size()));
      ++ids_checked;
    }
    ++checked;
  }
  require(checked == static_cast<size_t>(header.root.at("cases").as_int()), "corpus case count");
  DGPP_LOG_INFO("gemma4_chat_test: {} renders byte-exact, {} id sequences equal (model {}, template {})", checked,
                ids_checked, w.model, hex16(w.tpl.source_hash()));
}

DGPP_TEST(gemma4_chat_template_knobs) {
  const World& w = world();
  // What the service's knob gate asks: the template reads enable_thinking and
  // tools; it has no reasoning_effort.
  require(w.tpl.reads("enable_thinking") && w.tpl.reads("tools") && w.tpl.reads("bos_token"), "globals the template reads");
  require(!w.tpl.reads("reasoning_effort") && !w.tpl.reads("clear_thinking"), "globals it does not");
  // A tool message whose call id matches nothing and that names no tool is an
  // error in the template that folds tool messages (None + str): a render
  // error here too. The earlier template writes the message as a turn.
  const std::string text =
      R"({"bos_token": "<bos>", "add_generation_prompt": true, "messages": [
          {"role": "user", "content": "x"},
          {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
              "function": {"name": "f", "arguments": {}}}]},
          {"role": "tool", "tool_call_id": "other", "content": "y"}]})";
  const auto parsed = dgpp::minijson::parse(text);
  bool threw = false;
  try {
    (void)w.tpl.render(dgpp::text::Value::from_minijson(parsed.root));
  } catch (const std::exception&) {
    threw = true;
  }
  require(threw == w.folds_tool_messages, w.folds_tool_messages ? "an unmatched, unnamed tool response must not render"
                                                                : "a tool message is a turn of its own under this template");
}

namespace {

// Structural equality of two JSON values (object members in any order).
bool same_json(const dgpp::minijson::Value& a, const dgpp::minijson::Value& b) {
  if (a.is_object() && b.is_object()) {
    if (a.members().size() != b.members().size()) return false;
    for (const auto& m : a.members()) {
      const dgpp::minijson::Value* other = b.find(m.key);
      if (other == nullptr || !same_json(m.value, *other)) return false;
    }
    return true;
  }
  if (a.is_array() && b.is_array()) {
    if (a.items().size() != b.items().size()) return false;
    for (size_t i = 0; i < a.items().size(); ++i)
      if (!same_json(a.items()[i], b.items()[i])) return false;
    return true;
  }
  if (a.is_string() && b.is_string()) return a.as_string() == b.as_string();
  if (a.is_number() && b.is_number()) return a.as_double() == b.as_double();
  if (a.is_bool() && b.is_bool()) return a.as_bool() == b.as_bool();
  return a.is_null() && b.is_null();
}

// Whether a string anywhere in `v` holds the string token's own text.
bool holds_quote_token(const dgpp::minijson::Value& v) {
  if (v.is_string()) return v.as_string().find("<|\"|>") != std::string_view::npos;
  for (const auto& m : v.members())
    if (holds_quote_token(m.value)) return true;
  for (const auto& item : v.items())
    if (holds_quote_token(item)) return true;
  return false;
}

}  // namespace

DGPP_TEST(gemma4_markers_come_off_the_tokenizer) {
  const World& w = world();
  const dgpp::text::ChatMarkers m = dgpp::text::ChatMarkers::from_tokenizer(w.tok);
  // The release's ids (tokenizer.json): the channel pair, the call pair, the string token, the turn pair.
  require(m.think_open.id == 100 && m.think_close.id == 101 && m.channel_thinking, "the reasoning channel");
  require(m.tool_call_open.id == 48 && m.tool_call_close.id == 49 && m.string_quote.id == 52, "the call markers");
  require(m.tool_format() == dgpp::text::ToolFormat::kGemma, "the Gemma call format");
  require(m.role_markers.size() == 2 && m.role_markers[0].id == 105 && m.role_markers[1].id == 106, "the turn markers");
  require(m.newline.id == 107, "the bare newline");
  // The template's generation prompt leaves the channel to the model: thinking on ends at the
  // turn's newline, thinking off at a closed, empty channel.
  const std::vector<int64_t> on = w.tok.encode("<bos><|turn>user\nHi<turn|>\n<|turn>model\n");
  const std::vector<int64_t> off = w.tok.encode("<bos><|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
  require(!m.prompt_opens_thinking(on) && m.prompt_leaves_thinking_to_model(on), "thinking on");
  require(!m.prompt_opens_thinking(off) && m.prompt_leaves_thinking_to_model(off), "thinking off");
  // No grammar exists for the notation: the vocabulary reports unusable, so the engine offers
  // no constrained decoding for this family.
  const dgpp::text::GrammarVocab vocab =
      dgpp::text::GrammarVocab::from_tokenizer(w.tok, {1, 106, 50}, static_cast<int>(w.tok.max_id() + 1));
  require(!vocab.usable(), "no constrained decoding for the Gemma notation");
}

DGPP_TEST(gemma4_frontend_renders_the_goldens) {
  // The service's path: the request's globals without bos_token / add_generation_prompt — the
  // frontend supplies both.
  const World& w = world();
  const dgpp::serve::TextFrontend frontend(&w.tok, &w.tpl, /*json_calls=*/false, /*instruct_only=*/false, "<bos>");
  require(frontend.template_reads("enable_thinking"), "the thinking switch is the template's");
  require(frontend.reasoning_settings("none").enable_thinking == std::optional<bool>(false) &&
              frontend.reasoning_settings("high").enable_thinking == std::optional<bool>(true),
          "reasoning_effort maps onto enable_thinking");
  require(frontend.boundary_token_ids() == std::vector<int64_t>({105, 106}), "prefix-cache boundaries");
  size_t checked = 0;
  for (size_t i = 1; i < w.lines.size(); ++i) {
    const auto parsed = dgpp::minijson::parse(w.lines[i]);
    const dgpp::minijson::Value& kwargs = parsed.root.at("kwargs");
    if (!kwargs.at("add_generation_prompt").as_bool()) continue;
    std::vector<dgpp::minijson::Member> members;
    for (const auto& m : kwargs.members())
      if (m.key != "bos_token" && m.key != "add_generation_prompt") members.push_back(m);
    const std::string got = frontend.render_chat(dgpp::minijson::Value::make_object(std::move(members)));
    require(got == std::string(parsed.root.at("render").as_string()),
            "case " + std::string(parsed.root.at("name").as_string()) + ": the frontend's render differs");
    ++checked;
  }
  require(checked >= 25, "cases rendered through the frontend: " + std::to_string(checked));
}

DGPP_TEST(gemma4_tool_call_render_encode_parse_roundTrip) {
  // The template's own spelling of an assistant's calls, in the tokenizer's own pieces, parses
  // back to the calls: every model turn of the corpus that opens with calls written from a map.
  const World& w = world();
  const dgpp::text::ChatMarkers markers = dgpp::text::ChatMarkers::from_tokenizer(w.tok);
  size_t turns = 0, calls = 0, undecidable = 0;
  for (size_t i = 1; i < w.lines.size(); ++i) {
    const auto parsed = dgpp::minijson::parse(w.lines[i]);
    const std::string name(parsed.root.at("name").as_string());
    const std::string render(parsed.root.at("render").as_string());
    size_t from = 0;
    for (const auto& message : parsed.root.at("kwargs").at("messages").items()) {
      const dgpp::minijson::Value* tool_calls = message.find("tool_calls");
      if (tool_calls == nullptr || !tool_calls->is_array() || tool_calls->items().empty()) continue;
      bool maps = true;
      for (const auto& tc : tool_calls->items()) maps = maps && tc.at("function").at("arguments").is_object();
      if (!maps) continue;  // arguments handed over as a string are written raw, not in the notation
      // This message's calls: from its first <|tool_call> to the end of the run of call blocks
      // (the model stops there: <|tool_response> is one of its EOS ids).
      const size_t open = render.find("<|tool_call>", from);
      require(open != std::string::npos, "case " + name + ": no call block in the render");
      size_t stop = open;
      while (render.compare(stop, 12, "<|tool_call>") == 0) {
        const size_t close = render.find("<tool_call|>", stop);
        require(close != std::string::npos, "case " + name + ": an unclosed call block in the render");
        stop = close + 12;
      }
      from = stop;
      // What the model would have produced: any reasoning channel before, the calls, then the EOS.
      size_t begin = open;
      const size_t channel = render.rfind("<|channel>thought\n", open);
      const size_t turn = render.rfind("<|turn>model\n", open);
      if (channel != std::string::npos && turn != std::string::npos && channel > turn) begin = channel;
      const std::vector<int64_t> ids = w.tok.encode(render.substr(begin, stop - begin));
      dgpp::text::ToolCallParser::Options o;
      o.start_in_reasoning = false;
      o.model_may_open_thinking = true;
      dgpp::text::ToolCallParser parser(
          markers, [&](const std::vector<int64_t>& v) { return w.tok.decode(v, true); },
          dgpp::text::ToolSchemas(), o);
      std::vector<dgpp::text::ToolCallParser::Event> events;
      for (const int64_t id : ids) parser.feed(id, &events);
      parser.finish(&events);
      std::vector<dgpp::text::ToolCallParser::Call> got;
      std::string content, reasoning;
      for (const auto& e : events) {
        if (e.kind == dgpp::text::ToolCallParser::Event::Kind::kToolCall) got.push_back(e.call);
        else if (e.kind == dgpp::text::ToolCallParser::Event::Kind::kContent) content += e.text;
        else if (e.kind == dgpp::text::ToolCallParser::Event::Kind::kReasoning) reasoning += e.text;
      }
      // The notation cannot carry a string that holds its own delimiter (the template writes it
      // raw): such a block is not a call for transformers' reader either. Here it flushes whole,
      // markers restored, and no half-read call escapes.
      bool ambiguous = false;
      for (const auto& tc : tool_calls->items()) ambiguous = ambiguous || holds_quote_token(tc.at("function").at("arguments"));
      if (ambiguous) {
        require(got.empty() && content == render.substr(open, stop - open),
                "case " + name + ": an undecidable block must flush as its literal text");
        ++undecidable;
        continue;
      }
      require(content.empty(), "case " + name + ": stray content '" + content + "'");
      require(got.size() == tool_calls->items().size(),
              "case " + name + ": parsed " + std::to_string(got.size()) + " calls of " +
                  std::to_string(tool_calls->items().size()));
      for (size_t c = 0; c < got.size(); ++c) {
        const dgpp::minijson::Value& fn = tool_calls->items()[c].at("function");
        require(got[c].name == fn.at("name").as_string(), "case " + name + ": call name " + got[c].name);
        const auto args = dgpp::minijson::parse(got[c].arguments);
        require(same_json(args.root, fn.at("arguments")),
                "case " + name + ": arguments " + got[c].arguments + " differ from the message's");
        ++calls;
      }
      // A rendered thought comes back as the reasoning, header and the template's newline apart.
      if (begin != open) {
        const dgpp::minijson::Value* r = message.find("reasoning_content");
        if (r == nullptr) r = message.find("reasoning");
        require(r != nullptr && reasoning == std::string(r->as_string()) + "\n",
                "case " + name + ": reasoning '" + reasoning + "'");
      }
      ++turns;
    }
  }
  require(turns >= 8 && calls >= 10 && undecidable == 1,
          "tool-call turns " + std::to_string(turns) + ", calls " + std::to_string(calls) + ", undecidable " +
              std::to_string(undecidable));
  DGPP_LOG_INFO("gemma4_chat_test: {} tool-call turns ({} calls) round-trip render -> encode -> parse; {} undecidable "
                "block flushed as text",
                turns, calls, undecidable);
}

int main(int argc, char** argv) {
  g_argc = argc;
  g_argv = argv;
  return ::dgpp::test::run_all();
}
