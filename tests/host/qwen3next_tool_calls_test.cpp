// The Qwen3-Next JSON tool-call form over the real tokenizer and template
// (host; skips with rc 2 when the checkpoint is not in the model cache).
// The unit gates pin the parser and the grammar on synthetic ids; this one
// pins what only the checkpoint can say:
//   * the template renders as jinja2 does (the corpus,
//     tests/data/qwen3next_chat_template_goldens.jsonl, from
//     tools/gen_chat_template_goldens.py; its ids are compared where a
//     case carries them — a --render-only corpus has none);
//   * the tokenizer cannot tell this form from the XML one, so the
//     frontend and the grammar vocabulary state it;
//   * the template's own spelling of a call, in the tokenizer's own
//     pieces, parses back to the call and is admitted by the grammar id by
//     id — the pieces run across the frame's seams (" {\"" ends the head and
//     opens the arguments, "\"}}\n" closes a string, the arguments and the
//     frame), which is what the grammar's byte-stream reading is for.
#include <chrono>
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

using dgpp::text::GrammarSpec;
using dgpp::text::GrammarState;
using dgpp::text::GrammarVocab;
using dgpp::text::TokenMask;
using dgpp::text::ToolCallParser;
using dgpp::text::ToolFormat;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::string read_text_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string golden_path() {
  return g_argc > 1 ? g_argv[1] : "tests/data/qwen3next_chat_template_goldens.jsonl";
}

// The checkpoint, its corpus and what the service derives from them.
struct World {
  std::vector<std::string> lines;  // the corpus: header, then one case a line
  std::filesystem::path snap;
  dgpp::text::Tokenizer tok;
  dgpp::text::ChatTemplate tpl;
  std::vector<int64_t> eos;  // generation_config.json's list
  int64_t vocab = 0;
  int64_t im_end = -1;

  World(std::vector<std::string> corpus, const std::string& dir)
      : lines(std::move(corpus)),
        snap(dir),
        tok(dgpp::text::Tokenizer::load((snap / "tokenizer.json").string())),
        tpl(dgpp::text::ChatTemplate::load((snap / "chat_template.jinja").string())) {
    const std::string generation = read_text_file((snap / "generation_config.json").string());
    const dgpp::minijson::ParseResult g = dgpp::minijson::parse(generation);
    const dgpp::minijson::Value& e = g.root.at("eos_token_id");
    if (e.is_array())
      for (const auto& v : e.items()) eos.push_back(v.as_int());
    else
      eos.push_back(e.as_int());
    const std::string config = read_text_file((snap / "config.json").string());
    // A flat config names its vocabulary at the root; a nested one (the
    // Qwen3-VL-MoE corpus, 2026-10-04) in text_config.
    const dgpp::minijson::ParseResult cfg = dgpp::minijson::parse(config);
    const dgpp::minijson::Value* text_config = cfg.root.find("text_config");
    vocab = (cfg.root.find("vocab_size") != nullptr ? cfg.root : text_config != nullptr ? *text_config : cfg.root)
                .at("vocab_size")
                .as_int();
    const std::vector<int64_t> end = tok.encode("<|im_end|>");
    require(end.size() == 1, "<|im_end|> is one added token");
    im_end = end[0];
  }
};

World load_world() {
  std::vector<std::string> lines;
  std::istringstream f(read_text_file(golden_path()));
  for (std::string line; std::getline(f, line);)
    if (!line.empty()) lines.push_back(line);
  require(!lines.empty(), "the corpus is empty");
  std::string model(dgpp::minijson::parse(lines[0]).root.at("model").as_string());
  if (const char* env = std::getenv("DGPP_TP_REAL_MODEL"); env && *env) model = env;
  std::string err;
  const std::string snap = dgpp::hf::model_dir(model, &err);
  if (snap.empty()) {
    DGPP_LOG_WARN("qwen3next_tool_calls_test: model {} unavailable ({}); skipping", model, err);
    std::exit(2);
  }
  return World(std::move(lines), snap);
}

const World& world() {
  static const World w = load_world();
  return w;
}

std::string hex16(uint64_t v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
  return buf;
}

// An assistant turn of the corpus that carries tool calls: the ids the model
// would have produced for it (the template's rendering of the turn, without
// the header the prompt supplies, ending in <|im_end|>), and what it says.
struct Turn {
  std::string where;
  std::vector<int64_t> ids;
  std::string content;
  std::vector<ToolCallParser::Call> calls;
  std::string tools;  // the request's tools, JSON text
};

std::vector<Turn> golden_turns() {
  const World& w = world();
  std::vector<Turn> out;
  for (size_t i = 1; i < w.lines.size(); ++i) {
    const dgpp::minijson::ParseResult parsed = dgpp::minijson::parse(w.lines[i]);
    const dgpp::minijson::Value& kwargs = parsed.root.at("kwargs");
    const dgpp::minijson::Value& messages = kwargs.at("messages");
    const dgpp::minijson::Value* tools = kwargs.find("tools");
    for (size_t k = 0; k < messages.items().size(); ++k) {
      const dgpp::minijson::Value& msg = messages.items()[k];
      const dgpp::minijson::Value* tcs = msg.find("tool_calls");
      if (msg.at("role").as_string() != "assistant" || tcs == nullptr) continue;
      // A turn that carries reasoning (the field, or <think> inside its
      // content) pins the template's history rendering, not a form the
      // model writes: the Qwen3-235B corpus has two (2026-10-04).
      if (msg.find("reasoning_content") != nullptr || !msg.at("content").is_string() ||
          msg.at("content").as_string().find("<think>") != std::string_view::npos)
        continue;
      const auto render_through = [&](size_t count) {
        std::vector<dgpp::text::Value> ms;
        for (size_t j = 0; j < count; ++j) ms.push_back(dgpp::text::Value::from_minijson(messages.items()[j]));
        dgpp::text::Value::Members g;
        g.emplace_back("messages", dgpp::text::Value::list_value(std::move(ms)));
        if (tools) g.emplace_back("tools", dgpp::text::Value::from_minijson(*tools));
        g.emplace_back("add_generation_prompt", dgpp::text::Value::boolean(false));
        return w.tpl.render(dgpp::text::Value::map_value(std::move(g)));
      };
      Turn turn;
      turn.where = std::string(parsed.root.at("name").as_string()) + " message " + std::to_string(k);
      const std::string before = render_through(k);
      const std::string through = render_through(k + 1);
      require(through.compare(0, before.size(), before) == 0, turn.where + ": the render grows by the turn");
      std::string text = through.substr(before.size());
      const std::string kAssistant = "<|im_start|>assistant\n", kEnd = "<|im_end|>\n";
      require(text.compare(0, kAssistant.size(), kAssistant) == 0 && text.size() >= kAssistant.size() + kEnd.size() &&
                  text.compare(text.size() - kEnd.size(), kEnd.size(), kEnd) == 0,
              turn.where + ": the turn sits between the assistant header and <|im_end|>: " + text);
      text = text.substr(kAssistant.size(), text.size() - kAssistant.size() - kEnd.size());
      // The Qwen3-235B-2507 template renders a TRAILING assistant turn
      // behind an empty reasoning block (the Qwen3 reasoning-history branch
      // it inherits; the Instruct checkpoint writes none): template text,
      // not part of what the model produces.
      const std::string kEmptyThink = "<think>\n\n</think>\n\n";
      if (text.compare(0, kEmptyThink.size(), kEmptyThink) == 0) text = text.substr(kEmptyThink.size());
      turn.ids = w.tok.encode(text);
      turn.ids.push_back(w.im_end);
      // The template writes the content, a newline before the first block
      // when there is content, and a newline between blocks.
      turn.content = std::string(msg.at("content").as_string());
      if (!turn.content.empty()) turn.content += "\n";
      for (size_t c = 0; c < tcs->items().size(); ++c) {
        if (c != 0) turn.content += "\n";
        const dgpp::minijson::Value& fn = tcs->items()[c].at("function");
        ToolCallParser::Call call;
        call.name = std::string(fn.at("name").as_string());
        call.arguments = dgpp::text::Value::from_minijson(fn.at("arguments")).to_json(false);
        turn.calls.push_back(std::move(call));
      }
      if (tools) turn.tools = dgpp::text::json_text_of(*tools);
      out.push_back(std::move(turn));
    }
  }
  require(!out.empty(), "the corpus carries assistant turns with tool calls");
  return out;
}

DGPP_TEST(qwen3next_template_renders_as_the_goldens) {
  const World& w = world();
  const dgpp::minijson::ParseResult header_doc = dgpp::minijson::parse(w.lines[0]);
  const dgpp::minijson::Value& header = header_doc.root;
  require(std::string(header.at("template_hash").as_string()) == hex16(w.tpl.source_hash()),
          "the corpus is keyed to another chat_template.jinja — regenerate it (tools/gen_chat_template_goldens.py)");
  require(std::string(header.at("tokenizer_revision").as_string()) == hex16(w.tok.revision_hash()),
          "the corpus is keyed to another tokenizer.json");
  size_t with_ids = 0;
  for (size_t i = 1; i < w.lines.size(); ++i) {
    const dgpp::minijson::ParseResult parsed = dgpp::minijson::parse(w.lines[i]);
    const std::string name(parsed.root.at("name").as_string());
    const std::string want(parsed.root.at("render").as_string());
    const std::string got = w.tpl.render(dgpp::text::Value::from_minijson(parsed.root.at("kwargs")));
    if (got != want) {
      size_t off = 0;
      while (off < got.size() && off < want.size() && got[off] == want[off]) ++off;
      require(false, "case '" + name + "' diverges at byte " + std::to_string(off) + ": expected " +
                         want.substr(off < 12 ? 0 : off - 12, 24) + ", got " + got.substr(off < 12 ? 0 : off - 12, 24));
    }
    // The generation prompt opens no reasoning block: the reply starts as
    // content, and the parser with it.
    const std::vector<int64_t> ids = w.tok.encode(got);
    const dgpp::text::ChatMarkers markers = dgpp::text::ChatMarkers::from_tokenizer(w.tok);
    require(!markers.prompt_opens_thinking(ids) && !markers.prompt_leaves_thinking_to_model(ids),
            "case '" + name + "': the prompt neither opens a reasoning block nor leaves one to the model");
    if (const dgpp::minijson::Value* want_ids = parsed.root.find("ids")) {
      std::vector<int64_t> reference;
      for (const auto& v : want_ids->items()) reference.push_back(v.as_int(-1));
      require(ids == reference, "case '" + name + "': the render's ids differ from the reference tokenizer's");
      ++with_ids;
    }
  }
  DGPP_LOG_INFO("qwen3next_tool_calls_test: {} renders byte-exact, {} with reference ids", w.lines.size() - 1,
                with_ids);
}

DGPP_TEST(qwen3next_tool_format_is_stated_not_read_off_the_tokenizer) {
  const World& w = world();
  // The tokenizer carries the two <tool_call> tokens and no GLM argument
  // markers — exactly what Qwen3.8's carries — so it reads as the XML
  // format until the family says otherwise.
  const dgpp::text::ChatMarkers inferred = dgpp::text::ChatMarkers::from_tokenizer(w.tok);
  require(inferred.tool_call_open.available() && inferred.tool_call_close.available() &&
              !inferred.arg_key_open.available() && !inferred.dsml.available() && !inferred.xml_compact,
          "the two markers, nothing else");
  require(inferred.tool_format() == ToolFormat::kQwenXml, "the tokenizer alone reads as the XML format");
  require(dgpp::serve::TextFrontend(&w.tok, &w.tpl).markers().tool_format() == ToolFormat::kQwenXml,
          "a frontend that states nothing keeps the XML reading");
  const dgpp::serve::TextFrontend frontend(&w.tok, &w.tpl, /*json_calls=*/true);
  require(frontend.markers().tool_format() == ToolFormat::kQwenJson, "the frontend states the JSON form");
  const GrammarVocab vocab = GrammarVocab::from_tokenizer(w.tok, w.eos, static_cast<int>(w.vocab),
                                                          dgpp::text::DsmlDialect::kV41, /*json_calls=*/true);
  require(vocab.usable() && vocab.markers().tool_format() == ToolFormat::kQwenJson,
          "the grammar vocabulary states the same");
  require(GrammarVocab::from_tokenizer(w.tok, w.eos, static_cast<int>(w.vocab)).markers().tool_format() ==
              ToolFormat::kQwenXml,
          "and reads as XML when it is not told");
  // The turn after a call ends at <|im_end|>, one of generation_config.json's
  // EOS ids; the markers decode literally (they are not special tokens), so
  // a malformed block comes back as the text the model wrote.
  require(vocab.call_turn_eos() == w.im_end && vocab.is_eos(w.im_end), "the call turn ends at <|im_end|>");
  require(w.tok.decode({frontend.markers().tool_call_open.id, frontend.markers().tool_call_close.id}, true) ==
              "<tool_call></tool_call>",
          "the markers decode to their text");
}

DGPP_TEST(qwen3next_tool_call_render_encode_parse_roundTrip) {
  const World& w = world();
  const dgpp::serve::TextFrontend frontend(&w.tok, &w.tpl, /*json_calls=*/true);
  size_t calls = 0;
  for (const Turn& turn : golden_turns()) {
    ToolCallParser::Options opts;
    opts.start_in_reasoning = false;  // the prompt ends in "assistant\n"
    require(!turn.tools.empty(), turn.where + ": the case declares its tools");
    const dgpp::minijson::ParseResult tools = dgpp::minijson::parse(turn.tools);
    ToolCallParser parser(
        frontend.markers(), [&](const std::vector<int64_t>& v) { return frontend.decode_ids(v); },
        dgpp::text::ToolSchemas(tools.root), opts);
    std::vector<ToolCallParser::Event> events;
    for (const int64_t id : turn.ids) parser.feed(id, &events);
    parser.finish(&events);
    std::string content;
    std::vector<ToolCallParser::Call> got;
    for (const auto& ev : events) {
      require(ev.kind == ToolCallParser::Event::Kind::kContent || ev.kind == ToolCallParser::Event::Kind::kToolCall,
              turn.where + ": content and calls only");
      if (ev.kind == ToolCallParser::Event::Kind::kContent) content += ev.text;
      else got.push_back(ev.call);
    }
    require(content == turn.content, turn.where + ": the content around the blocks: '" + content + "'");
    require(got.size() == turn.calls.size(), turn.where + ": every block is a call");
    for (size_t c = 0; c < got.size(); ++c) {
      require(got[c].name == turn.calls[c].name, turn.where + ": the name " + got[c].name);
      require(got[c].arguments == turn.calls[c].arguments, turn.where + ": the arguments " + got[c].arguments);
    }
    calls += got.size();
    // The same ids under the XML reading: no call, the blocks literal content.
    ToolCallParser xml(
        dgpp::text::ChatMarkers::from_tokenizer(w.tok),
        [&](const std::vector<int64_t>& v) { return frontend.decode_ids(v); }, dgpp::text::ToolSchemas(tools.root),
        opts);
    std::vector<ToolCallParser::Event> xml_events;
    for (const int64_t id : turn.ids) xml.feed(id, &xml_events);
    xml.finish(&xml_events);
    require(xml.calls() == 0, turn.where + ": the XML reading finds no call in a JSON block");
  }
  DGPP_LOG_INFO("qwen3next_tool_calls_test: {} calls round-tripped through the real tokenizer", calls);
}

DGPP_TEST(qwen3next_tool_grammar_accepts_the_golden_turns_over_the_real_tokenizer) {
  const World& w = world();
  const auto t0 = std::chrono::steady_clock::now();
  const GrammarVocab vocab = GrammarVocab::from_tokenizer(w.tok, w.eos, static_cast<int>(w.vocab),
                                                          dgpp::text::DsmlDialect::kV41, /*json_calls=*/true);
  vocab.prepare_json();
  const double build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const int64_t open = vocab.markers().tool_call_open.id, close = vocab.markers().tool_call_close.id;
  size_t positions = 0, straddles = 0;
  double total_us = 0.0, max_us = 0.0;
  std::string max_where;
  // One turn under one spec: every id inside the mask, the mask and the
  // pointwise rule agreeing, the turn ending on its EOS.
  const auto walk = [&](const Turn& turn, const GrammarSpec& spec, const std::vector<int64_t>& ids,
                        const std::string& mode) {
    GrammarState g(&vocab, spec, /*prompt_opens_thinking=*/false);
    for (size_t j = 0; j < ids.size(); ++j) {
      TokenMask m;
      const auto a = std::chrono::steady_clock::now();
      g.mask(&m);
      const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count();
      ++positions;
      total_us += us;
      if (us > max_us) {
        max_us = us;
        max_where = g.state_name();
      }
      const std::string at = turn.where + " (" + mode + ") id " + std::to_string(ids[j]) + " '" +
                             vocab.text(ids[j]) + "' in state " + g.state_name();
      require(m.allows(ids[j]) == g.allows(ids[j]), "mask and allows() disagree at " + at);
      require(m.allows(ids[j]), "the template's own tokenization is refused at " + at);
      // A piece that ends the head with the object's brace inside it, or
      // closes the object and the frame together: the seams the byte-stream
      // reading exists for.
      const std::string before = g.state_name();
      const std::string& piece = vocab.text(ids[j]);
      g.advance(ids[j]);
      const std::string after = g.state_name();
      if ((before == "j-head" && after != "j-head" && piece.find('{') != std::string::npos) ||
          (before == "j-args" && after == "q-close" && piece.size() >= 3 &&
           piece.compare(piece.size() - 3, 3, "}}\n") == 0))
        ++straddles;
    }
    require(g.active(), turn.where + " (" + mode + "): the grammar is alive at the end");
  };
  for (const Turn& turn : golden_turns()) {
    require(!turn.tools.empty(), turn.where + ": the case declares its tools");
    const dgpp::minijson::ParseResult tools = dgpp::minijson::parse(turn.tools);
    GrammarSpec spec;
    for (const dgpp::minijson::Value& t : tools.root.items()) {
      const dgpp::minijson::Value* fn = t.find("function");
      spec.tools.push_back(
          dgpp::text::grammar_tool_from_function(fn ? *fn : t, nullptr, nullptr, ToolFormat::kQwenJson));
    }
    // auto: the whole turn, prose included.
    spec.mode = GrammarSpec::Mode::kAuto;
    walk(turn, spec, turn.ids, "auto");
    // required: a call is owed, so the turn is its blocks (the opener is
    // forced; prose before it is the unconstrained turn's).
    std::vector<int64_t> blocks;
    for (size_t j = 0; j < turn.ids.size(); ++j)
      if (turn.ids[j] == open) {
        blocks.assign(turn.ids.begin() + static_cast<std::ptrdiff_t>(j), turn.ids.end());
        break;
      }
    require(!blocks.empty(), turn.where + ": the turn opens a block");
    spec.mode = GrammarSpec::Mode::kRequired;
    walk(turn, spec, blocks, "required");
    // named: the first block alone, then the turn's end.
    std::vector<int64_t> first;
    for (const int64_t id : blocks) {
      first.push_back(id);
      if (id == close) break;
    }
    first.push_back(w.im_end);
    spec.mode = GrammarSpec::Mode::kNamed;
    spec.named = turn.calls[0].name;
    spec.parallel = false;
    walk(turn, spec, first, "named");
    // The XML spelling of the same call is outside this grammar.
    GrammarState x(&vocab, spec, false);
    x.advance(open);
    bool refused = false;
    for (const int64_t id : w.tok.encode("\n<function=" + turn.calls[0].name + ">\n</function>\n")) {
      if (!x.allows(id)) {
        refused = true;
        break;
      }
      x.advance(id);
    }
    require(refused, turn.where + ": the XML form is refused under the JSON grammar");
  }
  require(straddles > 0, "the tokenizer's pieces cross the frame's seams in the corpus");
  // The whole vocabulary at every position of one call: the mask is the
  // pointwise rule, id for id.
  size_t compared = 0, compared_positions = 0;
  {
    GrammarSpec spec;
    spec.mode = GrammarSpec::Mode::kRequired;
    const dgpp::minijson::ParseResult def = dgpp::minijson::parse(
        R"({"name":"get_weather","strict":true,"parameters":{"type":"object","properties":{)"
        R"("city":{"type":"string"},"days":{"type":"integer"}},"required":["city"],"additionalProperties":false}})");
    spec.tools.push_back(dgpp::text::grammar_tool_from_function(def.root, nullptr, nullptr, ToolFormat::kQwenJson));
    std::vector<int64_t> ids =
        w.tok.encode("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"São Paulo\", \"days\": 3}}\n</tool_call>");
    ids.push_back(w.im_end);
    GrammarState g(&vocab, spec, false);
    for (const int64_t id : ids) {
      TokenMask m;
      g.mask(&m);
      int counted = 0;
      for (int64_t v = 0; v < w.vocab; ++v) {
        const bool pointwise = g.allows(v);
        require(m.allows(v) == pointwise, std::string("mask and allows() disagree on id ") + std::to_string(v) + " '" +
                                              vocab.text(v) + "' in state " + g.state_name());
        counted += pointwise ? 1 : 0;
        ++compared;
      }
      ++compared_positions;
      require(counted == m.allowed, std::string("the allowed count is the set's size in state ") + g.state_name());
      require(m.allows(id), std::string("the call's own id refused in state ") + g.state_name());
      g.advance(id);
    }
    require(std::string(g.state_name()) == "done", std::string("the strict call ended the turn: ") + g.state_name());
  }
  DGPP_LOG_INFO(
      "qwen3next_tool_calls_test: grammar vocabulary and JSON tables built in {:.2f} s; {} positions, mask {:.1f} us "
      "avg, {:.0f} us max (at {}); {} ids crossed a seam of the frame; mask == allows() over {} ids at {} positions",
      build_s, positions, total_us / static_cast<double>(positions), max_us, max_where, straddles, compared,
      compared_positions);
}

}  // namespace

int main(int argc, char** argv) {
  g_argc = argc;
  g_argv = argv;
  return ::dgpp::test::run_all();
}
