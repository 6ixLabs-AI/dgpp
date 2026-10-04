// MiniMax-M2.7 over its real tokenizer and template (host; skips with rc 2
// when the checkpoint is not in the model cache). The unit gates pin the
// parser on synthetic ids; this one pins what only the checkpoint can say:
//   * the template renders as jinja2 does, its raise_exception message
//     included (the corpus, tests/data/minimax_chat_template_goldens.jsonl,
//     from tools/gen_chat_template_goldens.py; its ids are compared where a
//     case carries them), and its generation prompt opens the reasoning;
//   * the markers are the tokenizer's own — <think> 200050, </think> 200051,
//     <minimax:tool_call> 200052, </minimax:tool_call> 200053 for the pinned
//     revision, none of them special — so the format is read off the
//     tokenizer and a malformed block comes back as its literal text;
//   * the template's own spelling of an assistant turn, in the tokenizer's
//     own pieces, parses back to its reasoning, content and calls — the
//     arguments typed from the request's schema — however the pieces are cut;
//   * the forced-call grammar does not cover this format: the vocabulary
//     says so (an engine then reports no constrained decoding).
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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
using dgpp::text::ToolCallParser;
using dgpp::text::ToolFormat;
using Kind = dgpp::text::ToolCallParser::Event::Kind;

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
  return g_argc > 1 ? g_argv[1] : "tests/data/minimax_chat_template_goldens.jsonl";
}

// The checkpoint, its corpus and what the service derives from them.
struct World {
  std::vector<std::string> lines;  // the corpus: header, then one case a line
  std::filesystem::path snap;
  dgpp::text::Tokenizer tok;
  dgpp::text::ChatTemplate tpl;
  std::vector<int64_t>
      eos;  // generation_config.json's (config.json's eos_token_id is not the turn's end)
  int64_t turn_end = -1;  // "[e~["

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
    const std::vector<int64_t> end = tok.encode("[e~[");
    require(end.size() == 1, "[e~[ is one added token");
    turn_end = end[0];
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
    DGPP_LOG_WARN("minimax_tool_calls_test: model {} unavailable ({}); skipping", model, err);
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

std::string trim_newlines(std::string x) {
  const size_t b = x.find_first_not_of('\n');
  if (b == std::string::npos) return std::string();
  return x.substr(b, x.find_last_not_of('\n') - b + 1);
}

struct Parsed {
  std::string reasoning, content;
  int reasoning_closed = 0;
  std::vector<ToolCallParser::Call> calls;
  std::vector<Kind> order;
};

// The ids as the service parses them: the frontend's markers and decode,
// the request's tool schemas; `in_reasoning` when the prompt ended in
// "<think>\n" (every generation prompt of this template does).
Parsed parse(const dgpp::serve::TextFrontend& frontend, const std::vector<int64_t>& ids,
             bool in_reasoning, const std::string& tools = "", size_t* calls_at = nullptr) {
  ToolCallParser::Options opts;
  opts.start_in_reasoning = in_reasoning;
  const dgpp::minijson::ParseResult schemas =
      dgpp::minijson::parse(tools.empty() ? std::string_view("[]") : tools);
  ToolCallParser parser(
      frontend.markers(), [&](const std::vector<int64_t>& v) { return frontend.decode_ids(v); },
      dgpp::text::ToolSchemas(schemas.root), opts);
  std::vector<ToolCallParser::Event> events;
  for (size_t i = 0; i < ids.size(); ++i) {
    const size_t before = events.size();
    parser.feed(ids[i], &events);
    if (calls_at != nullptr)
      for (size_t k = before; k < events.size(); ++k)
        if (events[k].kind == Kind::kToolCall && *calls_at == static_cast<size_t>(-1))
          *calls_at = i;
  }
  parser.finish(&events);
  Parsed out;
  for (const auto& ev : events) {
    out.order.push_back(ev.kind);
    if (ev.kind == Kind::kReasoning) out.reasoning += ev.text;
    if (ev.kind == Kind::kReasoningClosed) ++out.reasoning_closed;
    if (ev.kind == Kind::kContent) out.content += ev.text;
    if (ev.kind == Kind::kToolCall) out.calls.push_back(ev.call);
  }
  return out;
}

// An assistant turn of the corpus: the ids the model would have produced for
// it — the template's rendering of the turn after the header "]~b]ai\n" and,
// where the turn renders its reasoning, after the "<think>\n" a generation
// prompt ends in — through "[e~[", and what it says.
struct Turn {
  std::string where;
  std::vector<int64_t> ids;
  bool in_reasoning = false;  // the turn carries reasoning: its ids follow the prompt's "<think>\n"
  std::string reasoning, content;
  std::vector<ToolCallParser::Call> calls;
  std::string tools;  // the request's tools, JSON text ("" when it has none)
};

std::vector<Turn> golden_turns() {
  const World& w = world();
  std::vector<Turn> out;
  for (size_t i = 1; i < w.lines.size(); ++i) {
    const dgpp::minijson::ParseResult parsed = dgpp::minijson::parse(w.lines[i]);
    if (parsed.root.find("error") != nullptr) continue;
    const dgpp::minijson::Value& kwargs = parsed.root.at("kwargs");
    const dgpp::minijson::Value& messages = kwargs.at("messages");
    for (size_t k = 0; k < messages.items().size(); ++k) {
      const dgpp::minijson::Value& msg = messages.items()[k];
      if (msg.at("role").as_string() != "assistant") continue;
      const dgpp::minijson::Value* content = msg.find("content");
      // The service always sends a string content (null becomes ""): the
      // cases that pin the template's own handling of a missing or null
      // content, and of reasoning carried inside the content, are renders.
      if (content == nullptr || !content->is_string() ||
          content->as_string().find("</think>") != std::string_view::npos)
        continue;
      // The conversation as it stood when the model wrote the turn: its
      // prefix through the turn (so the turn follows the prefix's last user
      // message and renders its reasoning), minus the prefix before it.
      const auto render_through = [&](size_t count) {
        std::vector<dgpp::text::Value> ms;
        for (size_t j = 0; j < count; ++j)
          ms.push_back(dgpp::text::Value::from_minijson(messages.items()[j]));
        dgpp::text::Value::Members g;
        for (const dgpp::minijson::Member& m : kwargs.members())
          if (m.key != "messages" && m.key != "add_generation_prompt")
            g.emplace_back(m.key, dgpp::text::Value::from_minijson(m.value));
        g.emplace_back("messages", dgpp::text::Value::list_value(std::move(ms)));
        g.emplace_back("add_generation_prompt", dgpp::text::Value::boolean(false));
        return w.tpl.render(dgpp::text::Value::map_value(std::move(g)));
      };
      Turn turn;
      turn.where =
          std::string(parsed.root.at("name").as_string()) + " message " + std::to_string(k);
      const std::string before = render_through(k);
      const std::string through = render_through(k + 1);
      require(through.compare(0, before.size(), before) == 0,
              turn.where + ": the render grows by the turn");
      std::string text = through.substr(before.size());
      const std::string kHeader = "]~b]ai\n", kEnd = "[e~[\n", kThink = "<think>\n";
      require(text.compare(0, kHeader.size(), kHeader) == 0 &&
                  text.size() >= kHeader.size() + kEnd.size() &&
                  text.compare(text.size() - kEnd.size(), kEnd.size(), kEnd) == 0,
              turn.where + ": the turn sits between the ai header and [e~[: " + text);
      text = text.substr(kHeader.size(),
                         text.size() - kHeader.size() - 1);  // keep "[e~[", drop its newline
      if (text.compare(0, kThink.size(), kThink) == 0) {
        turn.in_reasoning = true;
        text.erase(0, kThink.size());
      }
      turn.ids = w.tok.encode(text);
      require(!turn.ids.empty() && turn.ids.back() == w.turn_end,
              turn.where + ": the turn ends at [e~[");
      if (const dgpp::minijson::Value* r = msg.find("reasoning_content");
          r != nullptr && r->is_string())
        turn.reasoning = std::string(r->as_string());
      require(turn.in_reasoning == !turn.reasoning.empty(),
              turn.where + ": the turn renders the reasoning it carries");
      turn.content = std::string(content->as_string());
      if (const dgpp::minijson::Value* tcs = msg.find("tool_calls");
          tcs != nullptr && tcs->is_array())
        for (const dgpp::minijson::Value& tc : tcs->items()) {
          const dgpp::minijson::Value* fn = tc.find("function");
          const dgpp::minijson::Value& def = fn ? *fn : tc;
          ToolCallParser::Call call;
          call.name = std::string(def.at("name").as_string());
          call.arguments = dgpp::text::Value::from_minijson(def.at("arguments")).to_json(false);
          turn.calls.push_back(std::move(call));
        }
      if (const dgpp::minijson::Value* tools = kwargs.find("tools");
          tools != nullptr && tools->is_array())
        turn.tools = dgpp::text::json_text_of(*tools);
      out.push_back(std::move(turn));
    }
  }
  require(!out.empty(), "the corpus carries assistant turns");
  return out;
}

DGPP_TEST(minimax_template_renders_as_the_goldens) {
  const World& w = world();
  const dgpp::minijson::ParseResult header_doc = dgpp::minijson::parse(w.lines[0]);
  const dgpp::minijson::Value& header = header_doc.root;
  require(std::string(header.at("template_hash").as_string()) == hex16(w.tpl.source_hash()),
          "the corpus is keyed to another chat_template.jinja — regenerate it "
          "(tools/gen_chat_template_goldens.py)");
  require(std::string(header.at("tokenizer_revision").as_string()) == hex16(w.tok.revision_hash()),
          "the corpus is keyed to another tokenizer.json");
  const dgpp::text::ChatMarkers markers = dgpp::text::ChatMarkers::from_tokenizer(w.tok);
  size_t renders = 0, errors = 0, with_ids = 0, prompts = 0;
  for (size_t i = 1; i < w.lines.size(); ++i) {
    const dgpp::minijson::ParseResult parsed = dgpp::minijson::parse(w.lines[i]);
    const std::string name(parsed.root.at("name").as_string());
    std::string got, thrown;
    try {
      got = w.tpl.render(dgpp::text::Value::from_minijson(parsed.root.at("kwargs")));
    } catch (const std::exception& e) {
      thrown = e.what();
    }
    // A raise_exception of the template is a render error that carries the
    // template's message and nothing else — what the service answers 400 with.
    if (const dgpp::minijson::Value* error = parsed.root.find("error")) {
      require(thrown == "chat-template: " + std::string(error->as_string()),
              "case '" + name + "': expected the template's refusal '" +
                  std::string(error->as_string()) +
                  "', got: " + (thrown.empty() ? "a render" : thrown));
      ++errors;
      continue;
    }
    require(thrown.empty(), "case '" + name + "': " + thrown);
    const std::string want(parsed.root.at("render").as_string());
    if (got != want) {
      size_t off = 0;
      while (off < got.size() && off < want.size() && got[off] == want[off]) ++off;
      require(false, "case '" + name + "' diverges at byte " + std::to_string(off) + ": expected " +
                         want.substr(off < 12 ? 0 : off - 12, 24) + ", got " +
                         got.substr(off < 12 ? 0 : off - 12, 24));
    }
    ++renders;
    const std::vector<int64_t> ids = w.tok.encode(got);
    // The generation prompt ends in "<think>\n": the reply starts inside
    // the reasoning, and the parser with it.
    const dgpp::minijson::Value* gen = parsed.root.at("kwargs").find("add_generation_prompt");
    const bool prompt = gen != nullptr && gen->is_bool() && gen->as_bool();
    require(markers.prompt_opens_thinking(ids) == prompt &&
                !markers.prompt_leaves_thinking_to_model(ids),
            "case '" + name + "': a generation prompt opens the reasoning, and only it does");
    prompts += prompt ? 1 : 0;
    if (const dgpp::minijson::Value* want_ids = parsed.root.find("ids")) {
      std::vector<int64_t> reference;
      for (const auto& v : want_ids->items()) reference.push_back(v.as_int(-1));
      require(ids == reference,
              "case '" + name + "': the render's ids differ from the reference tokenizer's (" +
                  std::to_string(ids.size()) + " vs " + std::to_string(reference.size()) + ")");
      ++with_ids;
    }
  }
  require(renders >= 24 && errors >= 2 && prompts >= 18,
          "the corpus covers the renders and the template's refusal");
  DGPP_LOG_INFO(
      "minimax_tool_calls_test: {} renders byte-exact ({} with reference ids, {} generation "
      "prompts), {} "
      "template refusals by message",
      renders, with_ids, prompts, errors);
}

DGPP_TEST(minimax_markers_are_read_off_the_tokenizer) {
  const World& w = world();
  const dgpp::text::ChatMarkers m = dgpp::text::ChatMarkers::from_tokenizer(w.tok);
  require(m.tool_format() == ToolFormat::kMinimaxXml && m.minimax_invoke &&
              m.tool_calls_available() && m.reasoning_available() && !m.bracket_think &&
              !m.xml_compact && !m.json_calls && !m.arg_key_open.available() &&
              !m.dsml.available() && !m.tool_calls.available(),
          "the tokenizer alone says MiniMax: nothing is stated by the family");
  require(
      dgpp::serve::TextFrontend(&w.tok, &w.tpl).markers().tool_format() == ToolFormat::kMinimaxXml,
      "a frontend that states nothing reads the same");
  DGPP_LOG_INFO(
      "minimax_tool_calls_test: markers <think> {} </think> {} <minimax:tool_call> {} "
      "</minimax:tool_call> {} "
      "[e~[ {} newline {}",
      m.think_open.id, m.think_close.id, m.tool_call_open.id, m.tool_call_close.id, w.turn_end,
      m.newline.id);
  // The pinned revision's ids (tokenizer.json fbebca183bacb5f8).
  require(m.think_open.id == 200050 && m.think_close.id == 200051 &&
              m.tool_call_open.id == 200052 && m.tool_call_close.id == 200053 &&
              w.turn_end == 200020,
          "the marker ids of the pinned tokenizer");
  require(w.eos == std::vector<int64_t>{200020}, "generation_config.json ends a turn at [e~[");
  // The four markers are not special: the decode prints them (a malformed
  // block is literal text); the turn's end and the role header are special.
  require(w.tok.decode({m.think_open.id, m.think_close.id, m.tool_call_open.id,
                        m.tool_call_close.id, w.turn_end},
                       true) == "<think></think><minimax:tool_call></minimax:tool_call>",
          "the markers decode to their text, [e~[ to nothing");
  std::vector<int64_t> roles;
  for (const dgpp::text::ChatMarker& r : m.role_markers) roles.push_back(r.id);
  require(roles == std::vector<int64_t>{200019}, "the role header ]~b] is the one boundary token");
  require(m.newline.available(), "the bare newline is one id");
  // The template has no reasoning knob and reads no special-token global:
  // its BOS and turn ends are literals.
  require(w.tpl.reads("tools") && w.tpl.reads("add_generation_prompt") &&
              w.tpl.reads("model_identity") && !w.tpl.reads("enable_thinking") &&
              !w.tpl.reads("reasoning_effort") && !w.tpl.reads("bos_token") &&
              !w.tpl.reads("eos_token"),
          "the template's globals");
  // The forced-call grammar does not cover the format: the vocabulary is
  // not usable, so an engine reports no constrained decoding, and a grammar
  // built on it anyway refuses by name.
  const GrammarVocab vocab =
      GrammarVocab::from_tokenizer(w.tok, w.eos, static_cast<int>(w.tok.max_id() + 1));
  require(!vocab.usable() && vocab.markers().tool_format() == ToolFormat::kMinimaxXml,
          "the vocabulary is not usable");
  GrammarSpec spec;
  spec.mode = GrammarSpec::Mode::kRequired;
  std::string refusal;
  try {
    GrammarState g(&vocab, spec, true);
  } catch (const std::invalid_argument& e) {
    refusal = e.what();
  }
  require(refusal.find("MiniMax-M2") != std::string::npos,
          "the grammar refuses the format by name: " + refusal);
}

DGPP_TEST(minimax_tool_call_render_encode_parse_roundTrip) {
  const World& w = world();
  const dgpp::serve::TextFrontend frontend(&w.tok, &w.tpl);
  size_t turns = 0, calls = 0, reasoned = 0;
  for (const Turn& turn : golden_turns()) {
    const Parsed got = parse(frontend, turn.ids, turn.in_reasoning, turn.tools);
    // The template separates the parts with newlines of its own ("\n</think>
    // \n\n" after the reasoning, "\n" before the block).
    require(trim_newlines(got.reasoning) == trim_newlines(turn.reasoning),
            turn.where + ": reasoning '" + got.reasoning + "' != '" + turn.reasoning + "'");
    require(trim_newlines(got.content) == trim_newlines(turn.content),
            turn.where + ": content '" + got.content + "' != '" + turn.content + "'");
    require(got.reasoning_closed == (turn.in_reasoning ? 1 : 0),
            turn.where + ": the reasoning closes once");
    require(got.calls.size() == turn.calls.size(),
            turn.where + ": " + std::to_string(got.calls.size()) + " calls parsed, " +
                std::to_string(turn.calls.size()) + " expected");
    for (size_t c = 0; c < got.calls.size(); ++c) {
      require(got.calls[c].name == turn.calls[c].name,
              turn.where + ": the name " + got.calls[c].name);
      require(got.calls[c].arguments == turn.calls[c].arguments,
              turn.where + ": the arguments " + got.calls[c].arguments +
                  " != " + turn.calls[c].arguments);
    }
    calls += got.calls.size();
    reasoned += turn.in_reasoning ? 1 : 0;
    ++turns;
  }
  require(turns >= 12 && calls >= 9 && reasoned >= 6,
          "the corpus' turns: " + std::to_string(turns) + " turns, " + std::to_string(calls) +
              " calls, " + std::to_string(reasoned) + " with reasoning");
  DGPP_LOG_INFO(
      "minimax_tool_calls_test: {} turns ({} calls, {} with reasoning) round-trip render -> encode "
      "-> parse",
      turns, calls, reasoned);
}

DGPP_TEST(minimax_calls_stream_split_and_fall_back_over_the_real_tokenizer) {
  const World& w = world();
  const dgpp::serve::TextFrontend frontend(&w.tok, &w.tpl);
  const dgpp::text::ChatMarkers& m = frontend.markers();
  const std::string tools =
      R"([{"type":"function","function":{"name":"search","parameters":{"type":"object","properties":{)"
      R"("query":{"type":"string"},"top_k":{"type":"integer"},"filters":{"type":"object"},)"
      R"("safe":{"type":"boolean"},"note":{"type":["string","null"]}}}}}])";
  const std::string block =
      "\n<invoke name=\"search\">\n<parameter name=\"query\">Zürich 天气 \"now\"\n</parameter>\n"
      "<parameter name=\"top_k\">3</parameter>\n<parameter name=\"filters\">{\"région\": [\"ZH\", "
      "1.5]}</parameter>\n"
      "<parameter name=\"safe\">true</parameter>\n<parameter name=\"note\">null</parameter>\n"
      "<parameter name=\"extra\">7</parameter>\n</invoke>\n<invoke name=\"ping\">\n</invoke>\n";
  const std::string want =
      "{\"query\": \"Zürich 天气 \\\"now\\\"\\n\", \"top_k\": 3, \"filters\": {\"région\": "
      "[\"ZH\", 1.5]}, "
      "\"safe\": true, \"note\": null, \"extra\": \"7\"}";
  // Streaming: reasoning, then content, then the block's calls together on
  // its closing marker, before the turn's end.
  {
    std::vector<int64_t> ids =
        w.tok.encode("Search it.\n</think>\n\nLooking.\n<minimax:tool_call>" + block +
                     "</minimax:tool_call>[e~[");
    size_t calls_at = static_cast<size_t>(-1);
    const Parsed got = parse(frontend, ids, /*in_reasoning=*/true, tools, &calls_at);
    require(got.reasoning == "Search it.\n" && got.reasoning_closed == 1 &&
                got.content == "\n\nLooking.\n",
            "reasoning '" + got.reasoning + "', content '" + got.content + "'");
    require(got.calls.size() == 2 && got.calls[0].name == "search" &&
                got.calls[0].arguments == want && got.calls[1].name == "ping" &&
                got.calls[1].arguments == "{}",
            "the block's calls, typed from the schema: " +
                (got.calls.empty() ? std::string("none") : got.calls[0].arguments));
    require(calls_at == ids.size() - 2 && ids[calls_at] == m.tool_call_close.id,
            "the calls arrive on the closing marker");
  }
  // The same block from every two-way cut of its text (the pieces a sampler
  // could have chosen instead): the same calls.
  {
    size_t cuts = 0;
    for (size_t j = 0; j <= block.size(); ++j) {
      if (j < block.size() && (static_cast<unsigned char>(block[j]) & 0xC0) == 0x80)
        continue;  // inside a character
      std::vector<int64_t> ids = {m.tool_call_open.id};
      for (const std::string& part : {block.substr(0, j), block.substr(j)})
        for (const int64_t id : w.tok.encode(part)) ids.push_back(id);
      ids.push_back(m.tool_call_close.id);
      ids.push_back(w.turn_end);
      const Parsed got = parse(frontend, ids, false, tools);
      require(got.calls.size() == 2 && got.calls[0].arguments == want &&
                  got.calls[1].name == "ping" && got.content.empty(),
              "cut " + std::to_string(j) + ": " +
                  (got.calls.empty() ? "no call, content '" + got.content + "'"
                                     : got.calls[0].arguments));
      ++cuts;
    }
    require(cuts > 250, "the cuts were walked: " + std::to_string(cuts));
  }
  // A call that opens inside the reasoning closes it.
  {
    const Parsed got =
        parse(frontend,
              w.tok.encode("Search now.\n<minimax:tool_call>\n<invoke name=\"ping\">\n</invoke>\n"
                           "</minimax:tool_call>[e~["),
              true, tools);
    require(got.reasoning == "Search now.\n" && got.reasoning_closed == 1 &&
                got.calls.size() == 1 && got.content.empty(),
            "the reasoning ends at the call");
  }
  // Malformed blocks come back as the text the model wrote, markers and all.
  for (const char* text :
       {"<minimax:tool_call>\n<invoke name=\"search\">\n<parameter name=\"query\">Par",
        "Sure.<minimax:tool_call>\n</minimax:tool_call>",
        "<minimax:tool_call>\n<invoke name=\"search\">\n<parameter "
        "name=\"query\">a</parameter>\n</minimax:tool_call>",
        "<minimax:tool_call>\n<function=search>\n</function>\n</minimax:tool_call> trailing",
        "<minimax:tool_call>\n{\"name\": \"search\", \"arguments\": {}}\n</minimax:tool_call>"}) {
    const Parsed got = parse(frontend, w.tok.encode(std::string(text) + "[e~["), false, tools);
    require(got.calls.empty() && got.content == text,
            std::string("literal content for: ") + text + " — got '" + got.content + "'");
  }
}

}  // namespace

int main(int argc, char** argv) {
  g_argc = argc;
  g_argv = argv;
  return ::dgpp::test::run_all();
}
