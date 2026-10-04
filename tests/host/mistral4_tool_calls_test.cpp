// Mistral-Small-4 over its real tokenizer and template (host; skips with rc 2
// when the checkpoint is not in the model cache). The unit gates pin the
// parser and the grammar on synthetic ids; this one pins what only the
// checkpoint can say:
//   * the template renders as jinja2 does, raise_exception messages
//     included (the corpus, tests/data/mistral4_chat_template_goldens.jsonl,
//     from tools/gen_chat_template_goldens.py; its ids are compared where a
//     case carries them);
//   * the markers are the tokenizer's SPECIAL tokens — [TOOL_CALLS] 9,
//     [ARGS] 32, [THINK] 34, [/THINK] 35 for the pinned revision — which the
//     service's decode skips, so the format is read off the tokenizer and a
//     malformed call's literal text restores them;
//   * the template's own spelling of an assistant turn, in the tokenizer's
//     own pieces, parses back to its reasoning, content and calls, however
//     the pieces are cut, and is admitted by the grammar id by id.
#include <chrono>
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
using dgpp::text::TokenMask;
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
  return g_argc > 1 ? g_argv[1] : "tests/data/mistral4_chat_template_goldens.jsonl";
}

// The checkpoint, its corpus and what the service derives from them. The
// release carries no config.json / generation_config.json (params.json is
// the lead's): the ids below come off the tokenizer.
struct World {
  std::vector<std::string> lines;  // the corpus: header, then one case a line
  std::filesystem::path snap;
  dgpp::text::Tokenizer tok;
  dgpp::text::ChatTemplate tpl;
  int64_t bos = -1, eos = -1;

  World(std::vector<std::string> corpus, const std::string& dir)
      : lines(std::move(corpus)),
        snap(dir),
        tok(dgpp::text::Tokenizer::load((snap / "tokenizer.json").string())),
        tpl(dgpp::text::ChatTemplate::load((snap / "chat_template.jinja").string())) {
    for (const auto& added : tok.added_tokens()) {
      if (added.content == "<s>") bos = added.id;
      if (added.content == "</s>") eos = added.id;
    }
    require(bos >= 0 && eos >= 0, "<s> and </s> are added tokens");
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
    DGPP_LOG_WARN("mistral4_tool_calls_test: model {} unavailable ({}); skipping", model, err);
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

struct Parsed {
  std::string reasoning, content;
  int reasoning_closed = 0;
  std::vector<ToolCallParser::Call> calls;
  std::vector<Kind> order;
};

// The ids as the service parses them: the frontend's markers and decode
// (special tokens skipped); the prompt ended in [/INST] or [/TOOL_RESULTS],
// so the reply starts as content and [THINK] is the model's to open.
Parsed parse(const dgpp::serve::TextFrontend& frontend, const std::vector<int64_t>& ids,
             size_t* call_at = nullptr) {
  ToolCallParser::Options opts;
  opts.start_in_reasoning = false;
  opts.model_may_open_thinking = true;
  ToolCallParser parser(
      frontend.markers(), [&](const std::vector<int64_t>& v) { return frontend.decode_ids(v); },
      dgpp::text::ToolSchemas(), opts);
  std::vector<ToolCallParser::Event> events;
  for (size_t i = 0; i < ids.size(); ++i) {
    const size_t before = events.size();
    parser.feed(ids[i], &events);
    if (call_at != nullptr)
      for (size_t k = before; k < events.size(); ++k)
        if (events[k].kind == Kind::kToolCall && *call_at == static_cast<size_t>(-1)) *call_at = i;
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
// it — the template's rendering of the turn, which follows [/INST] or
// [/TOOL_RESULTS] with no header and ends in </s> — and what it says.
struct Turn {
  std::string where;
  std::string text;  // the turn's text, </s> included
  std::vector<int64_t> ids;
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
      // The template merges consecutive assistant messages into one turn:
      // the single ones are the turns a model writes.
      if (k > 0 && messages.items()[k - 1].at("role").as_string() == "assistant") continue;
      if (k + 1 < messages.items().size() &&
          messages.items()[k + 1].at("role").as_string() == "assistant")
        continue;
      const auto render_through = [&](size_t count) {
        std::vector<dgpp::text::Value> ms;
        for (size_t j = 0; j < count; ++j)
          ms.push_back(dgpp::text::Value::from_minijson(messages.items()[j]));
        dgpp::text::Value::Members g;
        for (const dgpp::minijson::Member& m : kwargs.members())
          if (m.key != "messages") g.emplace_back(m.key, dgpp::text::Value::from_minijson(m.value));
        g.emplace_back("messages", dgpp::text::Value::list_value(std::move(ms)));
        return w.tpl.render(dgpp::text::Value::map_value(std::move(g)));
      };
      Turn turn;
      turn.where =
          std::string(parsed.root.at("name").as_string()) + " message " + std::to_string(k);
      const std::string before = render_through(k);
      const std::string through = render_through(k + 1);
      require(through.compare(0, before.size(), before) == 0,
              turn.where + ": the render grows by the turn");
      turn.text = through.substr(before.size());
      const std::string kEnd = "</s>";
      require(turn.text.size() > kEnd.size() &&
                  turn.text.compare(turn.text.size() - kEnd.size(), kEnd.size(), kEnd) == 0,
              turn.where + ": the turn ends in </s>: " + turn.text);
      turn.ids = w.tok.encode(turn.text);
      require(!turn.ids.empty() && turn.ids.back() == w.eos,
              turn.where + ": </s> encodes to the EOS id");
      // What the turn says: the reasoning (a field or thinking chunks), the
      // text (a string or text chunks), the calls (arguments as JSON).
      if (const dgpp::minijson::Value* r = msg.find("reasoning_content");
          r != nullptr && r->is_string())
        turn.reasoning = std::string(r->as_string());
      else if (const dgpp::minijson::Value* r2 = msg.find("reasoning");
               r2 != nullptr && r2->is_string())
        turn.reasoning = std::string(r2->as_string());
      if (const dgpp::minijson::Value* c = msg.find("content"); c != nullptr) {
        if (c->is_string()) turn.content = std::string(c->as_string());
        if (c->is_array())
          for (const dgpp::minijson::Value& chunk : c->items()) {
            const std::string_view type = chunk.at("type").as_string();
            if (type == "text") turn.content += std::string(chunk.at("text").as_string());
            if (type == "thinking") turn.reasoning += std::string(chunk.at("thinking").as_string());
          }
      }
      if (const dgpp::minijson::Value* tcs = msg.find("tool_calls");
          tcs != nullptr && tcs->is_array())
        for (const dgpp::minijson::Value& tc : tcs->items()) {
          const dgpp::minijson::Value& fn = tc.at("function");
          ToolCallParser::Call call;
          call.name = std::string(fn.at("name").as_string());
          const dgpp::minijson::Value& args = fn.at("arguments");
          if (args.is_string()) {
            // The template writes a string verbatim ('' as {}); the parser
            // re-serializes the object it holds. (The parsed values view
            // the text: it has to outlive them.)
            const std::string text =
                args.as_string().empty() ? std::string("{}") : std::string(args.as_string());
            const dgpp::minijson::ParseResult object = dgpp::minijson::parse(text);
            call.arguments = dgpp::text::Value::from_minijson(object.root).to_json(false);
          } else {
            call.arguments = dgpp::text::Value::from_minijson(args).to_json(false);
          }
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

DGPP_TEST(mistral4_template_renders_as_the_goldens) {
  const World& w = world();
  const dgpp::minijson::ParseResult header_doc = dgpp::minijson::parse(w.lines[0]);
  const dgpp::minijson::Value& header = header_doc.root;
  require(std::string(header.at("template_hash").as_string()) == hex16(w.tpl.source_hash()),
          "the corpus is keyed to another chat_template.jinja — regenerate it "
          "(tools/gen_chat_template_goldens.py)");
  require(std::string(header.at("tokenizer_revision").as_string()) == hex16(w.tok.revision_hash()),
          "the corpus is keyed to another tokenizer.json");
  const dgpp::text::ChatMarkers markers = dgpp::text::ChatMarkers::from_tokenizer(w.tok);
  size_t renders = 0, errors = 0, with_ids = 0;
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
    require(!ids.empty() && ids[0] == w.bos,
            "case '" + name + "': the render opens with <s> (the template's, not encode's)");
    // No generation prompt: a render that awaits a reply ends in [/INST] or
    // [/TOOL_RESULTS], opens no reasoning block and leaves it to the model.
    const auto ends_with = [&](const char* tail) {
      const std::string t(tail);
      return got.size() >= t.size() && got.compare(got.size() - t.size(), t.size(), t) == 0;
    };
    if (ends_with("[/INST]") || ends_with("[/TOOL_RESULTS]"))
      require(!markers.prompt_opens_thinking(ids) && markers.prompt_leaves_thinking_to_model(ids),
              "case '" + name + "': the prompt leaves the reasoning to the model");
    if (const dgpp::minijson::Value* want_ids = parsed.root.find("ids")) {
      std::vector<int64_t> reference;
      for (const auto& v : want_ids->items()) reference.push_back(v.as_int(-1));
      require(ids == reference,
              "case '" + name + "': the render's ids differ from the reference tokenizer's (" +
                  std::to_string(ids.size()) + " vs " + std::to_string(reference.size()) + ")");
      ++with_ids;
    }
  }
  require(renders >= 30 && errors >= 12,
          "the corpus covers the renders and the template's refusals");
  DGPP_LOG_INFO(
      "mistral4_tool_calls_test: {} renders byte-exact ({} with reference ids), {} template "
      "refusals by message",
      renders, with_ids, errors);
}

DGPP_TEST(mistral4_markers_are_special_tokens_read_off_the_tokenizer) {
  const World& w = world();
  const dgpp::text::ChatMarkers m = dgpp::text::ChatMarkers::from_tokenizer(w.tok);
  require(m.tool_format() == ToolFormat::kMistral && m.tool_calls_available() &&
              m.reasoning_available() && m.bracket_think && !m.minimax_invoke && !m.xml_compact &&
              !m.tool_call_open.available() && !m.dsml.available(),
          "the tokenizer alone says Mistral: nothing is stated by the family");
  require(dgpp::serve::TextFrontend(&w.tok, &w.tpl).markers().tool_format() == ToolFormat::kMistral,
          "a frontend that states nothing reads the same");
  DGPP_LOG_INFO(
      "mistral4_tool_calls_test: markers [TOOL_CALLS] {} [ARGS] {} [THINK] {} [/THINK] {} <s> {} "
      "</s> {} newline {}",
      m.tool_calls.id, m.args.id, m.think_open.id, m.think_close.id, w.bos, w.eos, m.newline.id);
  // The pinned revision's ids (tokenizer.json f4c0a904bc06b16e): a parser
  // keyed on ids must know them, and they are all special.
  require(m.tool_calls.id == 9 && m.args.id == 32 && m.think_open.id == 34 &&
              m.think_close.id == 35 && w.bos == 1 && w.eos == 2,
          "the marker ids of the pinned tokenizer");
  for (const auto& added : w.tok.added_tokens())
    for (const char* marker : {"[TOOL_CALLS]", "[ARGS]", "[THINK]", "[/THINK]", "[INST]", "[/INST]",
                               "</s>", "<s>", "[SYSTEM_PROMPT]", "[TOOL_RESULTS]",
                               "[AVAILABLE_TOOLS]", "[MODEL_SETTINGS]", "[CALL_ID]"})
      if (added.content == marker)
        require(added.special, std::string(marker) + " is a special token");
  const std::vector<int64_t> call =
      w.tok.encode("[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\"}</s>");
  require(call.size() > 4 && call[0] == m.tool_calls.id && call.back() == w.eos,
          "the brackets encode to their ids");
  require(w.tok.decode(call, true) == "get_weather{\"city\": \"Paris\"}",
          "the service's decode skips the brackets: " + w.tok.decode(call, true));
  require(w.tok.decode(call, false) == "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\"}</s>",
          "the raw decode prints them");
  // The turn openers are the prefix cache's boundaries; "\n" is one id.
  std::vector<int64_t> roles;
  for (const dgpp::text::ChatMarker& r : m.role_markers) roles.push_back(r.id);
  require(roles == (std::vector<int64_t>{17, 3, 7}),
          "[SYSTEM_PROMPT] 17, [INST] 3, [TOOL_RESULTS] 7 are the boundaries");
  require(m.newline.available(), "the bare newline is one id");
  // The knobs the template reads: the effort (none / high) and the two
  // special-token globals the serving frontend has to pass.
  require(w.tpl.reads("reasoning_effort") && w.tpl.reads("bos_token") && w.tpl.reads("eos_token") &&
              w.tpl.reads("tools") && !w.tpl.reads("enable_thinking") &&
              !w.tpl.reads("add_generation_prompt"),
          "the template's globals");
}

DGPP_TEST(mistral4_tool_call_render_encode_parse_roundTrip) {
  const World& w = world();
  const dgpp::serve::TextFrontend frontend(&w.tok, &w.tpl);
  size_t turns = 0, calls = 0, reasoned = 0;
  for (const Turn& turn : golden_turns()) {
    const Parsed got = parse(frontend, turn.ids);
    require(got.reasoning == turn.reasoning,
            turn.where + ": reasoning '" + got.reasoning + "' != '" + turn.reasoning + "'");
    require(got.content == turn.content,
            turn.where + ": content '" + got.content + "' != '" + turn.content + "'");
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
    reasoned += got.reasoning.empty() ? 0 : 1;
    ++turns;
  }
  require(turns >= 22 && calls >= 12 && reasoned >= 6,
          "the corpus' turns: " + std::to_string(turns) + " turns, " + std::to_string(calls) +
              " calls, " + std::to_string(reasoned) + " with reasoning");
  DGPP_LOG_INFO(
      "mistral4_tool_calls_test: {} turns ({} calls, {} with reasoning) round-trip render -> "
      "encode -> parse",
      turns, calls, reasoned);
}

DGPP_TEST(mistral4_calls_stream_split_and_fall_back_over_the_real_tokenizer) {
  const World& w = world();
  const dgpp::serve::TextFrontend frontend(&w.tok, &w.tpl);
  const dgpp::text::ChatMarkers& m = frontend.markers();
  const auto encoded = [&](const std::string& text) { return w.tok.encode(text); };
  // Streaming: the content arrives before the call; the call goes out with
  // the id that closes its arguments, before the EOS.
  {
    const std::string args =
        "{\"city\": \"São Paulo\", \"days\": 3, \"opts\": {\"a\": [1, 2, {\"b\": \"}\"}]}}";
    const std::vector<int64_t> ids =
        encoded("Let me check.[TOOL_CALLS]get_weather[ARGS]" + args + "</s>");
    size_t call_at = static_cast<size_t>(-1);
    const Parsed got = parse(frontend, ids, &call_at);
    require(got.content == "Let me check." && got.calls.size() == 1 &&
                got.calls[0].name == "get_weather" && got.calls[0].arguments == args,
            "the streamed turn: " + (got.calls.empty() ? got.content : got.calls[0].arguments));
    require(call_at == ids.size() - 2, "the call is emitted at its closing brace, before </s>");
    require(got.order.back() == Kind::kToolCall, "the content deltas, then the call");
  }
  // The same call from every two-way cut of its name and of its arguments
  // (the pieces a sampler could have chosen instead): the same call.
  {
    const std::string
        name = "get_weather",
        args = "{\"city\": \"Zürich 天气\", \"n\": [1.5, null, true], \"o\": {\"k\": \"v\"}}";
    const std::string want = args;
    size_t cuts = 0;
    const auto boundary = [](const std::string& s, size_t i) {
      return i == s.size() || (static_cast<unsigned char>(s[i]) & 0xC0) != 0x80;
    };
    for (size_t i = 0; i <= name.size(); ++i)
      for (size_t j = 0; j <= args.size(); ++j) {
        if (!boundary(args, j) || (i != name.size() && j != args.size() && (i + j) % 3 != 0))
          continue;
        std::vector<int64_t> ids = {m.tool_calls.id};
        for (const std::string& part : {name.substr(0, i), name.substr(i)})
          for (const int64_t id : encoded(part)) ids.push_back(id);
        ids.push_back(m.args.id);
        for (const std::string& part : {args.substr(0, j), args.substr(j)})
          for (const int64_t id : encoded(part)) ids.push_back(id);
        ids.push_back(w.eos);
        const Parsed got = parse(frontend, ids);
        require(got.calls.size() == 1 && got.calls[0].name == name &&
                    got.calls[0].arguments == want && got.content.empty(),
                "cut " + std::to_string(i) + "/" + std::to_string(j) + ": " +
                    (got.calls.empty() ? "no call, content '" + got.content + "'"
                                       : got.calls[0].arguments));
        ++cuts;
      }
    require(cuts > 60, "the cuts were walked: " + std::to_string(cuts));
  }
  // Reasoning the model opened, with and without its close before a call.
  {
    Parsed got = parse(
        frontend,
        encoded("[THINK]I should call the tool.[/THINK]Checking.[TOOL_CALLS]ping[ARGS]{}</s>"));
    require(got.reasoning == "I should call the tool." && got.reasoning_closed == 1 &&
                got.content == "Checking." && got.calls.size() == 1 &&
                got.calls[0].arguments == "{}",
            "reasoning, content, a call");
    got = parse(frontend, encoded("[THINK]I should call the tool.[TOOL_CALLS]ping[ARGS]{}</s>"));
    require(got.reasoning == "I should call the tool." && got.reasoning_closed == 1 &&
                got.calls.size() == 1,
            "a call closes the reasoning it opens in");
  }
  // Malformed calls come back as the text the model wrote, brackets and all
  // (the EOS is no text).
  for (const char* text :
       {"[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Par", "[TOOL_CALLS]get_weather",
        "Sure.[TOOL_CALLS]get_weather[ARGS][1, 2]", "[TOOL_CALLS][ARGS]{\"a\": 1}",
        "[TOOL_CALLS]f[ARGS]{\"a\": 1, \"a\": 2} trailing"}) {
    const Parsed got = parse(frontend, encoded(std::string(text) + "</s>"));
    require(got.calls.empty() && got.content == text,
            std::string("literal content for: ") + text + " — got '" + got.content + "'");
  }
  // Text after a call's object is content; a second call still parses.
  {
    const Parsed got = parse(
        frontend, encoded("[TOOL_CALLS]ping[ARGS]{} and then[TOOL_CALLS]ping[ARGS]{\"n\": 1}</s>"));
    require(got.calls.size() == 2 && got.content == " and then" &&
                got.calls[1].arguments == "{\"n\": 1}",
            "content between calls: '" + got.content + "'");
  }
}

DGPP_TEST(mistral4_tool_grammar_accepts_the_golden_turns_over_the_real_tokenizer) {
  const World& w = world();
  const auto t0 = std::chrono::steady_clock::now();
  const int vocab_size = static_cast<int>(w.tok.max_id() + 1);
  const GrammarVocab vocab = GrammarVocab::from_tokenizer(w.tok, {w.eos}, vocab_size);
  vocab.prepare_json();
  const double build_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  require(vocab.usable() && vocab.markers().tool_format() == ToolFormat::kMistral &&
              vocab.call_turn_eos() == w.eos,
          "the grammar vocabulary is Mistral's, its call turn ends at </s>");
  const int64_t opener = vocab.markers().tool_calls.id, args_id = vocab.markers().args.id;
  size_t positions = 0, walked = 0;
  double total_us = 0.0, max_us = 0.0;
  std::string max_where;
  // One turn under one spec: every id inside the mask, the mask and the
  // pointwise rule agreeing, the grammar alive at the end.
  const auto walk = [&](const Turn& turn, const GrammarSpec& spec, const std::vector<int64_t>& ids,
                        const std::string& mode) {
    GrammarState g(&vocab, spec, /*prompt_opens_thinking=*/false, /*model_may_open_thinking=*/true);
    for (size_t j = 0; j < ids.size(); ++j) {
      TokenMask mask;
      const auto a = std::chrono::steady_clock::now();
      g.mask(&mask);
      const double us =
          std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count();
      ++positions;
      total_us += us;
      if (us > max_us) {
        max_us = us;
        max_where = g.state_name();
      }
      const std::string at = turn.where + " (" + mode + ") id " + std::to_string(ids[j]) + " '" +
                             w.tok.decode(ids[j], false) + "' in state " + g.state_name();
      require(mask.allows(ids[j]) == g.allows(ids[j]), "mask and allows() disagree at " + at);
      require(mask.allows(ids[j]), "the template's own tokenization is refused at " + at);
      g.advance(ids[j]);
    }
    require(g.active(), turn.where + " (" + mode + "): the grammar is alive at the end");
    ++walked;
  };
  for (const Turn& turn : golden_turns()) {
    if (turn.calls.empty() || turn.tools.empty()) continue;
    // The template writes a client's argument STRING verbatim (compact JSON
    // is still one object); the grammar reads either spelling.
    const dgpp::minijson::ParseResult tools = dgpp::minijson::parse(turn.tools);
    GrammarSpec spec;
    bool declared = true;
    for (const ToolCallParser::Call& call : turn.calls) {
      bool found = false;
      for (const dgpp::minijson::Value& t : tools.root.items()) {
        const dgpp::minijson::Value* fn = t.find("function");
        found = found || (fn ? *fn : t).at("name").as_string() == call.name;
      }
      declared = declared && found;
    }
    if (!declared) continue;  // a call to a tool the case does not declare: outside any grammar
    for (const dgpp::minijson::Value& t : tools.root.items()) {
      const dgpp::minijson::Value* fn = t.find("function");
      spec.tools.push_back(dgpp::text::grammar_tool_from_function(fn ? *fn : t, nullptr, nullptr,
                                                                  ToolFormat::kMistral));
    }
    // auto: the whole turn — reasoning, prose, calls, the EOS.
    spec.mode = GrammarSpec::Mode::kAuto;
    walk(turn, spec, turn.ids, "auto");
    // required: a call is owed, so the turn is its reasoning (if any) and
    // its calls: the prose before the first call is the unconstrained turn's.
    std::vector<int64_t> forced;
    {
      size_t first_call = 0;
      while (first_call < turn.ids.size() && turn.ids[first_call] != opener) ++first_call;
      require(first_call < turn.ids.size(), turn.where + ": the turn opens a call");
      size_t think_end = 0;
      if (turn.ids[0] == vocab.markers().think_open.id)
        while (think_end < first_call && turn.ids[think_end] != vocab.markers().think_close.id)
          ++think_end;
      if (think_end > 0 && think_end < first_call)
        forced.assign(turn.ids.begin(),
                      turn.ids.begin() + static_cast<std::ptrdiff_t>(think_end) + 1);
      forced.insert(forced.end(), turn.ids.begin() + static_cast<std::ptrdiff_t>(first_call),
                    turn.ids.end());
    }
    spec.mode = GrammarSpec::Mode::kRequired;
    walk(turn, spec, forced, "required");
    // named, one call: the first call alone, then the turn's end; a second
    // opener is refused there.
    std::vector<int64_t> first;
    {
      size_t at = 0;
      while (forced[at] != opener) first.push_back(forced[at++]);
      first.push_back(forced[at++]);
      while (at < forced.size() && forced[at] != opener && forced[at] != w.eos)
        first.push_back(forced[at++]);
      first.push_back(w.eos);
    }
    spec.mode = GrammarSpec::Mode::kNamed;
    spec.named = turn.calls[0].name;
    spec.parallel = false;
    walk(turn, spec, first, "named");
    {
      GrammarState g(&vocab, spec, false, true);
      for (size_t j = 0; j + 1 < first.size(); ++j) g.advance(first[j]);
      require(g.active() && !g.allows(opener) && g.allows(w.eos),
              turn.where + ": after the named call only the turn's end");
    }
    // A name outside the tools is refused at its first id; EOS and [ARGS]
    // are refused while the call is owed.
    {
      spec.mode = GrammarSpec::Mode::kRequired;
      GrammarState g(&vocab, spec, false, false);
      require(!g.allows(w.eos) && !g.allows(args_id) && g.allows(opener),
              turn.where + ": the opener is owed");
      g.advance(opener);
      const std::vector<int64_t> zebra = w.tok.encode("zebra_tool");
      require(!zebra.empty() && !g.allows(zebra[0]) && !g.allows(args_id),
              turn.where + ": an undeclared name is refused");
    }
  }
  require(walked >= 15,
          "the corpus' call turns were walked under three modes: " + std::to_string(walked));
  // The whole vocabulary at every position of one strict call: the mask is
  // the pointwise rule, id for id, and the call ends the turn.
  size_t compared = 0, compared_positions = 0;
  {
    GrammarSpec spec;
    spec.mode = GrammarSpec::Mode::kRequired;
    spec.parallel = false;
    const dgpp::minijson::ParseResult def = dgpp::minijson::parse(
        R"({"name":"get_weather","strict":true,"parameters":{"type":"object","properties":{)"
        R"("city":{"type":"string"},"days":{"type":"integer"}},"required":["city"],"additionalProperties":false}})");
    spec.tools.push_back(
        dgpp::text::grammar_tool_from_function(def.root, nullptr, nullptr, ToolFormat::kMistral));
    std::vector<int64_t> ids =
        w.tok.encode("[TOOL_CALLS]get_weather[ARGS]{\"city\": \"São Paulo\", \"days\": 3}");
    ids.push_back(w.eos);
    GrammarState g(&vocab, spec, false, false);
    for (const int64_t id : ids) {
      TokenMask mask;
      g.mask(&mask);
      int counted = 0;
      for (int64_t v = 0; v < vocab_size; ++v) {
        const bool pointwise = g.allows(v);
        require(mask.allows(v) == pointwise, std::string("mask and allows() disagree on id ") +
                                                 std::to_string(v) + " '" + vocab.text(v) +
                                                 "' in state " + g.state_name());
        counted += pointwise ? 1 : 0;
        ++compared;
      }
      ++compared_positions;
      require(counted == mask.allowed,
              std::string("the allowed count is the set's size in state ") + g.state_name());
      require(mask.allows(id), std::string("the call's own id refused in state ") + g.state_name());
      g.advance(id);
    }
    require(std::string(g.state_name()) == "done",
            std::string("the strict call ended the turn: ") + g.state_name());
  }
  DGPP_LOG_INFO(
      "mistral4_tool_calls_test: grammar vocabulary and JSON tables built in {:.2f} s; {} walks, "
      "{} positions, mask "
      "{:.1f} us avg, {:.0f} us max (at {}); mask == allows() over {} ids at {} positions",
      build_s, walked, positions, total_us / static_cast<double>(positions), max_us, max_where,
      compared, compared_positions);
}

}  // namespace

int main(int argc, char** argv) {
  g_argc = argc;
  g_argv = argv;
  return ::dgpp::test::run_all();
}
