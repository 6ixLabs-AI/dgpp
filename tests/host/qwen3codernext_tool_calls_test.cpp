// Qwen3-Coder-Next's tool-call form over the real tokenizer and template
// (host; skips with rc 2 when the checkpoint is not in the model cache).
// The model class and the tokenizer are Qwen3-Next-80B's, whose template
// writes one JSON object between the <tool_call> tokens; this checkpoint's
// writes the XML tags of the Qwen3.8 template (vLLM's qwen3_coder parser).
// What only the checkpoint can say:
//   * its template states the XML form (text::chat_template_writes_json_calls
//     reads it off the source), and the frontend and the grammar vocabulary
//     built from that statement read as the XML format;
//   * the template's own spelling of a call, in the tokenizer's own pieces,
//     parses back to the call and is admitted by the forced-call grammar id
//     by id — in the auto, required and named modes;
//   * the 80B's JSON spelling is refused under this grammar.
// There is no jinja2 corpus for this template yet (none could be generated
// where this was written): the two plain prompts are compared with their
// hand-derived renders, the tool header by its structure.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common/log.hpp"
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

const char* kModel = "RedHatAI/Qwen3-Coder-Next-NVFP4";

// The checkpoint and what the service derives from it.
struct World {
  std::filesystem::path snap;
  dgpp::text::Tokenizer tok;
  dgpp::text::ChatTemplate tpl;
  std::string tpl_source;
  std::vector<int64_t> eos;  // generation_config.json's list
  int64_t vocab = 0;
  int64_t im_end = -1;

  explicit World(const std::string& dir)
      : snap(dir),
        tok(dgpp::text::Tokenizer::load((snap / "tokenizer.json").string())),
        tpl(dgpp::text::ChatTemplate::load((snap / "chat_template.jinja").string())),
        tpl_source(read_text_file((snap / "chat_template.jinja").string())) {
    const std::string generation = read_text_file((snap / "generation_config.json").string());
    const dgpp::minijson::ParseResult g = dgpp::minijson::parse(generation);
    const dgpp::minijson::Value& e = g.root.at("eos_token_id");
    if (e.is_array())
      for (const auto& v : e.items()) eos.push_back(v.as_int());
    else
      eos.push_back(e.as_int());
    const std::string config = read_text_file((snap / "config.json").string());
    vocab = dgpp::minijson::parse(config).root.at("vocab_size").as_int();
    const std::vector<int64_t> end = tok.encode("<|im_end|>");
    require(end.size() == 1, "<|im_end|> is one added token");
    im_end = end[0];
  }
};

const World& world() {
  static const World w = [] {
    std::string model = kModel;
    if (const char* env = std::getenv("DGPP_TP_REAL_MODEL"); env && *env) model = env;
    std::string err;
    const std::string snap = dgpp::hf::model_dir(model, &err);
    if (snap.empty()) {
      DGPP_LOG_WARN("qwen3codernext_tool_calls_test: model {} unavailable ({}); skipping", model, err);
      std::exit(2);
    }
    return World(snap);
  }();
  return w;
}

std::string render(const World& w, const std::string& messages_json, const std::string& tools_json,
                   bool generation_prompt) {
  const dgpp::minijson::ParseResult messages = dgpp::minijson::parse(messages_json);
  dgpp::text::Value::Members g;
  g.emplace_back("messages", dgpp::text::Value::from_minijson(messages.root));
  dgpp::minijson::ParseResult tools;
  if (!tools_json.empty()) {
    tools = dgpp::minijson::parse(tools_json);
    g.emplace_back("tools", dgpp::text::Value::from_minijson(tools.root));
  }
  g.emplace_back("add_generation_prompt", dgpp::text::Value::boolean(generation_prompt));
  return w.tpl.render(dgpp::text::Value::map_value(std::move(g)));
}

const char* kTools = R"([
  {"type": "function", "function": {"name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
      "city": {"type": "string", "description": "City name"},
      "days": {"type": "integer", "minimum": 0}}, "required": ["city"]}}},
  {"type": "function", "function": {"name": "run",
    "description": "Run a shell command.",
    "parameters": {"type": "object", "properties": {
      "command": {"type": "string"}, "timeout": {"type": "number"}}, "required": ["command"]}}},
  {"type": "function", "function": {"name": "search",
    "description": "Search the index.",
    "parameters": {"type": "object", "properties": {
      "query": {"type": "string"}, "limit": {"type": "integer"},
      "filters": {"type": "object"}}, "required": ["query"]}}}])";

// An assistant turn that carries tool calls: its request (the user turn and
// the assistant message), what it says, and the ids the model would have
// produced for it (the template's rendering of the turn without the header
// the prompt supplies, ending in <|im_end|>).
struct Turn {
  std::string where;
  std::string assistant_json;  // the assistant message
  std::string content;
  std::vector<ToolCallParser::Call> calls;
  std::vector<int64_t> ids;
};

std::vector<Turn> turns() {
  const World& w = world();
  const char* cases[][2] = {
      {"one call, no content",
       R"({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function":
          {"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}}]})"},
      {"content, then a multi-line string argument",
       R"({"role": "assistant", "content": "Let me look.", "tool_calls": [{"type": "function", "function":
          {"name": "run", "arguments": {"command": "ls -la\necho 'done'  # two lines", "timeout": 2.5}}}]})"},
      {"two calls, non-ASCII text and an object argument",
       R"({"role": "assistant", "content": "", "tool_calls": [
          {"type": "function", "function": {"name": "search", "arguments":
            {"query": "naïve café 北京", "limit": 5, "filters": {"lang": "fr", "exact": true}}}},
          {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "São Paulo"}}}]})"},
  };
  const std::string user = R"({"role": "user", "content": "Do it."})";
  std::vector<Turn> out;
  for (const auto& c : cases) {
    Turn t;
    t.where = c[0];
    t.assistant_json = c[1];
    const std::string before = render(w, "[" + user + "]", kTools, false);
    const std::string through = render(w, "[" + user + ", " + t.assistant_json + "]", kTools, false);
    require(through.compare(0, before.size(), before) == 0, t.where + ": the render grows by the turn");
    std::string text = through.substr(before.size());
    const std::string kAssistant = "<|im_start|>assistant\n", kEnd = "<|im_end|>\n";
    require(text.compare(0, kAssistant.size(), kAssistant) == 0 &&
                text.size() >= kAssistant.size() + kEnd.size() &&
                text.compare(text.size() - kEnd.size(), kEnd.size(), kEnd) == 0,
            t.where + ": the turn sits between the assistant header and <|im_end|>: " + text);
    text = text.substr(kAssistant.size(), text.size() - kAssistant.size() - kEnd.size());
    t.ids = w.tok.encode(text);
    t.ids.push_back(w.im_end);
    const dgpp::minijson::ParseResult msg = dgpp::minijson::parse(t.assistant_json);
    t.content = std::string(msg.root.at("content").as_string());
    for (const dgpp::minijson::Value& tc : msg.root.at("tool_calls").items()) {
      const dgpp::minijson::Value& fn = tc.at("function");
      ToolCallParser::Call call;
      call.name = std::string(fn.at("name").as_string());
      call.arguments = dgpp::text::Value::from_minijson(fn.at("arguments")).to_json(false);
      t.calls.push_back(std::move(call));
    }
    out.push_back(std::move(t));
  }
  return out;
}

std::string trimmed(std::string s) {
  const char* ws = " \t\r\n";
  const size_t a = s.find_first_not_of(ws);
  if (a == std::string::npos) return "";
  return s.substr(a, s.find_last_not_of(ws) - a + 1);
}

}  // namespace

DGPP_TEST(qwen3codernext_template_states_the_xml_form) {
  const World& w = world();
  require(!dgpp::text::chat_template_writes_json_calls(w.tpl_source),
          "this checkpoint's template writes <function=...> tags, not a JSON object");
  // The tokenizer is the 80B's: the two <tool_call> tokens and nothing else.
  const dgpp::text::ChatMarkers inferred = dgpp::text::ChatMarkers::from_tokenizer(w.tok);
  require(inferred.tool_call_open.available() && inferred.tool_call_close.available() &&
              !inferred.arg_key_open.available() && !inferred.dsml.available() && !inferred.xml_compact,
          "the two markers, nothing else");
  // What the server builds from the template's statement (apps/dgpp_serve.cpp
  // family_json_calls): the XML format, on a checkpoint with no reasoning.
  const bool json_calls = dgpp::text::chat_template_writes_json_calls(w.tpl_source);
  const dgpp::serve::TextFrontend frontend(&w.tok, &w.tpl, json_calls, /*instruct_only=*/true);
  require(frontend.markers().tool_format() == ToolFormat::kQwenXml, "the frontend reads the XML format");
  const GrammarVocab vocab = GrammarVocab::from_tokenizer(w.tok, w.eos, static_cast<int>(w.vocab),
                                                          dgpp::text::DsmlDialect::kV41, json_calls);
  require(vocab.usable() && vocab.markers().tool_format() == ToolFormat::kQwenXml,
          "the grammar vocabulary reads the same");
  require(vocab.call_turn_eos() == w.im_end && vocab.is_eos(w.im_end), "the call turn ends at <|im_end|>");
  // The template has no reasoning switch and its prompt opens no block.
  require(!w.tpl.reads("enable_thinking") && !w.tpl.reads("reasoning_effort"), "no reasoning switch");
}

DGPP_TEST(qwen3codernext_template_renders_the_plain_prompts) {
  const World& w = world();
  require(render(w, R"([{"role": "user", "content": "The capital of France is"}])", "", true) ==
              "<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n",
          "a user turn and the generation prompt");
  const std::string sys =
      render(w, R"([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "写一首诗。"}])", "",
             true);
  require(sys == "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\n写一首诗。<|im_end|>\n"
                 "<|im_start|>assistant\n",
          "a system turn: " + sys);
  // With tools and no system message the template supplies its own system
  // text, then the <tools> block and the call format, then the turns.
  const std::string tools = render(w, R"([{"role": "user", "content": "Weather in Paris?"}])", kTools, true);
  const std::string head = "<|im_start|>system\nYou are Qwen, a helpful AI assistant that can interact with a "
                           "computer to solve tasks.\n\n# Tools\n\nYou have access to the following functions:\n\n"
                           "<tools>\n<function>\n<name>get_weather</name>\n<description>Get the current weather "
                           "for a city.</description>\n<parameters>\n<parameter>\n<name>city</name>\n<type>string"
                           "</type>\n<description>City name</description>\n</parameter>\n<parameter>\n<name>days"
                           "</name>\n<type>integer</type>\n<minimum>0</minimum>\n</parameter>\n<required>[\"city\"]"
                           "</required>\n</parameters>\n</function>\n<function>\n<name>run</name>";
  require(tools.compare(0, head.size(), head) == 0, "the tools header: " + tools.substr(0, head.size() + 40));
  require(tools.find("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO "
                     "suffix:\n\n<tool_call>\n<function=example_function_name>\n") != std::string::npos,
          "the call format follows the tools");
  const std::string tail = "</IMPORTANT><|im_end|>\n<|im_start|>user\nWeather in Paris?<|im_end|>\n"
                           "<|im_start|>assistant\n";
  require(tools.size() > tail.size() && tools.compare(tools.size() - tail.size(), tail.size(), tail) == 0,
          "the turns follow the header");
  const dgpp::text::ChatMarkers markers = dgpp::text::ChatMarkers::from_tokenizer(w.tok);
  const std::vector<int64_t> ids = w.tok.encode(tools);
  require(!markers.prompt_opens_thinking(ids) && !markers.prompt_leaves_thinking_to_model(ids),
          "the prompt neither opens a reasoning block nor leaves one to the model");
  // A tool result returns inside a user turn.
  const std::string result = render(
      w,
      R"([{"role": "user", "content": "Go."}, {"role": "assistant", "content": "", "tool_calls": [{"type":
          "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
          {"role": "tool", "content": "18C"}])",
      kTools, true);
  const std::string result_tail =
      "<|im_start|>assistant\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
      "</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n18C\n</tool_response><|im_end|>\n"
      "<|im_start|>assistant\n";
  require(result.size() > result_tail.size() &&
              result.compare(result.size() - result_tail.size(), result_tail.size(), result_tail) == 0,
          "a call and its result: " + result.substr(result.size() > 260 ? result.size() - 260 : 0));
}

DGPP_TEST(qwen3codernext_tool_call_render_encode_parse_roundTrip) {
  const World& w = world();
  const dgpp::serve::TextFrontend frontend(&w.tok, &w.tpl, /*json_calls=*/false, /*instruct_only=*/true);
  const dgpp::minijson::ParseResult tools = dgpp::minijson::parse(kTools);
  size_t calls = 0;
  for (const Turn& turn : turns()) {
    ToolCallParser::Options opts;
    opts.start_in_reasoning = false;  // the prompt ends in "assistant\n"
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
    // The template puts newlines around the content and between the blocks.
    require(trimmed(content) == turn.content, turn.where + ": the content around the blocks: '" + content + "'");
    require(got.size() == turn.calls.size(),
            turn.where + ": every block is a call (" + std::to_string(got.size()) + ")");
    for (size_t c = 0; c < got.size(); ++c) {
      require(got[c].name == turn.calls[c].name, turn.where + ": the name " + got[c].name);
      require(got[c].arguments == turn.calls[c].arguments, turn.where + ": the arguments " + got[c].arguments);
    }
    calls += got.size();
    // The same ids under the 80B's JSON reading: no call.
    dgpp::text::ChatMarkers json = dgpp::text::ChatMarkers::from_tokenizer(w.tok);
    json.json_calls = true;
    ToolCallParser as_json(json, [&](const std::vector<int64_t>& v) { return frontend.decode_ids(v); },
                           dgpp::text::ToolSchemas(tools.root), opts);
    std::vector<ToolCallParser::Event> json_events;
    for (const int64_t id : turn.ids) as_json.feed(id, &json_events);
    as_json.finish(&json_events);
    require(as_json.calls() == 0, turn.where + ": the JSON reading finds no call in an XML block");
  }
  DGPP_LOG_INFO("qwen3codernext_tool_calls_test: {} calls round-tripped through the real tokenizer", calls);
}

DGPP_TEST(qwen3codernext_tool_grammar_accepts_the_templates_calls_over_the_real_tokenizer) {
  const World& w = world();
  const GrammarVocab vocab = GrammarVocab::from_tokenizer(w.tok, w.eos, static_cast<int>(w.vocab));
  const int64_t open = vocab.markers().tool_call_open.id, close = vocab.markers().tool_call_close.id;
  const dgpp::minijson::ParseResult tools = dgpp::minijson::parse(kTools);
  size_t positions = 0;
  // One turn under one spec: every id inside the mask, the mask and the
  // pointwise rule agreeing.
  const auto walk = [&](const Turn& turn, const GrammarSpec& spec, const std::vector<int64_t>& ids,
                        const std::string& mode) {
    GrammarState g(&vocab, spec, /*prompt_opens_thinking=*/false);
    for (size_t j = 0; j < ids.size(); ++j) {
      TokenMask m;
      g.mask(&m);
      ++positions;
      const std::string at = turn.where + " (" + mode + ") id " + std::to_string(ids[j]) + " '" +
                             vocab.text(ids[j]) + "' in state " + g.state_name();
      require(m.allows(ids[j]) == g.allows(ids[j]), "mask and allows() disagree at " + at);
      require(m.allows(ids[j]), "the template's own tokenization is refused at " + at);
      g.advance(ids[j]);
    }
    require(g.active(), turn.where + " (" + mode + "): the grammar is alive at the end");
  };
  for (const Turn& turn : turns()) {
    GrammarSpec spec;
    for (const dgpp::minijson::Value& t : tools.root.items())
      spec.tools.push_back(
          dgpp::text::grammar_tool_from_function(t.at("function"), nullptr, nullptr, ToolFormat::kQwenXml));
    // auto: the whole turn, prose included.
    spec.mode = GrammarSpec::Mode::kAuto;
    walk(turn, spec, turn.ids, "auto");
    // required: a call is owed, so the turn is its blocks.
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
    // The 80B's JSON spelling of the same call is outside this grammar.
    GrammarState x(&vocab, spec, false);
    x.advance(open);
    bool refused = false;
    for (const int64_t id : w.tok.encode("\n{\"name\": \"" + turn.calls[0].name + "\", \"arguments\": {}}\n")) {
      if (!x.allows(id)) {
        refused = true;
        break;
      }
      x.advance(id);
    }
    require(refused, turn.where + ": the JSON form is refused under the XML grammar");
  }
  DGPP_LOG_INFO("qwen3codernext_tool_calls_test: the grammar admitted {} positions of the template's own calls",
                positions);
}

int main() {
  return ::dgpp::test::run_all();
}
