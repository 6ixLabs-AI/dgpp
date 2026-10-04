// M6 6g: the tool-call grammar over a fake vocabulary (host-only, always
// runs). Pins: the masks of every state (which ids a position allows and
// the allowed count the sampler treats as the vocabulary), the name/key
// automaton over token texts (any tokenization of a name, nothing else,
// prefix-sharing names), the modes (required / named / auto-single /
// forbid, parallel or not), the EOS discipline while a call is owed, a
// disallowed id killing the grammar, and a complete valid turn being
// accepted position by position.
#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "text/tool_grammar.hpp"

namespace {

using dgpp::text::ChatMarker;
using dgpp::text::ChatMarkers;
using dgpp::text::GrammarArg;
using dgpp::text::GrammarSpec;
using dgpp::text::GrammarState;
using dgpp::text::GrammarTool;
using dgpp::text::GrammarVocab;
using dgpp::text::TokenMask;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// The fake vocabulary: ids 0..255 are single bytes, then a few multi-byte
// word tokens, the eight markers, and three EOS ids; vocab_size pads it.
constexpr int64_t kGet = 256, kWeather = 257, kGetWeather = 258,
                  kUnderscore = 259, kWea = 260, kTher = 261, kCity = 262,
                  kGetT = 263, kIme = 264;
constexpr int64_t kThinkOpen = 300, kThinkClose = 301, kToolOpen = 302,
                  kToolClose = 303, kKeyOpen = 304, kKeyClose = 305,
                  kValueOpen = 306, kValueClose = 307;
constexpr int64_t kEosText = 310, kEosUser = 311, kEosObs = 312;
constexpr int kVocab = 320;

GrammarVocab fake_vocab() {
  std::vector<std::string> texts(static_cast<size_t>(kVocab));
  for (int b = 0; b < 256; ++b) texts[static_cast<size_t>(b)] = std::string(1, static_cast<char>(b));
  texts[kGet] = "get";
  texts[kWeather] = "_weather";
  texts[kGetWeather] = "get_weather";
  texts[kUnderscore] = "_";
  texts[kWea] = "wea";
  texts[kTher] = "ther";
  texts[kCity] = "city";
  texts[kGetT] = "get_t";
  texts[kIme] = "ime";
  ChatMarkers m;
  m.think_open = ChatMarker{kThinkOpen, "<think>"};
  m.think_close = ChatMarker{kThinkClose, "</think>"};
  m.tool_call_open = ChatMarker{kToolOpen, "<tool_call>"};
  m.tool_call_close = ChatMarker{kToolClose, "</tool_call>"};
  m.arg_key_open = ChatMarker{kKeyOpen, "<arg_key>"};
  m.arg_key_close = ChatMarker{kKeyClose, "</arg_key>"};
  m.arg_value_open = ChatMarker{kValueOpen, "<arg_value>"};
  m.arg_value_close = ChatMarker{kValueClose, "</arg_value>"};
  return GrammarVocab(std::move(texts), m, {kEosText, kEosUser, kEosObs}, kVocab,
                      kEosObs);
}

GrammarSpec spec_of(GrammarSpec::Mode mode, bool parallel = true,
                    const std::string& named = "") {
  GrammarSpec s;
  s.mode = mode;
  s.parallel = parallel;
  s.named = named;
  GrammarTool weather;
  weather.name = "get_weather";
  weather.constrain_keys = true;
  weather.keys = {"city", "days"};
  GrammarTool time;
  time.name = "get_time";  // shares the "get_" prefix
  time.constrain_keys = false;
  GrammarTool bare;
  bare.name = "ping";
  bare.constrain_keys = true;  // closed and empty: no arguments at all
  s.tools = {weather, time, bare};
  return s;
}

std::vector<int64_t> allowed_ids(const GrammarState& g) {
  TokenMask m;
  g.mask(&m);
  std::vector<int64_t> out;
  if (!m.constrained()) return out;
  for (int64_t id = 0; id < kVocab; ++id)
    if (m.allows(id)) out.push_back(id);
  require(static_cast<int>(out.size()) == m.allowed,
          "allowed count equals the set bits: " + std::to_string(out.size()) +
              " vs " + std::to_string(m.allowed));
  return out;
}

bool same(std::vector<int64_t> a, std::vector<int64_t> b) {
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  return a == b;
}

std::string show(const std::vector<int64_t>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size() && i < 12; ++i)
    s += (i ? "," : "") + std::to_string(v[i]);
  if (v.size() > 12) s += ",…";
  return s + "]";
}

void feed(GrammarState& g, const std::vector<int64_t>& ids) {
  for (const int64_t id : ids) {
    require(g.allows(id), "id " + std::to_string(id) + " allowed in state " +
                              g.state_name());
    g.advance(id);
  }
}

// ---- the Qwen3.8 XML format --------------------------------------
GrammarVocab qwen_vocab(bool compact = false) {
  std::vector<std::string> texts(static_cast<size_t>(kVocab));
  for (int b = 0; b < 256; ++b) texts[static_cast<size_t>(b)] = std::string(1, static_cast<char>(b));
  texts[kGet] = "get";
  texts[kWeather] = "_weather";
  texts[kGetWeather] = "get_weather";
  texts[kUnderscore] = "_";
  texts[kWea] = "wea";
  texts[kTher] = "ther";
  texts[kCity] = "city";
  texts[kGetT] = "get_t";
  texts[kIme] = "ime";
  texts[kThinkOpen] = "<think>";
  texts[kThinkClose] = "</think>";
  texts[kToolOpen] = "<tool_call>";
  texts[kToolClose] = "</tool_call>";
  ChatMarkers m;
  m.think_open = ChatMarker{kThinkOpen, "<think>"};
  m.think_close = ChatMarker{kThinkClose, "</think>"};
  m.tool_call_open = ChatMarker{kToolOpen, "<tool_call>"};
  m.tool_call_close = ChatMarker{kToolClose, "</tool_call>"};
  texts[280] = "></";
  texts[281] =
      "<function=get_weather><parameter=city>Paris</parameter><parameter=days>3</parameter></"
      "function>";
  texts[282] = "></function>";
  texts[283] = "><parameter=unknown>";
  texts[284] = "3</parameter><parameter=city>Paris</parameter></function>";
  m.xml_compact = compact;
  return GrammarVocab(std::move(texts), m, {kEosText, kEosUser}, kVocab, kEosUser);
}
std::vector<int64_t> bytes_of(const std::string& s) {
  std::vector<int64_t> out;
  for (const char c : s) out.push_back(static_cast<unsigned char>(c));
  return out;
}

DGPP_TEST(tool_grammar_qwen_required_call_walks_the_xml_shape) {
  const GrammarVocab vocab = qwen_vocab();
  require(vocab.usable() && vocab.markers().tool_format() == dgpp::text::ToolFormat::kQwenXml,
          "the two-marker vocabulary is the Qwen format");
  GrammarState g(&vocab, spec_of(GrammarSpec::Mode::kRequired), /*prompt_opens_thinking=*/false);
  // A call is owed: only <tool_call> opens the turn. (The template's
  // "optional reasoning BEFORE the function call" lives in <think>; free
  // prose here only lets a greedy model ramble until max_tokens instead of
  // calling — TC-45.) The turn may not end while the call is owed.
  require(!g.allows('H') && g.allows(kToolOpen) && !g.allows(kEosUser) && !g.allows(kEosText) &&
              !g.allows(kToolClose),
          "top: <tool_call> only, never prose or EOS while owed");
  g.advance(kToolOpen);
  // "\n<function=": any tokenization of the literal, nothing else.
  require(g.allows('\n') && !g.allows('x') && !g.allows(kToolClose) && !g.allows(kEosUser),
          "q-name starts with the literal's newline");
  feed(g, bytes_of("\n<function="));
  // The name: get_weather / get_time / ping over token texts.
  require(g.allows(kGet) && g.allows(kGetWeather) && g.allows('p') && g.allows(kGetT) && !g.allows('x'),
          "names over token texts");
  feed(g, {kGet, kWeather});
  require(g.allows('>') && !g.allows('\n'), "the name closes with '>\\n'");
  feed(g, bytes_of(">\n"));
  require(std::string(g.state_name()) == "q-key-or-close", std::string("after the name: ") + g.state_name());
  // Closed keys city/days, or the close: everything starts with '<'.
  require(same(allowed_ids(g), {'<'}), "key-or-close: '<' only: " + show(allowed_ids(g)));
  feed(g, bytes_of("<parameter="));
  require(g.allows('c') && g.allows('d') && !g.allows('x'), "closed keys city/days");
  feed(g, bytes_of("city>\n"));
  require(std::string(g.state_name()) == "q-value", "a free value follows the key");
  // The free value: anything but the markers, the think markers and EOS.
  const std::vector<int64_t> v = allowed_ids(g);
  require(!v.empty() && std::find(v.begin(), v.end(), kToolClose) == v.end() &&
              std::find(v.begin(), v.end(), kEosUser) == v.end() &&
              std::find(v.begin(), v.end(), kThinkOpen) == v.end() &&
              std::find(v.begin(), v.end(), 'P') != v.end(),
          "free value: text only");
  feed(g, bytes_of("Paris\n</parameter>\n"));
  require(std::string(g.state_name()) == "q-key-or-close", "the terminator closes the value");
  // city is used: after "<parameter=" only days remains.
  feed(g, bytes_of("<parameter="));
  require(g.allows('d') && !g.allows('c'), "a closed key is offered once");
  feed(g, bytes_of("days>\n3\n</parameter>\n"));
  // Every key used: only the close remains.
  require(g.allows('<') && !g.allows('\n'), "only </function> remains");
  feed(g, bytes_of("</function>"));
  require(!g.allows('p'), "after </function> the newline");
  feed(g, bytes_of("\n"));
  require(same(allowed_ids(g), {kToolClose}), "q-close: </tool_call> only: " + show(allowed_ids(g)));
  g.advance(kToolClose);
  // Required (parallel): free text, another call, or the turn's end.
  require(g.allows(kToolOpen) && g.allows(kEosUser) && g.allows('x') && !g.allows(kToolClose),
          "after the call: text, another call or EOS");
  g.advance(kEosUser);
  require(std::string(g.state_name()) == "done", "done after EOS");
}

DGPP_TEST(tool_grammar_qwen_free_keys_typed_values_and_named_single) {
  const GrammarVocab vocab = qwen_vocab();
  // get_time: an open key set — free text through ">\n".
  GrammarState g(&vocab, spec_of(GrammarSpec::Mode::kNamed, false, "get_time"), false);
  g.advance(kToolOpen);
  feed(g, bytes_of("\n<function="));
  require(!g.allows(kGetWeather) && g.allows(kGetT), "named: only get_time");
  feed(g, {kGetT, kIme});
  feed(g, bytes_of(">\n<parameter="));
  require(std::string(g.state_name()) == "q-free-key", "an open key set is free text");
  require(g.allows('t') && !g.allows(kToolClose) && !g.allows(kEosUser), "free key text");
  feed(g, bytes_of("tz>\nUTC\n</parameter>\n</function>\n"));
  g.advance(kToolClose);
  // Named: exactly one call, then the turn ends.
  require(same(allowed_ids(g), {kEosUser}), "named single: EOS only: " + show(allowed_ids(g)));
  // Typed values: a JSON integer and an enum string.
  GrammarSpec spec = spec_of(GrammarSpec::Mode::kRequired);
  GrammarArg days;
  days.key = "days";
  days.kind = GrammarArg::Kind::kJson;
  days.schema = R"({"type": "integer"})";
  GrammarArg unit;
  unit.key = "unit";
  unit.kind = GrammarArg::Kind::kText;
  unit.texts = {"celsius", "fahrenheit"};
  spec.tools[0].keys = {"days", "unit"};
  spec.tools[0].args = {days, unit};
  GrammarState t(&vocab, spec, false);
  t.advance(kToolOpen);
  feed(t, bytes_of("\n<function=get_weather>\n<parameter=days>\n"));
  // A JSON text may open with whitespace; a letter never.
  require(t.allows('3') && !t.allows('x') && !t.allows('<'), "a JSON integer: digits first");
  feed(t, bytes_of("3"));
  require(t.allows('4') && t.allows('\n'), "more digits, or the terminator once complete");
  feed(t, bytes_of("\n"));
  require(t.allows('<') && !t.allows('4') && !t.allows('\n'), "inside the terminator only its bytes");
  feed(t, bytes_of("</parameter>\n<parameter=unit>\n"));
  require(t.allows('c') && t.allows('f') && !t.allows('k'), "an enum value: its texts");
  feed(t, bytes_of("celsius"));
  require(t.allows('\n') && !t.allows('c'), "the enum text then its terminator");
  feed(t, bytes_of("\n</parameter>\n</function>\n"));
  t.advance(kToolClose);
  require(t.active() && std::string(t.state_name()) == "top", "the call closed cleanly");
}

// The MiMo-V2.6 flows on the XML grammar (2026-09-22): the prompt leaves
// the thinking to the model (model_may_open_thinking) — a <think> as the
// first id opens the free reasoning and </think> returns to the mode's
// opening state; any other first id settles the grammar as before — and
// the Qwen2 tokenizer's straddling token that closes a JSON value and runs
// into the terminator's newline ('"\n').
DGPP_TEST(tool_grammar_xml_modelOpensThinkingAndStraddledTerminator) {
  std::vector<std::string> texts(static_cast<size_t>(kVocab));
  for (int b = 0; b < 256; ++b) texts[static_cast<size_t>(b)] = std::string(1, static_cast<char>(b));
  texts[kGetWeather] = "get_weather";
  texts[kCity] = "\"\n";  // the straddling token: a string's close quote + the terminator's newline
  texts[kThinkOpen] = "<think>";
  texts[kThinkClose] = "</think>";
  texts[kToolOpen] = "<tool_call>";
  texts[kToolClose] = "</tool_call>";
  ChatMarkers m;
  m.think_open = ChatMarker{kThinkOpen, "<think>"};
  m.think_close = ChatMarker{kThinkClose, "</think>"};
  m.tool_call_open = ChatMarker{kToolOpen, "<tool_call>"};
  m.tool_call_close = ChatMarker{kToolClose, "</tool_call>"};
  m.xml_compact = false;
  const GrammarVocab vocab(std::move(texts), m, {kEosText, kEosUser}, kVocab, kEosUser);
  // Required call, the choice left to the model: <think> allowed first.
  GrammarState g(&vocab, spec_of(GrammarSpec::Mode::kRequired), /*prompt_opens_thinking=*/false,
                 /*model_may_open_thinking=*/true);
  require(g.allows(kThinkOpen) && g.allows(kToolOpen) && !g.allows('H') && !g.allows(kEosUser),
          "first position: the opener or the call; never prose or EOS while owed");
  g.advance(kThinkOpen);
  require(std::string(g.state_name()) == "think", std::string("the opener enters the reasoning: ") + g.state_name());
  require(g.allows('x') && g.allows(kThinkClose) && !g.allows(kEosUser), "free reasoning, EOS refused while owed");
  feed(g, bytes_of("plan"));
  g.advance(kThinkClose);
  require(std::string(g.state_name()) == "top", std::string("</think> returns to the top: ") + g.state_name());
  g.advance(kToolOpen);
  feed(g, bytes_of("\n<function="));
  feed(g, {kGetWeather});
  feed(g, bytes_of(">\n<parameter=city>\n"));
  // JSON mode for the value: a typed string closing with the straddling token.
  // (city is a free value in spec_of; a typed one is built below.)
  feed(g, bytes_of("Paris\n</parameter>\n</function>\n"));
  g.advance(kToolClose);
  require(g.active(), "the call closed");
  // A JSON-typed value whose close quote and terminator newline share a token.
  GrammarSpec spec = spec_of(GrammarSpec::Mode::kRequired);
  GrammarArg city;
  city.key = "city";
  city.kind = GrammarArg::Kind::kJson;
  city.schema = R"({"type": "string"})";
  spec.tools[0].keys = {"city"};
  spec.tools[0].args = {city};
  GrammarState t(&vocab, spec, false, true);
  // Any other first id settles the opening (the top's rules apply).
  t.advance(kToolOpen);
  require(std::string(t.state_name()) == "q-name" && !t.allows(kThinkOpen), "a call first: inside the call no opener");
  feed(t, bytes_of("\n<function="));
  feed(t, {kGetWeather});
  feed(t, bytes_of(">\n<parameter=city>\n\"Oslo"));
  require(t.allows(kCity), "the straddling '\"\\n' token closes the value and opens the terminator");
  t.advance(kCity);
  require(t.allows('<') && !t.allows('\n') && !t.allows('x'), "inside the terminator: '</' next");
  feed(t, bytes_of("</parameter>\n</function>\n"));
  t.advance(kToolClose);
  require(t.active() && std::string(t.state_name()) == "top", "the typed call closed cleanly");
  // Without the choice (a Qwen prompt that opened the block, or one that
  // closed it): the opener is not offered at the first position, and prose
  // is refused — the owed call is forced, so the marker id kills the grammar.
  GrammarState settled(&vocab, spec_of(GrammarSpec::Mode::kRequired), false, false);
  require(!settled.allows(kThinkOpen) && settled.allows(kToolOpen) && !settled.allows('H'),
          "no choice: the call forced, the opener and prose refused");
  settled.advance(kThinkOpen);  // a disallowed id kills the grammar
  require(!settled.active() && std::string(settled.state_name()) == "dead",
          std::string("no choice: the marker kills the forced top: ") + settled.state_name());
  GrammarSpec json_settled;
  json_settled.mode = GrammarSpec::Mode::kJson;
  GrammarState js(&vocab, json_settled, false, false);
  require(!js.allows(kThinkOpen) && js.allows('{'), "no choice in JSON mode: the body only");
  GrammarState opened(&vocab, spec_of(GrammarSpec::Mode::kRequired), true, true);
  require(std::string(opened.state_name()) == "think", "a prompt that opened the block starts in the reasoning");
  // JSON mode: <think> first, then the JSON body after </think>.
  GrammarSpec json;
  json.mode = GrammarSpec::Mode::kJson;
  GrammarState j(&vocab, json, false, true);
  require(j.allows(kThinkOpen) && j.allows('{') && !j.allows('x'), "JSON mode: the opener or the body");
  j.advance(kThinkOpen);
  feed(j, bytes_of("why"));
  j.advance(kThinkClose);
  require(std::string(j.state_name()) == "json-body" || j.allows('{'), std::string("the body after the block: ") + j.state_name());
  require(j.allows('{') && !j.allows('x'), "the JSON body follows");
}

DGPP_TEST(tool_grammar_mimo_modelOpensThinkingAndStraddledTerminator) {
  std::vector<std::string> texts(static_cast<size_t>(kVocab));
  for (int b = 0; b < 256; ++b)
    texts[static_cast<size_t>(b)] = std::string(1, static_cast<char>(b));
  texts[kGetWeather] = "get_weather";
  texts[kCity] = "\"</";  // the straddling token: a string's close quote + the terminator's newline
  texts[kThinkOpen] = "<think>";
  texts[kThinkClose] = "</think>";
  texts[kToolOpen] = "<tool_call>";
  texts[kToolClose] = "</tool_call>";
  ChatMarkers m;
  m.think_open = ChatMarker{kThinkOpen, "<think>"};
  m.think_close = ChatMarker{kThinkClose, "</think>"};
  m.tool_call_open = ChatMarker{kToolOpen, "<tool_call>"};
  m.tool_call_close = ChatMarker{kToolClose, "</tool_call>"};
  m.xml_compact = true;
  const GrammarVocab vocab(std::move(texts), m, {kEosText, kEosUser}, kVocab, kEosUser);
  // Required call, the choice left to the model: <think> allowed first.
  GrammarState g(&vocab, spec_of(GrammarSpec::Mode::kRequired), /*prompt_opens_thinking=*/false,
                 /*model_may_open_thinking=*/true);
  require(g.allows(kThinkOpen) && g.allows(kToolOpen) && g.allows('H') && !g.allows(kEosUser),
          "first position: the opener, the call or text; never EOS while owed");
  g.advance(kThinkOpen);
  require(std::string(g.state_name()) == "think",
          std::string("the opener enters the reasoning: ") + g.state_name());
  require(g.allows('x') && g.allows(kThinkClose) && !g.allows(kEosUser),
          "free reasoning, EOS refused while owed");
  feed(g, bytes_of("plan"));
  g.advance(kThinkClose);
  require(std::string(g.state_name()) == "top",
          std::string("</think> returns to the top: ") + g.state_name());
  g.advance(kToolOpen);
  feed(g, bytes_of("<function="));
  feed(g, {kGetWeather});
  feed(g, bytes_of("><parameter=city>"));
  // JSON mode for the value: a typed string closing with the straddling token.
  // (city is a free value in spec_of; a typed one is built below.)
  feed(g, bytes_of("Paris</parameter></function>"));
  g.advance(kToolClose);
  require(g.active(), "the call closed");
  // A JSON-typed value whose close quote and terminator newline share a token.
  GrammarSpec spec = spec_of(GrammarSpec::Mode::kRequired);
  GrammarArg city;
  city.key = "city";
  city.kind = GrammarArg::Kind::kJson;
  city.schema = R"({"type": "string"})";
  spec.tools[0].keys = {"city"};
  spec.tools[0].args = {city};
  GrammarState t(&vocab, spec, false, true);
  // Any other first id settles the opening (the top's rules apply).
  t.advance(kToolOpen);
  require(std::string(t.state_name()) == "q-name" && !t.allows(kThinkOpen),
          "a call first: inside the call no opener");
  feed(t, bytes_of("<function="));
  feed(t, {kGetWeather});
  feed(t, bytes_of("><parameter=city>\"Oslo"));
  require(t.allows(kCity),
          "the straddling '\"\\n' token closes the value and opens the terminator");
  t.advance(kCity);
  require(t.allows('p') && !t.allows('<') && !t.allows('x'), "inside the terminator: '</' next");
  feed(t, bytes_of("parameter></function>"));
  t.advance(kToolClose);
  require(t.active() && std::string(t.state_name()) == "top", "the typed call closed cleanly");
  // Without the choice (a Qwen prompt that opened the block, or one that
  // closed it): the opener is not offered at the first position.
  GrammarState settled(&vocab, spec_of(GrammarSpec::Mode::kRequired), false, false);
  settled.advance(
      kThinkOpen);  // free text at the XML top (the format's own rule): no reasoning state
  require(std::string(settled.state_name()) == "top",
          std::string("no choice: the marker is text at the top: ") + settled.state_name());
  GrammarSpec json_settled;
  json_settled.mode = GrammarSpec::Mode::kJson;
  GrammarState js(&vocab, json_settled, false, false);
  require(!js.allows(kThinkOpen) && js.allows('{'), "no choice in JSON mode: the body only");
  GrammarState opened(&vocab, spec_of(GrammarSpec::Mode::kRequired), true, true);
  require(std::string(opened.state_name()) == "think",
          "a prompt that opened the block starts in the reasoning");
  // JSON mode: <think> first, then the JSON body after </think>.
  GrammarSpec json;
  json.mode = GrammarSpec::Mode::kJson;
  GrammarState j(&vocab, json, false, true);
  require(j.allows(kThinkOpen) && j.allows('{') && !j.allows('x'),
          "JSON mode: the opener or the body");
  j.advance(kThinkOpen);
  feed(j, bytes_of("why"));
  j.advance(kThinkClose);
  require(std::string(j.state_name()) == "json-body" || j.allows('{'),
          std::string("the body after the block: ") + j.state_name());
  require(j.allows('{') && !j.allows('x'), "the JSON body follows");
}

// A schema that declares properties and does not opt out with an explicit
// `additionalProperties: true` closes the parameter names: an open name slot
// is free text, which is what lets the model write an undeclared or a
// repeated name (the wire evidence in
// benchmarks/results/2026-09-19-tool-key-closure.md).
DGPP_TEST(tool_grammar_open_schema_closes_the_parameter_names) {
  const GrammarVocab vocab = qwen_vocab();
  const dgpp::minijson::ParseResult def = dgpp::minijson::parse(
      R"({"name":"get_weather","parameters":{"type":"object","properties":)"
      R"({"city":{"type":"string"},"days":{"type":"number"}},"required":["city"]}})");
  std::vector<std::string> notes;
  GrammarSpec spec;
  spec.mode = GrammarSpec::Mode::kRequired;
  spec.tools.push_back(dgpp::text::grammar_tool_from_function(def.root, nullptr, &notes));
  require(spec.tools[0].constrain_keys &&
              spec.tools[0].keys == std::vector<std::string>{"city", "days"} && notes.empty(),
          "an unspecified additionalProperties closes the declared names");
  GrammarState g(&vocab, spec, /*prompt_opens_thinking=*/false);
  g.advance(kToolOpen);
  feed(g, bytes_of("\n<function=get_weather>\n<parameter="));
  require(std::string(g.state_name()) == "q-key-or-close",
          std::string("after <parameter=: ") + g.state_name());
  require(g.allows('c') && g.allows('d') && !g.allows('l') && !g.allows('x'),
          "only the declared names start a key: " + show(allowed_ids(g)));
  feed(g, bytes_of("city>\nRome\n</parameter>\n<parameter="));
  require(g.allows('d') && !g.allows('c'),
          "a declared name is offered once: " + show(allowed_ids(g)));
  // days is a number: its value is the JSON machine, not free text.
  feed(g, bytes_of("days>\n"));
  require(g.allows('5') && !g.allows('x'),
          "a number property opens the JSON machine: " + show(allowed_ids(g)));
  feed(g, bytes_of("5\n</parameter>\n</function>\n"));
  require(same(allowed_ids(g), {kToolClose}),
          "every declared name used: only </tool_call>: " + show(allowed_ids(g)));
  g.advance(kToolClose);
  require(g.active() && std::string(g.state_name()) == "top", "the call closed cleanly");
  // An explicit opt-out keeps the free name slot, and says so once.
  const dgpp::minijson::ParseResult open =
      dgpp::minijson::parse(R"({"name":"get_weather","parameters":{"type":"object","properties":)"
                            R"({"city":{"type":"string"}},"additionalProperties":true}})");
  std::vector<std::string> free_notes;
  const GrammarTool f = dgpp::text::grammar_tool_from_function(open.root, nullptr, &free_notes);
  require(!f.constrain_keys && f.keys.empty() && free_notes.size() == 1 &&
              free_notes[0].find("get_weather") != std::string::npos &&
              free_notes[0].find("additionalProperties") != std::string::npos,
          "an explicit additionalProperties:true keeps the keys free, noted");
}

// ---- the DeepSeek-V4.1 DSML format --------------------------------
// One marker id (the tag token, empty text like every special token) inside
// text tags: "<" TAG " calls>\n", "<" TAG " invoke name=\"NAME\">\n", the
// parameters "<" TAG " parameter name=\"K\" string=\"true|false\">" V "</" TAG
// " parameter>\n", "</" TAG " invoke>\n", "</" TAG " calls>" then EOS.
constexpr int64_t kDsmlTag = 308;
constexpr int64_t kQuotedLt = 265;
// The V4 vocabulary's id that spans an invoke header's end and the blank
// line of a call without parameters (the real tokenizer has one).
constexpr int64_t kHeadEndBlank = 266;
GrammarVocab dsml_vocab(dgpp::text::DsmlDialect dialect = dgpp::text::DsmlDialect::kV41) {
  std::vector<std::string> texts(static_cast<size_t>(kVocab));
  for (int b = 0; b < 256; ++b) texts[static_cast<size_t>(b)] = std::string(1, static_cast<char>(b));
  texts[kGet] = "get";
  texts[kWeather] = "_weather";
  texts[kGetWeather] = "get_weather";
  texts[kUnderscore] = "_";
  texts[kWea] = "wea";
  texts[kTher] = "ther";
  texts[kCity] = "city";
  texts[kGetT] = "get_t";
  texts[kIme] = "ime";
  texts[kThinkOpen] = "<think>";
  texts[kThinkClose] = "</think>";
  texts[kQuotedLt] = "\"x<";
  if (dialect == dgpp::text::DsmlDialect::kV4) texts[kHeadEndBlank] = "\">\n\n";
  ChatMarkers m;
  m.think_open = ChatMarker{kThinkOpen, "<think>"};
  m.think_close = ChatMarker{kThinkClose, "</think>"};
  m.dsml = ChatMarker{kDsmlTag, "｜DSML｜"};
  m.dsml_dialect = dialect;
  return GrammarVocab(std::move(texts), m, {kEosText}, kVocab, kEosText);
}

DGPP_TEST(tool_grammar_dsml_required_call_walks_the_tagged_shape) {
  const GrammarVocab vocab = dsml_vocab();
  require(vocab.usable() && vocab.markers().tool_format() == dgpp::text::ToolFormat::kDsml,
          "the tag-token vocabulary is the DSML format");
  GrammarState g(&vocab, spec_of(GrammarSpec::Mode::kRequired), /*prompt_opens_thinking=*/false);
  // The top is free text (the content before the block); the tag needs
  // its "<"; the turn may not end while the call is owed.
  require(g.allows('H') && !g.allows(kDsmlTag) && !g.allows(kEosText), "top: text, no bare tag, no EOS while owed");
  feed(g, bytes_of("Sure.\n\n"));
  require(!g.allows(kDsmlTag), "the tag is not offered until a '<'");
  g.advance('<');
  require(g.allows(kDsmlTag) && g.allows('a'), "after '<': the tag (or more text)");
  g.advance('a');
  require(!g.allows(kDsmlTag), "text after the '<' withdraws the tag");
  feed(g, bytes_of(" <"));
  g.advance(kDsmlTag);
  require(std::string(g.state_name()) == "d-calls", std::string("after the tag: ") + g.state_name());
  require(same(allowed_ids(g), {' '}), "the block opens with ' calls>': " + show(allowed_ids(g)));
  feed(g, bytes_of(" calls>\n"));
  require(std::string(g.state_name()) == "d-invoke", "then an invoke");
  // "<" TAG " invoke name=\"" NAME "\">\n": the '<', the tag alone, the literal, the names.
  require(same(allowed_ids(g), {'<'}), "an invoke opens with '<'");
  g.advance('<');
  require(same(allowed_ids(g), {kDsmlTag}), "the tag alone after the '<' in the block: " + show(allowed_ids(g)));
  g.advance(kDsmlTag);
  require(!g.allows(kDsmlTag) && g.allows(' '), "no token runs across the tag");
  feed(g, bytes_of(" invoke name=\""));
  require(g.allows(kGet) && g.allows(kGetWeather) && g.allows('p') && g.allows(kGetT) && !g.allows('x'),
          "names over token texts");
  feed(g, {kGet, kWeather});
  feed(g, bytes_of("\">\n"));
  require(std::string(g.state_name()) == "d-param-or-close", std::string("after the name: ") + g.state_name());
  // Closed keys city/days, or the invoke's close: everything starts with '<'.
  require(same(allowed_ids(g), {'<'}), "param-or-close: '<' only");
  g.advance('<');
  require(g.allows(kDsmlTag) && g.allows('/') && !g.allows('x'), "the tag (a parameter) or '/' (the close)");
  g.advance(kDsmlTag);
  feed(g, bytes_of(" parameter name=\""));
  require(g.allows('c') && g.allows('d') && !g.allows('x'), "closed keys city/days");
  feed(g, bytes_of("city\" string=\""));
  require(std::string(g.state_name()) == "d-flag", "the string flag follows a key");
  require(g.allows('t') && g.allows('f'), "a free value: string true or false");
  feed(g, bytes_of("true\">"));
  require(std::string(g.state_name()) == "d-value", "a free value follows the flag");
  // The free value: anything but the markers, the think markers, EOS and
  // the tag — until a "</" makes the tag the closer's start.
  const std::vector<int64_t> v = allowed_ids(g);
  require(!v.empty() && std::find(v.begin(), v.end(), kDsmlTag) == v.end() &&
              std::find(v.begin(), v.end(), kEosText) == v.end() &&
              std::find(v.begin(), v.end(), kThinkOpen) == v.end() &&
              std::find(v.begin(), v.end(), 'P') != v.end(),
          "free value: text only");
  feed(g, bytes_of("Paris </b> <"));
  require(!g.allows(kDsmlTag), "a lone '<' in the value is text");
  g.advance('/');
  require(g.allows(kDsmlTag) && g.allows('x'), "after \"</\" the tag opens the closer (or the text goes on)");
  g.advance(kDsmlTag);
  require(same(allowed_ids(g), {' '}), "after the closer's tag: its literal only: " + show(allowed_ids(g)));
  feed(g, bytes_of(" parameter>\n"));
  require(std::string(g.state_name()) == "d-param-or-close", "the closer ends the value");
  // days is JSON-typed: the flag is forced false; the value a JSON integer.
  GrammarSpec typed = spec_of(GrammarSpec::Mode::kRequired);
  (void)typed;
  feed(g, bytes_of("<"));
  g.advance(kDsmlTag);
  feed(g, bytes_of(" parameter name=\""));
  require(g.allows('d') && !g.allows('c'), "a closed key is offered once");
  feed(g, bytes_of("days\" string=\"false\">3</"));
  g.advance(kDsmlTag);
  feed(g, bytes_of(" parameter>\n"));
  // Every key used: only the invoke's close remains.
  require(same(allowed_ids(g), {'<'}), "only the close remains");
  g.advance('<');
  require(same(allowed_ids(g), {'/'}), "the close's '/'");
  g.advance('/');
  require(same(allowed_ids(g), {kDsmlTag}), "the close's tag");
  g.advance(kDsmlTag);
  feed(g, bytes_of(" invoke>\n"));
  require(std::string(g.state_name()) == "d-invoke-or-close", std::string("after the invoke: ") + g.state_name());
  // Required (parallel): another invoke or the block's close, then EOS.
  g.advance('<');
  require(g.allows(kDsmlTag) && g.allows('/'), "another invoke or the close");
  g.advance('/');
  g.advance(kDsmlTag);
  feed(g, bytes_of(" calls>"));
  require(same(allowed_ids(g), {kEosText}), "after the block: EOS only: " + show(allowed_ids(g)));
  g.advance(kEosText);
  require(std::string(g.state_name()) == "done", "done after EOS");
}

DGPP_TEST(tool_grammar_dsml_typed_values_named_single_and_open_keys) {
  const GrammarVocab vocab = dsml_vocab();
  // Typed values: a JSON integer (the flag forced false) and an enum
  // string (forced true); named: exactly one invoke.
  GrammarSpec spec = spec_of(GrammarSpec::Mode::kNamed, false, "get_weather");
  GrammarArg days;
  days.key = "days";
  days.kind = GrammarArg::Kind::kJson;
  days.schema = R"({"type": "integer"})";
  GrammarArg unit;
  unit.key = "unit";
  unit.kind = GrammarArg::Kind::kText;
  unit.texts = {"celsius", "fahrenheit"};
  spec.tools[0].keys = {"days", "unit"};
  spec.tools[0].args = {days, unit};
  GrammarState t(&vocab, spec, false);
  t.advance('<');
  t.advance(kDsmlTag);
  feed(t, bytes_of(" calls>\n<"));
  t.advance(kDsmlTag);
  feed(t, bytes_of(" invoke name=\""));
  require(t.allows(kGetWeather) && !t.allows(kGetT) && !t.allows('p'), "named: only get_weather");
  feed(t, {kGetWeather});
  feed(t, bytes_of("\">\n<"));
  t.advance(kDsmlTag);
  feed(t, bytes_of(" parameter name=\"days\" string=\""));
  require(t.allows('f') && !t.allows('t'), "a JSON-typed value is not a string");
  feed(t, bytes_of("false\">"));
  require(t.allows('3') && !t.allows('x') && !t.allows('<'), "a JSON integer: digits first");
  feed(t, bytes_of("3"));
  require(t.allows('4') && t.allows('<') && !t.allows(kDsmlTag), "more digits, or the closer's '<' once complete");
  feed(t, bytes_of("<"));
  require(same(allowed_ids(t), {'/'}), "inside the closer only its bytes");
  t.advance('/');
  require(same(allowed_ids(t), {kDsmlTag}), "then the tag");
  t.advance(kDsmlTag);
  feed(t, bytes_of(" parameter>\n<"));
  t.advance(kDsmlTag);
  feed(t, bytes_of(" parameter name=\"unit\" string=\""));
  require(t.allows('t') && !t.allows('f'), "an enum value is a string");
  feed(t, bytes_of("true\">"));
  require(t.allows('c') && t.allows('f') && !t.allows('k'), "an enum value: its texts");
  feed(t, bytes_of("celsius"));
  require(t.allows('<') && !t.allows('c'), "the enum text then its closer");
  feed(t, bytes_of("</"));
  t.advance(kDsmlTag);
  feed(t, bytes_of(" parameter>\n</"));
  t.advance(kDsmlTag);
  feed(t, bytes_of(" invoke>\n"));
  // Named: no second invoke — the block closes.
  require(same(allowed_ids(t), {'<'}), "the block's close only");
  t.advance('<');
  require(same(allowed_ids(t), {'/'}), "no second invoke under named");
  feed(t, bytes_of("/"));
  t.advance(kDsmlTag);
  feed(t, bytes_of(" calls>"));
  require(same(allowed_ids(t), {kEosText}), "named single: EOS only");
  // An open key set (get_time): free text through "\" string=\"".
  GrammarState o(&vocab, spec_of(GrammarSpec::Mode::kNamed, false, "get_time"), false);
  o.advance('<');
  o.advance(kDsmlTag);
  feed(o, bytes_of(" calls>\n<"));
  o.advance(kDsmlTag);
  feed(o, bytes_of(" invoke name=\""));
  feed(o, {kGetT, kIme});
  feed(o, bytes_of("\">\n<"));
  o.advance(kDsmlTag);
  feed(o, bytes_of(" parameter name=\""));
  require(std::string(o.state_name()) == "d-free-key", "an open key set is free text");
  require(o.allows('t') && !o.allows(kDsmlTag) && !o.allows(kEosText), "free key text");
  feed(o, bytes_of("tz\" string=\"true\">UTC</"));
  o.advance(kDsmlTag);
  feed(o, bytes_of(" parameter>\n</"));
  o.advance(kDsmlTag);
  feed(o, bytes_of(" invoke>\n</"));
  o.advance(kDsmlTag);
  feed(o, bytes_of(" calls>"));
  require(same(allowed_ids(o), {kEosText}), "the open-key call closed cleanly");
  // Auto: text may end the turn without a call; a block, once opened,
  // completes; a disallowed id kills the grammar.
  GrammarState a(&vocab, spec_of(GrammarSpec::Mode::kAuto), false);
  require(a.allows(kEosText) && a.allows('x') && !a.allows(kDsmlTag), "auto: free, EOS allowed, no bare tag");
  a.advance('<');
  a.advance(kDsmlTag);
  require(std::string(a.state_name()) == "d-calls" && !a.allows(kEosText), "a block owes its close");
  a.advance(kEosText);
  require(!a.active() && std::string(a.state_name()) == "dead", "a disallowed id kills the grammar");
}

// The DeepSeek-V4 dialect (encoding_dsv4.py): the same tag token and
// states, the tags spelled without the space after the tag and the block
// named "tool_calls" — "<" TAG "tool_calls>\n", "<" TAG "invoke name=...",
// "<" TAG "parameter name=...", "</" TAG "parameter>\n", "</" TAG
// "invoke>\n", "</" TAG "tool_calls>".
DGPP_TEST(tool_grammar_dsml_v4_dialect_walks_its_own_spelling) {
  const GrammarVocab vocab = dsml_vocab(dgpp::text::DsmlDialect::kV4);
  require(vocab.usable() && vocab.markers().tool_format() == dgpp::text::ToolFormat::kDsml &&
              vocab.markers().dsml_dialect == dgpp::text::DsmlDialect::kV4,
          "the tag-token vocabulary in the V4 dialect");
  GrammarState g(&vocab, spec_of(GrammarSpec::Mode::kRequired), /*prompt_opens_thinking=*/false);
  require(g.allows('H') && !g.allows(kDsmlTag) && !g.allows(kEosText), "top: text, no bare tag, no EOS while owed");
  feed(g, bytes_of("Sure.\n\n<"));
  g.advance(kDsmlTag);
  require(std::string(g.state_name()) == "d-calls", std::string("after the tag: ") + g.state_name());
  require(same(allowed_ids(g), {'t'}), "the block opens with 'tool_calls>', no space: " + show(allowed_ids(g)));
  feed(g, bytes_of("tool_calls>\n"));
  require(std::string(g.state_name()) == "d-invoke", "then an invoke");
  require(same(allowed_ids(g), {'<'}), "an invoke opens with '<'");
  g.advance('<');
  require(same(allowed_ids(g), {kDsmlTag}), "the tag alone after the '<' in the block: " + show(allowed_ids(g)));
  g.advance(kDsmlTag);
  require(same(allowed_ids(g), {'i'}), "'invoke' right after the tag: " + show(allowed_ids(g)));
  feed(g, bytes_of("invoke name=\""));
  require(g.allows(kGet) && g.allows(kGetWeather) && g.allows('p') && g.allows(kGetT) && !g.allows('x'),
          "names over token texts");
  feed(g, {kGet, kWeather});
  feed(g, bytes_of("\">\n"));
  require(std::string(g.state_name()) == "d-param-or-close", std::string("after the name: ") + g.state_name());
  // A parameter, the plain close, or the reference's blank-line close of a
  // call without parameters (offered only before the first parameter).
  require(same(allowed_ids(g), {'\n', '<'}), "param-or-close: '<', or the blank line: " + show(allowed_ids(g)));
  g.advance('<');
  require(g.allows(kDsmlTag) && g.allows('/') && !g.allows('x'), "the tag (a parameter) or '/' (the close)");
  g.advance(kDsmlTag);
  require(same(allowed_ids(g), {'p'}), "'parameter' right after the tag: " + show(allowed_ids(g)));
  feed(g, bytes_of("parameter name=\""));
  require(g.allows('c') && g.allows('d') && !g.allows('x'), "closed keys city/days");
  feed(g, bytes_of("city\" string=\""));
  require(std::string(g.state_name()) == "d-flag", "the string flag follows a key");
  feed(g, bytes_of("true\">"));
  require(std::string(g.state_name()) == "d-value", "a free value follows the flag");
  feed(g, bytes_of("Paris </b> <"));
  require(!g.allows(kDsmlTag), "a lone '<' in the value is text");
  g.advance('/');
  require(g.allows(kDsmlTag) && g.allows('x'), "after \"</\" the tag opens the closer (or the text goes on)");
  g.advance(kDsmlTag);
  require(same(allowed_ids(g), {'p'}), "after the closer's tag: its literal only: " + show(allowed_ids(g)));
  feed(g, bytes_of("parameter>\n"));
  require(std::string(g.state_name()) == "d-param-or-close", "the closer ends the value");
  require(same(allowed_ids(g), {'<'}), "after a parameter: no blank-line close: " + show(allowed_ids(g)));
  feed(g, bytes_of("<"));
  g.advance(kDsmlTag);
  feed(g, bytes_of("parameter name=\""));
  require(g.allows('d') && !g.allows('c'), "a closed key is offered once");
  feed(g, bytes_of("days\" string=\"false\">3</"));
  g.advance(kDsmlTag);
  feed(g, bytes_of("parameter>\n"));
  require(same(allowed_ids(g), {'<'}), "only the close remains");
  feed(g, bytes_of("</"));
  require(same(allowed_ids(g), {kDsmlTag}), "the close's tag");
  g.advance(kDsmlTag);
  feed(g, bytes_of("invoke>\n"));
  require(std::string(g.state_name()) == "d-invoke-or-close", std::string("after the invoke: ") + g.state_name());
  // A second invoke without parameters, closed the reference's way (a blank
  // line between its tags), then the block's close and EOS.
  g.advance('<');
  g.advance(kDsmlTag);
  feed(g, bytes_of("invoke name=\"ping\">\n"));
  require(same(allowed_ids(g), {'\n', '<'}), "an empty closed key set: the close, plain or blank-line: " + show(allowed_ids(g)));
  g.advance('\n');
  require(same(allowed_ids(g), {'<'}), "after the blank line: the close");
  g.advance('<');
  require(same(allowed_ids(g), {'/'}), "the blank line commits the close (no parameter after it)");
  g.advance('/');
  g.advance(kDsmlTag);
  feed(g, bytes_of("invoke>\n"));
  require(std::string(g.state_name()) == "d-invoke-or-close", "the blank-line close ends the call");
  // The same call when one id spans the header's end and the blank line
  // (the real tokenizer's "\">\n\n"): allowed after a name whose call may
  // close empty, and then only the close follows.
  g.advance('<');
  g.advance(kDsmlTag);
  feed(g, bytes_of("invoke name=\"ping"));
  require(same(allowed_ids(g), {'"', kHeadEndBlank}), "the header's end, alone or with the blank line: " + show(allowed_ids(g)));
  g.advance(kHeadEndBlank);
  require(std::string(g.state_name()) == "d-invoke-or-close" && same(allowed_ids(g), {'<'}),
          std::string("inside the whole-call target: ") + g.state_name() + " " + show(allowed_ids(g)));
  g.advance('<');
  require(same(allowed_ids(g), {'/'}), "the close, not a parameter");
  g.advance('/');
  g.advance(kDsmlTag);
  feed(g, bytes_of("invoke>\n"));
  require(std::string(g.state_name()) == "d-invoke-or-close", "the spanning id's call closed");
  feed(g, bytes_of("</"));
  g.advance(kDsmlTag);
  require(same(allowed_ids(g), {'t'}), "the block's close is 'tool_calls>': " + show(allowed_ids(g)));
  feed(g, bytes_of("tool_calls>"));
  require(same(allowed_ids(g), {kEosText}), "after the block: EOS only: " + show(allowed_ids(g)));
  g.advance(kEosText);
  require(std::string(g.state_name()) == "done", "done after EOS");
}

DGPP_TEST(tool_grammar_dsml_dialects_do_not_cross_and_v4_typed_named) {
  const GrammarVocab v41 = dsml_vocab();
  const GrammarVocab v4 = dsml_vocab(dgpp::text::DsmlDialect::kV4);
  // The V4.1 spelling dies in the V4 grammar at the block's name, and the
  // V4 spelling in the V4.1 grammar: a wrong dialect never half-works.
  GrammarState a(&v4, spec_of(GrammarSpec::Mode::kRequired), false);
  a.advance('<');
  a.advance(kDsmlTag);
  require(!a.allows(' ') && a.allows('t'), "V4: no space after the tag");
  a.advance(' ');
  require(!a.active() && std::string(a.state_name()) == "dead", "the V4.1 spelling kills the V4 grammar");
  GrammarState b(&v41, spec_of(GrammarSpec::Mode::kRequired), false);
  b.advance('<');
  b.advance(kDsmlTag);
  require(b.allows(' ') && !b.allows('t'), "V4.1: the space after the tag");
  b.advance('t');
  require(!b.active() && std::string(b.state_name()) == "dead", "the V4 spelling kills the V4.1 grammar");
  // V4.1 keeps its single close of an invoke without parameters.
  GrammarState c(&v41, spec_of(GrammarSpec::Mode::kNamed, false, "ping"), false);
  c.advance('<');
  c.advance(kDsmlTag);
  feed(c, bytes_of(" calls>\n<"));
  c.advance(kDsmlTag);
  feed(c, bytes_of(" invoke name=\"ping\">\n"));
  require(same(allowed_ids(c), {'<'}), "V4.1: the plain close only: " + show(allowed_ids(c)));

  // V4, named and typed: a JSON integer (the flag forced false), an enum
  // string (forced true), exactly one invoke; an open key set.
  GrammarSpec spec = spec_of(GrammarSpec::Mode::kNamed, false, "get_weather");
  GrammarArg days;
  days.key = "days";
  days.kind = GrammarArg::Kind::kJson;
  days.schema = R"({"type": "integer"})";
  GrammarArg unit;
  unit.key = "unit";
  unit.kind = GrammarArg::Kind::kText;
  unit.texts = {"celsius", "fahrenheit"};
  spec.tools[0].keys = {"days", "unit"};
  spec.tools[0].args = {days, unit};
  spec.tools[0].required_keys = {"days"};
  spec.tools[0].strict = true;
  GrammarState t(&v4, spec, false);
  t.advance('<');
  t.advance(kDsmlTag);
  feed(t, bytes_of("tool_calls>\n<"));
  t.advance(kDsmlTag);
  feed(t, bytes_of("invoke name=\""));
  require(t.allows(kGetWeather) && !t.allows(kGetT) && !t.allows('p'), "named: only get_weather");
  feed(t, {kGetWeather});
  feed(t, bytes_of("\">\n"));
  require(same(allowed_ids(t), {'<'}), "a strict call owing a required key: no close of either form: " + show(allowed_ids(t)));
  t.advance('<');
  require(same(allowed_ids(t), {kDsmlTag}), "a parameter must open");
  t.advance(kDsmlTag);
  feed(t, bytes_of("parameter name=\"days\" string=\""));
  require(t.allows('f') && !t.allows('t'), "a JSON-typed value is not a string");
  feed(t, bytes_of("false\">"));
  require(t.allows('3') && !t.allows('x') && !t.allows('<'), "a JSON integer: digits first");
  feed(t, bytes_of("3"));
  require(t.allows('4') && t.allows('<') && !t.allows(kDsmlTag), "more digits, or the closer's '<' once complete");
  feed(t, bytes_of("</"));
  require(same(allowed_ids(t), {kDsmlTag}), "then the tag");
  t.advance(kDsmlTag);
  feed(t, bytes_of("parameter>\n<"));
  t.advance(kDsmlTag);
  feed(t, bytes_of("parameter name=\"unit\" string=\""));
  require(t.allows('t') && !t.allows('f'), "an enum value is a string");
  feed(t, bytes_of("true\">"));
  require(t.allows('c') && t.allows('f') && !t.allows('k'), "an enum value: its texts");
  feed(t, bytes_of("celsius</"));
  t.advance(kDsmlTag);
  feed(t, bytes_of("parameter>\n</"));
  t.advance(kDsmlTag);
  feed(t, bytes_of("invoke>\n"));
  require(same(allowed_ids(t), {'<'}), "the block's close only");
  t.advance('<');
  require(same(allowed_ids(t), {'/'}), "no second invoke under named");
  t.advance('/');
  t.advance(kDsmlTag);
  feed(t, bytes_of("tool_calls>"));
  require(same(allowed_ids(t), {kEosText}), "named single: EOS only");
  GrammarState o(&v4, spec_of(GrammarSpec::Mode::kNamed, false, "get_time"), false);
  o.advance('<');
  o.advance(kDsmlTag);
  feed(o, bytes_of("tool_calls>\n<"));
  o.advance(kDsmlTag);
  feed(o, bytes_of("invoke name=\""));
  feed(o, {kGetT, kIme});
  feed(o, bytes_of("\">\n<"));
  o.advance(kDsmlTag);
  feed(o, bytes_of("parameter name=\""));
  require(std::string(o.state_name()) == "d-free-key", "an open key set is free text");
  feed(o, bytes_of("tz\" string=\"true\">UTC</"));
  o.advance(kDsmlTag);
  feed(o, bytes_of("parameter>\n"));
  require(same(allowed_ids(o), {'<'}), "after a free-key parameter: no blank-line close: " + show(allowed_ids(o)));
  feed(o, bytes_of("</"));
  o.advance(kDsmlTag);
  feed(o, bytes_of("invoke>\n</"));
  o.advance(kDsmlTag);
  feed(o, bytes_of("tool_calls>"));
  require(same(allowed_ids(o), {kEosText}), "the open-key call closed cleanly");
  // Auto: free text may end the turn; a block, once opened, completes.
  GrammarState u(&v4, spec_of(GrammarSpec::Mode::kAuto), false);
  require(u.allows(kEosText) && u.allows('x') && !u.allows(kDsmlTag), "auto: free, EOS allowed, no bare tag");
  u.advance('<');
  u.advance(kDsmlTag);
  require(std::string(u.state_name()) == "d-calls" && !u.allows(kEosText), "a block owes its close");
}

DGPP_TEST(tool_grammar_requiredOwesACallAndForcesItsShape) {
  const GrammarVocab v = fake_vocab();
  GrammarState g(&v, spec_of(GrammarSpec::Mode::kRequired), /*thinking=*/true);
  // Thinking: free except the three EOS ids (the turn may not end).
  {
    const std::vector<int64_t> a = allowed_ids(g);
    require(a.size() == static_cast<size_t>(kVocab - 3), "think allows all but EOS: " +
                                                             std::to_string(a.size()));
    require(g.allows('a') && g.allows(kThinkClose) && g.allows(kToolOpen) &&
                !g.allows(kEosObs) && !g.allows(kEosUser),
            "think mask membership");
  }
  feed(g, {'h', 'm', kThinkClose});
  // After </think> under required: only <tool_call>.
  require(same(allowed_ids(g), {kToolOpen}), "top owes a call: " + show(allowed_ids(g)));
  feed(g, {kToolOpen});
  // The name: any token that continues a tool name from its start.
  require(same(allowed_ids(g), {'g', 'p', kGet, kGetWeather, kGetT}),
          "name starts: " + show(allowed_ids(g)));
  feed(g, {kGet});
  // "get" emitted: "_" (either name), "_weather", "_t" is not a token here.
  require(same(allowed_ids(g), {'_', kUnderscore, kWeather}),
          "after 'get': " + show(allowed_ids(g)));
  feed(g, {kUnderscore, kWea});  // "get_wea"
  require(same(allowed_ids(g), {'t', kTher}), "after 'get_wea': " + show(allowed_ids(g)));
  feed(g, {kTher});  // "get_weather" complete, no longer name extends it
  require(same(allowed_ids(g), {kKeyOpen, kToolClose}),
          "complete name: <arg_key> or </tool_call>: " + show(allowed_ids(g)));
  feed(g, {kKeyOpen});
  // Closed keys: "city" or "days" — by their tokens.
  require(same(allowed_ids(g), {'c', 'd', kCity}), "keys: " + show(allowed_ids(g)));
  feed(g, {kCity});
  require(same(allowed_ids(g), {kKeyClose}), "complete key");
  feed(g, {kKeyClose});
  require(same(allowed_ids(g), {kValueOpen}), "after key");
  feed(g, {kValueOpen});
  // Values are free text: everything but the markers (and EOS: a call is
  // still owed until it closes), plus the closer.
  {
    const std::vector<int64_t> a = allowed_ids(g);
    require(g.allows('P') && g.allows(kValueClose) && g.allows(kThinkOpen) &&
                !g.allows(kKeyOpen) && !g.allows(kToolOpen) && !g.allows(kEosObs),
            "value mask membership");
    require(a.size() == static_cast<size_t>(kVocab - 5 - 3), "value allows all but 5 "
            "markers and 3 EOS: " + std::to_string(a.size()));
  }
  feed(g, {'P', 'a', kValueClose});
  require(same(allowed_ids(g), {kKeyOpen, kToolClose}), "after value");
  feed(g, {kToolClose});
  // One call closed, parallel: another call or the turn end (only the
  // call-turn EOS).
  require(same(allowed_ids(g), {kToolOpen, kEosObs}),
          "after a call (parallel): " + show(allowed_ids(g)));
  feed(g, {kEosObs});
  require(same(allowed_ids(g), {kEosObs}), "done: EOS again");
  require(g.active(), "still active (never died)");
}

DGPP_TEST(tool_grammar_namedSingleCallThenEos) {
  const GrammarVocab v = fake_vocab();
  GrammarState g(&v, spec_of(GrammarSpec::Mode::kNamed, true, "get_time"),
                 /*thinking=*/false);
  require(same(allowed_ids(g), {kToolOpen}), "no think: a call right away");
  feed(g, {kToolOpen});
  require(same(allowed_ids(g), {'g', kGet, kGetT}),
          "only get_time's starts: " + show(allowed_ids(g)));
  feed(g, {kGetT, kIme});
  // get_time's keys are open: free text or the closers.
  require(same(allowed_ids(g), {kKeyOpen, kToolClose}), "name complete");
  feed(g, {kKeyOpen});
  require(g.allows('z') && g.allows(kKeyClose) && !g.allows(kValueOpen),
          "open keys are free text");
  feed(g, {'z', kKeyClose, kValueOpen, '1', kValueClose, kToolClose});
  // Named: exactly one call, then the turn ends.
  require(same(allowed_ids(g), {kEosObs}), "named: EOS after the one call");
  // A tool without arguments closes right after its name.
  GrammarState p(&v, spec_of(GrammarSpec::Mode::kNamed, true, "ping"), false);
  feed(p, {kToolOpen, 'p', 'i', 'n', 'g'});
  require(same(allowed_ids(p), {kToolClose}), "closed empty keys: no <arg_key>");
}

DGPP_TEST(tool_grammar_autoSingleAndForbidModes) {
  const GrammarVocab v = fake_vocab();
  // auto + parallel false: free until a call opens, then that call, then EOS.
  GrammarState a(&v, spec_of(GrammarSpec::Mode::kAuto, false), true);
  require(!a.active() || allowed_ids(a).empty(), "think is unconstrained (no call owed)");
  feed(a, {'x', kThinkClose});
  require(a.allows('H') && a.allows(kEosUser) && a.allows(kToolOpen) &&
              !a.allows(kKeyOpen),
          "top: free, calls may open, stray markers may not");
  feed(a, {'H', 'i', kToolOpen, kGetWeather, kToolClose});
  require(same(allowed_ids(a), {kEosObs}), "auto single: EOS after the one call");
  // forbid: everything but <tool_call> (and the other markers).
  GrammarState f(&v, spec_of(GrammarSpec::Mode::kForbidCalls), true);
  feed(f, {kThinkClose});
  require(f.allows('H') && f.allows(kEosUser) && !f.allows(kToolOpen) &&
              !f.allows(kToolClose),
          "forbid: no call may open");
  {
    TokenMask m;
    f.mask(&m);
    require(m.allowed == kVocab - 6, "forbid allows all but the six markers");
  }
  // required + parallel false: exactly one call.
  GrammarState r(&v, spec_of(GrammarSpec::Mode::kRequired, false), false);
  feed(r, {kToolOpen, kGetWeather, kToolClose});
  require(same(allowed_ids(r), {kEosObs}), "required single: EOS");
}

DGPP_TEST(tool_grammar_disallowedIdKillsTheGrammar) {
  const GrammarVocab v = fake_vocab();
  GrammarState g(&v, spec_of(GrammarSpec::Mode::kRequired), false);
  require(!g.allows('H'), "text before the call is not allowed");
  g.advance('H');  // the MTP draft may propose it; the verify rejects it
  require(!g.active(), "dead after a disallowed id");
  TokenMask m;
  g.mask(&m);
  require(!m.constrained(), "a dead grammar constrains nothing");
  // An inactive spec is never active.
  GrammarState none(&v, GrammarSpec{}, true);
  require(!none.active(), "kNone is inactive");
  none.mask(&m);
  require(!m.constrained(), "inactive: no mask");
  // A named function outside the tools refuses.
  bool threw = false;
  try {
    GrammarState bad(&v, spec_of(GrammarSpec::Mode::kNamed, true, "nope"), false);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "named outside tools refuses");
}

DGPP_TEST(tool_grammar_jsonModeSpellsOneTextThenEos) {
  // response_format (M6 6h) as a grammar: thinking stays free but the
  // turn cannot end; </think> opens the JSON body, where the machine's
  // mask rules, the markers never appear, and EOS comes only once the
  // text is complete.
  const GrammarVocab v = fake_vocab();
  GrammarSpec spec;
  spec.mode = GrammarSpec::Mode::kJson;  // json_schema "" = json_object
  GrammarState g(&v, spec, /*prompt_opens_thinking=*/true);
  require(g.active() && std::string(g.state_name()) == "think", "starts thinking");
  require(g.allows('x') && g.allows(kThinkClose) && g.allows(kToolOpen) &&
              !g.allows(kEosText) && !g.allows(kEosObs),
          "thinking is free, EOS withheld");
  g.advance(kThinkClose);
  require(std::string(g.state_name()) == "json-value", "the body opens");
  require(g.allows('{') && g.allows(' ') && g.allows('\n') && !g.allows('"') &&
              !g.allows('[') && !g.allows('1') && !g.allows(kEosText) &&
              !g.allows(kThinkOpen) && !g.allows(kThinkClose) && !g.allows(kToolOpen),
          "json_object: an object opens, nothing else");
  TokenMask m;
  g.mask(&m);
  require(m.constrained() && m.allows('{') && !m.allows(kEosText) && m.allowed >= 5,
          "the body mask is a real constraint");
  for (const char c : std::string("{\"a\":1")) g.advance(static_cast<unsigned char>(c));
  require(g.active() && !g.allows(kEosText) && g.allows(',') && g.allows('}'),
          "inside the object: no EOS yet");
  g.advance('}');
  require(g.allows(kEosText) && g.allows(kEosObs) && g.allows(' ') && !g.allows(',') &&
              !g.allows('{'),
          "complete: EOS or whitespace only");
  g.mask(&m);
  require(m.allows(kEosText) && m.allows(kEosUser) && !m.allows('}'), "done mask");
  // Trailing whitespace is capped: past sixteen bytes only EOS remains.
  for (int i = 0; i < 16; ++i) g.advance(' ');
  g.mask(&m);
  require(m.allows(kEosText) && !m.allows(' ') && !m.allows('\n') && m.allowed == 3,
          "the whitespace cap leaves the EOS ids alone");
  g.advance(kEosText);
  require(std::string(g.state_name()) == "done", "EOS ends the turn");

  // A schema: closed keys spelled from the declared names, an integer
  // value, the closer only once the required key is in.
  GrammarSpec typed;
  typed.mode = GrammarSpec::Mode::kJson;
  typed.json_schema =
      "{\"type\":\"object\",\"properties\":{\"k\":{\"type\":\"integer\"},"
      "\"s\":{\"enum\":[\"on\",\"off\"]}},\"required\":[\"k\"],"
      "\"additionalProperties\":false}";
  GrammarState t(&v, typed, /*prompt_opens_thinking=*/false);
  require(std::string(t.state_name()) == "json-value", "no think block: body at once");
  t.advance('{');
  require(t.allows('"') && !t.allows('}'), "the required key is owed");
  t.advance('"');
  require(t.allows('k') && t.allows('s') && !t.allows('z') && !t.allows('"'),
          "keys from the declared names");
  for (const char c : std::string("k\":")) t.advance(static_cast<unsigned char>(c));
  require(t.allows('-') && t.allows('7') && !t.allows('"') && !t.allows('t'),
          "an integer value");
  t.advance('4');
  require(!t.allows('.') && !t.allows('e') && t.allows('}') && t.allows(','),
          "integer: no fraction; the object may close");
  for (const char c : std::string(",\"s\":\"o")) t.advance(static_cast<unsigned char>(c));
  require(t.allows('n') && t.allows('f') && !t.allows('x') && !t.allows('"'),
          "an enum string is spelled from its targets");
  for (const char c : std::string("ff\"}")) t.advance(static_cast<unsigned char>(c));
  require(t.allows(kEosText), "done under the schema");
  // A disallowed id kills the grammar (the sampler never produces one).
  GrammarState k(&v, typed, false);
  k.advance('[');
  require(!k.active(), "a disallowed id kills the JSON grammar");
  // A schema text that does not parse, or is outside the subset, refuses
  // at construction.
  GrammarSpec bad = typed;
  bad.json_schema = "{not json";
  bool threw = false;
  try {
    GrammarState b(&v, bad, false);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "unparsable schema text refused");
  bad.json_schema = "{\"type\":\"string\",\"pattern\":\"[\"}";
  threw = false;
  try {
    GrammarState b(&v, bad, false);
  } catch (const std::invalid_argument& e) {
    threw = std::string(e.what()).rfind("schema.pattern", 0) == 0;
  }
  require(threw, "an unsupported keyword refused by name");
  // The spec's equality covers the schema text.
  require(!(spec == typed), "specs differ by schema");
  GrammarSpec same = typed;
  require(same == typed, "equal specs");
}

DGPP_TEST(tool_grammar_jsonOrToolsCommitsToOneBranch) {
  for (const GrammarVocab& v : {fake_vocab(), qwen_vocab(), dsml_vocab()}) {
    const bool dsml = v.markers().tool_format() == dgpp::text::ToolFormat::kDsml;
    const bool qwen = v.markers().tool_format() == dgpp::text::ToolFormat::kQwenXml;
    for (const bool parallel : {false, true}) {
      GrammarSpec spec = spec_of(GrammarSpec::Mode::kJsonOrTools, parallel);
      spec.json_schema = R"({"type":"object","properties":{"n":{"type":"number","minimum":0,"maximum":10}},"required":["n"],"additionalProperties":false})";
      GrammarState json(&v, spec, true);
      require(json.allows('x') && !json.allows(kEosText), "thinking precedes JSON or tools");
      feed(json, {kThinkClose, ' ', '\n'});
      require(json.allows('{') && json.allows(dsml ? '<' : kToolOpen) &&
                  !json.allows('x') && !json.allows(kEosText), "choose JSON or a call");
      feed(json, bytes_of("{\"n\":0.5}"));
      require(json.allows(kEosText) && !json.allows(kToolOpen) && !json.allows('<') &&
                  !json.allows('{'), "JSON answer cannot become a call");
      GrammarState call(&v, spec, false);
      if (dsml) {
        feed(call, {'<', kDsmlTag});
        feed(call, bytes_of(" calls>\n<"));
        feed(call, {kDsmlTag});
        feed(call, bytes_of(" invoke name=\"ping\">\n</"));
        feed(call, {kDsmlTag});
        feed(call, bytes_of(" invoke>\n</"));
        feed(call, {kDsmlTag});
        feed(call, bytes_of(" calls>"));
      } else {
        feed(call, {kToolOpen});
        feed(call, bytes_of(qwen ? "\n<function=ping>\n</function>\n" : "ping"));
        feed(call, {kToolClose});
      }
      require(call.allows(v.call_turn_eos()) && !call.allows('{') && !call.allows('x'),
              "tool turn ends without an unconstrained answer");
      if (!dsml) require(call.allows(kToolOpen) == parallel, "parallel flag still enforced");
    }
  }
  const auto v = dsml_vocab();
  GrammarSpec scalar = spec_of(GrammarSpec::Mode::kJsonOrTools);
  scalar.json_schema = R"({"type":"string"})";
  GrammarState string(&v, scalar, false);
  feed(string, {kQuotedLt, '"', kEosText});
  require(string.active(), "a '<' inside a JSON string is not a DSML opener");
}

DGPP_TEST(tool_grammar_typedValuesFollowTheSchema) {
  // Typed arguments (M6 6i): a JSON-typed value runs the JSON machine
  // under the property's schema with </arg_value> only when complete; an
  // enum string is spelled from its texts; a plain string and an unknown
  // key stay free; auto mode arms the shape with calls at will.
  const GrammarVocab v = fake_vocab();
  GrammarSpec spec;
  spec.mode = GrammarSpec::Mode::kAuto;
  spec.parallel = true;
  GrammarTool w;
  w.name = "get_weather";
  w.constrain_keys = false;  // open keys: an unknown key is allowed and free
  w.args.push_back(GrammarArg{"city", GrammarArg::Kind::kFree, "", {}});
  w.args.push_back(GrammarArg{"days", GrammarArg::Kind::kJson, "{\"type\":\"integer\"}", {}});
  w.args.push_back(GrammarArg{"unit", GrammarArg::Kind::kText, "", {"celsius", "fahrenheit"}});
  w.args.push_back(GrammarArg{"opts", GrammarArg::Kind::kJson,
                              "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"boolean\"}},"
                              "\"required\":[\"a\"],\"additionalProperties\":false}", {}});
  spec.tools.push_back(w);
  GrammarState g(&v, spec, /*prompt_opens_thinking=*/false);
  const auto feed = [&](const std::string& text) {
    for (const char c : text) {
      require(g.allows(static_cast<unsigned char>(c)),
              std::string("byte '") + c + "' refused in state " + g.state_name());
      g.advance(static_cast<unsigned char>(c));
    }
  };
  const auto marker = [&](int64_t id) {
    require(g.allows(id), "marker refused in state " + std::string(g.state_name()));
    g.advance(id);
  };
  require(g.allows(kToolOpen) && g.allows(kEosText) && g.allows('x'), "auto: free top");
  marker(kToolOpen);
  feed("get_weather");
  marker(kKeyOpen);
  feed("days");
  marker(kKeyClose);
  marker(kValueOpen);
  // An integer: digits, no quote, no fraction; the closer once complete.
  require(g.allows('-') && g.allows('7') && !g.allows('"') && !g.allows('t') &&
              !g.allows(kValueClose),
          "integer value: the closer waits");
  feed("12");
  require(g.allows('3') && !g.allows('.') && !g.allows('e') && g.allows(kValueClose) &&
              !g.allows(kToolClose) && !g.allows(kEosText),
          "12 is complete: only the closer or more digits");
  TokenMask m;
  g.mask(&m);
  require(m.constrained() && m.allows(kValueClose) && m.allows('3') && !m.allows('.'),
          "the value mask agrees with allows()");
  marker(kValueClose);
  // An enum string: spelled from its texts.
  marker(kKeyOpen);
  feed("unit");
  marker(kKeyClose);
  marker(kValueOpen);
  require(g.allows('c') && g.allows('f') && !g.allows('k') && !g.allows(kValueClose),
          "enum: first bytes only");
  feed("celsiu");
  require(!g.allows(kValueClose) && g.allows('s'), "enum: incomplete");
  feed("s");
  require(g.allows(kValueClose) && !g.allows('x'), "enum: complete");
  marker(kValueClose);
  // A nested object under its schema: closed keys, a boolean, required.
  marker(kKeyOpen);
  feed("opts");
  marker(kKeyClose);
  marker(kValueOpen);
  require(g.allows('{') && !g.allows('[') && !g.allows('1'), "an object value");
  feed("{\"");
  require(g.allows('a') && !g.allows('b'), "closed key");
  feed("a\":");
  require(g.allows('t') && g.allows('f') && !g.allows('1'), "boolean");
  feed("true");
  require(!g.allows(kValueClose) && g.allows('}'), "not closed yet");
  feed("}");
  require(g.allows(kValueClose) && !g.allows(','), "complete object");
  marker(kValueClose);
  // A free string, then an unknown key (open schema): free text.
  marker(kKeyOpen);
  feed("city");
  marker(kKeyClose);
  marker(kValueOpen);
  require(g.allows('P') && g.allows('"') && g.allows(' ') && g.allows(kValueClose) &&
              !g.allows(kToolClose),
          "a string is free (closer allowed at once)");
  feed("Paris 75");
  marker(kValueClose);
  marker(kKeyOpen);
  feed("zzz");
  marker(kKeyClose);
  marker(kValueOpen);
  require(g.allows('{') && g.allows('x') && g.allows(kValueClose), "unknown key: free value");
  marker(kValueClose);
  marker(kToolClose);
  // auto with parallel: another call may open, or the turn may end.
  require(g.allows(kToolOpen) && g.allows(kEosText) && g.allows('x'), "auto: calls at will");
  g.advance(kEosText);
  require(std::string(g.state_name()) == "done", "done");
  // A disallowed byte inside a typed value kills the grammar.
  GrammarState k(&v, spec, false);
  k.advance(kToolOpen);
  for (const char c : std::string("get_weather")) k.advance(static_cast<unsigned char>(c));
  k.advance(kKeyOpen);
  for (const char c : std::string("days")) k.advance(static_cast<unsigned char>(c));
  k.advance(kKeyClose);
  k.advance(kValueOpen);
  k.advance('x');
  require(!k.active(), "a non-integer byte kills the grammar");
  // A JSON-typed argument's schema tolerates a keyword that only narrows
  // the value (2026-09-06: the value stays typed — a letter still kills the
  // grammar) and refuses one outside the subset in shape at construction.
  GrammarSpec lax = spec;
  lax.tools[0].args[1].schema = "{\"type\":\"integer\",\"minimum\":0}";
  {
    GrammarState l(&v, lax, false);
    l.advance(kToolOpen);
    for (const char c : std::string("get_weather")) l.advance(static_cast<unsigned char>(c));
    l.advance(kKeyOpen);
    for (const char c : std::string("days")) l.advance(static_cast<unsigned char>(c));
    l.advance(kKeyClose);
    l.advance(kValueOpen);
    require(l.allows('7') && !l.allows('x'),
            "a tolerated minimum: the value is still an integer");
    l.advance('x');
    require(!l.active(), "a non-integer byte kills the grammar under a tolerated keyword");
  }
  GrammarSpec bad = spec;
  bad.tools[0].args[1].schema = "{\"type\":\"integer\",\"$ref\":\"#/x\"}";
  bool threw = false;
  try {
    GrammarState b(&v, bad, false);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "an unsupported argument schema refused");
  // Equality covers the arguments.
  GrammarSpec same = spec;
  require(same == spec && !(bad == spec), "spec equality over arguments");
}

// The 6i follow-on: a closed key is offered ONCE per call (a duplicate key
// is never a valid object), and a strict tool's call cannot close while a
// required key is missing — OpenAI's strict guarantee, enforced by the
// mask rather than hoped for. A non-strict tool keeps the closer open (its
// required keys are the client's business) but still spends its keys.
DGPP_TEST(tool_grammar_closedKeysOnceAndStrictRequiredKeysGateTheClose) {
  const GrammarVocab v = fake_vocab();
  GrammarSpec s = spec_of(GrammarSpec::Mode::kNamed, false, "get_weather");
  s.tools[0].strict = true;
  s.tools[0].required_keys = {"city"};
  GrammarState g(&v, s, /*thinking=*/false);
  feed(g, {kToolOpen, kGetWeather});
  // The name is complete: a strict tool owing a key cannot close yet.
  require(same(allowed_ids(g), {kKeyOpen}),
          "strict name: only <arg_key>: " + show(allowed_ids(g)));
  feed(g, {kKeyOpen});
  require(same(allowed_ids(g), {'c', 'd', kCity}), "keys: " + show(allowed_ids(g)));
  feed(g, {'d', 'a', 'y', 's', kKeyClose, kValueOpen, '3', kValueClose});
  // "days" used, "city" (required) still missing: another key, no close.
  require(same(allowed_ids(g), {kKeyOpen}),
          "after days: city still owed: " + show(allowed_ids(g)));
  feed(g, {kKeyOpen});
  // "days" is spent: only "city" continues, and 'd' never starts again.
  require(same(allowed_ids(g), {'c', kCity}),
          "keys less the used one: " + show(allowed_ids(g)));
  require(!g.allows('d'), "a duplicate key never starts");
  feed(g, {kCity, kKeyClose, kValueOpen, 'X', kValueClose});
  // Every closed key used: no <arg_key>; every required key present: close.
  require(same(allowed_ids(g), {kToolClose}),
          "all keys spent: only </tool_call>: " + show(allowed_ids(g)));
  feed(g, {kToolClose});
  require(same(allowed_ids(g), {kEosObs}), "named: the turn ends");
  require(g.active(), "still active");

  // The same tool, non-strict: the closer stays open from the name on ...
  GrammarSpec ns = s;
  ns.tools[0].strict = false;
  GrammarState h(&v, ns, /*thinking=*/false);
  feed(h, {kToolOpen, kGetWeather});
  require(same(allowed_ids(h), {kKeyOpen, kToolClose}),
          "non-strict: closable at once: " + show(allowed_ids(h)));
  // ... but a closed key is still offered once.
  feed(h, {kKeyOpen, kCity, kKeyClose, kValueOpen, kValueClose, kKeyOpen});
  require(same(allowed_ids(h), {'d'}),
          "non-strict: the used key is gone: " + show(allowed_ids(h)));
  feed(h, {'d', 'a', 'y', 's', kKeyClose, kValueOpen, kValueClose});
  require(same(allowed_ids(h), {kToolClose}),
          "non-strict, keys spent: only the close: " + show(allowed_ids(h)));

  // The derivation from a function definition: strict rides, required keys
  // are the declared ones, an undeclared name warns and is dropped.
  {
    const dgpp::minijson::ParseResult def = dgpp::minijson::parse(
        R"({"name":"get_weather","strict":true,"parameters":{"type":"object",)"
        R"("properties":{"city":{"type":"string"},"days":{"type":"integer"}},)"
        R"("required":["city","city"],"additionalProperties":false}})");
    std::vector<std::string> warnings;
    const GrammarTool t = dgpp::text::grammar_tool_from_function(def.root, &warnings);
    require(t.strict && t.constrain_keys &&
                t.required_keys == std::vector<std::string>{"city"},
            "derived: strict, closed, the declared required key once");
    require(warnings.empty(), "valid strict schema has no warnings");
    bool invalid_required = false;
    try {
      const auto bad = dgpp::minijson::parse(R"({"name":"f","strict":true,"parameters":{"type":"object","properties":{},"required":["missing"]}})");
      dgpp::text::grammar_tool_from_function(bad.root, nullptr);
    } catch (const std::invalid_argument&) { invalid_required = true; }
    require(invalid_required, "strict tools reject unsatisfiable required keys");
    const dgpp::minijson::ParseResult lax = dgpp::minijson::parse(
        R"({"name":"f","parameters":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}})");
    const GrammarTool u = dgpp::text::grammar_tool_from_function(lax.root, nullptr);
    require(!u.strict && u.constrain_keys && u.keys == std::vector<std::string>{"a"} &&
                u.required_keys == std::vector<std::string>{"a"},
            "derived: a non-strict tool closes its declared names and records its "
            "required keys unenforced");
  }
}

DGPP_TEST(tool_grammar_keyClosureSchemaEdges) {
  struct Case {
    std::string params;
    bool closed;
    std::vector<std::string> keys;
    bool note;
  };
  for (const auto& c : std::vector<Case>{
           {R"({"type":"object","properties":{"x":{"type":"string"}}})", true, {"x"}, false},
           {R"({"properties":{"x":{}},"additionalProperties":false})", true, {"x"}, false},
           {R"({"properties":{"x":{}},"additionalProperties":true})", false, {}, true},
           {R"({"properties":{"x":{}},"additionalProperties":{"type":"number"}})",
            true,
            {"x"},
            false},
           {R"({"properties":{}})", true, {}, false},
           {R"({"properties":{},"additionalProperties":false})", true, {}, false},
           {R"({"properties":{},"additionalProperties":true})", false, {}, true},
           {R"({"type":"object"})", false, {}, false},
           {R"({"type":"object","additionalProperties":false})", false, {}, false}}) {
    const std::string text = R"({"name":"f","parameters":)" + c.params + "}";
    const auto def = dgpp::minijson::parse(text);
    std::vector<std::string> notes;
    const auto tool = dgpp::text::grammar_tool_from_function(def.root, nullptr, &notes);
    require(tool.constrain_keys == c.closed && tool.keys == c.keys, "key policy: " + c.params);
    require(notes.size() == static_cast<size_t>(c.note), "opt-out note: " + c.params);
  }
}

DGPP_TEST(tool_grammar_keyClosureLeavesNestedJsonOpen) {
  const auto def = dgpp::minijson::parse(
      R"({"name":"get_weather","parameters":{"properties":{"options":{"type":"object","properties":{"x":{"type":"number"}}}}}})");
  GrammarSpec spec;
  spec.mode = GrammarSpec::Mode::kRequired;
  spec.tools.push_back(dgpp::text::grammar_tool_from_function(def.root, nullptr));
  require(spec.tools[0].constrain_keys && spec.tools[0].keys == std::vector<std::string>{"options"},
          "only the top-level argument name closes");
  const auto vocab = qwen_vocab();
  GrammarState state(&vocab, spec, false);
  state.advance(kToolOpen);
  feed(state, bytes_of("\n<function=get_weather>\n<parameter=options>\n{\"x\":1,\"extra\":2}\n"
                       "</parameter>\n</function>\n"));
  require(state.allows(kToolClose), "a nested object retains JSON Schema's open default");
}

// ---- the Qwen3-Next JSON format -----------------------------------
// The same two marker ids as the XML format, the block one JSON object in
// the template's spelling: "\n{\"name\": \"NAME\", \"arguments\": " ARGS "}\n".
// The vocabulary carries the Qwen2 tokenizer's pieces around the frame's
// seams (each is an id of the real vocabulary): " {\"" ends the head and
// opens the arguments, "\"}}\n" closes a string, the arguments and the
// frame at once.
constexpr int64_t kJBraceQuote = 270, kJName = 271, kJQuoteColon = 272, kJSpaceQuote = 273,
                  kJQuoteComma = 274, kJArguments = 275, kJSpaceBraceQuote = 276, kJSpaceEmpty = 277,
                  kJQuoteCloseAll = 278, kJCloseBoth = 279, kJCloseBothNl = 280, kJCloseNl = 281,
                  kJColonSpaceBrace = 282, kJSpaceBraceNl = 283, kJTwoSpaces = 284, kJCloseBothNlNl = 285,
                  kJSpaceBrace = 286, kJQuoteClose = 287, kJSpaceEmptyNl = 288;
GrammarVocab json_vocab() {
  std::vector<std::string> texts(static_cast<size_t>(kVocab));
  for (int b = 0; b < 256; ++b) texts[static_cast<size_t>(b)] = std::string(1, static_cast<char>(b));
  texts[kGet] = "get";
  texts[kWeather] = "_weather";
  texts[kGetWeather] = "get_weather";
  texts[kUnderscore] = "_";
  texts[kWea] = "wea";
  texts[kTher] = "ther";
  texts[kCity] = "city";
  texts[kGetT] = "get_t";
  texts[kIme] = "ime";
  texts[kThinkOpen] = "<think>";
  texts[kThinkClose] = "</think>";
  texts[kToolOpen] = "<tool_call>";
  texts[kToolClose] = "</tool_call>";
  texts[kJBraceQuote] = "{\"";
  texts[kJName] = "name";
  texts[kJQuoteColon] = "\":";
  texts[kJSpaceQuote] = " \"";
  texts[kJQuoteComma] = "\",";
  texts[kJArguments] = "arguments";
  texts[kJSpaceBraceQuote] = " {\"";
  texts[kJSpaceEmpty] = " {}";
  texts[kJQuoteCloseAll] = "\"}}\n";
  texts[kJCloseBoth] = "}}";
  texts[kJCloseBothNl] = "}}\n";
  texts[kJCloseNl] = "}\n";
  texts[kJColonSpaceBrace] = "\": {\"";
  texts[kJSpaceBraceNl] = " {\n";
  texts[kJTwoSpaces] = "  ";
  texts[kJCloseBothNlNl] = "}}\n\n";
  texts[kJSpaceBrace] = " {";
  texts[kJQuoteClose] = "\"}";
  texts[kJSpaceEmptyNl] = " {}\n";
  ChatMarkers m;
  m.think_open = ChatMarker{kThinkOpen, "<think>"};
  m.think_close = ChatMarker{kThinkClose, "</think>"};
  m.tool_call_open = ChatMarker{kToolOpen, "<tool_call>"};
  m.tool_call_close = ChatMarker{kToolClose, "</tool_call>"};
  m.json_calls = true;  // the family's statement: the tokenizer cannot tell
  return GrammarVocab(std::move(texts), m, {kEosText, kEosUser}, kVocab, kEosUser);
}

// The mask, the pointwise rule and the commit agree at this position: the
// same ids either way, and every allowed id leaves the grammar alive.
void require_consistent(const GrammarState& g, const std::string& where) {
  TokenMask m;
  g.mask(&m);
  for (int64_t id = 0; id < kVocab; ++id) {
    const bool pointwise = g.allows(id);
    require(m.allows(id) == pointwise, where + ": mask and allows() disagree on id " + std::to_string(id) +
                                           " in state " + g.state_name());
    if (!m.constrained() || !pointwise) continue;
    GrammarState next = g;
    next.advance(id);
    require(next.active(), where + ": allowed id " + std::to_string(id) + " killed the grammar in state " +
                               g.state_name());
  }
}
void feed_consistent(GrammarState& g, const std::vector<int64_t>& ids, const std::string& where) {
  for (const int64_t id : ids) {
    require_consistent(g, where);
    require(g.allows(id), where + ": id " + std::to_string(id) + " allowed in state " + g.state_name());
    g.advance(id);
  }
  require_consistent(g, where);
}

// What the grammar let through, as the parser reads it.
std::vector<dgpp::text::ToolCallParser::Call> parse_json_calls(const GrammarVocab& vocab,
                                                               const std::vector<int64_t>& ids,
                                                               std::string* content) {
  dgpp::text::ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  dgpp::text::ToolCallParser parser(
      vocab.markers(),
      [&](const std::vector<int64_t>& run) {
        std::string out;
        for (const int64_t id : run) out += vocab.text(id);
        return out;
      },
      dgpp::text::ToolSchemas(), plain);
  std::vector<dgpp::text::ToolCallParser::Event> events;
  for (const int64_t id : ids) parser.feed(id, &events);
  parser.finish(&events);
  std::vector<dgpp::text::ToolCallParser::Call> calls;
  for (const auto& ev : events) {
    if (ev.kind == dgpp::text::ToolCallParser::Event::Kind::kToolCall) calls.push_back(ev.call);
    if (ev.kind == dgpp::text::ToolCallParser::Event::Kind::kContent) *content += ev.text;
  }
  return calls;
}

DGPP_TEST(tool_grammar_json_format_isTheVocabularysStatement) {
  // The same marker ids select the XML grammar or the JSON one by the
  // markers' statement alone; the existing families keep theirs.
  using dgpp::text::ToolFormat;
  const GrammarVocab json = json_vocab();
  require(json.usable() && json.markers().tool_format() == ToolFormat::kQwenJson, "the stated JSON form");
  require(!json.json_call_end_ids().empty(), "the JSON form indexes the ids that can close the arguments and run on");
  const GrammarVocab xml = qwen_vocab();
  require(xml.markers().tool_format() == ToolFormat::kQwenXml && xml.json_call_end_ids().empty(),
          "Qwen3.8's vocabulary stays the XML format");
  require(qwen_vocab(/*compact=*/true).markers().tool_format() == ToolFormat::kQwenXml,
          "MiMo's vocabulary stays the XML format");
  require(fake_vocab().markers().tool_format() == ToolFormat::kGlmMarkers, "the GLM vocabulary is unaffected");
  // The first bytes of a block under each: the JSON grammar refuses the XML
  // form, the XML grammar the JSON one.
  GrammarState j(&json, spec_of(GrammarSpec::Mode::kRequired), false);
  GrammarState x(&xml, spec_of(GrammarSpec::Mode::kRequired), false);
  j.advance(kToolOpen);
  x.advance(kToolOpen);
  require(std::string(j.state_name()) == "j-head" && std::string(x.state_name()) == "q-name",
          std::string("each format's own block: ") + j.state_name() + " / " + x.state_name());
  feed(j, bytes_of("\n"));
  feed(x, bytes_of("\n"));
  require(j.allows('{') && !j.allows('<'), "the JSON block opens with its brace, never <function=");
  require(x.allows('<') && !x.allows('{'), "the XML block opens with <function=, never a brace");
  j.advance('<');  // a disallowed id kills the grammar
  require(!j.active() && std::string(j.state_name()) == "dead", "the XML form is outside the JSON grammar");
}

DGPP_TEST(tool_grammar_json_required_call_walks_the_canonical_shape) {
  const GrammarVocab vocab = json_vocab();
  GrammarState g(&vocab, spec_of(GrammarSpec::Mode::kRequired), /*prompt_opens_thinking=*/false);
  // A call is owed: the opener alone, as under the XML format.
  require(!g.allows('H') && g.allows(kToolOpen) && !g.allows(kEosUser) && !g.allows(kEosText) && !g.allows(kToolClose),
          "top: <tool_call> only, never prose or EOS while owed");
  g.advance(kToolOpen);
  require(std::string(g.state_name()) == "j-head", std::string("the head follows the opener: ") + g.state_name());
  // The head is the template's spelling, byte for byte: a newline, the
  // brace, "name" first.
  require(same(allowed_ids(g), {'\n'}), "the head starts with the template's newline: " + show(allowed_ids(g)));
  feed(g, bytes_of("\n"));
  require(same(allowed_ids(g), {'{', kJBraceQuote}), "then the brace: " + show(allowed_ids(g)));
  feed(g, bytes_of("{\""));
  require(g.allows('n') && g.allows(kJName) && !g.allows('a') && !g.allows(kJArguments),
          "the name comes first, the arguments second");
  feed(g, bytes_of("name\": \""));
  // The name: get_weather / get_time / ping over token texts.
  require(g.allows(kGet) && g.allows(kGetWeather) && g.allows('p') && g.allows(kGetT) && !g.allows('x') &&
              !g.allows('"'),
          "names over token texts");
  feed(g, {kGet, kWeather});
  require(same(allowed_ids(g), {'"', kJQuoteComma}), "the name closes with its quote: " + show(allowed_ids(g)));
  feed(g, bytes_of("\", \"arguments\":"));
  // The head's last byte is one space; the tokens that carry it into the
  // object are offered with it, two spaces or a newline are not.
  require(same(allowed_ids(g), {' ', kJSpaceBraceQuote, kJSpaceEmpty, kJSpaceBraceNl, kJSpaceBrace}),
          "the head's space, alone or with the object's first bytes: " + show(allowed_ids(g)));
  feed(g, bytes_of(" "));
  require(std::string(g.state_name()) == "j-args", std::string("the arguments follow the head: ") + g.state_name());
  // The object opens at once: no whitespace before its brace, no other value.
  require(same(allowed_ids(g), {'{', kJBraceQuote}), "the arguments are an object: " + show(allowed_ids(g)));
  feed(g, bytes_of("{\""));
  // get_weather's keys are closed: city / days.
  require(g.allows('c') && g.allows('d') && g.allows(kCity) && !g.allows('x') && !g.allows('"'), "closed keys city/days");
  feed(g, {kCity});
  feed(g, bytes_of("\": \"Paris\""));
  // The value closed: another key, or the object's end — alone, or with the
  // frame's brace and newline behind it. A newline right after the object's
  // brace is not the tail.
  require(g.allows(',') && g.allows('}') && g.allows(kJCloseBoth) && g.allows(kJCloseBothNl) && !g.allows(kJCloseNl) &&
              !g.allows(kJCloseBothNlNl) && !g.allows(kToolClose) && !g.allows(kEosUser),
          "after a value: a comma, or the close with the frame's tail");
  feed(g, bytes_of(", \""));
  require(g.allows('d') && !g.allows('c') && !g.allows(kCity), "a closed key is offered once");
  feed(g, bytes_of("days\": 3"));
  feed(g, bytes_of("}"));
  // The object is whole: the tail and nothing else, no whitespace before it.
  require(std::string(g.state_name()) == "j-args" && same(allowed_ids(g), {'}', kJCloseNl}),
          "only the frame's brace follows the object: " + show(allowed_ids(g)));
  feed(g, bytes_of("}"));
  require(same(allowed_ids(g), {'\n'}), "then the template's newline: " + show(allowed_ids(g)));
  feed(g, bytes_of("\n"));
  require(std::string(g.state_name()) == "q-close" && same(allowed_ids(g), {kToolClose}),
          "then </tool_call> only: " + show(allowed_ids(g)));
  g.advance(kToolClose);
  // Required (parallel): free text, another call, or the turn's end.
  require(g.allows(kToolOpen) && g.allows(kEosUser) && g.allows('x') && !g.allows(kToolClose),
          "after the call: text, another call or EOS");
  g.advance(kEosUser);
  require(std::string(g.state_name()) == "done", "done after EOS");
}

DGPP_TEST(tool_grammar_json_naturalTokenizationCrossesBothSeams) {
  const GrammarVocab vocab = json_vocab();
  // The Qwen2 tokenizer's own pieces for the template's block: " {\"" spans
  // the head's space and the object's first two bytes; "\"}}\n" closes the
  // string, the object and the frame. Every id is allowed in turn, the
  // mask, allows() and the commit agree at every position, and the parser
  // reads the result as the call.
  const std::vector<int64_t> call = {
      kToolOpen, '\n', kJBraceQuote, kJName, kJQuoteColon, kJSpaceQuote, kGet, kWeather, kJQuoteComma, kJSpaceQuote,
      kJArguments, kJQuoteColon, kJSpaceBraceQuote, kCity, kJQuoteColon, kJSpaceQuote, 'P', 'a', 'r', 'i', 's',
      kJQuoteCloseAll, kToolClose, kEosUser};
  GrammarState g(&vocab, spec_of(GrammarSpec::Mode::kRequired, /*parallel=*/false), false);
  feed_consistent(g, call, "natural tokens");
  require(std::string(g.state_name()) == "done", std::string("the turn ended: ") + g.state_name());
  std::string content;
  auto calls = parse_json_calls(vocab, call, &content);
  require(calls.size() == 1 && calls[0].name == "get_weather" && calls[0].arguments == "{\"city\": \"Paris\"}" &&
              content.empty(),
          "the parser reads what the grammar wrote");
  // No arguments: " {}" ends the head and is the whole object, "}\n" the tail.
  const std::vector<int64_t> empty = {kToolOpen, '\n',       kJBraceQuote, kJName,      kJQuoteColon, kJSpaceQuote,
                                      'p',       'i',        'n',          'g',         kJQuoteComma, kJSpaceQuote,
                                      kJArguments, kJQuoteColon, kJSpaceEmpty, kJCloseNl, kToolClose,   kEosUser};
  GrammarState e(&vocab, spec_of(GrammarSpec::Mode::kRequired, false), false);
  feed_consistent(e, empty, "no arguments");
  content.clear();
  calls = parse_json_calls(vocab, empty, &content);
  require(calls.size() == 1 && calls[0].name == "ping" && calls[0].arguments == "{}", "the empty call parses");
  // A piece across three bytes of the head and two of the object, a
  // separate close, and two calls with the template's newline between them.
  const std::vector<int64_t> two = {
      kToolOpen, '\n', kJBraceQuote, kJName, kJQuoteColon, kJSpaceQuote, kGetT, kIme, kJQuoteComma, kJSpaceQuote,
      kJArguments, kJColonSpaceBrace, 't', 'z', kJQuoteColon, kJSpaceQuote, 'U', 'T', 'C', kJQuoteClose, kJCloseNl,
      kToolClose, '\n',
      kToolOpen, '\n', kJBraceQuote, kJName, kJQuoteColon, kJSpaceQuote, kGetWeather, kJQuoteComma, kJSpaceQuote,
      kJArguments, kJQuoteColon, kJSpaceBrace, '"', 'd', 'a', 'y', 's', kJQuoteColon, ' ', '3', kJCloseBothNl,
      kToolClose, kEosUser};
  GrammarState t(&vocab, spec_of(GrammarSpec::Mode::kRequired), false);
  feed_consistent(t, two, "two calls");
  require(std::string(t.state_name()) == "done", "two calls, then the end");
  content.clear();
  calls = parse_json_calls(vocab, two, &content);
  require(calls.size() == 2 && calls[0].name == "get_time" && calls[0].arguments == "{\"tz\": \"UTC\"}" &&
              calls[1].name == "get_weather" && calls[1].arguments == "{\"days\": 3}" && content == "\n",
          "both calls parse, the newline between them is content");
}

DGPP_TEST(tool_grammar_json_frameIsExactArgumentsAreJson) {
  const GrammarVocab vocab = json_vocab();
  const auto at = [&](const std::string& bytes) {
    GrammarState g(&vocab, spec_of(GrammarSpec::Mode::kRequired), false);
    g.advance(kToolOpen);
    feed(g, bytes_of(bytes));
    return g;
  };
  // The frame has no latitude: the template's bytes or nothing.
  require(!at("").allows('{') && !at("").allows(' '), "no brace before the newline, no other whitespace");
  require(!at("\n").allows('\n') && !at("\n").allows(' '), "one newline only");
  require(!at("\n{").allows(' ') && !at("\n{\"name\"").allows(' '), "no space the template does not write");
  require(!at("\n{\"").allows('a'), "the arguments never come first");
  GrammarState head = at("\n{\"name\": \"ping\", \"arguments\":");
  require(head.allows(' ') && !head.allows(kJTwoSpaces) && !head.allows('\n') && !head.allows('{'),
          "exactly one space before the object");
  // ping declares no parameters at all: its object is empty. An id may
  // carry the head's space and the whole object, but not a newline after
  // it: the frame's brace comes first.
  require(head.allows(kJSpaceEmpty) && !head.allows(kJSpaceBraceQuote), "a closed, empty key set: {} only");
  require(!head.allows(kJSpaceEmptyNl), "an id that runs past the object must run into the tail");
  GrammarState open = at("\n{\"name\": \"ping\", \"arguments\": ");
  require(!open.allows(' ') && !open.allows('\n') && !open.allows('[') && !open.allows('"') && !open.allows('1') &&
              open.allows('{'),
          "an object, at once");
  feed(open, bytes_of("{"));
  require(open.allows('}') && !open.allows('"'), "no key to offer");
  feed(open, bytes_of("}"));
  require(!open.allows(' ') && !open.allows('\n') && !open.allows(',') && open.allows('}'),
          "nothing between the object's brace and the frame's");
  feed(open, bytes_of("}"));
  require(!open.allows(' ') && !open.allows('}') && !open.allows(kToolClose) && open.allows('\n'),
          "the newline before </tool_call>");
  // Inside the arguments the JSON machine's own whitespace applies, as for
  // any JSON-typed value.
  GrammarState loose = at("\n{\"name\": \"get_weather\", \"arguments\": {");
  require(loose.allows(' ') && loose.allows('\n') && loose.allows('"') && loose.allows('}'),
          "whitespace inside the object is the machine's");
  feed(loose, bytes_of("\n  \"city\" :\t\"Oslo\"\n"));
  // The object's own close may carry a newline only as the tail's: "}\n"
  // here would put it between the two braces.
  require(loose.allows('}') && loose.allows(kJCloseBoth) && loose.allows(kJCloseBothNl) && !loose.allows(kJCloseNl) &&
              !loose.allows(kJCloseBothNlNl),
          "the close, with the tail or without, never past it");
  loose.advance(kJCloseBothNl);
  require(std::string(loose.state_name()) == "q-close", "one id closed the object and the frame");
  // A marker or an EOS never rides inside the block, not even as string
  // content.
  GrammarState str = at("\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Par");
  require(str.allows('i') && !str.allows(kThinkOpen) && !str.allows(kThinkClose) && !str.allows(kToolOpen) &&
              !str.allows(kToolClose) && !str.allows(kEosUser) && !str.allows(kEosText),
          "inside a string: text only");
  // A structural piece that would close the object from inside a string does
  // not: it is the string's content (and a raw newline is not JSON there).
  feed(str, bytes_of("is"));
  require(str.allows(kJQuoteCloseAll) && str.allows(kJQuoteClose) && str.allows(kJCloseBoth) && !str.allows(kJCloseBothNl),
          "the quote ends the string; braces without it are content");
  GrammarState content = str;
  content.advance(kJCloseBoth);
  require(std::string(content.state_name()) == "j-args" && content.allows('"') && content.allows(kJQuoteCloseAll),
          "the braces were string content");
  str.advance(kJQuoteCloseAll);
  require(std::string(str.state_name()) == "q-close", "the quote, both braces and the newline in one id");
}

// The arguments from a function definition. Under the JSON format every
// declared property is a JSON text — a string is quoted like any other
// value — so each rides as kJson under its own schema and the whole object
// is one machine; under the other formats the derivation is what it was.
DGPP_TEST(tool_grammar_json_typesEveryDeclaredProperty) {
  const dgpp::minijson::ParseResult def = dgpp::minijson::parse(
      R"({"name":"get_weather","strict":true,"parameters":{"type":"object","properties":{)"
      R"("city":{"type":"string"},"days":{"type":"integer","minimum":1},)"
      R"("unit":{"type":"string","enum":["celsius","fahrenheit"]},)"
      R"("opts":{"type":"object","properties":{"metric":{"type":"boolean"}},"additionalProperties":false}},)"
      R"("required":["city","days"],"additionalProperties":false}})");
  const GrammarTool raw = dgpp::text::grammar_tool_from_function(def.root, nullptr);
  require(raw.args.size() == 4 && raw.args[0].kind == GrammarArg::Kind::kFree &&
              raw.args[1].kind == GrammarArg::Kind::kJson && raw.args[2].kind == GrammarArg::Kind::kText &&
              raw.args[2].texts == std::vector<std::string>{"celsius", "fahrenheit"},
          "the raw-string formats: a string is free text, a string enum its raw texts");
  const GrammarTool xml =
      dgpp::text::grammar_tool_from_function(def.root, nullptr, nullptr, dgpp::text::ToolFormat::kQwenXml);
  require(xml.args == raw.args && xml.keys == raw.keys, "stating the XML format changes nothing");
  const GrammarTool tool =
      dgpp::text::grammar_tool_from_function(def.root, nullptr, nullptr, dgpp::text::ToolFormat::kQwenJson);
  require(tool.strict && tool.constrain_keys && tool.keys == raw.keys && tool.required_keys == raw.required_keys,
          "the key set, the required keys and strict are the format's to share");
  for (const GrammarArg& a : tool.args)
    require(a.kind == GrammarArg::Kind::kJson && !a.schema.empty(), "the JSON format types '" + a.key + "' by its schema");

  const GrammarVocab vocab = json_vocab();
  GrammarSpec spec;
  spec.mode = GrammarSpec::Mode::kNamed;
  spec.named = "get_weather";
  spec.parallel = false;
  spec.tools.push_back(tool);
  GrammarState g(&vocab, spec, false);
  g.advance(kToolOpen);
  feed(g, bytes_of("\n{\"name\": \"get_weather\", \"arguments\":"));
  // Strict: the required keys gate the close, so the empty object is not on
  // offer, with the head's space or after it.
  require(g.allows(kJSpaceBraceQuote) && !g.allows(kJSpaceEmpty), "strict: {} is refused while keys are required");
  feed(g, bytes_of(" {"));
  require(g.allows('"') && !g.allows('}'), "strict: the object cannot close empty");
  feed(g, bytes_of("\"city\": "));
  require(g.allows('"') && !g.allows('1') && !g.allows('{') && !g.allows('t') && !g.allows('n') && !g.allows('['),
          "a string property is a JSON string");
  feed(g, bytes_of("\"Rome\""));
  require(g.allows(',') && !g.allows('}') && !g.allows(kJCloseBoth), "strict: days is still required");
  feed(g, bytes_of(", \"days\": "));
  require(g.allows('3') && !g.allows('0') && !g.allows('"') && !g.allows('-'), "an integer under its minimum");
  feed(g, bytes_of("3"));
  require(g.allows('}') && g.allows(kJCloseBoth) && g.allows(kJCloseBothNl) && g.allows(','),
          "every required key present: the object may close");
  feed(g, bytes_of(", \"unit\": "));
  require(g.allows('"') && !g.allows('c'), "a string enum is quoted here");
  feed(g, bytes_of("\""));
  require(g.allows('c') && g.allows('f') && !g.allows('k') && !g.allows('"'), "one of the enum's texts");
  feed(g, bytes_of("celsius\", \"opts\": {\"metric\": "));
  require(g.allows('t') && g.allows('f') && !g.allows('"') && !g.allows('1'), "a nested object under its own schema");
  feed(g, bytes_of("true}"));
  require(g.allows('}') && !g.allows(','), "every declared key used: only the close");
  feed(g, bytes_of("}}\n"));
  g.advance(kToolClose);
  require(same(allowed_ids(g), {kEosUser}), "named: exactly one call, then the turn ends: " + show(allowed_ids(g)));

  // Not strict: the same definition closes at will, as the other formats'
  // non-strict calls do.
  const dgpp::minijson::ParseResult lax = dgpp::minijson::parse(
      R"({"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},)"
      R"("required":["city"]}})");
  GrammarSpec lax_spec = spec;
  lax_spec.tools = {dgpp::text::grammar_tool_from_function(lax.root, nullptr, nullptr, dgpp::text::ToolFormat::kQwenJson)};
  GrammarState l(&vocab, lax_spec, false);
  l.advance(kToolOpen);
  feed(l, bytes_of("\n{\"name\": \"get_weather\", \"arguments\":"));
  require(l.allows(kJSpaceEmpty), "not strict: the required key is the client's to check");
  feed(l, bytes_of(" {\""));
  require(l.allows('c') && !l.allows('d'), "the declared names still close the key set");

  // A property outside the constrained subset stays any JSON value, with
  // the warning saying so.
  const dgpp::minijson::ParseResult odd = dgpp::minijson::parse(
      R"({"name":"f","parameters":{"type":"object","properties":{"a":{"oneOf":[{"type":"string"}]},"b":{"type":"string"}}}})");
  std::vector<std::string> warnings;
  const GrammarTool o =
      dgpp::text::grammar_tool_from_function(odd.root, &warnings, nullptr, dgpp::text::ToolFormat::kQwenJson);
  require(o.args[0].kind == GrammarArg::Kind::kFree && o.args[1].kind == GrammarArg::Kind::kJson && warnings.size() == 1 &&
              warnings[0].find("any JSON value") != std::string::npos,
          "outside the subset: free, and said so");
  GrammarSpec odd_spec = spec;
  odd_spec.named = "f";
  odd_spec.tools = {o};
  GrammarState f(&vocab, odd_spec, false);
  f.advance(kToolOpen);
  feed(f, bytes_of("\n{\"name\": \"f\", \"arguments\": {\"a\": "));
  require(f.allows('"') && f.allows('1') && f.allows('[') && f.allows('{') && f.allows('t'), "any JSON value");
  feed(f, bytes_of("[1, {\"k\": null}], \"b\": "));
  require(f.allows('"') && !f.allows('1'), "beside a typed one");
}

DGPP_TEST(tool_grammar_json_modes) {
  const GrammarVocab vocab = json_vocab();
  // Named: the one function, an open key set (get_time), exactly one call.
  GrammarState named(&vocab, spec_of(GrammarSpec::Mode::kNamed, false, "get_time"), false);
  require(same(allowed_ids(named), {kToolOpen}), "named: the opener alone: " + show(allowed_ids(named)));
  named.advance(kToolOpen);
  feed(named, bytes_of("\n{\"name\": \""));
  require(!named.allows(kGetWeather) && !named.allows('p') && named.allows(kGetT) && named.allows(kGet),
          "named: only get_time");
  feed(named, {kGetT, kIme});
  feed(named, bytes_of("\", \"arguments\": {\""));
  require(named.allows('t') && named.allows('x') && !named.allows(kToolClose) && !named.allows(kEosUser),
          "an open key set: any key");
  feed(named, bytes_of("tz\": \"UTC\"}}\n"));
  named.advance(kToolClose);
  require(same(allowed_ids(named), {kEosUser}), "named: EOS only: " + show(allowed_ids(named)));
  // Required, parallel calls off: one call, then the end.
  GrammarState single(&vocab, spec_of(GrammarSpec::Mode::kRequired, /*parallel=*/false), false);
  single.advance(kToolOpen);
  feed(single, bytes_of("\n{\"name\": \"ping\", \"arguments\": {}}\n"));
  single.advance(kToolClose);
  require(same(allowed_ids(single), {kEosUser}), "one call when parallel calls are off: " + show(allowed_ids(single)));
  // Auto: prose, a call when the model opens one — well-formed then — and
  // the end at will.
  GrammarState autos(&vocab, spec_of(GrammarSpec::Mode::kAuto), false);
  require(autos.allows('H') && autos.allows(kToolOpen) && autos.allows(kEosUser) && !autos.allows(kToolClose),
          "auto: free text, the opener, or the end");
  feed(autos, bytes_of("Sure."));
  autos.advance(kToolOpen);
  require(std::string(autos.state_name()) == "j-head" && same(allowed_ids(autos), {'\n'}),
          "auto: an opened block is the template's");
  feed(autos, bytes_of("\n{\"name\": \"ping\", \"arguments\": {}}\n"));
  autos.advance(kToolClose);
  require(autos.allows(kToolOpen) && autos.allows(kEosUser) && autos.allows('x'), "auto: more text, another call, or the end");
  // None: the opener never.
  GrammarState none(&vocab, spec_of(GrammarSpec::Mode::kForbidCalls), false);
  require(none.allows('H') && none.allows(kEosUser) && !none.allows(kToolOpen), "none: no block opens");
  // Tools beside response_format: a JSON answer or a call, the call in the
  // JSON call format.
  GrammarSpec both = spec_of(GrammarSpec::Mode::kJsonOrTools);
  GrammarState either(&vocab, both, false);
  require(either.allows('{') && either.allows(kToolOpen) && !either.allows('H'), "a JSON answer or a call");
  either.advance(kToolOpen);
  require(std::string(either.state_name()) == "j-head", std::string("the call's head: ") + either.state_name());
  // A reasoning block the prompt opened runs free first; the call is still owed.
  GrammarState thinks(&vocab, spec_of(GrammarSpec::Mode::kRequired), /*prompt_opens_thinking=*/true);
  require(std::string(thinks.state_name()) == "think" && thinks.allows('x') && !thinks.allows(kEosUser),
          "reasoning first, the turn cannot end");
  thinks.advance(kThinkClose);
  require(same(allowed_ids(thinks), {kToolOpen}), "then the owed call");
}

// The entry as it rides the journal: the untyped arguments are dropped
// (fabric_serve.cpp), a string enum built for a raw-string format may still
// arrive as kText. Every rank composes the same arguments schema from what
// arrives.
DGPP_TEST(tool_grammar_json_sameMasksFromTheJournalsEntry) {
  const GrammarVocab vocab = json_vocab();
  GrammarSpec head = spec_of(GrammarSpec::Mode::kRequired);
  GrammarArg city, days, unit;
  city.key = "city";  // kFree: listed on rank 0, absent on the peers
  days.key = "days";
  days.kind = GrammarArg::Kind::kJson;
  days.schema = R"({"type": "integer"})";
  unit.key = "unit";
  unit.kind = GrammarArg::Kind::kText;
  unit.texts = {"celsius", "fahrenheit"};
  head.tools[0].keys = {"city", "days", "unit"};
  head.tools[0].args = {city, days, unit};
  GrammarSpec peer = head;
  peer.tools[0].args = {days, unit};
  GrammarState a(&vocab, head, false), b(&vocab, peer, false);
  const std::vector<int64_t> ids = [&] {
    std::vector<int64_t> out = {kToolOpen};
    for (const int64_t id : bytes_of("\n{\"name\": \"get_weather\", \"arguments\": {\"city\": [\"x\"], \"days\": 3, "
                                     "\"unit\": \"celsius\"}}\n"))
      out.push_back(id);
    out.push_back(kToolClose);
    return out;
  }();
  for (const int64_t id : ids) {
    require(same(allowed_ids(a), allowed_ids(b)), std::string("the same mask on every rank in state ") + a.state_name());
    require_consistent(a, "journal entry");
    require(a.allows(id), "id " + std::to_string(id) + " allowed in state " + a.state_name());
    a.advance(id);
    b.advance(id);
  }
  require(a.active() && std::string(a.state_name()) == "top" && std::string(b.state_name()) == "top", "both closed the call");
  // The typed entries held: an integer for days, a quoted enum text for unit.
  GrammarState t(&vocab, peer, false);
  t.advance(kToolOpen);
  feed(t, bytes_of("\n{\"name\": \"get_weather\", \"arguments\": {\"days\": "));
  require(t.allows('3') && !t.allows('"'), "the kJson entry types its value");
  feed(t, bytes_of("3, \"unit\": "));
  require(t.allows('"') && !t.allows('c'), "the kText entry's texts are JSON strings here");
  feed(t, bytes_of("\"f"));
  require(t.allows('a') && !t.allows('c'), "one of its texts");
}

}  // namespace

DGPP_TEST(tool_grammar_mimo_compact_calls_empty_text_and_typed_values) {
  const GrammarVocab vocab = qwen_vocab(true);
  for (const std::string& value :
       {std::string(""), std::string("."), std::string("/home/jon/dgpp")}) {
    auto spec = spec_of(GrammarSpec::Mode::kAuto, false);
    GrammarState g(&vocab, spec, false);
    g.advance(kToolOpen);
    require(!g.allows('\n') && g.allows('<'), "MiMo uses compact function tags");
    feed(g, bytes_of("<function=get_weather><parameter=city>" + value + "</parameter>"));
    require(std::string(g.state_name()) == "q-key-or-close", "compact/empty value closes");
    feed(g, bytes_of("</function>"));
    require(same(allowed_ids(g), {kToolClose}), "call closes without a newline");
    g.advance(kToolClose);
    g.advance(kEosUser);
    require(std::string(g.state_name()) == "done", "compact automatic call completes");
  }
  auto spec = spec_of(GrammarSpec::Mode::kNamed, false, "get_weather");
  GrammarArg days;
  days.key = "days";
  days.kind = GrammarArg::Kind::kJson;
  days.schema = R"({"type":"integer"})";
  GrammarArg city;
  city.key = "city";
  city.kind = GrammarArg::Kind::kText;
  city.texts = {"Paris"};
  spec.tools[0].args = {days, city};
  GrammarState g(&vocab, spec, false);
  g.advance(kToolOpen);
  feed(g, bytes_of("<function=get_weather><parameter=days>3</parameter>"));
  feed(g, bytes_of("<parameter=city>Paris</parameter></function>"));
  g.advance(kToolClose);
  g.advance(kEosUser);
  require(std::string(g.state_name()) == "done", "typed compact call completes");
}

DGPP_TEST(tool_grammar_mimo_tokens_can_cross_xml_fields_without_bypassing_masks) {
  const GrammarVocab vocab = qwen_vocab(true);
  GrammarState g(&vocab, spec_of(GrammarSpec::Mode::kAuto), false);
  g.advance(kToolOpen);
  feed(g, bytes_of("<function=get_weather><parameter=city>Paris</parameter"));
  require(g.allows(280), "merged closing delimiter/next tag is legal");
  require(!g.allows(283), "unknown next key cannot bypass its mask");
  g.advance(280);
  feed(g, bytes_of("function>"));
  g.advance(kToolClose);
  require(std::string(g.state_name()) == "top", "cross-token call completed");
  g.advance(kToolOpen);
  require(g.allows(281), "one token may span an entire valid call body");
  g.advance(281);
  g.advance(kToolClose);
  require(std::string(g.state_name()) == "top", "multi-field token completed");
  auto spec = spec_of(GrammarSpec::Mode::kAuto);
  spec.tools[0].required_keys = {"city", "days"};
  spec.tools[0].args = {{"days", GrammarArg::Kind::kJson, R"({"type":"integer"})", {}},
                        {"city", GrammarArg::Kind::kText, "", {"Paris"}}};
  GrammarState typed(&vocab, spec, false);
  typed.advance(kToolOpen);
  feed(typed, bytes_of("<function=get_weather><parameter=days>"));
  require(typed.allows(284), "JSON/enum values and their closing tags can share a token");
  typed.advance(284);
  typed.advance(kToolClose);
  require(std::string(typed.state_name()) == "top", "typed multi-field token completed");
  GrammarState required(&vocab, spec, false);
  required.advance(kToolOpen);
  feed(required, bytes_of("<function=get_weather><parameter=days>3</parameter"));
  require(!required.allows(282), "cross-state token cannot skip a required argument");
}

DGPP_TEST(tool_grammar_mimo_non_strict_required_fields_block_empty_and_partial_calls) {
  const GrammarVocab vocab = qwen_vocab(true);
  for (const bool closed : {true, false}) {
    auto spec = spec_of(GrammarSpec::Mode::kAuto);
    spec.tools[0].strict = false;
    spec.tools[0].required_keys = {"city", "days"};
    spec.tools[0].constrain_keys = closed;
    GrammarState g(&vocab, spec, false);
    g.advance(kToolOpen);
    feed(g, bytes_of("<function=get_weather><"));
    require(!g.allows('/'), "non-strict MiMo call cannot close before required fields");
    feed(g, bytes_of("parameter=city>Paris</parameter><"));
    require(!g.allows('/'), "partial required fields cannot close the call");
    feed(g, bytes_of("parameter=days>3</parameter></function>"));
    g.advance(kToolClose);
    require(std::string(g.state_name()) == "top", "all required keys permit closing");
    // A legitimate no-argument tool remains callable.
    g.advance(kToolOpen);
    feed(g, bytes_of("<function=ping></function>"));
    g.advance(kToolClose);
    require(std::string(g.state_name()) == "top", "no required keys permits an empty call");
  }
}

DGPP_TEST(tool_grammar_mimo_free_fields_keep_constraints_for_every_allowed_token) {
  const GrammarVocab vocab = qwen_vocab(true);
  for (const std::string& prefix : {
           std::string("<function=get_time><parameter="),
           std::string("<function=get_weather><parameter=city>Paris"),
           std::string("<function=get_weather><parameter=city>Paris</parameter"),
       }) {
    GrammarState g(&vocab, spec_of(GrammarSpec::Mode::kAuto), false);
    g.advance(kToolOpen);
    feed(g, bytes_of(prefix));
    // The vocabulary includes zero-width tokens and merged XML delimiters.
    // Every token admitted by the mask must leave the grammar enforceable.
    for (const int64_t id : allowed_ids(g)) {
      GrammarState next = g;
      next.advance(id);
      require(next.active(), "allowed token disabled compact XML constraints: " +
                                 std::to_string(id) + " after " + prefix);
    }
    g.advance(319);  // a zero-width token in a free field
    require(g.active(), "zero-width token preserves the field state");
    require(!g.allows(kToolClose) && !g.allows(kEosUser),
            "a zero-width token cannot permit an unfinished call to close");
  }
}

DGPP_TEST(tool_grammar_mimo_open_keys_reject_duplicates_across_token_boundaries) {
  const GrammarVocab vocab = qwen_vocab(true);
  auto spec = spec_of(GrammarSpec::Mode::kAuto);
  spec.tools[0].constrain_keys = false;
  GrammarState g(&vocab, spec, false);
  g.advance(kToolOpen);
  feed(g, bytes_of("<function=get_weather><parameter=city>Paris</parameter>"));
  feed(g, bytes_of("<parameter=city"));
  require(!g.allows('>') && !g.allows(280) && !g.allows(283),
          "duplicate key cannot close alone or inside a merged token");
  // An open schema still permits new names sharing the used key's prefix.
  feed(g, bytes_of("_code>FR</parameter>"));
  feed(g, bytes_of("<parameter=city_code"));
  require(!g.allows('>'), "undeclared keys also enter the duplicate ledger");
  feed(g, bytes_of("_extra>75</parameter></function>"));
  g.advance(kToolClose);
  require(g.active() && std::string(g.state_name()) == "top",
          "distinct open keys produce a complete call");
  g.advance(kToolOpen);
  feed(g, bytes_of("<function=get_weather><parameter=city>Paris</parameter></function>"));
  g.advance(kToolClose);
  require(g.active(), "the duplicate ledger resets between calls");
}
