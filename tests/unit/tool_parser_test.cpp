// M6 6f: the tool-call / reasoning parser over a fake decoder (host-only,
// always runs). The real tokenizer's round trip — render(assistant
// tool_calls) → encode → parse — lives in glm_chat_template_test; here
// the state machine's contract is pinned on synthetic ids: the reasoning
// split, exact streamed deltas, schema-typed and inferred values, nested
// JSON, several calls per turn, and every malformed shape falling back to
// literal content with no half-parsed call. The same for every later
// format — Qwen's XML and JSON, DeepSeek's DSML, Mistral's brackets,
// MiniMax's invokes — each in its own section below.
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "text/chat_template.hpp"
#include "text/tool_parser.hpp"

namespace {

using dgpp::text::ChatMarker;
using dgpp::text::ChatMarkers;
using dgpp::text::DsmlDialect;
using dgpp::text::ToolCallParser;
using dgpp::text::ToolSchemas;
using Event = ToolCallParser::Event;
using Kind = ToolCallParser::Event::Kind;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// The fake tokenizer: ids below 256 are bytes; the markers are 1001..1008
// and decode to their literal text (they are not special tokens, exactly
// like the real ones); 999 is a special EOS that decodes to nothing.
constexpr int64_t kEos = 999;
constexpr int64_t kThinkOpen = 1001, kThinkClose = 1002, kToolOpen = 1003,
                  kToolClose = 1004, kKeyOpen = 1005, kKeyClose = 1006,
                  kValueOpen = 1007, kValueClose = 1008;
// DeepSeek-V4.1's tag token: SPECIAL in its tokenizer (the service's decode
// skips it), so the fake decode drops it like the EOS.
constexpr int64_t kDsml = 1010;

ChatMarkers fake_markers() {
  ChatMarkers m;
  m.think_open = ChatMarker{kThinkOpen, "<think>"};
  m.think_close = ChatMarker{kThinkClose, "</think>"};
  m.tool_call_open = ChatMarker{kToolOpen, "<tool_call>"};
  m.tool_call_close = ChatMarker{kToolClose, "</tool_call>"};
  m.arg_key_open = ChatMarker{kKeyOpen, "<arg_key>"};
  m.arg_key_close = ChatMarker{kKeyClose, "</arg_key>"};
  m.arg_value_open = ChatMarker{kValueOpen, "<arg_value>"};
  m.arg_value_close = ChatMarker{kValueClose, "</arg_value>"};
  return m;
}

std::string fake_decode(const std::vector<int64_t>& ids) {
  static const std::map<int64_t, std::string> markers = {
      {kThinkOpen, "<think>"},   {kThinkClose, "</think>"},
      {kToolOpen, "<tool_call>"}, {kToolClose, "</tool_call>"},
      {kKeyOpen, "<arg_key>"},   {kKeyClose, "</arg_key>"},
      {kValueOpen, "<arg_value>"}, {kValueClose, "</arg_value>"},
  };
  std::string out;
  for (const int64_t id : ids) {
    if (id == kEos || id == kDsml) continue;
    const auto m = markers.find(id);
    if (m != markers.end())
      out += m->second;
    else if (id >= 0 && id < 256)
      out.push_back(static_cast<char>(id));
  }
  return out;
}

// A token stream from text: bytes, with the marker strings mapped to
// their ids (leftmost-longest, like the real added-token scan).
std::vector<int64_t> ids_of(const std::string& text) {
  static const std::vector<std::pair<std::string, int64_t>> table = {
      {"</tool_call>", kToolClose}, {"<tool_call>", kToolOpen},
      {"</arg_value>", kValueClose}, {"<arg_value>", kValueOpen},
      {"</arg_key>", kKeyClose},   {"<arg_key>", kKeyOpen},
      {"</think>", kThinkClose},   {"<think>", kThinkOpen},
  };
  std::vector<int64_t> out;
  for (size_t i = 0; i < text.size();) {
    bool matched = false;
    for (const auto& [s, id] : table) {
      if (text.compare(i, s.size(), s) == 0) {
        out.push_back(id);
        i += s.size();
        matched = true;
        break;
      }
    }
    if (!matched) out.push_back(static_cast<unsigned char>(text[i++]));
  }
  return out;
}

ToolSchemas weather_schemas() {
  static const std::string tools =
      R"([{"type":"function","function":{"name":"get_weather",)"
      R"("parameters":{"type":"object","properties":{)"
      R"("city":{"type":"string"},"days":{"type":"integer"},)"
      R"("code":{"type":"string"},"opts":{"type":"object"}}}}},)"
      R"({"name":"flat_tool","parameters":{"type":"object",)"
      R"("properties":{"x":{"type":"number"}}}}])";
  static const dgpp::minijson::ParseResult parsed = dgpp::minijson::parse(tools);
  return ToolSchemas(parsed.root);
}

struct Run {
  std::string reasoning, content;
  int reasoning_closed = 0;
  std::vector<ToolCallParser::Call> calls;
  std::vector<Kind> order;
};

Run drive(const std::vector<int64_t>& ids, ToolCallParser::Options opts = {},
          ToolSchemas schemas = weather_schemas()) {
  ToolCallParser parser(fake_markers(), fake_decode, std::move(schemas), opts);
  std::vector<Event> events;
  for (const int64_t id : ids) parser.feed(id, &events);
  parser.finish(&events);
  Run run;
  for (const Event& ev : events) {
    run.order.push_back(ev.kind);
    switch (ev.kind) {
      case Kind::kReasoning: run.reasoning += ev.text; break;
      case Kind::kReasoningClosed: ++run.reasoning_closed; break;
      case Kind::kContent: run.content += ev.text; break;
      case Kind::kToolCall: run.calls.push_back(ev.call); break;
    }
  }
  require(static_cast<int>(run.calls.size()) == parser.calls(),
          "calls() counts the emitted calls");
  return run;
}

// ---- the Qwen3.8 format: only the outer markers are ids; the
// block's "<function=...><parameter=...>" structure is text, parsed when
// the block closes.
ChatMarkers qwen_markers() {
  ChatMarkers m;
  m.think_open = ChatMarker{kThinkOpen, "<think>"};
  m.think_close = ChatMarker{kThinkClose, "</think>"};
  m.tool_call_open = ChatMarker{kToolOpen, "<tool_call>"};
  m.tool_call_close = ChatMarker{kToolClose, "</tool_call>"};
  return m;
}
std::vector<int64_t> qwen_ids_of(const std::string& text) {
  static const std::vector<std::pair<std::string, int64_t>> table = {
      {"</tool_call>", kToolClose}, {"<tool_call>", kToolOpen},
      {"</think>", kThinkClose},   {"<think>", kThinkOpen},
  };
  std::vector<int64_t> out;
  for (size_t i = 0; i < text.size();) {
    bool matched = false;
    for (const auto& [s, id] : table) {
      if (text.compare(i, s.size(), s) == 0) {
        out.push_back(id);
        i += s.size();
        matched = true;
        break;
      }
    }
    if (!matched) out.push_back(static_cast<unsigned char>(text[i++]));
  }
  return out;
}
Run drive_qwen(const std::string& text, ToolCallParser::Options opts = {}) {
  ToolCallParser parser(qwen_markers(), fake_decode, weather_schemas(), opts);
  std::vector<Event> events;
  for (const int64_t id : qwen_ids_of(text)) parser.feed(id, &events);
  parser.finish(&events);
  Run run;
  for (const Event& ev : events) {
    run.order.push_back(ev.kind);
    switch (ev.kind) {
      case Kind::kReasoning: run.reasoning += ev.text; break;
      case Kind::kReasoningClosed: ++run.reasoning_closed; break;
      case Kind::kContent: run.content += ev.text; break;
      case Kind::kToolCall: run.calls.push_back(ev.call); break;
    }
  }
  require(static_cast<int>(run.calls.size()) == parser.calls(), "calls() counts the emitted calls");
  return run;
}

// ---- the DeepSeek-V4.1 DSML format: the tag token is the only id; the
// brackets and tag names are text, the block opens at the tag token after
// a "<" and closes at "</｜DSML｜ calls>".
ChatMarkers dsml_markers(DsmlDialect dialect = DsmlDialect::kV41) {
  ChatMarkers m;
  m.think_open = ChatMarker{kThinkOpen, "<think>"};
  m.think_close = ChatMarker{kThinkClose, "</think>"};
  m.dsml = ChatMarker{kDsml, "｜DSML｜"};
  m.dsml_dialect = dialect;
  return m;
}
std::vector<int64_t> dsml_ids_of(const std::string& text) {
  static const std::vector<std::pair<std::string, int64_t>> table = {
      {"｜DSML｜", kDsml}, {"</think>", kThinkClose}, {"<think>", kThinkOpen},
  };
  std::vector<int64_t> out;
  for (size_t i = 0; i < text.size();) {
    bool matched = false;
    for (const auto& [s, id] : table) {
      if (text.compare(i, s.size(), s) == 0) {
        out.push_back(id);
        i += s.size();
        matched = true;
        break;
      }
    }
    if (!matched) out.push_back(static_cast<unsigned char>(text[i++]));
  }
  return out;
}
Run drive_dsml(const std::string& text, ToolCallParser::Options opts = {}, DsmlDialect dialect = DsmlDialect::kV41) {
  ToolCallParser parser(dsml_markers(dialect), fake_decode, weather_schemas(), opts);
  std::vector<Event> events;
  for (const int64_t id : dsml_ids_of(text)) parser.feed(id, &events);
  parser.finish(&events);
  Run run;
  for (const Event& ev : events) {
    run.order.push_back(ev.kind);
    switch (ev.kind) {
      case Kind::kReasoning: run.reasoning += ev.text; break;
      case Kind::kReasoningClosed: ++run.reasoning_closed; break;
      case Kind::kContent: run.content += ev.text; break;
      case Kind::kToolCall: run.calls.push_back(ev.call); break;
    }
  }
  require(static_cast<int>(run.calls.size()) == parser.calls(), "calls() counts the emitted calls");
  return run;
}

DGPP_TEST(tool_parser_dsml_format_one_call_string_and_json_values) {
  require(dsml_markers().tool_format() == dgpp::text::ToolFormat::kDsml, "the tag token alone is the DSML format");
  require(dsml_markers().tool_calls_available(), "DSML tool calls are available");
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  const Run run = drive_dsml(
      "Sure, let me check.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n"
      "<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n"
      "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n"
      "</｜DSML｜ invoke>\n</｜DSML｜ calls>",
      plain);
  require(run.content == "Sure, let me check.", "the content stops before the block's blank line: '" + run.content + "'");
  require(run.calls.size() == 1, "one call");
  require(run.calls[0].name == "get_weather", "the call's name");
  require(run.calls[0].arguments == "{\"city\": \"Paris\", \"days\": 3}", "typed arguments: " + run.calls[0].arguments);
}

DGPP_TEST(tool_parser_content_token_provenance) {
  ToolCallParser::Options opts;
  opts.start_in_reasoning = false;
  opts.track_tokens = true;
  for (const std::string text : {"if a < b\n\n", "x ｜DSML｜ y",
                                 "ok\n\n<｜DSML｜ calls>broken"}) {
    const auto ids = dsml_ids_of(text);
    ToolCallParser parser(dsml_markers(), fake_decode, weather_schemas(), opts);
    std::vector<Event> events;
    for (const auto id : ids) parser.feed(id, &events);
    parser.finish(&events);
    std::string content, attributed;
    for (const auto& ev : events) {
      if (ev.kind != Kind::kContent) continue;
      content += ev.text;
      size_t end = 0;
      for (const auto& span : ev.tokens) {
        require(span.begin == end && span.end <= ev.text.size(), "complete, ordered byte attribution");
        require(span.token < ids.size() && ids[span.token] != kEos, "source token index");
        const auto decoded = ids[span.token] == kDsml ? std::string("｜DSML｜") : fake_decode({ids[span.token]});
        require(decoded == ev.text.substr(span.begin, span.end - span.begin), "attributed bytes match source");
        attributed += decoded;
        end = span.end;
      }
      require(end == ev.text.size(), "no unattributed content bytes");
    }
    require(content == text && attributed == text, "held and malformed DSML retains token provenance");
  }
}

DGPP_TEST(tool_parser_dsml_format_two_calls_reasoning_and_namespace) {
  const Run run = drive_dsml(
      "think first</think>\n\n<｜DSML｜ calls>\n"
      "<｜DSML｜ invoke name=\"search::lookup\">\n<｜DSML｜ parameter name=\"query\" string=\"true\">a \"quoted\" value</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n"
      "<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">Rome</｜DSML｜ parameter>\n"
      "<｜DSML｜ parameter name=\"flags\" string=\"false\">{\"metric\": true, \"n\": [1, 2]}</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n"
      "</｜DSML｜ calls>");
  require(run.reasoning == "think first" && run.reasoning_closed == 1, "the reasoning split");
  require(run.content.empty(), "no content before the block: '" + run.content + "'");
  require(run.calls.size() == 2, "two calls");
  require(run.calls[0].name == "lookup", "a namespaced call reports its bare name: " + run.calls[0].name);
  require(run.calls[0].arguments == "{\"query\": \"a \\\"quoted\\\" value\"}", "the quoted string value: " + run.calls[0].arguments);
  require(run.calls[1].arguments == "{\"city\": \"Rome\", \"flags\": {\"metric\": true, \"n\": [1, 2]}}",
          "the JSON value normalized: " + run.calls[1].arguments);
}

DGPP_TEST(tool_parser_dsml_format_holds_back_only_a_block_prefix) {
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  // A lone "<" and blank lines that never open a block are content, in full.
  const Run a = drive_dsml("if a < b then\n\nc < d\n\n", plain);
  require(a.content == "if a < b then\n\nc < d\n\n", "held prefixes flush as content: '" + a.content + "'");
  // The tag token without its "<" is literal text (the decode skips the special token; the parser restores it).
  const Run b = drive_dsml("x ｜DSML｜ y", plain);
  require(b.content == "x ｜DSML｜ y", "a stray tag token prints verbatim: '" + b.content + "'");
}

DGPP_TEST(tool_parser_dsml_format_malformed_block_falls_back_to_content) {
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  for (const char* bad : {
           "ok\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"city\">Paris</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
           "ok\n\n<｜DSML｜ calls><｜DSML｜ invoke name=\"get_weather\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
           "ok\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n",
       }) {
    const Run run = drive_dsml(bad, plain);
    require(run.calls.empty(), std::string("no call from a malformed block: ") + bad);
    require(run.content == bad, "the malformed block is literal content: '" + run.content + "'");
  }
  // Text after a closed block is content (the streaming parser completes
  // the block at its closing tag; the reference's whole-completion check
  // would reject the turn, which a client sees as the trailing content).
  const Run tail = drive_dsml(
      "ok\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls> trailing", plain);
  require(tail.calls.size() == 1 && tail.calls[0].arguments == "{}", "the closed block's call stands");
  require(tail.content == "ok trailing", "the text around the block: '" + tail.content + "'");
}

// ---- the DeepSeek-V4 dialect of DSML (encoding_dsv4.py): the same tag
// token, but no space after it and the block named "tool_calls"; no tool
// namespaces. The tokenizer cannot tell the dialects apart — the markers
// state it.
DGPP_TEST(tool_parser_dsml_v4_dialect_one_call_string_and_json_values) {
  require(dsml_markers().dsml_dialect == DsmlDialect::kV41 && dsml_markers().dsml_namespaces(),
          "the default dialect is V4.1's, with namespaces");
  const ChatMarkers v4 = dsml_markers(DsmlDialect::kV4);
  require(v4.tool_format() == dgpp::text::ToolFormat::kDsml && v4.tool_calls_available() && !v4.dsml_namespaces(),
          "the V4 dialect is the DSML format without namespaces");
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  const Run run = drive_dsml(
      "Sure, let me check.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n"
      "<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n"
      "<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>\n"
      "</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
      plain, DsmlDialect::kV4);
  require(run.content == "Sure, let me check.", "the content stops before the block's blank line: '" + run.content + "'");
  require(run.calls.size() == 1, "one call");
  require(run.calls[0].name == "get_weather", "the call's name");
  require(run.calls[0].arguments == "{\"city\": \"Paris\", \"days\": 3}", "typed arguments: " + run.calls[0].arguments);
}

DGPP_TEST(tool_parser_dsml_v4_dialect_calls_reasoning_and_no_namespace) {
  // The reference's own shape (encoding/tests/test_output_1.txt): the
  // reasoning, "</think>", a blank line, the block. A "::" in a name is
  // part of the name here — V4 has no namespaces to strip. An invoke
  // without parameters comes in both forms: the reference RENDERS it with
  // a blank line between its tags (the empty parameter list between the
  // template's two newlines) and its parser reads that and the plain form.
  const Run run = drive_dsml(
      "think first</think>\n\n<｜DSML｜tool_calls>\n"
      "<｜DSML｜invoke name=\"search::lookup\">\n<｜DSML｜parameter name=\"query\" string=\"true\">a \"quoted\" value</｜DSML｜parameter>\n</｜DSML｜invoke>\n"
      "<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Rome</｜DSML｜parameter>\n"
      "<｜DSML｜parameter name=\"flags\" string=\"false\">{\"metric\": true, \"n\": [1, 2]}</｜DSML｜parameter>\n</｜DSML｜invoke>\n"
      "<｜DSML｜invoke name=\"now\">\n\n</｜DSML｜invoke>\n"
      "<｜DSML｜invoke name=\"ping\">\n</｜DSML｜invoke>\n"
      "</｜DSML｜tool_calls>",
      {}, DsmlDialect::kV4);
  require(run.reasoning == "think first" && run.reasoning_closed == 1, "the reasoning split");
  require(run.content.empty(), "no content before the block: '" + run.content + "'");
  require(run.calls.size() == 4, "four calls: " + std::to_string(run.calls.size()));
  require(run.calls[0].name == "search::lookup", "a V4 name is reported as written: " + run.calls[0].name);
  require(run.calls[0].arguments == "{\"query\": \"a \\\"quoted\\\" value\"}", "the quoted string value: " + run.calls[0].arguments);
  require(run.calls[1].arguments == "{\"city\": \"Rome\", \"flags\": {\"metric\": true, \"n\": [1, 2]}}",
          "the JSON value normalized: " + run.calls[1].arguments);
  require(run.calls[2].name == "now" && run.calls[2].arguments == "{}", "the reference's blank-line form of a call without parameters");
  require(run.calls[3].name == "ping" && run.calls[3].arguments == "{}", "the plain form of a call without parameters");
  // The blank line is the header's only slack: two of them, or one before
  // a later tag, is not the reference's shape.
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  for (const char* bad : {
           "ok\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"now\">\n\n\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
           "ok\n\n<｜DSML｜tool_calls>\n\n<｜DSML｜invoke name=\"now\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
           "ok\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"now\">\n</｜DSML｜invoke>\n\n</｜DSML｜tool_calls>",
       }) {
    const Run r = drive_dsml(bad, plain, DsmlDialect::kV4);
    require(r.calls.empty() && r.content == bad, std::string("not the reference's shape: ") + bad);
  }
}

DGPP_TEST(tool_parser_dsml_dialects_do_not_cross) {
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  const std::string v41 =
      "ok\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n"
      "<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
  const std::string v4 =
      "ok\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n"
      "<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
  // Each parser reads its own spelling...
  require(drive_dsml(v41, plain, DsmlDialect::kV41).calls.size() == 1, "V4.1 reads the V4.1 spelling");
  require(drive_dsml(v4, plain, DsmlDialect::kV4).calls.size() == 1, "V4 reads the V4 spelling");
  // ...and the other one's block never closes for it: literal content at
  // the end of the turn, nothing lost, no call.
  const Run a = drive_dsml(v4, plain, DsmlDialect::kV41);
  require(a.calls.empty() && a.content == v4, "the V4 spelling under the V4.1 dialect is content: '" + a.content + "'");
  const Run b = drive_dsml(v41, plain, DsmlDialect::kV4);
  require(b.calls.empty() && b.content == v41, "the V4.1 spelling under the V4 dialect is content: '" + b.content + "'");
}

DGPP_TEST(tool_parser_dsml_v4_dialect_malformed_block_and_held_prefix) {
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  for (const char* bad : {
           "ok\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\">Paris</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
           "ok\n\n<｜DSML｜tool_calls><｜DSML｜invoke name=\"get_weather\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
           "ok\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n"
           "<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n"
           "<｜DSML｜parameter name=\"city\" string=\"true\">Rome</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
           "ok\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n",
       }) {
    const Run run = drive_dsml(bad, plain, DsmlDialect::kV4);
    require(run.calls.empty(), std::string("no call from a malformed block: ") + bad);
    require(run.content == bad, "the malformed block is literal content: '" + run.content + "'");
  }
  const Run held = drive_dsml("if a < b then\n\nc < d\n\n", plain, DsmlDialect::kV4);
  require(held.content == "if a < b then\n\nc < d\n\n", "held prefixes flush as content: '" + held.content + "'");
  const Run stray = drive_dsml("x ｜DSML｜ y", plain, DsmlDialect::kV4);
  require(stray.content == "x ｜DSML｜ y", "a stray tag token prints verbatim: '" + stray.content + "'");
  const Run tail = drive_dsml(
      "ok\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls> trailing", plain,
      DsmlDialect::kV4);
  require(tail.calls.size() == 1 && tail.calls[0].arguments == "{}", "the closed block's call stands");
  require(tail.content == "ok trailing", "the text around the block: '" + tail.content + "'");
}

// The arguments' re-serialization is json.dumps' — floats included: the
// notation follows the decimal exponent as Python's repr does, not the
// shorter spelling (100000.0 is not "1e+05").
DGPP_TEST(tool_parser_json_floats_print_as_python_repr) {
  const auto dumps = [](const std::string& json) {
    const dgpp::minijson::ParseResult parsed = dgpp::minijson::parse(json);
    return dgpp::text::Value::from_minijson(parsed.root).to_json(false);
  };
  for (const auto& [in, want] : std::vector<std::pair<std::string, std::string>>{
           {"0.5", "0.5"}, {"1.0", "1.0"}, {"100000.0", "100000.0"}, {"1e5", "100000.0"}, {"1E+5", "100000.0"},
           {"123456.789", "123456.789"}, {"1e15", "1000000000000000.0"}, {"1e16", "1e+16"}, {"1.5e16", "1.5e+16"},
           {"1e22", "1e+22"}, {"0.0001", "0.0001"}, {"0.00015", "0.00015"}, {"0.00001", "1e-05"}, {"1.5e-7", "1.5e-07"},
           {"-2.50", "-2.5"}, {"-0.0", "-0.0"}, {"0.0", "0.0"}, {"3.141592653589793", "3.141592653589793"},
           {"1.7976931348623157e308", "1.7976931348623157e+308"}, {"5e-324", "5e-324"}, {"12", "12"}, {"-7", "-7"}})
    require(dumps(in) == want, "json.dumps(" + in + ") is " + want + ", got " + dumps(in));
  require(dumps("[0.1, 1e5, {\"a\": 2.0}]") == "[0.1, 100000.0, {\"a\": 2.0}]", "nested floats");
}

DGPP_TEST(tool_parser_qwen_format_markers_and_one_call) {
  require(qwen_markers().tool_format() == dgpp::text::ToolFormat::kQwenXml, "the two-marker set is the Qwen format");
  require(fake_markers().tool_format() == dgpp::text::ToolFormat::kGlmMarkers, "the six-marker set is the GLM format");
  require(qwen_markers().tool_calls_available(), "Qwen tool calls are available");
  // The template's exact shape: typed by the schema (city string, days integer).
  const Run run = drive_qwen(
      "reasoning\n</think>\n\n<tool_call>\n<function=get_weather>\n"
      "<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n"
      "</function>\n</tool_call>");
  require(run.reasoning == "reasoning\n", "reasoning: " + run.reasoning);
  require(run.content == "\n\n", "content before the call: '" + run.content + "'");
  require(run.calls.size() == 1 && run.calls[0].name == "get_weather", "one call to get_weather");
  require(run.calls[0].arguments == "{\"city\": \"Paris\", \"days\": 3}", "arguments: " + run.calls[0].arguments);
}

DGPP_TEST(tool_parser_qwen_format_multiline_nested_and_two_calls) {
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  const Run run = drive_qwen(
      "Let me check both.\n\n<tool_call>\n<function=get_weather>\n"
      "<parameter=code>\nprint(1)\nprint(2)\n\n</parameter>\n"
      "<parameter=opts>\n{\"a\": 1, \"b\": [1, 2]}\n</parameter>\n"
      "<parameter=note>\n\n</parameter>\n"
      "</function>\n</tool_call>\n<tool_call>\n<function=flat_tool>\n"
      "<parameter=x>\n0.5\n</parameter>\n</function>\n</tool_call>", plain);
  require(run.content == "Let me check both.\n\n\n", "content: '" + run.content + "'");
  require(run.calls.size() == 2, "two calls");
  require(run.calls[0].arguments ==
              "{\"code\": \"print(1)\\nprint(2)\\n\", \"opts\": {\"a\": 1, \"b\": [1, 2]}, \"note\": \"\"}",
          "first arguments: " + run.calls[0].arguments);
  require(run.calls[1].name == "flat_tool" && run.calls[1].arguments == "{\"x\": 0.5}",
          "second call: " + run.calls[1].arguments);
}

DGPP_TEST(tool_parser_qwen_format_malformed_blocks_fall_back_to_content) {
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  // No </function>, text after </function>, a missing name, an unterminated
  // parameter, the stream ending inside the block: literal content, no call.
  for (const char* bad : {
           "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</tool_call>",
           "<tool_call>\n<function=get_weather>\n</function>\nextra</tool_call>",
           "<tool_call>\n<function=>\n</function>\n</tool_call>",
           "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis</function>\n</tool_call>",
           "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n",
       }) {
    const Run run = drive_qwen(bad, plain);
    require(run.calls.empty(), std::string("malformed block parsed as a call: ") + bad);
    require(run.content == bad, std::string("literal fallback differs: ") + run.content);
  }
  // A nested opener restarts the block: the first block's text is content,
  // the second parses.
  const Run run = drive_qwen(
      "<tool_call>\n<function=get_weather>\n<tool_call>\n<function=get_weather>\n"
      "<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>", plain);
  require(run.content == "<tool_call>\n<function=get_weather>\n", "the aborted block is content: " + run.content);
  require(run.calls.size() == 1 && run.calls[0].arguments == "{\"city\": \"Oslo\"}", "the restarted block parses");
}

DGPP_TEST(tool_parser_rejects_aRepeatedParameterName) {
  // A repeated <parameter=NAME> is how the XML format writes the same
  // argument twice, and a duplicate key is never a valid object
  // (tool_grammar.hpp). The block is rejected — as the DSML path already
  // rejects one — and its text stands as content instead.
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  const std::string dup =
      "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n"
      "<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>";
  const Run qwen = drive_qwen(dup, plain);
  require(qwen.calls.empty(), "a repeated parameter is not a call");
  require(qwen.content == dup, "the block stands as content: " + qwen.content);
  // Distinct names over the same schema still parse.
  const Run ok = drive_qwen(
      "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n"
      "<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>",
      plain);
  require(ok.calls.size() == 1, "distinct names produce one call");
  require(ok.calls[0].arguments == "{\"city\": \"Rome\", \"days\": 2}",
          "distinct names parse: " + ok.calls[0].arguments);
  // The DSML format's ledger, for parity: a repeat is content there too.
  const std::string dsml_dup =
      "ok\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n"
      "<｜DSML｜ parameter name=\"city\" string=\"true\">Rome</｜DSML｜ parameter>\n"
      "<｜DSML｜ parameter name=\"city\" string=\"true\">Oslo</｜DSML｜ parameter>\n"
      "</｜DSML｜ invoke>\n</｜DSML｜ calls>";
  const Run dsml = drive_dsml(dsml_dup, plain);
  require(dsml.calls.empty() && dsml.content == dsml_dup,
          "a repeated DSML parameter is content: " + dsml.content);
}

DGPP_TEST(tool_parser_qwen_duplicateKeysFallbackAndRecover) {
  const auto parameter = [](const std::string& name, const std::string& value) {
    return "<parameter=" + name + ">\n" + value + "\n</parameter>\n";
  };
  const auto block = [](const std::string& body) {
    return "<tool_call>\n<function=get_weather>\n" + body + "</function>\n</tool_call>";
  };
  const std::string good = block(parameter("city", "Paris"));
  for (const std::string key : {"city", "days", "unknown", "cité"}) {
    for (const std::string second : {"1", "2"}) {
      const std::string bad =
          block(parameter(key, "1") + parameter("between", "x") + parameter(key, second));
      const Run run = drive_qwen("Reason</think>Before" + good + bad + good + "After");
      require(run.reasoning == "Reason" && run.content == "Before" + bad + "After",
              "a duplicate preserves literal content and reasoning: " + key);
      require(run.calls.size() == 2, "valid calls before and after a duplicate survive");
      require(run.calls[0].arguments == "{\"city\": \"Paris\"}" &&
                  run.calls[1].arguments == run.calls[0].arguments,
              "key use is scoped to each call");
    }
  }
}

DGPP_TEST(tool_parser_splitsReasoningFromContentExactly) {
  // The prompt opened <think>; the ids before </think> are reasoning, the
  // rest content, both streamed as exact deltas; a repeated <think> in the
  // reasoning is dropped; </think> yields the structural event once.
  const Run run = drive(ids_of("<think>Let me think.</think>Hello, world."));
  require(run.reasoning == "Let me think.", "reasoning: " + run.reasoning);
  require(run.content == "Hello, world.", "content: " + run.content);
  require(run.reasoning_closed == 1, "one </think> event");
  require(run.calls.empty(), "no calls");
  // Deltas arrive per id, in order: every reasoning delta before the close,
  // every content delta after.
  bool closed = false;
  for (const Kind k : run.order) {
    if (k == Kind::kReasoningClosed) closed = true;
    require(k != Kind::kReasoning || !closed, "reasoning before the close");
    require(k != Kind::kContent || closed, "content after the close");
  }
  // Without the opening think (a prompt that does not end in <think>),
  // everything is content and a stray </think> is literal text.
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  const Run flat = drive(ids_of("A</think>B"), plain);
  require(flat.reasoning.empty() && flat.content == "A</think>B",
          "no split without the opening think: " + flat.content);
}

DGPP_TEST(tool_parser_oneCallTypedByTheSchema) {
  // Schema-typed values: city (string) stays text even though "123" parses,
  // days (integer) parses, code (string) keeps JSON-looking text verbatim,
  // an unknown key infers JSON when it parses and text otherwise.
  const Run run = drive(ids_of(
      "</think><tool_call>get_weather"
      "<arg_key>city</arg_key><arg_value>123</arg_value>"
      "<arg_key>days</arg_key><arg_value>3</arg_value>"
      "<arg_key>code</arg_key><arg_value>{\"a\": 1}</arg_value>"
      "<arg_key>ratio</arg_key><arg_value>0.5</arg_value>"
      "<arg_key>note</arg_key><arg_value>asked twice</arg_value>"
      "<arg_key>flag</arg_key><arg_value>true</arg_value>"
      "</tool_call>"));
  require(run.calls.size() == 1, "one call");
  require(run.calls[0].name == "get_weather", "name: " + run.calls[0].name);
  require(run.calls[0].arguments ==
              "{\"city\": \"123\", \"days\": 3, \"code\": \"{\\\"a\\\": 1}\", "
              "\"ratio\": 0.5, \"note\": \"asked twice\", \"flag\": true}",
          "arguments: " + run.calls[0].arguments);
  require(run.content.empty() && run.reasoning.empty(), "nothing else");
}

DGPP_TEST(tool_parser_preservesNumbersAcrossSchemaBoundaries) {
  for (const std::string value : {
           "0.100000000000000000001", "1e-5000", "1e5000", "9223372036854775809",
           "{\"n\":[0.100000000000000000001],\"s\":\"123\"}"}) {
    const Run glm = drive(ids_of("</think><tool_call>flat_tool<arg_key>x</arg_key><arg_value>" +
                                 value + "</arg_value></tool_call>"));
    const Run qwen = drive_qwen("</think><tool_call>\n<function=flat_tool>\n<parameter=x>\n" +
                                value + "\n</parameter>\n</function>\n</tool_call>");
    const Run dsml = drive_dsml("</think><｜DSML｜ calls>\n<｜DSML｜ invoke name=\"flat_tool\">\n"
                                "<｜DSML｜ parameter name=\"x\" string=\"false\">" + value +
                                "</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>");
    for (const Run& run : {glm, qwen, dsml})
      require(run.calls.size() == 1 && run.calls[0].arguments == "{\"x\": " + value + "}",
              "tool output preserves the numeric value accepted by the grammar: " + value);
  }
}

DGPP_TEST(tool_parser_nestedJsonAndNoArgsAndUnicode) {
  const Run run = drive(ids_of(
      "</think>Calling.<tool_call>get_weather"
      "<arg_key>opts</arg_key><arg_value>{\"a\": [1, 2, {\"b\": null}], \"c\": \"x\"}</arg_value>"
      "</tool_call><tool_call>flat_tool</tool_call>"
      "<tool_call>other<arg_key>k</arg_key><arg_value>[1, \"two\"]</arg_value>"
      "<arg_key>city</arg_key><arg_value>Zürich</arg_value></tool_call>Done."));
  require(run.calls.size() == 3, "three calls, got " +
                                     std::to_string(run.calls.size()));
  require(run.calls[0].arguments ==
              "{\"opts\": {\"a\": [1, 2, {\"b\": null}], \"c\": \"x\"}}",
          "nested JSON re-serialized in json.dumps form: " +
              run.calls[0].arguments);
  require(run.calls[1].name == "flat_tool" && run.calls[1].arguments == "{}",
          "a call without arguments is an empty object");
  // A function outside the schema: JSON when it parses, else text (the
  // UTF-8 passes through unescaped, ensure_ascii=False).
  require(run.calls[2].arguments ==
              "{\"k\": [1, \"two\"], \"city\": \"Zürich\"}",
          "unknown function: " + run.calls[2].arguments);
  require(run.content == "Calling.Done.", "content around the calls: " +
                                              run.content);
  // Order: content, call, call, call, content.
  require(run.order.size() >= 5 && run.order.back() == Kind::kContent,
          "trailing content after the calls");
}

DGPP_TEST(tool_parser_malformedBlocksFallBackToLiteralContent) {
  // A value without a key.
  {
    const Run run = drive(ids_of(
        "</think><tool_call>f<arg_value>x</arg_value></tool_call>tail"));
    require(run.calls.empty(), "no call from a value without a key");
    require(run.content == "<tool_call>f<arg_value>x</arg_value></tool_call>tail",
            "the literal block is content: " + run.content);
  }
  // A key without a value, then the close.
  {
    const Run run = drive(ids_of("</think><tool_call>f<arg_key>k</arg_key></tool_call>"));
    require(run.calls.empty() &&
                run.content == "<tool_call>f<arg_key>k</arg_key></tool_call>",
            "key without value: " + run.content);
  }
  // Text between </arg_key> and <arg_value>.
  {
    const Run run = drive(ids_of(
        "</think><tool_call>f<arg_key>k</arg_key> <arg_value>v</arg_value></tool_call>"));
    require(run.calls.empty(), "stray text inside the block");
    require(run.content ==
                "<tool_call>f<arg_key>k</arg_key> <arg_value>v</arg_value></tool_call>",
            "the block aborts at the stray id and everything is literal "
            "content: " + run.content);
  }
  // Unterminated at the end of the generation (the steps cap).
  {
    const Run run = drive(ids_of(
        "</think>Sure.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Par"));
    require(run.calls.empty(), "no call from an unterminated block");
    require(run.content ==
                "Sure.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Par",
            "unterminated: " + run.content);
  }
  // A nested <tool_call> aborts the open block and starts a fresh one.
  {
    const Run run = drive(ids_of(
        "</think><tool_call>f<arg_key>k</arg_key><tool_call>g<arg_key>x</arg_key>"
        "<arg_value>1</arg_value></tool_call>"));
    require(run.calls.size() == 1 && run.calls[0].name == "g" &&
                run.calls[0].arguments == "{\"x\": 1}",
            "the second block parses");
    require(run.content == "<tool_call>f<arg_key>k</arg_key>",
            "the first block is literal content: " + run.content);
  }
  // Markers in the wrong state outside a block are literal content.
  {
    const Run run = drive(ids_of("</think>a</arg_key>b<arg_value>c"));
    require(run.calls.empty() && run.content == "a</arg_key>b<arg_value>c",
            "stray markers outside a block: " + run.content);
  }
  // An EOS id inside a block decodes to nothing and leaves it open.
  {
    std::vector<int64_t> ids = ids_of("</think><tool_call>f");
    ids.push_back(kEos);
    const Run run = drive(ids);
    require(run.calls.empty() && run.content == "<tool_call>f",
            "EOS inside a block: " + run.content);
  }
}

DGPP_TEST(tool_parser_glm_duplicateKeysFallBackToContent) {
  for (const std::string key : {"city", "extra", "café"}) {
    for (const std::string middle : {"", "<arg_key>days</arg_key><arg_value>3</arg_value>"}) {
      for (const std::string value : {"Rome", "Oslo"}) {
        const std::string block = "<tool_call>get_weather<arg_key>" + key +
                                  "</arg_key><arg_value>Rome</arg_value>" + middle + "<arg_key>" +
                                  key + "</arg_key><arg_value>" + value +
                                  "</arg_value></tool_call>";
        const Run run = drive(ids_of("Think</think>Before" + block + "After"));
        require(run.calls.empty(), "a repeated GLM argument key must not emit a call: " + key);
        require(run.reasoning == "Think", "reasoning survives duplicate-key rejection");
        require(run.content == "Before" + block + "After",
                "the complete malformed block and surrounding content are preserved");
      }
    }
  }
}

DGPP_TEST(tool_parser_glm_duplicateKeysDoNotLeakAcrossCalls) {
  const std::string valid =
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Rome</arg_value>"
      "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>";
  const std::string duplicate =
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Rome</arg_value>"
      "<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>";
  const Run run = drive(ids_of("</think>" + valid + duplicate + valid + valid));
  require(run.content == duplicate, "only the duplicate-key call becomes content");
  require(run.calls.size() == 3, "valid calls before and after rejection still parse");
  for (const auto& call : run.calls) {
    require(call.name == "get_weather" && call.arguments == "{\"city\": \"Rome\", \"days\": 3}",
            "distinct keys retain their types and may recur in a separate call");
  }
}

DGPP_TEST(tool_parser_forcedPrefixSeedsTheBlock) {
  // tool_choice required: the prompt ends in "</think><tool_call>", so the
  // first ids are the name.
  ToolCallParser::Options forced;
  forced.start_in_reasoning = false;
  forced.start_in_tool_call = true;
  forced.forced_prefix_text = "<tool_call>";
  {
    const Run run = drive(ids_of("get_weather<arg_key>city</arg_key>"
                                 "<arg_value>Paris</arg_value></tool_call>"),
                          forced);
    require(run.calls.size() == 1 && run.calls[0].name == "get_weather" &&
                run.calls[0].arguments == "{\"city\": \"Paris\"}",
            "required: the block parses from the seeded state");
  }
  // tool_choice named: the name is in the prompt; the model may extend it
  // and then argues.
  ToolCallParser::Options named = forced;
  named.seeded_name = "get_";
  named.forced_prefix_text = "<tool_call>get_";
  {
    const Run run = drive(ids_of("weather<arg_key>days</arg_key>"
                                 "<arg_value>2</arg_value></tool_call>"),
                          named);
    require(run.calls.size() == 1 && run.calls[0].name == "get_weather" &&
                run.calls[0].arguments == "{\"days\": 2}",
            "named: seeded name + generated tail: " + run.calls[0].name);
  }
  // A forced block that never closes flushes with the prefix text, so the
  // client sees what the model effectively produced.
  {
    const Run run = drive(ids_of("weather<arg_key>days"), named);
    require(run.calls.empty() &&
                run.content == "<tool_call>get_weather<arg_key>days",
            "forced + unterminated: " + run.content);
  }
}

// The opened-thinking test over a rendered prompt: GLM's turn ends in
// <think>, Qwen's in <think> + the bare newline.
DGPP_TEST(tool_parser_promptOpensThinkingSeesTheQwenNewline) {
  ChatMarkers m = qwen_markers();
  constexpr int64_t kNewline = 198;
  require(m.prompt_opens_thinking({7, kThinkOpen}), "a bare <think> tail opens");
  require(!m.prompt_opens_thinking({7, kThinkOpen, kNewline}),
          "without the newline marker the Qwen tail is not recognized");
  m.newline = ChatMarker{kNewline, "\n"};
  require(m.prompt_opens_thinking({7, kThinkOpen, kNewline}), "<think> + newline opens");
  require(m.prompt_opens_thinking({7, kThinkOpen}), "a bare <think> tail still opens");
  require(!m.prompt_opens_thinking({7, kNewline}), "a newline alone does not open");
  require(!m.prompt_opens_thinking({kThinkOpen, kNewline, 7}), "text after the tail closes nothing");
  require(!m.prompt_opens_thinking({}), "an empty prompt");
}

// ---- the MiMo-V2.6 dialect of the XML format (2026-09-22): the same two
// markers, the template writing the call without newlines, the model
// opening its own <think> block (the prompt ends in "assistant\n").
ChatMarkers mimo_markers() {
  ChatMarkers m = qwen_markers();
  m.xml_compact = true;
  return m;
}
Run drive_mimo(const std::string& text, ToolCallParser::Options opts = {}) {
  ToolCallParser parser(mimo_markers(), fake_decode, weather_schemas(), opts);
  std::vector<Event> events;
  for (const int64_t id : qwen_ids_of(text)) parser.feed(id, &events);
  parser.finish(&events);
  Run run;
  for (const Event& ev : events) {
    run.order.push_back(ev.kind);
    switch (ev.kind) {
      case Kind::kReasoning: run.reasoning += ev.text; break;
      case Kind::kReasoningClosed: ++run.reasoning_closed; break;
      case Kind::kContent: run.content += ev.text; break;
      case Kind::kToolCall: run.calls.push_back(ev.call); break;
    }
  }
  require(static_cast<int>(run.calls.size()) == parser.calls(), "calls() counts the emitted calls");
  return run;
}

DGPP_TEST(tool_parser_mimo_compactCallsKeepTheirNewlines) {
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  // The template's exact shape: no newlines around the tags, so a value's
  // trailing newline is the value's (the Qwen dialect would strip it).
  const Run run = drive_mimo(
      "Checking.<tool_call><function=get_weather><parameter=city>Paris</parameter>"
      "<parameter=days>3</parameter></function></tool_call>"
      "<tool_call><function=get_weather><parameter=code>print(1)\nprint(2)\n</parameter>"
      "<parameter=opts>{\"a\": 1, \"b\": [1, 2]}</parameter></function></tool_call>", plain);
  require(run.content == "Checking.", "content: '" + run.content + "'");
  require(run.calls.size() == 2, "two calls");
  require(run.calls[0].arguments == "{\"city\": \"Paris\", \"days\": 3}", "first: " + run.calls[0].arguments);
  require(run.calls[1].arguments == "{\"code\": \"print(1)\\nprint(2)\\n\", \"opts\": {\"a\": 1, \"b\": [1, 2]}}",
          "second keeps the trailing newline: " + run.calls[1].arguments);
  // The newline form (a forced call: the grammar emits it) parses under the
  // compact dialect with the format's own newlines stripped.
  const Run forced = drive_mimo(
      "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
      "<parameter=code>\nprint(1)\nprint(2)\n\n</parameter>\n</function>\n</tool_call>", plain);
  require(forced.calls.size() == 1 && forced.calls[0].arguments == "{\"city\": \"Paris\", \"code\": \"print(1)\\nprint(2)\\n\"}",
          "the newline form under the compact dialect: " + (forced.calls.empty() ? std::string("no call") : forced.calls[0].arguments));
  // The Qwen dialect on the same newline form: identical.
  const Run qwen = drive_qwen(
      "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
      "<parameter=code>\nprint(1)\nprint(2)\n\n</parameter>\n</function>\n</tool_call>", plain);
  require(qwen.calls.size() == 1 && qwen.calls[0].arguments == forced.calls[0].arguments, "the dialects agree on the newline form");
}

DGPP_TEST(tool_parser_mimo_jsonBodyIsTheArguments) {
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  // The template renders a client's pre-serialized arguments as one JSON
  // object inside the function tags (no parameter tags).
  const Run run = drive_mimo("<tool_call><function=get_weather>{\"city\": \"Oslo\", \"days\": 2}</function></tool_call>", plain);
  require(run.calls.size() == 1 && run.calls[0].name == "get_weather", "one call");
  require(run.calls[0].arguments == "{\"city\": \"Oslo\", \"days\": 2}", "the object's members: " + run.calls[0].arguments);
  // Not an object, or malformed: the block falls back to content.
  const Run bad = drive_mimo("<tool_call><function=get_weather>{\"city\": </function></tool_call>", plain);
  require(bad.calls.empty() && bad.content.find("<function=get_weather>") != std::string::npos, "a malformed body is content");
}

DGPP_TEST(tool_parser_mimo_unclosed_and_mixed_dialect_calls_stay_content) {
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  const std::string function =
      "<tool_call>\n<function=read_file>\n"
      "<parameter=path>/tmp/dgpp-tool-probe.txt</parameter>\n</function>\n";
  const std::string mixed =
      function + "</invoke>\n<invoke name=\"read_file\">\n"
                 "<parameter name=\"path\">/tmp/dgpp-tool-probe.txt</parameter>";
  for (const std::string& text : {
           function, mixed, mixed + "</tool_call>",
           std::string("<tool_call><function=read_file><parameter=path>/tmp/dgpp"),
           std::string("<tool_call><function=read_file><parameter=path>a</parameter>"
                       "<parameter=path>b</parameter></function></tool_call>"),
       }) {
    const Run run = drive_mimo(text, plain);
    require(run.calls.empty() && run.content == text,
            "malformed MiMo output must remain literal content: " + text);
  }
}

DGPP_TEST(tool_parser_mimo_modelOpensItsOwnThinking) {
  ChatMarkers m = mimo_markers();
  constexpr int64_t kNewline = 198;
  m.newline = ChatMarker{kNewline, "\n"};
  // The generation prompt ends in "assistant\n": neither marker in the tail.
  require(m.prompt_leaves_thinking_to_model({7, 8, kNewline}), "a bare assistant header leaves the choice");
  require(!m.prompt_leaves_thinking_to_model({7, kThinkOpen}), "an opened block is the prompt's");
  require(!m.prompt_leaves_thinking_to_model({7, kThinkOpen, kThinkClose}), "enable_thinking=false settles it");
  require(!m.prompt_leaves_thinking_to_model({7, kThinkClose, kNewline}), "a closed block then a newline settles it");
  require(!m.prompt_opens_thinking({7, 8, kNewline}), "the bare header does not open");
  require(!ChatMarkers{}.prompt_leaves_thinking_to_model({7}), "no markers, no choice");
  // The parser: <think> as the first id opens the reasoning; </think> closes
  // it; the call follows as content-state structure.
  ToolCallParser::Options opts;
  opts.start_in_reasoning = false;
  opts.model_may_open_thinking = true;
  const Run run = drive_mimo(
      "<think>plan the call</think>Sure.<tool_call><function=get_weather><parameter=city>Rome</parameter></function></tool_call>", opts);
  require(run.reasoning == "plan the call", "reasoning: '" + run.reasoning + "'");
  require(run.reasoning_closed == 1, "one close event");
  require(run.content == "Sure.", "content: '" + run.content + "'");
  require(run.calls.size() == 1 && run.calls[0].arguments == "{\"city\": \"Rome\"}", "the call after the block");
  // Thinking disabled (the prompt ended in <think></think>): a stray opener
  // after content is text.
  ToolCallParser::Options settled;
  settled.start_in_reasoning = false;
  const Run plain = drive_mimo("Hi <think>x</think> there", settled);
  require(plain.reasoning.empty() && plain.content == "Hi <think>x</think> there", "markers are text once settled: '" + plain.content + "'");
  // With the choice left but content already out, an opener is text too.
  const Run late = drive_mimo("Hi <think>x</think>", opts);
  require(late.reasoning.empty() && late.content == "Hi <think>x</think>", "a late opener is content: '" + late.content + "'");
}

// ---- the Qwen3-Next JSON format: the same two markers as the XML format,
// the block one JSON object {"name": NAME, "arguments": {...}} (the Hermes
// form its template asks for). The tokenizer cannot tell the two apart —
// the markers state it — and the prompt opens no reasoning block.
ChatMarkers json_markers() {
  ChatMarkers m = qwen_markers();
  m.json_calls = true;
  return m;
}
Run run_of(const ToolCallParser& parser, const std::vector<Event>& events) {
  Run run;
  for (const Event& ev : events) {
    run.order.push_back(ev.kind);
    switch (ev.kind) {
      case Kind::kReasoning: run.reasoning += ev.text; break;
      case Kind::kReasoningClosed: ++run.reasoning_closed; break;
      case Kind::kContent: run.content += ev.text; break;
      case Kind::kToolCall: run.calls.push_back(ev.call); break;
    }
  }
  require(static_cast<int>(run.calls.size()) == parser.calls(), "calls() counts the emitted calls");
  return run;
}
Run drive_json(const std::string& text) {
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  ToolCallParser parser(json_markers(), fake_decode, weather_schemas(), plain);
  std::vector<Event> events;
  for (const int64_t id : qwen_ids_of(text)) parser.feed(id, &events);
  parser.finish(&events);
  return run_of(parser, events);
}

DGPP_TEST(tool_parser_json_format_isStatedNotInferred) {
  // The two-marker set is the XML format unless the markers say otherwise;
  // the flag changes nothing where the tokenizer already decides.
  using dgpp::text::ToolFormat;
  require(json_markers().tool_format() == ToolFormat::kQwenJson, "the stated JSON form");
  require(json_markers().tool_calls_available(), "JSON tool calls are available");
  require(qwen_markers().tool_format() == ToolFormat::kQwenXml, "Qwen3.8's two markers stay the XML format");
  require(mimo_markers().tool_format() == ToolFormat::kQwenXml, "MiMo's compact dialect stays the XML format");
  require(!ChatMarkers{}.json_calls, "the default is the XML reading");
  ChatMarkers glm = fake_markers();
  glm.json_calls = true;
  require(glm.tool_format() == ToolFormat::kGlmMarkers, "the six GLM markers decide first");
  ChatMarkers dsml = dsml_markers();
  dsml.json_calls = true;
  require(dsml.tool_format() == ToolFormat::kDsml, "the DSML tag token is unaffected");
  ChatMarkers none;
  none.json_calls = true;
  require(none.tool_format() == ToolFormat::kNone && !none.tool_calls_available(),
          "without the two markers there is no format to state");
  // The same block under each reading: a call for the one that owns it,
  // literal content for the other.
  const std::string json = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
  const std::string xml =
      "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  require(drive_json(json).calls.size() == 1 && drive_qwen(xml, plain).calls.size() == 1, "each reads its own form");
  const Run a = drive_json(xml);
  require(a.calls.empty() && a.content == xml, "the XML body under the JSON format is content: '" + a.content + "'");
  const Run b = drive_qwen(json, plain);
  require(b.calls.empty() && b.content == json, "the JSON body under the XML format is content: '" + b.content + "'");
  const Run c = drive_mimo(json, plain);
  require(c.calls.empty() && c.content == json, "and under the compact dialect: '" + c.content + "'");
}

DGPP_TEST(tool_parser_json_format_oneCall) {
  // The template's exact shape. The arguments are already JSON: nothing is
  // typed from the schema (city is declared a string and stays the number
  // the model wrote; the XML format would have made the text a string).
  const Run run = drive_json("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3}}\n</tool_call>");
  require(run.content.empty() && run.reasoning.empty(), "nothing but the call: '" + run.content + "'");
  require(run.calls.size() == 1 && run.calls[0].name == "get_weather", "one call to get_weather");
  require(run.calls[0].arguments == "{\"city\": \"Paris\", \"days\": 3}", "arguments: " + run.calls[0].arguments);
  require(run.order.size() == 1 && run.order[0] == Kind::kToolCall, "one event");
  const Run typed = drive_json("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": 123}}\n</tool_call>");
  require(typed.calls.size() == 1 && typed.calls[0].arguments == "{\"city\": 123}",
          "a value keeps the JSON type it was written with: " + typed.calls[0].arguments);
}

DGPP_TEST(tool_parser_json_format_proseTwoCallsAndTrailingContent) {
  // Text before the first call, the template's "\n" between two blocks, and
  // text after the last: content in order around the two calls.
  const Run run = drive_json(
      "Let me check both.\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n"
      "<tool_call>\n{\"name\": \"flat_tool\", \"arguments\": {\"x\": 0.5}}\n</tool_call>Done.");
  require(run.content == "Let me check both.\n\n\nDone.", "content: '" + run.content + "'");
  require(run.calls.size() == 2, "two calls");
  require(run.calls[0].name == "get_weather" && run.calls[0].arguments == "{\"city\": \"Paris\"}",
          "first: " + run.calls[0].arguments);
  require(run.calls[1].name == "flat_tool" && run.calls[1].arguments == "{\"x\": 0.5}",
          "second: " + run.calls[1].arguments);
  // In arrival order: the prose, a call, the newline, a call, the tail.
  std::vector<Kind> order;
  for (const Kind k : run.order)
    if (order.empty() || order.back() != k || k == Kind::kToolCall) order.push_back(k);
  require(order == std::vector<Kind>{Kind::kContent, Kind::kToolCall, Kind::kContent, Kind::kToolCall, Kind::kContent},
          "content and calls interleave in arrival order");
  // A reasoning block the model was given (a template that opens one) still
  // splits off before the content and the call.
  ToolCallParser parser(json_markers(), fake_decode, weather_schemas(), {});
  std::vector<Event> events;
  for (const int64_t id : qwen_ids_of("plan</think>\n\n<tool_call>\n{\"name\": \"flat_tool\", \"arguments\": {}}\n</tool_call>"))
    parser.feed(id, &events);
  parser.finish(&events);
  const Run thought = run_of(parser, events);
  require(thought.reasoning == "plan" && thought.reasoning_closed == 1 && thought.content == "\n\n",
          "the reasoning split: '" + thought.reasoning + "' / '" + thought.content + "'");
  require(thought.calls.size() == 1 && thought.calls[0].arguments == "{}", "the call after the block");
}

DGPP_TEST(tool_parser_json_format_lenientAboutLayoutStrictAboutMembers) {
  const auto one = [](const std::string& body) {
    const Run run = drive_json("<tool_call>" + body + "</tool_call>");
    require(run.calls.size() == 1 && run.content.empty(), "one call from: " + body + " (content '" + run.content + "')");
    return run.calls[0];
  };
  // No newlines, extra whitespace, a pretty-printed object.
  require(one("{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Oslo\"}}").arguments == "{\"city\": \"Oslo\"}",
          "the compact spelling is re-serialized in json.dumps form");
  require(one(" \n\t{ \"name\" : \"get_weather\" ,\r\n \"arguments\" : { \"city\" : \"Oslo\" } } \n\n").arguments ==
              "{\"city\": \"Oslo\"}",
          "whitespace around and inside");
  require(one("\n{\n  \"name\": \"get_weather\",\n  \"arguments\": {\n    \"city\": \"Oslo\",\n    \"days\": 2\n  }\n}\n")
                  .arguments == "{\"city\": \"Oslo\", \"days\": 2}",
          "a pretty-printed block");
  // The arguments first.
  const ToolCallParser::Call swapped = one("\n{\"arguments\": {\"days\": 2}, \"name\": \"get_weather\"}\n");
  require(swapped.name == "get_weather" && swapped.arguments == "{\"days\": 2}", "either member order");
  // The arguments as a JSON string holding the object (the OpenAI wire form).
  const ToolCallParser::Call wire =
      one("\n{\"name\": \"get_weather\", \"arguments\": \"{\\\"city\\\": \\\"Rome\\\", \\\"days\\\": 1}\"}\n");
  require(wire.arguments == "{\"city\": \"Rome\", \"days\": 1}", "a string holding the object: " + wire.arguments);
  // No arguments, nested values, every scalar, a name the request never declared.
  require(one("\n{\"name\": \"flat_tool\", \"arguments\": {}}\n").arguments == "{}", "an empty object");
  require(one("\n{\"name\": \"other\", \"arguments\": {\"opts\": {\"a\": [1, 2, {\"b\": null}], \"c\": \"x\"}, "
              "\"ok\": true, \"no\": false, \"n\": -1.5e3, \"s\": \"\"}}\n")
                  .arguments ==
              "{\"opts\": {\"a\": [1, 2, {\"b\": null}], \"c\": \"x\"}, \"ok\": true, \"no\": false, \"n\": -1500.0, \"s\": \"\"}",
          "nested values and scalars");
  // Braces, brackets, quotes and the markers' own text inside strings do
  // not end anything.
  const ToolCallParser::Call tricky = one(
      "\n{\"name\": \"get_weather\", \"arguments\": {\"code\": \"if (a) { b[0] = \\\"}\\\"; } // </tool_call\", "
      "\"city\": \"Zürich, \\\\ 東京\"}}\n");
  require(tricky.arguments ==
              "{\"code\": \"if (a) { b[0] = \\\"}\\\"; } // </tool_call\", \"city\": \"Zürich, \\\\ 東京\"}",
          "structure inside strings: " + tricky.arguments);
  // A raw newline inside a string — a model that did not escape it — is
  // read as the newline (the reader's leniency, shared with the other
  // formats' JSON values) and comes back escaped.
  require(one("\n{\"name\": \"get_weather\", \"arguments\": {\"code\": \"a\nb\"}}\n").arguments ==
              "{\"code\": \"a\\nb\"}",
          "an unescaped newline in a string");
  // Numbers the DOM would move keep their spelling, as in the other formats.
  for (const std::string value : {"0.100000000000000000001", "1e-5000", "1e5000", "9223372036854775809",
                                  "{\"n\":[0.100000000000000000001],\"s\":\"123\"}"})
    require(one("\n{\"name\": \"flat_tool\", \"arguments\": {\"x\": " + value + "}}\n").arguments ==
                "{\"x\": " + value + "}",
            "the numeric value is preserved: " + value);
}

DGPP_TEST(tool_parser_json_format_malformedBlocksFallBackToContent) {
  // Anything but one object with a string name and an object of arguments
  // is literal content, in full, and no call.
  for (const char* body : {
           "\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}\n",          // the object never closes
           "\n{\"name\": \"get_weather\", \"arguments\": {\"city\": }}\n",                  // a missing value
           "\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\",}}\n",        // a trailing comma
           "\n{\"name\": \"get_weather\", \"arguments\": {\"city\": Paris}}\n",             // a bare word
           "\n{\"name\": \"get_weather\", \"arguments\": {city: \"Paris\"}}\n",             // an unquoted key
           "\n{'name': 'get_weather', 'arguments': {}}\n",                                  // single quotes
           "\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Par\n",               // an open string
           "\n{\"arguments\": {\"city\": \"Paris\"}}\n",                                    // no name
           "\n{\"name\": 7, \"arguments\": {}}\n",                                          // a name that is not a string
           "\n{\"name\": \"\", \"arguments\": {}}\n",                                       // an empty name
           "\n{\"name\": null, \"arguments\": {}}\n",
           "\n{\"name\": \"get_weather\"}\n",                                               // no arguments
           "\n{\"name\": \"get_weather\", \"arguments\": [\"Paris\"]}\n",                   // arguments not an object
           "\n{\"name\": \"get_weather\", \"arguments\": 3}\n",
           "\n{\"name\": \"get_weather\", \"arguments\": null}\n",
           "\n{\"name\": \"get_weather\", \"arguments\": \"Paris\"}\n",                     // a string that holds no object
           "\n{\"name\": \"get_weather\", \"arguments\": \"\"}\n",
           "\n{\"name\": \"get_weather\", \"arguments\": \"[1]\"}\n",
           "\n{\"name\": \"get_weather\", \"arguments\": {}, \"id\": \"call_1\"}\n",        // a third member
           "\n{\"name\": \"get_weather\", \"name\": \"flat_tool\", \"arguments\": {}}\n",   // a repeated member
           "\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\", \"city\": \"Oslo\"}}\n",  // a repeated argument
           "\n{\"name\": \"get_weather\", \"arguments\": {}} trailing\n",                   // text after the object
           "\n{\"name\": \"get_weather\", \"arguments\": {}}\n{\"name\": \"flat_tool\", \"arguments\": {}}\n",
           "\n[{\"name\": \"get_weather\", \"arguments\": {}}]\n",                          // not an object
           "\nget_weather\n",
           "",
           "\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n",  // the XML form
       }) {
    const std::string text = std::string("Before<tool_call>") + body + "</tool_call>After";
    const Run run = drive_json(text);
    require(run.calls.empty(), std::string("malformed block parsed as a call: ") + body);
    require(run.content == text, std::string("literal fallback differs for ") + body + ": '" + run.content + "'");
  }
  // The stream ending inside a block (the steps cap): content, no call.
  const std::string open = "Sure.<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Par";
  const Run cut = drive_json(open);
  require(cut.calls.empty() && cut.content == open, "unterminated: '" + cut.content + "'");
  // A nested opener restarts the block; a malformed block does not poison
  // the calls around it.
  const std::string good = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}\n</tool_call>";
  const Run nested = drive_json("<tool_call>\n{\"name\": \"get_weather\", " + good);
  require(nested.content == "<tool_call>\n{\"name\": \"get_weather\", ", "the aborted block is content: " + nested.content);
  require(nested.calls.size() == 1 && nested.calls[0].arguments == "{\"city\": \"Oslo\"}", "the restarted block parses");
  const std::string bad = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": [1]}\n</tool_call>";
  const Run mixed = drive_json(good + bad + good);
  require(mixed.calls.size() == 2 && mixed.content == bad, "valid calls before and after a malformed one survive");
  // An EOS id inside a block decodes to nothing and leaves it open.
  std::vector<int64_t> ids = qwen_ids_of("<tool_call>\n{\"name\": \"flat_tool\"");
  ids.push_back(kEos);
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  ToolCallParser parser(json_markers(), fake_decode, weather_schemas(), plain);
  std::vector<Event> events;
  for (const int64_t id : ids) parser.feed(id, &events);
  parser.finish(&events);
  const Run eos = run_of(parser, events);
  require(eos.calls.empty() && eos.content == "<tool_call>\n{\"name\": \"flat_tool\"", "EOS inside a block: " + eos.content);
}

// Streaming, one id at a time. The fake tokenizer's ids are bytes, so every
// byte boundary is a token boundary: inside a key, inside a string's
// escapes, between the bytes of a \uXXXX escape and of a UTF-8 character.
// The contract is the XML format's: content streams as it arrives, a block
// is silent until its closing id, and that id yields the whole call.
DGPP_TEST(tool_parser_json_format_streamsLikeTheXmlFormat) {
  const std::string before = "Checking café… ";
  const std::string body =
      "\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"S\\u00e3o \\\"Paulo\\\" \\ud83d\\ude00 naïve\\n\\t\\\\\", "
      "\"days\": 3}}\n";
  const std::string after = " done";
  const std::vector<int64_t> ids = qwen_ids_of(before + "<tool_call>" + body + "</tool_call>" + after);
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  ToolCallParser parser(json_markers(), fake_decode, weather_schemas(), plain);
  std::string content;
  std::vector<ToolCallParser::Call> calls;
  size_t opened_at = 0, closed_at = 0;
  for (size_t i = 0; i < ids.size(); ++i) {
    std::vector<Event> events;
    const bool was_open = parser.in_tool_call();
    parser.feed(ids[i], &events);
    if (ids[i] == kToolOpen) {
      opened_at = i;
      require(events.empty() && parser.in_tool_call(), "the opening id starts a silent block");
      require(content == before, "the prose streamed before the block opened: '" + content + "'");
    } else if (ids[i] == kToolClose) {
      closed_at = i;
      require(was_open && !parser.in_tool_call(), "the closing id ends the block");
      require(events.size() == 1 && events[0].kind == Kind::kToolCall, "the closing id yields the call, whole");
      calls.push_back(events[0].call);
    } else if (was_open) {
      require(events.empty() && parser.in_tool_call(),
              "nothing is surfaced from inside a block (byte " + std::to_string(i - opened_at) + ")");
    } else {
      for (const Event& ev : events) {
        require(ev.kind == Kind::kContent, "only content outside the block");
        content += ev.text;
      }
    }
  }
  std::vector<Event> rest;
  parser.finish(&rest);
  require(rest.empty(), "nothing is left at the end");
  require(opened_at > 0 && closed_at > opened_at && parser.calls() == 1 && calls.size() == 1, "one block, one call");
  require(content == before + after, "content around the block: '" + content + "'");
  require(calls[0].name == "get_weather", "the name: " + calls[0].name);
  // The escapes decoded and re-serialized (ensure_ascii off): the ã and
  // the surrogate pair are the characters themselves, the rest stay escapes.
  require(calls[0].arguments == "{\"city\": \"São \\\"Paulo\\\" 😀 naïve\\n\\t\\\\\", \"days\": 3}",
          "arguments: " + calls[0].arguments);
  // The steps cap at every byte of the block: whatever was produced comes
  // back as content, byte for byte, and never as a call.
  const std::string block = "<tool_call>" + body;
  for (size_t cut = 0; cut <= body.size(); ++cut) {
    const std::string partial = before + "<tool_call>" + body.substr(0, cut);
    const Run run = drive_json(partial);
    require(run.calls.empty() && run.content == partial,
            "a block cut after " + std::to_string(cut) + " bytes is content: '" + run.content + "'");
  }
  // The tag spelled as ordinary text is not the tag: only the marker ids
  // open and close a block (the XML format's rule), so a split "inside the
  // tag" cannot happen — its bytes are content.
  std::vector<int64_t> spelled;
  for (const char c : block + "</tool_call>") spelled.push_back(static_cast<unsigned char>(c));
  ToolCallParser text_only(json_markers(), fake_decode, weather_schemas(), plain);
  std::vector<Event> events;
  for (const int64_t id : spelled) text_only.feed(id, &events);
  text_only.finish(&events);
  const Run literal = run_of(text_only, events);
  require(literal.calls.empty() && literal.content == block + "</tool_call>", "a spelled-out tag is content");
}

DGPP_TEST(tool_parser_json_format_malformedBlockKeepsTokenProvenance) {
  // Logprobs: a block that becomes visible text attributes every byte to
  // the token that produced it, the markers included.
  const std::string text = "ok <tool_call>\n{\"name\": \"get_weather\", \"arguments\": [1]}\n</tool_call> end";
  const std::vector<int64_t> ids = qwen_ids_of(text);
  ToolCallParser::Options opts;
  opts.start_in_reasoning = false;
  opts.track_tokens = true;
  ToolCallParser parser(json_markers(), fake_decode, weather_schemas(), opts);
  std::vector<Event> events;
  for (const int64_t id : ids) parser.feed(id, &events);
  parser.finish(&events);
  std::string content, attributed;
  for (const Event& ev : events) {
    require(ev.kind == Kind::kContent, "content only");
    content += ev.text;
    size_t end = 0;
    for (const auto& span : ev.tokens) {
      require(span.begin == end && span.end <= ev.text.size(), "complete, ordered byte attribution");
      require(span.token < ids.size(), "source token index");
      const std::string decoded = fake_decode({ids[span.token]});
      require(decoded == ev.text.substr(span.begin, span.end - span.begin), "attributed bytes match source");
      attributed += decoded;
      end = span.end;
    }
    require(end == ev.text.size(), "no unattributed content bytes");
  }
  require(content == text && attributed == text, "the malformed JSON block retains token provenance");
}

DGPP_TEST(tool_parser_schemasReadBothToolForms) {
  const ToolSchemas s = weather_schemas();
  require(s.has("get_weather") && s.has("flat_tool") && !s.has("none"),
          "names from the wrapped and the flat form");
  require(s.type_of("get_weather", "city") == ToolSchemas::Type::kString,
          "string");
  require(s.type_of("get_weather", "days") == ToolSchemas::Type::kJson,
          "integer is JSON-typed");
  require(s.type_of("get_weather", "nope") == ToolSchemas::Type::kUnknown,
          "unknown key");
  require(s.type_of("flat_tool", "x") == ToolSchemas::Type::kJson, "number");
  require(s.type_of("missing", "x") == ToolSchemas::Type::kUnknown,
          "unknown function");
  // Markers missing from a tokenizer disable the features, never guess.
  ChatMarkers none;
  require(!none.reasoning_available() && !none.tool_calls_available(),
          "no markers, no features");
  ToolCallParser::Options opts;
  ToolCallParser parser(none, fake_decode, ToolSchemas(), opts);
  std::vector<Event> events;
  for (const int64_t id : ids_of("</think><tool_call>f</tool_call>x"))
    parser.feed(id, &events);
  std::string content;
  for (const Event& ev : events) {
    require(ev.kind == Kind::kContent, "everything is content");
    content += ev.text;
  }
  require(content == "</think><tool_call>f</tool_call>x",
          "literal without markers: " + content);
}

// ---- the Mistral format (tokenizer v13+ — Mistral-Small-4): the call is
// "[TOOL_CALLS]NAME[ARGS]{json}" with no closing marker; the two brackets
// and the [THINK] / [/THINK] pair are ids — SPECIAL ones, so the fake
// decode drops them like the EOS (the service's decode does) and a
// malformed call's literal text has to restore them.
constexpr int64_t kMToolCalls = 1021, kMArgs = 1022, kMThink = 1023, kMThinkClose = 1024;
// Multi-byte pieces (ids from 2000): a token that carries a closing brace
// and what follows it, a tag split mid-name — what a byte-per-id stream
// cannot show.
std::vector<std::string>& pieces() {
  static std::vector<std::string> p;
  return p;
}
int64_t piece(const std::string& text) {
  pieces().push_back(text);
  return 2000 + static_cast<int64_t>(pieces().size()) - 1;
}
constexpr int64_t kMmToolOpen = 1031, kMmToolClose = 1032;
// The two new formats' decode: fake_decode (bytes, the <think> pair's text,
// nothing for the Mistral brackets), the pieces, and MiniMax's two markers,
// which are not special and print.
std::string port_decode(const std::vector<int64_t>& ids) {
  std::string out;
  for (const int64_t id : ids) {
    if (id >= 2000 && id < 2000 + static_cast<int64_t>(pieces().size()))
      out += pieces()[static_cast<size_t>(id - 2000)];
    else if (id == kMmToolOpen)
      out += "<minimax:tool_call>";
    else if (id == kMmToolClose)
      out += "</minimax:tool_call>";
    else
      out += fake_decode({id});
  }
  return out;
}
std::vector<int64_t> ids_from(const std::vector<std::pair<std::string, int64_t>>& table, const std::string& text) {
  std::vector<int64_t> out;
  for (size_t i = 0; i < text.size();) {
    bool matched = false;
    for (const auto& [s, id] : table) {
      if (text.compare(i, s.size(), s) == 0) {
        out.push_back(id);
        i += s.size();
        matched = true;
        break;
      }
    }
    if (!matched) out.push_back(static_cast<unsigned char>(text[i++]));
  }
  return out;
}
ChatMarkers mistral_markers() {
  ChatMarkers m;
  m.think_open = ChatMarker{kMThink, "[THINK]"};
  m.think_close = ChatMarker{kMThinkClose, "[/THINK]"};
  m.tool_calls = ChatMarker{kMToolCalls, "[TOOL_CALLS]"};
  m.args = ChatMarker{kMArgs, "[ARGS]"};
  m.bracket_think = true;
  return m;
}
std::vector<int64_t> mistral_ids_of(const std::string& text) {
  return ids_from({{"[TOOL_CALLS]", kMToolCalls}, {"[ARGS]", kMArgs}, {"[/THINK]", kMThinkClose},
                   {"[THINK]", kMThink}, {"</s>", kEos}},
                  text);
}
// The service's options for a Mistral prompt: it ends in [/INST], so the
// reply starts as content and the reasoning block is the model's to open.
ToolCallParser::Options mistral_options() {
  ToolCallParser::Options opts;
  opts.start_in_reasoning = false;
  opts.model_may_open_thinking = true;
  return opts;
}
Run drive_ids(const ChatMarkers& markers, const std::vector<int64_t>& ids, ToolCallParser::Options opts,
              ToolSchemas schemas = weather_schemas()) {
  ToolCallParser parser(markers, port_decode, std::move(schemas), opts);
  std::vector<Event> events;
  for (const int64_t id : ids) parser.feed(id, &events);
  parser.finish(&events);
  return run_of(parser, events);
}
Run drive_mistral(const std::string& text) {
  return drive_ids(mistral_markers(), mistral_ids_of(text), mistral_options());
}

DGPP_TEST(tool_parser_mistral_format_markers_and_one_call) {
  const ChatMarkers m = mistral_markers();
  require(m.tool_format() == dgpp::text::ToolFormat::kMistral && m.tool_calls_available(),
          "the two brackets are the Mistral format");
  require(m.reasoning_available(), "[/THINK] splits the reasoning");
  ChatMarkers half = m;
  half.args = ChatMarker{};
  require(half.tool_format() == dgpp::text::ToolFormat::kNone, "[TOOL_CALLS] without [ARGS] is no format");
  // Every older format outranks it: a tokenizer with the <tool_call> pair
  // keeps its reading whatever else it carries.
  ChatMarkers both = qwen_markers();
  both.tool_calls = m.tool_calls;
  both.args = m.args;
  require(both.tool_format() == dgpp::text::ToolFormat::kQwenXml, "the older formats keep their precedence");
  // The prompt ends in [/INST] (any id that is not a think marker): the
  // model may open the block, the prompt opened none.
  require(m.prompt_leaves_thinking_to_model({17, 18, 3, 72, 4}) && !m.prompt_opens_thinking({17, 18, 3, 72, 4}),
          "a Mistral prompt leaves the reasoning to the model");
  require(!m.prompt_leaves_thinking_to_model({3, 72, 4, kMThink}) && m.prompt_opens_thinking({3, 72, 4, kMThink}),
          "a prompt that ends in [THINK] opened it");
  require(!qwen_markers().prompt_leaves_thinking_to_model({1, 2, 3}), "the rule stays off the <think> families");

  const Run run = drive_mistral("Let me check.[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\", \"days\": 3}</s>");
  require(run.content == "Let me check.", "the content before the call: '" + run.content + "'");
  require(run.reasoning.empty() && run.reasoning_closed == 0, "no reasoning");
  require(run.calls.size() == 1 && run.calls[0].name == "get_weather", "one call, its name");
  require(run.calls[0].arguments == "{\"city\": \"Paris\", \"days\": 3}", "the arguments: " + run.calls[0].arguments);
  require(run.order == (std::vector<Kind>{Kind::kContent, Kind::kContent, Kind::kContent, Kind::kContent,
                                          Kind::kContent, Kind::kContent, Kind::kContent, Kind::kContent,
                                          Kind::kContent, Kind::kContent, Kind::kContent, Kind::kContent,
                                          Kind::kContent, Kind::kToolCall}),
          "thirteen content deltas, then the call");
}

DGPP_TEST(tool_parser_mistral_format_severalCallsLayoutAndValues) {
  // Calls follow one another with nothing between them (the template's
  // spelling); the arguments keep their member order, their nesting and
  // their raw UTF-8 whatever their layout, and come out in json.dumps form
  // (1e2 is the float 100.0 there, as in every other format).
  const Run run = drive_mistral(
      "[TOOL_CALLS]get_weather[ARGS]{\"city\":\"São Paulo\",\"days\":1e2}"
      "[TOOL_CALLS]flat_tool[ARGS] {\n  \"x\": 3.0,\n  \"o\": {\"a\": [1, {\"b\": \"}\"}], \"c\": null}\n}"
      "[TOOL_CALLS]ping[ARGS]{}</s>");
  require(run.content.empty(), "no content: '" + run.content + "'");
  require(run.calls.size() == 3, "three calls, got " + std::to_string(run.calls.size()));
  require(run.calls[0].name == "get_weather" &&
              run.calls[0].arguments == "{\"city\": \"São Paulo\", \"days\": 100.0}",
          "compact arguments re-serialized: " + run.calls[0].arguments);
  require(run.calls[1].name == "flat_tool" &&
              run.calls[1].arguments == "{\"x\": 3.0, \"o\": {\"a\": [1, {\"b\": \"}\"}], \"c\": null}}",
          "a brace inside a string closes nothing: " + run.calls[1].arguments);
  require(run.calls[2].name == "ping" && run.calls[2].arguments == "{}", "no arguments");
  // A name the tools do not declare is reported as written (the grammar,
  // not the parser, is what confines the names).
  const Run other = drive_mistral("[TOOL_CALLS] zebra [ARGS]{\"n\": 1}");
  require(other.calls.size() == 1 && other.calls[0].name == "zebra", "the name is trimmed, not checked");
}

DGPP_TEST(tool_parser_mistral_format_reasoningIsTheModelsToOpen) {
  // reasoning_effort "high": [THINK] is the model's first id.
  Run run = drive_mistral("[THINK]17 * 3 = 51.[/THINK]51.</s>");
  require(run.reasoning == "17 * 3 = 51." && run.reasoning_closed == 1 && run.content == "51.",
          "reasoning '" + run.reasoning + "', content '" + run.content + "'");
  // Reasoning, then a call, with and without the closing bracket: a call
  // that opens inside the reasoning closes it.
  run = drive_mistral("[THINK]I should call the tool.[/THINK][TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\"}</s>");
  require(run.reasoning == "I should call the tool." && run.reasoning_closed == 1 && run.content.empty() &&
              run.calls.size() == 1,
          "reasoning, then the call");
  run = drive_mistral("[THINK]I should call the tool.[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\"}</s>");
  require(run.reasoning == "I should call the tool." && run.reasoning_closed == 1 && run.calls.size() == 1 &&
              run.calls[0].arguments == "{\"city\": \"Paris\"}",
          "an unclosed block ends at the call");
  // [THINK] is never content: it opens the block after content too, and a
  // stray [/THINK] prints nothing.
  run = drive_mistral("Hmm.[THINK]second thoughts[/THINK]Done.[/THINK]</s>");
  require(run.content == "Hmm.Done." && run.reasoning == "second thoughts" && run.reasoning_closed == 1,
          "content '" + run.content + "', reasoning '" + run.reasoning + "'");
  // reasoning_effort "none": the model writes no block; everything is content.
  run = drive_mistral("Paris.</s>");
  require(run.content == "Paris." && run.reasoning.empty() && run.reasoning_closed == 0, "plain content");
  // Without the service's statement (a parser that was not told the block
  // is the model's to open) the bracket is skipped like any special token.
  ToolCallParser::Options untold;
  untold.start_in_reasoning = false;
  run = drive_ids(mistral_markers(), mistral_ids_of("[THINK]a[/THINK]b"), untold);
  require(run.content == "ab" && run.reasoning.empty(), "untold: '" + run.content + "'");
}

DGPP_TEST(tool_parser_mistral_format_malformedCallsFallBackToLiteralContent) {
  // The brackets are special tokens — the decode drops them — so the
  // literal text restores them: a client sees what the model wrote.
  struct Case {
    const char* what;
    const char* text;
    const char* content;
  };
  const Case cases[] = {
      {"the stream ends in the name", "a[TOOL_CALLS]get_weather</s>", "a[TOOL_CALLS]get_weather"},
      {"the stream ends in the arguments", "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Par",
       "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Par"},
      {"an inner brace does not close the call", "[TOOL_CALLS]f[ARGS]{\"o\": {\"a\": 1}",
       "[TOOL_CALLS]f[ARGS]{\"o\": {\"a\": 1}"},
      {"no arguments at all", "[TOOL_CALLS]get_weather[ARGS]</s>", "[TOOL_CALLS]get_weather[ARGS]"},
      {"an empty name", "[TOOL_CALLS][ARGS]{\"a\": 1}", "[TOOL_CALLS][ARGS]{\"a\": 1}"},
      {"a whitespace name", "[TOOL_CALLS] \n[ARGS]{\"a\": 1}", "[TOOL_CALLS] \n[ARGS]{\"a\": 1}"},
      {"an array of arguments", "[TOOL_CALLS]f[ARGS][{\"a\": 1}]", "[TOOL_CALLS]f[ARGS][{\"a\": 1}]"},
      {"text before the object", "[TOOL_CALLS]f[ARGS]args: {\"a\": 1}", "[TOOL_CALLS]f[ARGS]args: {\"a\": 1}"},
      {"a repeated key", "[TOOL_CALLS]f[ARGS]{\"a\": 1, \"a\": 2}", "[TOOL_CALLS]f[ARGS]{\"a\": 1, \"a\": 2}"},
      {"a member that is not JSON", "[TOOL_CALLS]f[ARGS]{\"a\": nope}", "[TOOL_CALLS]f[ARGS]{\"a\": nope}"},
      {"the name written with a brace and no [ARGS]", "[TOOL_CALLS]f{\"a\": 1}", "[TOOL_CALLS]f{\"a\": 1}"},
  };
  for (const Case& c : cases) {
    const Run run = drive_mistral(c.text);
    require(run.calls.empty(), std::string(c.what) + ": no call");
    require(run.content == c.content, std::string(c.what) + ": '" + run.content + "'");
  }
  // A second [TOOL_CALLS] before the first call's object closed: the first
  // is literal, the second a call; and a call after a malformed one parses.
  Run run = drive_mistral("[TOOL_CALLS]get_weather[ARGS]{\"city\": [TOOL_CALLS]flat_tool[ARGS]{\"x\": 2}</s>");
  require(run.content == "[TOOL_CALLS]get_weather[ARGS]{\"city\": " && run.calls.size() == 1 &&
              run.calls[0].name == "flat_tool" && run.calls[0].arguments == "{\"x\": 2}",
          "the open call is flushed, the next parsed: '" + run.content + "'");
  run = drive_mistral("[TOOL_CALLS]nameless[TOOL_CALLS]flat_tool[ARGS]{\"x\": 2}");
  require(run.content == "[TOOL_CALLS]nameless" && run.calls.size() == 1, "a name without arguments, then a call");
  // A stray [ARGS] in content is a skipped special token, like the EOS.
  run = drive_mistral("a[ARGS]b</s>");
  require(run.content == "ab" && run.calls.empty(), "a stray [ARGS]: '" + run.content + "'");
}

DGPP_TEST(tool_parser_mistral_format_streamsAndEndsACallAtItsLastBrace) {
  // Content streams id by id; the call goes out with the id that closes its
  // object — not at the EOS, there is no closing marker to wait for — and
  // what follows is content again.
  const std::vector<int64_t> ids = mistral_ids_of(
      "Hi[TOOL_CALLS]flat_tool[ARGS]{\"x\": {\"y\": 1}} then[TOOL_CALLS]flat_tool[ARGS]{\"x\": 2}</s>");
  ToolCallParser parser(mistral_markers(), port_decode, weather_schemas(), mistral_options());
  std::vector<Event> events;
  size_t first_call_at = 0, id_index = 0;
  bool in_call_seen = false;
  for (const int64_t id : ids) {
    const size_t before = events.size();
    parser.feed(id, &events);
    for (size_t k = before; k < events.size(); ++k)
      if (events[k].kind == Kind::kToolCall && first_call_at == 0) first_call_at = id_index;
    in_call_seen = in_call_seen || parser.in_tool_call();
    ++id_index;
  }
  parser.finish(&events);
  const Run run = run_of(parser, events);
  require(in_call_seen, "the parser reports the open call");
  // "Hi" (ids 0-1), [TOOL_CALLS] (2), "flat_tool" (3-11), [ARGS] (12), "{\"x\": {\"y\": 1}}" (13-27).
  require(first_call_at == 27, "the first call is emitted at its closing brace, id " + std::to_string(first_call_at));
  require(run.content == "Hi then" && run.calls.size() == 2, "content around the calls: '" + run.content + "'");
  require(run.calls[0].arguments == "{\"x\": {\"y\": 1}}" && run.calls[1].arguments == "{\"x\": 2}", "both calls");
  // One piece that closes the object and runs on: the call, then the rest
  // of the piece as content. One piece that holds the whole object.
  const std::vector<int64_t> tail = {kMToolCalls, piece("flat"), piece("_tool"), kMArgs, piece("{\"x\""),
                                     piece(": 2"), piece("}\n\nDone"), piece("."), kEos};
  const Run straddle = drive_ids(mistral_markers(), tail, mistral_options());
  require(straddle.calls.size() == 1 && straddle.calls[0].name == "flat_tool" &&
              straddle.calls[0].arguments == "{\"x\": 2}" && straddle.content == "\n\nDone.",
          "a piece across the object's end: '" + straddle.content + "'");
  require(straddle.order == (std::vector<Kind>{Kind::kToolCall, Kind::kContent, Kind::kContent}),
          "the call, the piece's tail, the next piece");
  const Run whole = drive_ids(mistral_markers(), {kMToolCalls, piece("ping"), kMArgs, piece("{}"), kEos},
                              mistral_options());
  require(whole.calls.size() == 1 && whole.calls[0].name == "ping" && whole.calls[0].arguments == "{}" &&
              whole.content.empty(),
          "one piece is the whole object");
}

DGPP_TEST(tool_parser_mistral_format_malformedCallKeepsTokenProvenance) {
  // Content logprobs: every byte of a flushed call maps back to the id that
  // produced it — a restored bracket to its own id.
  const std::string text = "ok [TOOL_CALLS]get_weather[ARGS]{\"city\": 1";
  const std::vector<int64_t> ids = mistral_ids_of(text);
  ToolCallParser::Options opts = mistral_options();
  opts.track_tokens = true;
  ToolCallParser parser(mistral_markers(), port_decode, weather_schemas(), opts);
  std::vector<Event> events;
  for (const int64_t id : ids) parser.feed(id, &events);
  parser.finish(&events);
  std::string content, attributed;
  for (const Event& ev : events) {
    require(ev.kind == Kind::kContent, "content only");
    content += ev.text;
    size_t end = 0;
    for (const auto& span : ev.tokens) {
      require(span.begin == end && span.end <= ev.text.size(), "complete, ordered byte attribution");
      require(span.token < ids.size(), "source token index");
      const int64_t id = ids[span.token];
      const std::string source = id == kMToolCalls ? "[TOOL_CALLS]" : id == kMArgs ? "[ARGS]" : port_decode({id});
      require(source == ev.text.substr(span.begin, span.end - span.begin), "attributed bytes match source");
      attributed += source;
      end = span.end;
    }
    require(end == ev.text.size(), "no unattributed content bytes");
  }
  require(content == text && attributed == text, "the flushed call retains token provenance: '" + content + "'");
}

// ---- the MiniMax-M2 format: the outer two markers are ids (not special:
// they print), the body is "<invoke name=...>" / "<parameter name=...>"
// text, one block holds one or more invokes, and a value is typed from the
// tool's schema the way MiniMax's reference parsers type it.
ChatMarkers minimax_markers() {
  ChatMarkers m;
  m.think_open = ChatMarker{kThinkOpen, "<think>"};
  m.think_close = ChatMarker{kThinkClose, "</think>"};
  m.tool_call_open = ChatMarker{kMmToolOpen, "<minimax:tool_call>"};
  m.tool_call_close = ChatMarker{kMmToolClose, "</minimax:tool_call>"};
  m.minimax_invoke = true;
  return m;
}
std::vector<int64_t> minimax_ids_of(const std::string& text) {
  return ids_from({{"</minimax:tool_call>", kMmToolClose}, {"<minimax:tool_call>", kMmToolOpen},
                   {"</think>", kThinkClose}, {"<think>", kThinkOpen}, {"[e~[", kEos}},
                  text);
}
ToolSchemas minimax_schemas() {
  static const std::string tools =
      R"([{"type":"function","function":{"name":"search","parameters":{"type":"object","properties":{)"
      R"("query":{"type":"string"},"top_k":{"type":"integer"},"ratio":{"type":"number"},)"
      R"("safe":{"type":"boolean"},"filters":{"type":"object"},"tags":{"type":"array"},)"
      R"("note":{"type":["string","null"]},"limit":{"type":["integer","null"]},)"
      R"("mode":{"enum":["fast",3,true]},"level":{"anyOf":[{"type":"integer"},{"type":"string"}]},)"
      R"("when":{"type":"date"},"plain":{"description":"no type"},"shorthand":true,)"
      R"("code":{"type":"string","pattern":"^[A-Z]{2}[0-9]{2}$"},"count":{"type":"int"}}}}}])";
  static const dgpp::minijson::ParseResult parsed = dgpp::minijson::parse(tools);
  return ToolSchemas(parsed.root);
}
// The reply after the generation prompt "]~b]ai\n<think>\n": inside the
// reasoning.
Run drive_minimax(const std::string& text, bool start_in_reasoning = false) {
  ToolCallParser::Options opts;
  opts.start_in_reasoning = start_in_reasoning;
  return drive_ids(minimax_markers(), minimax_ids_of(text), opts, minimax_schemas());
}

DGPP_TEST(tool_parser_minimax_format_markers_and_one_call) {
  const ChatMarkers m = minimax_markers();
  require(m.tool_format() == dgpp::text::ToolFormat::kMinimaxXml && m.tool_calls_available(),
          "MiniMax's two markers are the invoke format");
  ChatMarkers stated = m;
  stated.json_calls = true;
  require(stated.tool_format() == dgpp::text::ToolFormat::kMinimaxXml, "the markers decide, not json_calls");
  require(!m.prompt_leaves_thinking_to_model({1, 2, 3}) && m.prompt_opens_thinking({1, 2, kThinkOpen}),
          "the prompt opens the reasoning; nothing is left to the model");
  // The template's own spelling of a turn, after the prompt's "<think>\n".
  const Run run = drive_minimax(
      "The user wants a search.\n</think>\n\nLet me look.\n<minimax:tool_call>\n<invoke name=\"search\">\n"
      "<parameter name=\"query\">print(\"hi\")\nline 2\n</parameter>\n<parameter name=\"top_k\">5</parameter>\n"
      "<parameter name=\"filters\">{\"a\": 1.5, \"b\": [1, 2, null]}</parameter>\n"
      "<parameter name=\"safe\">true</parameter>\n</invoke>\n</minimax:tool_call>[e~[",
      /*start_in_reasoning=*/true);
  require(run.reasoning == "The user wants a search.\n" && run.reasoning_closed == 1, "the reasoning: '" + run.reasoning + "'");
  require(run.content == "\n\nLet me look.\n", "the content around the block: '" + run.content + "'");
  require(run.calls.size() == 1 && run.calls[0].name == "search", "one call");
  require(run.calls[0].arguments ==
              "{\"query\": \"print(\\\"hi\\\")\\nline 2\\n\", \"top_k\": 5, "
              "\"filters\": {\"a\": 1.5, \"b\": [1, 2, null]}, \"safe\": true}",
          "a string keeps its text, newlines included; the rest is typed: " + run.calls[0].arguments);
}

DGPP_TEST(tool_parser_minimax_format_severalInvokesAndBlocks) {
  // One block, two invokes: two calls, in order. A second block, single
  // quotes around the names, no newlines between the tags, an invoke
  // without parameters: all calls.
  const Run run = drive_minimax(
      "<minimax:tool_call>\n<invoke name=\"search\">\n<parameter name=\"query\">a</parameter>\n</invoke>\n"
      "<invoke name=\"search\">\n<parameter name=\"query\">b</parameter>\n</invoke>\n</minimax:tool_call>"
      " and <minimax:tool_call><invoke name='ping'></invoke><invoke name='search'>"
      "<parameter name='top_k'>7</parameter></invoke></minimax:tool_call>[e~[");
  require(run.calls.size() == 4, "four calls, got " + std::to_string(run.calls.size()));
  require(run.calls[0].arguments == "{\"query\": \"a\"}" && run.calls[1].arguments == "{\"query\": \"b\"}",
          "the first block's two invokes");
  require(run.calls[2].name == "ping" && run.calls[2].arguments == "{}", "an invoke without parameters");
  require(run.calls[3].name == "search" && run.calls[3].arguments == "{\"top_k\": 7}", "single quotes");
  require(run.content == " and ", "the text between the blocks: '" + run.content + "'");
  require(run.order == (std::vector<Kind>{Kind::kToolCall, Kind::kToolCall, Kind::kContent, Kind::kContent,
                                          Kind::kContent, Kind::kContent, Kind::kContent, Kind::kToolCall,
                                          Kind::kToolCall}),
          "a block's calls arrive together, in arrival order");
  // A call that opens inside the reasoning closes it.
  const Run unclosed = drive_minimax(
      "Search now.\n<minimax:tool_call>\n<invoke name=\"ping\">\n</invoke>\n</minimax:tool_call>[e~[", true);
  require(unclosed.reasoning == "Search now.\n" && unclosed.reasoning_closed == 1 && unclosed.calls.size() == 1 &&
              unclosed.content.empty(),
          "the reasoning ends at the call");
}

DGPP_TEST(tool_parser_minimax_format_valuesAreTypedFromTheSchema) {
  const ToolSchemas s = minimax_schemas();
  using B = ToolSchemas;
  require(s.type_bits("search", "query") == (B::kStringBit | B::kDeclaredBit), "string");
  require(s.type_bits("search", "note") == (B::kStringBit | B::kNullBit | B::kDeclaredBit), "a type list");
  require(s.type_bits("search", "mode") == (B::kStringBit | B::kIntegerBit | B::kBooleanBit | B::kDeclaredBit),
          "the enum values' types");
  require(s.type_bits("search", "level") == (B::kStringBit | B::kIntegerBit | B::kDeclaredBit), "the anyOf alternatives'");
  require(s.type_bits("search", "when") == B::kDeclaredBit, "a type name outside JSON Schema: declared, no type");
  require(s.type_bits("search", "plain") == (B::kStringBit | B::kDeclaredBit) &&
              s.type_bits("search", "shorthand") == (B::kStringBit | B::kDeclaredBit),
          "a schema that names no type is a string's");
  require(s.type_bits("search", "count") == (B::kIntegerBit | B::kDeclaredBit), "vLLM's aliases");
  require(s.type_bits("search", "nope") == 0 && s.type_bits("none", "query") == 0, "undeclared: 0");

  // text -> JSON, by the declared types in the reference order (null,
  // integer, number, boolean, object / array, string), else JSON-or-text.
  struct Case {
    const char* key;
    const char* text;
    const char* json;
  };
  const Case cases[] = {
      {"query", "Paris", "\"Paris\""},
      {"query", "123", "\"123\""},            // a string stays one
      {"query", "null", "\"null\""},          // null is not among its types
      {"query", " padded \n", "\" padded \\n\""},  // verbatim
      {"query", "{\"a\": 1}", "\"{\\\"a\\\": 1}\""},
      {"top_k", "5", "5"},
      {"top_k", " +007\n", "7"},
      {"top_k", "-0", "0"},
      {"top_k", "5.5", "5.5"},                // not an integer: the JSON it parses as
      {"top_k", "five", "\"five\""},          // nor JSON: the text
      {"top_k", "null", "null"},              // the fallback's JSON
      {"ratio", "0.25", "0.25"},
      {"ratio", "3.0", "3.0"},                // a float stays one (vLLM would say 3)
      {"ratio", "1e-3", "0.001"},             // json.dumps form, like every JSON value here
      {"ratio", "+2", "2"},
      {"ratio", "fast", "\"fast\""},
      {"safe", "true", "true"},
      {"safe", "False", "false"},
      {"safe", "1", "true"},
      {"safe", "0", "false"},
      {"safe", "yes", "\"yes\""},             // vLLM's set, not SGLang's yes/on
      {"filters", "{\"a\": {\"b\": [1, 2]}}", "{\"a\": {\"b\": [1, 2]}}"},
      {"filters", "{\"a\":1}", "{\"a\": 1}"},
      {"filters", "{broken", "\"{broken\""},
      {"tags", "[\"x\", \"y\"]", "[\"x\", \"y\"]"},
      {"tags", "x, y", "\"x, y\""},
      {"note", "null", "null"},
      {"note", "NULL", "null"},
      {"note", "none", "\"none\""},           // a nullable string is not always null
      {"note", "", "\"\""},
      {"limit", "10", "10"},
      {"limit", "null", "null"},
      {"limit", "ten", "\"ten\""},
      {"mode", "fast", "\"fast\""},
      {"mode", "3", "3"},
      {"mode", "true", "true"},
      {"level", "4", "4"},
      {"level", "high", "\"high\""},
      {"when", "2026-10-04", "\"2026-10-04\""},  // an unknown type: JSON if it parses
      {"when", "42", "42"},
      {"plain", "42", "\"42\""},
      {"shorthand", "42", "\"42\""},
      {"count", "42", "42"},
      {"code", "AB12", "\"AB12\""},
      {"code", "\"AB12\"", "\"AB12\""},       // the grammar's spelling of a constrained string
      {"undeclared", "42", "\"42\""},         // not in the schema: the text
      {"undeclared", "{\"a\": 1}", "\"{\\\"a\\\": 1}\""},
  };
  for (const Case& c : cases) {
    const Run run = drive_minimax(std::string("<minimax:tool_call>\n<invoke name=\"search\">\n<parameter name=\"") +
                                  c.key + "\">" + c.text + "</parameter>\n</invoke>\n</minimax:tool_call>");
    require(run.calls.size() == 1, std::string(c.key) + " '" + c.text + "': one call");
    const std::string want = std::string("{\"") + c.key + "\": " + c.json + "}";
    require(run.calls[0].arguments == want,
            std::string(c.key) + " '" + c.text + "': " + run.calls[0].arguments + " (want " + want + ")");
  }
  // A function the request does not declare: every value is its text.
  const Run unknown = drive_minimax(
      "<minimax:tool_call>\n<invoke name=\"zebra\">\n<parameter name=\"n\">3</parameter>\n</invoke>\n</minimax:tool_call>");
  require(unknown.calls.size() == 1 && unknown.calls[0].name == "zebra" && unknown.calls[0].arguments == "{\"n\": \"3\"}",
          "an undeclared function: " + (unknown.calls.empty() ? std::string("no call") : unknown.calls[0].arguments));
}

DGPP_TEST(tool_parser_minimax_format_malformedBlocksFallBackToContent) {
  // The markers are not special: the decode prints them, so the literal
  // block is exactly what the model wrote.
  const char* blocks[] = {
      "<minimax:tool_call>\n</minimax:tool_call>",                                             // no invoke
      "<minimax:tool_call>just text</minimax:tool_call>",
      "<minimax:tool_call>\n<invoke name=\"search\">\n<parameter name=\"query\">a\n</invoke>\n</minimax:tool_call>",
      "<minimax:tool_call>\n<invoke name=\"search\">\n<parameter name=\"query\">a</parameter>\n</minimax:tool_call>",
      ("<minimax:tool_call>\n<invoke name=\"search\">\n<parameter name=\"query\">a</parameter>\n"
       "<parameter name=\"query\">b</parameter>\n</invoke>\n</minimax:tool_call>"),            // a repeated name
      "<minimax:tool_call>\n<invoke name=\"\">\n</invoke>\n</minimax:tool_call>",              // an empty name
      "<minimax:tool_call>\n<invoke name=search>\n</invoke>\n</minimax:tool_call>",            // no quotes
      "<minimax:tool_call>\n<invoke name=\"search\">\n</invoke>\ntrailing\n</minimax:tool_call>",
      "<minimax:tool_call>\n<invoke name=\"search\">\nstray\n</invoke>\n</minimax:tool_call>",
      "<minimax:tool_call>\n<function=search>\n</function>\n</minimax:tool_call>",             // the Qwen XML body
      "<minimax:tool_call>\n{\"name\": \"search\", \"arguments\": {}}\n</minimax:tool_call>",    // the JSON body
  };
  for (const char* block : blocks) {
    const Run run = drive_minimax(std::string("x ") + block + " y");
    require(run.calls.empty(), std::string("no call from: ") + block);
    require(run.content == std::string("x ") + block + " y", std::string("literal content: ") + run.content);
  }
  // The stream ends inside a block; a second opener before the first block
  // closed flushes it and starts over; a stray closer is content.
  Run run = drive_minimax("a<minimax:tool_call>\n<invoke name=\"search\">\n<parameter name=\"query\">Par");
  require(run.calls.empty() && run.content == "a<minimax:tool_call>\n<invoke name=\"search\">\n<parameter name=\"query\">Par",
          "unterminated: '" + run.content + "'");
  run = drive_minimax("<minimax:tool_call>\n<invoke name=\"search\">\n<minimax:tool_call>\n<invoke name=\"ping\">\n"
                      "</invoke>\n</minimax:tool_call></minimax:tool_call>");
  require(run.calls.size() == 1 && run.calls[0].name == "ping" &&
              run.content == "<minimax:tool_call>\n<invoke name=\"search\">\n</minimax:tool_call>",
          "a nested opener restarts: '" + run.content + "'");
  // The same body between the Qwen markers is not this format's (and the
  // reverse): each tokenizer reads its own.
  const Run qwen = drive_qwen("<tool_call>\n<invoke name=\"search\">\n</invoke>\n</tool_call>");
  require(qwen.calls.empty(), "the XML format does not read an invoke");
}

DGPP_TEST(tool_parser_minimax_format_streamsAndSplitsAnywhere) {
  // Reasoning and content stream as they arrive; a block's ids buffer until
  // its closing marker, then its calls go out together. Byte-per-id is every
  // possible split of the block's text; the pieces below add tokens that
  // run across the tags' seams.
  const std::vector<int64_t> ids = minimax_ids_of(
      "think</think>Hi<minimax:tool_call>\n<invoke name=\"search\">\n<parameter name=\"top_k\">5</parameter>\n"
      "</invoke>\n</minimax:tool_call>!");
  ToolCallParser::Options opts;
  ToolCallParser parser(minimax_markers(), port_decode, minimax_schemas(), opts);
  std::vector<Event> events;
  std::vector<size_t> at;  // the id index each event arrived on
  for (size_t i = 0; i < ids.size(); ++i) {
    const size_t before = events.size();
    parser.feed(ids[i], &events);
    for (size_t k = before; k < events.size(); ++k) at.push_back(i);
    if (ids[i] == kMmToolOpen) require(parser.in_tool_call(), "the opener opens the block");
  }
  parser.finish(&events);
  const Run run = run_of(parser, events);
  require(run.reasoning == "think" && run.content == "Hi!" && run.calls.size() == 1 &&
              run.calls[0].arguments == "{\"top_k\": 5}",
          "the turn: '" + run.content + "'");
  // 5 reasoning deltas (ids 0-4), the close (5), 2 content deltas (6, 7),
  // nothing while the block is open, the call on its closing marker, "!".
  require(events.size() == 10 && at[7] == 7 && events[8].kind == Kind::kToolCall && at[8] == ids.size() - 2 &&
              events[9].kind == Kind::kContent,
          "the call arrives on the closing marker, events " + std::to_string(events.size()));
  const std::vector<int64_t> split = {
      piece("Hi"), kMmToolOpen, piece("\n<inv"), piece("oke name=\"sea"), piece("rch\">\n<parameter"),
      piece(" name=\"query\">S"), piece("\xC3"), piece("\xA3o Pau"), piece("lo</param"), piece("eter>\n<parameter name=\"top_k\">"),
      piece("1"), piece("2</parameter>\n</invoke>"), piece("\n"), kMmToolClose, kEos};
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  const Run pieces_run = drive_ids(minimax_markers(), split, plain, minimax_schemas());
  require(pieces_run.content == "Hi" && pieces_run.calls.size() == 1 &&
              pieces_run.calls[0].arguments == "{\"query\": \"São Paulo\", \"top_k\": 12}",
          "tokens across the tags' seams and a UTF-8 sequence: " +
              (pieces_run.calls.empty() ? pieces_run.content : pieces_run.calls[0].arguments));
}

// ---- the chat-template constructs the two ports added (Mistral-Small-4's
// and MiniMax-M2.7's templates need them; the differential goldens of the
// host gates pin whole renders — these pin each construct without a
// checkpoint, against what jinja2 prints).
std::string render_with(const std::string& source, const std::string& globals_json) {
  const dgpp::minijson::ParseResult parsed = dgpp::minijson::parse(globals_json);
  return dgpp::text::ChatTemplate::compile(source).render(dgpp::text::Value::from_minijson(parsed.root));
}
std::string render_error(const std::string& source, const std::string& globals_json) {
  try {
    (void)render_with(source, globals_json);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "(rendered)";
}

DGPP_TEST(chat_template_constructsOfTheMistralAndMiniMaxTemplates) {
  struct Case {
    const char* source;
    const char* globals;
    const char* want;
  };
  const Case cases[] = {
      // Dict literals, list concatenation, dict.get, join, list.
      {"{{ ([1] + [2, 'a'])|tojson }}", "{}", "[1, 2, \"a\"]"},
      {"{{ {'a': 1, 'b': [1, 2]}|tojson }}", "{}", "{\"a\": 1, \"b\": [1, 2]}"},
      {"{{ {'a': {'b': 1}}|tojson }}", "{}", "{\"a\": {\"b\": 1}}"},
      {"{{ {}|tojson }}|{{ {'a': 1, 'a': 2,}|tojson }}", "{}", "{}|{\"a\": 2}"},
      {"{% set ns = namespace(l=[]) %}{% for m in ms + [{'role': 'end'}] %}"
       "{% set ns.l = ns.l + [m['role']] %}{% endfor %}{{ ns.l|join('/') }}",
       "{\"ms\": [{\"role\": \"user\"}, {\"role\": \"tool\"}]}", "user/tool/end"},
      {"{{ d.get('a') }}|{{ d.get('z') }}|{{ d.get('z', 5) }}|{{ d.get('z', none) is none }}|"
       "{{ d.get('r', d.get('a', none)) }}",
       "{\"d\": {\"a\": 1}}", "1|None|5|True|1"},
      {"{{ ['a', 'b']|join('\\n\\n')|tojson }}|{{ []|join('x') }}|{{ [1, 2]|join }}|{{ ['a', 1]|join(', ') }}|{{ y|join('-') }}",
       "{}", "\"a\\n\\nb\"||12|a, 1|"},
      {"{{ ('abc'|list)|tojson }}|{{ (l|list)|tojson }}|{{ (d|list)|tojson }}|{{ (y|list)|tojson }}",
       "{\"l\": [1, 2], \"d\": {\"k\": 1, \"j\": 2}}", "[\"a\", \"b\", \"c\"]|[1, 2]|[\"k\", \"j\"]|[]"},
      {"{{ ([x] + c | list)|tojson }}", "{\"x\": 0, \"c\": [1, 2]}", "[0, 1, 2]"},
      // Macros: keyword arguments, an absent argument undefined, defaults.
      {"{% macro m(a, b, c=5) %}[{{ a }}|{{ b }}|{{ c }}]{% endmacro %}{{ m(1, c=7) }}{{ m(b=2, a=1) }}{{ m(1, 2, 3) }}",
       "{}", "[1||7][1|2|5][1|2|3]"},
      {"{% macro m(a, flag) %}{% if flag and a %}yes{% else %}no{% endif %}:{{ flag is defined }}{% endmacro %}"
       "{{ m('x') }}|{{ m('x', flag=true) }}",
       "{}", "no:False|yes:True"},
      // Undefined values: no iterations, length 0.
      {"{% for x in m['content'] %}a{% endfor %}b{{ m['content']|length }}", "{\"m\": {}}", "b0"},
      // A comment is raw text to its end; a tag's own newlines count as lines.
      {"{# it's \"quoted\" #}{{ 'ok' }}", "{}", "ok"},
      {"{%- set s = 'a\nb\nc' -%}\n{{ s }}", "{}", "a\nb\nc"},
      // The whitespace the MiniMax template's unmarked tags leave behind.
      {"x\n        {% set a = 1 %}\n        {%- if a %}y{% endif %}", "{}", "x\ny"},
      {"{{- '<p>' }}\n    {% for k, v in d.items() %}\n    {{- k }}={{ v | tojson if v is not string else v }}\n"
       "    {% endfor %}\n    {{- '</p>' }}",
       "{\"d\": {\"a\": \"x\", \"n\": 2}}", "<p>\na=x\nn=2\n</p>"},
      {"{% set ns = namespace(i=-1) %}{{ ns.i }}|{{ l[1:]|tojson }}|{{ 'a</t>b'.split('</t>')[-1].strip('\\n') }}",
       "{\"l\": [1, 2, 3]}", "-1|[2, 3]|b"},
  };
  for (const Case& c : cases) {
    const std::string got = render_with(c.source, c.globals);
    require(got == c.want, std::string(c.source) + " rendered '" + got + "', want '" + c.want + "'");
  }
  // The template's own refusals surface as its message; the constructs'
  // misuse is named.
  struct Bad {
    const char* source;
    const char* globals;
    const char* needle;
  };
  const Bad bad[] = {
      {"{{ raise_exception('Unexpected role \\'' + r + '\\' after role \\'user\\'') }}", "{\"r\": \"tool\"}",
       "chat-template: Unexpected role 'tool' after role 'user'"},
      {"{% macro m(a) %}{{ a }}{% endmacro %}{{ m(1, z=7) }}", "{}", "macro 'm' takes no keyword argument 'z'"},
      {"{% macro m(a) %}{{ a }}{% endmacro %}{{ m(1, a=7) }}", "{}", "macro 'm' takes no keyword argument 'a'"},
      {"{% macro m(a) %}{{ a }}{% endmacro %}{{ m(1, 2) }}", "{}", "expected at most 1 argument(s)"},
      {"{{ {1: 2}|tojson }}", "{}", "dict literal: keys must be strings"},
      {"{{ [1] + 'a' }}", "{}", "two numbers"},
      {"{{ 'abc'|join(',') }}", "{}", "join: not a list"},
      {"{{ n|list }}", "{\"n\": 5}", "list: not a list/map/string"},
      {"{{ n|length }}", "{\"n\": null}", "length: not a list/map/string"},
      {"{% for x in n %}{% endfor %}", "{\"n\": null}", "cannot iterate a non-list"},
      {"{{ s.get('a') }}", "{\"s\": \"str\"}", "cannot call a non-callable value"},
      {"{{ {'a': 1 }}", "{}", "unterminated tag"},
      {"{# never closed", "{}", "unterminated tag"},
  };
  for (const Bad& b : bad) {
    const std::string got = render_error(b.source, b.globals);
    require(got.find(b.needle) != std::string::npos, std::string(b.source) + ": expected '" + b.needle + "', got: " + got);
  }
  // An error after a 28-line tag names the line the statement stands on.
  const std::string late = render_error("{%- set s = 'a\nb\nc' %}\n\n{{ raise_exception('x') }}{{ 1 + 'a' }}", "{}");
  require(late == "chat-template: x", "raise_exception's message alone: " + late);
  const std::string line = render_error("{%- set s = 'a\nb\nc' %}\n\n{{ 1 + 'a' }}", "{}");
  require(line.find("line 5:") != std::string::npos, "the line after a multi-line tag: " + line);
}

}  // namespace
