// The Gemma 4 call notation and reasoning channel in the tool-call parser
// (text/tool_parser.hpp, ToolFormat::kGemma) over a fake decoder — host-only,
// always runs. The fake mirrors what makes this family different: every
// marker (<|channel>, <channel|>, <|tool_call>, <tool_call|>, <|"|>) is a
// SPECIAL token, so the decode the parser is given drops it. Pinned here:
// the channel's "thought\n" header dropped from the reasoning, the call
// notation read into json.dumps-form arguments (bare keys, token-quoted
// strings, bare scalars, nesting), several calls per turn, streaming
// deltas, and every malformed block flushed as literal content with its
// markers restored. The real tokenizer's round trip — the template's own
// rendering of a call, encoded and parsed back — is in
// tests/host/gemma4_chat_test.cpp.
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "text/tool_grammar.hpp"
#include "text/tool_parser.hpp"

namespace {

using dgpp::text::ChatMarker;
using dgpp::text::ChatMarkers;
using dgpp::text::ToolCallParser;
using dgpp::text::ToolFormat;
using dgpp::text::ToolSchemas;
using Event = ToolCallParser::Event;
using Kind = ToolCallParser::Event::Kind;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// The fake tokenizer: ids below 256 are bytes; the markers decode to
// NOTHING (special tokens under skip_special_tokens), like the turn end.
constexpr int64_t kChannelOpen = 1001, kChannelClose = 1002, kCallOpen = 1003, kCallClose = 1004, kQuote = 1005,
                  kTurnEnd = 1006, kResponseOpen = 1007;

ChatMarkers gemma_markers() {
  ChatMarkers m;
  m.think_open = ChatMarker{kChannelOpen, "<|channel>"};
  m.think_close = ChatMarker{kChannelClose, "<channel|>"};
  m.channel_thinking = true;
  m.tool_call_open = ChatMarker{kCallOpen, "<|tool_call>"};
  m.tool_call_close = ChatMarker{kCallClose, "<tool_call|>"};
  m.string_quote = ChatMarker{kQuote, "<|\"|>"};
  return m;
}

std::string fake_decode(const std::vector<int64_t>& ids) {
  std::string out;
  for (const int64_t id : ids)
    if (id >= 0 && id < 256) out.push_back(static_cast<char>(id));
  return out;
}

// A token stream from text: bytes, the marker strings as their ids.
std::vector<int64_t> ids_of(const std::string& text) {
  static const std::vector<std::pair<std::string, int64_t>> table = {
      {"<|tool_response>", kResponseOpen}, {"<|tool_call>", kCallOpen}, {"<tool_call|>", kCallClose},
      {"<|channel>", kChannelOpen},        {"<channel|>", kChannelClose}, {"<|\"|>", kQuote},
      {"<turn|>", kTurnEnd},
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

struct Parsed {
  std::string reasoning, content;
  std::vector<ToolCallParser::Call> calls;
  int reasoning_closed = 0;
  std::vector<Event> events;
};

// The prompt left the thinking to the model (the template's generation
// prompt never opens the channel): the parser starts in content and a
// leading <|channel> opens the reasoning.
Parsed parse(const std::string& text, bool model_may_open = true) {
  ToolCallParser::Options o;
  o.start_in_reasoning = false;
  o.model_may_open_thinking = model_may_open;
  ToolCallParser p(gemma_markers(), fake_decode, ToolSchemas(), o);
  Parsed out;
  for (const int64_t id : ids_of(text)) p.feed(id, &out.events);
  p.finish(&out.events);
  for (const Event& e : out.events) {
    if (e.kind == Kind::kReasoning) out.reasoning += e.text;
    else if (e.kind == Kind::kContent) out.content += e.text;
    else if (e.kind == Kind::kToolCall) out.calls.push_back(e.call);
    else ++out.reasoning_closed;
  }
  return out;
}

// One call whose arguments must be `json`, and no content.
void call_is(const std::string& block, const char* name, const std::string& json) {
  const Parsed p = parse(block);
  require(p.calls.size() == 1, "one call from '" + block + "' (content '" + p.content + "')");
  require(p.calls[0].name == name, "name '" + p.calls[0].name + "'");
  require(p.calls[0].arguments == json, "arguments '" + p.calls[0].arguments + "', want '" + json + "'");
  require(p.content.empty(), "no content beside the call, got '" + p.content + "'");
  // The arguments are JSON a client can parse.
  (void)dgpp::minijson::parse(p.calls[0].arguments);
}

// A malformed block: no call, and the literal text — markers restored — as content.
void flushed(const std::string& block) {
  const Parsed p = parse(block);
  require(p.calls.empty(), "no call from '" + block + "'");
  require(p.content == block, "the block flushed as content: got '" + p.content + "', want '" + block + "'");
}

}  // namespace

DGPP_TEST(gemma4_markers_state_the_format) {
  const ChatMarkers m = gemma_markers();
  require(m.tool_format() == ToolFormat::kGemma && m.tool_calls_available() && m.reasoning_available(), "format");
  // Without the string token the pair is the Qwen XML format's.
  ChatMarkers q = m;
  q.string_quote = ChatMarker{};
  require(q.tool_format() == ToolFormat::kQwenXml, "no string token: not Gemma");
  // The prompt's tail: thinking on ends at the model turn's newline, thinking off at a closed,
  // empty channel — the opener is the model's either way; a prompt ending in the opener is inside.
  const ChatMarkers n = [&] {
    ChatMarkers x = m;
    x.newline = ChatMarker{'\n', "\n"};
    return x;
  }();
  require(n.prompt_leaves_thinking_to_model(ids_of("<|turn>model\n")), "thinking on");
  require(n.prompt_leaves_thinking_to_model(ids_of("model\n<|channel>thought\n<channel|>")), "thinking off");
  require(!n.prompt_opens_thinking(ids_of("model\n<|channel>thought\n<channel|>")), "a closed channel is not open");
  require(n.prompt_opens_thinking(ids_of("model\n<|channel>")) && !n.prompt_leaves_thinking_to_model(ids_of("model\n<|channel>")),
          "a prompt ending in the opener");
  // No grammar is written for this notation: a vocabulary over it is unusable.
  const dgpp::text::GrammarVocab vocab(std::vector<std::string>(1100, "x"), m, {kTurnEnd}, 1100);
  require(!vocab.usable(), "no constrained decoding for the Gemma notation");
}

DGPP_TEST(gemma4_reasoning_channel_drops_its_header) {
  // <|channel>thought\n ... <channel|> then the answer.
  const Parsed p = parse("<|channel>thought\nThe user wants a sum.\n<channel|>4");
  require(p.reasoning == "The user wants a sum.\n", "reasoning '" + p.reasoning + "'");
  require(p.content == "4" && p.reasoning_closed == 1 && p.calls.empty(), "content '" + p.content + "'");
  // The header is held back until it is known to be one: no delta shows a part of it.
  for (const Event& e : p.events)
    if (e.kind == Kind::kReasoning)
      require(e.text.find("thought") == std::string::npos, "a header fragment was streamed: '" + e.text + "'");
  // An empty channel (what the template writes with thinking off, were the model to repeat it).
  const Parsed empty = parse("<|channel>thought\n<channel|>Hello");
  require(empty.reasoning.empty() && empty.content == "Hello" && empty.reasoning_closed == 1, "empty channel");
  // A channel that does not start with the header keeps all its text.
  const Parsed other = parse("<|channel>thinking hard<channel|>ok");
  require(other.reasoning == "thinking hard" && other.content == "ok", "another opening: '" + other.reasoning + "'");
  const Parsed prefix = parse("<|channel>thou<channel|>ok");
  require(prefix.reasoning.empty() && prefix.content == "ok", "a header prefix alone shows nothing");
  // Reasoning that itself begins with the header's letters.
  const Parsed again = parse("<|channel>thought\nthought\nagain<channel|>x");
  require(again.reasoning == "thought\nagain", "only the first header goes: '" + again.reasoning + "'");
  // No channel: everything is content.
  const Parsed plain = parse("Just an answer.<turn|>");
  require(plain.reasoning.empty() && plain.content == "Just an answer." && plain.reasoning_closed == 0, "plain");
  // After content has started a <|channel> is not an opener (its marker is skipped text).
  const Parsed late = parse("Answer <|channel>thought\nlate<channel|> end");
  require(late.reasoning.empty() && late.content == "Answer thought\nlate end", "late channel: '" + late.content + "'");
  // The prompt forbade it (another family's rule): the same.
  const Parsed off = parse("<|channel>thought\nx<channel|>y", /*model_may_open=*/false);
  require(off.reasoning.empty() && off.content == "thought\nxy", "not allowed to open: '" + off.content + "'");
}

DGPP_TEST(gemma4_call_notation_reads_into_json) {
  // What the template writes for {"city": "Paris", "days": 3}.
  call_is("<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:3}<tool_call|>", "get_weather",
          R"({"city": "Paris", "days": 3})");
  call_is("<|tool_call>call:ping{}<tool_call|>", "ping", "{}");
  // Scalars: booleans, null — and None, the template's own print of a null in the history —
  // and numbers, re-serialized in json.dumps form like every other format's arguments.
  call_is("<|tool_call>call:f{a:true,b:false,c:null,d:None,e:-3,f:0.5,g:12345678901234,h:1e-6,i:1.50}<tool_call|>", "f",
          R"({"a": true, "b": false, "c": null, "d": null, "e": -3, "f": 0.5, "g": 12345678901234, "h": 1e-06, "i": 1.5})");
  // Strings are whatever stands between two string tokens: quotes, braces, commas, newlines, colons.
  call_is("<|tool_call>call:run{code:<|\"|>print(\"hi\")\nif x: {a, b}[0]\n<|\"|>,note:<|\"|><|\"|>}<tool_call|>", "run",
          "{\"code\": \"print(\\\"hi\\\")\\nif x: {a, b}[0]\\n\", \"note\": \"\"}");
  call_is("<|tool_call>call:say{text:<|\"|>caf\xC3\xA9 \xE4\xB8\xAD\xE6\x96\x87 \\ back<|\"|>}<tool_call|>", "say",
          "{\"text\": \"caf\xC3\xA9 \xE4\xB8\xAD\xE6\x96\x87 \\\\ back\"}");
  // Nesting: maps with bare keys, lists, mixed.
  call_is("<|tool_call>call:f{opts:{a:{x:1.25,Y:<|\"|>y<|\"|>},b:[1,2,<|\"|>three<|\"|>,{k:None}]},e:[],m:{}}<tool_call|>", "f",
          R"({"opts": {"a": {"x": 1.25, "Y": "y"}, "b": [1, 2, "three", {"k": null}]}, "e": [], "m": {}})");
  // Keys may be token-quoted too (the template writes declarations that way).
  call_is("<|tool_call>call:f{<|\"|>my key<|\"|>:1,plain:2}<tool_call|>", "f", R"({"my key": 1, "plain": 2})");
  // Whitespace between the parts is tolerated.
  call_is("<|tool_call> call:f { a : 1 , b : [ 1 , 2 ] , c : <|\"|> s <|\"|> } <tool_call|>", "f",
          R"({"a": 1, "b": [1, 2], "c": " s "})");
  // A bare value that is not a JSON literal is taken as a string.
  call_is("<|tool_call>call:f{unit:celsius,when:2026-10-04,v:1.2.3}<tool_call|>", "f",
          R"({"unit": "celsius", "when": "2026-10-04", "v": "1.2.3"})");
  // A name transformers' \w+ would not read.
  call_is("<|tool_call>call:mcp.server-tool_v2{q:1}<tool_call|>", "mcp.server-tool_v2", R"({"q": 1})");
}

DGPP_TEST(gemma4_calls_among_reasoning_and_content) {
  // Reasoning, then two calls, then the turn stops at <|tool_response> (an EOS id of the model).
  const Parsed p = parse("<|channel>thought\nTwo cities.\n<channel|><|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}"
                         "<tool_call|><|tool_call>call:get_weather{city:<|\"|>Rome<|\"|>,days:2}<tool_call|><|tool_response>");
  require(p.reasoning == "Two cities.\n" && p.content.empty(), "reasoning / content");
  require(p.calls.size() == 2, "two calls");
  require(p.calls[0].arguments == R"({"city": "Paris"})" && p.calls[1].arguments == R"({"city": "Rome", "days": 2})", "arguments");
  // Content before and after a call.
  const Parsed q = parse("Let me check.<|tool_call>call:f{x:1}<tool_call|> Done.");
  require(q.content == "Let me check. Done." && q.calls.size() == 1 && q.calls[0].name == "f", "content around a call");
  // Event order: content, call, content.
  std::vector<Kind> kinds;
  for (const Event& e : q.events)
    if (kinds.empty() || kinds.back() != e.kind) kinds.push_back(e.kind);
  require(kinds.size() == 3 && kinds[0] == Kind::kContent && kinds[1] == Kind::kToolCall && kinds[2] == Kind::kContent,
          "event order");
}

DGPP_TEST(gemma4_malformed_blocks_flush_as_literal_text) {
  // The markers are special tokens — the decode drops them — and still come back in the flush.
  flushed("<|tool_call>get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>");            // no "call:"
  flushed("<|tool_call>call:{a:1}<tool_call|>");                                        // no name
  flushed("<|tool_call>call:f<tool_call|>");                                            // no arguments
  flushed("<|tool_call>call:f{a:1<tool_call|>");                                        // unclosed object
  flushed("<|tool_call>call:f{a:1}}<tool_call|>");                                      // text after the object
  flushed("<|tool_call>call:f{a:1} and more<tool_call|>");
  flushed("<|tool_call>call:f{a:<|\"|>open<tool_call|>");                               // unclosed string
  flushed("<|tool_call>call:f{a:1,a:2}<tool_call|>");                                   // a repeated key
  flushed("<|tool_call>call:f{:1}<tool_call|>");                                        // an empty key
  flushed("<|tool_call>call:f{a:}<tool_call|>");                                        // an empty value
  flushed("<|tool_call>call:f{a:[1,2}<tool_call|>");                                    // unclosed list
  flushed("<|tool_call>call:f{a:1,}<tool_call|>");                                      // a trailing comma
  flushed("<|tool_call>call:f{a:x<|\"|>y<|\"|>}<tool_call|>");                          // a string not at the value's start
  flushed("<|tool_call>{\"name\": \"f\", \"arguments\": {}}<tool_call|>");              // another family's JSON form
  // The stream ending inside a block.
  const Parsed open = parse("ok <|tool_call>call:f{a:1");
  require(open.calls.empty() && open.content == "ok <|tool_call>call:f{a:1", "unterminated block: '" + open.content + "'");
  // A nested opener aborts the first block and starts over.
  const Parsed nested = parse("<|tool_call>call:f{a:<|tool_call>call:g{b:2}<tool_call|>");
  require(nested.calls.size() == 1 && nested.calls[0].name == "g" && nested.content == "<|tool_call>call:f{a:",
          "nested opener: '" + nested.content + "'");
}

DGPP_TEST(gemma4_block_token_spans_cover_the_flushed_text) {
  // With token tracking (content logprobs) an aborted block's text is attributed id by id, the
  // restored markers included.
  ToolCallParser::Options o;
  o.start_in_reasoning = false;
  o.track_tokens = true;
  ToolCallParser p(gemma_markers(), fake_decode, ToolSchemas(), o);
  std::vector<Event> events;
  const std::string text = "<|tool_call>call:f{a:<|\"|>x<tool_call|>";
  const std::vector<int64_t> ids = ids_of(text);
  for (const int64_t id : ids) p.feed(id, &events);
  p.finish(&events);
  require(events.size() == 1 && events[0].kind == Kind::kContent && events[0].text == text, "flushed text");
  require(events[0].tokens.size() == ids.size(), "one span per id");
  size_t at = 0;
  for (size_t i = 0; i < ids.size(); ++i) {
    const auto& span = events[0].tokens[i];
    require(span.token == i && span.begin == at && span.end > span.begin, "span " + std::to_string(i));
    at = span.end;
  }
  require(at == text.size(), "the spans cover the text");
}
