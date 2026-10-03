#pragma once
// The DeepSeek-V4 prompt renderer (DeepSeek-V4-Flash-0731, HF model_type
// deepseek_v4): the checkpoint ships no chat_template.jinja —
// `encoding/encoding_dsv4.py` (its `encode_messages`) is the reference, and
// this is that function over the service's template globals. It is the
// OLDER format than DeepSeek-V4.1's (text/dsv41_prompt.hpp) and differs
// from it materially:
//   <｜begin▁of▁sentence｜>[effort preamble]{system}
//   <｜User｜>{user}<｜Assistant｜><think>{reasoning}</think>{content}[\n\n<｜DSML｜tool_calls>...]<｜end▁of▁sentence｜>
//   ... the generation header after a user turn that nothing but an
//   assistant message follows: <｜Assistant｜> then <think> (thinking mode)
//   or </think> (chat mode).
//   * There is NO <｜System｜> token (its tokenizer has none): a system
//     message is its bare text wherever it stands, and a conversation that
//     ENDS in a system message gets no generation header (the reference
//     writes one after a user / developer message only).
//   * The DSML tags have no space after the tag token and the block is
//     "tool_calls": <｜DSML｜tool_calls>, <｜DSML｜invoke name="…">,
//     <｜DSML｜parameter name="…" string="true|false"> (DsmlDialect::kV4 in
//     text/tool_parser.hpp). No tool namespaces: a schema line is the
//     entry's `function` object as given (a flat entry is that object less
//     its "type").
//   * `reasoning_effort` is one of three fixed TEXT preambles at the very
//     start of the prompt in thinking mode (the reference's
//     REASONING_EFFORT_PROMPTS): low — its default — adds nothing, high and
//     max each prepend a paragraph. The service's other strings map onto
//     them: minimal, low and medium -> low (the reference's "high" reads
//     "Absolute maximum with no shortcuts permitted", above what medium
//     asks; low is the model's own default deliberation); high -> high;
//     xhigh and max -> max; absent -> low. Anything else is refused, as
//     the reference refuses it ("none" is the service's enable_thinking
//     false and never reaches the renderer).
// What it shares with V4.1: tool results are <tool_result>...</tool_result>
// blocks inside the user turn (a `tool` message merges into the preceding
// user turn or opens one; consecutive user messages merge, "\n\n" between
// the blocks), ordered by the previous assistant turn's call ids; the
// "## Tools" block is appended to the system message (the service's
// `tools` global attaches to the first message when it is a system
// message, or to an empty one inserted at the front); earlier turns'
// reasoning is dropped in thinking mode unless the conversation carries
// tools (`clear_thinking`, or the reference's own name `drop_thinking`;
// default true); `enable_thinking` false (alias `thinking`) is the
// reference's chat mode.
//
// The mapping from the globals to the reference's call, where the service's
// shape is not the reference's own:
//   * tool-call arguments arrive as a JSON object (the service parses the
//     wire string): the object's members are the parameters, exactly what
//     the reference derives from the wire string of that object. A string
//     is parsed as the reference does (not JSON: one parameter named
//     "arguments" carrying it; JSON but not an object: refused);
//   * a content-part array of a system / user / assistant message is its
//     text parts "\n\n"-joined (the reference has no such form; a non-text
//     part is refused); a tool message's array is the reference's own
//     tool_result list form (text parts, "[Unsupported TYPE]" for others).
// Beyond the service's roles the renderer carries the rest of the
// reference, so that the encoder's own shipped cases are goldens: the
// `developer` (a user-side turn with its own tools) and `latest_reminder`
// roles — the service never passes either (it renders OpenAI's developer
// as system) — and the message fields `tools`, `response_format`, `task`
// (the quick-instruction tokens) and `wo_eos`, which the service passes
// through untouched when a client sends them.
//
// Validation: tests/host/dsv4_prompt_test.cpp renders
// tests/data/dsv4_prompt_goldens.jsonl (tools/gen_dsv4_prompt_goldens.py:
// the snapshot's own encoder over the same globals, its four shipped cases
// included) byte-exact, and the tokenizer ids of every render.
#include <cstdint>
#include <string>
#include <string_view>

#include "loaders/minijson.hpp"

namespace dgpp::text {

class Dsv4Prompt {
 public:
  // The service's globals: "messages" (the normalized OpenAI array; roles
  // system / user / assistant / tool, and the reference's developer /
  // latest_reminder), optional "tools" (the OpenAI array),
  // "reasoning_effort" (a string), "clear_thinking" / "drop_thinking" and
  // "enable_thinking" / "thinking" (bools). Throws std::runtime_error
  // naming the offending field on anything the reference refuses (an
  // unknown role, an unknown effort or task, tool-call arguments that are
  // JSON but not an object, a developer message without content) and on
  // the forms it has no rendering for (a non-text content part).
  static std::string render(const minijson::Value& globals);
  // The knobs this renderer reads (the service's chat_template_kwargs gate).
  static bool reads(std::string_view name);
  // The renderer's revision (the prefix cache's key stands on it, as on a
  // Jinja template's source hash): bumped whenever the render changes.
  static uint64_t source_hash();

  static constexpr const char* kBos = "<｜begin▁of▁sentence｜>";
  static constexpr const char* kEos = "<｜end▁of▁sentence｜>";
  static constexpr const char* kUser = "<｜User｜>";
  static constexpr const char* kAssistant = "<｜Assistant｜>";
  static constexpr const char* kLatestReminder = "<｜latest_reminder｜>";
  static constexpr const char* kThinkOpen = "<think>";
  static constexpr const char* kThinkClose = "</think>";
  static constexpr const char* kDsml = "｜DSML｜";
};

}  // namespace dgpp::text
