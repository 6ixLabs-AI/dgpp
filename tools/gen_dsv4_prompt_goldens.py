#!/usr/bin/env python3
"""Generates tests/data/dsv4_prompt_goldens.jsonl — the differential goldens
for the DeepSeek-V4 prompt renderer (src/text/dsv4_prompt.cpp; the family is
DeepSeek-V4-Flash-0731, HF model_type deepseek_v4).

The GOLDEN SOURCE is the checkpoint's own encoder,
`encoding/encoding_dsv4.py` in the Hub snapshot (no chat_template.jinja
ships with this model), applied to the SERVICE'S template globals:
"messages" (OpenAI messages: system / user / assistant with
reasoning_content and tool_calls / tool with tool_call_id; plus the
reference's own developer and latest_reminder roles and its message fields
tools / response_format / task / wo_eos), optional "tools" (the OpenAI
array), "reasoning_effort" (the service's strings), "clear_thinking" (or the
reference's name, "drop_thinking") and "enable_thinking" (alias "thinking").
The mapping from the globals to the reference's call is the renderer's
contract, written here once:
  * tools attach to the first message when it is a system message ("tools"
    field), or to an empty system message inserted at the front; a flat
    tool entry ({name, description, parameters}, no "function" wrapper) is
    wrapped, less its "type";
  * enable_thinking false -> thinking_mode "chat", else "thinking";
  * clear_thinking / drop_thinking -> drop_thinking (default true);
  * reasoning_effort: minimal, low, medium -> "low" (the reference's
    default: no preamble); high -> "high"; xhigh, max -> "max"; absent ->
    the reference's default (low);
  * tool-call arguments given as an object (the service parses the wire
    string) -> the JSON string of that object, which is what the reference
    reads; a string passes as is;
  * a content-part array on a system / user / assistant message -> its text
    parts joined with a blank line (the reference renders strings there; a
    tool message's array is the reference's own list form and passes as is).
The first four cases are the encoder's own shipped tests
(encoding/tests/test_input_N.json): their renders must equal the shipped
test_output_N.txt byte for byte, or the script refuses to write.

Each case records the render and the tokenizer ids of the render (HF
tokenizers on the snapshot). The header carries the encoder's FNV-1a-64
hash and the tokenizer revision; the C++ gate refuses a different
tokenizer.json.

Regenerating (the tools venv):
  .venv/bin/python3 tools/gen_dsv4_prompt_goldens.py [--model deepseek-ai/DeepSeek-V4-Flash-0731]
      [--snapshot DIR] [--out tests/data/dsv4_prompt_goldens.jsonl]
"""
import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cluster_doctor import cache_root, cached_snapshot  # noqa: E402
from site_env import cache_environment  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
EFFORT = {"minimal": "low", "low": "low", "medium": "low", "high": "high", "xhigh": "max", "max": "max"}


def fnv1a64(data: bytes) -> int:
    h = 1469598103934665603
    for b in data:
        h = (h ^ b) * 1099511628211 & 0xFFFFFFFFFFFFFFFF
    return h


def weather_tool():
    return {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city (城市天气查询).",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "days": {"type": "integer", "minimum": 0}}, "required": ["city"]}}}


def search_tool():
    return {"type": "function", "function": {
        "name": "search", "description": "Search the web", "strict": True,
        "parameters": {"type": "object", "properties": {"query": {"type": "string"},
                                                        "top_k": {"type": "integer", "default": 5}},
                       "required": ["query"], "additionalProperties": False}}}


def flat_tool():
    return {"type": "function", "name": "now", "description": "The current time.",
            "parameters": {"type": "object", "properties": {}}}


def call(cid, name, arguments):
    c = {"type": "function", "function": {"name": name, "arguments": arguments}}
    if cid is not None:
        c["id"] = cid
    return c


SHIPPED = [
    # (name, input file, the kwargs built from it, the shipped output)
    ("shipped_1_thinking_with_tools", 1, lambda td: {"messages": td["messages"], "tools": td["tools"]}),
    ("shipped_2_thinking_without_tools", 2, lambda td: {"messages": td}),
    ("shipped_3_interleaved_thinking_search", 3, lambda td: {"messages": td}),
    ("shipped_4_quick_instruction_task", 4, lambda td: {"messages": td, "enable_thinking": False}),
]

HELLO = [{"role": "user", "content": "Hi"}]
TWO_TURNS = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "reasoning_content": "The user said hello, I should greet back.", "content": "Hi there! How can I help you?"},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "reasoning_content": "It is Paris.", "content": "The capital of France is Paris."},
    {"role": "user", "content": "And of Spain?"}]

CASES = [
    # ---- plain chat, the two modes ---------------------------------------------
    ("simple_user", {"messages": [{"role": "user", "content": "The capital of France is"}]}),
    ("system_user", {"messages": [{"role": "system", "content": "You are a concise geography assistant."},
                                  {"role": "user", "content": "The capital of France is"}]}),
    ("chat_mode", {"messages": [{"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": "What is 2+2?"}], "enable_thinking": False}),
    ("chat_mode_thinking_alias", {"messages": HELLO, "thinking": False}),
    ("thinking_explicit", {"messages": HELLO, "enable_thinking": True}),
    # ---- every effort string -----------------------------------------------------
    ("effort_minimal", {"messages": HELLO, "reasoning_effort": "minimal"}),
    ("effort_low", {"messages": HELLO, "reasoning_effort": "low"}),
    ("effort_medium", {"messages": HELLO, "reasoning_effort": "medium"}),
    ("effort_high", {"messages": HELLO, "reasoning_effort": "high"}),
    ("effort_xhigh", {"messages": HELLO, "reasoning_effort": "xhigh"}),
    ("effort_max", {"messages": HELLO, "reasoning_effort": "max"}),
    ("effort_high_system", {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
                            "reasoning_effort": "high"}),
    ("effort_max_chat_mode_no_preamble", {"messages": [{"role": "system", "content": "Be brief."},
                                                       {"role": "user", "content": "Hi"}],
                                          "reasoning_effort": "max", "enable_thinking": False}),
    ("effort_high_tools_no_system", {"messages": [{"role": "user", "content": "Weather in Paris?"}],
                                     "tools": [weather_tool()], "reasoning_effort": "high"}),
    ("effort_max_multi_turn", {"messages": TWO_TURNS, "reasoning_effort": "max"}),
    # ---- earlier turns' reasoning --------------------------------------------------
    ("multi_turn_drop_thinking", {"messages": TWO_TURNS}),
    ("multi_turn_keep_thinking", {"messages": TWO_TURNS, "clear_thinking": False}),
    ("multi_turn_keep_thinking_native_name", {"messages": TWO_TURNS, "drop_thinking": False}),
    ("multi_turn_drop_thinking_explicit", {"messages": TWO_TURNS, "clear_thinking": True}),
    ("multi_turn_chat_mode", {"messages": TWO_TURNS, "enable_thinking": False}),
    ("multi_turn_tools_keep_reasoning", {"messages": TWO_TURNS, "tools": [weather_tool()]}),
    ("assistant_last_no_header", {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "reasoning_content": "r", "content": "Hello."}]}),
    ("assistant_last_chat_mode", {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "reasoning_content": "r", "content": "Hello."}], "enable_thinking": False}),
    ("assistant_without_reasoning", {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "Again"},
        {"role": "assistant", "content": "Hello again."}]}),
    # ---- system messages past the front: bare text, no generation header ------------
    ("mid_system_last", {"messages": [
        {"role": "system", "content": "Root."},
        {"role": "user", "content": "One"},
        {"role": "assistant", "content": "Two"},
        {"role": "system", "content": "Now answer in French."}]}),
    ("mid_system_then_user", {"messages": [
        {"role": "system", "content": "Root."},
        {"role": "user", "content": "One"},
        {"role": "assistant", "reasoning_content": "t", "content": "Two"},
        {"role": "system", "content": "Now answer in French."},
        {"role": "user", "content": "Three"}]}),
    ("user_then_system", {"messages": [
        {"role": "user", "content": "One"},
        {"role": "system", "content": "Answer in French."}]}),
    ("two_systems_front", {"messages": [
        {"role": "system", "content": "First."},
        {"role": "system", "content": "Second."},
        {"role": "user", "content": "Go"}], "tools": [weather_tool()]}),
    # ---- tools -----------------------------------------------------------------------
    ("tools_no_system", {"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [weather_tool()]}),
    ("tools_with_system", {"messages": [{"role": "system", "content": "Use tools when helpful."},
                                        {"role": "user", "content": "Weather in Paris for 3 days?"}],
                           "tools": [weather_tool(), search_tool(), flat_tool()]}),
    ("tools_chat_mode", {"messages": [{"role": "system", "content": "Use tools."},
                                      {"role": "user", "content": "Weather in Paris?"}],
                         "tools": [weather_tool()], "enable_thinking": False}),
    ("tools_empty_array", {"messages": HELLO, "tools": []}),
    ("tool_call_and_result", {"messages": [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "reasoning_content": "Call the tool.", "content": "",
         "tool_calls": [call("c1", "get_weather", {"city": "Paris", "days": 3})]},
        {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 21, \"unit\": \"C\"}"}],
        "tools": [weather_tool()]}),
    ("tool_call_with_content_and_reasoning", {"messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "reasoning_content": "I will look it up.", "content": "Let me check.",
         "tool_calls": [call("c1", "get_weather", "{\"city\": \"Paris\"}")]},
        {"role": "tool", "tool_call_id": "c1", "content": "21C"},
        {"role": "assistant", "reasoning_content": "Done.", "content": "It is 21C in Paris."},
        {"role": "user", "content": "Thanks. And Rome?"}],
        "tools": [weather_tool()]}),
    ("tool_loop_two_steps", {"messages": [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Compare Paris and Rome."},
        {"role": "assistant", "reasoning_content": "Paris first.", "content": "",
         "tool_calls": [call("p", "get_weather", {"city": "Paris"})]},
        {"role": "tool", "tool_call_id": "p", "content": "Paris: 21C"},
        {"role": "assistant", "reasoning_content": "Now Rome.", "content": "",
         "tool_calls": [call("r", "get_weather", {"city": "Rome"})]},
        {"role": "tool", "tool_call_id": "r", "content": "Rome: 25C"}],
        "tools": [weather_tool()]}),
    ("tool_loop_chat_mode", {"messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [call("p", "get_weather", {"city": "Paris"})]},
        {"role": "tool", "tool_call_id": "p", "content": "Paris: 21C"}],
        "tools": [weather_tool()], "enable_thinking": False}),
    ("tool_call_without_arguments", {"messages": [
        {"role": "user", "content": "What time is it, and the weather in Paris?"},
        {"role": "assistant", "reasoning_content": "Two calls.", "content": "",
         "tool_calls": [call("t", "now", {}), call("w", "get_weather", {"city": "Paris"})]},
        {"role": "tool", "tool_call_id": "t", "content": "12:00"},
        {"role": "tool", "tool_call_id": "w", "content": "21C"}],
        "tools": [flat_tool(), weather_tool()]}),
    ("parallel_calls_results_reordered", {"messages": [
        {"role": "user", "content": "Weather in Paris and Rome?"},
        {"role": "assistant", "content": "", "tool_calls": [
            call("a", "get_weather", {"city": "Paris"}), call("b", "get_weather", {"city": "Rome"}),
            call("c", "search", {"query": "weather", "top_k": 3})]},
        {"role": "tool", "tool_call_id": "c", "content": "results"},
        {"role": "tool", "tool_call_id": "b", "content": "Rome: 25C"},
        {"role": "tool", "tool_call_id": "a", "content": "Paris: 21C"},
        {"role": "user", "content": "Thanks, summarize."}],
        "tools": [weather_tool(), search_tool()]}),
    ("parallel_calls_unknown_and_missing_ids", {"messages": [
        {"role": "user", "content": "Go"},
        {"role": "assistant", "content": "", "tool_calls": [
            call("a", "get_weather", {"city": "Paris"}), call("b", "get_weather", {"city": "Rome"})]},
        {"role": "tool", "tool_call_id": "b", "content": "Rome: 25C"},
        {"role": "tool", "tool_call_id": "zzz", "content": "stray"},
        {"role": "tool", "content": "no id"},
        {"role": "tool", "tool_call_id": "a", "content": "Paris: 21C"}],
        "tools": [weather_tool()]}),
    ("call_ids_on_the_function", {"messages": [
        {"role": "user", "content": "Go"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"id": "f1", "name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}},
            {"type": "function", "function": {"id": "f2", "name": "get_weather", "arguments": "{\"city\": \"Rome\"}"}}]},
        {"role": "tool", "tool_call_id": "f2", "content": "Rome"},
        {"role": "tool", "tool_call_id": "f1", "content": "Paris"}],
        "tools": [weather_tool()]}),
    ("results_without_tools_block_drop_thinking", {"messages": [
        {"role": "user", "content": "Go"},
        {"role": "assistant", "reasoning_content": "call", "content": "",
         "tool_calls": [call("a", "get_weather", {"city": "Paris"})]},
        {"role": "tool", "tool_call_id": "a", "content": "Paris: 21C"},
        {"role": "assistant", "reasoning_content": "answer", "content": "21C."},
        {"role": "user", "content": "And tomorrow?"}]}),
    ("tool_result_then_user_text_merges", {"messages": [
        {"role": "user", "content": "Go"},
        {"role": "assistant", "content": "", "tool_calls": [call("a", "get_weather", {"city": "Paris"})]},
        {"role": "tool", "tool_call_id": "a", "content": "Paris: 21C"},
        {"role": "user", "content": "Also tell me about Rome."},
        {"role": "user", "content": "Briefly."}],
        "tools": [weather_tool()]}),
    ("consecutive_users_merge", {"messages": [
        {"role": "user", "content": "First part."},
        {"role": "user", "content": "Second part."},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": "Third."},
        {"role": "user", "content": ""}]}),
    ("tool_message_first", {"messages": [
        {"role": "tool", "tool_call_id": "x", "content": "orphan result"},
        {"role": "user", "content": "What was that?"}]}),
    ("tool_result_list_content", {"messages": [
        {"role": "user", "content": "Look it up."},
        {"role": "assistant", "content": "", "tool_calls": [call("s1", "search", {"query": "value"})]},
        {"role": "tool", "tool_call_id": "s1", "content": [
            {"type": "text", "text": "found"}, {"type": "audio", "data": "x"}, {"output": "untyped"},
            {"type": "text", "text": "第二段"}]}],
        "tools": [search_tool()]}),
    ("call_args_forms", {"messages": [
        {"role": "user", "content": "Go"},
        {"role": "assistant", "content": "On it.", "tool_calls": [
            call("o1", "get_weather", {"city": "Oslo", "days": 1, "flags": {"metric": True, "n": [1, 2.5, None]},
                                       "note": "a \"quoted\" <tag> & 换行\nline"}),
            call("o2", "get_weather", "{\"city\": \"Bergen\", \"days\": 2}"),
            call("o3", "get_weather", "not json"),
            call("o4", "get_weather", "{}"),
            call("o5", "get_weather", {}),
            call("o6", "search", "{\"query\": \"x\", \"top_k\": 10, \"exact\": false, \"filter\": null, "
                                 "\"range\": [0.5, 1e-07, 100000.0, -3], \"nested\": {\"k\": \"v\"}}"),
            call("o7", "search", " {\"query\" : \"spaced\" ,\"top_k\":7 } ")]},
        {"role": "tool", "tool_call_id": "o1", "content": "ok"}],
        "tools": [weather_tool(), search_tool()]}),
    # ---- content forms -----------------------------------------------------------------
    ("content_parts", {"messages": [
        {"role": "system", "content": [{"type": "text", "text": "System part."}]},
        {"role": "user", "content": [{"type": "text", "text": "第一段"}, {"type": "text", "text": "second part"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        {"role": "user", "content": [{"type": "text", "text": "again"}]}]}),
    ("empty_content", {"messages": [{"role": "user", "content": ""}]}),
    ("empty_system", {"messages": [{"role": "system", "content": ""}, {"role": "user", "content": "Hi"}]}),
    ("unicode_and_code", {"messages": [{"role": "system", "content": "Répondez en français, s'il vous plaît."},
                                       {"role": "user", "content": "def add(a, b):\n    return a + b  # 注释\n"}]}),
    ("special_token_text_in_content", {"messages": [
        {"role": "user", "content": "What does <think> mean, and <｜System｜> or ｜DSML｜?"}]}),
    # ---- the reference's own roles and message fields ------------------------------------
    ("latest_reminder", {"messages": [
        {"role": "system", "content": "You are DeepSeek."},
        {"role": "latest_reminder", "content": "2026-07-31,Friday,Paris,App,English"},
        {"role": "user", "content": "What day is it?"}]}),
    ("developer_with_tools", {"messages": [
        {"role": "system", "content": "You are DeepSeek."},
        {"role": "developer", "content": "Find the weather in Paris.", "tools": [weather_tool()]},
        {"role": "assistant", "reasoning_content": "call", "content": "",
         "tool_calls": [call(None, "get_weather", {"city": "Paris"})]},
        {"role": "tool", "content": "21C"}]}),
    ("developer_dropped_before_last_user", {"messages": [
        {"role": "developer", "content": "Earlier instruction."},
        {"role": "assistant", "reasoning_content": "t", "content": "Noted."},
        {"role": "user", "content": "Now answer."}]}),
    ("system_message_tools_and_response_format", {"messages": [
        {"role": "system", "content": "Reply in JSON.", "tools": [weather_tool()],
         "response_format": {"type": "object", "properties": {"answer": {"type": "string"}}}},
        {"role": "user", "content": "Weather in Paris?"}]}),
    ("task_query", {"messages": [
        {"role": "user", "content": "best pizza in Naples", "task": "query"}], "enable_thinking": False}),
    ("task_action_thinking", {"messages": [
        {"role": "user", "content": "Who won the match yesterday?", "task": "action"},
        {"role": "assistant", "content": "Search"}]}),
    ("task_title_after_assistant", {"messages": [
        {"role": "user", "content": "Explain BPE."},
        {"role": "assistant", "reasoning_content": "explain", "content": "BPE merges frequent pairs.", "task": "title"},
        {"role": "assistant", "content": "BPE explained"}]}),
    ("task_user_does_not_merge", {"messages": [
        {"role": "user", "content": "first", "task": "domain"},
        {"role": "user", "content": "second"}], "enable_thinking": False}),
    ("assistant_wo_eos", {"messages": [
        {"role": "user", "content": "Continue the list."},
        {"role": "assistant", "reasoning_content": "go on", "content": "1. apples\n2.", "wo_eos": True}]}),
]


def flatten(content):
    return "\n\n".join(part["text"] for part in content)


def to_reference(kwargs):
    messages = copy.deepcopy(kwargs["messages"])
    for m in messages:
        if m.get("role") != "tool" and isinstance(m.get("content"), list):
            m["content"] = flatten(m["content"])
        for tc in m.get("tool_calls") or []:
            arguments = tc["function"].get("arguments")
            if isinstance(arguments, dict):
                tc["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)
    tools = kwargs.get("tools")
    if tools:
        tools = [t if "function" in t else {"type": "function", "function": {k: v for k, v in t.items() if k != "type"}}
                 for t in copy.deepcopy(tools)]
        if messages and messages[0].get("role") == "system":
            messages[0]["tools"] = tools
        else:
            messages.insert(0, {"role": "system", "content": "", "tools": tools})
    thinking = kwargs.get("thinking", kwargs.get("enable_thinking", True))
    thinking_mode = "chat" if thinking is False else "thinking"
    drop = kwargs.get("drop_thinking", kwargs.get("clear_thinking", True))
    effort = kwargs.get("reasoning_effort")
    if effort is not None:
        effort = EFFORT[effort]
    return messages, thinking_mode, drop, effort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--out", default="tests/data/dsv4_prompt_goldens.jsonl")
    args = ap.parse_args()
    snap = Path(args.snapshot) if args.snapshot else cached_snapshot(args.model, cache_root(cache_environment()))
    enc_path = snap / "encoding" / "encoding_dsv4.py"
    spec = importlib.util.spec_from_file_location("dsv4_encoding", enc_path)
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    import tokenizers
    tok_path = snap / "tokenizer.json"
    tok = tokenizers.Tokenizer.from_file(str(tok_path))

    cases = []
    tests_dir = snap / "encoding" / "tests"
    for name, n, build in SHIPPED:
        kwargs = build(json.loads((tests_dir / f"test_input_{n}.json").read_text(encoding="utf-8")))
        cases.append((name, kwargs, (tests_dir / f"test_output_{n}.txt").read_text(encoding="utf-8")))
    cases += [(name, kwargs, None) for name, kwargs in CASES]
    names = [name for name, _, _ in cases]
    if len(set(names)) != len(names):
        sys.exit("duplicate case names")

    header = {
        "model": args.model,
        "encoder_hash": f"{fnv1a64(enc_path.read_bytes()):016x}",
        "tokenizer_revision": f"{fnv1a64(tok_path.read_bytes()):016x}",
        "tokenizers": tokenizers.__version__,
        "cases": len(cases),
    }
    lines = [json.dumps(header)]
    for name, kwargs, shipped in cases:
        messages, mode, drop, effort = to_reference(kwargs)
        render = enc.encode_messages(messages, thinking_mode=mode, drop_thinking=drop, reasoning_effort=effort)
        if shipped is not None and render != shipped:
            sys.exit(f"case {name}: the mapping does not reproduce the encoder's shipped output")
        ids = tok.encode(render, add_special_tokens=False).ids
        lines.append(json.dumps({"name": name, "kwargs": kwargs, "render": render, "ids": ids}, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {len(cases)} cases (encoder {header['encoder_hash']}, tokenizer {header['tokenizer_revision']})")


if __name__ == "__main__":
    main()
