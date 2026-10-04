#!/usr/bin/env python3
"""Generates tests/data/glm_chat_template_goldens.jsonl — the differential
goldens for the GLM chat template (M6 Stage 3b, DESIGN §10).

The GOLDEN SOURCE is jinja2 3.1.2 configured EXACTLY like transformers'
chat-template environment (src/transformers/utils/chat_template_utils.py,
verified 2026-08-31):
  * trim_blocks=True, lstrip_blocks=True
  * extensions=[jinja2.ext.loopcontrols]  (the template uses {% break %};
    plain jinja2 rejects it)
  * the tojson OVERRIDE: json.dumps(ensure_ascii=..., indent=None,
    separators=None, sort_keys=False) — jinja's built-in tojson both
    escapes HTML (<, >, &, ') and rejects the ensure_ascii kwarg, so the
    override is REQUIRED to render this template at all.

The corpus is written with ensure_ascii=False (raw UTF-8) so minijson's
reader never meets surrogate pairs (it decodes single \\uXXXX only; the
tokenizer gate had to learn that lesson the hard way).

Each case carries the render globals as JSON ("kwargs"), the expected
render ("render"), and the token ids HF tokenizers 0.23.1 produces for
the render ("ids" — the template->tokenizer integration differential;
the C++ gate re-encodes with glm_tokenizer and compares).

Case selection (per DESIGN §10: English, Chinese, code, reasoning
blocks, tool calls — plus the structural edges the template branches
on): the plain/system/multimodal user paths; the reasoning-keep path
(trailing assistant) vs the reasoning-strip path (mid-conversation
assistant, and clear_thinking=false); reasoning_content as a field vs
think-markers inside content (the content.split path) vs both; the
close-without-open split edge; tool headers flat vs OpenAI-wrapped vs
defer_loading; aligned/sorted/unaligned tool-response blocks; string
vs JSON-serialized arguments; tool_reference entries direct and nested
in outputs; empty conversation and empty content.

Keyed by chat_template.jinja's FNV-1a-64 hash (header "template_hash"),
cross-referenced with the tokenizer revision ("tokenizer_revision").

Regenerating: python3
    tools/gen_chat_template_goldens.py
(the venv pins jinja2 3.1.2 + tokenizers 0.23.1).
"""
import glob
import json
import os
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cluster_doctor import cache_root, cached_snapshot
from site_env import cache_environment


MODEL = "unsloth/GLM-5.3-Flash-FP8"


def fnv1a64(data: bytes) -> int:
    # The house hash (glm_loader.cpp / glm_tokenizer.cpp), not the FNV
    # standard basis.
    h = 1469598103934665603
    for b in data:
        h = (h ^ b) * 1099511628211 & 0xFFFFFFFFFFFFFFFF
    return h


# The template's reasoning markers, built by concatenation (heredocs and
# raw literals mangle them in tool transcripts; see the record).
THINK_OPEN = "<" + "think>"
THINK_CLOSE = "</" + "think>"


def env():
    import jinja2
    e = jinja2.Environment(
        trim_blocks=True, lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"],
    )
    # transformers' tojson override, character for character.
    def tojson(x, ensure_ascii=False, indent=None, separators=None,
               sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)
    e.filters["tojson"] = tojson
    # transformers' raise_exception global (a TemplateError).
    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)
    e.globals["raise_exception"] = raise_exception
    return e


def weather_tool():
    return {
        "name": "get_weather",
        "description": "Get the current weather for a city (城市天气查询).",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {"type": "integer", "minimum": 0},
            },
            "required": ["city"],
        },
    }


def call(cid, name, arguments):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": arguments}}


CASES = [
    ("simple_user_gen", {
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "add_generation_prompt": True,
    }),
    ("system_user_gen", {
        "messages": [
            {"role": "system",
             "content": "You are a concise geography assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "add_generation_prompt": True,
    }),
    ("chinese_user_gen", {
        "messages": [{"role": "user", "content": "写一首关于秋天的诗。"}],
        "add_generation_prompt": True,
    }),
    ("code_user_gen", {
        "messages": [{"role": "user", "content":
                      "def add(a, b):\n    return a + b\nExplain this code."}],
        "add_generation_prompt": True,
    }),
    ("empty_conversation_gen", {
        "messages": [],
        "add_generation_prompt": True,
    }),
    ("empty_user_content_gen", {
        "messages": [{"role": "user", "content": ""}],
        "add_generation_prompt": True,
    }),
    ("effort_low", {
        "messages": [{"role": "user", "content": "Hi"}],
        "add_generation_prompt": True,
        "reasoning_effort": "low",
    }),
    ("effort_high", {
        "messages": [{"role": "user", "content": "Hi"}],
        "add_generation_prompt": True,
        "reasoning_effort": "high",
    }),
    ("multimodal_parts_gen", {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe these in order."},
            {"type": "image_url", "image_url": {"url": "https://x/cat.png"}},
            {"type": "video_url"},
            {"type": "input_audio"},
        ]}],
        "add_generation_prompt": True,
    }),
    ("multi_turn_mid_reasoning_stripped", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant", "reasoning_content": "Two plus two is four.",
             "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ],
        "add_generation_prompt": True,
    }),
    ("trailing_assistant_reasoning_content", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant", "reasoning_content": "Two plus two is four.",
             "content": "4"},
        ],
        "add_generation_prompt": True,
    }),
    ("trailing_assistant_think_markers", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant",
             "content": THINK_OPEN + " Two plus two is four." + THINK_CLOSE +
                        " The answer is 4."},
        ],
        "add_generation_prompt": True,
    }),
    ("think_close_without_open", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant",
             "content": " partial thinking" + THINK_CLOSE + " The answer is 4."},
        ],
        "add_generation_prompt": True,
    }),
    ("trailing_assistant_both_reasoning_forms", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant",
             "reasoning_content": "Field form wins over markers.",
             "content": THINK_OPEN + " marker form" + THINK_CLOSE +
                        " The answer is 4."},
        ],
        "add_generation_prompt": True,
    }),
    ("clear_thinking_false_mid_turn", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant", "reasoning_content": "Two plus two is four.",
             "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ],
        "add_generation_prompt": True,
        "clear_thinking": False,
    }),
    ("no_generation_prompt", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant", "content": "4"},
        ],
    }),
    ("tools_flat", {
        "messages": [{"role": "user",
                      "content": "What's the weather in Paris?"}],
        "tools": [weather_tool(), {
            "name": "search_web", "description": "Search the web."}],
        "add_generation_prompt": True,
    }),
    ("tools_function_wrap", {
        "messages": [{"role": "user",
                      "content": "What's the weather in Paris?"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "description": "Weather <queries> & conditions 天气",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}}],
        "add_generation_prompt": True,
    }),
    ("tools_deferred", {
        "messages": [{"role": "user",
                      "content": "What's the weather in Paris?"}],
        "tools": [weather_tool(), {
            "name": "lazy_tool", "description": "Loaded on demand.",
            "defer_loading": True}],
        "add_generation_prompt": True,
    }),
    ("tool_call_aligned", {
        "messages": [
            {"role": "user", "content": "Weather in Paris and Tokyo?"},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather",
                     {"city": "Paris", "days": 3, "factor": 0.5}),
                call("call_2", "get_weather", {"city": "Tokyo"})]},
            {"role": "tool", "content": "18C, partly cloudy",
             "tool_call_id": "call_1"},
            {"role": "tool", "content": "22C, clear",
             "tool_call_id": "call_2"},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("tool_call_unsorted", {
        "messages": [
            {"role": "user", "content": "Weather in Paris and Tokyo?"},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather", {"city": "Paris"}),
                call("call_2", "get_weather", {"city": "Tokyo"})]},
            {"role": "tool", "content": "22C, clear",
             "tool_call_id": "call_2"},
            {"role": "tool", "content": "18C, partly cloudy",
             "tool_call_id": "call_1"},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("tool_call_str_args", {
        # NOTE: arguments MUST be a dict (a bare JSON string crashes the
        # template at `_args.items()`); the `v is not string` branch is
        # for string VALUES inside it, rendered verbatim.
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather",
                     {"city": "Paris", "note": "asked twice"})]},
            {"role": "tool", "content": "18C, partly cloudy",
             "tool_call_id": "call_1"},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("tool_call_float_args", {
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather",
                     {"scale": 1.0, "ratio": 0.5, "count": 3})]},
            {"role": "tool", "content": "18C",
             "tool_call_id": "call_1"},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("tool_call_outputs_and_references", {
        "messages": [
            {"role": "user", "content": "Weather in Paris and Tokyo?"},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather", {"city": "Paris"}),
                call("call_2", "get_weather", {"city": "Tokyo"})]},
            {"role": "tool", "content": [
                {"tool_call_id": "call_1", "output": "18C, partly cloudy"},
                {"tool_call_id": "call_2", "output": [
                    {"type": "tool_reference", "name": "get_weather",
                     "tool_call_id": "call_9"}]},
            ]},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("tool_reference_direct", {
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather", {"city": "Paris"})]},
            {"role": "tool", "content": [
                {"type": "tool_reference", "name": "get_weather",
                 "tool_call_id": "call_1"},
            ]},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("tool_no_prior_call", {
        "messages": [
            {"role": "user", "content": "Weather?"},
            {"role": "tool", "content": "18C, partly cloudy",
             "tool_call_id": "call_0"},
            {"role": "user", "content": "Thanks!"},
        ],
        "add_generation_prompt": True,
    }),
]


# The Qwen3.8-Flash-Next template: the reasoning-effort system
# preamble, the tools block, <think> reasoning kept only after the last
# user query (preserve_thinking), the <function=...><parameter=...> tool
# calls, tool responses grouped into one user turn, enable_thinking false
# emitting an empty think block.
def qwen_call(cid, name, arguments):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": arguments}}


QWEN_CASES = [
    ("simple_user_gen", {
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "add_generation_prompt": True,
    }),
    ("system_user_gen", {
        "messages": [
            {"role": "system", "content": "You are a concise geography assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "add_generation_prompt": True,
    }),
    ("chinese_user_gen", {
        "messages": [{"role": "user", "content": "写一首关于秋天的诗。"}],
        "add_generation_prompt": True,
    }),
    ("code_user_gen", {
        "messages": [{"role": "user", "content": "def add(a, b):\n    return a + b\nExplain this code."}],
        "add_generation_prompt": True,
    }),
    ("effort_low", {
        "messages": [{"role": "user", "content": "Hi"}],
        "add_generation_prompt": True, "reasoning_effort": "low",
    }),
    ("effort_medium_system", {
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
        "add_generation_prompt": True, "reasoning_effort": "medium",
    }),
    ("thinking_disabled", {
        "messages": [{"role": "user", "content": "Hi"}],
        "add_generation_prompt": True, "enable_thinking": False,
    }),
    ("thinking_disabled_with_system", {
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
        "add_generation_prompt": True, "enable_thinking": False,
    }),
    ("multi_turn_reasoning_kept_after_last_query", {
        "messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "thinking about Q1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "thinking about Q2"},
        ],
        "add_generation_prompt": False,
    }),
    ("multi_turn_preserve_thinking_false", {
        "messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "reasoning_content": "thinking about Q1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2", "reasoning_content": "thinking about Q2"},
            {"role": "user", "content": "Q3"},
        ],
        "add_generation_prompt": True, "preserve_thinking": False,
    }),
    ("assistant_without_reasoning", {
        "messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ],
        "add_generation_prompt": True,
    }),
    ("content_parts_text", {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]}],
        "add_generation_prompt": True,
    }),
    ("empty_user_content_gen", {
        "messages": [{"role": "user", "content": ""}],
        "add_generation_prompt": True,
    }),
    ("tools_no_system", {
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "tools": [weather_tool()], "add_generation_prompt": True,
    }),
    ("tools_with_system", {
        "messages": [{"role": "system", "content": "You are helpful."},
                     {"role": "user", "content": "What's the weather in Paris?"}],
        "tools": [weather_tool()], "add_generation_prompt": True, "reasoning_effort": "low",
    }),
    ("tool_call_and_response", {
        "messages": [
            {"role": "user", "content": "What's the weather in Paris for 3 days?"},
            {"role": "assistant", "content": "", "reasoning_content": "I should call the tool.",
             "tool_calls": [qwen_call("c1", "get_weather", {"city": "Paris", "days": 3})]},
            {"role": "tool", "content": "{\"temp\": 21, \"unit\": \"C\"}"},
        ],
        "tools": [weather_tool()], "add_generation_prompt": True,
    }),
    ("tool_call_with_content_and_two_calls", {
        "messages": [
            {"role": "user", "content": "Weather in Paris and Rome?"},
            {"role": "assistant", "content": "Let me check both.",
             "tool_calls": [qwen_call("c1", "get_weather", {"city": "Paris"}),
                            qwen_call("c2", "get_weather", {"city": "Rome", "days": 2})]},
            {"role": "tool", "content": "Paris: 21C"},
            {"role": "tool", "content": "Rome: 25C"},
            {"role": "assistant", "content": "Paris 21C, Rome 25C."},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()], "add_generation_prompt": True,
    }),
    ("tool_call_multiline_and_nested_args", {
        "messages": [
            {"role": "user", "content": "Run it."},
            {"role": "assistant", "content": "",
             "tool_calls": [qwen_call("c1", "run", {"code": "print(1)\nprint(2)\n", "opts": {"a": 1, "b": [1, 2]}, "flag": True})]},
            {"role": "tool", "content": "1\n2"},
        ],
        "tools": [{"name": "run", "description": "Run code", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}],
        "add_generation_prompt": True,
    }),
    ("tool_response_then_generation", {
        "messages": [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "", "tool_calls": [qwen_call("c1", "get_weather", {"city": "Oslo"})]},
            {"role": "tool", "content": "cold"},
        ],
        "tools": [weather_tool()], "add_generation_prompt": True, "enable_thinking": False,
    }),
    ("no_generation_prompt", {
        "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
        "add_generation_prompt": False,
    }),
]


# The GLM-4.7 template: the GLM-5.3 one without the reasoning
# effort preamble, the deferred tools, the tool references and the media
# reminders; the reasoning split uses rstrip('\n') / lstrip('\n') and
# enable_thinking=false closes the think block in the generation prompt.
# The GLM-5.3 cases that render on it, plus the thinking switch.
GLM4_SKIP = {"effort_low", "effort_high", "multimodal_parts_gen", "tools_deferred",
             "tool_call_outputs_and_references", "tool_reference_direct"}
GLM4_CASES = [c for c in CASES if c[0] not in GLM4_SKIP] + [
    ("thinking_disabled", {
        "messages": [{"role": "user", "content": "Hi"}],
        "add_generation_prompt": True, "enable_thinking": False,
    }),
    ("thinking_disabled_with_system", {
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
        "add_generation_prompt": True, "enable_thinking": False,
    }),
    ("trailing_assistant_think_markers_newlines", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant",
             "content": THINK_OPEN + "\n\nTwo plus two is four.\n\n" + THINK_CLOSE +
                        "\n\nThe answer is 4.\n"},
        ],
        "add_generation_prompt": True,
    }),
    ("mid_turn_think_markers_stripped", {
        "messages": [
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant",
             "content": THINK_OPEN + "\nreasoning\n" + THINK_CLOSE + "\n4"},
            {"role": "user", "content": "And 3+3?"},
        ],
        "add_generation_prompt": True,
    }),
    ("tool_call_two_turns", {
        "messages": [
            {"role": "user", "content": "Weather in Paris, then Tokyo."},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather", {"city": "Paris"})]},
            {"role": "tool", "content": "18C, partly cloudy",
             "tool_call_id": "call_1"},
            {"role": "assistant", "tool_calls": [
                call("call_2", "get_weather", {"city": "Tokyo", "days": 2})]},
            {"role": "tool", "content": "22C, clear",
             "tool_call_id": "call_2"},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
    ("tool_call_outputs_list", {
        "messages": [
            {"role": "user", "content": "Weather in Paris and Tokyo?"},
            {"role": "assistant", "tool_calls": [
                call("call_1", "get_weather", {"city": "Paris"}),
                call("call_2", "get_weather", {"city": "Tokyo"})]},
            {"role": "tool", "content": [
                {"tool_call_id": "call_1", "output": "18C, partly cloudy"},
                {"tool_call_id": "call_2", "output": "22C, clear"},
            ]},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": [weather_tool()],
        "add_generation_prompt": True,
    }),
]


# The MiMo-V2.6-Flash template (2026-09-22): ChatML roles, the tools as
# one JSON line each inside <tools> in a leading system turn (or after a
# message's own `tools`), assistant turns always opening with the
# reasoning in think markers (empty when none), tool calls as
# <tool_call><function=NAME><parameter=K>V</parameter>...</function></tool_call>
# with non-string values JSON-serialized, tool responses as a plain `tool`
# role turn, image / audio / video parts as their pad markers, and
# enable_thinking=false closing the think block in the generation prompt.
MIMO_SKIP = {"effort_low", "effort_medium_system", "multi_turn_preserve_thinking_false"}
MIMO_CASES = [c for c in QWEN_CASES if c[0] not in MIMO_SKIP] + [
    ("mimo_multimodal_parts", {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe "},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": " and "},
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
            {"type": "video", "video": "x.mp4"},
            {"type": "text", "text": "."},
        ]}],
        "add_generation_prompt": True,
    }),
    ("mimo_message_tools", {
        "messages": [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Weather in Oslo?", "tools": [weather_tool()]},
        ],
        "add_generation_prompt": True,
    }),
    ("mimo_message_tools_empty_body", {
        "messages": [{"role": "user", "content": "", "tools": [weather_tool()]}],
        "add_generation_prompt": True,
    }),
    ("mimo_string_arguments", {
        "messages": [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Oslo\"}"}}]},
            {"role": "tool", "content": "cold"},
        ],
        "tools": [weather_tool()], "add_generation_prompt": True,
    }),
    ("mimo_custom_tool_input", {
        "messages": [
            {"role": "user", "content": "Run it."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "custom", "custom": {"name": "shell", "input": "ls -la\n"}}]},
            {"role": "tool", "content": "total 0"},
        ],
        "tools": [{"type": "custom", "custom": {"name": "shell", "description": "Run a shell command"}}],
        "add_generation_prompt": True,
    }),
    ("mimo_reasoning_then_tool_call", {
        "messages": [
            {"role": "user", "content": "Weather in Paris and Rome?"},
            {"role": "assistant", "content": "Checking.", "reasoning_content": "two cities",
             "tool_calls": [qwen_call("c1", "get_weather", {"city": "Paris"}),
                            qwen_call("c2", "get_weather", {"city": "Rome", "days": 2})]},
            {"role": "tool", "content": "Paris: 21C"},
            {"role": "tool", "content": "Rome: 25C"},
        ],
        "tools": [weather_tool()], "add_generation_prompt": True, "enable_thinking": False,
    }),
    ("mimo_chinese_tool_args", {
        "messages": [
            {"role": "user", "content": "巴黎天气？"},
            {"role": "assistant", "content": "", "tool_calls": [qwen_call("c1", "get_weather", {"city": "巴黎", "days": 1})]},
            {"role": "tool", "content": "晴"},
        ],
        "tools": [weather_tool()], "add_generation_prompt": True,
    }),
]


# The Qwen3-Next-80B-A3B-Instruct template: ChatML roles, no reasoning
# block, the tools one JSON line each inside <tools> in the system turn,
# and tool calls in the JSON form —
#   <tool_call>\n{"name": NAME, "arguments": {...}}\n</tool_call>
# one block per call with a newline between blocks — with tool responses
# grouped into one user turn as <tool_response> blocks.
def wrapped(tool):
    return {"type": "function", "function": tool}


RUN_TOOL = {"name": "run", "description": "Run code.",
            "parameters": {"type": "object",
                           "properties": {"code": {"type": "string"}, "opts": {"type": "object"}},
                           "required": ["code"]}}

QWEN3_NEXT_CASES = [
    ("simple_user_gen", {
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "add_generation_prompt": True,
    }),
    ("system_user_gen", {
        "messages": [{"role": "system", "content": "Be brief."},
                     {"role": "user", "content": "写一首关于秋天的诗。"}],
        "add_generation_prompt": True,
    }),
    ("tools_no_system", {
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "tools": [weather_tool()], "add_generation_prompt": True,
    }),
    ("tools_two_calls_and_results", {
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Weather in Paris and Rome? 巴黎呢?"},
            {"role": "assistant", "content": "Let me check both.",
             "tool_calls": [qwen_call("c1", "get_weather", {"city": "Paris", "days": 3}),
                            qwen_call("c2", "get_weather", {"city": "Rome"})]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
            {"role": "tool", "tool_call_id": "c2", "content": "rain"},
            {"role": "assistant", "content": "",
             "tool_calls": [qwen_call("c3", "run", {"code": "print(\"hi\")\nprint(2)\n",
                                                    "opts": {"a": 1, "b": [1, 2]}})]},
            {"role": "tool", "tool_call_id": "c3", "content": "hi\n2"},
            {"role": "assistant", "content": "Paris is sunny; Rome has rain."},
            {"role": "user", "content": "And Zürich?"},
        ],
        "tools": [wrapped(weather_tool()), wrapped(RUN_TOOL)], "add_generation_prompt": True,
    }),
    ("tool_call_no_arguments_then_generation", {
        "messages": [
            {"role": "user", "content": "Ping it."},
            {"role": "assistant", "content": "", "tool_calls": [qwen_call("c1", "ping", {})]},
            {"role": "tool", "tool_call_id": "c1", "content": "pong"},
        ],
        "tools": [wrapped({"name": "ping", "description": "Ping.",
                           "parameters": {"type": "object", "properties": {}}})],
        "add_generation_prompt": True,
    }),
]


# The Gemma 4 template (2026-10-04; nvidia/Gemma-4-31B-IT-NVFP4's
# chat_template.jinja): turns as <|turn>ROLE\n ... <turn|>\n with the
# assistant's role spelled "model"; a system turn that carries <|think|>
# (enable_thinking), the system text and the tool declarations; an EMPTY
# thought channel after the generation prompt when thinking is off; tool
# declarations, calls and responses in the model's own notation —
#   <|tool>declaration:NAME{description:<|"|>..<|"|>,parameters:{..}}<tool|>
#   <|tool_call>call:NAME{key:value,..}<tool_call|>
#   <|tool_response>response:NAME{key:value,..}<tool_response|>
# keys bare and sorted (dictsort: case-insensitive), strings between <|"|>
# tokens, numbers / booleans / null bare, schema types upper-cased — and the
# OpenAI shapes folded into it: role "tool" messages after an assistant's
# tool_calls become that turn's tool responses, a second assistant message
# continues the turn. `bos_token` is a global transformers passes from the
# tokenizer; the engine's frontend passes the same.
def g4(**kw):
    kw.setdefault("bos_token", "<bos>")
    kw.setdefault("add_generation_prompt", True)
    return kw


G4_SEARCH_TOOL = wrapped({
    "name": "search_docs",
    "description": "Search the documentation.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for"},
            "Limit": {"type": "integer"},
            "filters": {"type": "object", "description": "Narrowing",
                        "properties": {"lang": {"type": "string", "enum": ["en", "fr", "de"]},
                                       "after": {"type": "string", "nullable": True}},
                        "required": ["lang"]},
            "tags": {"type": "array", "description": "Any of", "items": {"type": "string"}},
            "ranges": {"type": "array", "items": {"type": "object",
                                                 "properties": {"lo": {"type": "number"}, "hi": {"type": "number"}},
                                                 "required": ["lo"]}},
            "exact": {"type": "boolean", "nullable": True},
            "_internal": {"type": ["string", "null"]},
            "blob": {"type": "object"},
        },
        "required": ["query"],
    },
})

GEMMA4_CASES = [
    ("simple_user_gen", g4(messages=[{"role": "user", "content": "The capital of France is"}])),
    ("simple_user_no_gen", g4(messages=[{"role": "user", "content": "Hello"}], add_generation_prompt=False)),
    ("system_user_gen", g4(messages=[{"role": "system", "content": "  Be brief.\n"},
                                     {"role": "user", "content": "写一首关于秋天的诗。"}])),
    ("developer_role_is_system", g4(messages=[{"role": "developer", "content": "Answer in French."},
                                              {"role": "user", "content": "Hi"}])),
    ("thinking_on_no_system", g4(messages=[{"role": "user", "content": "Why is the sky blue?"}], enable_thinking=True)),
    ("thinking_on_with_system", g4(messages=[{"role": "system", "content": "You are terse."},
                                             {"role": "user", "content": "Why?"}], enable_thinking=True)),
    ("thinking_off_explicit", g4(messages=[{"role": "user", "content": "Why?"}], enable_thinking=False)),
    ("content_parts_text", g4(messages=[
        {"role": "system", "content": [{"type": "text", "text": "Part one. "}, {"type": "text", "text": " Part two."}]},
        {"role": "user", "content": [{"type": "text", "text": "  First.  "}, {"type": "text", "text": "Second."}]}])),
    ("content_parts_image", g4(messages=[
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]}])),
    ("multi_turn", g4(messages=[
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Tell me a joke.\n\n"},
        {"role": "assistant", "content": "Why did the chicken cross the road?"},
        {"role": "user", "content": "Why?"}])),
    ("assistant_thought_is_stripped", g4(messages=[
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "<|channel>thought\nsimple sum<channel|>4"},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "a<|channel>x<channel|>b<|channel>y<channel|> c "},
        {"role": "user", "content": "ok"}])),
    ("consecutive_assistant_messages", g4(messages=[
        {"role": "user", "content": "Go on."},
        {"role": "assistant", "content": "First part."},
        {"role": "assistant", "content": "Second part."},
        {"role": "user", "content": "More."}])),
    ("unicode_whitespace_trim", g4(messages=[
        {"role": "system", "content": "\u3000系统提示\u00a0"},
        {"role": "user", "content": "\u2003text with a trailing thin space\u2009\n\u2028"},
        {"role": "assistant", "content": "\u0085answer\u001f"},
        {"role": "user", "content": "\u200bzero width stays\u200b"}])),
    ("empty_content", g4(messages=[{"role": "user", "content": ""}])),
    ("tools_no_system", g4(messages=[{"role": "user", "content": "What's the weather in Paris?"}],
                           tools=[wrapped(weather_tool())])),
    ("tools_with_system_and_thinking", g4(messages=[{"role": "system", "content": "Use tools."},
                                                    {"role": "user", "content": "Weather in Rome?"}],
                                          tools=[wrapped(weather_tool()), wrapped(RUN_TOOL)], enable_thinking=True)),
    ("tool_schema_shapes", g4(messages=[{"role": "user", "content": "Find it."}], tools=[G4_SEARCH_TOOL])),
    ("tool_without_parameters", g4(messages=[{"role": "user", "content": "Ping."}], tools=[
        wrapped({"name": "ping", "description": "Ping."}),
        wrapped({"name": "now", "description": "The time.", "parameters": {"type": "object", "properties": {}}}),
        wrapped({"name": "typed", "description": "With a response.",
                 "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
                 "response": {"description": "What came back", "type": "object"}})])),
    ("tool_keys_sort_case_insensitively", g4(messages=[{"role": "user", "content": "Sort."}], tools=[wrapped({
        "name": "sorter", "description": "Keys in many cases.",
        "parameters": {"type": "object", "properties": {
            "zeta": {"type": "string"}, "Alpha": {"type": "string"}, "beta": {"type": "string"},
            "BETA2": {"type": "string"}, "_under": {"type": "string"}, "a1": {"type": "string"},
            "A0": {"type": "string"}, "\u00c4rger": {"type": "string"}, "\u00e4hre": {"type": "string"},
            "\u0401\u0436": {"type": "string"}, "\u0451\u043b\u043a\u0430": {"type": "string"},
            "\u0394": {"type": "string"}, "\u03b1": {"type": "string"}, "\u57ce\u5e02": {"type": "string"},
            "\u0160koda": {"type": "string"}, "\u0161ok": {"type": "string"}}}})])),
    ("tool_call_then_response", g4(messages=[
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [qwen_call("c1", "get_weather", {"city": "Paris", "days": 3})]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny, 21C"}],
        tools=[wrapped(weather_tool())])),
    ("tool_call_pending_no_response", g4(messages=[
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [qwen_call("c1", "get_weather", {"city": "Paris"})]}],
        tools=[wrapped(weather_tool())])),
    ("two_calls_two_responses_then_answer", g4(messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Weather in Paris and Rome? 巴黎呢?"},
        {"role": "assistant", "content": "Let me check both.",
         "tool_calls": [qwen_call("c1", "get_weather", {"city": "Paris", "days": 3}),
                        qwen_call("c2", "get_weather", {"city": "Rome"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        {"role": "tool", "tool_call_id": "c2", "content": "rain"},
        {"role": "assistant", "content": "Paris is sunny; Rome has rain."},
        {"role": "user", "content": "And Zürich?"}],
        tools=[wrapped(weather_tool()), wrapped(RUN_TOOL)])),
    ("argument_value_types", g4(messages=[
        {"role": "user", "content": "Run it."},
        {"role": "assistant", "content": "", "tool_calls": [qwen_call("c1", "run", {
            "code": "print(\"hi\")\nprint(2)\n", "Zed": None, "flag": True, "off": False, "n": 7, "ratio": 0.5,
            "big": 12345678901234, "neg": -3, "opts": {"b": [1, 2, "three", {"k": None}], "a": {"x": 1.25, "Y": "y"}},
            "empty_list": [], "empty_map": {}, "text": "quote \" and <|\"|> and 中文"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "hi\n2"}],
        tools=[wrapped(RUN_TOOL)])),
    ("arguments_as_a_string", g4(messages=[
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "tool_calls": [qwen_call("c1", "get_weather", "city:<|\"|>Oslo<|\"|>")]},
        {"role": "tool", "tool_call_id": "c1", "content": "cold"}],
        tools=[wrapped(weather_tool())])),
    ("tool_response_names", g4(messages=[
        {"role": "user", "content": "Do three things."},
        {"role": "assistant", "content": "", "tool_calls": [qwen_call("c1", "alpha", {}), qwen_call("c2", "beta", {"x": 1}),
                                                             qwen_call("c3", "gamma", {})]},
        {"role": "tool", "tool_call_id": "c2", "content": "second first"},
        {"role": "tool", "name": "named", "content": "no id, a name"},
        # (an id that matches no call, on a message without a name, is a TypeError in the template
        # itself — None + str — and a render error here: not a case)
        {"role": "tool", "tool_call_id": "nope", "name": "fallback", "content": "an id nothing matches"},
        {"role": "user", "content": "Thanks."}])),
    ("tool_response_content_parts", g4(messages=[
        {"role": "user", "content": "Read it."},
        {"role": "assistant", "content": "", "tool_calls": [qwen_call("c1", "read", {"path": "/tmp/x"})]},
        {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "line 1\n"},
                                                           {"type": "image"}, {"type": "text", "text": "line 2"}]}])),
    ("reasoning_with_tool_calls", g4(messages=[
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "reasoning_content": "I should call the tool.",
         "tool_calls": [qwen_call("c1", "get_weather", {"city": "Paris"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"}],
        tools=[wrapped(weather_tool())], enable_thinking=True)),
    ("reasoning_before_last_user_is_dropped", g4(messages=[
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "reasoning": "old thought",
         "tool_calls": [qwen_call("c1", "get_weather", {"city": "Paris"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        {"role": "assistant", "content": "It is sunny.", "reasoning_content": "no call here"},
        {"role": "user", "content": "And Rome?"}],
        tools=[wrapped(weather_tool())])),
    ("legacy_tool_responses_on_the_assistant", g4(messages=[
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "tool_calls": [qwen_call("c1", "get_weather", {"city": "Oslo"})],
         "tool_responses": [{"name": "get_weather", "response": {"temp": -2, "sky": "clear", "Wind": {"kph": 12}}},
                            {"response": "plain text"}]},
        {"role": "user", "content": "Thanks"}])),
    ("call_then_text_in_the_same_turn", g4(messages=[
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "tool_calls": [qwen_call("c1", "get_weather", {"city": "Oslo"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "cold"},
        {"role": "assistant", "content": "It is cold in Oslo."},
        {"role": "assistant", "content": "", "tool_calls": [qwen_call("c2", "get_weather", {"city": "Bergen"})]},
        {"role": "tool", "tool_call_id": "c2", "content": "wet"}],
        tools=[wrapped(weather_tool())])),
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--template", default=None)
    ap.add_argument("--tokenizer-json", default=None)
    ap.add_argument("--out", dest="out_opt", default=None)
    # Renders only: the corpus carries no "ids" and its header says so.
    # For a host without the tokenizers package; the gate that reads the
    # corpus compares ids only where a case has them.
    ap.add_argument("--render-only", action="store_true")
    args = ap.parse_args()
    model = args.model
    is_qwen3_next = "Qwen3-Next" in model
    is_qwen = "Qwen" in model and not is_qwen3_next
    is_glm4 = "GLM-4" in model
    is_mimo = "MiMo" in model
    is_gemma = "gemma" in model.lower()
    cases = (GEMMA4_CASES if is_gemma else QWEN3_NEXT_CASES if is_qwen3_next else
             MIMO_CASES if is_mimo else QWEN_CASES if is_qwen else GLM4_CASES if is_glm4 else CASES)
    out_path = args.out_opt or args.out or (
        "tests/data/gemma4_chat_template_goldens.jsonl" if is_gemma else
        "tests/data/qwen3next_chat_template_goldens.jsonl" if is_qwen3_next else
        "tests/data/mimo_chat_template_goldens.jsonl" if is_mimo else
        "tests/data/qwen_chat_template_goldens.jsonl" if is_qwen else
        "tests/data/glm4_chat_template_goldens.jsonl" if is_glm4 else "tests/data/glm_chat_template_goldens.jsonl")
    if args.template and args.tokenizer_json:
        tpl_path, tok_path = args.template, args.tokenizer_json
    else:
        snap = cached_snapshot(model, cache_root(cache_environment()))
        tpl_path = str(snap / "chat_template.jinja")
        tok_path = str(snap / "tokenizer.json")
    tpl_raw = pathlib.Path(tpl_path).read_bytes()
    tok_raw = pathlib.Path(tok_path).read_bytes()
    template_hash = f"{fnv1a64(tpl_raw):016x}"
    tok_hash = f"{fnv1a64(tok_raw):016x}"
    if not is_gemma and not is_qwen3_next and not is_qwen and not is_glm4 and not is_mimo and tok_hash != "700b4469fc43f23b":
        sys.exit(f"unexpected tokenizer revision {tok_hash} — the tokenizer "
                 "goldens are keyed to 700b4469fc43f23b")

    jinja_env = env()
    template = jinja_env.from_string(tpl_raw.decode("utf-8"))
    tok = None
    if not args.render_only:
        import tokenizers
        tok = tokenizers.Tokenizer.from_file(tok_path)

    with open(out_path, "w", encoding="utf-8") as f:
        header = {
            "model": model,
            "jinja2": jinja_env.jinja2.__version__ if hasattr(
                jinja_env, "jinja2") else "3.1.2",
            "template_hash": template_hash,
            "tokenizer_revision": tok_hash,
            "cases": len(cases),
        }
        if tok is None:
            header["ids"] = False
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for name, kwargs in cases:
            rendered = template.render(**kwargs)
            if not rendered:
                sys.exit(f"case {name} rendered empty — generator bug")
            case = {"name": name, "kwargs": kwargs, "render": rendered}
            ids = tok.encode(rendered).ids if tok is not None else []
            if tok is not None:
                case["ids"] = ids
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
            print(f"{name:42s} {len(rendered):5d} chars  {len(ids):4d} ids")
    print(f"wrote {out_path}: {len(cases)} cases, template {template_hash}")


if __name__ == "__main__":
    main()
