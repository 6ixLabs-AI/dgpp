#include "text/dsv4_prompt.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "text/chat_template.hpp"

namespace dgpp::text {
namespace {

using dgpp::minijson::Member;

[[noreturn]] void refuse(const std::string& what) { throw std::runtime_error("dsv4 prompt: " + what); }

// The reference's REASONING_EFFORT_PROMPTS ("low" is the empty one).
constexpr const char* kEffortHigh =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, "
    "rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered "
    "alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";
constexpr const char* kEffortMax =
    "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
    "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose "
    "the problem into its most fundamental components, trace every causal chain to its root, and resolve the "
    "underlying cause rather than any surface symptom.\n"
    "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain "
    "that no assumption remains unchecked and no error remains undiscovered.\n\n";
constexpr const char* kResponseFormatHead =
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n";

struct Param {
  std::string key;
  std::string value;  // the raw string, or the JSON text
  bool is_string = false;
};

struct ToolCall {
  std::string id;
  std::string name;
  std::vector<Param> params;
};

struct Block {
  bool tool_result = false;
  std::string text;         // a text block, or the tool result's content
  std::string tool_use_id;  // tool_result
};

struct Msg {
  std::string role;
  std::string content;        // system / developer / latest_reminder / assistant
  std::vector<Block> blocks;  // user turns (the reference's content_blocks)
  std::string reasoning;
  std::vector<ToolCall> calls;
  bool has_tools = false;
  std::string tool_schemas;  // one JSON line per tool
  bool has_response_format = false;
  std::string response_format;  // the JSON text
  bool has_task = false;
  std::string task;
  bool wo_eos = false;
};

// Python truthiness of a JSON value (`if tools:`, `content or ""`).
bool truthy(const minijson::Value* v) {
  if (v == nullptr) return false;
  switch (v->kind()) {
    case minijson::Value::Kind::Null: return false;
    case minijson::Value::Kind::Bool: return v->as_bool();
    case minijson::Value::Kind::Int: return v->as_int() != 0;
    case minijson::Value::Kind::Double: return v->as_double() != 0.0;
    case minijson::Value::Kind::String: return !v->as_string().empty();
    case minijson::Value::Kind::Array: return !v->items().empty();
    case minijson::Value::Kind::Object: return !v->members().empty();
  }
  return false;
}

// A JSON value as the reference holds it — a Python dict keeps one entry
// per key (a repeated key's last value, at its first position) — for the
// reference's to_json (json.dumps, ensure_ascii=False).
Value py_value(const minijson::Value& v) {
  if (v.is_object()) {
    Value::Members members;
    for (const Member& m : v.members()) {
      Value item = py_value(m.value);
      const auto it = std::find_if(members.begin(), members.end(), [&](const auto& kv) { return kv.first == m.key; });
      if (it != members.end()) it->second = std::move(item);
      else members.emplace_back(m.key, std::move(item));
    }
    return Value::map_value(std::move(members));
  }
  if (v.is_array()) {
    std::vector<Value> items;
    for (const minijson::Value& item : v.items()) items.push_back(py_value(item));
    return Value::list_value(std::move(items));
  }
  return Value::from_minijson(v);
}

std::string json_of(const minijson::Value& v) { return py_value(v).to_json(/*ensure_ascii=*/false); }

// The text of a system / user / assistant / developer content field: a
// string, or the "\n\n"-joined text parts of a content-part array (the
// reference renders strings only; the service passes OpenAI's part arrays
// through — a non-text part is refused, the API serves text).
std::string content_text(const minijson::Value& content, const std::string& where) {
  if (content.is_string()) return std::string(content.as_string());
  if (!content.is_array()) refuse(where + ".content must be a string or an array of content parts");
  std::string out;
  for (size_t i = 0; i < content.items().size(); ++i) {
    const minijson::Value& part = content.items()[i];
    const std::string at = where + ".content[" + std::to_string(i) + "]";
    if (!part.is_object()) refuse(at + " must be an object");
    const minijson::Value* type = part.find("type");
    if (!type || !type->is_string() || type->as_string() != "text")
      refuse(at + ": only text content parts are rendered (this model's encoder is text only)");
    const minijson::Value* text = part.find("text");
    if (!text || !text->is_string()) refuse(at + ".text must be a string");
    if (i) out += "\n\n";
    out += std::string(text->as_string());
  }
  return out;
}

// A tool result's content: a string, or the reference's list form — the
// text parts and "[Unsupported TYPE]" for every other part, "\n\n"-joined.
std::string tool_result_text(const minijson::Value& content, const std::string& where) {
  if (content.is_string()) return std::string(content.as_string());
  if (!content.is_array()) refuse(where + ".content must be a string or an array of content parts");
  std::string out;
  for (size_t i = 0; i < content.items().size(); ++i) {
    const minijson::Value& part = content.items()[i];
    const std::string at = where + ".content[" + std::to_string(i) + "]";
    if (!part.is_object()) refuse(at + " must be an object");
    if (i) out += "\n\n";
    const minijson::Value* type = part.find("type");
    if (type && type->is_string() && type->as_string() == "text") {
      const minijson::Value* text = part.find("text");
      if (text == nullptr) continue;  // the reference's block.get("text", "")
      if (!text->is_string()) refuse(at + ".text must be a string");
      out += std::string(text->as_string());
    } else if (type == nullptr || type->is_null()) {
      out += "[Unsupported None]";
    } else if (type->is_string()) {
      out += "[Unsupported " + std::string(type->as_string()) + "]";
    } else {
      refuse(at + ".type must be a string");
    }
  }
  return out;
}

// The "\n"-joined schema lines of a tools array: each entry's `function`
// object as given (the reference's tools_from_openai_format + to_json); a
// flat entry — the service accepts {name, description, parameters} without
// the wrapper — is that object less its "type".
std::string tool_schema_lines(const minijson::Value& tools, const std::string& where) {
  if (!tools.is_array()) refuse(where + " must be an array");
  std::string out;
  for (size_t i = 0; i < tools.items().size(); ++i) {
    const minijson::Value& tool = tools.items()[i];
    const std::string at = where + "[" + std::to_string(i) + "]";
    if (!tool.is_object()) refuse(at + " must be an object");
    if (i) out += "\n";
    if (const minijson::Value* fn = tool.find("function")) {
      out += json_of(*fn);
      continue;
    }
    if (tool.find("name") == nullptr) refuse(at + ".function is required");
    std::vector<Member> flat;
    for (const Member& m : tool.members())
      if (m.key != "type") flat.push_back(m);
    out += json_of(minijson::Value::make_object(std::move(flat)));
  }
  return out;
}

// The reference's TOOLS_TEMPLATE.
std::string render_tools(const std::string& schemas) {
  const std::string D = Dsv4Prompt::kDsml;
  std::string out;
  out += "## Tools\n\n";
  out += "You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<" + D +
         "tool_calls>\" block like the following:\n\n";
  out += "<" + D + "tool_calls>\n";
  out += "<" + D + "invoke name=\"$TOOL_NAME\">\n";
  out += "<" + D + "parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</" + D + "parameter>\n";
  out += "...\n";
  out += "</" + D + "invoke>\n";
  out += "<" + D + "invoke name=\"$TOOL_NAME2\">\n";
  out += "...\n";
  out += "</" + D + "invoke>\n";
  out += "</" + D + "tool_calls>\n\n";
  out += "String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, "
         "arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n";
  out += "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside "
         "<think>...</think> BEFORE any tool calls or final response.\n\n";
  out += "Otherwise, output directly after </think> with tool calls or final response.\n\n";
  out += "### Available Tool Schemas\n\n";
  out += schemas;
  out += "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";
  return out;
}

// The parameters of a JSON object's members (the reference's
// encode_arguments_to_dsml over the parsed arguments): a string raw, any
// other value as JSON; a repeated key keeps its last value at its first
// position, as the parsed dict does.
void params_of_object(const minijson::Value& obj, std::vector<Param>* params) {
  for (const Member& m : obj.members()) {
    Param p;
    p.key = m.key;
    if (m.value.is_string()) {
      p.is_string = true;
      p.value = std::string(m.value.as_string());
    } else {
      p.value = json_of(m.value);
    }
    const auto it = std::find_if(params->begin(), params->end(), [&](const Param& q) { return q.key == p.key; });
    if (it != params->end()) *it = std::move(p);
    else params->push_back(std::move(p));
  }
}

// An assistant tool call off the OpenAI form. The arguments: the service's
// parsed object (its members are the parameters), or the wire's string as
// the reference reads it — a JSON object's members; a string that is not
// JSON is one parameter named "arguments"; JSON that is not an object is
// refused (the reference raises on it).
ToolCall parse_tool_call(const minijson::Value& tc, const std::string& where) {
  if (!tc.is_object()) refuse(where + " must be an object");
  ToolCall call;
  const minijson::Value* fn = tc.find("function");
  if (!fn || !fn->is_object()) refuse(where + ".function must be an object");
  // tc.get("id") or tc.get("function", {}).get("id", "")
  if (const minijson::Value* id = tc.find("id"); truthy(id)) {
    if (!id->is_string()) refuse(where + ".id must be a string");
    call.id = std::string(id->as_string());
  } else if (const minijson::Value* fid = fn->find("id"); fid && !fid->is_null()) {
    if (!fid->is_string()) refuse(where + ".function.id must be a string");
    call.id = std::string(fid->as_string());
  }
  const minijson::Value* name = fn->find("name");
  if (!name || !name->is_string()) refuse(where + ".function.name must be a string");
  call.name = std::string(name->as_string());
  const minijson::Value* args = fn->find("arguments");
  if (args == nullptr) refuse(where + ".function.arguments is required");
  if (args->is_object()) {
    params_of_object(*args, &call.params);
  } else if (args->is_string()) {
    const std::string text(args->as_string());
    minijson::ParseResult parsed;
    bool is_json = false;
    try {
      parsed = minijson::parse(text);
      size_t rest = parsed.consumed;
      while (rest < text.size() && (text[rest] == ' ' || text[rest] == '\n' || text[rest] == '\r' || text[rest] == '\t'))
        ++rest;
      is_json = rest == text.size();
    } catch (const std::exception&) {
    }
    if (is_json) {
      if (!parsed.root.is_object())
        refuse(where + ".function.arguments must be a JSON object (or text that is not JSON)");
      params_of_object(parsed.root, &call.params);
    } else {
      Param p;
      p.key = "arguments";
      p.is_string = true;
      p.value = text;
      call.params.push_back(std::move(p));
    }
  } else {
    // Not a string: the reference's json.loads raises and the value rides
    // whole under "arguments".
    Param p;
    p.key = "arguments";
    p.value = json_of(*args);
    call.params.push_back(std::move(p));
  }
  return call;
}

std::string render_call(const ToolCall& c) {
  const std::string D = Dsv4Prompt::kDsml;
  std::string lines;
  for (size_t i = 0; i < c.params.size(); ++i) {
    if (i) lines += "\n";
    lines += "<" + D + "parameter name=\"" + c.params[i].key + "\" string=\"" + (c.params[i].is_string ? "true" : "false") +
             "\">" + c.params[i].value + "</" + D + "parameter>";
  }
  return "<" + D + "invoke name=\"" + c.name + "\">\n" + lines + "\n</" + D + "invoke>";
}

// The reference's find_last_user_index: the last user / developer message.
int find_last_user_index(const std::vector<Msg>& msgs) {
  for (int i = static_cast<int>(msgs.size()) - 1; i >= 0; --i)
    if (msgs[static_cast<size_t>(i)].role == "user" || msgs[static_cast<size_t>(i)].role == "developer") return i;
  return -1;
}

// The effort preamble: the reference's three levels, and the service's
// other strings folded onto them (the header comment's mapping).
const char* effort_prompt(const minijson::Value* v) {
  if (v == nullptr || v->is_null()) return "";  // the reference's default, "low"
  if (!v->is_string()) refuse("reasoning_effort must be a string (low, high or max)");
  const std::string_view s = v->as_string();
  if (s == "low" || s == "minimal" || s == "medium") return "";
  if (s == "high") return kEffortHigh;
  if (s == "max" || s == "xhigh") return kEffortMax;
  refuse("reasoning_effort must be low, high or max (minimal and medium render as low, xhigh as max), got '" +
         std::string(s) + "'");
}

const char* task_token(const std::string& task, const std::string& where) {
  if (task == "action") return "<｜action｜>";
  if (task == "query") return "<｜query｜>";
  if (task == "authority") return "<｜authority｜>";
  if (task == "domain") return "<｜domain｜>";
  if (task == "title") return "<｜title｜>";
  if (task == "read_url") return "<｜read_url｜>";
  refuse(where + ".task '" + task + "' is not one of action, query, authority, domain, title, read_url");
}

bool bool_knob(const minijson::Value& globals, const char* name, bool* out) {
  const minijson::Value* v = globals.find(name);
  if (v == nullptr || v->is_null()) return false;
  if (!v->is_bool()) refuse(std::string(name) + " must be a boolean");
  *out = v->as_bool();
  return true;
}

}  // namespace

bool Dsv4Prompt::reads(std::string_view name) {
  // `thinking`: the key the vLLM DeepSeek templates and their clients use
  // for enable_thinking (the alias Dsv41Prompt reads too). `drop_thinking`:
  // the reference's own name for the service's clear_thinking.
  return name == "enable_thinking" || name == "thinking" || name == "clear_thinking" || name == "drop_thinking" ||
         name == "reasoning_effort";
}

uint64_t Dsv4Prompt::source_hash() { return 0x647376342d2d3031ull; }  // "dsv4--01": the renderer's first revision

std::string Dsv4Prompt::render(const minijson::Value& globals) {
  if (!globals.is_object()) refuse("globals must be an object");
  const minijson::Value* messages = globals.find("messages");
  if (!messages || !messages->is_array() || messages->items().empty())
    refuse("messages is required and must be a non-empty array");
  const minijson::Value* tools = globals.find("tools");
  if (tools && !tools->is_null() && !tools->is_array()) refuse("tools must be an array");
  if (tools && !truthy(tools)) tools = nullptr;
  bool thinking = true, drop_thinking = true;
  {
    bool a = true, b = true;
    const bool has_a = bool_knob(globals, "enable_thinking", &a);
    const bool has_b = bool_knob(globals, "thinking", &b);
    if (has_a && has_b && a != b) refuse("thinking and enable_thinking disagree");
    if (has_a) thinking = a;
    if (has_b) thinking = b;
  }
  {
    bool a = true, b = true;
    const bool has_a = bool_knob(globals, "clear_thinking", &a);
    const bool has_b = bool_knob(globals, "drop_thinking", &b);
    if (has_a && has_b && a != b) refuse("drop_thinking and clear_thinking disagree");
    if (has_a) drop_thinking = a;
    if (has_b) drop_thinking = b;
  }
  const char* effort = effort_prompt(globals.find("reasoning_effort"));

  // ---- the messages in the reference's shape --------------------------------
  std::vector<Msg> raw;
  for (size_t i = 0; i < messages->items().size(); ++i) {
    const minijson::Value& m = messages->items()[i];
    const std::string where = "messages[" + std::to_string(i) + "]";
    if (!m.is_object()) refuse(where + " must be an object");
    const minijson::Value* role = m.find("role");
    if (!role || !role->is_string()) refuse(where + ".role must be a string");
    Msg msg;
    msg.role = std::string(role->as_string());
    const minijson::Value* content = m.find("content");
    if (msg.role == "system" || msg.role == "assistant") {
      if (truthy(content)) msg.content = content_text(*content, where);  // `content or ""`
    } else if (msg.role == "developer") {
      if (!truthy(content)) refuse(where + ".content is required for a developer message");
      msg.content = content_text(*content, where);
    } else if (msg.role == "latest_reminder") {
      if (content == nullptr || content->is_null()) refuse(where + ".content is required for a latest_reminder message");
      msg.content = content_text(*content, where);
    } else if (msg.role == "user") {
      Block b;
      if (content != nullptr) {  // msg.get("content", "")
        if (content->is_null()) refuse(where + ".content must be a string or an array of content parts");
        b.text = content_text(*content, where);
      }
      msg.blocks.push_back(std::move(b));
    } else if (msg.role == "tool") {
      Block b;
      b.tool_result = true;
      if (content != nullptr) {
        if (content->is_null()) refuse(where + ".content must be a string or an array of content parts");
        b.text = tool_result_text(*content, where);
      }
      if (const minijson::Value* id = m.find("tool_call_id"); id && !id->is_null()) {
        if (!id->is_string()) refuse(where + ".tool_call_id must be a string");
        b.tool_use_id = std::string(id->as_string());
      }
      msg.blocks.push_back(std::move(b));
    } else {
      refuse(where + ": unknown role '" + msg.role + "' (system, user, assistant, tool, developer, latest_reminder)");
    }
    if (msg.role == "assistant") {
      if (const minijson::Value* rc = m.find("reasoning_content"); truthy(rc)) {  // `reasoning_content or ""`
        if (!rc->is_string()) refuse(where + ".reasoning_content must be a string");
        msg.reasoning = std::string(rc->as_string());
      }
      if (const minijson::Value* calls = m.find("tool_calls"); truthy(calls)) {
        if (!calls->is_array()) refuse(where + ".tool_calls must be an array");
        for (size_t k = 0; k < calls->items().size(); ++k)
          msg.calls.push_back(parse_tool_call(calls->items()[k], where + ".tool_calls[" + std::to_string(k) + "]"));
      }
      msg.wo_eos = truthy(m.find("wo_eos"));
    }
    // The fields the reference's merge keeps: a user or tool message
    // becomes a fresh user message that carries the task alone; every other
    // role is copied whole (its tools count towards keeping the reasoning
    // even where the role does not render them).
    if (msg.role != "user" && msg.role != "tool") {
      if (const minijson::Value* t = m.find("tools"); truthy(t)) {
        msg.has_tools = true;
        msg.tool_schemas = tool_schema_lines(*t, where + ".tools");
      }
      if (const minijson::Value* rf = m.find("response_format"); truthy(rf)) {
        msg.has_response_format = true;
        msg.response_format = json_of(*rf);
      }
    }
    if (msg.role != "tool") {
      if (const minijson::Value* task = m.find("task"); task && !task->is_null()) {
        if (!task->is_string()) refuse(where + ".task must be a string");
        msg.has_task = true;
        msg.task = std::string(task->as_string());
        (void)task_token(msg.task, where);
      }
    }
    raw.push_back(std::move(msg));
  }
  // The tools ride the first message when it is a system message, or an
  // empty system message in front.
  if (tools) {
    if (raw[0].role != "system") {
      Msg sys;
      sys.role = "system";
      raw.insert(raw.begin(), std::move(sys));
    }
    raw[0].has_tools = true;
    raw[0].tool_schemas = tool_schema_lines(*tools, "tools");
  }

  // ---- merge_tool_messages ------------------------------------------------------
  std::vector<Msg> merged;
  for (Msg& m : raw) {
    const bool into_user = !merged.empty() && merged.back().role == "user";
    if (m.role == "tool") {
      if (into_user) {
        merged.back().blocks.push_back(std::move(m.blocks[0]));
      } else {
        m.role = "user";
        merged.push_back(std::move(m));
      }
    } else if (m.role == "user") {
      // A user turn that carries a task closes: the next user message opens
      // its own turn (its own task rides only when it opens one).
      if (into_user && !merged.back().has_task) merged.back().blocks.push_back(std::move(m.blocks[0]));
      else merged.push_back(std::move(m));
    } else {
      merged.push_back(std::move(m));
    }
  }
  // ---- sort_tool_results_by_call_order --------------------------------------------
  {
    std::vector<std::pair<std::string, int>> order;  // the last calling assistant's ids
    const auto rank = [&](const Block& b) {
      for (const auto& [id, idx] : order)
        if (id == b.tool_use_id) return idx;
      return 0;
    };
    for (Msg& m : merged) {
      if (m.role == "assistant" && !m.calls.empty()) {
        order.clear();
        for (size_t i = 0; i < m.calls.size(); ++i) {
          if (m.calls[i].id.empty()) continue;
          const auto it = std::find_if(order.begin(), order.end(), [&](const auto& e) { return e.first == m.calls[i].id; });
          if (it != order.end()) it->second = static_cast<int>(i);
          else order.emplace_back(m.calls[i].id, static_cast<int>(i));
        }
      } else if (m.role == "user") {
        std::vector<Block> sorted;
        for (const Block& b : m.blocks)
          if (b.tool_result) sorted.push_back(b);
        if (sorted.size() > 1 && !order.empty()) {
          std::stable_sort(sorted.begin(), sorted.end(), [&](const Block& a, const Block& b) { return rank(a) < rank(b); });
          size_t k = 0;
          for (Block& b : m.blocks)
            if (b.tool_result) b = sorted[k++];
        }
      }
    }
  }
  // ---- drop_thinking --------------------------------------------------------------
  bool any_tools = false;
  for (const Msg& m : merged)
    if (m.has_tools) any_tools = true;
  const bool effective_drop = drop_thinking && !any_tools;
  std::vector<Msg> msgs;
  if (thinking && effective_drop) {
    // _drop_thinking_messages: before the last user turn an assistant
    // message loses its reasoning and a developer message goes entirely.
    const int last_user = find_last_user_index(merged);
    for (size_t i = 0; i < merged.size(); ++i) {
      Msg& m = merged[i];
      if (m.role == "user" || m.role == "system" || m.role == "latest_reminder" || static_cast<int>(i) >= last_user) {
        msgs.push_back(std::move(m));
      } else if (m.role == "assistant") {
        m.reasoning.clear();
        msgs.push_back(std::move(m));
      }
    }
  } else {
    msgs = std::move(merged);
  }

  // ---- render ---------------------------------------------------------------------
  const int last_user = find_last_user_index(msgs);
  const std::string D = kDsml;
  std::string prompt = kBos;
  for (size_t index = 0; index < msgs.size(); ++index) {
    const Msg& m = msgs[index];
    std::string out;
    // The effort preamble: at the very start, in thinking mode.
    if (index == 0 && thinking) out += effort;
    const auto append_tools_and_format = [&] {
      if (m.has_tools) out += "\n\n" + render_tools(m.tool_schemas);
      if (m.has_response_format) out += std::string("\n\n") + kResponseFormatHead + m.response_format;
    };
    if (m.role == "system") {
      out += m.content;
      append_tools_and_format();
    } else if (m.role == "developer") {
      out += kUser;
      out += m.content;
      append_tools_and_format();
    } else if (m.role == "user") {
      out += kUser;
      for (size_t b = 0; b < m.blocks.size(); ++b) {
        if (b) out += "\n\n";
        out += m.blocks[b].tool_result ? "<tool_result>" + m.blocks[b].text + "</tool_result>" : m.blocks[b].text;
      }
    } else if (m.role == "latest_reminder") {
      out += kLatestReminder;
      out += m.content;
    } else if (m.role == "assistant") {
      std::string tc;
      if (!m.calls.empty()) {
        std::string calls;
        for (size_t i = 0; i < m.calls.size(); ++i) {
          if (i) calls += "\n";
          calls += render_call(m.calls[i]);
        }
        tc = "\n\n<" + D + "tool_calls>\n" + calls + "\n</" + D + "tool_calls>";
      }
      // The reply to a quick-instruction task carries no reasoning.
      const bool prev_has_task = index > 0 && msgs[index - 1].has_task;
      std::string thinking_part;
      if (thinking && !prev_has_task && (!effective_drop || static_cast<int>(index) > last_user))
        thinking_part = m.reasoning + kThinkClose;
      out += thinking_part + m.content + tc;
      if (!m.wo_eos) out += kEos;
    } else {
      refuse("unknown role after preprocessing: " + m.role);
    }
    // The transition: nothing when another turn of the prompt follows; a
    // task's token; else the generation header after a user / developer
    // message (not after a system message — the reference writes none).
    const bool followed_by_other =
        index + 1 < msgs.size() && msgs[index + 1].role != "assistant" && msgs[index + 1].role != "latest_reminder";
    if (!followed_by_other) {
      if (m.has_task) {
        const char* token = task_token(m.task, "messages[]");
        if (m.task != "action") {
          out += token;
        } else {
          out += kAssistant;
          out += thinking ? kThinkOpen : kThinkClose;
          out += token;
        }
      } else if (m.role == "user" || m.role == "developer") {
        out += kAssistant;
        if (!effective_drop && thinking) out += kThinkOpen;
        else if (effective_drop && thinking && static_cast<int>(index) >= last_user) out += kThinkOpen;
        else out += kThinkClose;
      }
    }
    prompt += out;
  }
  return prompt;
}

}  // namespace dgpp::text
