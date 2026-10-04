#pragma once

#include <optional>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"

namespace dgpp::serve {

// The chat globals with every message whose content is an array of text
// parts only ({"type":"text","text":…}; an empty array counts) rewritten to
// carry the parts' texts joined by "\n" as one string. nullopt when no
// message has such content: there is nothing to render differently.
//
// OpenAI allows either form, and they mean the same, but the templates do
// not agree on the array: some read the parts, Qwen3-Coder-Next's
// concatenates message.content and throws, Qwen3-Next's renders nothing for
// it (the model then answers an empty message). The service probes the
// template once and renders this form where the template loses the array.
// An array with any other part (image_url) is left as it is: only the
// template can place that.
//
// The strings of the result that were not joined here still view the
// storage `globals` views; keep that alive while the result is in use.
inline std::optional<minijson::Value> flatten_text_content(const minijson::Value& globals) {
  using minijson::Member;
  using minijson::Value;
  const Value* messages = globals.is_object() ? globals.find("messages") : nullptr;
  if (messages == nullptr || !messages->is_array()) return std::nullopt;

  bool changed = false;
  std::vector<Value> msgs;
  msgs.reserve(messages->items().size());
  for (const Value& msg : messages->items()) {
    const Value* content = msg.is_object() ? msg.find("content") : nullptr;
    bool text_only = content != nullptr && content->is_array();
    std::string joined;
    if (text_only) {
      bool first = true;
      for (const Value& part : content->items()) {
        const Value* type = part.is_object() ? part.find("type") : nullptr;
        const Value* text = part.is_object() ? part.find("text") : nullptr;
        if (type == nullptr || !type->is_string() || type->as_string() != "text" ||
            text == nullptr || !text->is_string()) {
          text_only = false;
          break;
        }
        if (!first) joined += '\n';
        joined += text->as_string();
        first = false;
      }
    }
    if (!text_only) {
      msgs.push_back(msg);
      continue;
    }
    std::vector<Member> members;
    members.reserve(msg.members().size());
    for (const Member& m : msg.members())
      members.push_back(m.key == "content" ? Member{"content", Value::make_owned_string(joined)} : m);
    msgs.push_back(Value::make_object(std::move(members)));
    changed = true;
  }
  if (!changed) return std::nullopt;

  std::vector<Member> out;
  out.reserve(globals.members().size());
  for (const Member& m : globals.members())
    out.push_back(m.key == "messages" ? Member{"messages", Value::make_array(msgs)} : m);
  return Value::make_object(std::move(out));
}

}  // namespace dgpp::serve
