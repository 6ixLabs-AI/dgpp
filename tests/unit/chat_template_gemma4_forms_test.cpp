// The Jinja forms the Gemma 4 chat template added to the interpreter
// (text/chat_template.cpp, 2026-10-04): the dictsort / upper / map / list
// filters, the boolean and sequence tests, a ternary without else, the
// block form of set, keyword arguments to a macro, dict.get, Python's
// whitespace set under trim, a for over an undefined value. Every expected
// string below is what jinja2 3.1.6 printed for the same source and
// globals under transformers' environment (trim_blocks, lstrip_blocks);
// the template itself is checked whole in tests/host/gemma4_chat_test.cpp.
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "text/chat_template.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// The globals every case renders with.
const char* kGlobals = R"JSON({
 "d": {"b": 1, "A": 2, "c": {"y": [1, "two", null, true, 2.5]}, "_z": "s", "É": 3, "é2": 4, "Zed": null},
 "l": ["string", "null", "Mixed Case"], "x": "plain text", "n": 42, "f": 0.25,
 "t": true, "fa": false, "nothing": null, "s": "it's \"q\"",
 "ws": "　  padded \u0085\t", "zw": "​ x ​",
 "nested": {"k": ["a'b", "c\"d", "e\\f\n"], "m": {"x": false}}
})JSON";

std::string render(const std::string& source) {
  const std::string globals = kGlobals;
  const auto parsed = dgpp::minijson::parse(globals);
  const auto tpl = dgpp::text::ChatTemplate::compile(source);
  return tpl.render(dgpp::text::Value::from_minijson(parsed.root));
}

void renders(const char* name, const std::string& source, const std::string& want) {
  const std::string got = render(source);
  require(got == want, std::string(name) + ": got '" + got + "', want '" + want + "'");
}

// Compiling or rendering `source` must throw a message holding `needle`.
void refused(const std::string& source, const char* needle) {
  std::string msg;
  try {
    (void)render(source);
  } catch (const std::exception& e) {
    msg = e.what();
  }
  require(msg.find(needle) != std::string::npos,
          "expected a refusal holding '" + std::string(needle) + "' for '" + source + "', got '" + msg + "'");
}

}  // namespace

DGPP_TEST(chat_template_dictsort_is_case_insensitive) {
  // Keys lowered, then compared by codepoint: "_z" < "a" < "b" < "c" < "zed" < "é" < "é2".
  renders("dictsort", "{% for k, v in d | dictsort %}{{ k }};{% endfor %}", "_z;A;b;c;Zed;É;é2;");
  // The map itself keeps its member order.
  renders("list of a map", "{% for k in d | list %}{{ k }},{% endfor %}", "b,A,c,_z,É,é2,Zed,");
  refused("{{ d | dictsort(true) }}", "dictsort");
  refused("{% for k, v in l | dictsort %}{% endfor %}", "dictsort");
}

DGPP_TEST(chat_template_upper_is_pythons_str_then_upper) {
  renders("upper",
          "{{ 'abc' | upper }}|{{ x | upper }}|{{ missing | upper }}|{{ n | upper }}|{{ f | upper }}|{{ l | upper }}|"
          "{{ nothing | upper }}|{{ t | upper }}|{{ s | upper }}",
          "ABC|PLAIN TEXT||42|0.25|['STRING', 'NULL', 'MIXED CASE']|NONE|TRUE|IT'S \"Q\"");
  // Containers print as Python prints them: elements by repr (the quote that avoids escaping,
  // backslash and newline escaped), None / True / False.
  renders("upper of containers", "{{ nested | upper }}|{{ d.c | upper }}",
          "{'K': [\"A'B\", 'C\"D', 'E\\\\F\\N'], 'M': {'X': FALSE}}|{'Y': [1, 'TWO', NONE, TRUE, 2.5]}");
  renders("dictsort values", "{% for k, v in nested | dictsort %}{{ k }}={{ v | upper }};{% endfor %}",
          "k=[\"A'B\", 'C\"D', 'E\\\\F\\N'];m={'X': FALSE};");
  // Python's upper() maps every cased script; this one refuses what it does not map.
  refused("{{ 'stra\xC3\x9F" "e' | upper }}", "non-ASCII");
}

DGPP_TEST(chat_template_map_and_list) {
  renders("map", "{{ l | map('upper') | list | length }}:{% for u in l | map('upper') | list %}{{ u }},{% endfor %}",
          "3:STRING,NULL,MIXED CASE,");
  refused("{{ l | map('reverse') | list }}", "map");
  refused("{{ l | map(attribute='x') | list }}", "map");
}

DGPP_TEST(chat_template_boolean_and_sequence_tests) {
  // sequence: strings, lists, maps — and an undefined value (Jinja's lenient Undefined has a
  // length and an item access); never a number, None or a boolean.
  renders("tests",
          "{{ t is boolean }}{{ n is boolean }}{{ nothing is boolean }}|{{ x is sequence }}{{ l is sequence }}"
          "{{ d is sequence }}{{ n is sequence }}{{ nothing is sequence }}{{ missing is sequence }}{{ t is sequence }}|"
          "{{ l is not sequence }}",
          "TrueFalseFalse|TrueTrueTrueFalseFalseTrueFalse|False");
}

DGPP_TEST(chat_template_ternary_without_else) {
  renders("ternary", "[{{ 'a' if t }}][{{ 'b' if fa }}][{{ ',' if not fa }}][{{ ('x' if fa) is defined }}]",
          "[a][][,][False]");
}

DGPP_TEST(chat_template_block_set) {
  renders("block set", "{% set cap %}x{{ n }}y{% endset %}[{{ cap }}][{{ cap | length }}]", "[x42y][4]");
  renders("block set, trimmed", "{% set cap -%}\n   padded {{ x }}  \n{%- endset %}[{{ cap }}]", "[padded plain text]");
  // The body is a scope of its own: a plain set inside does not outlive it, a namespace attribute does.
  renders("block set scope",
          "{% set z = 1 %}{% set ns = namespace(v=0) %}{% set cap %}{% set z = 2 %}{% set ns.v = 5 %}{{ z }}{% endset %}"
          "{{ z }}{{ cap }}{{ ns.v }}",
          "125");
  renders("block set in a loop", "{% for i in [1, 2] %}{% set cap %}<{{ i }}>{% endset %}{{ cap }}{{ cap }}{% endfor %}",
          "<1><1><2><2>");
  refused("{% set cap %}x", "endset");
  refused("{% set ns.v %}x{% endset %}", "block form");
}

DGPP_TEST(chat_template_macro_keyword_arguments) {
  renders("kwargs",
          "{% macro m(a, b=2, c=3) %}{{ a }}-{{ b }}-{{ c }}{% endmacro %}{{ m(1) }} {{ m(1, c=9) }} {{ m(1, 5, c=7) }} "
          "{{ m(a=4) }} {{ m(c=1, a=2, b=3) }}",
          "1-2-3 1-2-9 1-5-7 4-2-3 2-3-1");
  refused("{% macro m(a) %}{{ a }}{% endmacro %}{{ m(1, zz=2) }}", "unknown argument 'zz'");
  refused("{% macro m(a) %}{{ a }}{% endmacro %}{{ m(1, a=2) }}", "given twice");
  refused("{% macro m(a, b) %}{{ a }}{% endmacro %}{{ m(b=2) }}", "missing argument 'a'");
}

DGPP_TEST(chat_template_dict_get) {
  // Python's dict.get: None (not undefined) for an absent key, so `default` does not apply to it.
  renders("get",
          "{{ d.get('b') }}|{{ d.get('zz') }}|{{ d.get('zz', 'dflt') }}|{{ d.get('zz') is none }}|{{ d.get('Zed') or 'x' }}|"
          "{{ d.get('zz') == d.get('yy') }}|{{ d.get('zz') | default('u') }}",
          "1|None|dflt|True|x|True|None");
  refused("{{ d.get() }}", "get()");
  refused("{{ x.get('a') }}", "non-callable");
}

DGPP_TEST(chat_template_trim_is_pythons_whitespace) {
  // str.strip(): the ideographic and no-break spaces, the line separator and NEL go; a zero-width
  // space is not whitespace and stays.
  renders("trim", "[{{ ws | trim }}][{{ '  a b  ' | trim }}][{{ zw | trim }}]",
          "[padded][a b][\xE2\x80\x8B x \xE2\x80\x8B]");
  // Of a non-string, its Python str() (Jinja's soft_str): the earlier Gemma 4 template trims a
  // system message's content even when it is a list of parts.
  renders("trim of non-strings", "[{{ l | trim }}][{{ n | trim }}][{{ missing | trim }}][{{ nothing | trim }}][{{ d.c | trim }}]",
          "[['string', 'null', 'Mixed Case']][42][][None][{'y': [1, 'two', None, True, 2.5]}]");
  refused("{{ x | trim('p') }}", "trim");
}

DGPP_TEST(chat_template_for_over_undefined_is_empty) {
  renders("for", "[{% for q in missing %}x{% endfor %}][{% for q in d.nope %}y{% endfor %}]", "[][]");
  // None is not iterable in Python either.
  refused("{% for q in nothing %}x{% endfor %}", "iterate");
}
