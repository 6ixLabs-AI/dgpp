# Qwen required-call grammar golden follow-up

The Qwen chat-template suite failed on master `e5fbbcd` and on PR #62's
base `652cd40`: `tool_call_and_response` rejected token 271 (`\n\n`) at
position 8 in grammar state `top`.

PR #38 (`0f2a87c`) intentionally made Qwen's required/named grammar force
`<tool_call>` when a call is still owed. The golden test continued replaying
the reference template's whitespace and optional prose between `</think>`
and the first call under a required-call spec. An isolated build restoring
only the earlier masking rule accepted all four golden tool-call turns,
confirming the cause. Restoring that rule in production would also restore
the bug where a required call could be deferred indefinitely by prose.

The updated test keeps the reference corpus unchanged and checks both
contracts over the real tokenizer:

- Automatic mode accepts the original rendered turn, including whitespace
  and optional prose before the first call.
- Required mode walks the same reasoning and calls with that intervening
  text removed, matching the forced first opener.
- Required and named masks admit exactly the first tool-call opener both
  without reasoning and immediately after reasoning closes.

The existing checks for undeclared keys, single-call limits, EOS while a
call is owed, and invalid tool names remain. MiMo retains its existing
separate handling. Production grammar, templates, and serving code are
unchanged.

## Validation

Rebuilt the affected host binary and unit suite with warnings treated as
errors, then ran:

```bash
cmake --build build-ci --target chat_template_test unit_tests -j4
ctest --test-dir build-ci -R '^(unit_tests|qwen_chat_template_test|chat_template_test|glm4_chat_template_test|glm_dsa_chat_template_test)$' --output-on-failure -j1
```

All five suites passed in 17.22 seconds with no skips. Linking the updated
test against the isolated pre-PR-38 masking rule fails the new assertion
that required/named mode must force only the first opener. This confirms
the update also guards the enforcement that motivated PR #38.

These are CPU host tests. The existing GLM service remained running;
no GPU or fabric execution was needed for this test-only change.
