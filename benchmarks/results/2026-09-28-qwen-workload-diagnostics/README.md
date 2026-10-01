# Qwen workload diagnostics captured through 2026-09-28

These files were left untracked in the primary checkout and are preserved
here without changing their bytes. They are operational evidence from a
single-node Qwen NVFP4 workload, separate from the W4A4 opt-in policy change.

The [diagnostic archive](dgpp-diagnostics-scrubbed.zip) contains rank-0 logs,
a redacted operation stream, its manifest and an export README. The export
identifies source `ac3b7246b17c3736c3db9c64739589d7263b2997`, model
`nvidia/Qwen3.8-Flash-Next-NVFP4`, five slots, MTP depth 3, a 397312-token
shared BF16 pool and a 6-GiB prefix budget. Its capture runs through
2026-09-28T03:03:15Z. See the preserved [export provenance](export-provenance.md)
for the collection method and redactions.

The [dashboard image](dgpp.webp) shows aggregate input/output tokens,
prefix reuse, admissions and wall-clock versus decode-active throughput.
It was stored alongside the archive; its exact time window has not been
independently matched to the exported logs.

At check-in, the archive's two payloads matched every manifest length,
line count and SHA256. All 453676 token operations retained the explicit
redaction marker, and every request identifier matched the redacted alias
format. A credential-pattern scan found no private-key blocks, Hugging Face
or GitHub tokens, or bearer credentials. The [SHA256SUMS](SHA256SUMS) identify
the original archive and image.

The token values were removed before this archive was supplied, so it
cannot support exact generation replay or a W4A4/W4A16 quality comparison.
No specific issue reproduction or root cause is established by preserving
these files. Keep future paired quality results with the issue #68 record.
