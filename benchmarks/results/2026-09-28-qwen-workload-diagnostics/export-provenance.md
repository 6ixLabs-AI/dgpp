# DGPP rank-0 diagnostic export — scrubbed for developer review

These are diagnostic copies, not untouched originals and not an exact token replay.

## Provenance

- Captured through 2026-09-28T03:03:15Z (September27 local evening).
- Serving source: ac3b7246b17c3736c3db9c64739589d7263b2997, stock source.
- Model: nvidia/Qwen3.8-Flash-Next-NVFP4, single GB10 / world_size1.
- Selected runtime: five slots, MTP3, shared BF16 token pool397312, prefix budget6GiB/55snapshots, busy/idle prefill budgets256/2048, native per-request ceiling262144.
- `serve_r0.log` is an export of retained rank-0 Docker stdout/stderr. This deployment does not create a native file with that name. Docker retention may limit the available history; this export is not claimed to reconstruct rotated-away logs.
- `serve_rank0.ops` is a fixed-size-prefix snapshot of the actual `/build/serve_rank0.ops` inside the running container. The server was not stopped or flushed for capture. Its capture boundary need not exactly coincide with the text log cutoff.

## Redactions and retained information

- Original request IDs are replaced by stable aliases, consistently across both files. The alias mapping is not included.
- Generated token values are removed from every `T` operation and from text-log `first token` fields. Token values can otherwise be decoded into generated text/tool calls. The literal `TOKEN_REDACTED` marker (inside angle brackets) is a redaction, not a replacement token ID.
- The `.ops` file retains all captured records in order, including token-event counts, step numbers, retirement reasons, snapshot/attach/rolling events, positions and slots. The exact original token sequence, token equality patterns, original audit digest and token replay are intentionally unavailable.
- Text logs retain allowlisted numerical scheduling, cache, throughput, retirement and sampling records, plus startup/model/configuration/memory information with local paths scrubbed. Other content is replaced by explicit redaction markers. Raw tool-schema warnings and arbitrary unclassified text are not shared.
- Exact timestamps, model identity, serving parameters and numerical workload sizes remain intentionally visible for diagnosis. These can reveal usage patterns, but the export is designed not to expose request/tool contents.
- `manifest.json` records file lengths, SHA256 checksums and redaction counts. These checksums identify the scrubbed copies, not the originals.

Please do not feed the scrubbed `.ops` file into a parser that expects numeric token IDs without accounting for the explicit redaction field. It remains suitable for event-order/count/position analysis, not exact output reconstruction.

The original files remain private. No prompts, credentials, raw request-ID map or token-ID map are included in this export. Any further request for original token contents needs a separate privacy decision.

## Review and preservation check

An independent review checked every original/scrubbed record for privacy and diagnostic preservation. Its usefulness findings were corrected, then the changed lines were rechecked against the originals: 878 retirement summaries, four disconnect notices, one empty-cache notice and one static MoE configuration notice were restored with request aliases only. All 557,380 ops records remain; 40,770 of 40,774 text-log lines are retained with targeted scrubbing. Only four tool-related error lines remain fully redacted.
