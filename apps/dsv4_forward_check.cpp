// dsv4_forward_check: the engine's per-layer residual streams over a prompt
// on a DeepSeek-V4-Flash checkpoint (world 1, the streaming loader: each
// layer read from the NVMe as the walk reaches it), dumped for the
// independent torch reference (tools/dsv4_torch_reference.py: the release's
// own inference/model.py layer code) — the cross-check on the real weights
// and on the test fixture.
//
//   dsv4_forward_check --write-fixture DIR
//   dsv4_forward_check --model ORG/NAME | --checkpoint-dir DIR --out FILE
//                      (--ids 1,2,3 | --text FILE [--no-bos] | --tokens N [--seed S])
//                      [--layers N] [--all-logits] [--decode-steps N] [--draft]
//
// The dump: "DSV4ST01", int32 L T H4, bf16 states [L][T][H4] (every walked
// layer's output streams, [4, H] per row), int64 ids [T], int32 K and the
// routed expert ids [L][T][K] (ascending), int32 Li and ms and the indexed
// layers' selections [Li][T][ms] (-1 padded), int32 V and R and fp32
// logits [R][V] (the last row, or every row with --all-logits; R 0 when
// the walk is limited).
// --decode-steps N: the prompt's last N tokens are fed as single decode
// steps after a session prefill of the first T - N; the dump then carries
// a second section: "DSV4DEC1", int32 N L H4 V Li ms, and per step the
// bf16 states [L][H4], the fp32 logits [V], the selections [Li][ms].
// --draft (with --decode-steps): the model carries the DSpark stages and
// after every step drafts the block off the step's winner; a third section
// "DSV4DRF1", int32 N B V, per step: int32 the winner (the block's first
// token), the first draft row's Markov-biased fp32 logits [V], the block's
// base logits [B][V] (before the Markov bias) and row 0's confidence logit.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "common/log.hpp"
#include "dsv4_fixture.hpp"
#include "loaders/architecture.hpp"
#include "loaders/hf_cache.hpp"
#include "models/dsv4/config.hpp"
#include "models/dsv4/model.hpp"
#include "text/tokenizer.hpp"

int main(int argc, char** argv) {
  std::string model_id, ckpt, out, ids_text, text_path, fixture_dir;
  int tokens = 0, layers = 0, decode_steps = 0;
  uint64_t seed = 7;
  bool bos = true, all_logits = false, draft = false;
  const auto next = [&](int& i) {
    if (i + 1 >= argc) throw std::runtime_error("missing value after " + std::string(argv[i]));
    return argv[++i];
  };
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--model") model_id = next(i);
      else if (a == "--checkpoint-dir") ckpt = next(i);
      else if (a == "--write-fixture") fixture_dir = next(i);
      else if (a == "--out") out = next(i);
      else if (a == "--ids") ids_text = next(i);
      else if (a == "--text") text_path = next(i);
      else if (a == "--tokens") tokens = std::stoi(next(i));
      else if (a == "--seed") seed = std::stoull(next(i));
      else if (a == "--layers") layers = std::stoi(next(i));
      else if (a == "--decode-steps") decode_steps = std::stoi(next(i));
      else if (a == "--no-bos") bos = false;
      else if (a == "--all-logits") all_logits = true;
      else if (a == "--draft") draft = true;
      else throw std::runtime_error("unknown argument " + a);
    }
    if (!fixture_dir.empty()) {
      dsv4fx::write_fixture(dsv4fx::tiny_config(), fixture_dir);
      return 0;
    }
    if (ckpt.empty()) {
      if (model_id.empty()) throw std::runtime_error("--model or --checkpoint-dir is required");
      std::string err;
      ckpt = dgpp::hf::model_dir(model_id, &err);
      if (ckpt.empty()) throw std::runtime_error("cannot resolve " + model_id + ": " + err);
    }
    if (out.empty()) throw std::runtime_error("--out is required");
    const std::string cfg_path = (std::filesystem::path(ckpt) / "config.json").string();
    if (dgpp::detect_architecture_file(cfg_path) != dgpp::ModelArchitecture::DeepseekV4)
      throw std::runtime_error("not a DeepseekV4 checkpoint: " + ckpt);
    const dgpp::Dsv4Config cfg = dgpp::Dsv4Config::from_json_file(cfg_path);
    // The prompt: explicit ids, a text through the checkpoint's tokenizer
    // (BOS first unless --no-bos), or random ids.
    std::vector<int64_t> ids;
    if (!ids_text.empty()) {
      std::stringstream ss(ids_text);
      std::string item;
      while (std::getline(ss, item, ',')) ids.push_back(std::stoll(item));
    } else if (!text_path.empty()) {
      std::ifstream f(text_path);
      if (!f) throw std::runtime_error("cannot open " + text_path);
      std::stringstream buf;
      buf << f.rdbuf();
      const dgpp::text::Tokenizer tok =
          dgpp::text::Tokenizer::load((std::filesystem::path(ckpt) / "tokenizer.json").string());
      if (bos) {
        int64_t bos_id = -1;
        for (const auto& added : tok.added_tokens())
          if (added.content == "<｜begin▁of▁sentence｜>") bos_id = added.id;
        if (bos_id < 0) throw std::runtime_error("the tokenizer has no BOS token");
        ids.push_back(bos_id);
      }
      const std::vector<int64_t> body = tok.encode(buf.str());
      ids.insert(ids.end(), body.begin(), body.end());
      if (tokens > 0 && static_cast<int>(ids.size()) > tokens) ids.resize(static_cast<size_t>(tokens));
    } else {
      if (tokens <= 0) throw std::runtime_error("--ids, --text or --tokens is required");
      std::mt19937_64 rng(seed);
      for (int i = 0; i < tokens; ++i) ids.push_back(static_cast<int64_t>(rng() % static_cast<uint64_t>(cfg.vocab_size)));
    }
    const int T = static_cast<int>(ids.size());
    if (T < 1) throw std::runtime_error("an empty prompt");
    if (decode_steps < 0 || decode_steps >= T) throw std::runtime_error("--decode-steps must leave a prefill row");
    if (draft && decode_steps == 0) throw std::runtime_error("--draft rides --decode-steps");
    if (decode_steps > 0 && layers > 0) throw std::runtime_error("--decode-steps walks every layer");
    DGPP_LOG_INFO("dsv4_forward_check: {} tokens, {} layers, {} decode steps", T, layers > 0 ? layers : cfg.num_hidden_layers,
                  decode_steps);
    const auto t0 = std::chrono::steady_clock::now();
    dgpp::Dsv4Model model(cfg, ckpt, /*max_tokens=*/std::max({T, cfg.sliding_window, 16}),
                          /*max_cache_tokens=*/std::max(T + 256, 1024), dgpp::Dsv4Residency::Streaming, nullptr, 0, 1, 1,
                          /*mtp=*/draft, /*decode_rows=*/8);
    if (layers > 0) model.set_debug_layer_limit(layers);
    const dgpp::Dsv4Model::Outputs o = model.forward(ids, /*capture_layers=*/true);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const int L = static_cast<int>(o.layer_states.size());
    const int H4 = cfg.hc_mult * cfg.hidden_size;
    if (L < 1) throw std::runtime_error("no layer captures");
    std::ofstream f(out, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + out);
    f.write("DSV4ST01", 8);
    const int32_t hdr[3] = {L, T, H4};
    f.write(reinterpret_cast<const char*>(hdr), 12);
    for (const auto& st : o.layer_states) {
      if (st.size() != static_cast<size_t>(T) * H4) throw std::runtime_error("a layer capture with the wrong rows");
      f.write(reinterpret_cast<const char*>(st.data()), static_cast<std::streamsize>(st.size() * 2));
    }
    f.write(reinterpret_cast<const char*>(ids.data()), static_cast<std::streamsize>(ids.size() * 8));
    const int32_t K = static_cast<int32_t>(cfg.num_experts_per_tok);
    f.write(reinterpret_cast<const char*>(&K), 4);
    if (static_cast<int>(o.route_ids.size()) < L) throw std::runtime_error("route captures");
    for (int l = 0; l < L; ++l) {
      if (o.route_ids[static_cast<size_t>(l)].size() != static_cast<size_t>(T) * K) throw std::runtime_error("route rows");
      f.write(reinterpret_cast<const char*>(o.route_ids[static_cast<size_t>(l)].data()), static_cast<std::streamsize>(T) * K * 4);
    }
    const int32_t Li = static_cast<int32_t>(o.dsa_selections.size());
    const int32_t ms = static_cast<int32_t>(cfg.index_topk);
    f.write(reinterpret_cast<const char*>(&Li), 4);
    f.write(reinterpret_cast<const char*>(&ms), 4);
    for (const auto& sel : o.dsa_selections) {
      if (sel.size() != static_cast<size_t>(T) * ms) throw std::runtime_error("selection rows");
      f.write(reinterpret_cast<const char*>(sel.data()), static_cast<std::streamsize>(sel.size() * 4));
    }
    const int32_t V = o.lm_vocab_count;
    const bool have_logits = layers <= 0 && !o.logits.empty();
    const int32_t R = !have_logits ? 0 : (all_logits ? T : 1);
    f.write(reinterpret_cast<const char*>(&V), 4);
    f.write(reinterpret_cast<const char*>(&R), 4);
    if (R > 0)
      f.write(reinterpret_cast<const char*>(o.logits.data() + static_cast<size_t>(T - R) * V),
              static_cast<std::streamsize>(R) * V * 4);
    DGPP_LOG_INFO("dsv4_forward_check: {} layers x {} rows x {} (+ routes, {} indexed layers) written to {} in {:.1f} s", L, T,
                  H4, Li, out, secs);
    if (have_logits) {
      const float* last = o.logits.data() + o.logits.size() - static_cast<size_t>(V);
      std::vector<int> order(static_cast<size_t>(V));
      for (int i = 0; i < V; ++i) order[static_cast<size_t>(i)] = i;
      std::partial_sort(order.begin(), order.begin() + 8, order.end(), [&](int a, int b) { return last[a] > last[b]; });
      std::string s;
      for (int i = 0; i < 8; ++i)
        s += std::format("{}{}:{:.3f}", i ? " " : "", order[static_cast<size_t>(i)], last[order[static_cast<size_t>(i)]]);
      DGPP_LOG_INFO("dsv4_forward_check: the last row's top-8 logits (id:value): {}", s);
    }
    if (decode_steps > 0) {
      // The decode path: a session prefill of the first T - N tokens, then
      // the last N as single steps, every layer's row captured.
      const int P = T - decode_steps;
      const std::vector<int64_t> head(ids.begin(), ids.begin() + P);
      const auto pre_out = model.session_prefill(0, head);
      setenv("DGPP_DSV4_CAPTURE_DECODE", "1", 1);
      f.write("DSV4DEC1", 8);
      const int32_t dh[6] = {decode_steps, cfg.num_hidden_layers, H4, V, Li, ms};
      f.write(reinterpret_cast<const char*>(dh), 24);
      std::vector<float> draft_rows, draft_base, draft_conf;
      std::vector<int32_t> draft_next;
      const int B = cfg.dspark_block_size;
      if (draft) {
        // The draft trails the session by the prompt's last row: draft it
        // (its block is not dumped) so every step drafts exactly its row.
        int w = 0;
        for (int v = 1; v < V; ++v)
          if (pre_out.logits[static_cast<size_t>(v)] > pre_out.logits[static_cast<size_t>(w)]) w = v;
        (void)model.session_draft(0, {static_cast<int64_t>(w)});
      }
      for (int s = 0; s < decode_steps; ++s) {
        const auto st = model.session_step(0, static_cast<int32_t>(ids[static_cast<size_t>(P + s)]));
        if (static_cast<int>(st.layer_states.size()) != cfg.num_hidden_layers) throw std::runtime_error("decode captures");
        for (const auto& ls : st.layer_states) {
          if (ls.size() != static_cast<size_t>(H4)) throw std::runtime_error("a decode capture with more than one row");
          f.write(reinterpret_cast<const char*>(ls.data()), static_cast<std::streamsize>(ls.size() * 2));
        }
        if (st.logits.size() != static_cast<size_t>(V)) throw std::runtime_error("decode logits");
        f.write(reinterpret_cast<const char*>(st.logits.data()), static_cast<std::streamsize>(V) * 4);
        if (static_cast<int>(st.dsa_selections.size()) != Li) throw std::runtime_error("decode selections");
        for (const auto& sel : st.dsa_selections)
          f.write(reinterpret_cast<const char*>(sel.data()), static_cast<std::streamsize>(ms) * 4);
        if (draft) {
          // The draft off the step's row: the block [winner, noise...] at the
          // next positions — row 0's Markov-biased logits, the block's base
          // logits (every row, before the Markov bias) and row 0's confidence.
          int next_tok = 0;
          for (int v = 1; v < V; ++v)
            if (st.logits[static_cast<size_t>(v)] > st.logits[static_cast<size_t>(next_tok)]) next_tok = v;
          const auto row = model.session_draft(0, {static_cast<int64_t>(next_tok)});
          if (row.logits.size() != static_cast<size_t>(V)) throw std::runtime_error("draft row");
          draft_rows.insert(draft_rows.end(), row.logits.begin(), row.logits.end());
          std::vector<float> base(static_cast<size_t>(B) * V), conf(static_cast<size_t>(B));
          DGPP_CUDA_OK(cudaDeviceSynchronize());
          DGPP_CUDA_OK(cudaMemcpy(base.data(), model.debug_base_logits(), base.size() * 4, cudaMemcpyDeviceToHost));
          DGPP_CUDA_OK(cudaMemcpy(conf.data(), model.debug_confidence(), static_cast<size_t>(B) * 4, cudaMemcpyDeviceToHost));
          draft_base.insert(draft_base.end(), base.begin(), base.end());
          draft_conf.push_back(conf[0]);
          draft_next.push_back(next_tok);
        }
      }
      unsetenv("DGPP_DSV4_CAPTURE_DECODE");
      if (draft) {
        f.write("DSV4DRF1", 8);
        const int32_t fh[3] = {decode_steps, B, V};
        f.write(reinterpret_cast<const char*>(fh), 12);
        for (int s = 0; s < decode_steps; ++s) {
          f.write(reinterpret_cast<const char*>(&draft_next[static_cast<size_t>(s)]), 4);
          f.write(reinterpret_cast<const char*>(draft_rows.data() + static_cast<size_t>(s) * V), static_cast<std::streamsize>(V) * 4);
          f.write(reinterpret_cast<const char*>(draft_base.data() + static_cast<size_t>(s) * B * V),
                  static_cast<std::streamsize>(B) * V * 4);
          f.write(reinterpret_cast<const char*>(&draft_conf[static_cast<size_t>(s)]), 4);
        }
      }
      DGPP_LOG_INFO("dsv4_forward_check: {} decode steps after a {}-token prefill appended", decode_steps, P);
    }
    return 0;
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("dsv4_forward_check: {}", e.what());
    return 1;
  }
}
