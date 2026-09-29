// The gated residual: every device stage against its
// exact host math on the device's own inputs (the elementwise kernels
// bitwise; the GEMV within an ulp of the sequential chain), and the whole
// site — mix then combine — within two bf16 ulps of the reference.
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/bf16_gemv.hpp"
#include "kernels/bf12_gemv.hpp"
#include "kernels/qwen_gr.hpp"
#include "kernels/qwen_norm.hpp"
#include "kernels/scale_gemm.hpp"
#include "loaders/fp8_quant.hpp"
#include "models/qwen/gr_reference.hpp"
#include "models/qwen/norm_reference.hpp"

using namespace dgpp::kda_test;

namespace {

struct Site {
  int rows = 3, hc = 4, hidden = 256, rank = 32;
  int width() const { return hc * hidden; }
  std::vector<uint16_t> r, w_norm, w_down, w_up, w_inj, y;
  static Site make(uint64_t seed) {
    Site s;
    s.r = random_bf16_normal(seed + 1, static_cast<int64_t>(s.rows) * s.width(), 1.0f);
    s.w_norm = random_bf16_uniform(seed + 2, s.width(), 0.3f);
    s.w_down = random_bf16_normal(seed + 3, static_cast<int64_t>(s.rank) * s.width(), 0.05f);
    s.w_up = random_bf16_normal(seed + 4, static_cast<int64_t>(s.width()) * s.rank, 0.2f);
    s.w_inj = random_bf16_normal(seed + 5, static_cast<int64_t>(s.hc) * s.width(), 0.05f);
    s.y = random_bf16_normal(seed + 6, static_cast<int64_t>(s.rows) * s.hidden, 0.5f);
    return s;
  }
};

std::vector<uint16_t> download(const DevBuf& b, size_t n) {
  std::vector<uint16_t> v(n);
  b.download(v.data(), n * 2);
  return v;
}

}  // namespace

// The FP8 twins (2026-09-10, engine.dense_weights = "fp8"): the down / up
// matrices encoded to block FP8 (the loader's recipe), the fused forms —
// the norm-staged down GEMV with the inject rows, the act-staged up GEMV,
// the batched down + inject — against the unfused fp8 chain (the norm
// kernel, launch_scale_gemm_bf16, gate_act, launch_scale_gemm_bf16, the
// dots kernel): bitwise, at one row (the scalar path), three, and six
// (two chunks).
DGPP_TEST(qwen_gr_fp8_fused_forms_are_bitwise_the_fp8_chain) {
  cudaStream_t st = test_stream();
  for (const int rows : {1, 3, 6}) {
    Site s = Site::make(20 + rows);
    s.rows = rows;
    s.r = random_bf16_normal(21 + rows, static_cast<int64_t>(rows) * s.width(), 1.0f);
    const int W = s.width();
    // The FP8 forms of down [rank, W] and up [W, rank].
    std::vector<uint8_t> down_p(static_cast<size_t>(s.rank) * W), up_p(static_cast<size_t>(W) * s.rank);
    std::vector<float> down_s(static_cast<size_t>(dgpp::fp8_quant::scale_rows(s.rank)) * dgpp::fp8_quant::scale_cols(W));
    std::vector<float> up_s(static_cast<size_t>(dgpp::fp8_quant::scale_rows(W)) * dgpp::fp8_quant::scale_cols(s.rank));
    dgpp::fp8_quant::encode_block128(s.w_down.data(), W, s.rank, W, down_p.data(), down_s.data(), 1);
    dgpp::fp8_quant::encode_block128(s.w_up.data(), s.rank, W, s.rank, up_p.data(), up_s.data(), 1);
    DevBuf dr(s.r.size() * 2), dnorm(s.w_norm.size() * 2), dinj(s.w_inj.size() * 2);
    DevBuf ddp(down_p.size()), dds(down_s.size() * 4), dup(up_p.size()), dus(up_s.size() * 4);
    dr.upload(s.r.data(), s.r.size() * 2);
    dnorm.upload(s.w_norm.data(), s.w_norm.size() * 2);
    dinj.upload(s.w_inj.data(), s.w_inj.size() * 2);
    ddp.upload(down_p.data(), down_p.size());
    dds.upload(down_s.data(), down_s.size() * 4);
    dup.upload(up_p.data(), up_p.size());
    dus.upload(up_s.data(), up_s.size() * 4);
    const size_t tn = static_cast<size_t>(rows) * s.rank, gn = static_cast<size_t>(rows) * s.hc;
    // The chain.
    DevBuf drn(s.r.size() * 2), dt(tn * 2), dlog(s.r.size() * 2), dg(gn * 4);
    dgpp::qwen_group_rmsnorm_bf16(dr.p, dnorm.p, drn.p, rows, s.hc, s.hidden, 1e-6f, st);
    dgpp::launch_scale_gemm_bf16(drn.as<uint16_t>(), static_cast<size_t>(W), ddp.as<uint8_t>(), static_cast<const float*>(dds.p),
                                 dt.as<uint16_t>(), rows, s.rank, W, st, static_cast<size_t>(s.rank));
    dgpp::qwen_gr_combine_dots_bf16(drn.p, dinj.p, static_cast<float*>(dg.p), rows, s.hc, s.hidden, st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    const std::vector<uint16_t> rn = download(drn, s.r.size());
    const std::vector<uint16_t> t_chain = download(dt, tn);
    dgpp::qwen_gr_gate_act_bf16(dt.p, rows, s.rank, s.hc, st);
    dgpp::launch_scale_gemm_bf16(dt.as<uint16_t>(), static_cast<size_t>(s.rank), dup.as<uint8_t>(), static_cast<const float*>(dus.p),
                                 dlog.as<uint16_t>(), rows, W, s.rank, st, static_cast<size_t>(W));
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    const std::vector<uint16_t> logits = download(dlog, s.r.size());
    std::vector<float> g_ref(gn);
    DGPP_CUDA_OK(cudaMemcpy(g_ref.data(), dg.p, gn * 4, cudaMemcpyDeviceToHost));
    // The fused forms.
    DevBuf drn2(s.r.size() * 2), dt2(tn * 2), dlog2(s.r.size() * 2), dg2(gn * 4), dt3(tn * 2), dg3(gn * 4);
    dgpp::qwen_gr_norm_down_fp8(dr.p, static_cast<size_t>(W), dnorm.p, s.hc, s.hidden, 1e-6f, drn2.p, ddp.as<uint8_t>(),
                                static_cast<const float*>(dds.p), dt2.p, s.rank, rows, st, dinj.p, static_cast<float*>(dg2.p));
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    const std::vector<uint16_t> t2 = download(dt2, tn);
    dgpp::qwen_gr_act_up_fp8(dt2.p, s.rank, s.hc, dup.as<uint8_t>(), static_cast<const float*>(dus.p), dlog2.p, s.hidden, rows, st);
    dgpp::qwen_gr_down_inject_fp8(drn.p, ddp.as<uint8_t>(), static_cast<const float*>(dds.p), dt3.p, s.rank, dinj.p,
                                  static_cast<float*>(dg3.p), s.hc, s.hidden, rows, st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    const std::vector<uint16_t> rn2 = download(drn2, s.r.size());
    const std::vector<uint16_t> logits2 = download(dlog2, s.r.size());
    const std::vector<uint16_t> t3 = download(dt3, tn);
    std::vector<float> g2(gn), g3(gn);
    DGPP_CUDA_OK(cudaMemcpy(g2.data(), dg2.p, gn * 4, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(g3.data(), dg3.p, gn * 4, cudaMemcpyDeviceToHost));
    require_bitwise("fp8 norm_down: Rn", rn2.data(), rn.data(), rn2.size() * 2);
    require_bitwise("fp8 norm_down: t", t2.data(), t_chain.data(), t2.size() * 2);
    require_bitwise("fp8 norm_down: inject gates", g2.data(), g_ref.data(), gn * 4);
    require_bitwise("fp8 act_up: logits", logits2.data(), logits.data(), logits2.size() * 2);
    require_bitwise("fp8 down_inject: t", t3.data(), t_chain.data(), t3.size() * 2);
    require_bitwise("fp8 down_inject: gates", g3.data(), g_ref.data(), gn * 4);
    // The norm staged into the batched down + inject GEMV: Rn, t, gates bitwise the chain.
    {
      DevBuf drn4(s.r.size() * 2), dt4b(tn * 2), dg4(gn * 4);
      dgpp::qwen_gr_norm_down_inject_fp8(dr.p, static_cast<size_t>(W), dnorm.p, s.hc, s.hidden, 1e-6f, drn4.p,
                                         ddp.as<uint8_t>(), static_cast<const float*>(dds.p), dt4b.p, s.rank, dinj.p,
                                         static_cast<float*>(dg4.p), rows, st);
      DGPP_CUDA_OK(cudaStreamSynchronize(st));
      const std::vector<uint16_t> rn4 = download(drn4, s.r.size());
      const std::vector<uint16_t> t4 = download(dt4b, tn);
      std::vector<float> g4(gn);
      DGPP_CUDA_OK(cudaMemcpy(g4.data(), dg4.p, gn * 4, cudaMemcpyDeviceToHost));
      require_bitwise("fp8 norm_down_inject: Rn", rn4.data(), rn.data(), rn4.size() * 2);
      require_bitwise("fp8 norm_down_inject: t", t4.data(), t_chain.data(), t4.size() * 2);
      require_bitwise("fp8 norm_down_inject: gates", g4.data(), g_ref.data(), gn * 4);
    }
    // act_up_mix on the chain's t and Rn against mix_finish on the chain's logits.
    DevBuf dx(static_cast<size_t>(rows) * s.hidden * 2), dx2(static_cast<size_t>(rows) * s.hidden * 2);
    dgpp::qwen_gr_mix_finish_bf16(dlog.p, drn.p, dx.p, rows, s.hc, s.hidden, st);
    // dt was activated in place above; dt2 is the chain's t (bitwise) untouched.
    dgpp::qwen_gr_act_up_mix_fp8(dt2.p, s.rank, s.hc, dup.as<uint8_t>(), static_cast<const float*>(dus.p), drn.p, dx2.p,
                                 s.hidden, rows, st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    const std::vector<uint16_t> x = download(dx, static_cast<size_t>(rows) * s.hidden);
    const std::vector<uint16_t> x2 = download(dx2, x.size());
    require_bitwise("fp8 act_up_mix: x", x2.data(), x.data(), x2.size() * 2);
    std::printf("[ OK ] the fp8 fused GR forms are bitwise the fp8 chain at %d rows\n", rows);
  }
}

DGPP_TEST(qwen_gr_stages_match_the_reference_on_the_device_inputs) {
  const Site s = Site::make(10);
  cudaStream_t st = test_stream();
  const int W = s.width();
  DevBuf dr(s.r.size() * 2), dnorm(s.w_norm.size() * 2), ddown(s.w_down.size() * 2),
      dup(s.w_up.size() * 2), dinj(s.w_inj.size() * 2), dy(s.y.size() * 2);
  DevBuf drn(s.r.size() * 2), dt(static_cast<size_t>(s.rows) * s.rank * 2),
      dlog(s.r.size() * 2), dx(static_cast<size_t>(s.rows) * s.hidden * 2);
  dr.upload(s.r.data(), s.r.size() * 2);
  dnorm.upload(s.w_norm.data(), s.w_norm.size() * 2);
  ddown.upload(s.w_down.data(), s.w_down.size() * 2);
  dup.upload(s.w_up.data(), s.w_up.size() * 2);
  dinj.upload(s.w_inj.data(), s.w_inj.size() * 2);
  dy.upload(s.y.data(), s.y.size() * 2);
  // 1. the group norm
  dgpp::qwen_group_rmsnorm_bf16(dr.p, dnorm.p, drn.p, s.rows, s.hc, s.hidden, 1e-6f, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> rn = download(drn, s.r.size());
  {
    std::vector<uint16_t> want(s.r.size());
    dgpp::qwen_ref::group_rmsnorm(s.r.data(), s.w_norm.data(), want.data(), s.rows, s.hc, s.hidden, 1e-6f);
    require_bf16("group norm", compare_bf16(rn, want, 1), 1e-3, 0.01);
  }
  // 2. the down GEMV on the device's Rn
  if (!dgpp::bf16_gemv_accepts(ddown.p, s.rows, W)) throw std::runtime_error("gemv shape refused");
  dgpp::launch_bf16_gemv(drn.as<uint16_t>(), W, ddown.as<uint16_t>(), dt.p, false, s.rows, s.rank, W, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> t_dev = download(dt, static_cast<size_t>(s.rows) * s.rank);
  {
    std::vector<uint16_t> want(t_dev.size());
    dgpp::qwen_ref::gemv_bf16(rn.data(), s.w_down.data(), want.data(), s.rows, s.rank, W);
    require_bf16("down gemv", compare_bf16(t_dev, want, 1), 1e-3, 0.02);
  }
  // 3. the activation (bitwise on the device's t)
  std::vector<uint16_t> t_ref = t_dev;
  dgpp::qwen_ref::gr_gate_act(t_ref.data(), s.rows, s.rank, s.hc);
  dgpp::qwen_gr_gate_act_bf16(dt.p, s.rows, s.rank, s.hc, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> t_act = download(dt, t_dev.size());
  require_bitwise("gate act", t_act.data(), t_ref.data(), t_act.size() * 2);
  // 4. the up GEMV
  dgpp::launch_bf16_gemv(dt.as<uint16_t>(), s.rank, dup.as<uint16_t>(), dlog.p, false, s.rows, W, s.rank, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> logits = download(dlog, s.r.size());
  {
    std::vector<uint16_t> want(logits.size());
    dgpp::qwen_ref::gemv_bf16(t_act.data(), s.w_up.data(), want.data(), s.rows, W, s.rank);
    require_bf16("up gemv", compare_bf16(logits, want, 1), 1e-3, 0.02);
  }
  // 4b. the fused decode GEMVs: the norm staged into the down
  // GEMV and the activation into the up GEMV — bitwise the chain above.
  {
    DevBuf drn2(s.r.size() * 2), dt2(static_cast<size_t>(s.rows) * s.rank * 2), dlog2(s.r.size() * 2);
    // The inject dots folded into the same launch against the
    // standalone dots kernel on the same Rn: bitwise, or the gate that
    // decides every combine has moved.
    DevBuf dg_fold(static_cast<size_t>(s.rows) * s.hc * 4), dg_ref(static_cast<size_t>(s.rows) * s.hc * 4);
    if (!dgpp::qwen_gr_fused_mix_accepts(s.hc, s.hidden, s.rank)) throw std::runtime_error("fused mix refuses the fixture");
    dgpp::qwen_gr_norm_down_bf16(dr.p, static_cast<size_t>(W), dnorm.p, s.hc, s.hidden, 1e-6f, drn2.p,
                                 ddown.p, dt2.p, s.rank, s.rows, st, dinj.p,
                                 static_cast<float*>(dg_fold.p));
    dgpp::qwen_gr_combine_dots_bf16(drn.p, dinj.p, static_cast<float*>(dg_ref.p), s.rows, s.hc,
                                    s.hidden, st);
    dgpp::qwen_gr_act_up_bf16(dt2.p, s.rank, s.hc, dup.p, dlog2.p, s.hidden, s.rows, st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    const std::vector<uint16_t> rn2 = download(drn2, s.r.size());
    const std::vector<uint16_t> t2 = download(dt2, t_dev.size());
    const std::vector<uint16_t> logits2 = download(dlog2, s.r.size());
    require_bitwise("fused norm_down: Rn", rn2.data(), rn.data(), rn2.size() * 2);
    require_bitwise("fused norm_down: t", t2.data(), t_dev.data(), t2.size() * 2);
    require_bitwise("fused act_up: logits", logits2.data(), logits.data(), logits2.size() * 2);
    std::vector<float> g_fold(static_cast<size_t>(s.rows) * s.hc), g_ref(g_fold.size());
    DGPP_CUDA_OK(cudaMemcpy(g_fold.data(), dg_fold.p, g_fold.size() * 4, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(g_ref.data(), dg_ref.p, g_ref.size() * 4, cudaMemcpyDeviceToHost));
    require_bitwise("fused norm_down: inject gates", g_fold.data(), g_ref.data(), g_fold.size() * 4);
    std::printf("[ OK ] the fused decode GEMVs are bitwise the norm/GEMV/act/GEMV chain at %d rows, "
                "and the folded inject gates the dots kernel's\n", s.rows);
  }
  // 5. the finish (bitwise on the device's logits and Rn)
  dgpp::qwen_gr_mix_finish_bf16(dlog.p, drn.p, dx.p, s.rows, s.hc, s.hidden, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> x = download(dx, static_cast<size_t>(s.rows) * s.hidden);
  {
    std::vector<uint16_t> want(x.size());
    dgpp::qwen_ref::gr_mix_finish(logits.data(), rn.data(), want.data(), s.rows, s.hc, s.hidden);
    require_bitwise("mix finish", x.data(), want.data(), x.size() * 2);
  }
  // 5b. the up GEMV with the mix in its epilogue, on the fused chain's t
  // (== the chain's t) and the device's Rn: bitwise the act_up + finish x.
  {
    DevBuf dt4(static_cast<size_t>(s.rows) * s.rank * 2), dx2(static_cast<size_t>(s.rows) * s.hidden * 2);
    dt4.upload(t_dev.data(), t_dev.size() * 2);
    dgpp::qwen_gr_act_up_mix_bf16(dt4.p, s.rank, s.hc, dup.p, drn.p, dx2.p, s.hidden, s.rows, st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    const std::vector<uint16_t> x2 = download(dx2, x.size());
    require_bitwise("act_up_mix: x", x2.data(), x.data(), x2.size() * 2);
    std::printf("[ OK ] act_up with the mix in its epilogue is bitwise act_up + mix_finish at %d rows\n", s.rows);
  }
  // 6. the combine: the inject dots are the kernel's own (a warp chain), so
  // s may differ from the host's sequential dot by an ulp — hold R' to one.
  DevBuf dgates(static_cast<size_t>(s.rows) * s.hc * 4);
  dgpp::qwen_gr_combine_bf16(dr.p, drn.p, dinj.p, dy.p, static_cast<float*>(dgates.p), s.rows,
                             s.hc, s.hidden, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> r2 = download(dr, s.r.size());
  {
    std::vector<uint16_t> want = s.r;
    dgpp::qwen_ref::gr_combine(want.data(), rn.data(), s.w_inj.data(), s.y.data(), s.rows, s.hc, s.hidden);
    require_bf16("combine", compare_bf16(r2, want, 1), 1e-3, 0.01);
  }
  // 6b. the batched rows' down GEMV with the inject rows appended
  //: t bitwise the chain's GEMV, the gates bitwise the dots
  // kernel's, at the fixture's rows.
  {
    DevBuf dt3(static_cast<size_t>(s.rows) * s.rank * 2), dg3(static_cast<size_t>(s.rows) * s.hc * 4);
    dgpp::qwen_gr_down_inject_bf16(drn.p, ddown.p, dt3.p, s.rank, dinj.p,
                                   static_cast<float*>(dg3.p), s.hc, s.hidden, s.rows, st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    const std::vector<uint16_t> t3 = download(dt3, t_dev.size());
    require_bitwise("down_inject: t", t3.data(), t_dev.data(), t3.size() * 2);
    std::vector<float> g3(static_cast<size_t>(s.rows) * s.hc), g_ref(g3.size());
    DGPP_CUDA_OK(cudaMemcpy(g3.data(), dg3.p, g3.size() * 4, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(g_ref.data(), dgates.p, g_ref.size() * 4, cudaMemcpyDeviceToHost));
    require_bitwise("down_inject: gates", g3.data(), g_ref.data(), g3.size() * 4);
    std::printf("[ OK ] the batched down GEMV with the inject rows is bitwise the chain at %d rows\n", s.rows);
    // 6c. the group norm staged into the batched down + inject GEMV (one
    // block reduction for every (row, group)) on the raw R: Rn bitwise the
    // group norm's, t and the gates the chain's.
    {
      DevBuf dr0(s.r.size() * 2), drn5(s.r.size() * 2), dt5(static_cast<size_t>(s.rows) * s.rank * 2),
          dg5(static_cast<size_t>(s.rows) * s.hc * 4);
      dr0.upload(s.r.data(), s.r.size() * 2);
      dgpp::qwen_gr_norm_down_inject_bf16(dr0.p, static_cast<size_t>(W), dnorm.p, s.hc, s.hidden, 1e-6f, drn5.p,
                                          ddown.p, dt5.p, s.rank, dinj.p, static_cast<float*>(dg5.p), s.rows, st);
      DGPP_CUDA_OK(cudaStreamSynchronize(st));
      const std::vector<uint16_t> rn5 = download(drn5, s.r.size());
      const std::vector<uint16_t> t5 = download(dt5, t_dev.size());
      std::vector<float> g5(static_cast<size_t>(s.rows) * s.hc);
      DGPP_CUDA_OK(cudaMemcpy(g5.data(), dg5.p, g5.size() * 4, cudaMemcpyDeviceToHost));
      require_bitwise("norm_down_inject: Rn", rn5.data(), rn.data(), rn5.size() * 2);
      require_bitwise("norm_down_inject: t", t5.data(), t_dev.data(), t5.size() * 2);
      require_bitwise("norm_down_inject: gates", g5.data(), g_ref.data(), g5.size() * 4);
      std::printf("[ OK ] the norm-staged batched down GEMV is bitwise the group norm + down_inject chain at %d rows\n",
                  s.rows);
    }
    // The down matrix's 12-bit companion: bitwise the bf16 chain.
    {
      const dgpp::Bf12Host h = dgpp::bf12_encode(s.w_down.data(), s.rank, W);
      DevBuf dp(h.packed.size()), db(h.rows.size() * 4), de(h.esc.size() * 4), draw(h.raw.size() * 2);
      dp.upload(h.packed.data(), h.packed.size());
      db.upload(h.rows.data(), h.rows.size() * 4);
      de.upload(h.esc.data(), h.esc.size() * 4);
      draw.upload(h.raw.data(), h.raw.size() * 2);
      dgpp::Bf12Matrix m;
      m.packed = dp.as<uint8_t>();
      m.rows = db.as<uint32_t>();
      m.esc = de.as<uint32_t>();
      m.raw = draw.as<uint16_t>();
      m.n = s.rank;
      m.k = W;
      DevBuf dt5(static_cast<size_t>(s.rows) * s.rank * 2), dg5(static_cast<size_t>(s.rows) * s.hc * 4);
      dgpp::qwen_gr_down_inject_bf12(drn.p, m, dt5.p, s.rank, dinj.p, static_cast<float*>(dg5.p), s.hc, s.hidden, s.rows, st);
      DGPP_CUDA_OK(cudaStreamSynchronize(st));
      const std::vector<uint16_t> t5 = download(dt5, t_dev.size());
      std::vector<float> g5(static_cast<size_t>(s.rows) * s.hc);
      DGPP_CUDA_OK(cudaMemcpy(g5.data(), dg5.p, g5.size() * 4, cudaMemcpyDeviceToHost));
      require_bitwise("down_inject bf12: t", t5.data(), t_dev.data(), t5.size() * 2);
      require_bitwise("down_inject bf12: gates", g5.data(), g_ref.data(), g5.size() * 4);
      std::printf("[ OK ] the down GEMV from the 12-bit companion is bitwise the bf16 chain at %d rows\n", s.rows);
    }
  }
  // 7. end to end: the device's x against the host's whole read from R.
  {
    std::vector<uint16_t> rn_ref(s.r.size()), x_ref(x.size());
    dgpp::qwen_ref::gr_mix(s.r.data(), s.w_norm.data(), s.w_down.data(), s.w_up.data(), rn_ref.data(),
                           x_ref.data(), s.rows, s.hc, s.hidden, s.rank, 1e-6f);
    require_bf16("mix end to end", compare_bf16(x, x_ref, 2), 2e-3, 0.02);
  }
}

int main() { return dgpp::test::run_all(); }
