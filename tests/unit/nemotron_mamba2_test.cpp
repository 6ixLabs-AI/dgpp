// The Nemotron-H Mamba2 host reference against tools/nemotron_reference.py:
// nemotron_mamba2_vectors.hpp holds one Mamba layer of the synthetic
// checkpoint (tools/nemotron_synth.py ckpt --preset nano --seed 0) and what
// the numpy reference computes from a fixed in_proj output in float64 —
// the convolution, the time step, the scan, the gated norm and both final
// states. The numpy side is itself held to the reference's chunked SSD
// algorithm by `nemotron_synth.py selftest`.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "models/nemotron/mamba2_reference.hpp"
#include "nemotron_mamba2_vectors.hpp"

namespace {

namespace v = nemotron_mamba2_vectors;
using dgpp::mamba2_ref::Geometry;
using dgpp::mamba2_ref::Rounding;
using dgpp::mamba2_ref::State;
using dgpp::mamba2_ref::Trace;
using dgpp::mamba2_ref::Weights;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

Geometry geometry() {
  return Geometry{v::kHeads, v::kHeadDim, v::kState, v::kGroups, v::kConvKernel};
}
Weights weights() {
  return Weights{v::kConvW, v::kConvB, v::kALog, v::kD, v::kDtBias, v::kNormW};
}

template <typename A, typename B>
double max_abs_diff(const A* a, const B* b, size_t n) {
  double worst = 0;
  for (size_t i = 0; i < n; ++i)
    worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  return worst;
}

template <typename Acc>
struct Run {
  std::vector<Acc> out;
  Trace<Acc> trace;
  State<Acc> state{geometry()};
};

// Feeds the vectors' tokens in the given pieces (which must sum to kTokens).
template <typename Acc>
Run<Acc> run(const std::vector<int>& pieces, Rounding rounding = Rounding::None) {
  const Geometry g = geometry();
  Run<Acc> r;
  r.out.resize(static_cast<size_t>(v::kTokens) * static_cast<size_t>(g.inner()));
  int done = 0;
  for (const int n : pieces) {
    dgpp::mamba2_ref::mixer<Acc>(
        g, weights(), v::kProj + static_cast<size_t>(done) * static_cast<size_t>(g.proj_rows()), n,
        r.state, v::kEps, r.out.data() + static_cast<size_t>(done) * static_cast<size_t>(g.inner()),
        rounding, &r.trace);
    done += n;
  }
  require(done == v::kTokens, "pieces must cover the vectors");
  return r;
}

template <typename Acc>
void expect_matches(const Run<Acc>& r, double tol, const char* what) {
  const auto worst = [&](const char* name, double d) {
    require(d <= tol, std::string(what) + ": " + name + " max |diff| " + std::to_string(d) + " > " +
                          std::to_string(tol));
  };
  worst("conv", max_abs_diff(r.trace.conv.data(), v::kWantConv, std::size(v::kWantConv)));
  worst("dt", max_abs_diff(r.trace.dt.data(), v::kWantDt, std::size(v::kWantDt)));
  worst("scan", max_abs_diff(r.trace.scan.data(), v::kWantScan, std::size(v::kWantScan)));
  worst("out", max_abs_diff(r.out.data(), v::kWantOut, std::size(v::kWantOut)));
  worst("ssm state", max_abs_diff(r.state.ssm.data(), v::kWantSsm, std::size(v::kWantSsm)));
  worst("conv tail",
        max_abs_diff(r.state.conv.data(), v::kWantConvTail, std::size(v::kWantConvTail)));
}

}  // namespace

DGPP_TEST(nemotron_mamba2_vectors_have_the_geometry) {
  const Geometry g = geometry();
  require(g.inner() == 32 && g.conv_dim() == 96 && g.proj_rows() == 132, "derived widths");
  require(std::size(v::kConvW) ==
              static_cast<size_t>(g.conv_dim()) * static_cast<size_t>(g.conv_kernel),
          "conv weight");
  require(std::size(v::kConvB) == static_cast<size_t>(g.conv_dim()), "conv bias");
  require(
      std::size(v::kProj) == static_cast<size_t>(v::kTokens) * static_cast<size_t>(g.proj_rows()),
      "in_proj output");
  require(std::size(v::kWantConv) ==
              static_cast<size_t>(v::kTokens) * static_cast<size_t>(g.conv_dim()),
          "conv output");
  require(
      std::size(v::kWantOut) == static_cast<size_t>(v::kTokens) * static_cast<size_t>(g.inner()),
      "norm output");
  require(std::size(v::kWantSsm) == static_cast<size_t>(g.heads) * static_cast<size_t>(g.head_dim) *
                                        static_cast<size_t>(g.state),
          "ssm state");
  require(std::size(v::kWantConvTail) ==
              static_cast<size_t>(g.conv_kernel - 1) * static_cast<size_t>(g.conv_dim()),
          "conv tail");
  // The vectors are not degenerate: the state carries signal and the scan
  // is not the D skip alone.
  double ssm = 0, scan = 0;
  for (const double x : v::kWantSsm) ssm = std::max(ssm, std::fabs(x));
  for (const double x : v::kWantScan) scan = std::max(scan, std::fabs(x));
  require(ssm > 1e-3 && scan > 1e-1, "the vectors carry signal");
}

DGPP_TEST(nemotron_mamba2_double_matches_the_numpy_reference) {
  // The same arithmetic in double on identical float32 inputs: the
  // difference is summation order and libm, far under 1e-12 on O(1) values.
  expect_matches(run<double>({v::kTokens}), 1e-12, "double, whole sequence");
}

DGPP_TEST(nemotron_mamba2_float_tracks_the_numpy_reference) {
  // The device's fp32 interior against the float64 reference.
  expect_matches(run<float>({v::kTokens}), 2e-5, "float, whole sequence");
}

DGPP_TEST(nemotron_mamba2_state_carries_across_calls) {
  // A prefill followed by decode steps, or any other split, performs the
  // same operations in the same order: every split is bit for bit the whole.
  const Run<double> whole = run<double>({v::kTokens});
  const std::vector<std::vector<int>> splits = {
      {4, 7}, {1, 10}, {10, 1}, {2, 3, 6}, std::vector<int>(static_cast<size_t>(v::kTokens), 1)};
  for (const auto& pieces : splits) {
    const Run<double> r = run<double>(pieces);
    require(r.out == whole.out,
            "outputs equal across a split of " + std::to_string(pieces.size()) + " calls");
    require(r.state.ssm == whole.state.ssm && r.state.conv == whole.state.conv,
            "states equal across a split");
    require(r.trace.scan == whole.trace.scan && r.trace.conv == whole.trace.conv,
            "intermediates equal across a split");
  }
  const Run<float> wf = run<float>({v::kTokens});
  const Run<float> sf = run<float>(std::vector<int>(static_cast<size_t>(v::kTokens), 1));
  require(sf.out == wf.out && sf.state.ssm == wf.state.ssm,
          "fp32: token by token equals the whole sequence");
}

DGPP_TEST(nemotron_mamba2_pieces_compose_to_the_mixer) {
  const Geometry g = geometry();
  const Weights w = weights();
  const Run<double> whole = run<double>({v::kTokens});
  State<double> st(g);
  const size_t T = static_cast<size_t>(v::kTokens);
  std::vector<double> conv(T * static_cast<size_t>(g.conv_dim())),
      y(T * static_cast<size_t>(g.inner()));
  std::vector<double> out(T * static_cast<size_t>(g.inner()));
  dgpp::mamba2_ref::conv_silu<double>(g, w, v::kProj + g.inner(), g.proj_rows(), v::kTokens,
                                      st.conv.data(), conv.data(), Rounding::None);
  dgpp::mamba2_ref::scan<double>(g, w, conv.data(), v::kProj + g.inner() + g.conv_dim(),
                                 g.proj_rows(), v::kTokens, st.ssm.data(), y.data(), nullptr,
                                 Rounding::None);
  dgpp::mamba2_ref::gated_group_rmsnorm<double>(g, y.data(), v::kProj, g.proj_rows(), w.norm_w,
                                                out.data(), v::kTokens, v::kEps, Rounding::None);
  require(out == whole.out && st.ssm == whole.state.ssm && st.conv == whole.state.conv,
          "the three pieces are the mixer");
}

DGPP_TEST(nemotron_mamba2_bf16_rounding_points) {
  // The BF16 form: the three rounded tensors hold BF16 values, the state
  // stays fp32, and the result stays within BF16's reach of the unrounded
  // run (the recurrence is contractive: one rounding per stage, 2^-8
  // relative each).
  const Run<float> r = run<float>({v::kTokens}, Rounding::Bf16);
  const auto is_bf16 = [](float x) {
    return dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(x)) == x;
  };
  require(std::all_of(r.out.begin(), r.out.end(), is_bf16), "the norm output is BF16");
  require(std::all_of(r.trace.conv.begin(), r.trace.conv.end(), is_bf16),
          "the conv output is BF16");
  require(std::all_of(r.trace.scan.begin(), r.trace.scan.end(), is_bf16),
          "the scan output is BF16");
  require(!std::all_of(r.state.ssm.begin(), r.state.ssm.end(), is_bf16),
          "the state is not rounded");
  const double d = max_abs_diff(r.out.data(), v::kWantOut, std::size(v::kWantOut));
  require(d < 5e-2, "the BF16 form stays near the reference, max |diff| " + std::to_string(d));
  require(d > 0, "the rounding is applied");
}

DGPP_TEST(nemotron_mamba2_refuses_a_bad_geometry) {
  Geometry g = geometry();
  g.groups = 3;  // does not divide four heads
  State<double> st(g);
  std::vector<double> out(static_cast<size_t>(g.inner()));
  std::vector<float> proj(static_cast<size_t>(g.proj_rows()), 0.0f);
  bool refused = false;
  try {
    dgpp::mamba2_ref::mixer<double>(g, weights(), proj.data(), 1, st, v::kEps, out.data());
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "groups must divide heads");
}
