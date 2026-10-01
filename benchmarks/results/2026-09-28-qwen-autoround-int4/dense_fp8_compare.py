"""S5 gate: the hybrid's shipped block-FP8 dense tensors against OUR at-load
recipe (loaders/fp8_quant.hpp: scale = amax(block)/448 in fp32, code =
e4m3(w/scale) RNE saturating) applied to NVIDIA's NVFP4 release, whose dense
classes are the BF16 originals. Bitwise equality = the hybrid's dense stack is
exactly what dgpp's dense_weights=fp8 template already serves."""
import json, struct, sys, os
import numpy as np

def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def index(snapshot):
    idx = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))["weight_map"]
    cache = {}
    def get(name):
        shard = os.path.join(snapshot, idx[name])
        if shard not in cache: cache[shard] = header(shard)
        h, base = cache[shard]
        info = h[name]; b, e = info["data_offsets"]
        with open(shard, "rb") as f:
            f.seek(base + b); buf = f.read(e - b)
        return info["dtype"], info["shape"], buf
    return get

# e4m3fn code table (OCP): value of every code 0..127 (positive), NaN at 0x7F.
def e4m3_values():
    vals = np.zeros(128, dtype=np.float64)
    for c in range(128):
        e = (c >> 3) & 0xF; m = c & 7
        if e == 0: vals[c] = m * 2.0 ** -9
        elif c == 0x7F: vals[c] = np.nan
        else: vals[c] = (1 + m / 8.0) * 2.0 ** (e - 7)
    return vals
VALS = e4m3_values()
FINITE = VALS[:127]  # sorted ascending, codes 0..126

def encode_e4m3(x):
    """RNE to the e4m3fn grid, saturating at 448 (values beyond map to 448)."""
    a = np.abs(x).astype(np.float64)
    a = np.minimum(a, 448.0)
    hi = np.searchsorted(FINITE, a, side="left")   # first value >= a
    hi = np.clip(hi, 1, 126)
    lo = hi - 1
    dlo = a - FINITE[lo]; dhi = FINITE[hi] - a
    pick_hi = dhi < dlo
    tie = dhi == dlo
    # ties to even: the code with an even mantissa LSB
    pick_hi = np.where(tie, (hi & 1) == 0, pick_hi)
    code = np.where(pick_hi, hi, lo).astype(np.uint8)
    code = np.where(a == FINITE[hi], hi, code).astype(np.uint8)  # exact hits
    sign = (np.signbit(x)).astype(np.uint8) << 7
    return (code | sign).astype(np.uint8)

def our_encode(w_bf16_bits, rows, cols):
    w = (w_bf16_bits.astype(np.uint32) << 16).view(np.float32).reshape(rows, cols)
    sr, sc = (rows + 127) // 128, (cols + 127) // 128
    codes = np.zeros((rows, cols), dtype=np.uint8); scales = np.zeros((sr, sc), dtype=np.float32)
    for br in range(sr):
        for bc in range(sc):
            blk = w[br*128:(br+1)*128, bc*128:(bc+1)*128]
            amax = np.float32(np.abs(blk).max())
            s = np.float32(amax / np.float32(448.0)) if amax > 0 else np.float32(1.0)
            scales[br, bc] = s
            q = (blk / s).astype(np.float32)
            codes[br*128:(br+1)*128, bc*128:(bc+1)*128] = encode_e4m3(q)
    return codes, scales

hyb = index(sys.argv[1]); nv = index(sys.argv[2])
names = sys.argv[3:]
for name in names:
    d, shape, buf = hyb(name)
    assert d == "F8_E4M3", (name, d)
    rows, cols = shape
    their_codes = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols)
    ds, sshape, sbuf = hyb(name + "_scale_inv")
    their_scales = np.frombuffer(sbuf, dtype=np.float32).reshape(sshape)
    dn, nshape, nbuf = nv(name)
    assert dn == "BF16" and nshape == shape, (name, dn, nshape)
    ours_codes, ours_scales = our_encode(np.frombuffer(nbuf, dtype=np.uint16), rows, cols)
    sc_eq = np.array_equal(their_scales, ours_scales)
    cd_eq = np.array_equal(their_codes, ours_codes)
    ndiff = int((their_codes != ours_codes).sum())
    sdiff = int((their_scales != ours_scales).sum())
    print("%s: scales %s (%d of %d differ), codes %s (%d of %d differ)" % (
        name, "bitwise" if sc_eq else "DIFFER", sdiff, their_scales.size, "bitwise" if cd_eq else "DIFFER", ndiff, their_codes.size), flush=True)
    if not cd_eq:
        i = np.argwhere(their_codes != ours_codes)[:3]
        for r, c in i: print("   e.g. (%d,%d): theirs 0x%02x ours 0x%02x" % (r, c, their_codes[r, c], ours_codes[r, c]))
