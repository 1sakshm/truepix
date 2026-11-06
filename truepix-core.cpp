// Truepix: experimental lossless image codec.
//
// Combines online least-squares prediction, parametric logistic models and
// logistic context mixing. The coder is symmetric: encoder and decoder run the
// same model, so decoding takes as long as encoding. Large images are split into
// tiles coded on separate threads, so nothing here may be shared between models.
//
// Files must decode on any machine, so the coded path uses only IEEE 754 add,
// subtract, multiply, divide and square root, which are exactly rounded everywhere.
// That rules out libm, reassociated sums and fused multiply-add, hence the flag:
//
// Build: g++ -std=c++17 -O3 -mavx2 -ffp-contract=off -static truepix-core.cpp -o truepix-core.exe
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#ifndef TRUEPIX_SERIAL
#include <thread>
#endif
#include <vector>

#if !defined(FLT_EVAL_METHOD) || FLT_EVAL_METHOD != 0
#error "Truepix needs strict float evaluation: on 32-bit x86 build with -msse2 -mfpmath=sse"
#endif
#if defined(__FAST_MATH__)
#error "Truepix must not be built with -ffast-math"
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef TRUEPIX_WASM
__attribute__((import_module("env"), import_name("tp_tick"))) void tp_tick(int samples);
#endif

static const char MAGIC[] = "TPX1";
static const int PCD_OR = -255504, PCD_OG = 199153, PCD_OB = -354277;

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;

static const int PAD = 4;

struct Image {
    int w = 0, h = 0, c = 0;
    std::vector<u8> data;
};

static Image read_pnm(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    std::vector<u8> b;
    u8 chunk[65536];
    size_t n;
    while ((n = fread(chunk, 1, sizeof chunk, f)) > 0) b.insert(b.end(), chunk, chunk + n);
    fclose(f);
    Image im;
    size_t p = 0;
    auto token = [&]() {
        while (b[p] == ' ' || b[p] == '\n' || b[p] == '\r') p++;
        size_t s = p;
        while (b[p] != ' ' && b[p] != '\n' && b[p] != '\r') p++;
        return std::string((char*)&b[s], p - s);
    };
    std::string magic = token();
    if (magic == "P7") {
        for (;;) {
            std::string t = token();
            if (t == "ENDHDR") break;
            if (t == "WIDTH") im.w = atoi(token().c_str());
            else if (t == "HEIGHT") im.h = atoi(token().c_str());
            else if (t == "DEPTH") im.c = atoi(token().c_str());
            else if (t == "MAXVAL") token();
            else if (t == "TUPLTYPE") token();
        }
    } else {
        im.c = magic == "P5" ? 1 : 3;
        im.w = atoi(token().c_str());
        im.h = atoi(token().c_str());
        token();
    }
    p++;
    im.data.assign(b.begin() + p, b.end());
    return im;
}

static void write_pnm(const char* path, const Image& im) {
    FILE* f = fopen(path, "wb");
    if (im.c == 1) fprintf(f, "P5\n%d %d\n255\n", im.w, im.h);
    else if (im.c == 3) fprintf(f, "P6\n%d %d\n255\n", im.w, im.h);
    else fprintf(f, "P7\nWIDTH %d\nHEIGHT %d\nDEPTH %d\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n", im.w, im.h, im.c);
    fwrite(im.data.data(), 1, im.data.size(), f);
    fclose(f);
}

// Carryless binary arithmetic coder with 16-bit probabilities.
struct Coder {
    u32 x1 = 0, x2 = 0xffffffff, x = 0;
    std::vector<u8>* out = nullptr;
    const u8* in = nullptr;
    size_t pos = 0, size = 0;
    int next() { return pos < size ? in[pos++] : 0; }
    void start_decode() { for (int i = 0; i < 4; i++) x = (x << 8) | next(); }
    template <bool ENC> int code(int bit, u32 p1) {
        u32 range = x2 - x1;
        u32 mid = x1 + (u32)(((uint64_t)range * p1) >> 16);
        if (!ENC) bit = x <= mid;
        if (bit) x2 = mid; else x1 = mid + 1;
        while (((x1 ^ x2) & 0xff000000) == 0) {
            if (ENC) out->push_back(x2 >> 24);
            else x = (x << 8) | next();
            x1 <<= 8;
            x2 = (x2 << 8) | 255;
        }
        return bit;
    }
    void flush() { for (int i = 0; i < 4; i++) { out->push_back(x1 >> 24); x1 <<= 8; } }
};

// Exact-by-construction log and exp, because libm results differ between platforms.
static const double LN2 = 0.693147180559945309417232;

static inline double det_exp(double x) {
    x = std::max(-40.0, std::min(40.0, x));
    double scaled = x / LN2;
    int k = (int)(scaled + (scaled < 0 ? -0.5 : 0.5));
    double r = x - k * LN2;
    double poly = 1 + r * (1 + r * (0.5 + r * (1.0 / 6 + r * (1.0 / 24 + r * (1.0 / 120 + r * (1.0 / 720 + r * (1.0 / 5040 + r / 40320)))))));
    uint64_t bits = (uint64_t)(k + 1023) << 52;
    double power;
    memcpy(&power, &bits, 8);
    return poly * power;
}

static inline double det_log(double v) {
    uint64_t bits;
    memcpy(&bits, &v, 8);
    int e = (int)((bits >> 52) & 0x7ff) - 1023;
    bits = (bits & 0xFFFFFFFFFFFFFull) | (uint64_t)1023 << 52;
    double m;
    memcpy(&m, &bits, 8);
    if (m > 1.41421356237309504880) { m *= 0.5; e++; }
    double t = (m - 1) / (m + 1), t2 = t * t;
    double series = 1 + t2 * (1.0 / 3 + t2 * (1.0 / 5 + t2 * (1.0 / 7 + t2 * (1.0 / 9 + t2 * (1.0 / 11 + t2 / 13)))));
    return 2 * t * series + e * LN2;
}

static inline float stretch(float p) { return (float)det_log((double)p / (1.0 - (double)p)); }
static inline float squash(float t) { return (float)(1.0 / (1.0 + det_exp(-(double)t))); }
static inline float clamp_p(float p) { return std::min(0.99998f, std::max(0.00002f, p)); }

// Fast table for the logistic CDF.
struct LogisticTable {
    static const int N = 8192;
    float t[N + 2];
    LogisticTable() { for (int i = 0; i <= N + 1; i++) t[i] = squash(-16.f + 32.f * i / N); }
    float operator()(float z) const {
        if (z <= -16.f) return t[0];
        if (z >= 16.f) return t[N];
        float f = (z + 16.f) * (N / 32.f);
        int i = (int)f;
        float a = f - i;
        return t[i] + (t[i + 1] - t[i]) * a;
    }
};
static LogisticTable cdf;

// Exponentially weighted recursive least squares, solved by Cholesky.
struct OLS {
    int n = 0, every = 1, count = 0;
    double lambda = 0.998, nu = 0.001;
    std::vector<double> cov, vec, w, chol, x, y, column;
    void init(int size, double lam, int solve_every) {
        n = size; lambda = lam; every = solve_every;
        cov.assign(n * n, 0); vec.assign(n, 0); w.assign(n, 0); chol.assign(n * n, 0); x.assign(n, 0);
        y.assign(n, 0); column.assign(n, 0);
    }
    double predict() const {
        double s = 0;
        const double* wi = w.data();
        const double* xi = x.data();
        for (int i = 0; i < n; i++) s += wi[i] * xi[i];
        return s;
    }
    void update(double target) {
        const double a = 1 - lambda, keep = lambda;
        const double* __restrict xs = x.data();
        double* __restrict bv = vec.data();
        double* cells = cov.data();
        const int size = n;
        for (int i = 0; i < size; i++) {
            double* __restrict row = cells + i * size;
            const double xi = xs[i] * a;
            for (int j = 0; j <= i; j++) row[j] = keep * row[j] + xi * xs[j];
            bv[i] = keep * bv[i] + xi * target;
        }
        if (++count < every) return;
        count = 0;
        solve();
    }
    // Cholesky, with chol[k * n + i] holding L(i, k). Working one column at a time gives each entry
    // the same subtractions in the same order as the row form, so results match bit for bit.
    void solve() {
        const int size = n;
        double* cells = chol.data();
        double* __restrict work = column.data();
        const double* source = cov.data();
        for (int j = 0; j < size; j++) {
            for (int i = j; i < size; i++) work[i] = source[i * size + j] + 0.0;
            work[j] += nu;
            for (int k = 0; k < j; k++) {
                const double* __restrict ck = cells + k * size;
                const double factor = ck[j];
                for (int i = j; i < size; i++) work[i] -= ck[i] * factor;
            }
            if (work[j] <= 1e-9) return;
            const double root = std::sqrt(work[j]);
            double* __restrict cj = cells + j * size;
            cj[j] = root;
            for (int i = j + 1; i < size; i++) cj[i] = work[i] / root;
        }
        double* __restrict ys = y.data();
        const double* bv = vec.data();
        for (int i = 0; i < size; i++) ys[i] = bv[i];
        for (int k = 0; k < size; k++) {
            const double* __restrict ck = cells + k * size;
            const double value = ys[k] / ck[k];
            ys[k] = value;
            for (int i = k + 1; i < size; i++) ys[i] -= ck[i] * value;
        }
        double* out = w.data();
        for (int i = size - 1; i >= 0; i--) {
            const double* ci = cells + i * size;
            double sum = ys[i];
            for (int k = i + 1; k < size; k++) sum -= ci[k] * out[k];
            out[i] = sum / ci[i];
        }
    }
};

struct Counter {
    u16 fast = 32768, slow = 32768;
    u16 n = 0;
};

static const int NCTX = 21;
static const int NPARAM = 7;
static const int NOLS = 4;
static const int NRAW = 11;
static const int NINPUT = NCTX * 2 + NPARAM + 1;
static const int NODES = 80;
static const int TABLE_BITS = 19;

static float STRETCH[4096];
static const bool stretch_ready = [] { for (int i = 0; i < 4096; i++) STRETCH[i] = stretch(clamp_p((i + 0.5f) / 4096.f)); return true; }();

struct Mixer {
    int inputs, selectors;
    std::vector<float> w;
    float* cur = nullptr;
    float dot = 0, pr = 0.5f;
    void init(int n, int s, float start) { inputs = n; selectors = s; w.assign(n * s, start); }
    float mix(const float* x, int sel) {
        const float result = weigh(x, sel);
        pr = squash(dot);
        return result;
    }
    float weigh(const float* x, int sel) {
        cur = &w[sel * inputs];
        float lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        const int n = inputs, whole = n & ~7;
        const float* c = cur;
        for (int i = 0; i < whole; i += 8)
            for (int j = 0; j < 8; j++) lane[j] += c[i + j] * x[i + j];
        float d = ((lane[0] + lane[1]) + (lane[2] + lane[3])) + ((lane[4] + lane[5]) + (lane[6] + lane[7]));
        for (int i = whole; i < n; i++) d += c[i] * x[i];
        dot = std::max(-14.f, std::min(14.f, d));
        return dot;
    }
    void update(const float* x, int bit, float lr) {
        float err = (bit - pr) * lr;
        const int n = inputs;
        float* c = cur;
        for (int i = 0; i < n; i++) c[i] += err * x[i];
    }
};

// Adaptive probability map that refines a probability from a small context.
struct APM {
    std::vector<float> t;
    int idx = 0;
    float a = 0;
    void init(int contexts) {
        t.resize(contexts * 33);
        for (int c = 0; c < contexts; c++)
            for (int i = 0; i < 33; i++) t[c * 33 + i] = squash((i - 16) * 0.75f);
    }
    float refine(float st, int ctx) {
        float pos = (std::max(-11.99f, std::min(11.99f, st)) + 12.f) / 0.75f;
        int i = (int)pos;
        a = pos - i;
        idx = ctx * 33 + i;
        return t[idx] * (1 - a) + t[idx + 1] * a;
    }
    void update(int bit, float rate) {
        t[idx] += (bit - t[idx]) * rate * (1 - a);
        t[idx + 1] += (bit - t[idx + 1]) * rate * a;
    }
};

static inline u32 hash32(u32 a, u32 b) {
    u32 h = a * 0x2F0B4C87u + b * 0x9E3779B1u + 0x7F4A7C15u;
    return h ^ (h >> 15);
}

// std::lround for the small values used while coding: halves round away from zero.
static inline int round_float(float v) {
    int t = (int)v;
    float rest = v - (float)t;
    return rest >= 0.5f ? t + 1 : rest <= -0.5f ? t - 1 : t;
}

static unsigned char QUANT_LOG[161];
static const bool quant_ready = [] {
    static const int limits[] = {0, 1, 2, 3, 5, 7, 10, 14, 20, 28, 40, 56, 80, 112, 160};
    for (int v = 0, i = 0; v <= 160; v++) {
        while (v > limits[i]) i++;
        QUANT_LOG[v] = (unsigned char)i;
    }
    return true;
}();

static inline int quant_log(int v) { return v > 160 ? 15 : QUANT_LOG[v < 0 ? 0 : v]; }

static inline int quant_signed(int v) {
    int a = std::abs(v);
    int q = a == 0 ? 0 : a == 1 ? 1 : a <= 3 ? 2 : a <= 7 ? 3 : a <= 15 ? 4 : a <= 31 ? 5 : 6;
    return v < 0 ? 7 + q : q;
}

static const int BUCKET_LOW[] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193};
static const int BUCKET_HIGH[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 255};
static const int BUCKETS = 16;

struct Model {
    int w, h, c;
    int stride;
    std::vector<u8> buf;
    std::vector<short> errs;
    std::vector<OLS> ols[NOLS];
    std::vector<std::vector<std::pair<int, int>>> offsets[NOLS];  // per plane: (dx, dy) + plane in `planes`
    std::vector<std::vector<int>> planes[NOLS];
    std::vector<std::vector<int>> reach[NOLS];  // byte offsets of the regressors from the current pixel
    std::vector<float> perr[NOLS];
    std::vector<Counter> tables[NCTX];
    Mixer mix[5], final_mix;
    APM apm[2];
    float blend[4][8][NRAW];
    float mae[4][16][NPARAM];
    float inputs[NINPUT];
    float second[5];
    int texture_ctx = 0;
    std::vector<float> bias_sum, bias_count;
    u32 ctx[NCTX];
    Counter* active[NCTX];
    double lr_scale = 1;
    int phase_ctx = 0;
    double cost[4][4] = {};
    long samples[4][4] = {};
    int cost_plane = 0, cost_phase = 0;
    long decisions = 0;

    size_t index(int x, int y, int p) const { return (size_t)(y + PAD) * stride + (x + PAD) * c + p; }
    u8& at(int x, int y, int p) { return buf[(y + PAD) * stride + (x + PAD) * c + p]; }
    short& err(int x, int y, int p) { return errs[(y + PAD) * stride + (x + PAD) * c + p]; }

    void build_regressors(int plane, const int order[4]) {
        static const int OWN[][2] = {{-1, 0}, {0, -1}, {-1, -1}, {1, -1}, {-2, 0}, {0, -2}, {1, -2}, {-2, -1}, {2, -1},
                                     {-1, -2}, {-3, 0}, {0, -3}, {2, -2}, {-2, -2}, {3, -1}, {-3, -1}, {1, -3}, {-1, -3},
                                     {3, -2}, {-3, -2}, {2, -3}, {-2, -3}, {3, -3}, {-3, -3}, {-4, 0}, {0, -4}, {1, -4}, {-1, -4}, {4, -1}, {-4, -1}};
        static const int CROSS[][2] = {{0, 0}, {-1, 0}, {0, -1}, {1, -1}, {-1, -1}, {1, 0}, {0, 1}};
        static const int OWN_N[NOLS] = {18, 10, 5, 30};
        static const int CROSS_N[NOLS] = {7, 4, 2, 7};
        static const double LAMBDA[NOLS] = {0.9985, 0.995, 0.97, 0.9995};
        static const int EVERY[NOLS] = {16, 8, 4, 32};
        for (int k = 0; k < NOLS; k++) {
            std::vector<std::pair<int, int>> off;
            std::vector<int> pl;
            for (int i = 0; i < OWN_N[k]; i++) { off.push_back({OWN[i][0], OWN[i][1]}); pl.push_back(order[plane]); }
            for (int q = 0; q < plane; q++)
                for (int i = 0; i < CROSS_N[k]; i++) {
                    // Rows below the current one are not decoded yet, so stop at the current row.
                    if (CROSS[i][1] > 0 || (CROSS[i][1] == 0 && CROSS[i][0] > 0)) continue;
                    off.push_back({CROSS[i][0], CROSS[i][1]});
                    pl.push_back(order[q]);
                }
            offsets[k][plane] = off;
            planes[k][plane] = pl;
            reach[k][plane].clear();
            for (size_t i = 0; i < off.size(); i++) reach[k][plane].push_back(off[i].second * stride + off[i].first * c + pl[i]);
            for (int phase = 0; phase < 4; phase++) ols[k][plane * 4 + phase].init((int)off.size(), LAMBDA[k], EVERY[k]);
        }
    }

    void init(int width, int height, int channels, const int order[4]) {
        w = width; h = height; c = channels;
        stride = (w + 2 * PAD) * c;
        buf.assign((h + 2 * PAD) * stride, 0);
        errs.assign((h + 2 * PAD) * stride, 0);
        for (int k = 0; k < NOLS; k++) {
            ols[k].resize(c * 4); offsets[k].resize(c); planes[k].resize(c); reach[k].resize(c);
            perr[k].assign((h + 2 * PAD) * stride, 0.f);
        }
        {
            for (int p = 0; p < c; p++) build_regressors(p, order);
        }
        for (auto& t : tables) t.assign(1 << TABLE_BITS, Counter());
        mix[0].init(NINPUT, NODES, 0.12f);
        mix[1].init(NINPUT, 32 * 16, 0.12f);
        mix[2].init(NINPUT, 32 * 4 * 8, 0.12f);
        mix[3].init(NINPUT, 32 * 16, 0.12f);
        mix[4].init(NINPUT, 32 * 256, 0.12f);
        final_mix.init(5, NODES, 0.2f);
        bias_sum.assign(4 << 12, 0.f);
        bias_count.assign(4 << 12, 0.f);
        apm[0].init(NODES * 16);
        apm[1].init(1 << 14);
        for (auto& a : blend) for (auto& b : a) for (auto& v : b) v = 0.1f;
        for (auto& a : mae) for (auto& b : a) for (auto& v : b) v = 4.f;
    }
};

#ifdef __AVX2__
static inline __m256d det_exp4(__m256d v) {
    const __m256d ln2 = _mm256_set1_pd(LN2);
    const __m256d x = _mm256_max_pd(_mm256_set1_pd(-40.0), _mm256_min_pd(_mm256_set1_pd(40.0), v));
    const __m256d scaled = _mm256_div_pd(x, ln2);
    const __m256d half = _mm256_blendv_pd(_mm256_set1_pd(0.5), _mm256_set1_pd(-0.5), _mm256_cmp_pd(scaled, _mm256_setzero_pd(), _CMP_LT_OQ));
    const __m128i k = _mm256_cvttpd_epi32(_mm256_add_pd(scaled, half));
    const __m256d r = _mm256_sub_pd(x, _mm256_mul_pd(_mm256_cvtepi32_pd(k), ln2));
    __m256d poly = _mm256_add_pd(_mm256_set1_pd(1.0 / 5040), _mm256_div_pd(r, _mm256_set1_pd(40320)));
    poly = _mm256_add_pd(_mm256_set1_pd(1.0 / 720), _mm256_mul_pd(r, poly));
    poly = _mm256_add_pd(_mm256_set1_pd(1.0 / 120), _mm256_mul_pd(r, poly));
    poly = _mm256_add_pd(_mm256_set1_pd(1.0 / 24), _mm256_mul_pd(r, poly));
    poly = _mm256_add_pd(_mm256_set1_pd(1.0 / 6), _mm256_mul_pd(r, poly));
    poly = _mm256_add_pd(_mm256_set1_pd(0.5), _mm256_mul_pd(r, poly));
    poly = _mm256_add_pd(_mm256_set1_pd(1), _mm256_mul_pd(r, poly));
    poly = _mm256_add_pd(_mm256_set1_pd(1), _mm256_mul_pd(r, poly));
    const __m256i bits = _mm256_slli_epi64(_mm256_add_epi64(_mm256_cvtepi32_epi64(k), _mm256_set1_epi64x(1023)), 52);
    return _mm256_mul_pd(poly, _mm256_castsi256_pd(bits));
}

static inline __m256d det_log4(__m256d v) {
    const __m256i bits = _mm256_castpd_si256(v);
    __m256i e = _mm256_sub_epi64(_mm256_and_si256(_mm256_srli_epi64(bits, 52), _mm256_set1_epi64x(0x7ff)), _mm256_set1_epi64x(1023));
    const __m256i tail = _mm256_and_si256(bits, _mm256_set1_epi64x(0xFFFFFFFFFFFFFll));
    __m256d m = _mm256_castsi256_pd(_mm256_or_si256(tail, _mm256_set1_epi64x((long long)1023 << 52)));
    const __m256d big = _mm256_cmp_pd(m, _mm256_set1_pd(1.41421356237309504880), _CMP_GT_OQ);
    m = _mm256_blendv_pd(m, _mm256_mul_pd(m, _mm256_set1_pd(0.5)), big);
    e = _mm256_sub_epi64(e, _mm256_castpd_si256(big));
    const __m128i low = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(e, _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0)));
    const __m256d one = _mm256_set1_pd(1);
    const __m256d t = _mm256_div_pd(_mm256_sub_pd(m, one), _mm256_add_pd(m, one));
    const __m256d t2 = _mm256_mul_pd(t, t);
    __m256d series = _mm256_add_pd(_mm256_set1_pd(1.0 / 11), _mm256_div_pd(t2, _mm256_set1_pd(13)));
    series = _mm256_add_pd(_mm256_set1_pd(1.0 / 9), _mm256_mul_pd(t2, series));
    series = _mm256_add_pd(_mm256_set1_pd(1.0 / 7), _mm256_mul_pd(t2, series));
    series = _mm256_add_pd(_mm256_set1_pd(1.0 / 5), _mm256_mul_pd(t2, series));
    series = _mm256_add_pd(_mm256_set1_pd(1.0 / 3), _mm256_mul_pd(t2, series));
    series = _mm256_add_pd(one, _mm256_mul_pd(t2, series));
    const __m256d scaled = _mm256_mul_pd(_mm256_add_pd(t, t), series);
    return _mm256_add_pd(scaled, _mm256_mul_pd(_mm256_cvtepi32_pd(low), _mm256_set1_pd(LN2)));
}

static inline __m128 squash4(__m128 t) {
    const __m256d negated = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_cvtps_pd(t));
    const __m256d one = _mm256_set1_pd(1);
    return _mm256_cvtpd_ps(_mm256_div_pd(one, _mm256_add_pd(one, det_exp4(negated))));
}

static inline __m128 stretch4(__m128 p) {
    const __m256d wide = _mm256_cvtps_pd(p);
    return _mm256_cvtpd_ps(det_log4(_mm256_div_pd(wide, _mm256_sub_pd(_mm256_set1_pd(1), wide))));
}

static inline __m256 cdf8(__m256 z) {
    const __m256 f = _mm256_mul_ps(_mm256_add_ps(z, _mm256_set1_ps(16.f)), _mm256_set1_ps(LogisticTable::N / 32.f));
    const __m256 low = _mm256_cmp_ps(z, _mm256_set1_ps(-16.f), _CMP_LE_OQ);
    const __m256 high = _mm256_cmp_ps(z, _mm256_set1_ps(16.f), _CMP_GE_OQ);
    const __m256i i = _mm256_andnot_si256(_mm256_castps_si256(_mm256_or_ps(low, high)), _mm256_cvttps_epi32(f));
    const __m256 a = _mm256_sub_ps(f, _mm256_cvtepi32_ps(i));
    const __m256 t0 = _mm256_i32gather_ps(cdf.t, i, 4), t1 = _mm256_i32gather_ps(cdf.t + 1, i, 4);
    const __m256 mixed = _mm256_add_ps(t0, _mm256_mul_ps(_mm256_sub_ps(t1, t0), a));
    return _mm256_blendv_ps(_mm256_blendv_ps(mixed, _mm256_set1_ps(cdf.t[0]), low), _mm256_set1_ps(cdf.t[LogisticTable::N]), high);
}
#endif

// Parametric mixture state for one sample.
struct alignas(32) Params {
    float mu[8] = {}, s[8] = {0, 0, 0, 0, 0, 0, 0, 1.f};  // NPARAM, padded to a whole vector
#ifdef __AVX2__
    static const int EDGES = 6;
    mutable __m256 edge_cdf[EDGES];
    mutable int edge_at[EDGES];
    mutable int edges = 0;
    // The logistic CDF at v - 0.5 for all mixtures, kept because decisions share their edges.
    __m256 edge(int v) const {
        for (int i = 0; i < edges; i++) if (edge_at[i] == v) return edge_cdf[i];
        const __m256 z = _mm256_div_ps(_mm256_sub_ps(_mm256_set1_ps(v - 0.5f), _mm256_load_ps(mu)), _mm256_load_ps(s));
        const __m256 value = cdf8(z);
        if (edges < EDGES) { edge_at[edges] = v; edge_cdf[edges++] = value; }
        return value;
    }
    __m256 mass8(int lo, int hi) const { return _mm256_sub_ps(edge(hi + 1), edge(lo)); }
#endif
    float mass(int i, int lo, int hi) const {
        return cdf((hi + 0.5f - mu[i]) / s[i]) - cdf((lo - 0.5f - mu[i]) / s[i]);
    }
};

template <bool ENC>
static int code_decision(Model& m, Coder& coder, int bit, int node, const Params& par, int plo1, int phi1, int plo2, int phi2,
                         int energy, int plane) {
    // P(bit = 1) is the mass of [plo1, phi1] within that interval plus [plo2, phi2].
    float* in = m.inputs;
    int k = 0;
    for (int i = 0; i < NCTX; i++) {
        m.active[i] = &m.tables[i][(m.ctx[i] + (u32)node * 0x9E3779B1u) >> (32 - TABLE_BITS)];
        __builtin_prefetch(m.active[i], 1);
    }
    for (int i = 0; i < NCTX; i++) {
        Counter* e = m.active[i];
        in[k++] = STRETCH[(int)e->fast >> 4];
        in[k++] = STRETCH[(int)e->slow >> 4];
    }
#ifdef __AVX2__
    {
        const __m256 yes = par.mass8(plo1, phi1);
        const __m256 no = phi2 >= plo2 ? par.mass8(plo2, phi2)
                                       : _mm256_max_ps(_mm256_sub_ps(par.mass8(0, 255), yes), _mm256_setzero_ps());
        __m256 p = _mm256_div_ps(_mm256_add_ps(yes, _mm256_set1_ps(1e-7f)),
                                 _mm256_add_ps(_mm256_add_ps(yes, no), _mm256_set1_ps(2e-7f)));
        p = _mm256_min_ps(_mm256_set1_ps(0.99998f), _mm256_max_ps(_mm256_set1_ps(0.00002f), p));
        alignas(32) float out[8];
        _mm_store_ps(out, stretch4(_mm256_castps256_ps128(p)));
        _mm_store_ps(out + 4, stretch4(_mm256_extractf128_ps(p, 1)));
        for (int i = 0; i < NPARAM; i++) in[k++] = out[i];
    }
#else
    for (int i = 0; i < NPARAM; i++) {
        float yes = par.mass(i, plo1, phi1);
        float no = phi2 >= plo2 ? par.mass(i, plo2, phi2) : std::max(0.f, par.mass(i, 0, 255) - yes);
        float p = (yes + 1e-7f) / (yes + no + 2e-7f);
        in[k++] = stretch(clamp_p(p));
    }
#endif
    in[k++] = 0.3f;

    int nc = std::min(node, 31);
    m.second[0] = m.mix[0].weigh(in, node);
    m.second[1] = m.mix[1].weigh(in, nc * 16 + energy);
    m.second[2] = m.mix[2].weigh(in, (nc * 4 + plane) * 8 + (m.ctx[2] & 7));
    m.second[3] = m.mix[3].weigh(in, nc * 16 + m.phase_ctx);
    m.second[4] = m.mix[4].weigh(in, nc * 256 + m.texture_ctx);
#ifdef __AVX2__
    {
        alignas(16) float squashed[4];
        _mm_store_ps(squashed, squash4(_mm_loadu_ps(m.second)));
        for (int i = 0; i < 4; i++) m.mix[i].pr = squashed[i];
        m.mix[4].pr = squash(m.second[4]);
    }
#else
    for (int i = 0; i < 5; i++) m.mix[i].pr = squash(m.second[i]);
#endif
    float st = m.final_mix.mix(m.second, node);
    float p0 = m.final_mix.pr;  // the same squash of the same value
    float a0 = m.apm[0].refine(st, node * 16 + energy);
    float a1 = m.apm[1].refine(st, (m.ctx[7] >> 18) & 0x3fff);
    float p = 0.5f * p0 + 0.25f * a0 + 0.25f * a1;
    u32 p16 = (u32)(clamp_p(p) * 65536.f);
    p16 = std::min<u32>(65535 - 16, std::max<u32>(16, p16));
    bit = coder.code<ENC>(bit, p16);
    if (ENC) m.cost[m.cost_plane][m.cost_phase] -= std::log2((bit ? p16 : 65536 - p16) / 65536.0);

    float lr = (float)(0.0015 + 0.02 * m.lr_scale);
    for (int i = 0; i < 5; i++) m.mix[i].update(in, bit, lr);
    m.final_mix.update(m.second, bit, 0.001f);
    m.apm[0].update(bit, 0.02f);
    m.apm[1].update(bit, 0.02f);
    // Done in arrays so the divisions run side by side.
    int fast[NCTX], slow[NCTX], seen[NCTX];
    for (int i = 0; i < NCTX; i++) {
        const Counter* e = m.active[i];
        fast[i] = e->fast; slow[i] = e->slow; seen[i] = e->n;
    }
    const int target = bit ? 65535 : 0;
    for (int i = 0; i < NCTX; i++) {
        fast[i] = (int)((float)fast[i] + (float)(target - fast[i]) / ((float)std::min(seen[i], 30) + 1.5f));
        slow[i] = (int)((float)slow[i] + (float)(target - slow[i]) / ((float)std::min(seen[i], 250) + 1.5f));
    }
    for (int i = 0; i < NCTX; i++) {
        Counter* e = m.active[i];
        e->fast = (u16)fast[i]; e->slow = (u16)slow[i];
        if (seen[i] < 1000) e->n = (u16)(seen[i] + 1);
    }
    m.decisions++;
    return bit;
}

template <bool ENC>
static void run(Model& m, Coder& coder, const Image* src, const int order[4]) {
    const int w = m.w, h = m.h, c = m.c;
    float raw[NRAW];
    int total = w * h * c, done = 0;
    for (int y = 0; y < h; y++) {
        for (int pl = 0; pl < c; pl++) m.at(-1, y, pl) = m.at(-2, y, pl) = m.at(-3, y, pl) = y ? m.at(0, y - 1, pl) : 128;
        for (int x = 0; x < w; x++) {
            for (int idx = 0; idx < c; idx++) {
                int pl = order[idx];
                int W = m.at(x - 1, y, pl), N = m.at(x, y - 1, pl), NW = m.at(x - 1, y - 1, pl), NE = m.at(x + 1, y - 1, pl);
                int WW = m.at(x - 2, y, pl), NN = m.at(x, y - 2, pl), NNE = m.at(x + 1, y - 2, pl);
                double base = (W + N) * 0.5;
                int phase = (x & 1) | (y & 1) << 1;
                int slot[NOLS];
                for (int k = 0; k < NOLS; k++) slot[k] = idx * 4 + (idx > 0 && k < 2 ? phase : 0);
                for (int k = 0; k < NOLS; k++) {
                    OLS& o = m.ols[k][slot[k]];
                    const int* steps = m.reach[k][idx].data();
                    const u8* here = &m.at(x, y, 0);
                    double* regressors = o.x.data();
                    for (int i = 0; i < o.n; i++) regressors[i] = here[steps[i]] - base;
                }
                int med = W + N - NW;
                int lo_n = std::min(W, N), hi_n = std::max(W, N);
                med = med < lo_n ? lo_n : med > hi_n ? hi_n : med;

                float ols_p[NOLS];
                for (int k = 0; k < NOLS; k++) ols_p[k] = std::max(-32.f, std::min(287.f, (float)(m.ols[k][slot[k]].predict() + base)));
                for (int k = 0; k < 3; k++) raw[k] = ols_p[k];
                // Favour whichever predictor has been most accurate right here.
                {
                    float sum = 0, total_w = 0;
                    size_t at = m.index(x, y, pl);
                    for (int k = 0; k < NOLS; k++) {
                        const float* e = m.perr[k].data();
                        float a = e[at - c] + e[at - m.stride] + 0.5f * (e[at - m.stride - c] + e[at - m.stride + c]) + 0.5f * e[at - 2 * c] + 0.5f * e[at - 2 * m.stride];
                        float weight = 1.f / (a * a + 0.5f);
                        sum += weight * ols_p[k];
                        total_w += weight;
                    }
                    raw[10] = sum / total_w;
                }
                raw[3] = (float)med;
                raw[4] = (float)W;
                raw[5] = (float)N;
                raw[6] = (W + N) * 0.5f + (NE - NW) * 0.25f;
                if (idx > 0) {
                    int q = order[idx - 1];
                    int C0 = m.at(x, y, q), W0 = m.at(x - 1, y, q), N0 = m.at(x, y - 1, q), NW0 = m.at(x - 1, y - 1, q);
                    raw[7] = (float)(W + C0 - W0);
                    raw[8] = (float)(N + C0 - N0);
                    int d = (W - W0) + (N - N0) - (NW - NW0);
                    int a = W - W0, b = N - N0;
                    d = d < std::min(a, b) ? std::min(a, b) : d > std::max(a, b) ? std::max(a, b) : d;
                    raw[9] = (float)(C0 + d);
                } else {
                    raw[7] = (W + NE) * 0.5f;
                    raw[8] = (float)(N + NE - NNE);
                    raw[9] = (float)(2 * W - WW);
                }

                int e_w = m.err(x - 1, y, pl), e_n = m.err(x, y - 1, pl), e_ne = m.err(x + 1, y - 1, pl), e_nw = m.err(x - 1, y - 1, pl);
                int grad = std::abs(W - NW) + std::abs(N - NW) + std::abs(N - NE) + std::abs(W - WW) + std::abs(N - NN);
                int noise = std::abs(e_w) + std::abs(e_n) + std::abs(e_ne) + std::abs(e_nw);
                int prev_e = idx > 0 ? m.err(x, y, order[idx - 1]) : 0;
                int active = grad / 2 + noise + 2 * std::abs(prev_e);
                int energy = quant_log(active);
                int e8 = energy / 2;

                float* bw = m.blend[idx][e8];
                float blended = (float)base, norm = 1.f;
                float d[NRAW];
                for (int i = 0; i < NRAW; i++) {
                    d[i] = raw[i] - (float)base;
                    blended += bw[i] * d[i];
                    norm += d[i] * d[i];
                }
                u32 tex_bias = ((W > blended) | (N > blended) << 1 | (NW > blended) << 2 | (NE > blended) << 3 |
                                (WW > blended) << 4 | (NN > blended) << 5 | (e_w > 0) << 6 | (e_n > 0) << 7);
                int bias_slot = idx << 12 | (int)tex_bias << 4 | energy;
                if (m.bias_count[bias_slot] > 0) blended += m.bias_sum[bias_slot] / m.bias_count[bias_slot];
                blended = std::max(0.f, std::min(255.f, blended));
                int P = round_float(blended);

                Params par;
                static const int PICK[6] = {0, 1, 10, 3, 7, 8};
                for (int i = 0; i < 6; i++) par.mu[i] = raw[PICK[i]];
                par.mu[6] = blended;
                float* scale = m.mae[idx][energy];
                for (int i = 0; i < NPARAM; i++) par.s[i] = std::max(0.15f, scale[i] * 0.72f);

                u32 pmix = (u32)idx << 28;
                int rW = W - P, rN = N - P, rNW = NW - P, rNE = NE - P, rWW = WW - P, rNN = NN - P;
                u32 tex = (rW > 0) | (rN > 0) << 1 | (rNW > 0) << 2 | (rNE > 0) << 3 | (rWW > 0) << 4 | (rNN > 0) << 5 |
                          (e_w > 0) << 6 | (e_n > 0) << 7;
                int q0 = idx > 0 ? m.at(x, y, order[idx - 1]) : 0;
                m.ctx[0] = hash32(pmix + 1, energy);
                m.ctx[1] = hash32(pmix + 2, quant_log(noise) * 16 + quant_log(std::abs(prev_e)));
                m.ctx[2] = hash32(pmix + 3, tex) & ~7u;
                m.ctx[2] |= std::min(7, std::abs(P - W) + std::abs(P - N));
                m.ctx[3] = hash32(pmix + 4, quant_signed(prev_e) * 16 + quant_log(active / 2));
                m.ctx[4] = hash32(pmix + 5, quant_signed(e_w) * 16 + quant_signed(e_n));
                m.ctx[5] = hash32(hash32(pmix + 6, W | N << 8 | NW << 16 | (u32)NE << 24), P);
                m.ctx[6] = hash32(hash32(pmix + 7, W | N << 8 | q0 << 16), P);
                m.ctx[7] = hash32(pmix + 8, P / 8 + 64 * energy);
                m.ctx[8] = hash32(pmix + 9, quant_signed((int)raw[0] - P) * 16 + quant_signed((int)raw[1] - P) + 256 * quant_signed((int)raw[3] - P));
                m.ctx[9] = hash32(hash32(pmix + 10, rW + 300 + (rN + 300) * 1024), rNE + 300 + (u32)(rNW + 300) * 1024);
                m.ctx[10] = hash32(pmix + 11, phase * 256 + energy * 16 + quant_log(std::abs(prev_e)));
                m.ctx[11] = hash32(pmix + 12, (x & 7) * 8 + (y & 7) + 64 * e8);
                m.phase_ctx = phase * 4 + idx;
                m.texture_ctx = tex;
                int spread = 0, lowest = 1000, highest = -1000;
                for (int k = 0; k < NOLS; k++) {
                    int v = round_float(ols_p[k]) - P;
                    lowest = std::min(lowest, v);
                    highest = std::max(highest, v);
                }
                spread = highest - lowest;
                m.ctx[12] = hash32(pmix + 13, quant_signed(W - NW) * 256 + quant_signed(N - NW) * 16 + quant_signed(NE - N));
                m.ctx[13] = hash32(pmix + 14, quant_log(spread) * 256 + quant_signed(lowest) * 16 + quant_signed(highest));
                m.ctx[14] = hash32(pmix + 15, quant_signed(e_w) * 256 + quant_signed(e_n) * 16 + quant_signed(e_ne));
                m.ctx[15] = hash32(pmix + 16, quant_signed(WW - 2 * W + P) * 256 + quant_signed(NN - 2 * N + P) * 16 + energy);
                m.ctx[16] = hash32(pmix + 17, quant_signed(W + NE - N - P) * 256 + quant_signed(N + NW - W - P) * 16 + quant_signed(N + NE - NNE - P));
                m.ctx[17] = hash32(hash32(pmix + 18, P), quant_log(active));
                // Exact colour differences: a counter here learns which differences occur at all,
                // so any colour grid is picked up with no knowledge of how it was made.
                {
                    int first = order[0];
                    int g = idx ? m.at(x, y, first) : 0, gW = m.at(x - 1, y, first), gN = m.at(x, y - 1, first);
                    int d = P - g + 300, known = idx == 2 ? m.at(x, y, order[1]) - g + 300 : 0;
                    int dW = W - gW + 300, dN = N - gN + 300;
                    m.ctx[18] = hash32(hash32(pmix + 19, d), known);
                    m.ctx[19] = hash32(hash32(pmix + 20, d + dW * 1024), known);
                    m.ctx[20] = hash32(hash32(pmix + 21, d + dW * 1024 + (u32)dN * (1u << 20)), known);
                }

                int value = ENC ? src->data[(y * w + x) * c + pl] : 0;
                m.cost_plane = idx;
                m.cost_phase = phase;
                m.samples[idx][phase]++;
                int bit;
                // Is the sample exactly the prediction?
                bit = code_decision<ENC>(m, coder, ENC ? value == P : 0, 0, par, P, P, 0, -1, energy, idx);
                if (bit) {
                    value = P;
                } else {
                    int up_room = 255 - P, down_room = P;
                    int positive;
                    if (up_room == 0) positive = 0;
                    else if (down_room == 0) positive = 1;
                    else {
                        positive = code_decision<ENC>(m, coder, ENC ? value > P : 0, 1, par, P + 1, 255, 0, P - 1, energy, idx);
                    }
                    int room = positive ? up_room : down_room;
                    int side = positive ? 0 : 1;
                    int mag = ENC ? std::abs(value - P) : 0;
                    int b = 0;
                    auto low_of = [&](int a, int z) { return positive ? P + a : P - z; };
                    auto high_of = [&](int a, int z) { return positive ? P + z : P - a; };
                    for (;; b++) {
                        int z = std::min(BUCKET_HIGH[b], room);
                        if (z >= room || b == BUCKETS - 1) break;
                        int a = BUCKET_LOW[b];
                        int restart = z + 1;
                        int in_bucket = code_decision<ENC>(m, coder, ENC ? mag <= z : 0, 2 + side * 16 + b, par, low_of(a, z), high_of(a, z),
                                                           low_of(restart, room), high_of(restart, room), energy, idx);
                        if (in_bucket) break;
                    }
                    int a = BUCKET_LOW[b], z = std::min(BUCKET_HIGH[b], room);
                    int depth = 0;
                    while (a < z) {
                        int mid = (a + z) / 2;
                        int lower = code_decision<ENC>(m, coder, ENC ? mag <= mid : 0, 34 + side * 20 + std::min(b, 15) / 2 * 2 + std::min(depth, 1) , par,
                                                       low_of(a, mid), high_of(a, mid), low_of(mid + 1, z), high_of(mid + 1, z), energy, idx);
                        if (lower) z = mid; else a = mid + 1;
                        depth++;
                    }
                    mag = a;
                    value = positive ? P + mag : P - mag;
                }

                m.at(x, y, pl) = (u8)value;
                m.err(x, y, pl) = (short)(value - P);
                for (int k = 0; k < NOLS; k++) {
                    m.ols[k][slot[k]].update(value - base);
                    m.perr[k][m.index(x, y, pl)] = std::abs((float)value - ols_p[k]);
                }
                float e = (float)value - blended;
                m.bias_sum[bias_slot] += e;
                m.bias_count[bias_slot] += 1;
                if (m.bias_count[bias_slot] > 255) { m.bias_sum[bias_slot] *= 0.5f; m.bias_count[bias_slot] *= 0.5f; }
                float gain = 0.002f * e / norm * 8.f;
                for (int i = 0; i < NRAW; i++) bw[i] += gain * d[i];
                for (int i = 0; i < NPARAM; i++) scale[i] += (std::abs((float)value - par.mu[i]) - scale[i]) * 0.008f;
                done++;
#ifdef TRUEPIX_WASM
                if ((done & 0x3fff) == 0) tp_tick(0x4000);
#endif
                m.lr_scale = std::max(0.0, 1.0 - (double)done / std::min(total, 400000));
            }
            if (x == w - 1)
                for (int pl = 0; pl < c; pl++)
                    for (int i = 1; i < PAD; i++) m.at(w - 1 + i, y, pl) = m.at(w - 1, y, pl);
        }
    }
}

// libjpeg's fixed-point YCbCr to RGB conversion.
static inline void ycc_to_rgb(int y, int cb, int cr, int& r, int& g, int& b) {
    auto clip = [](int v) { return v < 0 ? 0 : v > 255 ? 255 : v; };
    const int half = 1 << 15;
    r = clip(y + ((91881 * (cr - 128) + half) >> 16));
    g = clip(y + ((-22554 * (cb - 128) + half - 46802 * (cr - 128)) >> 16));
    b = clip(y + ((116130 * (cb - 128) + half) >> 16));
}

// A decoded JPEG's RGB values lie on the image of this conversion, which covers
// only about a quarter of all colours. Inverting it recovers the original
// YCbCr samples, whose chroma is far more compressible.
static void invert_lattice(const Image& src, Image& ycc, std::vector<signed char>& delta, long& hits) {
    ycc = src;
    delta.assign((size_t)src.w * src.h * 3, 0);
    hits = 0;
    size_t count = (size_t)src.w * src.h;
    for (size_t i = 0; i < count; i++) {
        const u8* px = &src.data[i * 3];
        int r = px[0], g = px[1], b = px[2];
        int y0 = (int)std::lround(0.299 * r + 0.587 * g + 0.114 * b);
        int b0 = (int)std::lround(-0.168736 * r - 0.331264 * g + 0.5 * b + 128);
        int r0 = (int)std::lround(0.5 * r - 0.418688 * g - 0.081312 * b + 128);
        int best = 1 << 30, by = y0, bb = b0, br = r0;
        for (int dy = -3; dy <= 3; dy++)
            for (int db = -3; db <= 3; db++)
                for (int dr = -3; dr <= 3; dr++) {
                    int yy = y0 + dy, cb = b0 + db, cr = r0 + dr;
                    if (yy < 0 || yy > 255 || cb < 0 || cb > 255 || cr < 0 || cr > 255) continue;
                    int x, s, t;
                    ycc_to_rgb(yy, cb, cr, x, s, t);
                    int dist = dy * dy + db * db + dr * dr;
                    if (x == r && s == g && t == b && dist < best) { best = dist; by = yy; bb = cb; br = cr; }
                }
        if (best < (1 << 30)) hits++;
        by = std::min(255, std::max(0, by));
        bb = std::min(255, std::max(0, bb));
        br = std::min(255, std::max(0, br));
        ycc.data[i * 3] = (u8)by;
        ycc.data[i * 3 + 1] = (u8)bb;
        ycc.data[i * 3 + 2] = (u8)br;
        int x, s, t;
        ycc_to_rgb(by, bb, br, x, s, t);
        delta[i * 3] = (signed char)(r - x);
        delta[i * 3 + 1] = (signed char)(g - s);
        delta[i * 3 + 2] = (signed char)(b - t);
    }
}

// Pixels off the lattice are rare and clustered, so a small adaptive model is enough.
template <bool ENC>
static void code_deltas(Coder& coder, std::vector<signed char>& delta, int w, int h) {
    std::vector<u16> flag(16, 32768), tree(3 * 256, 32768);
    std::vector<u8> marks((size_t)w * h, 0);
    auto bit_of = [&](u16& p, int bit) {
        u32 prob = std::min<u32>(65000, std::max<u32>(500, p));
        bit = coder.code<ENC>(bit, prob);
        p += ((bit ? 65535 : 0) - p) / 16;
        return bit;
    };
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            size_t i = (size_t)y * w + x;
            int left = x ? marks[i - 1] : 0, up = y ? marks[i - w] : 0, ul = x && y ? marks[i - w - 1] : 0;
            int ctx = left | up << 1 | ul << 2;
            int any = ENC ? (delta[i * 3] || delta[i * 3 + 1] || delta[i * 3 + 2]) : 0;
            any = bit_of(flag[ctx], any);
            marks[i] = (u8)any;
            if (!any) continue;
            for (int k = 0; k < 3; k++) {
                int value = ENC ? (u8)delta[i * 3 + k] : 0, node = 1;
                for (int b = 7; b >= 0; b--) {
                    int bit = bit_of(tree[k * 256 + node], ENC ? (value >> b) & 1 : 0);
                    node = node * 2 + bit;
                }
                if (!ENC) delta[i * 3 + k] = (signed char)(node - 256);
            }
        }
}

// Photo CD's fixed-point PhotoYCC to RGB conversion, with gains and offsets fitted to pixels.
static inline void pcd_to_rgb(int y, int c1, int c2, int& r, int& g, int& b) {
    auto clip = [](int v) { return v < 0 ? 0 : v > 255 ? 255 : v; };
    int luma = 1391 * y;
    r = clip((luma + 1865 * c2 + PCD_OR) >> 10);
    g = clip((luma - 441 * c1 - 949 * c2 + PCD_OG) >> 10);
    b = clip((luma + 2271 * c1 + PCD_OB) >> 10);
}

// Same idea as invert_lattice, with a wider search because this map stretches colours
// and clipped pixels sit far from the ideal inverse. Colours repeat, so results are cached.
static void invert_pcd(const Image& src, Image& ycc, std::vector<signed char>& delta, long& hits) {
    ycc = src;
    delta.assign((size_t)src.w * src.h * 3, 0);
    hits = 0;
    struct Found { u8 y, c1, c2; signed char d[3]; bool hit; };
    std::unordered_map<u32, Found> cache;
    size_t count = (size_t)src.w * src.h;
    for (size_t i = 0; i < count; i++) {
        const u8* px = &src.data[i * 3];
        int r = px[0], g = px[1], b = px[2];
        u32 key = r << 16 | g << 8 | b;
        auto it = cache.find(key);
        if (it == cache.end()) {
            double dr = r - g, db = b - g;
            // Solve the 2x2 system dr = 2.7482 d2 + 0.4307 d1, db = 0.9268 d2 + 2.6486 d1 (1024ths of the integer gains).
            double a = (1865 + 949) / 1024.0, bb = 441 / 1024.0, cc = 949 / 1024.0, dd = (2271 + 441) / 1024.0;
            double det = a * dd - bb * cc;
            double d2 = (dr * dd - db * bb) / det, d1 = (db * a - dr * cc) / det;
            int c10 = (int)std::lround(d1 + 156), c20 = (int)std::lround(d2 + 137);
            int y0 = (int)std::lround((g * 1024.0 + 441 * c10 + 949 * c20 - PCD_OG) / 1391.0);
            int best = 1 << 30;
            Found f{};
            for (int span : {3, 9}) {
                for (int e = -span; e <= span; e++)
                    for (int m = -span; m <= span; m++)
                        for (int n = -span; n <= span; n++) {
                            int yy = y0 + e, c1 = c10 + m, c2 = c20 + n;
                            if (yy < 0 || yy > 255 || c1 < 0 || c1 > 255 || c2 < 0 || c2 > 255) continue;
                            int x, t, w;
                            pcd_to_rgb(yy, c1, c2, x, t, w);
                            int miss = std::abs(x - r) + std::abs(t - g) + std::abs(w - b);
                            int cost = miss * 100000 + e * e + m * m + n * n;
                            if (cost < best) {
                                best = cost;
                                f.y = (u8)yy; f.c1 = (u8)c1; f.c2 = (u8)c2;
                                f.d[0] = (signed char)(r - x); f.d[1] = (signed char)(g - t); f.d[2] = (signed char)(b - w);
                                f.hit = miss == 0;
                            }
                        }
                if (f.hit) break;
            }
            it = cache.emplace(key, f).first;
        }
        const Found& f = it->second;
        if (f.hit) hits++;
        ycc.data[i * 3] = f.y;
        ycc.data[i * 3 + 1] = f.c1;
        ycc.data[i * 3 + 2] = f.c2;
        for (int k = 0; k < 3; k++) delta[i * 3 + k] = f.d[k];
    }
}

static void plane_order(int mode, int channels, int order[4]) {
    for (int i = 0; i < 4; i++) order[i] = i;
    if (mode == 0 && channels >= 3) { order[0] = 1; order[1] = 0; }
}

// One pixel in 61 is enough to rule out both fixed grids.
static bool grid_unlikely(const Image& im) {
    Image thin;
    size_t count = (size_t)im.w * im.h;
    thin.w = (int)(count / 61);
    thin.h = 1;
    thin.c = 3;
    if (thin.w < 1000) return false;
    thin.data.resize((size_t)thin.w * 3);
    for (int i = 0; i < thin.w; i++)
        for (int k = 0; k < 3; k++) thin.data[(size_t)i * 3 + k] = im.data[(size_t)i * 61 * 3 + k];
    Image out;
    std::vector<signed char> delta;
    long a = 0, b = 0;
    invert_lattice(thin, out, delta, a);
    invert_pcd(thin, out, delta, b);
    return std::max(a, b) < thin.w * 0.5;
}

static std::vector<u8> encode_mode(const Image& im, int mode, std::vector<signed char>* delta) {
    int order[4];
    plane_order(mode, im.c, order);
    std::vector<u8> out;
    for (int v : {im.w, im.h}) for (int i = 0; i < 4; i++) out.push_back(v >> (8 * i));
    out.push_back(im.c);
    out.push_back(mode);
    Model* m = new Model();
    m->init(im.w, im.h, im.c, order);
    Coder coder;
    coder.out = &out;
    run<true>(*m, coder, &im, order);
    if (mode >= 1) code_deltas<true>(coder, *delta, im.w, im.h);
    coder.flush();
    if (getenv("Truepix_STATS")) {
        for (int a = 0; a < im.c; a++) {
            printf("mode %d plane %d:", mode, order[a]);
            for (int b = 0; b < 4; b++) printf("  phase %d %.3f b/s", b, m->cost[a][b] / std::max(1L, m->samples[a][b]));
            printf("\n");
        }
    }
    delete m;
    return out;
}

static std::vector<u8> compress(const Image& im, int& chosen) {
    std::vector<u8> best = encode_mode(im, 0, nullptr);
    chosen = 0;
    if (im.c == 3 && !grid_unlikely(im)) {
        Image ycc;
        std::vector<signed char> delta;
        long hits = 0;
        invert_lattice(im, ycc, delta, hits);
        double share = (double)hits / ((double)im.w * im.h);
        if (getenv("Truepix_STATS")) printf("lattice share %.4f\n", share);
        if (share > 0.6) {
            std::vector<u8> other = encode_mode(ycc, 1, &delta);
            if (other.size() < best.size()) { best.swap(other); chosen = 1; }
        }
        invert_pcd(im, ycc, delta, hits);
        share = (double)hits / ((double)im.w * im.h);
        if (getenv("Truepix_STATS")) printf("photo cd share %.4f\n", share);
        if (share > 0.6) {
            std::vector<u8> other = encode_mode(ycc, 2, &delta);
            if (other.size() < best.size()) { best.swap(other); chosen = 2; }
        }
    }
    return best;
}

// Runs task(0) to task(count - 1), at once where threads exist.
template <class Task>
static void run_tiles(int count, Task task) {
#ifdef TRUEPIX_SERIAL
    for (int t = 0; t < count; t++) task(t);
#else
    std::vector<std::thread> threads;
    for (int t = 0; t < count; t++) threads.emplace_back(task, t);
    for (auto& thread : threads) thread.join();
#endif
}

static u32 crc32_of(const std::vector<u8>& data) {
    static const std::vector<u32> table = [] {
        std::vector<u32> t(256);
        for (u32 i = 0; i < 256; i++) {
            u32 c = i;
            for (int k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    u32 c = 0xFFFFFFFFu;
    for (u8 v : data) c = table[(c ^ v) & 255] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

static u32 get32(const u8* b) { return b[0] | b[1] << 8 | b[2] << 16 | (u32)b[3] << 24; }

// A tile must be exactly the rectangle the file header implies, because the model
// and the copy back into the image are both sized from it.
static bool tile_valid(const u8* b, size_t size, int w, int h, int c) {
    if (size < 10) return false;
    int mode = b[9];
    return (int)get32(b) == w && (int)get32(b + 4) == h && b[8] == c && mode <= 2 && (mode == 0 || c == 3);
}

static Image expand(const u8* b, size_t size) {
    Image im;
    im.w = (int)get32(b);
    im.h = (int)get32(b + 4);
    im.c = b[8];
    int mode = b[9], order[4];
    plane_order(mode, im.c, order);
    Model* m = new Model();
    m->init(im.w, im.h, im.c, order);
    Coder coder;
    coder.in = b + 10;
    coder.size = size - 10;
    coder.start_decode();
    run<false>(*m, coder, nullptr, order);
    im.data.resize((size_t)im.w * im.h * im.c);
    for (int y = 0; y < im.h; y++)
        for (int x = 0; x < im.w; x++)
            for (int p = 0; p < im.c; p++) im.data[(y * im.w + x) * im.c + p] = m->at(x, y, p);
    if (mode >= 1) {
        std::vector<signed char> delta((size_t)im.w * im.h * 3, 0);
        code_deltas<false>(coder, delta, im.w, im.h);
        for (size_t i = 0; i < (size_t)im.w * im.h; i++) {
            int r, g, bl;
            if (mode == 1) ycc_to_rgb(im.data[i * 3], im.data[i * 3 + 1], im.data[i * 3 + 2], r, g, bl);
            else pcd_to_rgb(im.data[i * 3], im.data[i * 3 + 1], im.data[i * 3 + 2], r, g, bl);
            im.data[i * 3] = (u8)(r + delta[i * 3]);
            im.data[i * 3 + 1] = (u8)(g + delta[i * 3 + 1]);
            im.data[i * 3 + 2] = (u8)(bl + delta[i * 3 + 2]);
        }
    }
    delete m;
    return im;
}

// Tile t of a side-by-side split, as a half-open rectangle.
static void tile_box(int w, int h, int side, int t, int& x0, int& y0, int& x1, int& y1) {
    int i = t % side, j = t / side;
    x0 = (int)((long long)w * i / side);
    x1 = (int)((long long)w * (i + 1) / side);
    y0 = (int)((long long)h * j / side);
    y1 = (int)((long long)h * (j + 1) / side);
}

// Packs an image into a complete Truepix file.
static std::vector<u8> pack(const Image& im, int side, std::vector<int>& modes) {
    if (im.w < side * 16 || im.h < side * 16) side = 1;
    int count = side * side;
    std::vector<std::vector<u8>> parts(count);
    modes.assign(count, 0);
    run_tiles(count, [&](int t) {
            int x0, y0, x1, y1;
            tile_box(im.w, im.h, side, t, x0, y0, x1, y1);
            Image tile;
            tile.w = x1 - x0;
            tile.h = y1 - y0;
            tile.c = im.c;
            tile.data.resize((size_t)tile.w * tile.h * tile.c);
            for (int y = y0; y < y1; y++)
                memcpy(&tile.data[(size_t)(y - y0) * tile.w * tile.c], &im.data[((size_t)y * im.w + x0) * im.c], (size_t)tile.w * tile.c);
            parts[t] = compress(tile, modes[t]);
        });
    std::vector<u8> head;
    auto put32 = [&](u32 v) { for (int i = 0; i < 4; i++) head.push_back((u8)(v >> (8 * i))); };
    for (const char* m = MAGIC; *m; m++) head.push_back((u8)*m);
    put32(im.w);
    put32(im.h);
    head.push_back((u8)im.c);
    head.push_back((u8)side);
    put32(crc32_of(im.data));
    for (auto& part : parts) put32((u32)part.size());
    for (auto& part : parts) head.insert(head.end(), part.begin(), part.end());
    return head;
}

// Unpacks a complete Truepix file, or returns the reason it could not.
static const char* unpack(const std::vector<u8>& b, Image& im) {
    const size_t HEAD = 18;
    if (b.size() < HEAD || memcmp(b.data(), MAGIC, 4)) return "not a Truepix file of this version";
    im.w = (int)get32(b.data() + 4);
    im.h = (int)get32(b.data() + 8);
    im.c = b[12];
    int side = b[13], count = side * side;
    u32 expected = get32(b.data() + 14);
    if (im.w < 1 || im.h < 1 || im.c < 1 || im.c > 4 || side < 1 || side > 16 || b.size() < HEAD + (size_t)count * 4) return "damaged Truepix header";
    im.data.resize((size_t)im.w * im.h * im.c);
    std::vector<size_t> start(count + 1);
    start[0] = HEAD + (size_t)count * 4;
    for (int t = 0; t < count; t++) start[t + 1] = start[t] + get32(b.data() + HEAD + (size_t)t * 4);
    if (start[count] != b.size()) return "damaged Truepix file";
    if (im.w < side || im.h < side) return "damaged Truepix header";
    for (int t = 0; t < count; t++) {
        int x0, y0, x1, y1;
        tile_box(im.w, im.h, side, t, x0, y0, x1, y1);
        if (!tile_valid(b.data() + start[t], start[t + 1] - start[t], x1 - x0, y1 - y0, im.c)) return "damaged Truepix file";
    }
    run_tiles(count, [&](int t) {
            Image tile = expand(b.data() + start[t], start[t + 1] - start[t]);
            int x0, y0, x1, y1;
            tile_box(im.w, im.h, side, t, x0, y0, x1, y1);
            for (int y = y0; y < y1; y++)
                memcpy(&im.data[((size_t)y * im.w + x0) * im.c], &tile.data[(size_t)(y - y0) * tile.w * tile.c], (size_t)tile.w * tile.c);
        });
    // Never hand back an image that differs from what was encoded.
    if (crc32_of(im.data) != expected) return "Truepix decode failed its checksum";
    return nullptr;
}

// Tiling trades a little size for speed, and only pays on large images.
static int default_side(const Image& im) { return (long long)im.w * im.h >= (1 << 20) ? 2 : 1; }

#ifdef TRUEPIX_WASM
// The browser build: no files, and results stay here until the page copies them out.
static std::vector<u8> packed;
static Image unpacked;

extern "C" {
__attribute__((export_name("tp_alloc"))) u8* tp_alloc(size_t size) { return (u8*)malloc(size); }
__attribute__((export_name("tp_free"))) void tp_free(u8* p) { free(p); }
__attribute__((export_name("tp_compress"))) size_t tp_compress(const u8* pixels, int w, int h, int c) {
    Image im;
    im.w = w; im.h = h; im.c = c;
    im.data.assign(pixels, pixels + (size_t)w * h * c);
    std::vector<int> modes;
    packed = pack(im, default_side(im), modes);
    return packed.size();
}
__attribute__((export_name("tp_packed"))) const u8* tp_packed() { return packed.data(); }
__attribute__((export_name("tp_decompress"))) int tp_decompress(const u8* file, size_t size) {
    std::vector<u8> b(file, file + size);
    unpacked = Image();
    return unpack(b, unpacked) ? 1 : 0;
}
__attribute__((export_name("tp_pixels"))) const u8* tp_pixels() { return unpacked.data.data(); }
__attribute__((export_name("tp_width"))) int tp_width() { return unpacked.w; }
__attribute__((export_name("tp_height"))) int tp_height() { return unpacked.h; }
__attribute__((export_name("tp_channels"))) int tp_channels() { return unpacked.c; }
}
#else
int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) { fprintf(stderr, "usage: truepix-core c|d input output [tiles per side]\n"); return 1; }
    if (argv[1][0] == 'c') {
        Image im = read_pnm(argv[2]);
        int side = argc == 5 ? atoi(argv[4]) : default_side(im);
        if (side < 1 || side > 16) { fprintf(stderr, "tiles per side must be 1 to 16\n"); return 1; }
        std::vector<int> modes;
        std::vector<u8> file = pack(im, side, modes);
        FILE* f = fopen(argv[3], "wb");
        fwrite(file.data(), 1, file.size(), f);
        fclose(f);
        printf("%zu mode", file.size());
        for (size_t t = 0; t < modes.size(); t++) printf("%s%d", t ? "," : " ", modes[t]);
        printf("\n");
    } else {
        FILE* f = fopen(argv[2], "rb");
        std::vector<u8> b;
        u8 chunk[65536];
        size_t n;
        while ((n = fread(chunk, 1, sizeof chunk, f)) > 0) b.insert(b.end(), chunk, chunk + n);
        fclose(f);
        Image im;
        if (const char* problem = unpack(b, im)) { fprintf(stderr, "%s\n", problem); return 2; }
        write_pnm(argv[3], im);
    }
    return 0;
}
#endif
