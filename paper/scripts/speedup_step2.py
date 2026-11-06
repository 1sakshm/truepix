from pathlib import Path

s = Path("fast.cpp").read_text(encoding="utf-8")


def sub(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)


# Mixers: separate the dot product from its squash, so the squashes can run side by side.
sub('''    float mix(const float* x, int sel) {
        cur = &w[sel * inputs];''', '''    float mix(const float* x, int sel) {
        const float result = weigh(x, sel);
        pr = squash(dot);
        return result;
    }
    float weigh(const float* x, int sel) {
        cur = &w[sel * inputs];''')
sub('''        dot = std::max(-14.f, std::min(14.f, d));
        pr = squash(dot);
        return dot;
''', '''        dot = std::max(-14.f, std::min(14.f, d));
        return dot;
''')

# AVX2 forms of the logistic maths. Each lane does the same operations in the same order as the
# scalar code, so both give the same bits. There are no fused multiplies here or in the scalar code.
sub('''// Parametric mixture state for one sample.
''', '''#ifdef __AVX2__
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
    const __m256d scaled = _mm256_mul_pd(_mm256_mul_pd(_mm256_add_pd(t, t), series), one);
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
''')
sub('''struct Params {
    float mu[NPARAM], s[NPARAM];
''', '''struct alignas(32) Params {
    float mu[8] = {}, s[8] = {0, 0, 0, 0, 0, 0, 0, 1.f};  // NPARAM, padded to a whole vector
#ifdef __AVX2__
    __m256 mass8(int lo, int hi) const {
        const __m256 m = _mm256_load_ps(mu), scale = _mm256_load_ps(s);
        const __m256 top = _mm256_div_ps(_mm256_sub_ps(_mm256_set1_ps(hi + 0.5f), m), scale);
        const __m256 bottom = _mm256_div_ps(_mm256_sub_ps(_mm256_set1_ps(lo - 0.5f), m), scale);
        return _mm256_sub_ps(cdf8(top), cdf8(bottom));
    }
#endif
''')
sub('''    for (int i = 0; i < NCTX; i++) {
        Counter* e = &m.tables[i][(m.ctx[i] + (u32)node * 0x9E3779B1u) >> (32 - TABLE_BITS)];
        m.active[i] = e;
        in[k++] = STRETCH[(int)e->fast >> 4];
''', '''    for (int i = 0; i < NCTX; i++) {
        m.active[i] = &m.tables[i][(m.ctx[i] + (u32)node * 0x9E3779B1u) >> (32 - TABLE_BITS)];
        __builtin_prefetch(m.active[i], 1);
    }
    for (int i = 0; i < NCTX; i++) {
        Counter* e = m.active[i];
        in[k++] = STRETCH[(int)e->fast >> 4];
''')
sub('''    for (int i = 0; i < NPARAM; i++) {
        float yes = par.mass(i, plo1, phi1);
        float no = phi2 >= plo2 ? par.mass(i, plo2, phi2) : std::max(0.f, par.mass(i, 0, 255) - yes);
        float p = (yes + 1e-7f) / (yes + no + 2e-7f);
        in[k++] = stretch(clamp_p(p));
    }
''', '''#ifdef __AVX2__
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
''')
sub('''    m.second[0] = m.mix[0].mix(in, node);
    m.second[1] = m.mix[1].mix(in, nc * 16 + energy);
    m.second[2] = m.mix[2].mix(in, (nc * 4 + plane) * 8 + (m.ctx[2] & 7));
    m.second[3] = m.mix[3].mix(in, nc * 16 + m.phase_ctx);
    m.second[4] = m.mix[4].mix(in, nc * 256 + m.texture_ctx);
    float st = m.final_mix.mix(m.second, node);
    float p0 = squash(st);
''', '''    m.second[0] = m.mix[0].weigh(in, node);
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
''')
sub('''    for (int i = 0; i < NCTX; i++) {
        Counter* e = m.active[i];
        int target = bit ? 65535 : 0;
        int n = e->n;
        e->fast += (target - e->fast) / (std::min(n, 30) + 1.5f);
        e->slow += (target - e->slow) / (std::min(n, 250) + 1.5f);
        if (n < 1000) e->n = n + 1;
    }
''', '''    // Done in arrays so the divisions run side by side.
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
''')
sub("static const char MAGIC[] = \"TPX1\";", "#ifdef __AVX2__\n#include <immintrin.h>\n#endif\n\nstatic const char MAGIC[] = \"TPX1\";")
Path("fast.cpp").write_text(s, encoding="utf-8")
