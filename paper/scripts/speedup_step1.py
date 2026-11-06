from pathlib import Path

s = Path("base.cpp").read_text(encoding="utf-8")


def sub(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)


# OLS: remove aliasing so the update vectorises. The factorisation keeps each value's sequence of
# subtractions, in the same order, so it stays bit-exact while running column by column.
start = s.index("    std::vector<double> cov, vec, w, chol, x, y;")
end = s.index("struct Counter {")
s = s[:start] + '''    std::vector<double> cov, vec, w, chol, x, y, column;
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

''' + s[end:]

# Exact stand-ins for slower library rounding and for the search in quant_log.
sub('''static inline int quant_log(int v) {
    static const int limits[] = {0, 1, 2, 3, 5, 7, 10, 14, 20, 28, 40, 56, 80, 112, 160};
    for (int i = 0; i < 15; i++) if (v <= limits[i]) return i;
    return 15;
}
''', '''// std::lround for the small values used while coding: halves round away from zero.
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
''')
sub("int P = (int)std::lround(blended);", "int P = round_float(blended);")
sub("int v = (int)std::lround(ols_p[k]) - P;", "int v = round_float(ols_p[k]) - P;")

# Regressors: flat offsets in place of two levels of vectors per lookup.
sub("    std::vector<std::vector<int>> planes[NOLS];\n", "    std::vector<std::vector<int>> planes[NOLS];\n    std::vector<std::vector<int>> reach[NOLS];  // byte offsets of the regressors from the current pixel\n")
sub('''            offsets[k][plane] = off;
            planes[k][plane] = pl;
''', '''            offsets[k][plane] = off;
            planes[k][plane] = pl;
            reach[k][plane].clear();
            for (size_t i = 0; i < off.size(); i++) reach[k][plane].push_back(off[i].second * stride + off[i].first * c + pl[i]);
''')
sub("ols[k].resize(c * 4); offsets[k].resize(c); planes[k].resize(c);", "ols[k].resize(c * 4); offsets[k].resize(c); planes[k].resize(c); reach[k].resize(c);")
sub('''                    OLS& o = m.ols[k][slot[k]];
                    for (int i = 0; i < o.n; i++)
                        o.x[i] = m.at(x + m.offsets[k][idx][i].first, y + m.offsets[k][idx][i].second, m.planes[k][idx][i]) - base;
''', '''                    OLS& o = m.ols[k][slot[k]];
                    const int* steps = m.reach[k][idx].data();
                    const u8* here = &m.at(x, y, 0);
                    double* regressors = o.x.data();
                    for (int i = 0; i < o.n; i++) regressors[i] = here[steps[i]] - base;
''')
Path("fast.cpp").write_text(s, encoding="utf-8")
