import sys
from pathlib import Path

s = Path("fast.cpp").read_text(encoding="utf-8")


def sub(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)


sub("    const __m256d scaled = _mm256_mul_pd(_mm256_mul_pd(_mm256_add_pd(t, t), series), one);\n",
    "    const __m256d scaled = _mm256_mul_pd(_mm256_add_pd(t, t), series);\n")

# One sample's decisions keep asking for the same interval edges, so remember them.
sub('''    __m256 mass8(int lo, int hi) const {
        const __m256 m = _mm256_load_ps(mu), scale = _mm256_load_ps(s);
        const __m256 top = _mm256_div_ps(_mm256_sub_ps(_mm256_set1_ps(hi + 0.5f), m), scale);
        const __m256 bottom = _mm256_div_ps(_mm256_sub_ps(_mm256_set1_ps(lo - 0.5f), m), scale);
        return _mm256_sub_ps(cdf8(top), cdf8(bottom));
    }
''', '''    static const int EDGES = 6;
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
''')
if "next" in sys.argv:
    sub('''        __builtin_prefetch(m.active[i], 1);
    }
''', '''        __builtin_prefetch(m.active[i], 1);
        if (node == 0) __builtin_prefetch(&m.tables[i][(m.ctx[i] + 0x9E3779B1u) >> (32 - TABLE_BITS)], 1);
    }
''')
Path("fast.cpp").write_text(s, encoding="utf-8")
