from pathlib import Path

s = Path("base.cpp").read_text(encoding="utf-8")


def sub(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)


sub('static const char MAGIC[] = "TPX1";', '''#include <x86intrin.h>
static unsigned long long tt[12], t0, nd, ns;
#define TICK(i) do { unsigned long long n_ = __rdtsc(); tt[i] += n_ - t0; t0 = n_; } while (0)
static const char* NAMES[12] = {"gather regressors", "predict+contexts", "table lookup", "parametric inputs", "mix+squash",
                                "apm+coder", "mixer update", "counter update", "ols update", "ols solve", "blend/bias update", ""};
static const char MAGIC[] = "TPX1";''')
sub('''        count = 0;
        solve();
''', '''        count = 0;
        TICK(8);
        solve();
        TICK(9);
''')
sub('''    float* in = m.inputs;
    int k = 0;
''', '''    float* in = m.inputs;
    int k = 0;
    nd++;
    TICK(1);
''')
sub('''    for (int i = 0; i < NPARAM; i++) {
        float yes''', '''    TICK(2);
    for (int i = 0; i < NPARAM; i++) {
        float yes''')
sub('''    in[k++] = 0.3f;

''', '''    in[k++] = 0.3f;
    TICK(3);

''')
sub('''    float p0 = squash(st);
''', '''    float p0 = squash(st);
    TICK(4);
''')
sub('''    float lr = (float)(0.0015 + 0.02 * m.lr_scale);
''', '''    TICK(5);
    float lr = (float)(0.0015 + 0.02 * m.lr_scale);
''')
sub('''    m.apm[0].update(bit, 0.02f);
''', '''    TICK(6);
    m.apm[0].update(bit, 0.02f);
''')
sub('''    m.decisions++;
    return bit;
''', '''    m.decisions++;
    TICK(7);
    return bit;
''')
sub('''                int pl = order[idx];
                int W = m.at''', '''                int pl = order[idx];
                ns++;
                TICK(10);
                int W = m.at''')
sub('''                int med = W + N - NW;
''', '''                TICK(0);
                int med = W + N - NW;
''')
sub('''                m.at(x, y, pl) = (u8)value;
''', '''                TICK(1);
                m.at(x, y, pl) = (u8)value;
''')
sub('''                float e = (float)value - blended;
''', '''                TICK(8);
                float e = (float)value - blended;
''')
sub('''int main(int argc, char** argv) {
''', '''int main(int argc, char** argv) {
    t0 = __rdtsc();
    struct Report { ~Report() {
        unsigned long long sum = 0;
        for (int i = 0; i < 11; i++) sum += tt[i];
        fprintf(stderr, "samples %llu decisions %llu (%.2f per sample), %.0f cycles per sample\\n", ns, nd, (double)nd / ns, (double)sum / ns);
        for (int i = 0; i < 11; i++) fprintf(stderr, "%-20s %5.1f%%  %6.0f cycles/sample\\n", NAMES[i], tt[i] * 100.0 / sum, (double)tt[i] / ns);
    } } report;
''')
Path("timed.cpp").write_text(s, encoding="utf-8")
