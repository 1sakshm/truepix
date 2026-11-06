"""Seven builds of the new source: each file must match the old baseline, and each build must decode each file."""
import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent.resolve()
S = HERE.parent.parent
WORK = HERE / "cross"
WORK.mkdir(exist_ok=True)
SRC = Path("D:/Truepix/truepix-core.cpp")
ZIG = [str(S / ".venv" / "Scripts" / "python.exe"), "-m", "ziglang", "c++"]
COMMON = ["-std=c++17", "-ffp-contract=off"]
BUILD = {
    "GCC AVX2": ("b_gcc_avx2.exe", ["g++", *COMMON, "-O3", "-mavx2", "-static"]),
    "GCC FMA": ("b_gcc_fma.exe", ["g++", *COMMON, "-O3", "-mavx2", "-mfma", "-static"]),
    "GCC SSE2": ("b_gcc_sse2.exe", ["g++", *COMMON, "-O3", "-static"]),
    "GCC -O0": ("b_gcc_o0.exe", ["g++", *COMMON, "-O0", "-mavx2", "-static"]),
    "Clang x64": ("b_clang.exe", [*ZIG, *COMMON, "-O2", "-mavx2", "-target", "x86_64-windows-gnu"]),
    "Clang x86": ("b_clang32.exe", [*ZIG, *COMMON, "-O2", "-msse2", "-mfpmath=sse", "-target", "x86-windows-gnu"]),
    "WebAssembly": ("b_wasm.wasm", [*ZIG, *COMMON, "-O2", "-DTRUEPIX_SERIAL", "-fno-exceptions", "-target", "wasm32-wasi"]),
}
BUILDS = {name: exe for name, (exe, _) in BUILD.items()}
INPUTS = {n: n for n in ("small.ppm", "kodak.ppm", "jpegish.ppm", "gray.pgm", "alpha.pam", "j_base.jpg", "j_prog.jpg", "j_grey.jpg", "j_cmyk.jpg", "j_q100.jpg")}


def compile_one(item):
    name, (exe, command) = item
    done = subprocess.run([*command, str(SRC), "-o", str(HERE / exe)], capture_output=True, text=True)
    print(name, "built" if done.returncode == 0 else "FAILED\n" + done.stderr[-1500:], flush=True)
    return done.returncode == 0


def call(build, *args):
    exe = BUILDS[build]
    if exe.endswith(".wasm"):
        names = ["/work/" + a if i else a for i, a in enumerate(args)]
        command = ["node", "--no-warnings", str(HERE / "run_wasi.mjs"), str(HERE / exe), str(WORK), *names]
    else:
        command = [str(HERE / exe), *[str(WORK / a) if i else a for i, a in enumerate(args)]]
    return subprocess.run(command, capture_output=True, text=True)


def encode(job):
    index, stem = job
    name = f"{stem}.{index}.tpx"
    done = call(list(BUILDS)[index], "c", f"{stem}.src", name)
    assert done.returncode == 0, (list(BUILDS)[index], done.stderr)
    return stem, index, hashlib.sha256((WORK / name).read_bytes()).hexdigest()


def decode(job):
    reader, writer, stem = job
    out = f"{stem}.{reader}.{writer}.back"
    done = call(list(BUILDS)[reader], "d", f"{stem}.{writer}.tpx", out)
    same = done.returncode == 0 and (WORK / out).read_bytes() == (WORK / f"{stem}.src").read_bytes()
    (WORK / out).unlink(missing_ok=True)
    return stem, reader, writer, same


with ThreadPoolExecutor(3) as pool:
    if not all(pool.map(compile_one, BUILD.items())):
        sys.exit(1)

stems = []
baseline = {}
for name in INPUTS:
    stem = Path(name).stem
    shutil.copy(HERE.parent / "set" / name, WORK / f"{stem}.src")
    stems.append(stem)
    done = subprocess.run(["D:/Truepix/truepix-core.exe", "c", str(WORK / f"{stem}.src"), str(WORK / f"{stem}.old")], capture_output=True)
    assert done.returncode == 0
    baseline[stem] = hashlib.sha256((WORK / f"{stem}.old").read_bytes()).hexdigest()

count = len(BUILDS)
with ThreadPoolExecutor(4) as pool:
    hashes = list(pool.map(encode, [(i, s) for s in stems for i in range(count)]))
    print("encoded", flush=True)
    decoded = list(pool.map(decode, [(r, w, s) for s in stems for r in range(count) for w in range(count)]))
matches = [h == baseline[t] for t, _, h in hashes]
result = {"builds": list(BUILDS), "files": len(hashes), "match_old_baseline": sum(matches), "decodes": len(decoded),
          "exact": sum(d[3] for d in decoded), "failed": [d[:3] for d in decoded if not d[3]],
          "mismatched": [(t, list(BUILDS)[i]) for (t, i, h) in hashes if h != baseline[t]]}
(HERE / "matrix.json").write_text(json.dumps(result, indent=1))
print(result)
shutil.rmtree(WORK)
