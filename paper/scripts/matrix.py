"""Full portability matrix: every build decodes every build's file, for five kinds of image."""
import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent.resolve()
WORK = HERE / "cross"
WORK.mkdir(exist_ok=True)
BUILDS = {"GCC AVX2": "b_gcc_avx2.exe", "GCC FMA": "b_gcc_fma.exe", "GCC SSE2": "b_gcc_sse2.exe", "GCC -O0": "b_gcc_o0.exe",
          "Clang x64": "b_clang.exe", "Clang x86": "b_clang32.exe", "WebAssembly": "b_wasm.wasm"}
INPUTS = {"small.ppm": "plain RGB", "kodak.ppm": "Photo CD", "jpegish.ppm": "JPEG history", "gray.pgm": "greyscale", "alpha.pam": "RGBA"}


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
    name = f"{stem}.{index}.pxm"
    assert call(list(BUILDS)[index], "c", f"{stem}.src", name).returncode == 0
    return stem, index, hashlib.sha256((WORK / name).read_bytes()).hexdigest()


def decode(job):
    reader, writer, stem = job
    out = f"{stem}.{reader}.{writer}.back"
    done = call(list(BUILDS)[reader], "d", f"{stem}.{writer}.pxm", out)
    same = done.returncode == 0 and (WORK / out).read_bytes() == (WORK / f"{stem}.src").read_bytes()
    (WORK / out).unlink(missing_ok=True)
    return stem, reader, writer, same


stems = []
for name in INPUTS:
    stem = Path(name).stem
    shutil.copy(HERE / name, WORK / f"{stem}.src")
    stems.append(stem)
count = len(BUILDS)
with ThreadPoolExecutor(4) as pool:
    hashes = list(pool.map(encode, [(i, s) for s in stems for i in range(count)]))
    decoded = list(pool.map(decode, [(r, w, s) for s in stems for r in range(count) for w in range(count)]))
identical = {s: len({h for t, _, h in hashes if t == s}) == 1 for s in stems}
result = {"builds": list(BUILDS), "inputs": list(INPUTS.values()), "identical": identical, "decodes": len(decoded), "exact": sum(d[3] for d in decoded), "failed": [d[:3] for d in decoded if not d[3]]}
(Path(sys.argv[1]) / "matrix.json").write_text(json.dumps(result, indent=1))
print(result)
shutil.rmtree(WORK)
