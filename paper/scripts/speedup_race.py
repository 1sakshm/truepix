import subprocess, sys, time, json
from pathlib import Path
from PIL import Image
S = Path(sys.argv[1]); D = S/"dec"; OLD, NEW = str(D/"base.exe"), r"D:\Truepix\truepix-core.exe"
src = sorted((S/"locked_final").glob("*.png"))
if not src: src = sorted((S/"locked_fresh").glob("*.png"))
src = src[:5]
print([p.name for p in src], flush=True)
te={"old":0,"new":0}; td={"old":0,"new":0}; same=0; exact=0; total=0
for p in src:
    ppm = D/"r.ppm"; Image.open(p).convert("RGB").save(ppm)
    raw = ppm.read_bytes()
    for name, exe in (("new",NEW),("old",OLD)):
        t=time.perf_counter(); subprocess.run([exe,"c",str(ppm),str(D/f"r.{name}")],check=True,capture_output=True); te[name]+=time.perf_counter()-t
    o,n=(D/"r.old").read_bytes(),(D/"r.new").read_bytes(); same+=o==n; total+=len(n)
    for rep in range(2):
        for name, exe in (("old",OLD),("new",NEW)) if rep==0 else (("new",NEW),("old",OLD)):
            t=time.perf_counter(); subprocess.run([exe,"d",str(D/"r.new"),str(D/"r.back")],check=True,capture_output=True); td[name]+=time.perf_counter()-t
            exact += (D/"r.back").read_bytes()==raw
    print(p.name, flush=True)
n=len(src)
print(json.dumps({"images":n,"identical_files":same,"exact_decodes":exact,"of":n*4,"encode_old":te["old"]/n,"encode_new":te["new"]/n,"decode_old":td["old"]/n/2,"decode_new":td["new"]/n/2}))
