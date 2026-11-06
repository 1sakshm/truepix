// Runs a WASI build of PXM under Node with one folder mapped to /work.
import { readFile } from "node:fs/promises";
import { WASI } from "node:wasi";

const [wasmPath, folder, ...args] = process.argv.slice(2);
const wasi = new WASI({ version: "preview1", args: ["pxm", ...args], preopens: { "/work": folder }, returnOnExit: true });
const module = await WebAssembly.compile(await readFile(wasmPath));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
process.exit(wasi.start(instance));
