// WebGPU path: decodes the connectome off the main thread and hands the matrices over.

import { buildBrainData } from "./build";
import { brainDataTransfer, type LoaderOut, type ResolvedLoadOptions } from "./types";

const post = (msg: LoaderOut, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(msg, transfer);

self.onmessage = async (ev: MessageEvent<ResolvedLoadOptions>) => {
  try {
    const d = await buildBrainData(ev.data, (label, done, total) => post({ kind: "progress", label, done, total }));
    post({ kind: "done", data: d }, brainDataTransfer(d));
  } catch (err) {
    post({ kind: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
