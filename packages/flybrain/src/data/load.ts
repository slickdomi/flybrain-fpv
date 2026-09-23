import { DEFAULT_LOAD, DEFAULT_MODEL, type ModelParams } from "../model";
import type { BrainData, LoadOptions, LoaderOut, ResolvedLoadOptions } from "./types";

export type ProgressCallback = (label: string, fraction: number) => void;

export function resolveLoadOptions(opts: LoadOptions, model: Pick<ModelParams, "wSyn" | "tauS"> = DEFAULT_MODEL): ResolvedLoadOptions {
  const base = typeof location !== "undefined" ? location.href : undefined;
  return {
    url: new URL(opts.url, base).href.replace(/\/$/, ""),
    prune: opts.prune ?? DEFAULT_LOAD.prune,
    laminaWeight: opts.laminaWeight ?? DEFAULT_LOAD.laminaWeight,
    wSyn: model.wSyn,
    tauS: model.tauS,
    plastic: opts.plastic ?? null,
    extract: opts.extract ?? [],
    silence: opts.silence ?? [],
    gain: opts.gain ?? [],
  };
}

/** Downloads and decodes the connectome in a worker. */
export function loadBrainData(req: ResolvedLoadOptions, onProgress?: ProgressCallback): Promise<BrainData> {
  const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  return new Promise((resolve, reject) => {
    worker.onmessage = (ev: MessageEvent<LoaderOut>) => {
      const msg = ev.data;
      // done can exceed total when the server inflates gzip on the wire
      if (msg.kind === "progress") onProgress?.(msg.label, msg.total > 0 ? Math.min(1, msg.done / msg.total) : 0);
      else if (msg.kind === "done") {
        worker.terminate();
        resolve(msg.data);
      } else {
        worker.terminate();
        reject(new Error(msg.message));
      }
    };
    worker.onerror = (e) => reject(new Error(e.message));
    worker.postMessage(req);
  });
}
