// CPU backend: loads the connectome and runs the brain simulation in this worker.

import { buildBrainData } from "../data/build";
import { copyInfo, type ConnectomeInfo, type ResolvedLoadOptions } from "../data/types";
import type { ModelParams } from "../model";
import { CpuSim } from "./sim";

export type CpuIn =
  | { kind: "init"; req: ResolvedLoadOptions; model: ModelParams }
  | { kind: "probes"; idx: Uint32Array }
  | { kind: "model"; model: ModelParams }
  | { kind: "plastic"; packed: Uint32Array }
  | { kind: "reset"; includeGraded: boolean }
  | { kind: "step"; brainMs: number; contrast: Float32Array | null; drive: { start: number; values: Float32Array } | null }
  | { kind: "activity"; id: number };

export type CpuOut =
  | { kind: "progress"; label: string; done: number; total: number }
  | { kind: "ready"; info: ConnectomeInfo }
  | { kind: "result"; counts: Uint32Array; brainTime: number; spikes: number; frameMs: number; wallMs: number }
  | { kind: "activity"; id: number; act: Float32Array; graded: Float32Array }
  | { kind: "error"; message: string };

const post = (msg: CpuOut, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(msg, transfer);

let sim: CpuSim | null = null;

self.onmessage = async (ev: MessageEvent<CpuIn>) => {
  const msg = ev.data;
  try {
    if (msg.kind === "init") {
      const data = await buildBrainData(msg.req, (label, done, total) => post({ kind: "progress", label, done, total }));
      sim = new CpuSim(data, msg.model);
      post({ kind: "ready", info: copyInfo(data) });
      return;
    }
    if (!sim) return;
    if (msg.kind === "probes") sim.setProbes(msg.idx);
    else if (msg.kind === "model") sim.setParams(msg.model);
    else if (msg.kind === "plastic") sim.setPlastic(msg.packed);
    else if (msg.kind === "reset") sim.reset(msg.includeGraded);
    else if (msg.kind === "activity") {
      const act = sim.act.slice();
      const graded = sim.graded().slice();
      post({ kind: "activity", id: msg.id, act, graded }, [act.buffer, graded.buffer]);
    } else if (msg.kind === "step") {
      const t0 = performance.now();
      if (msg.drive) sim.setDrive(msg.drive.start, msg.drive.values);
      if (msg.contrast) sim.setContrast(msg.contrast);
      const r = sim.step(msg.brainMs);
      post({ kind: "result", ...r, brainTime: sim.brainTime, wallMs: performance.now() - t0 }, [r.counts.buffer]);
    }
  } catch (err) {
    post({ kind: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
