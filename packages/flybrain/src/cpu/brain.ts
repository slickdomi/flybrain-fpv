import type { Activity, FlyBrain, Readback } from "../brain";
import type { ProgressCallback } from "../data/load";
import type { ConnectomeInfo, ResolvedLoadOptions } from "../data/types";
import { DriveTable } from "../drive";
import type { ModelParams } from "../model";
import type { CpuIn, CpuOut } from "./worker";

/**
 * Main-thread handle for the brain running in a web worker on the CPU. It takes one chunk at a time
 * (check `ready`), so on a slow CPU brain time runs slower than real time.
 */
export class CpuBrain implements FlyBrain {
  readonly backend = "cpu";
  brainTime = 0;
  onReadback: ((r: Readback) => void) | null = null;
  onError: ((e: Error) => void) | null = null;
  /** wall-clock ms the worker needed for the last chunk */
  lastWallMs = 0;
  private busy = false;
  private drive: DriveTable;
  private contrast: Float32Array | null = null;
  private waiters = new Map<number, (a: Activity) => void>();
  private nextId = 0;

  private constructor(
    private worker: Worker,
    readonly info: ConnectomeInfo,
    private _model: ModelParams,
  ) {
    this.drive = new DriveTable(info.n);
    worker.onmessage = (ev: MessageEvent<CpuOut>) => this.handle(ev.data);
  }

  static create(req: ResolvedLoadOptions, model: ModelParams, onProgress?: ProgressCallback): Promise<CpuBrain> {
    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    return new Promise((resolve, reject) => {
      worker.onerror = (e) => reject(new Error(e.message));
      worker.onmessage = (ev: MessageEvent<CpuOut>) => {
        const msg = ev.data;
        if (msg.kind === "progress") onProgress?.(msg.label, msg.total > 0 ? Math.min(1, msg.done / msg.total) : 0);
        else if (msg.kind === "ready") resolve(new CpuBrain(worker, msg.info, { ...model }));
        else if (msg.kind === "error") {
          worker.terminate();
          reject(new Error(msg.message));
        }
      };
      worker.postMessage({ kind: "init", req, model } satisfies CpuIn);
    });
  }

  private send(msg: CpuIn, transfer: Transferable[] = []) {
    this.worker.postMessage(msg, transfer);
  }

  private handle(msg: CpuOut) {
    if (msg.kind === "result") {
      this.busy = false;
      this.brainTime = msg.brainTime;
      this.lastWallMs = msg.wallMs;
      this.onReadback?.({ counts: msg.counts, brainTime: msg.brainTime, spikes: msg.spikes, frameMs: msg.frameMs });
    } else if (msg.kind === "activity") {
      this.waiters.get(msg.id)?.({ act: msg.act, graded: msg.graded });
      this.waiters.delete(msg.id);
    } else if (msg.kind === "error") {
      this.busy = false;
      const err = new Error(msg.message);
      if (this.onError) this.onError(err);
      else console.error("flybrain worker:", err);
    }
  }

  get ready() {
    return !this.busy;
  }

  get model(): Readonly<ModelParams> {
    return this._model;
  }

  setModel(params: Partial<ModelParams>) {
    const { dt: _dt, tDelay: _tDelay, ...rest } = params;
    Object.assign(this._model, rest);
    this.send({ kind: "model", model: { ...this._model } });
  }

  setVision(contrast: Float32Array) {
    this.contrast = contrast.slice();
  }

  setDrive(neurons: ArrayLike<number>, mV: number) {
    this.drive.setTonic(neurons, mV);
  }

  pulse(neurons: ArrayLike<number>, mV: number, durationMs: number) {
    this.drive.pulse(neurons, mV, durationMs, this.brainTime);
  }

  setProbes(neurons: Uint32Array) {
    this.send({ kind: "probes", idx: neurons.slice() });
  }

  setPlasticWeights(packed: Uint32Array) {
    const copy = packed.slice();
    this.send({ kind: "plastic", packed: copy }, [copy.buffer]);
  }

  step(ms: number): boolean {
    if (this.busy) return false;
    this.busy = true;
    this.drive.advance(this.brainTime);
    const dirty = this.drive.takeDirty();
    const drive = dirty ? { start: dirty[0], values: this.drive.values.slice(dirty[0], dirty[1]) } : null;
    const contrast = this.contrast;
    this.contrast = null;
    const transfer: Transferable[] = [];
    if (drive) transfer.push(drive.values.buffer);
    if (contrast) transfer.push(contrast.buffer);
    this.send({ kind: "step", brainMs: ms, contrast, drive }, transfer);
    return true;
  }

  reset(includeGraded = true) {
    this.send({ kind: "reset", includeGraded });
  }

  readActivity(): Promise<Activity> {
    const id = this.nextId++;
    return new Promise((resolve) => {
      this.waiters.set(id, resolve);
      this.send({ kind: "activity", id });
    });
  }

  destroy() {
    this.worker.terminate();
  }
}
