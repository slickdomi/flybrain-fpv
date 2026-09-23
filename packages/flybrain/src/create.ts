import type { FlyBrain } from "./brain";
import { CpuBrain } from "./cpu/brain";
import { loadBrainData, resolveLoadOptions, type ProgressCallback } from "./data/load";
import type { LoadOptions } from "./data/types";
import { GpuBrain, type GpuBrainOptions } from "./gpu/brain";
import { DEFAULT_MODEL, type ModelParams } from "./model";

export interface CreateFlyBrainOptions extends LoadOptions {
  /** "auto" (default) uses WebGPU when a device is available and falls back to the CPU worker */
  backend?: "auto" | "webgpu" | "cpu";
  /** an existing WebGPU device (e.g. the one your renderer uses); see requestFlyDevice() for the limits it needs */
  device?: GPUDevice | null;
  model?: Partial<ModelParams>;
  onProgress?: ProgressCallback;
  gpu?: Omit<GpuBrainOptions, "model">;
}

/** Requests a WebGPU device with storage buffer limits large enough for the connectome, or null. */
export async function requestFlyDevice(): Promise<GPUDevice | null> {
  if (typeof navigator === "undefined" || !navigator.gpu) return null;
  const adapter = (await navigator.gpu.requestAdapter({ powerPreference: "high-performance" })) ?? (await navigator.gpu.requestAdapter());
  if (!adapter) return null;
  const want = (name: keyof GPUSupportedLimits, value: number) => Math.min(value, adapter.limits[name] as number);
  return adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: want("maxStorageBufferBindingSize", 512 * 2 ** 20),
      maxBufferSize: want("maxBufferSize", 512 * 2 ** 20),
      maxStorageBuffersPerShaderStage: want("maxStorageBuffersPerShaderStage", 10),
    },
  });
}

/** Downloads the connectome and starts a brain on the best available backend. */
export async function createFlyBrain(opts: CreateFlyBrainOptions): Promise<FlyBrain> {
  const model = { ...DEFAULT_MODEL, ...opts.model };
  const req = resolveLoadOptions(opts, model);
  const want = opts.backend ?? "auto";
  if (want !== "cpu") {
    const device = opts.device ?? (await requestFlyDevice().catch((e) => (console.warn("WebGPU init failed:", e), null)));
    if (device) {
      const data = await loadBrainData(req, opts.onProgress);
      return new GpuBrain(device, data, { ...opts.gpu, model });
    }
    if (want === "webgpu") throw new Error("WebGPU is not available");
  }
  return CpuBrain.create(req, model, opts.onProgress);
}
