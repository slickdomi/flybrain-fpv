import type { FlyBrain, Readback } from "./brain";

/**
 * The model occasionally seizes: the cholinergic lLN1_bc clique in the antennal lobe ignites and the whole brain
 * runs away (over 90 k spikes/s). The watchdog resets the spiking state when that lasts, keeping the optic lobe.
 */
export class SeizureWatchdog {
  count = 0;
  /** runaway episodes that `confirm` did not accept (e.g. a busy visual scene, not a seizure) */
  rejected = 0;
  /** smoothed spikes per second of brain time */
  spikesPerSec = 0;
  /**
   * Optional extra check before a reset, e.g. that the known seizure clique is really firing:
   * `() => rates.instHz("lLN1_bc") > 100`. Rich visual scenes alone can push the brain past the threshold.
   */
  confirm: (() => boolean) | null = null;
  private runawayMs = 0;
  private episodeRejected = false;

  constructor(
    private brain: FlyBrain,
    public thresholdSpikesPerSec = 90000,
    public holdMs = 300,
    public onSeizure: ((count: number, brainTime: number) => void) | null = null,
  ) {}

  /** Returns true if the brain was reset. */
  update(r: Readback): boolean {
    if (r.frameMs <= 0) return false;
    const inst = (r.spikes * 1000) / r.frameMs;
    this.spikesPerSec += (inst - this.spikesPerSec) * 0.1;
    this.runawayMs = inst > this.thresholdSpikesPerSec ? this.runawayMs + r.frameMs : 0;
    if (this.runawayMs === 0) this.episodeRejected = false;
    if (this.runawayMs <= this.holdMs) return false;
    if (this.confirm && !this.confirm()) {
      if (!this.episodeRejected) this.rejected++;
      this.episodeRejected = true;
      return false;
    }
    this.episodeRejected = false;
    this.count++;
    this.runawayMs = 0;
    this.spikesPerSec = 0;
    this.onSeizure?.(this.count, r.brainTime);
    this.brain.reset(false);
    return true;
  }
}
