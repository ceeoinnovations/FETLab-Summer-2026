#Run in VSCODE
"""
Event-based velocimetry: sanity checks + global contrast-maximization
velocity estimate for a single constant-velocity recording.

Run this on your HOST PC (not the OpenMV board) against the events.bin
file produced by genx320_event_logger.py.

Usage:
    python analyze_events.py events.bin

Requires: numpy, matplotlib, scipy
    pip install numpy matplotlib scipy
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def detect_hot_pixels(x, y, size=320, factor=20.0, min_count=50):
    """Flag pixels whose total event count is far above the rest of the
    sensor - the signature of a stuck/hot pixel rather than real scene
    motion, which otherwise spreads events across many locations over
    time. Returns (hot_pixel_set, counts_grid, threshold_used)."""
    xi = x.astype(np.int64)
    yi = y.astype(np.int64)
    counts = np.zeros((size, size), dtype=np.int64)
    np.add.at(counts, (xi, yi), 1)

    nonzero = counts[counts > 0]
    if nonzero.size == 0:
        return set(), counts, 0.0

    median = np.median(nonzero)
    threshold = max(factor * median, min_count)
    hot_idx = np.argwhere(counts > threshold)
    return set(map(tuple, hot_idx)), counts, threshold


def mask_hot_pixels(x, y, hot_pixels):
    if not hot_pixels:
        return np.ones(len(x), dtype=bool)
    xi = x.astype(np.int64)
    yi = y.astype(np.int64)
    mask = np.ones(len(x), dtype=bool)
    for (px, py) in hot_pixels:
        mask &= ~((xi == px) & (yi == py))
    return mask


def load_events(path):
    raw = np.fromfile(path, dtype="<u2").reshape(-1, 6)
    ev_type = raw[:, 0].astype(np.int64)
    s = raw[:, 1].astype(np.int64)
    ms = raw[:, 2].astype(np.int64)
    us = raw[:, 3].astype(np.int64)
    x = raw[:, 4].astype(np.float64)
    y = raw[:, 5].astype(np.float64)
    t_us = s * 1_000_000 + ms * 1000 + us
    return ev_type, t_us, x, y


def sanity_check(ev_type, t_us, x, y):
    n = len(t_us)
    duration_s = (t_us.max() - t_us.min()) / 1e6
    print(f"Total events: {n}")
    print(f"Duration: {duration_s:.3f} s")
    if duration_s > 0:
        print(f"Event rate: {n / duration_s:.1f} events/s")
    print(f"x range: {x.min():.0f} - {x.max():.0f}")
    print(f"y range: {y.min():.0f} - {y.max():.0f}")
    print(f"ON events: {np.sum(ev_type == 1)}  OFF events: {np.sum(ev_type == 0)}")

    dt = np.diff(t_us)
    if np.any(dt < 0):
        worst = int(dt.min())  # most negative jump, in microseconds
        n_bad = int(np.sum(dt < 0))
        print(f"WARNING: timestamps are not monotonically increasing - "
              f"{n_bad} out of {n - 1} consecutive pairs go backward "
              f"(worst jump: {worst} us).")
        if abs(worst) <= 1000 and n_bad / n < 0.01:
            print("This looks like normal microsecond-scale reordering "
                  "from the sensor's parallel readout - usually fine to "
                  "proceed with.")
        else:
            print("This is a LARGER or more frequent violation than typical "
                  "readout jitter - worth investigating (timer rollover, "
                  "buffer reuse, or a logging bug) before trusting results.")
    print()


def plot_raw(t_us, x, y, out_path="raw_events.png"):
    t_s = (t_us - t_us.min()) / 1e6
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(t_s, x, c=t_s, cmap="coolwarm", s=2)
    axes[0].set(xlabel="t (s)", ylabel="x (px)", title="x vs t")

    axes[1].scatter(t_s, y, c=t_s, cmap="coolwarm", s=2)
    axes[1].set(xlabel="t (s)", ylabel="y (px)", title="y vs t")

    plt.tight_layout()
    plt.savefig(out_path, dpi=125)
    plt.close(fig)
    print(f"Saved raw event plot to {out_path}")
    print("Look for a clean, mostly-straight diagonal trend in both panels "
          "- that slope IS the velocity you're about to estimate. A "
          "scattered plot with no visible trend usually means lighting or "
          "background noise is dominating the recording.\n")


def plot_event_rate(t_us, out_path="event_rate.png", bin_ms=10):
    t_s = (t_us - t_us.min()) / 1e6
    edges = np.arange(0, t_s.max() + bin_ms / 1000, bin_ms / 1000)
    counts, _ = np.histogram(t_s, bins=edges)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(edges[:-1], counts)
    plt.xlabel("t (s)")
    plt.ylabel(f"events per {bin_ms} ms")
    plt.title("Event rate over time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=125)
    plt.close(fig)
    print(f"Saved event rate plot to {out_path}")
    print("A real moving marker should show up as a distinct bump/plateau "
          "somewhere in this timeline - that's your actual motion window. "
          "If the rate looks roughly flat and high across the ENTIRE "
          "recording instead, that's noise dominating the whole thing, "
          "not a moving marker.\n")


def plot_spatial_heatmap(x, y, out_path="spatial_heatmap.png", bins=320):
    fig = plt.figure(figsize=(6, 6))
    plt.hist2d(x, y, bins=bins, range=[[0, 320], [0, 320]], cmap="inferno")
    plt.colorbar(label="event count")
    plt.xlabel("x (px)")
    plt.ylabel("y (px)")
    plt.title("Spatial distribution of ALL logged events")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=125)
    plt.close(fig)
    print(f"Saved spatial heatmap to {out_path}")
    print("A real marker's path should look like a line/band tracing its "
          "trajectory. A handful of extremely bright single pixels "
          "(especially at edges/corners/rows) instead of a path means you "
          "have hot/noisy pixels dominating the count - those need to be "
          "masked out or fixed at the bias/lighting level, not analyzed.\n")


def plot_cropped_window(t_us, x, y, start, end, out_path="raw_events_cropped.png"):
    t_s = (t_us - t_us.min()) / 1e6
    mask = (t_s >= start) & (t_s <= end)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(t_s[~mask], x[~mask], c="lightgray", s=2,
                     label="outside window")
    axes[0].scatter(t_s[mask], x[mask], c="crimson", s=3,
                     label="used for estimate")
    axes[0].axvspan(start, end, color="crimson", alpha=0.08)
    axes[0].set(xlabel="t (s)", ylabel="x (px)", title="x vs t (cropped window in red)")
    axes[0].legend(loc="upper right", fontsize=8, markerscale=3)

    axes[1].scatter(t_s[~mask], y[~mask], c="lightgray", s=2)
    axes[1].scatter(t_s[mask], y[mask], c="crimson", s=3)
    axes[1].axvspan(start, end, color="crimson", alpha=0.08)
    axes[1].set(xlabel="t (s)", ylabel="y (px)", title="y vs t (cropped window in red)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=125)
    plt.close(fig)
    print(f"Saved cropped-window plot to {out_path} - the red points/band "
          f"are exactly what's being fed into the velocity estimate. If "
          f"you don't see a red band roughly in the [{start:.2f}, "
          f"{end:.2f}] s range, something's off with the times you "
          f"picked.\n")


def iwe(x, y, t_s, vx, vy, t0, bins, extent):
    xw = x - vx * (t_s - t0)
    yw = y - vy * (t_s - t0)
    img, _, _ = np.histogram2d(xw, yw, bins=bins, range=extent)
    return img


def estimate_velocity(x, y, t_s, extent, bins=160,
                       coarse_range=200, coarse_steps=41):
    t0 = t_s.mean()

    # ---- Stage 1: coarse grid search ----
    # coarse_range is in px/s - widen this if your motion is faster than
    # +/-200 px/s, or you'll never find the true peak.
    grid = np.linspace(-coarse_range, coarse_range, coarse_steps)
    best_var = -1.0
    best_v = (0.0, 0.0)
    for vx in grid:
        for vy in grid:
            v = iwe(x, y, t_s, vx, vy, t0, bins, extent).var()
            if v > best_var:
                best_var = v
                best_v = (vx, vy)

    print(f"Coarse search peak: vx={best_v[0]:.1f} px/s, "
          f"vy={best_v[1]:.1f} px/s (variance={best_var:.3f})")

    # ---- Stage 2: local refinement ----
    def neg_var(v):
        return -iwe(x, y, t_s, v[0], v[1], t0, bins, extent).var()

    res = minimize(neg_var, x0=best_v, method="Nelder-Mead",
                    options={"xatol": 0.5, "fatol": 1e-3})

    vx_final, vy_final = res.x
    print(f"Refined estimate:    vx={vx_final:.2f} px/s, "
          f"vy={vy_final:.2f} px/s (variance={-res.fun:.3f})\n")

    return vx_final, vy_final, t0


def plot_variance_surface(x, y, t_s, t0, extent, bins, vx_est, vy_est,
                           span=60, steps=61, out_path="variance_surface.png"):
    gx = np.linspace(vx_est - span, vx_est + span, steps)
    gy = np.linspace(vy_est - span, vy_est + span, steps)
    S = np.zeros((steps, steps))
    for i, vy in enumerate(gy):
        for j, vx in enumerate(gx):
            S[i, j] = iwe(x, y, t_s, vx, vy, t0, bins, extent).var()

    fig = plt.figure(figsize=(6, 5))
    plt.pcolormesh(gx, gy, S, cmap="viridis", shading="auto")
    plt.plot(vx_est, vy_est, "r+", ms=14, mew=2, label="estimate")
    plt.xlabel("vx (px/s)")
    plt.ylabel("vy (px/s)")
    plt.title("Variance surface around estimate")
    plt.colorbar(label="variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=125)
    plt.close(fig)
    print(f"Saved variance surface plot to {out_path}")
    print("A single, clean peak here is a good sign. Multiple comparable "
          "peaks, a flat plateau, or the estimate sitting right at the "
          "edge of this window all mean something's off (see notes "
          "below).\n")


def plot_iwe(x, y, t_s, t0, vx, vy, bins, extent, out_path="iwe_at_estimate.png"):
    img = iwe(x, y, t_s, vx, vy, t0, bins, extent)
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(img.T, origin="lower", cmap="magma",
               extent=[extent[0][0], extent[0][1], extent[1][0], extent[1][1]])
    plt.xlabel("x' (px)")
    plt.ylabel("y' (px)")
    plt.title(f"Image of Warped Events @ v=({vx:.1f}, {vy:.1f}) px/s")
    plt.colorbar(label="events/bin")
    plt.tight_layout()
    plt.savefig(out_path, dpi=125)
    plt.close(fig)
    print(f"Saved IWE plot to {out_path}")
    print("This should look like a sharp, tight blob (or a few tight "
          "blobs, one per marker) if the velocity estimate is correct - a "
          "smeared streak means the motion wasn't fully compensated.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Sanity-check a GENX320 event log and estimate velocity "
                     "via contrast maximization.")
    parser.add_argument("path", help="path to the events.bin log file")
    parser.add_argument("--start", type=float, default=None,
                         help="start time in seconds (relative to the first "
                              "logged event) - restricts the VELOCITY "
                              "ESTIMATE to this window. Diagnostic plots "
                              "still use the full recording so you can find "
                              "this value in the first place.")
    parser.add_argument("--end", type=float, default=None,
                         help="end time in seconds (relative to the first "
                              "logged event) - see --start.")
    parser.add_argument("--hot-pixel-factor", type=float, default=20.0,
                         help="flag any pixel with an event count more than "
                              "this many times the median nonzero pixel "
                              "count as a hot/stuck pixel (default 20)")
    parser.add_argument("--no-hot-pixel-filter", action="store_true",
                         help="disable hot pixel detection/filtering")
    args = parser.parse_args()

    ev_type, t_us, x, y = load_events(args.path)

    if not args.no_hot_pixel_filter:
        hot_pixels, counts, threshold = detect_hot_pixels(
            x, y, factor=args.hot_pixel_factor)
        if hot_pixels:
            keep = mask_hot_pixels(x, y, hot_pixels)
            removed = len(x) - int(keep.sum())
            print(f"Hot-pixel filter: flagged {len(hot_pixels)} pixel(s) "
                  f"with count > {threshold:.0f} events, removing "
                  f"{removed} events ({100 * removed / len(x):.2f}% of "
                  f"total).")
            for (px, py) in sorted(hot_pixels):
                print(f"  hot pixel at (x={px}, y={py}): "
                      f"{int(counts[px, py])} events")
            print()
            ev_type, t_us, x, y = (ev_type[keep], t_us[keep],
                                   x[keep], y[keep])
        else:
            print("Hot-pixel filter: no pixels flagged.\n")
    else:
        print("Hot-pixel filter disabled.\n")

    sanity_check(ev_type, t_us, x, y)

    t_s = (t_us - t_us.min()) / 1e6

    plot_raw(t_us, x, y)
    plot_event_rate(t_us)
    plot_spatial_heatmap(x, y)

    # ---- crop to the real motion window before estimating velocity ----
    if args.start is not None or args.end is not None:
        start = args.start if args.start is not None else 0.0
        end = args.end if args.end is not None else t_s.max()
        mask = (t_s >= start) & (t_s <= end)
        x_est, y_est, t_s_est = x[mask], y[mask], t_s[mask]
        print(f"Restricting velocity estimate to t in [{start:.2f}, "
              f"{end:.2f}] s -> {mask.sum()} events "
              f"(out of {len(t_s)} total)\n")
        plot_cropped_window(t_us, x, y, start, end)
    else:
        x_est, y_est, t_s_est = x, y, t_s
        print("No --start/--end given - using the FULL recording for the "
              "velocity estimate. Once you've picked a window from "
              "event_rate.png, re-run with e.g. --start 1.2 --end 2.4\n")

    if len(t_s_est) < 50:
        print("Too few events in this window to estimate anything useful - "
              "widen --start/--end and try again.")
        return

    # Region of interest - crop this further to where the marker actually
    # moves if you know it, to avoid diluting the variance signal with
    # background noise elsewhere in the frame. Defaults to the full extent
    # of the (possibly time-cropped) events being used for the estimate.
    extent = [[float(x_est.min()) - 1, float(x_est.max()) + 1],
              [float(y_est.min()) - 1, float(y_est.max()) + 1]]
    bins = 160

    vx, vy, t0 = estimate_velocity(x_est, y_est, t_s_est, extent, bins=bins)
    speed = (vx ** 2 + vy ** 2) ** 0.5
    print(f"Estimated speed: {speed:.2f} px/s\n")

    plot_variance_surface(x_est, y_est, t_s_est, t0, extent, bins, vx, vy)
    plot_iwe(x_est, y_est, t_s_est, t0, vx, vy, bins, extent)


if __name__ == "__main__":
    main()
