import os
import math
import pickle
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def ensure_xy(traj: Any) -> np.ndarray:
    arr = np.asarray(to_numpy(traj), dtype=np.float32)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Trajectory shape should end with [T, 2], but got {arr.shape}")
    return arr


def maybe_to_absolute(traj_xy: np.ndarray, assume_relative: bool) -> np.ndarray:
    if assume_relative:
        abs_xy = np.cumsum(traj_xy, axis=0)
    else:
        abs_xy = traj_xy.copy()

    # prepend ego origin at t=0
    return np.vstack([np.array([[0.0, 0.0]], dtype=np.float32), abs_xy])


def load_results(pkl_path: str) -> List[dict]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list in pkl, got {type(data)}")
    return data


def parse_results(data: List[dict]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for item in data:
        token = list(item.keys())[0]
        inner = item[token]
        parsed.append(
            {
                "token": token,
                "metric": inner.get("metric_results", {}),
                "plan": inner.get("plan_results", {}),
            }
        )
    return parsed


def split_l2_cases(
    parsed: List[Dict[str, Any]],
    ratios: Tuple[float, float, float] = (0.3333, 0.3334, 0.3333),
) -> Dict[str, List[Dict[str, Any]]]:
    """Split samples by plan_L2_3s quantile buckets: best / middle / worst."""
    valid = []
    for x in parsed:
        l2 = x["metric"].get("plan_L2_3s", None)
        if l2 is None:
            continue
        try:
            l2 = float(l2)
        except Exception:
            continue
        if np.isfinite(l2):
            valid.append((x, l2))

    if len(valid) == 0:
        return {"best": [], "middle": [], "worst": []}

    w_best, w_mid, w_worst = ratios
    total = w_best + w_mid + w_worst
    if total <= 0:
        w_best, w_mid, w_worst = 0.3333, 0.3334, 0.3333
        total = 1.0
    w_best, w_mid, w_worst = w_best / total, w_mid / total, w_worst / total
    q1 = w_best
    q2 = w_best + w_mid

    l2_vals = np.array([v for _, v in valid], dtype=np.float32)
    q1, q2 = np.quantile(l2_vals, [q1, q2])

    best, middle, worst = [], [], []
    for x, l2 in valid:
        if l2 <= q1:
            best.append(x)
        elif l2 <= q2:
            middle.append(x)
        else:
            worst.append(x)
    return {"best": best, "middle": middle, "worst": worst}


def global_to_ego_xy(
    points_global: np.ndarray,
    ego_translation: List[float],
    ego_rotation: List[float],
) -> np.ndarray:
    pts = np.asarray(points_global, dtype=np.float32)
    trans = np.array(ego_translation[:2], dtype=np.float32)
    yaw = Quaternion(ego_rotation).yaw_pitch_roll[0]

    pts = pts - trans[None, :]
    c, s = math.cos(-yaw), math.sin(-yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return pts @ rot.T


def rebuild_gt_from_nuscenes(nusc: NuScenes, sample_token: str, future_steps: int = 6) -> np.ndarray:
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", lidar_token)
    ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])

    cur = sample
    future_xy_global: List[List[float]] = []

    for _ in range(future_steps):
        nxt = cur["next"]
        if nxt == "":
            break
        cur = nusc.get("sample", nxt)
        lidar_token_next = cur["data"]["LIDAR_TOP"]
        sd_next = nusc.get("sample_data", lidar_token_next)
        ego_pose_next = nusc.get("ego_pose", sd_next["ego_pose_token"])
        future_xy_global.append(ego_pose_next["translation"][:2])

    if len(future_xy_global) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    gt = global_to_ego_xy(
        np.array(future_xy_global, dtype=np.float32),
        ego_pose["translation"],
        ego_pose["rotation"],
    )
    return np.vstack([np.array([[0.0, 0.0]], dtype=np.float32), gt])


def pick_best_index(plan: Dict[str, Any], source: str = "best_prev") -> int:
    """Pick best trajectory index from desired source."""
    if source in ("best_prev", "auto") and "best_prev_traj_idx" in plan:
        idx = to_numpy(plan["best_prev_traj_idx"])
        return int(np.asarray(idx).reshape(-1)[0])

    if source in ("score", "auto") and "pred_wm_cls" in plan:
        scores = to_numpy(plan["pred_wm_cls"])
        while scores.ndim > 2:
            scores = scores[0]
        if scores.ndim == 2 and scores.shape[-1] == 1:
            scores = scores[:, 0]
        if scores.ndim == 1:
            return int(np.argmax(scores))

    return 0


def infer_relative_mode(traj_xy: np.ndarray) -> bool:
    """Heuristic: if cumulative span is much larger than raw span, treat as relative offsets."""
    arr = ensure_xy(traj_xy)
    raw_span = float(np.max(np.ptp(arr, axis=0))) if len(arr) > 0 else 0.0
    csum = np.cumsum(arr, axis=0)
    csum_span = float(np.max(np.ptp(csum, axis=0))) if len(csum) > 0 else 0.0
    return csum_span > max(1.0, raw_span * 3.0)


def to_absolute_with_mode(traj_xy: np.ndarray, mode: str) -> np.ndarray:
    if mode == "relative":
        return maybe_to_absolute(traj_xy, assume_relative=True)
    if mode == "absolute":
        return maybe_to_absolute(traj_xy, assume_relative=False)
    # auto
    is_relative = infer_relative_mode(traj_xy)
    return maybe_to_absolute(traj_xy, assume_relative=is_relative)


def extract_gt_traj(
    metric: Dict[str, Any],
    nusc: Optional[NuScenes],
    token: str,
    future_steps: int,
    gt_mode: str,
    gt_source: str = "nusc",
) -> np.ndarray:
    def _from_metric() -> np.ndarray:
        for key in ["gt_ego_fut_trajs", "gt_ego_fut_traj", "fut_gt_traj"]:
            if key in metric:
                gt_xy = ensure_xy(metric[key])
                return to_absolute_with_mode(gt_xy, mode=gt_mode)
        return np.zeros((0, 2), dtype=np.float32)

    if gt_source in ("metric", "auto"):
        gt_metric = _from_metric()
        if len(gt_metric) > 0:
            return gt_metric

    if nusc is not None:
        gt_nusc = rebuild_gt_from_nuscenes(nusc, token, future_steps=future_steps)
        # If reconstructed GT is almost static, fallback to metric GT if available.
        if len(gt_nusc) > 1:
            disp = float(np.linalg.norm(gt_nusc[-1] - gt_nusc[0]))
            if disp < 1e-4:
                gt_metric = _from_metric()
                if len(gt_metric) > 0:
                    return gt_metric
        return gt_nusc
    return np.zeros((0, 2), dtype=np.float32)


def sanitize_traj(traj: np.ndarray, max_len: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(traj, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    valid = np.isfinite(arr).all(axis=1)
    arr = arr[valid]
    if max_len is not None and len(arr) > max_len:
        arr = arr[:max_len]
    return arr


def calc_l2_3s(pred: np.ndarray, gt: np.ndarray) -> float:
    t = min(len(pred), len(gt))
    if t <= 1:
        return float("nan")
    d = np.linalg.norm(pred[:t] - gt[:t], axis=1)
    return float(np.mean(d[1:]))


def apply_axis_transform(traj: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(traj, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[1] != 2:
        return arr
    if mode == "none":
        return arr
    if mode == "swap_xy":
        return arr[:, [1, 0]]
    if mode == "flip_x":
        arr[:, 0] = -arr[:, 0]
        return arr
    if mode == "flip_y":
        arr[:, 1] = -arr[:, 1]
        return arr
    if mode == "swap_flipx":
        arr = arr[:, [1, 0]]
        arr[:, 0] = -arr[:, 0]
        return arr
    if mode == "swap_flipy":
        arr = arr[:, [1, 0]]
        arr[:, 1] = -arr[:, 1]
        return arr
    return arr


def auto_align_gt_axis(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, str]:
    modes = ["none", "swap_xy", "flip_x", "flip_y", "swap_flipx", "swap_flipy"]
    best_mode = "none"
    best_gt = gt
    best_score = calc_l2_3s(pred, gt)
    for m in modes[1:]:
        g = apply_axis_transform(gt, m)
        s = calc_l2_3s(pred, g)
        if np.isfinite(s) and (not np.isfinite(best_score) or s < best_score):
            best_score = s
            best_mode = m
            best_gt = g
    return best_gt, best_mode


def ensure_origin_start(traj: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Prepend ego origin (0, 0) if trajectory does not already start there."""
    arr = np.asarray(traj, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    if len(arr) == 0:
        return np.array([[0.0, 0.0]], dtype=np.float32)
    if np.linalg.norm(arr[0]) <= tol:
        return arr
    return np.vstack([np.array([[0.0, 0.0]], dtype=np.float32), arr])


def draw_sample(
    item: Dict[str, Any],
    save_path: str,
    nusc: Optional[NuScenes],
    future_steps: int,
    pred_mode: str,
    gt_mode: str,
    gt_source: str,
    gt_axis_mode: str,
    best_index_source: str,
    show_all_candidates: bool,
    debug: bool,
) -> bool:
    token = item["token"]
    metric = item["metric"]
    plan = item["plan"]

    if "pred_ego_fut_trajs" not in plan:
        print(f"skip {token}: missing pred_ego_fut_trajs")
        return False

    pred_all = to_numpy(plan["pred_ego_fut_trajs"])  # expected [B, K, T, 2]
    while pred_all.ndim > 4:
        pred_all = pred_all[0]
    if pred_all.ndim != 4:
        print(f"skip {token}: pred_ego_fut_trajs shape invalid: {pred_all.shape}")
        return False

    batch = 0
    best_idx = pick_best_index(plan, source=best_index_source)
    best_idx = int(np.clip(best_idx, 0, pred_all.shape[1] - 1))

    pred_raw = ensure_xy(pred_all[batch, best_idx])
    pred_best = to_absolute_with_mode(pred_raw, mode=pred_mode)

    gt = extract_gt_traj(metric, nusc, token, future_steps, gt_mode=gt_mode, gt_source=gt_source)
    if gt.shape[0] == 0:
        # Never skip plotting: fallback to a static GT at ego origin.
        gt = np.zeros((future_steps + 1, 2), dtype=np.float32)

    pred_best = sanitize_traj(pred_best, max_len=future_steps + 1)
    gt = sanitize_traj(gt, max_len=future_steps + 1)
    pred_best = ensure_origin_start(pred_best)
    gt = ensure_origin_start(gt)
    if len(pred_best) <= 1:
        print(f"skip {token}: invalid pred/gt after sanitize")
        return False
    if len(gt) <= 1:
        gt = np.repeat(gt, future_steps + 1, axis=0)

    selected_axis_mode = gt_axis_mode
    if gt_axis_mode == "auto":
        gt, selected_axis_mode = auto_align_gt_axis(pred_best, gt)
    else:
        gt = apply_axis_transform(gt, gt_axis_mode)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("#f3f3f3")

    if show_all_candidates:
        for k in range(pred_all.shape[1]):
            p_raw = ensure_xy(pred_all[batch, k])
            p = to_absolute_with_mode(p_raw, mode=pred_mode)
            ax.plot(p[:, 0], p[:, 1], color="#6b7eff", linewidth=1.2, alpha=0.35, zorder=1)

    ax.plot(
        pred_best[:, 0],
        pred_best[:, 1],
        color="#1f2bd6",
        linewidth=2.2,
        linestyle="-",
        marker="o",
        markersize=4,
        label=f"Pred (idx={best_idx})",
        zorder=3,
    )

    ax.plot(gt[:, 0], gt[:, 1], color="#ff2d2d", linewidth=3.0, alpha=0.95, marker="o", markersize=4, label="GT", zorder=5)
    ax.scatter(gt[:, 0], gt[:, 1], s=22, color="#ff2d2d", zorder=6)
    ax.scatter(pred_best[:, 0], pred_best[:, 1], s=20, color="#1f2bd6", zorder=3)
    ax.scatter([0], [0], c="black", s=20, label="ego", zorder=4)

    l2_3s = metric.get("plan_L2_3s", None)
    col_3s = metric.get("plan_obj_box_col_3s", None)
    title = f"{token[:8]}"
    if l2_3s is not None:
        title += f" | L2_3s={float(l2_3s):.3f}"
    if col_3s is not None:
        title += f" | col_3s={float(col_3s):.4f}"
    ax.set_title(title, fontsize=12)

    all_pts = np.vstack([pred_best, gt, np.array([[0.0, 0.0]], dtype=np.float32)])
    all_pts = sanitize_traj(all_pts)
    if len(all_pts) == 0:
        all_pts = np.array([[0.0, 0.0]], dtype=np.float32)

    # Use robust bounds to avoid one outlier shrinking trajectories into a dot.
    low = np.percentile(all_pts, 5, axis=0)
    high = np.percentile(all_pts, 95, axis=0)
    xmin, ymin = low
    xmax, ymax = high
    span = float(max(xmax - xmin, ymax - ymin))
    if not np.isfinite(span) or span < 1e-3:
        span = 0.05
        xmin, xmax = -0.025, 0.025
        ymin, ymax = -0.025, 0.025
    elif span > 120:
        # Fallback window around ego to keep local trajectories visible.
        xmin, xmax = -30.0, 30.0
        ymin, ymax = -30.0, 30.0
        span = 60.0
    margin = max(0.02, 0.15 * span)

    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    if debug:
        print("\n=== DEBUG SAMPLE ===")
        print("token:", token)
        print("pred_all shape:", pred_all.shape)
        print("best_idx:", best_idx)
        if "pred_wm_cls" in plan:
            print("pred_wm_cls:", to_numpy(plan["pred_wm_cls"]).reshape(-1))
        print("pred mode:", pred_mode)
        print("gt mode:", gt_mode)
        print("gt source:", gt_source)
        print("gt axis mode:", selected_axis_mode)
        print("best index source:", best_index_source)
        print("pred_best[-1]:", pred_best[-1])
        print("gt[-1]:", gt[-1])
        print("calc_l2_3s_from_plot:", calc_l2_3s(pred_best, gt))
        print("metric plan_L2_3s:", metric.get("plan_L2_3s", None))
        print("pred span:", np.ptp(pred_best, axis=0))
        print("gt span:", np.ptp(gt, axis=0))
        print("pred_raw:\n", pred_raw)
        print("title:", title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize World4Drive predicted trajectory vs GT.")
    parser.add_argument("--pkl", required=True, help="Path to evaluation result pkl")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--dataroot", default=None, help="nuScenes dataroot (optional if GT in metric)")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--future-steps", type=int, default=6)
    parser.add_argument(
        "--pred-mode",
        choices=["auto", "relative", "absolute"],
        default="auto",
        help="How to interpret pred_ego_fut_trajs values",
    )
    parser.add_argument(
        "--gt-mode",
        choices=["auto", "relative", "absolute"],
        default="auto",
        help="How to interpret GT values when loaded from metric_results",
    )
    parser.add_argument("--pred-is-relative", action="store_true", help="Deprecated alias for --pred-mode relative")
    parser.add_argument("--gt-is-relative", action="store_true", help="Deprecated alias for --gt-mode relative")
    parser.add_argument(
        "--gt-source",
        choices=["nusc", "metric", "auto"],
        default="auto",
        help="Where GT comes from: auto tries metric GT first then nuScenes reconstruction",
    )
    parser.add_argument(
        "--best-index-source",
        choices=["best_prev", "score", "auto"],
        default="best_prev",
        help="Which field to use for best candidate index (default best_prev_traj_idx)",
    )
    parser.add_argument(
        "--gt-axis-mode",
        choices=["auto", "none", "swap_xy", "flip_x", "flip_y", "swap_flipx", "swap_flipy"],
        default="auto",
        help="Axis transform for GT to match prediction convention; auto picks the lowest L2 transform",
    )
    parser.add_argument("--show-all-candidates", action="store_true", help="Draw 6 candidate trajectories")
    parser.add_argument(
        "--mode",
        choices=["worst_l2_3s", "best_l2_3s", "middle_l2_3s", "first"],
        default="worst_l2_3s",
    )
    parser.add_argument(
        "--print-case-stats",
        action="store_true",
        help="Print best/middle/worst case counts based on L2 quantile buckets",
    )
    parser.add_argument(
        "--case-ratios",
        default="20,60,20",
        help="Best/Middle/Worst split ratios, e.g. 20,60,20",
    )
    parser.add_argument("--debug-first", action="store_true")
    args = parser.parse_args()
    if args.pred_is_relative:
        args.pred_mode = "relative"
    if args.gt_is_relative:
        args.gt_mode = "relative"
    try:
        parts = [float(x.strip()) for x in args.case_ratios.split(",")]
        if len(parts) != 3:
            raise ValueError
        case_ratios = (parts[0], parts[1], parts[2])
    except Exception:
        raise ValueError("--case-ratios must be three numbers like 20,60,20")

    os.makedirs(args.save_dir, exist_ok=True)

    nusc = None
    if args.dataroot:
        print("Loading nuScenes...")
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    print("Loading result pkl...")
    data = load_results(args.pkl)
    parsed = parse_results(data)

    parsed = [x for x in parsed if x["metric"].get("fut_valid_flag", True)]
    print("After filter, valid samples:", len(parsed))

    cases = split_l2_cases(parsed, ratios=case_ratios)
    total_case = len(cases["best"]) + len(cases["middle"]) + len(cases["worst"])
    if args.print_case_stats and total_case > 0:
        print(
            "Case stats (L2 quantiles): "
            f"best={len(cases['best'])} ({len(cases['best']) / total_case:.2%}), "
            f"middle={len(cases['middle'])} ({len(cases['middle']) / total_case:.2%}), "
            f"worst={len(cases['worst'])} ({len(cases['worst']) / total_case:.2%})"
        )

    if args.mode == "worst_l2_3s":
        selected_pool = cases["worst"] if len(cases["worst"]) > 0 else parsed
        selected_pool.sort(key=lambda x: x["metric"].get("plan_L2_3s", -1), reverse=True)
    elif args.mode == "best_l2_3s":
        selected_pool = cases["best"] if len(cases["best"]) > 0 else parsed
        selected_pool.sort(key=lambda x: x["metric"].get("plan_L2_3s", 1e9))
    elif args.mode == "middle_l2_3s":
        selected_pool = cases["middle"] if len(cases["middle"]) > 0 else parsed
        selected_pool.sort(key=lambda x: x["metric"].get("plan_L2_3s", 1e9))
    else:
        selected_pool = parsed

    selected = selected_pool[: args.num]

    saved = 0
    for i, item in enumerate(selected):
        save_path = os.path.join(args.save_dir, f"{i:03d}_{item['token']}.png")
        try:
            ok = draw_sample(
                item=item,
                save_path=save_path,
                nusc=nusc,
                future_steps=args.future_steps,
                pred_mode=args.pred_mode,
                gt_mode=args.gt_mode,
                gt_source=args.gt_source,
                gt_axis_mode=args.gt_axis_mode,
                best_index_source=args.best_index_source,
                show_all_candidates=args.show_all_candidates,
                debug=(args.debug_first and i == 0),
            )
            if ok:
                saved += 1
                print(f"[{i+1}/{len(selected)}] saved: {save_path}")
            else:
                print(f"[{i+1}/{len(selected)}] skipped: {item['token']}")
        except Exception as e:
            print(f"[{i+1}/{len(selected)}] failed on {item['token']}: {e}")

    print(f"Done. Saved {saved} images to: {args.save_dir}")


if __name__ == "__main__":
    main()
