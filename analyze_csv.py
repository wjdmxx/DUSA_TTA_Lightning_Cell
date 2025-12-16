from pathlib import Path
import pandas as pd
import numpy as np


# =========================
# 你的两个实验目录（直接放若干 CSV）
# =========================
DIR_CONVNEXT = Path("/mnt/bit/liyuanxi/projects/DUSA_TTA_Lightning/outputs/conv_next_4e-5_sample_log/2025-12-16/02-38-03/sample_logs")
DIR_VIT      = Path("/mnt/bit/liyuanxi/projects/DUSA_TTA_Lightning/outputs/ViT_2e-5_sample_log/2025-12-16/02-45-32/sample_logs")

# 分桶数（用于单调性统计）
N_BINS = 10

# 过滤曲线：保留 top x%（按指标从大到小排序）
KEEP_FRACS = (0.8, 0.6, 0.5, 0.4, 0.2)


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def auc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    AUROC via rank statistic (tie-aware).
    Returns NaN if labels are all same or no valid samples.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    m = np.isfinite(y_score)
    y_true = y_true[m]
    y_score = y_score[m]
    if y_true.size == 0:
        return np.nan

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    ranks = pd.Series(y_score).rank(method="average").to_numpy()  # 1..n
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def decile_sep(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float]:
    """
    (acc_top10, acc_bottom10, sep)
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    m = np.isfinite(y_score)
    y_true = y_true[m]
    y_score = y_score[m]
    n = y_true.size
    if n < 10:
        return (np.nan, np.nan, np.nan)

    idx = np.argsort(y_score)  # ascending
    k = max(1, n // 10)
    bottom = idx[:k]
    top = idx[-k:]
    acc_bottom = float(y_true[bottom].mean()) if bottom.size else np.nan
    acc_top = float(y_true[top].mean()) if top.size else np.nan
    sep = acc_top - acc_bottom if np.isfinite(acc_top) and np.isfinite(acc_bottom) else np.nan
    return acc_top, acc_bottom, sep


def retention_acc(y_true: np.ndarray, y_score: np.ndarray, keep_fracs=KEEP_FRACS) -> dict:
    """
    Sort by score desc, keep top frac, compute acc.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    m = np.isfinite(y_score)
    y_true = y_true[m]
    y_score = y_score[m]
    n = y_true.size
    if n == 0:
        return {f"keep{int(f*100)}_acc": np.nan for f in keep_fracs}

    order = np.argsort(-y_score)  # descending
    out = {}
    for f in keep_fracs:
        k = max(1, int(round(n * f)))
        kept = order[:k]
        out[f"keep{int(f*100)}_acc"] = float(y_true[kept].mean())
    return out


def quantile_bin_monotonicity(y_true: np.ndarray, y_score: np.ndarray, n_bins=N_BINS) -> dict:
    """
    Bin by quantiles, compute:
      - acc_span = acc_last - acc_first (ascending bins)
      - mono_violations = count of decreases between adjacent bins
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    m = np.isfinite(y_score)
    y_true = y_true[m]
    y_score = y_score[m]
    if y_true.size < n_bins:
        return {"acc_span": np.nan, "mono_violations": np.nan}

    s = pd.Series(y_score)
    try:
        bins = pd.qcut(s, q=n_bins, duplicates="drop")
    except ValueError:
        bins = pd.cut(s, bins=n_bins, include_lowest=True)

    dfb = pd.DataFrame({"b": bins, "y": y_true})
    accs = dfb.groupby("b", observed=False)["y"].mean().to_numpy()
    if accs.size < 2:
        return {"acc_span": np.nan, "mono_violations": np.nan}

    return {
        "acc_span": float(accs[-1] - accs[0]),
        "mono_violations": int(np.sum(np.diff(accs) < 0)),
    }


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return np.nan
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def analyze_csv(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)

    if "correct" not in df.columns:
        return {"task": csv_path.stem, "file": str(csv_path), "n": 0, "acc": np.nan, "error": "missing_correct"}

    y = safe_numeric(df["correct"]).fillna(0).astype(int).to_numpy()
    n = int(len(y))
    acc = float(np.mean(y)) if n > 0 else np.nan

    out = {
        "task": csv_path.stem,
        "file": str(csv_path),
        "n": n,
        "acc": acc,
        "has_tau": int("kendall_tau" in df.columns),
        "has_gap": int("gap_norm" in df.columns),
    }

    # Kendall tau block
    if "kendall_tau" in df.columns:
        tau = safe_numeric(df["kendall_tau"]).to_numpy()
        out["tau_auc"] = auc_rank(y, tau)
        ttop, tbot, tsep = decile_sep(y, tau)
        out["tau_acc_top10"] = ttop
        out["tau_acc_bottom10"] = tbot
        out["tau_sep_topbot10"] = tsep
        out.update({f"tau_{k}": v for k, v in retention_acc(y, tau).items()})
        out.update({f"tau_{k}": v for k, v in quantile_bin_monotonicity(y, tau).items()})

    # gap block
    if "gap_norm" in df.columns:
        gap = safe_numeric(df["gap_norm"]).to_numpy()
        out["gap_auc"] = auc_rank(y, gap)
        gtop, gbot, gsep = decile_sep(y, gap)
        out["gap_acc_top10"] = gtop
        out["gap_acc_bottom10"] = gbot
        out["gap_sep_topbot10"] = gsep
        out.update({f"gap_{k}": v for k, v in retention_acc(y, gap).items()})
        out.update({f"gap_{k}": v for k, v in quantile_bin_monotonicity(y, gap).items()})

    # redundancy check
    if ("kendall_tau" in df.columns) and ("gap_norm" in df.columns):
        tau = safe_numeric(df["kendall_tau"]).to_numpy()
        gap = safe_numeric(df["gap_norm"]).to_numpy()
        out["tau_gap_spearman"] = spearman_corr(tau, gap)

    return out


def analyze_dir(exp_dir: Path, exp_name: str) -> pd.DataFrame:
    csvs = sorted([p for p in exp_dir.glob("*.csv") if p.is_file()])
    rows = [analyze_csv(p) for p in csvs]
    df = pd.DataFrame(rows)
    df.insert(0, "experiment", exp_name)
    return df


def weighted_mean(df: pd.DataFrame, col: str, wcol: str = "n") -> float:
    if col not in df.columns:
        return np.nan
    x = df[col].to_numpy(dtype=float, copy=True)
    w = df[wcol].to_numpy(dtype=float, copy=True)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if m.sum() == 0:
        return np.nan
    return float(np.sum(x[m] * w[m]) / np.sum(w[m]))


def experiment_summary(df: pd.DataFrame) -> dict:
    s = {
        "experiment": df["experiment"].iloc[0] if len(df) else "unknown",
        "n_tasks": int(len(df)),
        "n_total": int(df["n"].sum()) if "n" in df.columns else 0,
        "acc": weighted_mean(df, "acc"),
        "tau_auc": weighted_mean(df, "tau_auc"),
        "gap_auc": weighted_mean(df, "gap_auc"),
        "tau_sep_topbot10": weighted_mean(df, "tau_sep_topbot10"),
        "gap_sep_topbot10": weighted_mean(df, "gap_sep_topbot10"),
        "tau_mono_violations": weighted_mean(df, "tau_mono_violations"),
        "gap_mono_violations": weighted_mean(df, "gap_mono_violations"),
        "tau_acc_span": weighted_mean(df, "tau_acc_span"),
        "gap_acc_span": weighted_mean(df, "gap_acc_span"),
        "tau_keep60_acc": weighted_mean(df, "tau_keep60_acc"),
        "gap_keep60_acc": weighted_mean(df, "gap_keep60_acc"),
        "tau_keep20_acc": weighted_mean(df, "tau_keep20_acc"),
        "gap_keep20_acc": weighted_mean(df, "gap_keep20_acc"),
        "tau_gap_spearman": weighted_mean(df, "tau_gap_spearman"),
    }
    # quick indicator: which proxy seems better
    if np.isfinite(s["tau_auc"]) and np.isfinite(s["gap_auc"]):
        s["proxy_auc_better"] = "tau" if s["tau_auc"] > s["gap_auc"] else "gap"
        s["proxy_auc_diff"] = float(s["tau_auc"] - s["gap_auc"])
    else:
        s["proxy_auc_better"] = ""
        s["proxy_auc_diff"] = np.nan
    return s


def main():
    assert DIR_CONVNEXT.exists(), f"Not found: {DIR_CONVNEXT}"
    assert DIR_VIT.exists(), f"Not found: {DIR_VIT}"

    df_c = analyze_dir(DIR_CONVNEXT, "ConvNeXt-L (lr=4e-5)")
    df_v = analyze_dir(DIR_VIT, "ViT-B/16 (lr=2e-5)")

    # 逐 task 对齐对比（用 task=文件名 stem）
    key = "task"
    df_merge = df_c.merge(df_v, on=key, how="outer", suffixes=("_convnext", "_vit"))

    # 你最关心的字段：按需增删
    compare_cols = [
        "task",
        "n_convnext", "acc_convnext", "tau_auc_convnext", "gap_auc_convnext",
        "tau_keep60_acc_convnext", "gap_keep60_acc_convnext",
        "tau_keep20_acc_convnext", "gap_keep20_acc_convnext",
        "tau_sep_topbot10_convnext", "gap_sep_topbot10_convnext",
        "tau_mono_violations_convnext", "gap_mono_violations_convnext",
        "n_vit", "acc_vit", "tau_auc_vit", "gap_auc_vit",
        "tau_keep60_acc_vit", "gap_keep60_acc_vit",
        "tau_keep20_acc_vit", "gap_keep20_acc_vit",
        "tau_sep_topbot10_vit", "gap_sep_topbot10_vit",
        "tau_mono_violations_vit", "gap_mono_violations_vit",
    ]
    # 自动过滤不存在的列
    compare_cols = [c for c in compare_cols if c in df_merge.columns]

    # delta
    for metric in ["acc", "tau_auc", "gap_auc", "tau_keep60_acc", "gap_keep60_acc", "tau_keep20_acc", "gap_keep20_acc"]:
        c1 = f"{metric}_convnext"
        c2 = f"{metric}_vit"
        if c1 in df_merge.columns and c2 in df_merge.columns:
            df_merge[f"{metric}_delta(convnext-vit)"] = df_merge[c1] - df_merge[c2]

    # 打印 task 对比表（按 acc_delta 排序）
    sort_col = "acc_delta(convnext-vit)" if "acc_delta(convnext-vit)" in df_merge.columns else None
    df_print = df_merge[compare_cols + ([sort_col] if sort_col else [])].copy()
    if sort_col:
        df_print = df_print.sort_values(by=sort_col, ascending=False, na_position="last")

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 240)

    print("\n" + "=" * 120)
    print("PER-TASK COMPARISON")
    print("=" * 120)
    print(df_print.to_string(index=False))

    # 实验级汇总
    sum_c = experiment_summary(df_c)
    sum_v = experiment_summary(df_v)
    df_sum = pd.DataFrame([sum_c, sum_v])

    print("\n" + "=" * 120)
    print("EXPERIMENT SUMMARY (weighted by n per task)")
    print("=" * 120)
    print(df_sum.to_string(index=False))

    # 保存结果到当前工作目录
    out_task = Path("per_task_comparison.csv")
    out_sum = Path("experiment_summary.csv")
    df_merge.to_csv(out_task, index=False)
    df_sum.to_csv(out_sum, index=False)

    print("\n" + "=" * 120)
    print("SAVED")
    print("=" * 120)
    print(f"[INFO] per-task:  {out_task.resolve()}")
    print(f"[INFO] summary:   {out_sum.resolve()}")


if __name__ == "__main__":
    main()
