#!/usr/bin/env python3
"""Merge N LeRobot v3 (PsiX-style) datasets with an identical feature layout into ONE.

Companion to ``split_lerobot_train_val.py`` -- same on-disk conventions (``data/``,
``videos/``, ``images/``, ``meta/``), same hardlink placement, same re-indexing rules.

What it does
------------
* Episodes are appended in the order the sources are given and renumbered to a
  contiguous ``0..N-1`` range: parquet / video / subgoal-image files are renamed, the
  ``episode_index`` column is rewritten, and any string column embedding an
  ``episode_XXXXXX`` path is patched. ``index`` / ``frame_index`` are left untouched
  (this dataset family stores ``index`` per-episode).
* ``tasks.jsonl`` is unioned by task NAME: a task string already seen keeps its index,
  a new one gets the next free index. ``task_index`` in every parquet and the
  ``tasks`` list in ``episodes.jsonl`` are remapped accordingly.
* ``episodes.jsonl`` / ``episodes_stats.jsonl`` are concatenated with the new indices
  and fresh ``dataset_from_index`` / ``dataset_to_index`` offsets.
* ``meta/info.json`` totals are rewritten; every other ``meta/*`` file (modality.json,
  ...) is copied from the FIRST source, with a warning if a later source's copy differs.

Stats
-----
``meta/stats.json`` is computed ONCE over every parquet of the ``--stats-srcs`` datasets
(default: the ``--srcs`` themselves) using the gear loader formula
(mean/std/min/max/q01/q99 over float features) and written into the output. To build a
train/val pair that normalises identically, run the script twice and point the val run's
``--stats-srcs`` at the TRAIN sources -- or simply pass ``--stats-from`` with the train
output's ``meta/stats.json`` to copy it byte-for-byte.

Example
-------
    # train
    python scripts/data/merge_lerobot_datasets.py \
        --srcs .data/finetune/pnp_multi_obj_standing_0916_train/g1 \
               .data/finetune/trash_or_cart_0918_train/g1 \
        --out  .data/finetune/psix_steerability_0918_train/g1

    # val, sharing the train stats
    python scripts/data/merge_lerobot_datasets.py \
        --srcs .data/finetune/pnp_multi_obj_standing_0916_val/g1 \
               .data/finetune/trash_or_cart_0918_val/g1 \
        --out  .data/finetune/psix_steerability_0918_val/g1 \
        --stats-from .data/finetune/psix_steerability_0918_train/g1/meta/stats.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# --------------------------------------------------------------------------- io
def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=4))


def place_file(src: Path, dst: Path, mode: str) -> None:
    """Materialize ``src`` at ``dst`` via hardlink (default), symlink, or copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # cross-device etc. -> fall back to copy
    if mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
        return
    shutil.copy2(src, dst)


def place_dir(src_dir: Path, dst_dir: Path, mode: str) -> None:
    for f in sorted(src_dir.rglob("*")):
        if f.is_file():
            place_file(f, dst_dir / f.relative_to(src_dir), mode)


# ------------------------------------------------------------------------ stats
def compute_full_stats(srcs: list[Path], info: dict) -> dict:
    """Stats over every parquet of every source -- mirrors the gear loader's
    ``calculate_dataset_statistics`` (float features only, mean/std/min/max/q01/q99)."""
    float_feats = [k for k, v in info["features"].items() if "float" in v["dtype"]]
    per_feat: dict[str, list[np.ndarray]] = {f: [] for f in float_feats}
    for src in srcs:
        for pf in sorted((src / "data").rglob("episode_*.parquet")):
            t = pq.read_table(pf, columns=float_feats)
            for f in float_feats:
                a = np.asarray(t.column(f).to_pylist(), dtype=np.float32)
                if a.ndim == 1:
                    a = a[:, None]
                per_feat[f].append(a)
    stats = {}
    for f in float_feats:
        arr = np.vstack(per_feat[f])
        stats[f] = {
            "max": np.max(arr, axis=0).tolist(),
            "min": np.min(arr, axis=0).tolist(),
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }
    return stats


# --------------------------------------------------------------------- checks
def feature_sig(info: dict) -> dict:
    return {k: (v.get("dtype"), tuple(v.get("shape") or ())) for k, v in info["features"].items()}


def check_compatible(infos: list[dict], srcs: list[Path]) -> None:
    ref = infos[0]
    for src, info in zip(srcs[1:], infos[1:]):
        for k in ("codebase_version", "fps", "robot_type", "chunks_size", "data_path", "video_path"):
            if info.get(k) != ref.get(k):
                raise SystemExit(f"{src}: info.json '{k}' = {info.get(k)!r} differs from {srcs[0]}: {ref.get(k)!r}")
        a, b = feature_sig(ref), feature_sig(info)
        if a != b:
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            raise SystemExit(f"{src}: features differ from {srcs[0]} on {diff}")


# ------------------------------------------------------------------------ merge
def merge(srcs: list[Path], out: Path, mode: str, stats_obj: dict | None, stats_srcs: list[Path]) -> None:
    infos = [json.loads((s / "meta" / "info.json").read_text()) for s in srcs]
    check_compatible(infos, srcs)
    info = infos[0]
    chunks_size = info["chunks_size"]
    data_tmpl = info["data_path"]
    video_tmpl = info["video_path"]
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    string_feats = [k for k, v in info["features"].items() if v["dtype"] == "string"]

    # ---- task table: union by task name -------------------------------------
    tasks_out: list[dict] = []
    name_to_new: dict[str, int] = {}

    new_eps, new_stats = [], []
    new_idx = 0
    running = 0
    for src in srcs:
        ep_meta = {int(e["episode_index"]): e for e in load_jsonl(src / "meta" / "episodes.jsonl")}
        st_path = src / "meta" / "episodes_stats.jsonl"
        st_meta = {int(e["episode_index"]): e for e in load_jsonl(st_path)} if st_path.exists() else {}
        tasks_path = src / "meta" / "tasks.jsonl"
        src_tasks = load_jsonl(tasks_path) if tasks_path.exists() else []
        old_to_new: dict[int, int] = {}
        for row in src_tasks:
            name = row["task"]
            if name not in name_to_new:
                name_to_new[name] = len(tasks_out)
                new_row = dict(row)
                new_row["task_index"] = name_to_new[name]
                tasks_out.append(new_row)
            old_to_new[int(row["task_index"])] = name_to_new[name]

        img_root = src / "images"
        img_subdirs = [d.name for d in img_root.iterdir() if d.is_dir()] if img_root.is_dir() else []
        n_src = 0
        for orig in sorted(ep_meta):
            chunk = new_idx // chunks_size
            old_tok, new_tok = f"episode_{orig:06d}", f"episode_{new_idx:06d}"

            # --- parquet ---
            src_pq = src / data_tmpl.format(episode_chunk=orig // chunks_size, episode_index=orig)
            t = pq.read_table(src_pq)
            n = t.num_rows
            ei = t.schema.get_field_index("episode_index")
            t = t.set_column(ei, "episode_index",
                             pa.array([new_idx] * n, type=t.schema.field("episode_index").type))
            ti = t.schema.get_field_index("task_index")
            if ti >= 0 and old_to_new:
                vals = [old_to_new.get(int(v), int(v)) if v is not None else v for v in t.column(ti).to_pylist()]
                t = t.set_column(ti, "task_index", pa.array(vals, type=t.schema.field("task_index").type))
            if old_tok != new_tok:
                for f in string_feats:
                    vals = t.column(f).to_pylist()
                    if any(v and old_tok in v for v in vals):
                        vals = [v.replace(old_tok, new_tok) if v else v for v in vals]
                        fi = t.schema.get_field_index(f)
                        t = t.set_column(fi, f, pa.array(vals, type=t.schema.field(f).type))
            dst_pq = out / data_tmpl.format(episode_chunk=chunk, episode_index=new_idx)
            dst_pq.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(t, dst_pq)

            # --- videos ---
            for vk in video_keys:
                s = src / video_tmpl.format(episode_chunk=orig // chunks_size, episode_index=orig, video_key=vk)
                if s.exists():
                    place_file(s, out / video_tmpl.format(episode_chunk=chunk, episode_index=new_idx, video_key=vk), mode)

            # --- subgoal image folders ---
            for sub in img_subdirs:
                s = img_root / sub / old_tok
                if s.is_dir():
                    place_dir(s, out / "images" / sub / new_tok, mode)

            # --- meta rows ---
            line = dict(ep_meta[orig])
            line["episode_index"] = new_idx
            line["length"] = n
            line["dataset_from_index"] = running
            line["dataset_to_index"] = running + n - 1
            if isinstance(line.get("tasks"), list):
                line["tasks"] = [old_to_new.get(int(x), int(x)) for x in line["tasks"]]
            new_eps.append(line)
            running += n
            if orig in st_meta:
                st = dict(st_meta[orig])
                st["episode_index"] = new_idx
                new_stats.append(st)
            new_idx += 1
            n_src += 1
        print(f"  {src}: {n_src} episodes, {len(src_tasks)} tasks")

    # ---- meta ----------------------------------------------------------------
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    new_info = dict(info)
    new_info["total_episodes"] = new_idx
    new_info["total_frames"] = running
    new_info["total_tasks"] = len(tasks_out)
    new_info["total_videos"] = new_idx * len(video_keys)
    new_info["total_chunks"] = (new_idx - 1) // chunks_size + 1 if new_idx else 0
    dump_json(meta / "info.json", new_info)
    dump_jsonl(meta / "episodes.jsonl", new_eps)
    if new_stats:
        dump_jsonl(meta / "episodes_stats.jsonl", new_stats)
    dump_jsonl(meta / "tasks.jsonl", tasks_out)

    if stats_obj is None:
        print(f"  computing stats.json over {[str(s) for s in stats_srcs]} ...")
        stats_obj = compute_full_stats(stats_srcs, info)
    dump_json(meta / "stats.json", stats_obj)

    handled = {"info.json", "episodes.jsonl", "episodes_stats.jsonl", "stats.json", "tasks.jsonl"}
    for f in (srcs[0] / "meta").iterdir():
        if f.is_file() and f.name not in handled:
            shutil.copy2(f, meta / f.name)
            for other in srcs[1:]:
                o = other / "meta" / f.name
                if o.exists() and o.read_bytes() != f.read_bytes():
                    print(f"  WARNING: {o} differs from {f}; kept the first source's copy")

    print(f"\n-> {out}  ({new_idx} eps, {running} frames, {len(tasks_out)} tasks)")


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--srcs", required=True, nargs="+", type=Path,
                    help="Source dataset roots (folders holding data/ videos/ meta/), in append order.")
    ap.add_argument("--out", required=True, type=Path, help="Output dataset root.")
    ap.add_argument("--stats-srcs", nargs="+", type=Path, default=None,
                    help="Datasets to compute meta/stats.json over (default: --srcs).")
    ap.add_argument("--stats-from", type=Path, default=None,
                    help="Copy this stats.json verbatim instead of computing (e.g. the train output's).")
    ap.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink",
                    help="How to place videos/images (parquet is always rewritten). Default hardlink.")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing output dir.")
    args = ap.parse_args()

    for s in args.srcs:
        if not (s / "meta" / "info.json").exists():
            raise SystemExit(f"not a dataset root: {s}")
    if args.out.exists():
        if not args.force:
            raise SystemExit(f"Output exists (use --force to overwrite): {args.out}")
        shutil.rmtree(args.out)

    stats_obj = json.loads(args.stats_from.read_text()) if args.stats_from else None
    print(f"Merging {len(args.srcs)} sources -> {args.out}")
    merge(args.srcs, args.out, args.mode, stats_obj, args.stats_srcs or args.srcs)


if __name__ == "__main__":
    main()
