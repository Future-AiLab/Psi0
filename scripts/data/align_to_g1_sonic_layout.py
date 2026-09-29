#!/usr/bin/env python3
"""Rewrite a psix-flavour finetune pack into the g1_sonic_lerobot_0810_merged layout.

The two families hold the same joints but lay them out differently:

    psix  (e.g. .data/finetune/psix_steerability_0918_*/g1)
        observation.state (45)       hand(14) | arm(14) | leg(12) | waist(3) | neck(2)   -- unnamed
        action (36)                  hand(14) | arm(14) | torso(3) | base(5)
        action.neck (2)              separate column
        action.body_token_v1_1 (64)  separate column, 1/16-grid token

    0810  (.data/g1_sonic_lerobot_0810_merged, the --reference)
        observation.state (45)       leg(12) | waist(3) | arm(14) | hand(14) | neck(2)   -- named
        action (80)                  hand(14) | neck(2) | token(64)                     -- named

This script rewrites a psix pack IN PLACE so it matches the reference:

* ``observation.state`` is permuted into the reference joint order. The permutation is
  derived by NAME: the reference names every column; the psix column names come from
  the pack's own info.json when present, otherwise from the reference names regrouped
  into ``--source-state-order`` (default hand arm leg waist neck).
* ``action`` is rebuilt as the concatenation the reference's modality.json describes
  (hand_joints | neck | token), pulling each block from the psix column that holds it
  (``action[0:14]``, ``action.neck``, ``action.body_token_v1_1``). The consumed
  columns are dropped from the parquets and from info.json.
* ``meta/stats.json`` and ``meta/episodes_stats.jsonl`` are transformed the same way
  (per-dim arrays are permuted / concatenated, so they stay exact).
* ``meta/info.json`` gets the reference's names for both columns and the new action
  shape; ``meta/modality.json`` takes the reference's ``state`` and ``action`` sections
  and keeps everything else (video, annotation, subgoal, meta) untouched.

Every meta file it edits is backed up next to itself as ``*.pre_align.bak``. Parquets are
rewritten atomically (tmp + rename). Re-running on an already aligned pack is a no-op.

NOT touched (differs between the families, decide separately): the video key
(``observation.images.egocentric`` vs ``observation.images.head``), the task string
column (``task_description`` vs ``annotation.task``), and the extra psix columns
(subgoal / memory / subtask fields).

Usage:
    python scripts/data/align_to_g1_sonic_layout.py \
        .data/finetune/psix_steerability_0918_train/g1 \
        .data/finetune/psix_steerability_0918_val/g1 \
        --reference .data/g1_sonic_lerobot_0810_merged [--jobs 16] [--dry-run]
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STATE_KEY = "observation.state"
ACTION_KEY = "action"
# reference modality.json action block name -> candidate psix modality.json block names
ACTION_BLOCK_ALIASES = {
    "hand_joints": ["hand_joints", "hand"],
    "neck": ["neck"],
    "token": ["token", "body_token_v1_1", "body_token", "body_token_v1"],
}
GROUP_PATTERNS = {
    "hand": lambda n: "_hand_" in n,
    "arm": lambda n: any(j in n for j in ("shoulder", "elbow", "wrist")),
    "leg": lambda n: any(j in n for j in ("hip", "knee", "ankle")),
    "waist": lambda n: n.startswith("waist"),
    "neck": lambda n: n.startswith("neck"),
}


# ------------------------------------------------------------------------ plan
def load_json(p: Path):
    return json.loads(p.read_text())


def names_of(info: dict, key: str) -> list[str] | None:
    n = info["features"][key].get("names")
    if isinstance(n, dict):
        n = list(n.values())[0]
    return list(n) if isinstance(n, list) else None


def group_of(name: str) -> str:
    for g, f in GROUP_PATTERNS.items():
        if f(name):
            return g
    raise SystemExit(f"cannot assign joint {name!r} to a group")


def regroup(ref_names: list[str], order: list[str]) -> list[str]:
    groups: dict[str, list[str]] = {g: [] for g in GROUP_PATTERNS}
    for n in ref_names:
        groups[group_of(n)].append(n)
    unknown = [g for g in order if g not in groups]
    if unknown:
        raise SystemExit(f"--source-state-order has unknown groups {unknown}; choose from {list(groups)}")
    return [n for g in order for n in groups[g]]


def build_perm(src: list[str], ref: list[str]) -> list[int]:
    """new[:, j] = old[:, perm[j]]"""
    if sorted(src) != sorted(ref):
        raise SystemExit(f"state joints differ: only in source {sorted(set(src) - set(ref))}, "
                         f"only in reference {sorted(set(ref) - set(src))}")
    idx = {n: i for i, n in enumerate(src)}
    return [idx[n] for n in ref]


def action_plan(src_mod: dict, ref_mod: dict) -> list[tuple[str, str, int, int]]:
    """[(ref_block, src_original_key, start, end), ...] in reference column order."""
    ref_blocks = sorted(ref_mod["action"].items(), key=lambda kv: kv[1]["start"])
    plan = []
    for name, spec in ref_blocks:
        if spec["original_key"] != ACTION_KEY:
            raise SystemExit(f"reference action block {name} lives in {spec['original_key']}, expected {ACTION_KEY}")
        want = spec["end"] - spec["start"]
        src_spec = None
        for alias in ACTION_BLOCK_ALIASES.get(name, [name]):
            if alias in src_mod["action"]:
                src_spec = src_mod["action"][alias]
                break
        if src_spec is None:
            raise SystemExit(f"source modality.json has no action block for reference block {name!r} "
                             f"(tried {ACTION_BLOCK_ALIASES.get(name, [name])})")
        got = src_spec["end"] - src_spec["start"]
        if got != want:
            raise SystemExit(f"action block {name}: reference wants {want} dims, source block has {got}")
        plan.append((name, src_spec["original_key"], src_spec["start"], src_spec["end"]))
    return plan


# --------------------------------------------------------------------- parquet
def _list_column(values: np.ndarray, like: pa.DataType) -> pa.Array:
    flat = pa.array(values.reshape(-1), type=like.value_type)
    if pa.types.is_fixed_size_list(like):
        return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])
    offsets = pa.array(np.arange(values.shape[0] + 1, dtype=np.int32) * values.shape[1])
    return pa.ListArray.from_arrays(offsets, flat)


def rewrite_parquet(job) -> tuple[str, int]:
    path, perm, plan, drop = job
    file = Path(path)
    t = pq.read_table(file)
    n = t.num_rows

    si = t.schema.get_field_index(STATE_KEY)
    sfield = t.schema.field(si)
    s = np.asarray(t.column(si).to_pylist(), dtype=np.float32)
    if s.shape != (n, len(perm)):
        raise RuntimeError(f"{file}: {STATE_KEY} is {s.shape}, expected ({n}, {len(perm)})")
    t = t.set_column(si, sfield, _list_column(np.ascontiguousarray(s[:, perm]), sfield.type))

    ai = t.schema.get_field_index(ACTION_KEY)
    afield = t.schema.field(ai)
    cols = {}
    parts = []
    for _name, key, lo, hi in plan:
        if key not in cols:
            cols[key] = np.asarray(t.column(key).to_pylist(), dtype=np.float32)
        parts.append(cols[key][:, lo:hi])
    a = np.ascontiguousarray(np.concatenate(parts, axis=1))
    t = t.set_column(ai, afield, _list_column(a, afield.type))
    for key in drop:
        if key in t.column_names:
            t = t.remove_column(t.schema.get_field_index(key))

    compression = pq.ParquetFile(file).metadata.row_group(0).column(0).compression.lower()
    tmp = file.with_suffix(".align.tmp")
    pq.write_table(t, tmp, compression="none" if compression == "uncompressed" else compression)
    tmp.replace(file)
    ep_idx = int(t.column("episode_index")[0].as_py())
    per_ep = {STATE_KEY: _ep_stats(s[:, perm]), ACTION_KEY: _ep_stats(a)}
    return path, n, ep_idx, per_ep


def _ep_stats(arr: np.ndarray) -> dict:
    """Per-episode stats in the episodes_stats.jsonl convention (max/min/mean/std/count)."""
    return {
        "max": np.max(arr, axis=0).tolist(),
        "min": np.min(arr, axis=0).tolist(),
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


# ------------------------------------------------------------------------ meta
def backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".pre_align.bak")
    if not bak.exists():
        shutil.copy2(path, bak)


def transform_stats(stats: dict, perm: list[int], plan, drop: list[str]) -> dict:
    """Apply the state permutation / action concatenation to one stats dict
    ({feature: {stat: [..]}}), e.g. stats.json or one episodes_stats.jsonl record."""
    out = dict(stats)
    if STATE_KEY in out:
        blk = dict(out[STATE_KEY])
        for k, v in blk.items():
            if isinstance(v, list) and len(v) == len(perm):
                blk[k] = [v[i] for i in perm]
        out[STATE_KEY] = blk
    if ACTION_KEY in out:
        src_blocks = {key: out.get(key, {}) for _n, key, _lo, _hi in plan}
        stat_names = list(out[ACTION_KEY].keys())
        new = {}
        for st in stat_names:
            pieces = []
            ok = True
            for _n, key, lo, hi in plan:
                v = src_blocks[key].get(st)
                if not isinstance(v, list) or len(v) < hi:
                    ok = False
                    break
                pieces.append(v[lo:hi])
            if ok:
                new[st] = [x for p in pieces for x in p]
            else:
                new[st] = out[ACTION_KEY][st]  # e.g. scalar "count": keep as is
        out[ACTION_KEY] = new
    for key in drop:
        out.pop(key, None)
    return out


def already_aligned(info: dict, ref_info: dict) -> bool:
    return (names_of(info, STATE_KEY) == names_of(ref_info, STATE_KEY)
            and list(info["features"][ACTION_KEY]["shape"]) == list(ref_info["features"][ACTION_KEY]["shape"]))


# ------------------------------------------------------------------------ pack
def align_pack(pack: Path, ref: Path, source_order: list[str], jobs: int, dry_run: bool) -> None:
    info = load_json(pack / "meta/info.json")
    ref_info = load_json(ref / "meta/info.json")
    src_mod = load_json(pack / "meta/modality.json")
    ref_mod = load_json(ref / "meta/modality.json")

    print(f"\n=== {pack}")
    if already_aligned(info, ref_info):
        print("  already aligned with reference; nothing to do")
        return

    ref_state = names_of(ref_info, STATE_KEY)
    ref_action = names_of(ref_info, ACTION_KEY)
    if ref_state is None or ref_action is None:
        raise SystemExit(f"{ref}: reference must name both {STATE_KEY} and {ACTION_KEY}")
    src_state = names_of(info, STATE_KEY)
    if src_state is None:
        src_state = regroup(ref_state, source_order)
        print(f"  {STATE_KEY} unnamed in source; assuming {' | '.join(source_order)} order")
    if len(src_state) != info["features"][STATE_KEY]["shape"][0]:
        raise SystemExit(f"{STATE_KEY} has {info['features'][STATE_KEY]['shape']} dims but {len(src_state)} names")
    perm = build_perm(src_state, ref_state)

    plan = action_plan(src_mod, ref_mod)
    drop = sorted({key for _n, key, _lo, _hi in plan if key != ACTION_KEY})
    total = sum(hi - lo for _n, _k, lo, hi in plan)
    if total != ref_info["features"][ACTION_KEY]["shape"][0]:
        raise SystemExit(f"action plan gives {total} dims, reference has {ref_info['features'][ACTION_KEY]['shape']}")
    for key in drop:
        if key not in info["features"]:
            raise SystemExit(f"source info.json has no feature {key!r} needed for the action")

    print(f"  state perm : {perm}")
    print("  action     : " + " | ".join(f"{n}={k}[{lo}:{hi}]" for n, k, lo, hi in plan))
    print(f"  drop cols  : {drop}")
    parquets = sorted((pack / "data").rglob("episode_*.parquet"))
    print(f"  parquets   : {len(parquets)}")
    if dry_run:
        return

    # --- parquets ---
    per_ep: dict[int, dict] = {}
    rows = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for _p, n, ep_idx, st in ex.map(rewrite_parquet, [(str(p), perm, plan, drop) for p in parquets]):
            rows += n
            per_ep[ep_idx] = st
    print(f"  rewrote {len(parquets)} parquets, {rows} rows")

    # --- stats.json ---
    sp = pack / "meta/stats.json"
    if sp.exists():
        backup(sp)
        sp.write_text(json.dumps(transform_stats(load_json(sp), perm, plan, drop), indent=4) + "\n")
        print("  stats.json transformed")

    # --- episodes_stats.jsonl ---
    # Per-episode records only carry stats for a subset of features (here: action,
    # timestamp) and none for the neck/token columns, so the action entry cannot be
    # rebuilt by concatenation. Recompute it from the rewritten parquet instead, keeping
    # exactly the stat fields the record already had.
    ep = pack / "meta/episodes_stats.jsonl"
    if ep.exists():
        backup(ep)
        lines = []
        for line in ep.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            st = rec.get("stats")
            if isinstance(st, dict):
                st = transform_stats(st, perm, plan, drop)
                fresh = per_ep.get(int(rec["episode_index"]), {})
                for key in (STATE_KEY, ACTION_KEY):
                    if key in st and key in fresh:
                        st[key] = {f: fresh[key][f] for f in st[key] if f in fresh[key]}
                rec["stats"] = st
            lines.append(json.dumps(rec))
        ep.write_text("\n".join(lines) + "\n")
        print(f"  episodes_stats.jsonl transformed ({len(lines)} episodes)")

    # --- info.json ---
    ip = pack / "meta/info.json"
    backup(ip)
    info["features"][STATE_KEY]["names"] = list(ref_state)
    af = info["features"][ACTION_KEY]
    af["shape"] = [total]
    af["names"] = list(ref_action)
    for key in drop:
        info["features"].pop(key, None)
    ip.write_text(json.dumps(info, indent=4) + "\n")
    print("  info.json updated")

    # --- modality.json ---
    mp = pack / "meta/modality.json"
    backup(mp)
    src_mod["state"] = copy.deepcopy(ref_mod["state"])
    src_mod["action"] = copy.deepcopy(ref_mod["action"])
    mp.write_text(json.dumps(src_mod, indent=4) + "\n")
    print("  modality.json state/action sections replaced with reference's")

    # --- other meta files that still mention dropped keys ---
    for f in (pack / "meta").iterdir():
        if f.is_file() and f.suffix in (".json", ".jsonl") and not f.name.endswith(".bak"):
            if f.name in ("info.json", "modality.json", "stats.json", "episodes_stats.jsonl"):
                continue
            txt = f.read_text()
            hits = [k for k in drop if k in txt]
            if hits:
                print(f"  WARNING: {f.name} still mentions {hits}; not rewritten")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("packs", nargs="+", type=Path, help="Dataset roots to rewrite in place.")
    ap.add_argument("--reference", type=Path, default=Path(".data/g1_sonic_lerobot_0810_merged"))
    ap.add_argument("--source-state-order", nargs="+", default=["hand", "arm", "leg", "waist", "neck"],
                    metavar="GROUP", help="Joint-group order of an UNNAMED source state column.")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for p in list(args.packs) + [args.reference]:
        if not (p / "meta/info.json").exists():
            raise SystemExit(f"not a dataset root: {p}")
    for pack in args.packs:
        align_pack(pack, args.reference, args.source_state_order, args.jobs, args.dry_run)


if __name__ == "__main__":
    main()
