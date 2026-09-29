#!/usr/bin/env python3
"""Push one checkpoint of a run directory to ModelScope.

The upload mirror of scripts/deploy/rsync_run_dir.sh: same positional
arguments, same checkpoint selection, same exclusion set, same retry loop.
Where the rsync script pulls

    REMOTE_HOST:REMOTE_PROJECT_DIR/RUN_DIR  ->  LOCAL_BASE/.runs/SUBFOLDER/

this pushes

    LOCAL_BASE/RUN_DIR  ->  MS_REPO_ID/PATH_PREFIX/SUBFOLDER/<run name>

so the first two positional args name the ModelScope destination instead of
an ssh one.

    upload_ckpt_ms [ms-repo-id] [path-prefix] [run-dir] [ckpt-step] [local-base]

Set DRY_RUN=1 to print what would be uploaded without transferring anything,
MAX_RETRIES to change the retry budget (default 50, as in the rsync script), and
EXTRA_IGNORE=posttrained (comma-separated globs) to leave extra run-level folders
out of the metadata upload.

The API token is read from the environment (MODELSCOPE_API_TOKEN, falling back
to MS_UPLOAD_TOKEN), loaded from the project .env if present. It is never
stored in this file.
"""

import os
import sys
import time
from pathlib import Path

# repo_root/scripts/deploy/modelscope_run_dir.py -> repo_root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REPO_ID = "uscpsix/psix-model"
DEFAULT_PATH_PREFIX = "runs"
DEFAULT_RUN_DIR = ".runs/psix_finetune/mem.rss26r.flow1000.cosine.lr5.0e-05.b128.gpus8.2606090813"
DEFAULT_CKPT_STEP = "40000"
DEFAULT_ENDPOINT = "https://www.modelscope.ai"

TOKEN_ENV_VAR = "MODELSCOPE_API_TOKEN"
FALLBACK_TOKEN_ENV_VAR = "MS_UPLOAD_TOKEN"

# Same states the rsync script refuses to move: they are large and rebuildable.
IGNORE_PATTERNS = [
    "random_states_*.pkl",
    "**/random_states_*.pkl",
    "optimizer.bin",
    "**/optimizer.bin",
    "scheduler.bin",
    "**/scheduler.bin",
]


def read_env_file(env_path):
    """Minimal .env reader, matching scripts/deploy/upload_ckpt.py."""
    env_values = {}
    if not env_path.exists():
        return env_values
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        env_values[key] = value
    return env_values


def load_env():
    env_values = read_env_file(PROJECT_ROOT / ".env")
    for key, value in env_values.items():
        os.environ.setdefault(key, value)
    return env_values


def resolve_token():
    token = os.environ.get(TOKEN_ENV_VAR) or os.environ.get(FALLBACK_TOKEN_ENV_VAR)
    if not token:
        sys.exit(
            f"Neither {TOKEN_ENV_VAR} nor {FALLBACK_TOKEN_ENV_VAR} is set. "
            f"Add one of them to {PROJECT_ROOT / '.env'} before uploading."
        )
    return token


def main():
    argv = sys.argv[1:]

    def arg(index, default):
        # Matches bash ${N:-default}: an empty argument falls back to the default.
        return argv[index] if len(argv) > index and argv[index] else default

    repo_id = arg(0, DEFAULT_REPO_ID)
    path_prefix = arg(1, DEFAULT_PATH_PREFIX).strip("/")
    run_dir = arg(2, DEFAULT_RUN_DIR)
    ckpt_step = arg(3, DEFAULT_CKPT_STEP)
    local_base = arg(4, os.getcwd())

    load_env()
    token = resolve_token()
    endpoint = os.environ.get("MODELSCOPE_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")

    source = (Path(local_base) / run_dir).resolve()
    if not source.is_dir():
        sys.exit(f"run directory not found: {source}")

    ckpt_dir = source / "checkpoints" / f"ckpt_{ckpt_step}"
    if not ckpt_dir.is_dir():
        sys.exit(f"checkpoint not found: {ckpt_dir}")

    # Mirror the rsync layout: strip the leading .runs/ so the remote path is
    # <prefix>/<subfolder>/<run name>, e.g. runs/psix_finetune/mem.rss26r...
    rel = run_dir.strip("/")
    if rel.startswith(".runs/"):
        rel = rel[len(".runs/"):]
    remote_dir = f"{path_prefix}/{rel}" if path_prefix else rel

    from modelscope.hub.api import HubApi

    api = HubApi(endpoint=endpoint, token=token)
    try:
        api.get_model(repo_id)
    except Exception as exc:  # noqa: BLE001 - surface auth/network problems up front
        sys.exit(
            f"Cannot access {endpoint}/models/{repo_id}: {exc}\n"
            f"Check that {TOKEN_ENV_VAR} is valid and has write access to the repo."
        )

    # EXTRA_IGNORE: comma-separated glob patterns dropped from the metadata job,
    # e.g. EXTRA_IGNORE=posttrained to leave the exported posttrained/ folder out.
    extra_ignore = []
    for pat in os.environ.get("EXTRA_IGNORE", "").split(","):
        pat = pat.strip().strip("/")
        if pat:
            extra_ignore += [pat, f"{pat}/**"]

    # Two uploads instead of a staging copy: checkpoints are tens of GB and
    # copying them just to filter would double the disk cost.
    #   1. run metadata (configs, logs) minus wandb/ and checkpoints/
    #   2. the one requested checkpoint, minus the excluded state files
    jobs = [
        (
            source,
            remote_dir,
            ["wandb", "wandb/**", "checkpoints", "checkpoints/**"] + extra_ignore,
            "run metadata",
        ),
        (
            ckpt_dir,
            f"{remote_dir}/checkpoints/ckpt_{ckpt_step}",
            list(IGNORE_PATTERNS),
            f"ckpt_{ckpt_step}",
        ),
    ]

    print(f"Uploading {source}")
    print(f"      to  {endpoint}/models/{repo_id} :: {remote_dir}")
    print(f"Checkpoint: ckpt_{ckpt_step} (optimizer/scheduler/random states excluded)")

    # Retry on failure (e.g. dropped connection) until the upload succeeds or we
    # run out of attempts, matching MAX_RETRIES in rsync_run_dir.sh.
    max_retries = int(os.environ.get("MAX_RETRIES", "50"))
    dry_run = os.environ.get("DRY_RUN", "") not in ("", "0")

    for folder, path_in_repo, ignore_patterns, label in jobs:
        if dry_run:
            print(f"[dry-run] would upload {folder}")
            print(f"[dry-run]           -> {repo_id}:{path_in_repo}")
            print(f"[dry-run]      ignoring {ignore_patterns}")
            continue
        attempt = 1
        while True:
            try:
                api.upload_folder(
                    repo_id=repo_id,
                    folder_path=str(folder),
                    path_in_repo=path_in_repo,
                    repo_type="model",
                    token=token,
                    ignore_patterns=ignore_patterns,
                    commit_message=f"Upload {source.name} {label}",
                )
                print(f"upload of {label} completed successfully.")
                break
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - retry on anything transient
                if attempt >= max_retries:
                    print(
                        f"upload of {label} failed after {attempt} attempts; giving up: {exc}",
                        file=sys.stderr,
                    )
                    return 1
                print(
                    f"upload of {label} failed (attempt {attempt}/{max_retries}): {exc}; retrying in 5s...",
                    file=sys.stderr,
                )
                attempt += 1
                time.sleep(5)

    print("dry run complete; nothing was uploaded." if dry_run else "upload completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
