import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ["HF_TOKEN"] = "..."

from scripts.data.download import download_from_huggingface

parser = argparse.ArgumentParser()
parser.add_argument("--psi0-raw-data-dir", type=str, default="/hfm/data/psi0")
args = parser.parse_args()

download_from_huggingface(
    repo_id="songlinwei/hfm",
    remote_dir="g1_real_raw",
    repo_type="dataset",
    local_dir=args.psi0_raw_data_dir,
    move_to_local_root=True,
)
