# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import csv
import argparse
from tqdm import tqdm
from ast import literal_eval
import gzip
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import ssl
import re

ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torchaudio
from torch.hub import download_url_to_file
from voxpopuli import ASR_LANGUAGES, ASR_ACCENTED_LANGUAGES, DOWNLOAD_BASE_URL
from voxpopuli.utils import multiprocess_run


SPLITS = ["train", "dev", "test"]


def cut_session(info: Tuple[str, Dict[str, List[Tuple[float, float]]]]) -> None:
    in_path, out_path_to_timestamps = info
    waveform, sr = torchaudio.load(in_path)
    duration = waveform.size(1)
    for out_path, timestamps in out_path_to_timestamps.items():
        segment = torch.cat(
            [waveform[:, int(s * sr): min(int(t * sr), duration)]
             for s, t in timestamps],
            dim=1
        )
        torchaudio.save(out_path, segment, sr)


def get(args):
    in_root = Path(args.root) / "raw_audios" / "original"
    out_root = Path(args.root) / "labelled_data"
    out_root.mkdir(exist_ok=True, parents=True)
    # Get metadata TSV
    url = f"{DOWNLOAD_BASE_URL}/annotations/asr/asr_{args.lang}.tsv.gz"
    tsv_path = out_root / Path(url).name
    if not tsv_path.exists():
        download_url_to_file(url, (out_root / Path(url).name).as_posix())
    with gzip.open(tsv_path, "rt") as f:
        metadata = [x for x in csv.DictReader(f, delimiter="|")]
    # Get segment into list
    items = defaultdict(dict)
    manifest = []
    for r in tqdm(metadata):
        split = r["split"]
        if split not in SPLITS:
            continue
        event_id = r["session_id"]
        year = event_id[:4]
        in_path = in_root / year / f"{event_id}_original.ogg"
        cur_out_root = out_root / year
        # cur_out_root.mkdir(exist_ok=True, parents=True)
        out_path = cur_out_root / "{}-{}.ogg".format(event_id, r["id_"])
        timestamps = [(t[0], t[1]) for t in literal_eval(r["vad"])]
        if not Path(out_path).exists() or not Path(out_path).stat().st_size:
                items[in_path.as_posix()][out_path.as_posix()] = timestamps
        # convo_id = '-'.join(out_path.stem.split('-')[:4])
        convo_id = re.split(r'-[a-z]{2}_', out_path.stem)[0]  # Split at 'de_' and take the first part
        manifest.append(
            (
            convo_id,
            # "20180704-0900-PLENARY-de_20180704"
            year,
            str(literal_eval(r["vad"])[0][0]),
            str(literal_eval(r["vad"])[0][1]),
            r["speaker_id"] + ": " + r["normed_text"],
            )
        )
    items = list(items.items())
    # Segment
    # multiprocess_run(items, cut_session)
    # Output per-split manifest
    header = [
        "id", "start_time", "end_time",  "speaker_id: normalized_text",
    ]
    # Create split-specific directory
    # (out_root / split).mkdir(exist_ok=True, parents=True)
    for cols in manifest:
        convo_id = cols[0]
        year = cols[1]
        (out_root / year).mkdir(exist_ok=True, parents=True)
        # with open(out_root / split / f"{cols[0]}.txt", "a", encoding="utf-8") as f_o:
        with open(out_root / year / f"{convo_id}_original.txt", "a", encoding="utf-8") as f_o:
            # f_o.write("\t".join(header) + "\n")
            f_o.write("\t".join(cols[2:]) + "\n")


def get_args():
    parser = argparse.ArgumentParser("Prepare transcribed data")
    parser.add_argument(
        "--root",
        help="data root path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--lang",
        required=True,
        type=str,
        choices=ASR_LANGUAGES + ASR_ACCENTED_LANGUAGES,
    )
    return parser.parse_args()


def main():
    args = get_args()
    get(args)


if __name__ == "__main__":
    main()
