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
    out_root = Path(args.root) / "rttm"
    out_root.mkdir(exist_ok=True, parents=True)

    for lang in ASR_LANGUAGES + ASR_ACCENTED_LANGUAGES:

        # Get metadata TSV
        url = f"{DOWNLOAD_BASE_URL}/annotations/asr/asr_{lang}.tsv.gz"
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
                r["speaker_id"],
                split,
                )
            )
        items = list(items.items())
        # Segment
        # multiprocess_run(items, cut_session)
        # Output per-split manifest in RTTM format
        for cols in manifest:
            convo_id = cols[0]
            year = cols[1]
            start_time = float(cols[2])
            end_time = float(cols[3])
            speaker_id = cols[4]
            split = cols[5]
            
            duration = end_time - start_time
            
            (out_root / year).mkdir(exist_ok=True, parents=True)
            
            # Write in RTTM format
            # Format: SPEAKER file_id channel_id start_time duration <NA> <NA> speaker_id <NA> <NA>
            with open(out_root / year / f"{convo_id}_original.rttm", "a", encoding="utf-8") as f_o:
                rttm_line = f"SPEAKER {convo_id} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA> <NA>\n"
                f_o.write(rttm_line)

            # Write in RTTM format
            # Format: SPEAKER file_id channel_id start_time duration <NA> <NA> speaker_id <NA> <NA>
            with open(out_root / f"{split}.rttm", "a", encoding="utf-8") as f_o:
                rttm_line = f"SPEAKER {convo_id} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA> <NA>\n"
                f_o.write(rttm_line)


def get_args():
    parser = argparse.ArgumentParser("Prepare transcribed data")
    parser.add_argument(
        "--root",
        help="data root path",
        type=str,
        required=True,
    )
    return parser.parse_args()


def main():
    args = get_args()
    get(args)


if __name__ == "__main__":
    main()
