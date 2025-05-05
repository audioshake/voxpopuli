# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import concurrent.futures
from tqdm import tqdm


def multiprocess_run(
        a_list: list, func: callable, n_workers: Optional[int] = None
):
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(executor.map(func, a_list), total=len(a_list)))
    return results
