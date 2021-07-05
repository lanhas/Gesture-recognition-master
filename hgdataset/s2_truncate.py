from pathlib import Path
from constants.enum_keys import HG
from numpy import random
import numpy as np
from hgdataset.s1_skeleton import HgdSkeleton


class HgdTruncate(HgdSkeleton):
    def __init__(self, mode,  data_path: Path, is_train: bool, resize_img_size: tuple, clip_len: int):
        super().__init__(mode, data_path, is_train, resize_img_size)
        self.clip_len = clip_len
        self.LABEL_DELAY = 0 # LABEL_DELAY frames are delayed to leave some time for RNN to observe the gesture

    def __getitem__(self, index):
        res_dict = super().__getitem__(index)
        if self.clip_len == -1:    # -1: use full video
            return res_dict
        v_len = len(res_dict[HG.GESTURE_LABEL])
        if v_len <= self.clip_len:
            raise ValueError("Video %s too short (%d) for clip_len %d" %
                            (res_dict[HG.VIDEO_PATH], v_len, self.clip_len))
        start = random.randint(v_len - self.clip_len)
        truncate = slice(start, start + self.clip_len)
        res_dict[HG.COORD] = res_dict[HG.COORD][truncate]
        res_dict[HG.GESTURE_LABEL] = res_dict[HG.GESTURE_LABEL][truncate]
        res_dict[HG.GESTURE_LABEL] = np.concatenate((np.zeros(self.LABEL_DELAY, dtype=np.int), res_dict[HG.GESTURE_LABEL]), axis=0)[: self.clip_len]
        return res_dict

