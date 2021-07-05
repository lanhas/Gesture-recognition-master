import numpy as np
from pathlib import Path
from hgdataset.s2_truncate import HgdTruncate
from constants.enum_keys import HG
from constants.keypoints import hand_bones, hand_bone_pairs


class HgdHandcraft(HgdTruncate):
    """Return handcrafted features: bone length and angle"""
    def __init__(self, mode, data_path: Path, is_train: bool, resize_img_size: tuple, clip_len: int):
        super().__init__(mode, data_path, is_train, resize_img_size, clip_len)
        self.bla = BoneLengthAngle()

    def __getitem__(self, index):
        res_dict = super().__getitem__(index)
        feature_dict = self.bla.handcrafted_features(res_dict[HG.COORD])
        res_dict.update(feature_dict)
        return res_dict


class BoneLengthAngle:
    def __init__(self):
        self.connections = np.asarray(hand_bones, np.int)
        self.pairs = np.asarray(hand_bone_pairs, np.int)

    def handcrafted_features(self, coord_norm):
        assert len(coord_norm.shape) == 3
        feature_dict = {}
        bone_len = self.bone_len(coord_norm)
        bone_sin, bone_cos = self.bone_pair_angle(coord_norm)
        feature_dict[HG.BONE_LENGTH] = bone_len
        feature_dict[HG.BONE_ANGLE_SIN] = bone_sin
        feature_dict[HG.BONE_ANGLE_COS] = bone_cos
        feature_dict[HG.BONE_DEPTH] = coord_norm[:, 2, :]
        return feature_dict

    def bone_len(self, coord):
        xy_coord = np.asarray(coord)
        xy_val = np.take(xy_coord, self.connections, axis=2)
        xy_diff = xy_val[:, :, :, 0] - xy_val[:, :, :, 1]
        xy_diff = xy_diff ** 2
        bone_len = np.sqrt(xy_diff[:, 0] + xy_diff[:, 1])
        return bone_len

    def bone_pair_angle(self, coord):
        xy_coord = np.asarray(coord)
        xy_val = np.take(xy_coord, self.pairs, axis=2)
        xy_vec = xy_val[:, :, :, : ,1] - xy_val[:, :, :, :, 0]
        ax = xy_vec[:, 0, :, 0]
        bx = xy_vec[:, 0, :, 1]
        ay = xy_vec[:, 1, :, 0]
        by = xy_vec[:, 1, :, 1]
        dot_product = ax * bx + ay * by
        cross_product = ax * by - ay * bx
        magnitude = np.einsum('fxpb,fxpb->fpb', xy_vec, xy_vec)  # a^2+b^2
        magnitude = np.sqrt(magnitude)  # shape: (F,P,B)
        magnitude[magnitude < 10e-3] = 10e-3  # Filter zero value
        mag_AxB = magnitude[:, :, 0] * magnitude[:, :, 1]  # shape: (F,P)
        cos = dot_product / mag_AxB
        sin = cross_product / mag_AxB
        return sin, cos

