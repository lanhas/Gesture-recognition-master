import torch
import numpy as np
from constants.enum_keys import HG
from models.hands_recognition_model import HandsRecognitionModel
from models.static_recognition_model import StaticRecognitionModel
from hgdataset.s3_handcraft import BoneLengthAngle
from pred.hands_keypoint_pred import HandsKeypointPredict


class HandsPred:
    def __init__(self, mode):
        self.mode = mode
        self.p_predictor = HandsKeypointPredict()
        self.bla = BoneLengthAngle()
        if self.mode == "dynamic":
            self.h_model = HandsRecognitionModel(1)
            self.h, self.c = self.h_model.h0(), self.h_model.c0()
        else:
            self.h_model = StaticRecognitionModel()
        self.h_model.load_ckpt()
        self.h_model.eval()

    def from_skeleton(self, coord_norm):
        assert coord_norm.ndim == 3 and coord_norm.shape[0] == 1
        ges_data = self.bla.handcrafted_features(coord_norm)
        features = np.concatenate((ges_data[HG.BONE_LENGTH], ges_data[HG.BONE_ANGLE_COS],
                                    ges_data[HG.BONE_ANGLE_SIN], ges_data[HG.BONE_DEPTH]), axis=1)
        features = features[np.newaxis]
        features = features.transpose((1, 0, 2))
        features = torch.from_numpy(features)
        features = features.to(self.h_model.device, dtype=torch.float32)
        with torch.no_grad():
            if self.mode == "dynamic":
                _, h, c, class_out = self.h_model(features, self.h, self.c)
                self.h, self.c = h, c
            else:
                class_out = self.h_model(features)
        np_out = class_out[0].cpu().numpy()
        max_arg = np.argmax(np_out)
        res_dict = {HG.OUT_ARGMAX: max_arg, HG.OUT_SCORES: np_out, HG.COORD: coord_norm}
        return res_dict

    def from_img(self, img: np.ndarray):
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8 and img.ndim == 3, "Expect ndarray of shape (H, W, C)"
        p_res = self.p_predictor.get_coordinates(img)
        res_dict = self.from_skeleton(p_res[np.newaxis])
        return res_dict

