# Predict each frame to coordinates and save to disk for further usages
from pred.hands_keypoint_pred import HandsKeypointPredict
import pickle
import shutil
from pathlib import Path
import cv2
import numpy as np
from hgdataset.s0_label import HgdLabel
from constants.enum_keys import HG


class HgdSkeleton(HgdLabel):
    """Load coords from disk if exists, else predict coords."""
    def __init__(self, mode, data_path: Path, is_train: bool, resize_img_size: tuple):
        super().__init__(mode, data_path, is_train)
        self.resize_img_size = resize_img_size
        if is_train:
            self.coord_folder = Path('generated/coords/' + str(self.mode) +'/train')
            self.video_folder = data_path / Path(str(self.mode) + '/train')
        else:
            self.coord_folder = Path('generated/coords/' + str(self.mode) +'/test')
            self.video_folder = data_path / Path(str(self.mode) + '/test')
        self.coord_folder.mkdir(parents=True, exist_ok=True)
        self.predictor = None

    def __getitem__(self, index):
        res_dict = super().__getitem__(index)
        v_name = res_dict[HG.VIDEO_NAME]
        coord_dict = self.vpath_to_coords(v_name)
        res_dict.update(coord_dict)
        return res_dict

    def vpath_to_coords(self, video_name: str):
        coord_dict = self.load_coords(video_name)
        if coord_dict is None:
            coord_dict = self.predict_from_video(video_name)
            self.save_coords(video_name, coord_dict)
        return coord_dict

    def save_coords(self, video_name, coords):
        pkl_path = self.coord_folder / video_name
        pkl_path = pkl_path.with_suffix('.pkl')
        with pkl_path.open('wb') as pickle_file:
            pickle.dump(coords, pickle_file)

    def load_coords(self, video_name):
        pkl_path = self.coord_folder / video_name
        pkl_path = pkl_path.with_suffix('.pkl')
        if not pkl_path.exists():
            return None
        with pkl_path.open('rb') as pickle_file:
            coords = pickle.load(pickle_file)
        return coords

    def predict_from_video(self, video_name):
        if self.predictor is None:
            self.predictor = HandsKeypointPredict()
        v_path = self.video_folder / video_name
        v_reader = self.video_reader(str(v_path))
        coords_list = []  # shape: (num_frame, xyz(3), num_keypoints)
        for i, frame in enumerate(v_reader):
            coords = self.predictor.get_coordinates(frame)
            coords_list.append(coords)
            print('Predicting %s: %d' % (video_name, i))
        coords_list = np.asarray(coords_list)
        return {HG.COORD: coords_list}

    def video_reader(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError("%s can't be opened" % video_path)
        v_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v_fps = int(cap.get(cv2.CAP_PROP_FPS))
        if v_fps != 15:
            raise ValueError("video %s must have a frame rate of 15, current %d" %(video_path, v_fps))

        for _ in range(v_size):
            ret, img = cap.read()
            re_img = cv2.resize(img, self.resize_img_size)
            yield re_img

        cap.release()
        print("video %s prediction finished" % video_path)

    @staticmethod
    def remove_generated_skeletons():
        p = Path('generated/coords')
        shutil.rmtree(p)

