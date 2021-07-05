import cv2
import numpy as np
from constants.enum_keys import HG
from warnings import warn
from hgdataset.s1_skeleton import HgdSkeleton
from imgaug import KeypointsOnImage
from imgaug.imgaug import draw_text
from utils.resize import ResizeKeepRatio
from PyQt5.QtCore import QObject,pyqtSignal
import pred.hands_pred
from pathlib import Path


class Player(QObject):
    hands_singal = pyqtSignal(int)

    def __init__(self, is_unittest=False):
        super(QObject, self).__init__()
        self.img_size = (512, 512)
        self.hpred = pred.hands_pred.HandsPred("dynamic")
        self.is_unittest = is_unittest
        self.show_camera = True

    def play_dataset_video(self, is_train, video_index, show=True):
        self.scd = HgdSkeleton('dynamic' , Path.home() / 'MeetingHands', is_train, self.img_size)
        res = self.scd[video_index]
        print('Playing %s' % res[HG.VIDEO_NAME])
        coord_norm_FXJ = res[HG.COORD]  # Shape: F,X,J
        coord_norm_FXJ = coord_norm_FXJ[:, :, :]
        coord_norm_FXJ2 = coord_norm_FXJ[:, :2, :]
        coord_norm_FJX2 = np.transpose(coord_norm_FXJ2, (0, 2, 1))  # FJX
        coord = coord_norm_FJX2 * np.array(self.img_size)
        img_shape = self.img_size[::-1] + (3,)
        kps = [KeypointsOnImage.from_xy_array(coord_JX, shape=img_shape) for coord_JX in coord]  # (frames, KeyOnImage)

        cap = cv2.VideoCapture(str(res[HG.VIDEO_PATH]))
        v_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v_fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = int(1000/(v_fps*3))
        hands = []
        for n in range(v_size):
            hdict = self.hpred.from_skeleton(coord_norm_FXJ[n][np.newaxis])
            hand = hdict[HG.OUT_ARGMAX]
            hands.append(hand)
            if not show:
                continue
            ret, img = cap.read()
            re_img = cv2.resize(img, self.img_size)
            hands_name = self.hands_dict[hand]
            re_img = draw_text(re_img, 50, 100, hands_name, (255, 50, 50), size=40)
            pOnImg = kps[n]
            img_kps = pOnImg.draw_on_image(re_img)
            if self.is_unittest:
                break
            cv2.imshow("Play saved keypoint results", img_kps)
            cv2.waitKey(duration)
        cap.release()
        hands = np.array(hands, np.int)
        res[HG.PRED_GESTURE] = hands
        print('The prediction of video ', res[HG.VIDEO_NAME], ' is completed')
        return res

    def play_custom_video(self, video_path):
        # 多线程
        hands_now = 0
        hands_continue = 0
        send_flag = False
        rkr = ResizeKeepRatio((512, 512))
        if video_path is None:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise IOError("Failed to open camera.")
        else:
            cap = cv2.VideoCapture(str(video_path))
            v_fps = int(cap.get(cv2.CAP_PROP_FPS))
            if v_fps != 15:
                warn('Suggested video frame rate is 15, currently %d, which may impact accuracy' % v_fps)
        duration = 10
        while True:
            ret, img = cap.read()
            if not ret:
                break
            re_img, _, _ = rkr.resize(img, np.zeros((2,)), np.zeros((4,)))
            hdict = self.hpred.from_img(re_img)
            hands = hdict[HG.OUT_ARGMAX]
            # 手势控制信号部分
            if hands == hands_now:
                hands_continue += 1
            else:
                hands_continue = 0
                hands_now = hands
                send_flag = False
            
            if hands_continue > 2 and send_flag==False:
                self.hands_singal.emit(hands_now)
                send_flag = True
            
            coord_norm_FXJ = hdict[HG.COORD]
            coord_norm_FXJ = coord_norm_FXJ[:, :2, :]
            coord_norm_FJX = np.transpose(coord_norm_FXJ, (0, 2, 1))  # FJX
            coord_FJX = coord_norm_FJX * np.array(self.img_size)
            koi = KeypointsOnImage.from_xy_array(coord_FJX[0], shape=re_img.shape)
            re_img = koi.draw_on_image(re_img)
            hands_name = self.hands_dict[hands]
            
            re_img = draw_text(re_img, 50, 100, hands_name, (255, 50, 50), size=40)
            if self.is_unittest:
                break
            if self.show_camera == True:
                cv2.imshow("Play saved keyPoint results", re_img)
            # cv2.waitKey(duration)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

    hands_dict = {
        0: "NO SINGAL",
        1: "TORCH",
        2: "ROTATE",
        3: "MOVE Left",
        4: "MOVE Right",
        5: "ZOOM",
        6: "EXPENSION",
        7: "CAPTURE",
        8: "BACK",
        9: "SCREEN SHOT"}

    hands_dict_c = {
        0: "无信号",
        1: "点击",
        2: "旋转",
        3: "左移",
        4: "右移",
        5: "缩小",
        6: "放大",
        7: "抓取",
        8: "返回",
        9: "截屏"}
