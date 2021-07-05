import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables import KeypointsOnImage, BoundingBoxesOnImage


class ResizeKeepRatio:
    def __init__(self, target_size: tuple):
        self.target_size = target_size
        assert len(self.target_size) == 2, 'Expect tuple (w, h) for target size'

    def resize(self, img, pts: np.ndarray, boxes: np.ndarray):
        pts_shape = pts.shape
        pts = pts.reshape((-1, 2))
        boxes_shape = boxes.shape
        boxes = boxes.reshape((-1, 4))

        tw, th = self.target_size
        ih, iw, ic = img.shape
        kps_on_image = KeypointsOnImage.from_xy_array(pts, shape=img.shape)
        boxes_on_img = BoundingBoxesOnImage.from_xyxy_array(boxes, shape=img.shape)

        seq = self.__aug_sequence((iw, ih), (tw, th))
        det = seq.to_deterministic()
        img_aug = det.augment_image(img)
        kps_aug = det.augment_keypoints(kps_on_image)
        boxes_aug = det.augment_bounding_boxes(boxes_on_img)

        np_kps_aug = kps_aug.to_xy_array()
        np_kps_aug = np_kps_aug.reshape(pts_shape)
        np_boxes_aug = boxes_aug.to_xy_array()
        np_boxes_aug = np_boxes_aug.reshape(boxes_shape)
        return img_aug, np_kps_aug, np_boxes_aug

    def __aug_sequence(self, input_size, target_size):
        iw, ih = input_size
        i_ratio = iw / ih
        tw, th = target_size
        t_ratio = tw / th
        if i_ratio > t_ratio:
            # Input image wider than target, resize width to target width
            resize_aug = iaa.Resize({"width": tw, "height": "keep-aspect-ratio"})
        else:
            # Input image higher than target, resize height to target height
            resize_aug = iaa.Resize({"width": "keep-aspect-ratio", "height": th})
        pad_aug = iaa.PadToFixedSize(width=tw, height=th, position="center")
        seq = iaa.Sequential([resize_aug, pad_aug])
        return seq
