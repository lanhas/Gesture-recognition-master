import os
import torch
import cv2 as cv
import numpy as np
from blazepalm.blazepalm import PalmDetector
from blazepalm.handlandmarks import HandLandmarks


m = PalmDetector()
m.load_weights("./checkpoints/palmdetector.pth")
m.load_anchors("./checkpoints/anchors.npy")
hl = HandLandmarks()
hl.load_weights("./checkpoints/handlandmarks.pth")
cap = cv.VideoCapture(0)
while True:
	ret, frame = cap.read()
	hh, ww, _ = frame.shape
	ll = min(hh, ww)
	img = cv.resize(frame[:ll, :ll][:, ::-1], (256, 256))
	predictions = m.predict_on_image(img)
	for pred in predictions:
		for pp in pred:
			p = pp*ll
			x = int(max(0, p[0]))
			y = int(max(0, p[1]))
			endx = int(min(ll, p[2]))
			endy = int(min(ll, p[3]))
			cropped_hand = frame[y:endy, x:endx]
			maxl = max(cropped_hand.shape[0], cropped_hand.shape[1])
			cropped_hand = np.pad(cropped_hand,
				( ((maxl-cropped_hand.shape[0])//2, (maxl-cropped_hand.shape[0]+1)//2), ((maxl-cropped_hand.shape[1])//2, (maxl-cropped_hand.shape[1]+1)//2), (0, 0) ),
				'constant')
			cropped_hand = cv.resize(cropped_hand, (256, 256))
			cropped_hand = torch.from_numpy(np.asarray(cropped_hand).astype(np.float32)).permute((2, 0, 1)).unsqueeze(0)
			# print(cropped_hand)
			landmarks = hl(cropped_hand)
			print(landmarks)
			cv.rectangle(img, (p[0], p[1]), (p[2], p[3]), (255, 0, 0), 2)
			cv.imshow('1', img)
			# cv.waitKey()
			break

	cv.imshow('frame', frame)
	if cv.waitKey(1) == ord('q'):
		break

cap.release()
cv.destroyAllWindows()
