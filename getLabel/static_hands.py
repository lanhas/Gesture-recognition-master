import sys
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import QApplication,QMainWindow
from PyQt5.QtWidgets import QFileDialog
from getLabel.static_ui import Ui_MainWindow as StaticUi

class MainWindowLabel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.hands_flag = 0
        self.show_delay = 0
        self.ishands_flag = False

    def openVideo(self):
        labels = []
        try:
            fileName, filetype = QFileDialog.getOpenFileName(self, "打开视频", str(Path.home() / 'MeetingHands' / 'static'), "All Files (*);;video File(*.mp4)")
            video_name = Path(fileName).name
            txt_name = Path(video_name).with_suffix('.csv')
            csv_path = Path(fileName).parent / txt_name
            cap = cv2.VideoCapture(fileName)
        except Exception:
            return True

        while cap.isOpened():
            ret, frame = cap.read()
            # if self.ishands_flag and self.show_delay > 20:
            #     self.hands_flag = 0
            #     self.ishands_flag = False
            if ret: 
                cv2.imshow("frame", frame)
                labels.append(self.hands_flag)
                self.show_delay += 1
            else:
                print("视频播放完成！")
                break

            key = cv2.waitKey(0)
            if key == 32:
                key = cv2.waitKey(50)
                continue
            if key == 27:
                break

        cap.release()
        self.hands_flag = 0
        self.show_delay = 0
        self.ishands_flag = False
        cv2.destroyAllWindows()
        labels = np.asarray(labels)
        labels = labels[np.newaxis].T
        np.savetxt(str(csv_path), labels.T, fmt='%d', delimiter=',')
        
    def set_up(self):
        self.hands_flag = 1
        self.ishands_flag = True
        self.show_delay = 0

    def set_down(self):
        self.hands_flag = 2
        self.ishands_flag = True
        self.show_delay = 0

    def set_left(self):
        self.hands_flag = 3
        self.ishands_flag = True
        self.show_delay = 0

    def set_right(self):
        self.hands_flag = 4
        self.ishands_flag = True
        self.show_delay = 0

    def set_start(self):
        self.hands_flag = 5
        self.ishands_flag = True
        self.show_delay = 0
    
    def set_end(self):
        self.hands_flag = 6
        self.ishands_flag = True
        self.show_delay = 0
    
    def set_pause(self):
        self.hands_flag = 7
        self.ishands_flag = True
        self.show_delay = 0

    def set_back(self):
        self.hands_flag = 8
        self.ishands_flag = True
        self.show_delay = 0

    def set_action1(self):
        self.hands_flag = 9
        self.ishands_flag = True
        self.show_delay = 0

    def set_action2(self):
        self.hands_flag = 10
        self.ishands_flag = False
        self.show_delay = 0

    def set_action3(self):
        self.hands_flag = 11
        self.ishands_flag = True
        self.show_delay = 0
    
    def set_nosignal(self):
        self.hands_flag = 0
        self.ishands_flag = False
        self.show_delay = 0



if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = MainWindowLabel()
    ui = StaticUi()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())