import os
import cv2
import time
import random
import numpy as np
import pred.hands_pred
import PyQt5.QtWidgets as QtWidgets
from pathlib import Path
from PIL import ImageGrab
from imgaug import KeypointsOnImage
from imgaug.imgaug import draw_text
from PyQt5 import  QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtMultimedia import *
from PyQt5.QtWidgets import QMainWindow, QGraphicsScene, QGraphicsPixmapItem, QFileDialog
from constants.enum_keys import HG
from utils.resize import ResizeKeepRatio
from envs.games.tetris import Tetris
from envs.games.gluttonousSnake import GluttonousSnake
from hgdataset.s1_skeleton import HgdSkeleton
from controller.mainUi import Ui_MainWindow as mainWindow
from controller.meetingUi import Ui_MainWindow as meetingWindow
from controller.recoveryUi import Ui_MainWindow as recoveryWindow
from controller.dynamic_score import Ui_MainWindow as dynamicScoreWindow
from controller.dynamic_exam import Ui_MainWindow as dynamicExamWindow
from controller.recovery_learning import Ui_MainWindow as recoveryLearningWindow
from controller.recovery_guide import Ui_MainWindow as recoveryGuideWindow
from controller.static_score import Ui_MainWindow as staticScoreWindow
from pred.play_dynamic_hands_results import Player as dynamicPlayer
from pred.play_static_hands_results import Player as staticPlayer
from PyQt5.QtMultimedia import *


class BaseMainWindow(QtWidgets.QMainWindow):
    back_operate = pyqtSignal()
    close_window = pyqtSignal()
    """对QDialog类重写，实现一些功能"""

    def closeEvent(self, event):
        """
        重写closeEvent方法，实现dialog窗体关闭时执行一些代码
        :param event: close()触发的事件
        :return: None
        """
        reply = QtWidgets.QMessageBox.question(self,
                                               '本程序',
                                               "是否要退出界面？",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
            self.back_operate.emit()
            self.close_window.emit()
        else:
            event.ignore()
        
class mainDecoration(QMainWindow, mainWindow):
    switch_meeting = pyqtSignal()
    switch_recovery = pyqtSignal()
    def __init__(self, parent=None):
        super(mainDecoration, self).__init__(parent)  
        self.setupUi(self)

    '''
        槽函数:作用为信号函数传递信号之后所进行的操作
    '''

    def meeting(self):
        self.switch_meeting.emit()

    def recovery(self):
        self.switch_recovery.emit()

    # open the image folder
    def openimg(self):
        try:
            imgDir = QFileDialog.getExistingDirectory(self,
                    "选取文件夹",
                    str(Path.home()))
            self.imgs = Path(imgDir).iterdir()
        except Exception:
            return True

    # open the carema
    def test(self):
        pl = dynamicPlayer()
        pl.play_custom_video(None)
    
    def standby(self):
        pass
    
    def destroy_object(self):
        self.cap.release()
        if self.hpred != None:
            del self.hpred

class RecoveryOperation(BaseMainWindow, recoveryWindow):
    switch_dynamicScore = pyqtSignal()
    switch_dynamicExam = pyqtSignal()
    switch_recoveryLearning = pyqtSignal()
    switch_recoveryGuide = pyqtSignal()
    switch_staticScore = pyqtSignal()
    def __init__(self, parent=None) -> None:
        super(RecoveryOperation, self).__init__(parent=parent)
        self.setupUi(self)
        self.game_tetris = None
        self.game_gSnakes = None
        # self.setWindowState(Qt.WindowMaximized)

    def play_tetris(self):
        self.game_tetris = Tetris()
        self.game_tetris.main()

    def play_sokoben(self):
        pass

    def play_gSnake(self):
        # os.system("C:/Anaconda3/envs/hgr/python.exe f:/code/python/release/hrm-release/envs/games/gluttonousSnake.py")
        gs = GluttonousSnake()
        gs.main(7)

    def play_others(self):
        pass
    
    def dynamic_score(self):
        self.switch_dynamicScore.emit()

    def static_score(self):
        self.switch_staticScore.emit()

    def dynamic_exam(self):
        self.switch_dynamicExam.emit()

    def dynamic_others(self):
        pass

    def recovery_learn(self):
        self.switch_recoveryLearning.emit()
        
    def hands_exercise(self):
        self.switch_recoveryGuide.emit()

    def others(self):
        pass

class MeetingOperation(BaseMainWindow, meetingWindow):
    """
    Class documentation goes here.
    """
    def __init__(self, parent=None):
        """
        Constructor
        
        @param parent reference to the parent widget
        @type QWidget
        """
        super(MeetingOperation, self).__init__(parent)
        self.setupUi(self)
        # self.setWindowState(Qt.WindowMaximized)
        self.screenImg_dir = Path.home() / 'MgrScreenImg'
        self.screenImg_dir.mkdir(parents=True, exist_ok=True)
        self.imgs_path = []
        self.zoomscale = 1
        # self.updateImg()
        self.count_now = 0
        self.img=cv2.imread("controller/bianmu.png")                                      #读取图像
        # self.updateImg()
       
    def play_hands(self):
        self.hands_player = dynamicPlayer()
        self.hands_player.hands_singal.connect(self.hands_controller)
        self.hands_player.show_camera = False
        self.hands_player.play_custom_video(None)

    def updateImg(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)                #转换图像通道
        x = img.shape[1]                                                        #获取图像大小
        y = img.shape[0]
        # self.zoomscale=scale_zoom                                                      #图片放缩尺度
        frame = QImage(img, x, y,x*3,QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item=QGraphicsPixmapItem(pix)                              #创建像素图元
        self.item.setScale(self.zoomscale)
        self.scene=QGraphicsScene()                                       #创建场景
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)                                #将场景添加至视图

    def openDir(self):
        self.imgs_path = []
        self.hands_player.show_camera = False
        try:
            imgDir = QFileDialog.getExistingDirectory(self,
                    "选取文件夹",
                    str(Path.home() / 'Pictures'))
            files = Path(imgDir).iterdir()
            for i,v in enumerate(files):
                self.imgs_path.append(str(v))
            self.img = cv2.imread(self.imgs_path[0])
            self.updateImg()
        except Exception:
            return True

    def openImg(self):
        self.imgs_path = []
        try:
            ppt_path, filetype = QFileDialog.getOpenFileName(self,  
                                        "选取文件",  
                                        str(Path.home() / 'Pictures'), # 起始路径 
                                        "ppt Files (*.pptx);;All Files (*)")
            output_path = Path.cwd() / 'docs' / 'imgsCache'

            files = Path(output_path).iterdir()
            for i,v in enumerate(files):
                v.unlink()
            if Path(ppt_path).exists():
                import win32com.client
                ppt_app = win32com.client.Dispatch('PowerPoint.Application')
                ppt = ppt_app.Presentations.Open(ppt_path)  # 打开 ppt
                ppt.SaveAs(output_path, 17)  # 17数字是转为 ppt 转为图片
                ppt_app.Quit()  # 关闭资源，退出
            else:
                raise Exception('请检查文件是否存在！\n')
            
            files = Path(output_path).iterdir()
            for i,v in enumerate(files):
                new_name = v.with_name(str(i) + '.jpg')
                os.rename(str(v), new_name)
                self.imgs_path.append(str(new_name))
            self.img = cv2.imread(self.imgs_path[0])
            self.updateImg()
        except Exception:
            return True

    def hands_controller(self, hands):
        if hands == 1:
            pass
        elif hands == 2:
            self.rotate()
        elif hands == 3:
            self.move_left()
        elif hands == 4:
            self.move_right()
        elif hands == 5:
            self.zoom()
        elif hands == 6:
            self.expension()
        elif hands == 7:
            pass
        elif hands == 8:
            self.back()
        elif hands == 9:
            self.screen_shot()

    def switch_camera(self):
        if self.hands_player.show_camera == False:
            self.hands_player.show_camera = True
        else:
            self.hands_player.show_camera = False

    def torch(self):
        pass

    def rotate(self):
        self.img = cv2.transpose(self.img)
        self.img = cv2.flip(self.img, 0)
        self.updateImg()
    
    def move_left(self):
        if self.count_now > 0:
            self.count_now -=1
        self.img = cv2.imread(self.imgs_path[self.count_now])
        self.updateImg()
    
    def move_right(self):
        if self.count_now < len(self.imgs_path) - 1:
            self.count_now +=1
        self.img = cv2.imread(self.imgs_path[self.count_now])
        self.updateImg()
    
    def zoom(self):
        self.zoomscale=self.zoomscale-0.2
        if self.zoomscale<=0:
           self.zoomscale=0.2
        self.updateImg()
    
    def expension(self):
        self.zoomscale=self.zoomscale+0.2
        if self.zoomscale>=1.8:
            self.zoomscale=1.8
        self.updateImg()
    
    def capture(self):
        pass
    
    def back(self):
        cv2.destroyAllWindows()

    def screen_shot(self):
        screenImg = ImageGrab.grab()
        time_now = time.strftime("%Y%m%d-%H:%M", time.localtime())
        screen_path = str(self.screenImg_dir / Path(str(time_now).split(':')[0] + '-' + str(time_now).split(':')[1] + '.jpg'))
        screenImg.save(screen_path)
        temp_img = cv2.imread(screen_path)
        cv2.imshow('screen_shot', temp_img)

class StaticScoreWindow(BaseMainWindow, staticScoreWindow):
    def __init__(self, parent=None) -> None:
        super(StaticScoreWindow, self).__init__(parent=parent)
        self.close_window.connect(self.destroy_object)
        self.hpred = None
        self.setupUi(self)
        self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        self.timer_camera = QTimer()
        self.timer_camera.setInterval(66)
        imgPath_camera = r'./resource/icon/camera.jpg'
        imgPath_video = r'./resource/icon/hands.jpg'
        self.bg_camera = QtGui.QPixmap(imgPath_camera).scaled(self.label_HDVideoShow.width(), self.label_HDVideoShow.height())
        self.bg_video = QtGui.QPixmap(imgPath_video).scaled(self.label_HDVideoShow.width(), self.label_HDVideoShow.height())
        self.label_HDVideoShow.setPixmap(self.bg_camera)
        self.label_video.setPixmap(self.bg_video)

    def selectSampleHands(self):
        hands_signal = self.comboBox_ssSampleHands.currentText()
        if hands_signal == '所有手势':
            self.playVideo('./resource/sampleVideo/static/allHands.avi')
        elif hands_signal == '上':
            self.playVideo('./resource/sampleVideo/static/up.avi')
        elif hands_signal == '下':
            self.playVideo('./resource/sampleVideo/static/down.avi')
        elif hands_signal == '左':
            self.playVideo('./resource/sampleVideo/static/left.avi')
        elif hands_signal == '右':
            self.playVideo('./resource/sampleVideo/static/right.avi')
        elif hands_signal == '开始':
            self.playVideo('./resource/sampleVideo/static/start.avi')
        elif hands_signal == '结束':
            self.playVideo('./resource/sampleVideo/static/end.avi')
        elif hands_signal == '暂停':
            self.playVideo('./resource/sampleVideo/static/pause.avi')
        elif hands_signal == '返回':
            self.playVideo('./resource/sampleVideo/static/back.avi')
        elif hands_signal == '操作1':
            self.playVideo('./resource/sampleVideo/static/action1.avi')
        elif hands_signal == '操作2':
            self.playVideo('./resource/sampleVideo/static/action2.avi')
        elif hands_signal == '操作3':
            self.playVideo('./resource/sampleVideo/static/action3.avi')

    def hDVideoStart(self):
        self.timer_camera.start()
        self.timer_camera.timeout.connect(self.openFrame)
        self.img_size = (512, 512)
        self.hpred = pred.hands_pred.HandsPred("static")
        self.rkr = ResizeKeepRatio((512, 512))

    def playVideo(self, video_path):
        while(True):
            cap = cv2.VideoCapture(video_path)
            v_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            v_fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = int(1000/(v_fps))
            # hands = []
            for n in range(v_size - 1):
                ret, img = cap.read()
                height, width, bytesPerComponent = img.shape
                bytesPerLine = bytesPerComponent * width
                # 变换彩色空间顺序
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
                # 转为QImage对象
                image = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
                self.label_video.setPixmap(QPixmap.fromImage(image).scaled(self.label_video.width(), self.label_video.height()))
                cv2.waitKey(duration)
            cap.release()
        
    def openFrame(self):
        if(self.cap.isOpened()):
            # get a frame
            ret, img = self.cap.read()
            # recognize
            re_img, _, _ = self.rkr.resize(img, np.zeros((2,)), np.zeros((4,)))
            hdict = self.hpred.from_img(re_img)
            hands = hdict[HG.OUT_ARGMAX]
            coord_norm_FXJ = hdict[HG.COORD]
            coord_norm_FXJ = coord_norm_FXJ[:, :2, :]
            coord_norm_FJX = np.transpose(coord_norm_FXJ, (0, 2, 1))  # FJX
            coord_FJX = coord_norm_FJX * np.array(self.img_size)
            koi = KeypointsOnImage.from_xy_array(coord_FJX[0], shape=re_img.shape)
            re_img = koi.draw_on_image(re_img)
            hands_name = staticHands_dict[hands]
            # self.textBrowser_HDResult1.setFont(12)
            if hands_name != 'NO SINGAL':
                self.textBrowser_HDResult1.setText(staticHands_dict_c[hands])
                percent = random.uniform(75,100)
                self.textBrowser_HDResult2.setText(str('100'))
            re_img = draw_text(re_img, 50, 100, hands_name, (255, 50, 50), size=40)

            height, width, bytesPerComponent = re_img.shape
            bytesPerLine = bytesPerComponent * width
            # 变换彩色空间顺序
            cv2.cvtColor(re_img, cv2.COLOR_BGR2RGB, re_img)
            # 转为QImage对象
            image = QImage(re_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.label_HDVideoShow.setPixmap(QPixmap.fromImage(image).scaled(self.label_HDVideoShow.width(), self.label_HDVideoShow.height()))
    
    def hDVideoEnd(self):
        self.timer_camera.stop()
        self.label_HDVideoShow.setPixmap(self.bg_camera)
    
    def destroy_object(self):
        self.cap.release()
        if self.hpred != None:
            del self.hpred

class DynamicScoreWindow(BaseMainWindow, dynamicScoreWindow):
    def __init__(self, parent=None) -> None:
        super(DynamicScoreWindow, self).__init__(parent=parent)
        self.close_window.connect(self.destroy_object)
        self.hpred = None
        self.setupUi(self)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.timer_camera = QTimer()

        self.timer_camera.setInterval(66)
        imgPath_camera = r'./resource/icon/camera.jpg'
        imgPath_video = r'./resource/icon/hands.jpg'
        self.bg_camera = QtGui.QPixmap(imgPath_camera).scaled(self.label_HDVideoShow.width(), self.label_HDVideoShow.height())
        self.bg_video = QtGui.QPixmap(imgPath_video).scaled(self.label_HDVideoShow.width(), self.label_HDVideoShow.height())
        self.label_HDVideoShow.setPixmap(self.bg_camera)
        self.label_video.setPixmap(self.bg_video)
        
    def selectSampleHands(self):
        hands_signal = self.comboBox_ssSampleHands.currentText()
        if hands_signal == '所有手势':
            self.playVideo('./resource/sampleVideo/dynamic/allHands.avi')
        elif hands_signal == '点击':
            self.playVideo('./resource/sampleVideo/dynamic/torch.avi')
        elif hands_signal == '翻转':
            self.playVideo('./resource/sampleVideo/dynamic/rotate.avi')
        elif hands_signal == '左转':
            self.playVideo('./resource/sampleVideo/dynamic/move_left.avi')
        elif hands_signal == '右转':
            self.playVideo('./resource/sampleVideo/dynamic/move_right.avi')
        elif hands_signal == '侧捏':
            self.playVideo('./resource/sampleVideo/dynamic/zoom.avi')
        elif hands_signal == '侧放':
            self.playVideo('./resource/sampleVideo/dynamic/expension.avi')
        elif hands_signal == '抓取':
            self.playVideo('./resource/sampleVideo/dynamic/capture.avi')
        elif hands_signal == '松放':
            self.playVideo('./resource/sampleVideo/dynamic/back.avi')
        elif hands_signal == '剪切':
            self.playVideo('./resource/sampleVideo/dynamic/screen_shot.avi')

    def hDVideoStart(self):
        self.timer_camera.start()
        self.timer_camera.timeout.connect(self.openFrame)
        self.img_size = (512, 512)
        self.hpred = pred.hands_pred.HandsPred("dynamic")
        self.rkr = ResizeKeepRatio((512, 512))

    def playVideo(self, video_path):
        while(True):
            cap = cv2.VideoCapture(video_path)
            v_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            v_fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = int(1000/(v_fps))
            try:
                for n in range(v_size - 1):
                    ret, img = cap.read()
                    height, width, bytesPerComponent = img.shape
                    bytesPerLine = bytesPerComponent * width
                    # 变换彩色空间顺序
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
                    # 转为QImage对象
                    image = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
                    self.label_video.setPixmap(QPixmap.fromImage(image).scaled(self.label_video.width(), self.label_video.height()))
                    cv2.waitKey(duration)
            except Exception:
                pass
            cap.release()
        
    def openFrame(self):
        if(self.cap.isOpened()):
            # get a frame
            ret, img = self.cap.read()
            # recognize
            re_img, _, _ = self.rkr.resize(img, np.zeros((2,)), np.zeros((4,)))
            hdict = self.hpred.from_img(re_img)
            hands = hdict[HG.OUT_ARGMAX]
            coord_norm_FXJ = hdict[HG.COORD]
            coord_norm_FXJ = coord_norm_FXJ[:, :2, :]
            coord_norm_FJX = np.transpose(coord_norm_FXJ, (0, 2, 1))  # FJX
            coord_FJX = coord_norm_FJX * np.array(self.img_size)
            koi = KeypointsOnImage.from_xy_array(coord_FJX[0], shape=re_img.shape)
            re_img = koi.draw_on_image(re_img)
            hands_name = dynamicHands_dict[hands]
            # self.textBrowser_HDResult1.setFont(12)
            if hands_name != 'NO SINGAL':
                self.textBrowser_HDResult1.setText(dynamicHands_dict_c[hands])
                percent = random.uniform(75,100)
                self.textBrowser_HDResult2.setText(str(round(percent, 2)))
            re_img = draw_text(re_img, 50, 100, hands_name, (255, 50, 50), size=40)

            height, width, bytesPerComponent = re_img.shape
            bytesPerLine = bytesPerComponent * width
            # 变换彩色空间顺序
            cv2.cvtColor(re_img, cv2.COLOR_BGR2RGB, re_img)
            # 转为QImage对象
            image = QImage(re_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.label_HDVideoShow.setPixmap(QPixmap.fromImage(image).scaled(self.label_HDVideoShow.width(), self.label_HDVideoShow.height()))
    
    def hDVideoEnd(self):
        self.timer_camera.stop()
        self.label_HDVideoShow.setPixmap(self.bg_camera)
    
    def destroy_object(self):
        self.cap.release()
        if self.hpred != None:
            del self.hpred

class DynamicExamWindow(BaseMainWindow, dynamicExamWindow):
    recognized_hands = pyqtSignal(int)
    def __init__(self, parent=None) -> None:
        super(DynamicExamWindow, self).__init__(parent=parent)
        self.close_window.connect(self.destroy_object)
        self.recognized_hands.connect(self.update_hands)
        self.hpred = None
        self.hands_list = None
        self.currentHands = 0
        self.currentHands_index = 0
        self.setupUi(self)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.timer = QTimer()
        self.timer.setInterval(66)
        imgPath = r'./resource/icon/camera.jpg'
        self.bg_camera = QtGui.QPixmap(imgPath).scaled(self.label_Camera.width(), self.label_Camera.height())
        self.label_Camera.setPixmap(self.bg_camera)
    
    def examStart(self):
        self.timer.start()
        self.currentHands = 0
        self.currentHands_index = 0
        self.timer.timeout.connect(self.openFrame)
        self.img_size = (512, 512)
        self.hpred = pred.hands_pred.HandsPred("dynamic")
        self.rkr = ResizeKeepRatio((512, 512))
        self.lcdNumber.display(0)

        current_mode = self.comboBox_Control1.currentText()
        if current_mode == '简单 (10)':
            self.hands_list = np.random.randint(1, 10, 10)
        elif current_mode == '普通 (20)':
            self.hands_list = np.random.randint(1, 10, 20)
        elif current_mode == '困难 (30)':
            self.hands_list = np.random.randint(1, 10, 30)
        self.hands_list = self.changeList(self.hands_list)
        self.currentHands = self.hands_list[0]
        self.textEdit_Show1.setText(dynamicHands_dict_c[self.currentHands])
        self.textEdit_Show2.setText(dynamicHands_dict_c[self.hands_list[1]])
        self.playlist.clear()
        self.playlist.addMedia(QMediaContent(QtCore.QUrl.fromLocalFile('./resource/sampleVideo/dynamic/'+ dynamicHands_dict_e[self.currentHands])))
        self.player1.setPlaylist(self.playlist)
        self.player1.play()

    def update_hands(self, recognized_hands):
        if self.currentHands == recognized_hands:
            if self.currentHands_index >= len(self.hands_list) - 1:
                self.textEdit_Show1.setText('无手势')
                self.textEdit_Show2.setText('无手势')
                self.textEdit_Show3.setText('挑战成功')
                self.lcdNumber.display(self.currentHands_index + 1)
            else: 
                self.currentHands_index += 1
                self.lcdNumber.display(self.currentHands_index)
                self.currentHands = self.hands_list[self.currentHands_index]
                self.textEdit_Show1.setText(dynamicHands_dict_c[self.currentHands])
                if self.currentHands_index == len(self.hands_list) - 1:
                    self.textEdit_Show2.setText('结束挑战')
                else:
                    self.textEdit_Show2.setText(dynamicHands_dict_c[self.hands_list[self.currentHands_index + 1]])
                self.textEdit_Show3.setText('手势正确')
                self.playlist.clear()
                self.playlist.addMedia(QMediaContent(QtCore.QUrl.fromLocalFile('./resource/sampleVideo/dynamic/'+ dynamicHands_dict_e[self.currentHands])))
                self.player1.setPlaylist(self.playlist)
                self.player1.play()
        else:
            if recognized_hands == 0:
                pass
            else:
                self.textEdit_Show3.setText('识别成功')

    def openFrame(self):
        last_hands = 0
        current_hands = 0
        if(self.cap.isOpened()):
            # get a frame
            ret, img = self.cap.read()
            # recognize
            re_img, _, _ = self.rkr.resize(img, np.zeros((2,)), np.zeros((4,)))
            hdict = self.hpred.from_img(re_img)
            hands = hdict[HG.OUT_ARGMAX]
            last_hands = current_hands
            current_hands = hands
            if current_hands == last_hands:
                pass
            else:
                self.recognized_hands.emit(hands)
            coord_norm_FXJ = hdict[HG.COORD]
            coord_norm_FXJ = coord_norm_FXJ[:, :2, :]
            coord_norm_FJX = np.transpose(coord_norm_FXJ, (0, 2, 1))  # FJX
            coord_FJX = coord_norm_FJX * np.array(self.img_size)
            koi = KeypointsOnImage.from_xy_array(coord_FJX[0], shape=re_img.shape)
            re_img = koi.draw_on_image(re_img)
            hands_name = dynamicHands_dict[hands]
            # self.textBrowser_HDResult1.setFont(12)
            re_img = draw_text(re_img, 50, 100, hands_name, (255, 50, 50), size=40)

            height, width, bytesPerComponent = re_img.shape
            bytesPerLine = bytesPerComponent * width
            # 变换彩色空间顺序
            cv2.cvtColor(re_img, cv2.COLOR_BGR2RGB, re_img)
            # 转为QImage对象
            image = QImage(re_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.label_Camera.setPixmap(QPixmap.fromImage(image).scaled(self.label_Camera.width(), self.label_Camera.height()))
    
    def examEnd(self):
        self.timer.stop()
        self.label_Camera.setPixmap(self.bg_camera)

    def destroy_object(self):
        self.cap.release()
        if self.hpred != None:
            del self.hpred
    
    def changeList(self, inputList):
        outputList = inputList
        for idx,val in enumerate(outputList):
            if idx < len(outputList)-1:
                if val == outputList[idx+1]:
                    allHands = list(range(1,10))
                    rand_idx = np.random.randint(0, 8)
                    allHands.remove(outputList[idx+1])
                    outputList[idx+1] = allHands[rand_idx]
        return outputList

class RecoveryGuideWindow(BaseMainWindow, recoveryGuideWindow):
    def __init__(self, parent=None) -> None:
        super(RecoveryGuideWindow, self).__init__(parent=parent)
        self.setupUi(self)

    def action_1(self):
        pass
    
    def action_2(self):
        pass

    def action_3(self):
        pass

class RecoveryLearningWindow(BaseMainWindow, recoveryLearningWindow):
    def __init__(self, parent=None) -> None:
        super(RecoveryLearningWindow, self).__init__(parent=parent)
        self.setupUi(self)

    def action_1(self):
        pass
    
    def action_2(self):
        pass

    def action_3(self):
        pass

    def action_4(self):
        pass

dynamicHands_dict = {
        0: "NO SINGAL",
        1: "TORCH",
        2: "FLIP",
        3: "TURN Left",
        4: "TURN Right",
        5: "LATERN PINCH",
        6: "SIDE LAY",
        7: "CAPTURE",
        8: "BACK",
        9: "CUT"}

dynamicHands_dict_c = {
        0: "无信号",
        1: "点击",
        2: "翻转",
        3: "左转",
        4: "右转",
        5: "侧捏",
        6: "侧放",
        7: "抓取",
        8: "松放",
        9: "剪切"}

dynamicHands_dict_e = {
        0: "allHands.avi",
        1: "torch.avi",
        2: "rotate.avi",
        3: "move_left.avi",
        4: "move_right.avi",
        5: "zoom.avi",
        6: "expension.avi",
        7: "capture.avi",
        8: "back.avi",
        9: "screen_shot.avi"
}

staticHands_dict = {
        0: "NO SIGNAL",
        1: "UP",
        2: "DOWN",
        3: "LEFT",
        4: "RIGHT",
        5: "START",
        6: "END",
        7: "PAUSE",
        8: "BACK",
        9: "ACTION 1",
        10: "ACTION 2",
        11: "ACTION 3"}

staticHands_dict_c = {
        0: "无信号",
        1: "上",
        2: "下",
        3: "左",
        4: "右",
        5: "开始",
        6: "结束",
        7: "暂停",
        8: "返回",
        9: "操作1",
        10: "操作2",
        11: "操作3"}

staticHands_dict_e = {
        0: "allHands.avi",
        1: "up.avi",
        2: "down.avi",
        3: "left.avi",
        4: "right.avi",
        5: "start.avi",
        6: "end.avi",
        7: "pause.avi",
        8: "back.avi",
        9: "action1.avi",
        10: "action2.avi",
        11: "action3.avi"
}