# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'f:\code\python\hrm-pytorch\controller\dynamic_exam.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import QVideoWidget


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1300, 900)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.widget_Title = QtWidgets.QWidget(self.centralwidget)
        self.widget_Title.setMinimumSize(QtCore.QSize(0, 0))
        self.widget_Title.setMaximumSize(QtCore.QSize(16777215, 150))
        self.widget_Title.setObjectName("widget_Title")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.widget_Title)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem = QtWidgets.QSpacerItem(436, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem)
        self.label_Title = QtWidgets.QLabel(self.widget_Title)
        font = QtGui.QFont()
        font.setPointSize(40)
        font.setBold(False)
        font.setWeight(50)
        self.label_Title.setFont(font)
        self.label_Title.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Title.setObjectName("label_Title")
        self.horizontalLayout_5.addWidget(self.label_Title)
        spacerItem1 = QtWidgets.QSpacerItem(436, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.verticalLayout_9.addWidget(self.widget_Title)
        self.widget_4 = QtWidgets.QWidget(self.centralwidget)
        self.widget_4.setObjectName("widget_4")
        self.widget_Show = QtWidgets.QWidget(self.widget_4)
        self.widget_Show.setGeometry(QtCore.QRect(9, 9, 418, 514))
        self.widget_Show.setObjectName("widget_Show")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.widget_Show)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.widget_Show1 = QtWidgets.QWidget(self.widget_Show)
        self.widget_Show1.setMaximumSize(QtCore.QSize(400, 90))
        self.widget_Show1.setObjectName("widget_Show1")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget_Show1)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.widget_Show11 = QtWidgets.QWidget(self.widget_Show1)
        self.widget_Show11.setObjectName("widget_Show11")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.widget_Show11)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_Show1 = QtWidgets.QLabel(self.widget_Show11)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_Show1.setFont(font)
        self.label_Show1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Show1.setObjectName("label_Show1")
        self.verticalLayout_4.addWidget(self.label_Show1)
        self.textEdit_Show1 = QtWidgets.QTextEdit(self.widget_Show11)
        self.textEdit_Show1.setEnabled(False)
        self.textEdit_Show1.setMinimumSize(QtCore.QSize(80, 35))
        self.textEdit_Show1.setMaximumSize(QtCore.QSize(80, 35))
        self.textEdit_Show1.setObjectName("textEdit_Show1")
        self.verticalLayout_4.addWidget(self.textEdit_Show1)
        self.horizontalLayout_3.addWidget(self.widget_Show11)
        self.widget_Show14 = QtWidgets.QWidget(self.widget_Show1)
        self.widget_Show14.setObjectName("widget_Show14")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.widget_Show14)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.label_3 = QtWidgets.QLabel(self.widget_Show14)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_10.addWidget(self.label_3)
        self.textEdit_Show3 = QtWidgets.QTextEdit(self.widget_Show14)
        self.textEdit_Show3.setEnabled(False)
        self.textEdit_Show3.setMinimumSize(QtCore.QSize(80, 35))
        self.textEdit_Show3.setMaximumSize(QtCore.QSize(80, 35))
        self.textEdit_Show3.setObjectName("textEdit_Show3")
        self.verticalLayout_10.addWidget(self.textEdit_Show3)
        self.horizontalLayout_3.addWidget(self.widget_Show14)
        self.widget_Show12 = QtWidgets.QWidget(self.widget_Show1)
        self.widget_Show12.setObjectName("widget_Show12")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.widget_Show12)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_Show2 = QtWidgets.QLabel(self.widget_Show12)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_Show2.setFont(font)
        self.label_Show2.setObjectName("label_Show2")
        self.verticalLayout_5.addWidget(self.label_Show2)
        self.textEdit_Show2 = QtWidgets.QTextEdit(self.widget_Show12)
        self.textEdit_Show2.setEnabled(False)
        self.textEdit_Show2.setMinimumSize(QtCore.QSize(80, 35))
        self.textEdit_Show2.setMaximumSize(QtCore.QSize(80, 35))
        self.textEdit_Show2.setObjectName("textEdit_Show2")
        self.verticalLayout_5.addWidget(self.textEdit_Show2)
        self.horizontalLayout_3.addWidget(self.widget_Show12)
        self.widget_Show13 = QtWidgets.QWidget(self.widget_Show1)
        self.widget_Show13.setObjectName("widget_Show13")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.widget_Show13)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_2 = QtWidgets.QLabel(self.widget_Show13)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_6.addWidget(self.label_2)
        self.lcdNumber = QtWidgets.QLCDNumber(self.widget_Show13)
        self.lcdNumber.setMinimumSize(QtCore.QSize(80, 35))
        self.lcdNumber.setMaximumSize(QtCore.QSize(80, 35))
        self.lcdNumber.setObjectName("lcdNumber")
        self.verticalLayout_6.addWidget(self.lcdNumber)
        self.horizontalLayout_3.addWidget(self.widget_Show13)
        self.verticalLayout_7.addWidget(self.widget_Show1)
        self.widget_Show2 = QtWidgets.QWidget(self.widget_Show)
        self.widget_Show2.setMinimumSize(QtCore.QSize(400, 400))
        self.widget_Show2.setMaximumSize(QtCore.QSize(400, 400))
        self.widget_Show2.setObjectName("widget_Show2")
        self.verticalLayout_7.addWidget(self.widget_Show2)
        self.widget_2 = QtWidgets.QWidget(self.widget_4)
        self.widget_2.setGeometry(QtCore.QRect(440, 90, 701, 451))
        self.widget_2.setObjectName("widget_2")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.widget_2)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.widget_Control = QtWidgets.QWidget(self.widget_2)
        self.widget_Control.setObjectName("widget_Control")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget_Control)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.widget_Control_camera = QtWidgets.QWidget(self.widget_Control)
        self.widget_Control_camera.setMinimumSize(QtCore.QSize(400, 400))
        self.widget_Control_camera.setMaximumSize(QtCore.QSize(400, 400))
        self.widget_Control_camera.setObjectName("widget_Control_camera")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget_Control_camera)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_Camera = QtWidgets.QLabel(self.widget_Control_camera)
        self.label_Camera.setEnabled(True)
        self.label_Camera.setMinimumSize(QtCore.QSize(400, 400))
        self.label_Camera.setMaximumSize(QtCore.QSize(400, 400))
        self.label_Camera.setText("")
        self.label_Camera.setObjectName("label_Camera")
        self.verticalLayout_3.addWidget(self.label_Camera)
        self.horizontalLayout_2.addWidget(self.widget_Control_camera)
        self.widget_Control2 = QtWidgets.QWidget(self.widget_Control)
        self.widget_Control2.setMaximumSize(QtCore.QSize(200, 400))
        self.widget_Control2.setObjectName("widget_Control2")
        self.label = QtWidgets.QLabel(self.widget_Control2)
        self.label.setGeometry(QtCore.QRect(80, 40, 101, 31))
        self.label.setMinimumSize(QtCore.QSize(0, 0))
        self.label.setMaximumSize(QtCore.QSize(564654, 40))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.pushButton_Control2 = QtWidgets.QPushButton(self.widget_Control2)
        self.pushButton_Control2.setGeometry(QtCore.QRect(80, 280, 95, 35))
        self.pushButton_Control2.setMinimumSize(QtCore.QSize(95, 35))
        self.pushButton_Control2.setMaximumSize(QtCore.QSize(95, 35))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButton_Control2.setFont(font)
        self.pushButton_Control2.setObjectName("pushButton_Control2")
        self.pushButton_Control1 = QtWidgets.QPushButton(self.widget_Control2)
        self.pushButton_Control1.setGeometry(QtCore.QRect(80, 180, 95, 35))
        self.pushButton_Control1.setMinimumSize(QtCore.QSize(95, 35))
        self.pushButton_Control1.setMaximumSize(QtCore.QSize(95, 35))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButton_Control1.setFont(font)
        self.pushButton_Control1.setObjectName("pushButton_Control1")
        self.comboBox_Control1 = QtWidgets.QComboBox(self.widget_Control2)
        self.comboBox_Control1.setGeometry(QtCore.QRect(80, 80, 95, 35))
        self.comboBox_Control1.setMinimumSize(QtCore.QSize(0, 0))
        self.comboBox_Control1.setMaximumSize(QtCore.QSize(95, 35))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.comboBox_Control1.setFont(font)
        self.comboBox_Control1.setObjectName("comboBox_Control1")
        self.comboBox_Control1.addItem("")
        self.comboBox_Control1.addItem("")
        self.comboBox_Control1.addItem("")
        self.horizontalLayout_2.addWidget(self.widget_Control2)
        self.verticalLayout_8.addWidget(self.widget_Control)
        self.verticalLayout_9.addWidget(self.widget_4)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1300, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        # add dynamic_exam
        self.widget_mediaw1 = QVideoWidget(self.widget_Show2)
        self.widget_mediaw1.setMinimumSize(QtCore.QSize(400, 400))
        self.widget_mediaw1.setMaximumSize(QtCore.QSize(400, 400))
        self.widget_mediaw1.setObjectName("widget_mediaw1")
        self.verticalLayout_media1 = QtWidgets.QVBoxLayout(self.widget_Show2)
        self.playlist = QMediaPlaylist(self.widget_Show2)
        self.playlist.addMedia(QMediaContent(QtCore.QUrl.fromLocalFile('./resource/sampleVideo/dynamic/allHands.avi')))
        self.playlist.setPlaybackMode(QMediaPlaylist.CurrentItemInLoop)
        self.verticalLayout_media1.setObjectName("verticalLayout_media1")
        
        self.player1 = QMediaPlayer(self.widget_Show2)
        self.player1.setObjectName("player1")
        self.player1.setVideoOutput(self.widget_mediaw1)
        self.player1.setPlaylist(self.playlist)
        self.player1.play() 
        self.verticalLayout_media1.addWidget(self.widget_mediaw1)
        # add end

        self.retranslateUi(MainWindow)
        self.pushButton_Control1.clicked.connect(MainWindow.examStart)
        self.pushButton_Control2.clicked.connect(MainWindow.examEnd)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_Title.setText(_translate("MainWindow", "????????????????????????"))
        self.label_Show1.setText(_translate("MainWindow", "????????????"))
        self.label_3.setText(_translate("MainWindow", "??????"))
        self.label_Show2.setText(_translate("MainWindow", "???????????????"))
        self.label_2.setText(_translate("MainWindow", "?????????"))
        self.label.setText(_translate("MainWindow", "???????????????"))
        self.pushButton_Control2.setText(_translate("MainWindow", "??????"))
        self.pushButton_Control1.setText(_translate("MainWindow", "??????"))
        self.comboBox_Control1.setItemText(0, _translate("MainWindow", "?????? (10)"))
        self.comboBox_Control1.setItemText(1, _translate("MainWindow", "?????? (20)"))
        self.comboBox_Control1.setItemText(2, _translate("MainWindow", "?????? (30)"))
