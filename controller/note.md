# if you change the interface by dynamic_exam.ui or dynamic_score.ui then use the instruct "PYQT:Compire Firm"
# copy this code to dynamic_score.ui and dynamic_exam.py


# add this to all of the py file in head
# copy start
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import QVideoWidget
# copy end

# add this to dynamic_exam.py
# add start
        self.widget_mediaw1 = QVideoWidget(self.widget_Show2)
        self.widget_mediaw1.setMinimumSize(QtCore.QSize(400, 400))
        self.widget_mediaw1.setMaximumSize(QtCore.QSize(400, 400))
        self.widget_mediaw1.setObjectName("widget_mediaw1")
        self.verticalLayout_media1 = QtWidgets.QVBoxLayout(self.widget_Show2)
        self.playlist = QMediaPlaylist(self.widget_Show2)
        self.playlist.addMedia(QMediaContent(QtCore.QUrl.fromLocalFile('./resource/sampleVideo/allHands.avi')))
        self.playlist.setPlaybackMode(QMediaPlaylist.CurrentItemInLoop)
        self.verticalLayout_media1.setObjectName("verticalLayout_media1")
        
        self.player1 = QMediaPlayer(self.widget_Show2)
        self.player1.setObjectName("player1")
        self.player1.setVideoOutput(self.widget_mediaw1)
        self.player1.setPlaylist(self.playlist)
        self.player1.play() 
        self.verticalLayout_media1.addWidget(self.widget_mediaw1)
# add end

# add this to dynamic_score.py
# add start
        self.widget_mediaw1 = QVideoWidget(self.widget_HSVideo)
        self.widget_mediaw1.setMinimumSize(QtCore.QSize(400, 400))
        self.widget_mediaw1.setMaximumSize(QtCore.QSize(400, 400))
        self.widget_mediaw1.setObjectName("widget_mediaw1")
        self.verticalLayout_media1 = QtWidgets.QVBoxLayout(self.widget_HSVideo)
        self.verticalLayout_media1.setObjectName("verticalLayout_media1")
        self.playlist = QMediaPlaylist(self.widget_HSVideo)
        self.playlist.addMedia(QMediaContent(QtCore.QUrl.fromLocalFile('./resource/sampleVideo/allHands.avi')))
        self.playlist.setPlaybackMode(QMediaPlaylist.CurrentItemInLoop)
        self.player1 = QMediaPlayer(self.widget_HSVideo)
        self.player1.setObjectName("player1")
        self.player1.setVideoOutput(self.widget_mediaw1)
        self.player1.setPlaylist(self.playlist)
        self.player1.play() 
        self.verticalLayout_media1.addWidget(self.widget_mediaw1)
# add end

# add this to recovery_learning.py
# add start
        self.widget_mediaw = QVideoWidget(self.widget_video)
        self.widget_mediaw.setMinimumSize(QtCore.QSize(640, 480))
        self.widget_mediaw.setMaximumSize(QtCore.QSize(640, 480))
        self.widget_mediaw.setObjectName("widget_mediaw")
        self.playlist = QMediaPlaylist(self.widget_video)
        self.playlist.addMedia(QMediaContent(QtCore.QUrl.fromLocalFile('./resource/sampleVideo/allHands.avi')))
        self.playlist.setPlaybackMode(QMediaPlaylist.CurrentItemInLoop)
        self.player1 = QMediaPlayer(self.widget_video)
        self.player1.setObjectName("player1")
        self.player1.setVideoOutput(self.widget_mediaw)
        self.player1.setPlaylist(self.playlist)
        self.player1.play() 
# add end
