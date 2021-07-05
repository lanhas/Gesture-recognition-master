'''
    To control windows jump to each other
'''
from controller.windows import *


class Controller:
    def __init__(self):
        pass

    def showMain(self):
        self.main = mainDecoration()
        self.main.setWindowTitle('手势识别及控制系统')
        self.main.switch_meeting.connect(self.showMeeting)
        self.main.switch_recovery.connect(self.showRecovery)
        self.main.show()
    
    def showMeeting(self):
        self.meetingWindow = MeetingOperation()
        self.meetingWindow.setWindowTitle('动态手势识别演示系统')
        self.meetingWindow.back_operate.connect(self.showMain)
        self.main.hide()
        self.meetingWindow.show()
        try:
            self.meetingWindow.play_hands()
        except Exception:
            return True

    def showRecovery(self):
        self.recoveryWindow = RecoveryOperation()
        self.recoveryWindow.setWindowTitle('手部康复系统')
        self.recoveryWindow.back_operate.connect(self.showMain)
        self.recoveryWindow.switch_dynamicScore.connect(self.showDynamicScore)
        self.recoveryWindow.switch_staticScore.connect(self.showStaticScore)
        self.recoveryWindow.switch_dynamicExam.connect(self.showDynamicExam)
        self.recoveryWindow.switch_recoveryLearning.connect(self.showRecoveryLearning)
        self.recoveryWindow.switch_recoveryGuide.connect(self.showRecoveryGuide)
        self.main.hide()
        self.recoveryWindow.show()

    def showStaticScore(self):
        self.staticScoreWindow = StaticScoreWindow()
        self.staticScoreWindow.setWindowTitle('静态手势评分系统')
        self.staticScoreWindow.back_operate.connect(self.showRecovery)
        self.recoveryWindow.hide()
        self.staticScoreWindow.show()

    def showDynamicScore(self):
        self.dynamicScoreWindow = DynamicScoreWindow()
        self.dynamicScoreWindow.setWindowTitle('动态手势评分系统')
        self.dynamicScoreWindow.back_operate.connect(self.showRecovery)
        self.recoveryWindow.hide()
        self.dynamicScoreWindow.show()

    def showDynamicExam(self):
        self.dynamicExamWindow = DynamicExamWindow()
        self.dynamicExamWindow.setWindowTitle('康复收拾通关系统')
        self.dynamicExamWindow.back_operate.connect(self.showRecovery)
        self.recoveryWindow.hide()
        self.dynamicExamWindow.show()

    def showRecoveryLearning(self):
        self.recoveryLearningWindow = RecoveryLearningWindow()
        self.recoveryLearningWindow.setWindowTitle('康复学习系统')
        self.recoveryLearningWindow.back_operate.connect(self.showRecovery)
        self.recoveryWindow.hide()
        self.recoveryLearningWindow.show()

    def showRecoveryGuide(self):
        self.recoveryGuideWindow = RecoveryGuideWindow()
        self.recoveryGuideWindow.setWindowTitle('康复指南系统')
        self.recoveryGuideWindow.back_operate.connect(self.showRecovery)
        self.recoveryWindow.hide()
        self.recoveryGuideWindow.show()


