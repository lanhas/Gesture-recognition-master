"""贪吃蛇"""

import random
import time
import pygame
import cv2
from pygame.locals import *
from collections import deque
from PyQt5.QtCore import *
import numpy as np
from constants.enum_keys import HG
from warnings import warn
from hgdataset.s1_skeleton import HgdSkeleton
from imgaug import KeypointsOnImage
from imgaug.imgaug import draw_text
from utils.resize import ResizeKeepRatio
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import pred.hands_pred
from pathlib import Path


class GluttonousSnake(QObject):
    def __init__(self) -> None:
        super(QObject, self).__init__()
        self.cap = cv2.VideoCapture(0)
        self.hands_player = StaticPlayer()
        self.currentEvent = 'NONE'
        self.update_flag = True

        self.SCREEN_WIDTH = 600      # 屏幕宽度
        self.SCREEN_HEIGHT = 480     # 屏幕高度
        self.SIZE = 20               # 小方格大小
        self.LINE_WIDTH = 1          # 网格线宽度

        # 游戏区域的坐标范围
        self.SCOPE_X = (0, self.SCREEN_WIDTH // self.SIZE - 1)
        self.SCOPE_Y = (2, self.SCREEN_HEIGHT // self.SIZE - 1)

        # 食物的分值及颜色
        self.FOOD_STYLE_LIST = [(10, (255, 100, 100)), (20, (100, 255, 100)), (30, (100, 100, 255))]

        self.LIGHT = (100, 100, 100)
        self.snakeColors = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), 
                            (0, 127, 255), (0, 0, 255), (139, 0, 255)]       # 蛇的颜色
        self.BLACK = (0, 0, 0)           # 网格线颜色
        self.RED = (200, 30, 30)         # 红色，GAME OVER 的字体颜色
        self.BGCOLOR = (40, 40, 60)      # 背景色


    def print_text(self, screen, font, x, y, text, fcolor=(255, 255, 255)):
        imgText = font.render(text, True, fcolor)
        screen.blit(imgText, (x, y))


    # 初始化蛇
    def init_snake(self, snakeSerial):
        snake = deque()
        snake.append((2, self.SCOPE_Y[0] + snakeSerial))
        snake.append((1, self.SCOPE_Y[0] + snakeSerial))
        snake.append((0, self.SCOPE_Y[0] + snakeSerial))
        return snake


    def create_food(self, Snakes):
        food_x = random.randint(self.SCOPE_X[0], self.SCOPE_X[1])
        food_y = random.randint(self.SCOPE_Y[0], self.SCOPE_Y[1])
        for snake in Snakes:
            while (food_x, food_y) in snake:
                # 如果食物出现在蛇身上，则重来
                food_x = random.randint(self.SCOPE_X[0], self.SCOPE_X[1])
                food_y = random.randint(self.SCOPE_Y[0], self.SCOPE_Y[1])
        return food_x, food_y


    def get_food_style(self):
        return self.FOOD_STYLE_LIST[random.randint(0, 2)]

    def staticHandsShow(self):
        self.hands_player.play_custom_video(None)

    def hands_controller(self, hands):
        if hands == 0:
            self.currentEvent = 'NONE'
        elif hands == 1:
            self.currentEvent = 'UP'
        elif hands == 2:
            pass
        elif hands == 3:
            self.currentEvent = 'LEFT'
        elif hands == 4:
            self.currentEvent = 'RIGHT'
        elif hands == 5:
            pass
        elif hands == 6:
            self.currentEvent = 'END'
        elif hands == 7:
            self.currentEvent = 'START'
        elif hands == 8:
            self.currentEvent = 'PAUSE'
        elif hands == 9:
            self.currentEvent = 'DOWN'
        elif hands == 10:
            self.currentEvent = 'SPEED_UP'
        elif hands == 11:
            self.currentEvent = 'SPEEN_DOWN'

    def main(self, snakeNum):

        self.hands_player.start()
        pygame.init()
        screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption('贪吃蛇')

        font1 = pygame.font.SysFont('SimHei', 24)  # 得分的字体
        font2 = pygame.font.Font(None, 72)  # GAME OVER 的字体
        fwidth, fheight = font2.size('GAME OVER')
        multySnake = []
        # 如果蛇正在向右移动，那么快速点击向下向左，由于程序刷新没那么快，向下事件会被向左覆盖掉，导致蛇后退，直接GAME OVER
        # b 变量就是用于防止这种情况的发生
        b = True
        #当前的蛇
        self.currentSnake = 0

        # 蛇
        for i in range(snakeNum):
            multySnake.append(self.init_snake(i))
        snake = multySnake[self.currentSnake]
        # 食物
        food = self.create_food(multySnake)
        food_style = self.get_food_style()
        # 方向
        pos = (1, 0)

        game_over = True
        start = False       # 是否开始，当start = True，game_over = True 时，才显示 GAME OVER
        score = 0           # 得分
        orispeed = 0.5      # 原始速度
        speed = orispeed
        last_move_time = None
        pause = False       # 暂停
        last_hands = None
        current_hands = None
        update_flag = True

        while True:
            last_hands = current_hands
            current_hands = self.hands_player.currentHands
            if last_hands == current_hands:
                update_flag = False
            else:
                update_flag = True
            self.hands_controller(current_hands)
            snake = multySnake[self.currentSnake]

            if self.currentEvent == 'UP':
                    # 这个判断是为了防止蛇向上移时按了向下键，导致直接 GAME OVER
                if b and not pos[1]:
                    pos = (0, -1)
                    b = False
            elif self.currentEvent == 'DOWN':
                if b and not pos[1]:
                    pos = (0, 1)
                    b = False
            elif self.currentEvent == 'LEFT':
                if b and not pos[0]:
                    pos = (-1, 0)
                    b = False
            elif self.currentEvent == 'RIGHT':
                if b and not pos[0]:
                    pos = (1, 0)
                    b = False

            if update_flag == True:
                update_flag = False
                if self.currentEvent == 'NONE':
                    pass
                elif self.currentEvent == 'END':
                    pygame.quit()
                    del self.hands_player
                    return True
                elif self.currentEvent == 'START':
                    if game_over:
                        multySnake = []
                        start = True
                        game_over = False
                        speed = orispeed
                        self.currentSnake = 0
                        b = True
                        for i in range(snakeNum):
                            multySnake.append(self.init_snake(i))
                        snake = multySnake[self.currentSnake]
                        food = self.create_food(multySnake)
                        food_style = self.get_food_style()
                        pos = (1, 0)
                        # 得分
                        score = 0
                        last_move_time = time.time()
                elif self.currentEvent == 'PAUSE':
                    if not game_over:
                        pause = not pause
                elif self.currentEvent == 'SPEED_UP':
                    if speed > 0 and speed < 1:
                        speed = speed - 0.1
                elif self.currentEvent == 'SPEED_DOWN':
                    if speed > 0 and speed < 1:
                        speed = speed + 0.1
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return True
                elif event.type == KEYDOWN:
                    if event.key == K_RETURN:
                        if game_over:
                            multySnake = []
                            start = True
                            game_over = False
                            b = True
                            for i in range(snakeNum):
                                multySnake.append(self.init_snake(i))
                            snake = multySnake[self.currentSnake]
                            food = self.create_food(multySnake)
                            food_style = self.get_food_style()
                            pos = (1, 0)
                            # 得分
                            score = 0
                            last_move_time = time.time()
                    elif event.key == K_SPACE:
                        if not game_over:
                            pause = not pause
                    elif event.key in (K_w, K_UP):
                        # 这个判断是为了防止蛇向上移时按了向下键，导致直接 GAME OVER
                        if b and not pos[1]:
                            pos = (0, -1)
                            b = False
                    elif event.key in (K_s, K_DOWN):
                        if b and not pos[1]:
                            pos = (0, 1)
                            b = False
                    elif event.key in (K_a, K_LEFT):
                        if b and not pos[0]:
                            pos = (-1, 0)
                            b = False
                    elif event.key in (K_d, K_RIGHT):
                        if b and not pos[0]:
                            pos = (1, 0)
                            b = False

            # 填充背景色
            screen.fill(self.BGCOLOR)
            # 画网格线 竖线
            for x in range(self.SIZE, self.SCREEN_WIDTH, self.SIZE):
                pygame.draw.line(screen, self.BLACK, (x, self.SCOPE_Y[0] * self.SIZE), (x, self.SCREEN_HEIGHT), self.LINE_WIDTH)
            # 画网格线 横线
            for y in range(self.SCOPE_Y[0] * self.SIZE, self.SCREEN_HEIGHT, self.SIZE):
                pygame.draw.line(screen, self.BLACK, (0, y), (self.SCREEN_WIDTH, y), self.LINE_WIDTH)

            if not game_over:
                curTime = time.time()
                if curTime - last_move_time > speed:
                    if not pause:
                        b = True
                        last_move_time = curTime
                        next_s = (snake[0][0] + pos[0], snake[0][1] + pos[1])
                        if next_s == food:
                            # 吃到了食物
                            update_flag = False
                            self.currentSnake = (self.currentSnake + 1) % snakeNum
                            snake.appendleft(next_s)
                            score += food_style[0]
                            speed = orispeed - 0.03 * (score // 100)
                            food = self.create_food(multySnake)
                            food_style = self.get_food_style()
                            pos = (1, 0)
                        else:
                            if self.SCOPE_X[0] <= next_s[0] <= self.SCOPE_X[1] and self.SCOPE_Y[0] <= next_s[1] <= self.SCOPE_Y[1] \
                                    and next_s not in snake:
                                snake.appendleft(next_s)
                                snake.pop()
                            else:
                                game_over = True

            # 画食物
            if not game_over:
                # 避免 GAME OVER 的时候把 GAME OVER 的字给遮住了
                pygame.draw.rect(screen, food_style[1], (food[0] * self.SIZE, food[1] * self.SIZE, self.SIZE, self.SIZE), 0)

            # 画蛇
            for index, snake in enumerate(multySnake):
                # print(index)
                for s in snake:
                    pygame.draw.rect(screen, self.snakeColors[index], (s[0] * self.SIZE + self.LINE_WIDTH, s[1] * self.SIZE + self.LINE_WIDTH,
                                                    self.SIZE - self.LINE_WIDTH * 2, self.SIZE - self.LINE_WIDTH * 2), 0)

            self.print_text(screen, font1, 30, 7, f'速度: {score//100}')
            self.print_text(screen, font1, 450, 7, f'得分: {score}')

            if game_over:
                if start:
                    self.print_text(screen, font2, (self.SCREEN_WIDTH - fwidth) // 2, (self.SCREEN_HEIGHT - fheight) // 2, 'GAME OVER', self.RED)

            pygame.display.update()




class StaticPlayer(QThread):
    hands_singal = pyqtSignal(int)
    def __init__(self, is_unittest=False):
        super(QThread, self).__init__()
        self.img_size = (512, 512)
        self.hpred = pred.hands_pred.HandsPred("static")
        self.is_unittest = is_unittest
        self.show_camera = True
        self.currentHands = None

    def play_dataset_video(self, is_train, video_index, show=True):
        self.scd = HgdSkeleton('static', Path.home() / 'MeetingHands', is_train, self.img_size)
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
                self.currentHands = hands_now
                # print(hands_now)
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

    def run(self):
        self.play_custom_video(None)
    

    hands_dict = {
        0: "NO SIGNAL",
        1: "UP",
        2: "DOWN",
        3: "LEFT",
        4: "RIGHT",
        5: "START",
        6: "END",
        7: "START",
        8: "PAUSE",
        9: "DOWN",
        10: "SPEED up",
        11: "speed down"}

    hands_dict_c = {
        0: "无信号",
        1: "上",
        2: "下",
        3: "左",
        4: "右",
        5: "开始",
        6: "结束",
        7: "开始",
        8: "暂停",
        9: "下",
        10: "加速",
        11: "减速"}

if __name__ == '__main__':
    gs = GluttonousSnake()
    gs.main(7)
