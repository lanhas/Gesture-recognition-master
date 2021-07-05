import random
import time
import pygame
import cv2
from pygame.locals import *
from PyQt5.QtCore import *
from collections import namedtuple
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

class Blocks():
    def __init__(self) -> None:
        self.Point = namedtuple('Point', 'X Y')
        self.Shape = namedtuple('Shape', 'X Y Width Height')
        self.Block = namedtuple('Block', 'template start_pos end_pos name next')

        # 方块形状的设计，我最初我是做成 4 × 4，因为长宽最长都是4，这样旋转的时候就不考虑怎么转了，就是从一个图形替换成另一个
        # 其实要实现这个功能，只需要固定左上角的坐标就可以了

        # S形方块
        self.S_BLOCK = [self.Block(['.OO',
                        'OO.',
                        '...'], self.Point(0, 0), self.Point(2, 1), 'S', 1),
                self.Block(['O..',
                        'OO.',
                        '.O.'], self.Point(0, 0), self.Point(1, 2), 'S', 0)]
        # Z形方块
        self.Z_BLOCK = [self.Block(['OO.',
                        '.OO',
                        '...'], self.Point(0, 0), self.Point(2, 1), 'Z', 1),
                self.Block(['.O.',
                        'OO.',
                        'O..'], self.Point(0, 0), self.Point(1, 2), 'Z', 0)]
        # I型方块
        self.I_BLOCK = [self.Block(['.O..',
                        '.O..',
                        '.O..',
                        '.O..'], self.Point(1, 0), self.Point(1, 3), 'I', 1),
                self.Block(['....',
                        '....',
                        'OOOO',
                        '....'], self.Point(0, 2), self.Point(3, 2), 'I', 0)]
        # O型方块
        self.O_BLOCK = [self.Block(['OO',
                        'OO'], self.Point(0, 0), self.Point(1, 1), 'O', 0)]
        # J型方块
        self.J_BLOCK = [self.Block(['O..',
                        'OOO',
                        '...'], self.Point(0, 0), self.Point(2, 1), 'J', 1),
                self.Block(['.OO',
                        '.O.',
                        '.O.'], self.Point(1, 0), self.Point(2, 2), 'J', 2),
                self.Block(['...',
                        'OOO',
                        '..O'], self.Point(0, 1), self.Point(2, 2), 'J', 3),
                self.Block(['.O.',
                        '.O.',
                        'OO.'], self.Point(0, 0), self.Point(1, 2), 'J', 0)]
        # L型方块
        self.L_BLOCK = [self.Block(['..O',
                        'OOO',
                        '...'], self.Point(0, 0), self.Point(2, 1), 'L', 1),
                self.Block(['.O.',
                        '.O.',
                        '.OO'], self.Point(1, 0), self.Point(2, 2), 'L', 2),
                self.Block(['...',
                        'OOO',
                        'O..'], self.Point(0, 1), self.Point(2, 2), 'L', 3),
                self.Block(['OO.',
                        '.O.',
                        '.O.'], self.Point(0, 0), self.Point(1, 2), 'L', 0)]
        # T型方块
        self.T_BLOCK = [self.Block(['.O.',
                        'OOO',
                        '...'], self.Point(0, 0), self.Point(2, 1), 'T', 1),
                self.Block(['.O.',
                        '.OO',
                        '.O.'], self.Point(1, 0), self.Point(2, 2), 'T', 2),
                self.Block(['...',
                        'OOO',
                        '.O.'], self.Point(0, 1), self.Point(2, 2), 'T', 3),
                self.Block(['.O.',
                        'OO.',
                        '.O.'], self.Point(0, 0), self.Point(1, 2), 'T', 0)]

        self.BLOCKS = {'O': self.O_BLOCK,
                'I': self.I_BLOCK,
                'Z': self.Z_BLOCK,
                'T': self.T_BLOCK,
                'L': self.L_BLOCK,
                'S': self.S_BLOCK,
                'J': self.J_BLOCK}


    def get_block(self):
        block_name = random.choice('OIZTLSJ')
        b = self.BLOCKS[block_name]
        idx = random.randint(0, len(b) - 1)
        return b[idx]


    def get_next_block(self, block):
        b = self.BLOCKS[block.name]
        return b[block.next]


class Tetris():
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0)
        self.hands_player = StaticPlayer()
        self.currentEvent = 'NONE'
        self.update_flag = True

        self.blocks = Blocks()
        self.SIZE = 30  # 每个小方格大小
        self.BLOCK_HEIGHT = 25  # 游戏区高度
        self.BLOCK_WIDTH = 10   # 游戏区宽度
        self.BORDER_WIDTH = 4   # 游戏区边框宽度
        self.BORDER_COLOR = (40, 40, 200)  # 游戏区边框颜色
        self.SCREEN_WIDTH = self.SIZE * (self.BLOCK_WIDTH + 5)  # 游戏屏幕的宽
        self.SCREEN_HEIGHT = self.SIZE * self.BLOCK_HEIGHT      # 游戏屏幕的高
        self.BG_COLOR = (40, 40, 60)  # 背景色
        self.BLOCK_COLOR = (20, 128, 200)  #
        self.BLACK = (0, 0, 0)
        self.RED = (200, 30, 30)      # GAME OVER 的字体颜色


    def print_text(self, screen, font, x, y, text, fcolor=(255, 255, 255)):
        imgText = font.render(text, True, fcolor)
        screen.blit(imgText, (x, y))

    def hands_controller(self, hands):
        if hands == 0:
            self.currentEvent = 'NONE'
        elif hands == 1:
            pass
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
            self.currentEvent = 'ROTATE'
        elif hands == 11:
            self.currentEvent = 'SPEEN_DOWN'

    def main(self):
        self.hands_player.start()
        pygame.init()
        screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption('俄罗斯方块')

        font1 = pygame.font.SysFont('SimHei', 24)  # 黑体24
        font2 = pygame.font.Font(None, 72)  # GAME OVER 的字体
        font_pos_x = self.BLOCK_WIDTH * self.SIZE + self.BORDER_WIDTH + 10  # 右侧信息显示区域字体位置的X坐标
        gameover_size = font2.size('GAME OVER')
        font1_height = int(font1.size('得分')[1])

        cur_block = None   # 当前下落方块
        next_block = None  # 下一个方块
        cur_pos_x, cur_pos_y = 0, 0

        game_area = None    # 整个游戏区域
        game_over = True
        start = False       # 是否开始，当start = True，game_over = True 时，才显示 GAME OVER
        score = 0           # 得分
        orispeed = 0.5      # 原始速度
        speed = orispeed    # 当前速度
        pause = False       # 暂停
        last_drop_time = None   # 上次下落时间
        last_press_time = None  # 上次按键时间
        last_hands = None
        current_hands = None
        update_flag = True

        def _dock():
            nonlocal cur_block, next_block, game_area, cur_pos_x, cur_pos_y, game_over, score, speed
            for _i in range(cur_block.start_pos.Y, cur_block.end_pos.Y + 1):
                for _j in range(cur_block.start_pos.X, cur_block.end_pos.X + 1):
                    if cur_block.template[_i][_j] != '.':
                        game_area[cur_pos_y + _i][cur_pos_x + _j] = '0'
            if cur_pos_y + cur_block.start_pos.Y <= 0:
                game_over = True
            else:
                # 计算消除
                remove_idxs = []
                for _i in range(cur_block.start_pos.Y, cur_block.end_pos.Y + 1):
                    if all(_x == '0' for _x in game_area[cur_pos_y + _i]):
                        remove_idxs.append(cur_pos_y + _i)
                if remove_idxs:
                    # 计算得分
                    remove_count = len(remove_idxs)
                    if remove_count == 1:
                        score += 100
                    elif remove_count == 2:
                        score += 300
                    elif remove_count == 3:
                        score += 700
                    elif remove_count == 4:
                        score += 1500
                    speed = orispeed - 0.03 * (score // 10000)
                    # 消除
                    _i = _j = remove_idxs[-1]
                    while _i >= 0:
                        while _j in remove_idxs:
                            _j -= 1
                        if _j < 0:
                            game_area[_i] = ['.'] * self.BLOCK_WIDTH
                        else:
                            game_area[_i] = game_area[_j]
                        _i -= 1
                        _j -= 1
                cur_block = next_block
                next_block = self.blocks.get_block()
                cur_pos_x, cur_pos_y = (self.BLOCK_WIDTH - cur_block.end_pos.X - 1) // 2, -1 - cur_block.end_pos.Y

        def _judge(pos_x, pos_y, block):
            nonlocal game_area
            for _i in range(block.start_pos.Y, block.end_pos.Y + 1):
                if pos_y + block.end_pos.Y >= self.BLOCK_HEIGHT:
                    return False
                for _j in range(block.start_pos.X, block.end_pos.X + 1):
                    if pos_y + _i >= 0 and block.template[_i][_j] != '.' and game_area[pos_y + _i][pos_x + _j] != '.':
                        return False
            return True

        while True:
            last_hands = current_hands
            current_hands = self.hands_player.currentHands
            if last_hands == current_hands:
                update_flag = False
            else:
                update_flag = True
            self.hands_controller(current_hands)

            if update_flag == True:
                update_flag = False
                if self.currentEvent == 'LEFT':
                    if not game_over and not pause:
                        if time.time() - last_press_time > 0.1:
                            last_press_time = time.time()
                            if cur_pos_x > - cur_block.start_pos.X:
                                if _judge(cur_pos_x - 1, cur_pos_y, cur_block):
                                    cur_pos_x -= 1
                elif self.currentEvent == 'RIGHT':
                    if not game_over and not pause:
                        if time.time() - last_press_time > 0.1:
                            last_press_time = time.time()
                            # 不能移除右边框
                            if cur_pos_x + cur_block.end_pos.X + 1 < self.BLOCK_WIDTH:
                                if _judge(cur_pos_x + 1, cur_pos_y, cur_block):
                                    cur_pos_x += 1
                elif self.currentEvent == 'DOWN':
                    if not game_over and not pause:
                        if time.time() - last_press_time > 0.1:
                            last_press_time = time.time()
                            if not _judge(cur_pos_x, cur_pos_y + 1, cur_block):
                                _dock()
                            else:
                                last_drop_time = time.time()
                                cur_pos_y += 1
                elif self.currentEvent == 'END':
                    pygame.quit()
                    return True
                    
                elif self.currentEvent == 'START':
                    if game_over:
                        start = True
                        game_over = False
                        score = 0
                        last_drop_time = time.time()
                        last_press_time = time.time()
                        game_area = [['.'] * self.BLOCK_WIDTH for _ in range(self.BLOCK_HEIGHT)]
                        cur_block = self.blocks.get_block()
                        next_block = self.blocks.get_block()
                        cur_pos_x, cur_pos_y = (self.BLOCK_WIDTH - cur_block.end_pos.X - 1) // 2, -1 - cur_block.end_pos.Y
                elif self.currentEvent == 'PAUSE':
                    if not game_over:
                        pause = not pause
                elif self.currentEvent == 'ROTATE':
                    # 旋转
                    # 其实记得不是很清楚了，比如
                    # .0.
                    # .00
                    # ..0
                    # 这个在最右边靠边的情况下是否可以旋转，我试完了网上的俄罗斯方块，是不能旋转的，这里我们就按不能旋转来做
                    # 我们在形状设计的时候做了很多的空白，这样只需要规定整个形状包括空白部分全部在游戏区域内时才可以旋转
                    if 0 <= cur_pos_x <= self.BLOCK_WIDTH - len(cur_block.template[0]):
                        _next_block = self.blocks.get_next_block(cur_block)
                        if _judge(cur_pos_x, cur_pos_y, _next_block):
                            cur_block = _next_block

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return True
                elif event.type == KEYDOWN:
                    if event.key == K_RETURN:
                        if game_over:
                            start = True
                            game_over = False
                            score = 0
                            last_drop_time = time.time()
                            last_press_time = time.time()
                            game_area = [['.'] * self.BLOCK_WIDTH for _ in range(self.BLOCK_HEIGHT)]
                            cur_block = self.blocks.get_block()
                            next_block = self.blocks.get_block()
                            cur_pos_x, cur_pos_y = (self.BLOCK_WIDTH - cur_block.end_pos.X - 1) // 2, -1 - cur_block.end_pos.Y
                    elif event.key == K_SPACE:
                        if not game_over:
                            pause = not pause
                    elif event.key in (K_w, K_UP):
                        # 旋转
                        # 其实记得不是很清楚了，比如
                        # .0.
                        # .00
                        # ..0
                        # 这个在最右边靠边的情况下是否可以旋转，我试完了网上的俄罗斯方块，是不能旋转的，这里我们就按不能旋转来做
                        # 我们在形状设计的时候做了很多的空白，这样只需要规定整个形状包括空白部分全部在游戏区域内时才可以旋转
                        if 0 <= cur_pos_x <= self.BLOCK_WIDTH - len(cur_block.template[0]):
                            _next_block = self.blocks.get_next_block(cur_block)
                            if _judge(cur_pos_x, cur_pos_y, _next_block):
                                cur_block = _next_block

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    if not game_over and not pause:
                        if time.time() - last_press_time > 0.1:
                            last_press_time = time.time()
                            if cur_pos_x > - cur_block.start_pos.X:
                                if _judge(cur_pos_x - 1, cur_pos_y, cur_block):
                                    cur_pos_x -= 1
                if event.key == pygame.K_RIGHT:
                    if not game_over and not pause:
                        if time.time() - last_press_time > 0.1:
                            last_press_time = time.time()
                            # 不能移除右边框
                            if cur_pos_x + cur_block.end_pos.X + 1 < self.BLOCK_WIDTH:
                                if _judge(cur_pos_x + 1, cur_pos_y, cur_block):
                                    cur_pos_x += 1
                if event.key == pygame.K_DOWN:
                    if not game_over and not pause:
                        if time.time() - last_press_time > 0.1:
                            last_press_time = time.time()
                            if not _judge(cur_pos_x, cur_pos_y + 1, cur_block):
                                _dock()
                            else:
                                last_drop_time = time.time()
                                cur_pos_y += 1

            self._draw_background(screen)

            self._draw_game_area(screen, game_area)

            self._draw_gridlines(screen)

            self._draw_info(screen, font1, font_pos_x, font1_height, score)
            # 画显示信息中的下一个方块
            self._draw_block(screen, next_block, font_pos_x, 30 + (font1_height + 6) * 5, 0, 0)

            if not game_over:
                cur_drop_time = time.time()
                if cur_drop_time - last_drop_time > speed:
                    if not pause:
                        # 不应该在下落的时候来判断到底没，我们玩俄罗斯方块的时候，方块落到底的瞬间是可以进行左右移动
                        if not _judge(cur_pos_x, cur_pos_y + 1, cur_block):
                            _dock()
                        else:
                            last_drop_time = cur_drop_time
                            cur_pos_y += 1
            else:
                if start:
                    self.print_text(screen, font2,
                            (self.SCREEN_WIDTH - gameover_size[0]) // 2, (self.SCREEN_HEIGHT - gameover_size[1]) // 2,
                            'GAME OVER', self.RED)

            # 画当前下落方块
            self._draw_block(screen, cur_block, 0, 0, cur_pos_x, cur_pos_y)

            pygame.display.flip()


    # 画背景
    def _draw_background(self,screen):
        # 填充背景色
        screen.fill(self.BG_COLOR)
        # 画游戏区域分隔线
        pygame.draw.line(screen, self.BORDER_COLOR,
                        (self.SIZE * self.BLOCK_WIDTH + self.BORDER_WIDTH // 2, 0),
                        (self.SIZE * self.BLOCK_WIDTH + self.BORDER_WIDTH // 2, self.SCREEN_HEIGHT), self.BORDER_WIDTH)


    # 画网格线
    def _draw_gridlines(self, screen):
        # 画网格线 竖线
        for x in range(self.BLOCK_WIDTH):
            pygame.draw.line(screen, self.BLACK, (x * self.SIZE, 0), (x * self.SIZE, self.SCREEN_HEIGHT), 1)
        # 画网格线 横线
        for y in range(self.BLOCK_HEIGHT):
            pygame.draw.line(screen, self.BLACK, (0, y * self.SIZE), (self.BLOCK_WIDTH * self.SIZE, y * self.SIZE), 1)


    # 画已经落下的方块
    def _draw_game_area(self, screen, game_area):
        if game_area:
            for i, row in enumerate(game_area):
                for j, cell in enumerate(row):
                    if cell != '.':
                        pygame.draw.rect(screen, self.BLOCK_COLOR, (j * self.SIZE, i * self.SIZE, self.SIZE, self.SIZE), 0)


    # 画单个方块
    def _draw_block(self, screen, block, offset_x, offset_y, pos_x, pos_y):
        if block:
            for i in range(block.start_pos.Y, block.end_pos.Y + 1):
                for j in range(block.start_pos.X, block.end_pos.X + 1):
                    if block.template[i][j] != '.':
                        pygame.draw.rect(screen, self.BLOCK_COLOR,
                                        (offset_x + (pos_x + j) * self.SIZE, offset_y + (pos_y + i) * self.SIZE, self.SIZE, self.SIZE), 0)


    # 画得分等信息
    def _draw_info(self, screen, font, pos_x, font_height, score):
        self.print_text(screen, font, pos_x, 10, f'得分: ')
        self.print_text(screen, font, pos_x, 10 + font_height + 6, f'{score}')
        self.print_text(screen, font, pos_x, 20 + (font_height + 6) * 2, f'速度: ')
        self.print_text(screen, font, pos_x, 20 + (font_height + 6) * 3, f'{score // 10000}')
        self.print_text(screen, font, pos_x, 30 + (font_height + 6) * 4, f'下一个：')

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
        10: "ROTATE",
        11: "ACTION3"}

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
    tr = Tetris()
    tr.main()
