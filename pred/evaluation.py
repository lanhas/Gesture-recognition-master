from pathlib import Path
import datetime
import numpy as np
from itertools import groupby
from constants.enum_keys import HG
from hgdataset.s0_label import HgdLabel
from pred.play_dynamic_hands_results import Player as DynamicPlayer
from pred.play_static_hands_results import Player as StaticPlayer


class Eval:
    def __init__(self, mode):
        self.mode = mode
        if self.mode == 'dynamic':
            self.player = DynamicPlayer()
        else:
            self.player = StaticPlayer()
        models_dir = Path.cwd() / 'checkpoints'
        self.models_path = models_dir.iterdir()
        self.num_video = len(HgdLabel(self.mode, Path.home() / 'MeetingHands' , is_train=False))
        self.ed = EditDistance()

    def main(self):
        file_handle = open('./docs/train_log/trainLog.txt', mode='a+')
        file_handle.writelines([
        '*---------------------------------------------------------------------------------------------------------------------------------*\n',
        '*--------------------------------------------------------- 训练日志 ---------------------------------------------------------------*\n',
        '训练日期： ' + str(datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')) +'\n',
            ])
        for idx, val in enumerate(self.models_path):
            print('*------------------------------------------------*')
            print('The prediction of model %s' %(val.name))
            file_handle.writelines([
            '*------------------------------------------------*\n'
            'The prediction of model %s\n' %(val.name),
            ])
            self.player.hpred.h_model.ckpt_path = Path(val)
            self.player.hpred.h_model.load_ckpt()
            self.sumDistence = [0, 0, 0, 0]
            for n in range(self.num_video):
                res = self.player.play_dataset_video(is_train=False, video_index=n, show=False)
                target = res[HG.GESTURE_LABEL]
                source = res[HG.PRED_GESTURE]
                (N, S, D, I), accuracy, self.sumDistence = self.ed.getAccuracy(source, target, self.sumDistence)
                print('N:%d, S:%d, D:%d, I:%d, accuracy:%.2f' %(N, S, D, I, accuracy))
                file_handle.writelines(['N:%d, S:%d, D:%d, I:%d, accuracy:%.2f\n' %(N, S, D, I, accuracy)])
                pass
            sum_accuracy = (self.sumDistence[0] - self.sumDistence[1] - self.sumDistence[2] - self.sumDistence[3]) / self.sumDistence[0]
            print("The predict of sum is completed")
            print('N:%d, S:%d, D:%d, I:%d, accuracy:%.2f' %(self.sumDistence[0], self.sumDistence[1], self.sumDistence[2], self.sumDistence[3], sum_accuracy))
            file_handle.writelines([
            "The predict of sum is completed\n",
            'N:%d, S:%d, D:%d, I:%d, accuracy:%.2f\n' %(self.sumDistence[0], self.sumDistence[1], self.sumDistence[2], self.sumDistence[3], sum_accuracy)
            ])


class EditDistance:
    # Order for edit distance: S,D,I
    def edit_distance(self, word1, word2):
        # Wrapper for __edit_distance with empty stack
        word1 = tuple(word1)
        word2 = tuple(word2)
        # target gesture num N
        N = len(list(filter(lambda x: x % 100 != 0, word2)))
        # tuple contains (S,D,I) for times of substitute, delete, insert
        return N, self.__edit_distance(word1, word2, {})

    def getAccuracy(self, source: list, target: list, sumDistence: list):
        assert len(source) == len(target)
        source_group = [k for k, g in groupby(source)]
        target_group = [k for k, g in groupby(target)]
        N, (S, D, I) = self.edit_distance(source_group, target_group)
        sumDistence[0]+= N
        sumDistence[1]+= S
        sumDistence[2]+= D
        sumDistence[3]+= I
        accuracy = (N - S - D - I) / N
        return (N, S, D, I), accuracy, sumDistence

    def __edit_distance(self, word1, word2, computed_solutions):
        # computed_solutions: dict{ tuple of word(w1,w2), tuple of integer(S,D,I)}
        if len(word1) == 0:
            return 0, 0, len(word2)  # Insert into word1
        if len(word2) == 0:
            return 0, len(word1), 0  # Delete from word1

        replace_tuple = (word1[1:], word2[1:])
        delete_tuple = (word1[1:], word2)
        insert_tuple = (word1, word2[1:])

        replace_dist = self.__distance_add(self.__replace_cost(word1, word2), self.__transformation_cost(replace_tuple, computed_solutions))
        delete_dist = self.__distance_add((0, 1, 0), self.__transformation_cost(delete_tuple, computed_solutions))
        insert_dist = self.__distance_add((0, 0, 1), self.__transformation_cost(insert_tuple, computed_solutions))

        min_dist = self.__distance_min(replace_dist, delete_dist, insert_dist)
        return min_dist

    def __replace_cost(self, word1, word2):
        """
        Cost of replacing 1st char of word1 to word2.
        :param word1:
        :param word2:
        :return: distance S,D,I
        """
        if word1[0] == word2[0]:
            return 0,0,0  # 1st chars are equal in A and B
        else:
            return 1,0,0  # Substitute 1st char in A to B

    def __transformation_cost(self, problem_tuple, solutions):
        """
        Use solution if case "tuple" was already solved, else solve the "tuple" and save to solutions
        :param problem_tuple:
        :param solutions:
        :return:
        """
        if problem_tuple in solutions:  # Already solved
            return solutions.get(problem_tuple)
        else:
            distSDI = self.__edit_distance(problem_tuple[0], problem_tuple[1], solutions)  # Compute and add to solutions
            solutions[problem_tuple] = distSDI
            return distSDI

    def __distance_add(self, dis1, dis2):
        """
        Add S,D,I separately from distance
        :param dis1:
        :param dis2:
        :return:
        """

        S = dis1[0] + dis2[0]
        D = dis1[1] + dis2[1]
        I = dis1[2] + dis2[2]
        return (S,D,I)

    def __distance_min(self, dis1, dis2, dis3):
        """
        Find minimum distance from distance tuple
        :param dis1:
        :param dis2:
        :return:
        """
        d1_total = dis1[0] + dis1[1] + dis1[2]
        d2_total = dis2[0] + dis2[1] + dis2[2]
        d3_total = dis3[0] + dis3[1] + dis3[2]
        arr123 = np.array([d1_total, d2_total, d3_total], np.int32)
        argminimum = int(np.argmin(arr123))
        if argminimum == 0:
            return dis1
        elif argminimum == 1:
            return dis2
        elif argminimum == 2:
            return dis3
        else:
            raise ValueError()
