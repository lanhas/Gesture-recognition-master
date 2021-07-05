import argparse
from pathlib import Path
import hgdataset.s1_skeleton
import train.train_dynamic_hands_model
import pred.prepare_skeleton_from_video
import pred.play_dynamic_hands_results
import pred.evaluation


def prepare_skeleton():
    pred.prepare_skeleton_from_video.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--train_hands', action='store_true',
                        help='Train hands recognition model from hands dataset')
    parser.add_argument('-c', '--clean_saved_skeleton', action='store_true',
                        help='Delete saved skeleton from generated/coords to regenerate them during next training')

    parser.add_argument('-a', '--play_keypoint', type=int,
                        help='Play keypoint estimation result')
    parser.add_argument('-b', '--play_hands', type=int,
                        help='Play hands recognition result')
    parser.add_argument('-p', '--play', type=str,
                        help='Assign a custom video path to play and recognize hands gestures')
    parser.add_argument('-r', '--play_realtime', action='store_true',
                        help='Open a camera and recognize hands on realtime')
    parser.add_argument('-e', '--eval', action='store_true',
                        help='Evaluate Edit Distance in test set')

    args = parser.parse_args()
    if args.train_hands:
        train.train_dynamic_hands_model.Trainer().train()
    elif args.clean_saved_skeleton:
        hgdataset.s1_skeleton.HgdSkeleton.remove_generated_skeletons()
    elif args.play_hands is not None:
        prepare_skeleton()
        pred.play_dynamic_hands_results.Player().play_dataset_video(is_train=False, video_index=args.play_hands)
    elif args.play is not None:
        video_path = args.play
        if not Path(video_path).is_file():
            raise FileNotFoundError(video_path, ' is not a file')
        pred.play_dynamic_hands_results.Player().play_custom_video(video_path)
    elif args.play_realtime:
        pred.play_dynamic_hands_results.Player().play_custom_video(None)
    elif args.eval:
        pred.evaluation.Eval().main()
