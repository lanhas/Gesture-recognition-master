import torch
from torch import nn
from pathlib import Path
from constants.keypoints import hand_bones, hand_bone_pairs


class StaticRecognitionModel(nn.Module):
    def __init__(self) -> None:
        super(StaticRecognitionModel, self).__init__()
        self.input_shape = len(hand_bones) + 2 * len(hand_bone_pairs) + 21
        self.ckpt_path = Path('checkpoints/static.pt')
        self.output_shape = 12
        self.hidden1 = nn.Linear(self.input_shape, 256)
        # self.drop = nn.Dropout(0.2)
        # self.bn1 = nn.BatchNorm1d(1)
        self.hidden2 = nn.Linear(256, 160)
        # self.bn2 = nn.BatchNorm1d(1)
        self.predict = nn.Linear(160, self.output_shape)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device, dtype=torch.float32)
        self.act = nn.ReLU()

    def save_ckpt(self):
        torch.save(self.state_dict(), self.ckpt_path)
        print('LSTM checkpoint saved')

    def load_ckpt(self, allow_new=True):
        if Path.is_file(self.ckpt_path):
            if torch.cuda.is_available():
                checkpoint = torch.load(self.ckpt_path)
            else:
                checkpoint = torch.load(self.ckpt_path, map_location=torch.device('cpu'))
            self.load_state_dict(checkpoint)
        else:
            if allow_new:
                print('LSTM ckpt not found.')
            else:
                raise FileNotFoundError('LSTM ckpt not found.')
    
    def forward(self, x):
        x = self.act(self.hidden1(x.contiguous().view(-1, self.input_shape)))
        # x = self.bn1(x)
        x = self.act(self.hidden2(x))
        # x = self.bn2(x)
        x = self.predict(x)
        
        return x


