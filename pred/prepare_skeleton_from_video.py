from pathlib import Path
from torch.utils.data import DataLoader
from hgdataset.s1_skeleton import HgdSkeleton


def save():

    ds = HgdSkeleton(Path.home() / 'MeetingHands', is_train=True, resize_img_size=(512, 512))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
    for d in loader:
        pass

    ds = HgdSkeleton(Path.home() / 'MeetingHands', is_train=False, resize_img_size=(512, 512))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
    for d in loader:
        pass
