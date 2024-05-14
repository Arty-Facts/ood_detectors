import torch
from PIL import Image
from pathlib import Path
import os


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(root, flist):
    """flist format: impath label\nimpath label\n."""
    imlist = []
    if flist == "all":
        for path in Path(root).rglob('*'):
            if path.is_file() and path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                imlist.append((str(path), 0))
    else:
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                data = line.strip().rsplit(maxsplit=1)
                if len(data) == 2:
                    impath, imlabel = data
                else:
                    impath, imlabel = data[0], 0
                path = os.path.join(root, impath)
                # if not os.path.isfile(path):
                #     raise FileNotFoundError(f"Image file not found: {path}")
                imlist.append((str(path), int(imlabel)))
    return imlist


class ImageFilelist(torch.utils.data.Dataset):

    def __init__(self,
                 root,
                 flist,
                 transform=None,
                 target_transform=None,
                 flist_reader=default_flist_reader,
                 loader=default_loader):
        self.root = root
        self.imlist = flist_reader(root, flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)
