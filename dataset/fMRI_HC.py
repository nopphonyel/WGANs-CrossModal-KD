import os
import pickle
from torch.nn import Sequential
from torch.utils.data import Dataset, DataLoader


class fMRI_HC_Dataset(Dataset):
    __DAT_PATH = 'content/DSC_2018.00114_120_v1'
    __SUB_LIST = ('s1', 's2', 's3')
    __VER_LIST = ('v1', 'v2')
    __IMG_SIZE = 64
    __DIR_NAME = os.path.dirname(__file__)
    __dev = 'cpu'

    def __init__(self, p_id: int, v: int = 1, train: bool = True, img_tf: Sequential = None):
        """
        :param p_id: Participant ID (range 1,3)
        :param v: Version (only V1 and V2 available)
        :param train: Using train data or not
        :param img_tf: image transform sequence
        """
        f_name = "fmriHC_train.dat" if train is True else "fmriHC_eval.dat"
        ld_path = os.path.join(
            fMRI_HC_Dataset.__DIR_NAME,
            fMRI_HC_Dataset.__DAT_PATH,
            f_name
        )
        self.dat = pickle.load(open(ld_path, "rb"))
        self.img_tf = img_tf

        if p_id < 1 or p_id > 3:
            raise IndexError("Participant ID not exist")
        else:
            self.p_id = fMRI_HC_Dataset.__SUB_LIST[p_id - 1]

        if v != 1 and v != 2:
            raise IndexError("Version not exist")
        else:
            self.v = fMRI_HC_Dataset.__VER_LIST[v - 1]

    def to(self, dev):
        self.__dev = dev
        return self

    def __len__(self):
        return self.dat[self.p_id][self.v].shape[0]

    def __getitem__(self, idx):
        fmri = self.dat[self.p_id][self.v][idx, :]
        img = self.dat['y'][idx, :, :, :]
        if self.img_tf is not None:
            img = self.img_tf(img)
        label = self.dat['l'][idx]
        return fmri.to(self.__dev), img.to(self.__dev), label.to(self.__dev)

    @staticmethod
    def get_name():
        return "fMRI_HC_Dataset"

    def get_num_classes(self):
        return 6


# # Testing fragment
# import matplotlib.pyplot as plt
# import torchvision.transforms as T
#
# if __name__ == '__main__':
#     img_tf = Sequential(
#         T.Resize((64, 64))
#     )
#     ds = fMRI_HC_Dataset(p_id=3, v=1, train=True, img_tf=img_tf)
#     ld = DataLoader(ds, batch_size=2)
#     for idx, (fmri, img, label) in enumerate(ld):
#         if idx == 16:
#             print(fmri.shape, img.shape, label.shape)
#             print(label[0])
#             plt.imshow(img[0].squeeze(0))
#             plt.show()
