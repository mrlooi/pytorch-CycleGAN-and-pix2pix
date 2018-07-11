import os.path
from data.base_dataset import BaseDataset
import random
import numpy as np
import scipy.io as sio
from transforms import cv2_transforms
import torch
import torchvision.transforms as ttransforms
import cv2

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)

    return images

def get_transform(opt, normalize=True):
    transform_list = []

    osize = [opt.loadSize, opt.loadSize]
    transform_list.append(cv2_transforms.Resize(osize, cv2.INTER_CUBIC))
    transform_list.append(ttransforms.ToTensor())
    if normalize:
        transform_list.append(ttransforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))

    return ttransforms.Compose(transform_list)

def read_depth(depth_file):
    assert depth_file.endswith(".mat")
    mat = sio.loadmat(depth_file)
    assert 'depth' in mat
    depth_data = mat['depth']
    if len(depth_data.shape) == 2:
        depth_data = np.expand_dims(depth_data,axis=-1)  # add extra channel so that (H,W,1)
    return depth_data

def read_img(img_file):
    img_data = cv2.imread(img_file) 
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY) # convert to grayscale 
    img_data = np.expand_dims(img_data,axis=-1)  # (H,W,1)
    return img_data

class ImageDepthDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt

        # self.opt.input_nc = 1 # grayscale
        # self.opt.output_nc = 1 # depth

        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, 'rgb/' + opt.phase)
        self.dir_B = os.path.join(opt.dataroot, 'depth/' + opt.phase)

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        assert self.A_size == self.B_size

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)

        self.A_transform = get_transform(opt, normalize=False)
        self.B_transform = get_transform(opt, normalize=False)


    def __getitem__(self, index):
        idx = index % self.A_size
        if not self.opt.serial_batches:
            idx = random.randint(0, self.A_size - 1) 
        A_path = self.A_paths[idx]
        B_path = self.B_paths[idx]
        A_img = read_img(A_path) 
        B_img = read_depth(B_path)

        A = self.A_transform(A_img)
        B = self.B_transform(B_img)

        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        # A = ttransforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        A = ttransforms.Normalize((0.5,), (0.5,))(A)

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'ImageDepthDataset'


if __name__ == '__main__':
    from sklearn.preprocessing import normalize

    class Opt():
        def __init__(self):
            self.dataroot = "/home/vincent/Documents/deep_learning/pytorch-CycleGAN-and-pix2pix/datasets/nyu_v2"
            self.phase = "train"
            self.resize_or_crop = 'resize_and_crop'
            self.loadSize = 286
            self.fineSize = 256
            self.serial_batches = False
            self.no_flip = False
            self.isTrain = True

    opt = Opt()

    idd = ImageDepthDataset()
    idd.initialize(opt)

    # A_path = idd.A_paths[0]
    # B_path = idd.B_paths[0]
    # A_img = cv2.imread(A_path) # Image.open(A_path).convert('RGB')
    # B_img = read_depth(B_path)
    
    # A = idd.A_transform(A_img)
    # B = idd.B_transform(B_img)

    for i in range(10):
        data = idd[i]
        A = data['A']
        B = data['B']
        A_path = data['A_paths']
        B_path = data['B_paths']

        print("Showing %s"%(A_path))

        A_img = cv2.imread(A_path)
        B_img = read_depth(B_path)

        cv2.imshow("A img", A_img)
        B_img = normalize(B_img.squeeze(), norm='max')
        cv2.imshow("B img", cv2.cvtColor(B_img, cv2.COLOR_GRAY2BGR))
        cv2.imshow("B img", B_img)
        
        A_t = np.transpose(A.numpy(), axes=[1,2,0])
        cv2.imshow("A", A_t)
        # cv2.imshow("A", tensor2im(A))

        B_t = np.transpose(B.numpy(), axes=[1,2,0]).squeeze()
        B_norm = normalize(B_t, norm='max')
        cv2.imshow("B", B_norm)

        # B = ttransforms.Normalize((0.5,), (0.5,))(B)
        # B = np.transpose(B.numpy(), axes=[1,2,0])
        # cv2.imshow("BB", B)

        cv2.waitKey(0)
