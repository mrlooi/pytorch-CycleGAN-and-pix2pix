from .pix2pix_model import Pix2PixModel
from collections import OrderedDict
from sklearn.preprocessing import normalize
import numpy as np
import torchvision.transforms as ttransforms
import cv2

class Pix2Depth2Model(Pix2PixModel):
    def name(self):
        return 'Pix2Depth2Model'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')

        parser.set_defaults(which_model_netG='unet_256')
        # parser.set_defaults(which_model_netG='unet_512')
        
        parser.set_defaults(dataset_mode='image_depth2')
        parser.set_defaults(input_nc=3) 
        parser.set_defaults(output_nc=3) 
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def set_input(self, input):
        self.real_A = input['A'].to(self.device) # img
        self.real_B = input['B'].to(self.device) # depth, 1 channel
        self.image_paths = input['A_paths']

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        # self.visual_names = ['real_A', 'fake_B', 'real_B']
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                image_tensor = getattr(self, name)
                # if name.endswith("_B"):  # depth tensors, normalize them for visualization purposes
                #     B = image_tensor.cpu().detach().float().numpy()
                #     # B_t = np.transpose(B, axes=[1,2,0]).squeeze()
                #     B_norm = normalize(B.squeeze(), norm='max')
                #     B_norm *= 255
                #     B_norm = np.expand_dims(B_norm, axis=0)
                #     image_tensor = B_norm.astype(np.uint8)
                image_data = image_tensor.cpu().detach().float().numpy().squeeze()  # B,3,H,W ->  3,H,W
                image_data = np.transpose(image_data, [1,2,0])  # 3,H,W -> H,W,3
                if name.endswith("_A"):
                    image_data = (image_data + 1) / 2
                # if name.endswith("real_B"):
                #     print(image_data)
                image_data *= 255
                image_data = image_data.astype(np.uint8)
                # print(image_data)
                visual_ret[name] = image_data
        return visual_ret