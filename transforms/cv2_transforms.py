# Used to replace TorchVision transforms, which mostly use PIL exclusively

import cv2
import numpy as np

import collections

class Resize(object):
    """Resize the input numpy array to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR``
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = tuple(size)
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (numpy array): Image to be scaled.

        Returns:
            Numpy array: Rescaled image.
        """
        resized = cv2.resize(img, self.size, interpolation=self.interpolation)
        if len(img.shape) == 3 and img.shape[2] == 1:  # since resize removes extra axis
        	resized = np.expand_dims(resized,axis=-1)  # add extra channel so that (H,W,1)
        return resized

    # def __repr__(self):
    #     interpolate_str = _pil_interpolation_to_str[self.interpolation]
    #     return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


