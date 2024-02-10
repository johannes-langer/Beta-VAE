"""utils.py"""

import argparse
import subprocess

import torch
import torch.nn as nn
from torch.autograd import Variable


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor

def mps(tensor, uses_mps):
    '''
    @Jona rebuilt cuda() function by example for metal support.
    '''
    return tensor.mps() if uses_mps else tensor

def device(tensor, uses_cuda, uses_mps):
    '''
    @Jona built this funcion to replace the cuda() function in all other applications.
    '''
    if uses_cuda:
        return tensor.cuda()
    elif uses_mps:
        return tensor.to('mps')
    else:
        return tensor


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def where(cond, x, y):
    """Do same operation as np.where

    code from:
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)


def grid2gif(image_str, output_gif, delay=100):
    """Make GIF from images.

    code from:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/34555939#34555939
    """
    str1 = 'convert -delay '+str(delay)+' -loop 0 ' + image_str  + ' ' + output_gif
    subprocess.call(str1, shell=True)
