from __future__ import absolute_import
import sys
sys.path.append('./')

import argparse
import os
import os.path as osp
import numpy as np
import math
import time
from PIL import Image, ImageFile

import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from config import get_args
from lib import datasets, evaluation_metrics, models
from lib.models.model_builder import ModelBuilder
from lib.datasets.dataset import LmdbDataset, AlignCollate
from lib.loss import SequenceCrossEntropyLoss
from lib.trainers import Trainer
from lib.evaluators import Evaluator
from lib.utils.logging import Logger, TFLogger
from lib.utils.serialization import load_checkpoint, save_checkpoint
from lib.utils.osutils import make_symlink_if_not_exists
from lib.evaluation_metrics.metrics import get_str_list
from lib.utils.labelmaps import get_vocabulary, labels2strs

global_args = get_args(sys.argv[1:])

def image_process(img, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
  if keep_ratio:
    w, h = img.size
    ratio = w / float(h)
    imgW = int(np.floor(ratio * imgH))
    imgW = max(imgH * min_ratio, imgW)

  img = img.resize((imgW, imgH), Image.BILINEAR)
  img = transforms.ToTensor()(img)
  img.sub_(0.5).div_(0.5)

  return img

class DataInfo(object):
  """
  Save the info about the dataset.
  This a code snippet from dataset.py
  """
  def __init__(self, voc_type):
    super(DataInfo, self).__init__()
    self.voc_type = voc_type

    assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    self.EOS = 'EOS'
    self.PADDING = 'PADDING'
    self.UNKNOWN = 'UNKNOWN'
    self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
    self.char2id = dict(zip(self.voc, range(len(self.voc))))
    self.id2char = dict(zip(range(len(self.voc)), self.voc))

    self.rec_num_classes = len(self.voc)


def recognizer(img_path, gt_path, model, device, dataset_info,
               savedir="outputs/", only_price=False):
    gt_file = open(gt_path, "r")

    save_filename = os.path.basename(gt_path)
    save_path = os.path.join(savedir, save_filename)

    if not os.path.isdir(savedir):
      os.mkdir(savedir)

    fp = open(save_path, "w")

    bounding_boxes = [[int(coord) if coord.isnumeric() else coord
                       for coord in bbox.split('\n')[0].split(',', 8)]
                      for bbox in gt_file.readlines()]

    img = Image.open(img_path)
    img_width, img_height = img.width, img.height

    for bbox in bounding_boxes:
        label = bbox[8]

        if only_price and not type(label) == int and len(label) == 1:
            continue

        x1, y1 = bbox[0], bbox[1]
        x2, y2 = bbox[2], bbox[3]
        x3, y3 = bbox[4], bbox[5]
        x4, y4 = bbox[6], bbox[7]
        
        cropped_img = img.crop((x1, y3, x2, y1))
            

        cropped_img = image_process(cropped_img.convert('RGB'))

        with torch.no_grad():
            cropped_img = cropped_img.to(device)

        input_dict = {}
        input_dict['images'] = cropped_img.unsqueeze(0)

        rec_targets = torch.IntTensor(1, args.max_len).fill_(1)
        rec_targets[:, args.max_len - 1] = dataset_info.char2id[
          dataset_info.EOS]

        input_dict['rec_targets'] = rec_targets
        input_dict['rec_lengths'] = [args.max_len]

        output_dict = model(input_dict)
        pred_rec = output_dict['output']['pred_rec']
        pred_str, _ = get_str_list(pred_rec, input_dict['rec_targets'],
                                   dataset=dataset_info)

        pred_label = pred_str[0]
        print('Recognition result: {0}'.format(pred_label))

        txt_line = "{},{},{},{},{},{},{},{},{}\n".format(x1, y1, x2, y2,
                                                         x3, y3, x4, y4,
                                                         pred_label)
        fp.write(txt_line)
    fp.close()


def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True

  args.cuda = args.cuda and torch.cuda.is_available()
  if args.cuda:
    print('using cuda.')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
  else:
    torch.set_default_tensor_type('torch.FloatTensor')
  
  # Create data loaders
  if args.height is None or args.width is None:
    args.height, args.width = (32, 100)

  dataset_info = DataInfo(args.voc_type)

  # Create model
  model = ModelBuilder(arch=args.arch, rec_num_classes=dataset_info.rec_num_classes,
                       sDim=args.decoder_sdim, attDim=args.attDim, max_len_labels=args.max_len,
                       eos=dataset_info.char2id[dataset_info.EOS], STN_ON=args.STN_ON)

  # Load from checkpoint
  if args.resume:
    checkpoint = load_checkpoint(args.resume)
    model.load_state_dict(checkpoint['state_dict'])

  if args.cuda:
    device = torch.device("cuda")
    model = model.to(device)
    model = nn.DataParallel(model)

  # Evaluation
  model.eval()
  images_path = args.images_path
  box_path = args.box_path
  imgs = os.listdir(images_path)

  for img in imgs:
      image_path = os.path.join(images_path, img)

      print("Image path:", image_path)

      gt_name = img.replace('jpg', 'txt')
      gt_path = os.path.join(box_path, gt_name)

      recognizer(image_path, gt_path, model, device, dataset_info,
       savedir="outputs/", only_price=False)


if __name__ == '__main__':
  # parse the config
  args = get_args(sys.argv[1:])
  main(args)
