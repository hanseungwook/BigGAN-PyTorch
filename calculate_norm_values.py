''' Calculate normalization values for dataset (WT)
 This script iterates over the dataset and calculates the normalization values
 (minimum, maximum, shift, scale).
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import inception_utils
from tqdm import tqdm, trange
from argparse import ArgumentParser
import utils

def prepare_parser():
  usage = 'Calculate and store inception metrics.'
  parser = ArgumentParser(description=usage)
  parser.add_argument(
    '--dataset', type=str, default='I128_hdf5',
    help='Which Dataset to train on, out of I128, I256, C10, C100...'
         'Append _hdf5 to use the hdf5 version of the dataset. (default: %(default)s)')
  parser.add_argument(
    '--data_root', type=str, default='data',
    help='Default location where data is stored (default: %(default)s)') 
  parser.add_argument(
    '--batch_size', type=int, default=64,
    help='Default overall batchsize (default: %(default)s)')
  parser.add_argument(
    '--parallel', action='store_true', default=False,
    help='Train with multiple GPUs (default: %(default)s)')
  parser.add_argument(
    '--augment', action='store_true', default=False,
    help='Augment with random crops and flips (default: %(default)s)')
  parser.add_argument(
    '--num_workers', type=int, default=8,
    help='Number of dataloader workers (default: %(default)s)')
  parser.add_argument(
    '--shuffle', action='store_true', default=False,
    help='Shuffle the data? (default: %(default)s)') 
  parser.add_argument(
    '--seed', type=int, default=0,
    help='Random seed to use.')
  return parser

def run(config):
  # Get loader
  config['drop_last'] = False
  loaders = utils.get_data_loaders(**config)

  device = 'cuda'
  filters = utils.create_filters(device=device)
  min_val = float('inf')
  max_val = float('-inf')
  for i, (x, y) in enumerate(tqdm(loaders[0])):
    x = utils.wt(x.to(device), filters, levels=2)[:, :, :64, :64]
    min_val = min(min_val, torch.min(x))
    max_val = max(max_val, torch.max(x))

  shift = torch.ceil(torch.abs(min_val))
  scale = shift + torch.ceil(max_val)

  print('Minimum: {} \t Maximum: {}\nShift: {} \t Scale: {}'.format(min_val, max_val, shift, scale))
  
  print('Saving normalization values to disk...')
  np.savez(config['dataset'].strip('_hdf5')+'_norm_values.npz', **{'min' : min_val, 'max' : max_val, 'shift': shift, 'scale': scale})

def main():
  # parse command line    
  parser = prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)


if __name__ == '__main__':    
    main()
