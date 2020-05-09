""" Convert dataset to HDF5
    This script preprocesses a dataset and saves it (images and labels) to 
    an HDF5 file for improved I/O. """
import os
import sys
from argparse import ArgumentParser
from tqdm import tqdm, trange
import h5py as h5

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import utils

def prepare_parser():
  usage = 'Parser for ImageNet HDF5 scripts.'
  parser = ArgumentParser(description=usage)
  parser.add_argument(
    '--dataset', type=str, default='train',
    help='Which dataset to convert to hdf5: train / valid')
  parser.add_argument(
    '--image_size', type=int, default=256,
    help='Image size')
  parser.add_argument(
    '--data_root', type=str, default='data',
    help='Default location where data is stored (default: %(default)s)')
  parser.add_argument(
    '--output_dir', type=str, default='./',
    help='Default location where hdf5 will be saved (default: %(default)s)')
  parser.add_argument(
    '--batch_size', type=int, default=256,
    help='Default overall batchsize (default: %(default)s)')
  parser.add_argument(
    '--num_workers', type=int, default=16,
    help='Number of dataloader workers (default: %(default)s)')
  parser.add_argument(
    '--chunk_size', type=int, default=500,
    help='Default overall batchsize (default: %(default)s)')
  parser.add_argument(
    '--compression', action='store_true', default=False,
    help='Use LZF compression? (default: %(default)s)')
  return parser


def run(config):
  if 'hdf5' in config['dataset']:
    raise ValueError('Reading from an HDF5 file which you will probably be '
                     'about to overwrite! Override this error only if you know '
                     'what you''re doing!')
  # Get image size
  # config['image_size'] = utils.imsize_dict[config['dataset']]

  # Update compression entry
  config['compression'] = 'lzf' if config['compression'] else None #No compression; can also use 'lzf' 

  # Get dataset
  kwargs = {'num_workers': config['num_workers'], 'pin_memory': False, 'drop_last': True}
  # train_loader = utils.get_data_loaders(dataset=config['dataset'],
  #                                       batch_size=config['batch_size'],
  #                                       shuffle=False,
  #                                       data_root=config['data_root'],
  #                                       use_multiepoch_sampler=False,
  #                                       **kwargs)[0]    

  # Create WT filter
  filters = utils.create_filters(device='cpu')

  # Create transforms
  train_transform = transforms.Compose([utils.CenterCropLongEdge(), 
                                        transforms.Resize(config['image_size']), 
                                        transforms.ToTensor(), 
                                        utils.Apply2WT64(filters)])

  train_dataset = ImageFolder(root=config['data_root'], transform=train_transform)
  train_loader = DataLoader(train_dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            **kwargs)

  # HDF5 supports chunking and compression. You may want to experiment 
  # with different chunk sizes to see how it runs on your machines.
  # Chunk Size/compression     Read speed @ 256x256   Read speed @ 128x128  Filesize @ 128x128    Time to write @128x128
  # 1 / None                   20/s
  # 500 / None                 ramps up to 77/s       102/s                 61GB                  23min
  # 500 / LZF                                         8/s                   56GB                  23min
  # 1000 / None                78/s
  # 5000 / None                81/s
  # auto:(125,1,16,32) / None                         11/s                  61GB        

  print('Starting to load %s into an HDF5 file with chunk size %i and compression %s...' % (config['dataset'], config['chunk_size'], config['compression']))
  # Loop over train loader
  for i,(x,y) in enumerate(tqdm(train_loader)):
    x = x.byte().numpy()
    # Numpyify y
    y = y.numpy()
    # If we're on the first batch, prepare the hdf5
    if i==0:
      with h5.File(config['output_dir'] + '/ILSVRC%i.hdf5' % config['image_size'], 'w') as f:
        print('Producing dataset of len %d' % len(train_loader.dataset))
        imgs_dset = f.create_dataset('imgs', x.shape,dtype='uint8', maxshape=(len(train_loader.dataset), 3, config['image_size'], config['image_size']),
                                     chunks=(config['chunk_size'], 3, config['image_size'], config['image_size']), compression=config['compression']) 
        print('Image chunks chosen as ' + str(imgs_dset.chunks))
        imgs_dset[...] = x
        labels_dset = f.create_dataset('labels', y.shape, dtype='int64', maxshape=(len(train_loader.dataset),), chunks=(config['chunk_size'],), compression=config['compression'])
        print('Label chunks chosen as ' + str(labels_dset.chunks))
        labels_dset[...] = y
    # Else append to the hdf5
    else:
      with h5.File(config['output_dir'] + '/ILSVRC%i.hdf5' % config['image_size'], 'a') as f:
        f['imgs'].resize(f['imgs'].shape[0] + x.shape[0], axis=0)
        f['imgs'][-x.shape[0]:] = x
        f['labels'].resize(f['labels'].shape[0] + y.shape[0], axis=0)
        f['labels'][-y.shape[0]:] = y


def main():
  # parse command line and run    
  parser = prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)

if __name__ == '__main__':    
  main()