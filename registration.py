
from constants import USABLE_SAMPLES
import pandas as pd
import os
import subprocess
import util
import numpy as np
import scipy.ndimage as sp

samples = pd.read_csv('slices.csv', index_col = 0)

labelled_frames = {}

def get_frame_num(s):
  s = str(s)
  if len(s) == 1:
    return '000' + s
  elif len(s) == 2:
    return '00' + s
  else:
    return '0' + s

for s in USABLE_SAMPLES:
  labelled_frames[s] = get_frame_num(int(samples.loc[s]['labeled frame']))

g_truth_dir = '../data/labels'
registration_dir = './registered/'
data_dir = '../data/raw/raw/'
seg_dir = '../data/data'
command = '../ITK_4D/build/bin/TemporalRegistration_Rigid'

def union(m1, m2):
  return np.ma.mask_or(m1, m2)

def dilate(mask, iterations=1):
  struct = sp.generate_binary_structure(2, 1)
  return sp.binary_dilation(mask, structure=struct, iterations=iterations)

def register(sample):
  g_truth_frame = labelled_frames[sample]
  all_frames = os.listdir(os.path.join(data_dir, sample))

  for i in range(len(all_frames)):
    all_frames[i] = os.path.join(os.path.join(data_dir, sample), all_frames[i])

  mask = '{}/{}/{}_{}_all_brains.nii.gz'.format(g_truth_dir, sample, sample, g_truth_frame)
  commands = [command, registration_dir, len(all_frames), mask, '1', '0']
  back_commands = commands.copy()
  prev_frames = all_frames[:int(g_truth_frame)]
  prev_frames.reverse()
  back_commands.extend(prev_frames)
  commands.extend(all_frames[int(g_truth_frame):])
  back_commands[2] = str(int(g_truth_frame))
  commands[2] = str(len(all_frames) - int(g_truth_frame))
  # print(commands)
  # print(back_commands)
  subprocess.run(back_commands)
  subprocess.run(commands)

def transform(image, q):  
  translation = np.array([[1, 0, 0, q[0]], 
                          [0, 1, 0, q[1]], 
                          [0, 0, 1, q[2]], 
                          [0, 0, 0, 1]])
  
  r_x = np.array([[1, 0, 0, 0], 
                  [0, np.cos(q[3]), np.sin(q[3]), 0], 
                  [0, -np.sin(q[3]), np.cos(q[3]), 0],
                  [0, 0, 0, 1]])

  r_y = np.array([[ np.cos(q[4]), 0, np.sin(q[4]), 0], 
                  [0, 1 , 0, 0], 
                  [-np.sin(q[4]), 0, np.cos(q[4]), 0],
                  [0, 0, 0, 1]])

  r_z = np.array([[ np.cos(q[5]),  np.sin(q[5]), 0, 0], 
                  [-np.sin(q[4]), np.cos(q[4]) , 0, 0], 
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

  M = translation@r_x@r_y@r_z
  return M@image

def register_by_frame(sample):
  g_truth_frame = labelled_frames[sample]
  all_frames = os.listdir(os.path.join(data_dir, sample))

  # for i in range(len(all_frames)):
  #   all_frames[i] = os.path.join(os.path.join(data_dir, sample), all_frames[i])

  mask = '{}/{}/{}_{}_all_brains.nii.gz'.format(g_truth_dir, sample, sample, g_truth_frame)
  commands = [command, registration_dir + '/byframe', '2' , 'mask', '1', '0']
  i = g_truth_frame
  q = np.ones(6)
  # forward pass
  while i < len(all_frames):
    frame = util.read_vol(data_dir + sample + '/' + all_frames[i])
    transformed = transform(frame, q)
    mask = util.read_vol(seg_dir + all_frames[i])
    mask_trans = transform(mask, q)
    commands[3] = mask_trans
    commands[6] = transformed
    commands[7] = data_dir + sample + '/' + all_frames[i+1]
    output = subprocess.run(commands, capture_output=True)
    q = output.stdout
    i += 1


labelled = ['010918L',  '013018L', '013118S',  '021218S',  '022415',  '031317L',  '031615',  '032217',   '032318c',  '040218',  '040716',	'041818',	'050318L', '051215',   '052218S',  '052516',  '080117'
,'010918S',  '013018S',  '021015',   '022318L',  '022618',  '031317T',  '031616',  '032318a',  '032318d',  '040417',  '041017',	'043015',	'050318S', '051718L',  '052418L',  '061715',  '083115',
'012115',	 '013118L',  '021218L',  '022318S',  '030315',  '031516',   '031716',  '032318b',  '032818',   '040617',  '041318S',	'043018',	'050917', '052218L',  '052418S',  '062515',  '102617']

# for l in labelled:
#   register(l)

register(labelled[0])