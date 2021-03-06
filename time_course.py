import util
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from constants import GOOD_FRAMES
# import eval_timecourse as et

usable = ['010918L', '010918S', '012115', '013018L', '013018S', '013118L',
          '021218L', '021218S', '022318L', '022318S', '030315', '031317T',
          '031615', '031716', '032217', '032318a', '032318b', '032318d',
          '040218', '040617', '040716', '041017', '041318S', 
          '043018', '050318S', '051718L', '052418L', '052418S', '053017',
          '053117L', '053117S', '060118', '061217', '062117', '062817L', 
          '062817S', '071218', '071717L', '071717S', '072017L', '080217',
           '081116', '081315', '082117L', '082517L', '082517S', '082917a',
           '082917b', '083017L', '083017S', '090517L', '090517S', '090817a', 
           '090817b', '091917L', '091917S', '092117S', '092817L', '092817S', 
           '100317L', '101317', '102717', '110214', '110217L', '110217S', 
           '111017L', '111017S', '112614', '120717', '122115']

greater_95 = ['021218L', '021218S', '030315', '032318a',
              '032318b', '041318S', '043018', '050318S', '051718L', '052418S',
              '060118', '061217', '062817S', 
              '071717L', '081315', '082117L', '082517L', '102717', '110217S',
              '111017S', '112614']

O2_TIME = {'021218L': [5, 5], '021218S': [5, 5], '030315': [10, 10, 10], '032318a': [5, 5], 
            '032318b': [5, 5], '041318S': [5, 5], '043018': [5, 10], '050318S': [5, 5],
            '051718L': [5, 5], '052418S': [5, 5], '060118': [10, 10, 10], '061217': [10, 10],
            '062817S': [5, 5], '071717L': [5, 5], '081315': [10, 10, 10], '082117L': [5, 5], 
            '082517L': [5, 5], '102717': [10, 10], '110217S': [5, 5], '111017S': [5, 5],
            '112164': [10, 10, 10]}


def get_frame_num(s):
  s = str(s)
  if len(s) == 1:
    return '000' + s
  elif len(s) == 2:
    return '00' + s
  else:
    return '0' + s

def timecourse_avg(sample):
  good_frames = GOOD_FRAMES[sample]
  frames = os.listdir('../data/raw/raw/{}'.format(sample))
  signal_all = np.zeros(len(frames))
  signal_good = np.zeros(len(frames))
  for i in range(len(frames)):
    frame = frames[i]
    predict = util.read_vol('../data/data/{}'.format(frame))
    scan = util.read_vol('../data/raw/raw/{}/{}'.format(sample, frame))
    brain = scan[np.nonzero(predict)]
    avg_signal = np.mean(brain)
    signal_all[i] = avg_signal
    if i in good_frames: 
      signal_good[i] = avg_signal

  fig = plt.figure()
  plt.plot(signal_good)
  plt.plot(signal_all)
  plt.legend(['good frames', 'all frames'])
  savefile = './avg_plots/less90/{}.png'.format(sample)
  plt.savefig(savefile)
  plt.close(fig)

def get_O2_frames(sample, num_frames):
  timestamps = O2_TIME[sample]
  proportion = np.array(timestamps) / np.sum(timestamps)
  frames = proportion * num_frames
  return np.cumsum(frames)

def timecourse_interpolated(sample):
  good_frames = GOOD_FRAMES[sample]
  frames = os.listdir('../data/raw/{}'.format(sample))
  fpm = len(frames) / sum(O2_TIME[sample])
  signal_good = np.empty(len(frames))
  signal_good[:] = np.nan
  for frame in good_frames:
    predict = util.read_vol('../data/data/{}_{}.nii.gz'.format(sample, get_frame_num(frame)))
    scan = util.read_vol('../data/raw/{}/{}_{}.nii.gz'.format(sample, sample, get_frame_num(frame)))
    brain = scan[np.nonzero(predict)]
    avg_signal = np.mean(brain)
    signal_good[frame] = avg_signal

  points = np.argwhere(np.isnan(signal_good))
  interpolate = np.zeros(points.shape)
  for i in range(len(points)):
    p = points[i]
    try:
      next_val = signal_good[p+1]
      j = 1
      while np.isnan(next_val) and p+j < len(signal_good):
        j += 1
        next_val = signal_good[p+j]
      avg_endpoints = np.mean([signal_good[p-1], next_val])
    except IndexError:
      avg_endpoints = signal_good[p-1]
    interpolate[i] = avg_endpoints
    signal_good[p] = avg_endpoints
  
  O2 = get_O2_frames(sample, len(frames))

  plt.figure()
  plt.plot(signal_good)
  plt.plot(points, interpolate, 'o')
  colors = ['r', 'g']
  for i in range(len(O2)):
    color = colors[i%2]
    plt.axvline(x=O2[i], color=color)
  # savefile = './interpolated/greater95/{}.png'.format(sample)
  # plt.savefig(savefile)
  plt.show()
  plt.close()

# timecourse_interpolated('052418S')
# for sample in greater_95:
#   timecourse_interpolated(sample)

# SEG_DIR = '/Users/rowcheung/urop2019/fetal/registered_segmentations/'
# REG_DIR = '/Users/rowcheung/urop2019/fetal/registered/'
SEG_DIR = '/data/vision/polina/projects/fetal_brain/fetal/registered_segmentations/'
REG_DIR = '/data/vision/polina/projects/fetal_brain/fetal/registered'

def voxel_wise(sample, num_frames):
  resting = int(num_frames / len(O2_TIME[sample]))
  courses = np.zeros((847000, resting))
  for i in range(1, resting):
    registered = util.read_vol('{}/{}/{}_{}.nii.gz_to_{}_0180.nii.gz'.format(REG_DIR, sample, sample, str(i).zfill(4), sample))
    mask = util.read_vol('{}/{}/{}_{}.nii.gz'.format(SEG_DIR, sample, sample, str(i).zfill(4)))
    #print(registered.shape)
    # print(mask.shape)
    brain = np.ravel_multi_index(np.argwhere(mask==1).T, mask.shape) 
    for voxel in brain:
      # print(courses[voxel, i].shape)
      # print(registered[voxel].shape)
      courses[voxel, i] = registered[np.unravel_index(voxel, registered.shape)]
  courses = courses[np.nonzero(courses)[0], :]
  means = np.mean(courses, axis=1)
  std = np.std(courses, axis=1)
  print(std)
  print(courses.T - means)
  normalized = ((courses.T - means) / std).T
  return normalized

courses = voxel_wise('021218L', 222)
#courses = courses[np.nonzero(courses)[0], :]
courses = courses[50:75, :]
for i in range(courses.shape[0]):
  plt.plot(courses[i, :])

plt.savefig('./voxel_wise.png')


