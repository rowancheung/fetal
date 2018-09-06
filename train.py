import os
import logging
logging.basicConfig(level=logging.INFO)

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--train',
                    metavar='INPUT_FILES, LABEL_FILES',
                    help='Train model',
                    dest='train', type=str, nargs=2)
parser.add_argument('--predict',
                    metavar='INPUT_FILES, SEED_FILES/LABEL_FILES, SAVE_PATH',
                    help='Predict segmentations',
                    dest='predict', type=str, nargs=3)
parser.add_argument('--test',
                    metavar='INPUT_FILES, [SEED_FILES,] LABEL_FILES',
                    help='Test model',
                    dest='test', type=str, nargs='+')
parser.add_argument('--organ',
                    metavar='ORGAN',
                    help='Organ to segment',
                    dest='organ', type=str, nargs=1)
parser.add_argument('--seed',
                    metavar='SEED_TYPE',
                    help='Seed slices',
                    dest='seed', type=str)
parser.add_argument('--concat',
                    metavar='INPUT_FILE, LABEL_FILE',
                    help='Concatenate first volume',
                    dest='concat', nargs=2)
parser.add_argument('--batch-size',
                    metavar='BATCH_SIZE',
                    help='Training batch size',
                    dest='batch_size', type=int, default=1)
parser.add_argument('--epochs',
                    metavar='EPOCHS',
                    help='Training epochs',
                    dest='epochs', type=int, default=1000)
parser.add_argument('--name',
                    metavar='MODEL_NAME',
                    help='Name of model',
                    dest='name', type=str)
parser.add_argument('--model-file',
                    metavar='MODEL_FILE',
                    help='Pretrained model file',
                    dest='model_file', type=str)
parser.add_argument('--size',
                    metavar='SIZE',
                    help='Size of UNet',
                    dest='size', type=str)
parser.add_argument('--run',
                    metavar='RUN',
                    dest='run', type=str)
parser.add_argument('--part',
                    metavar='PART',
                    dest='part', type=int)
options = parser.parse_args()

import constants
import glob
import time
import util
from data import AugmentGenerator, VolumeGenerator
from models import UNet, UNetSmall, UNetBig


def main(options):
    start = time.time()

    logging.info('Creating model.')
    shape = constants.SHAPE
    if options.seed:
        shape = tuple(list(shape[:-1]) + [shape[-1] + 1])
    if options.concat:
        shape = tuple(list(shape[:-1]) + [shape[-1] + 2])
    if options.size == 'small':
        m = UNetSmall
    elif options.size == 'big':
        m = UNetBig
    else:
        m = UNet
    model = m(shape, name=options.name, filename=options.model_file)

    gen_seed = (options.seed == 'slice' or options.seed == 'volume')

    if options.train:
        logging.info('Creating data generator.')

        input_path = options.train[0].split('*')[0]
        label_path = options.train[1].split('*')[0]

        label_files = glob.glob(options.train[1])
        input_files = [label_file.replace(label_path, input_path) for label_file in label_files]

        aug_gen = AugmentGenerator(input_files,
                                   label_files=label_files,
                                   batch_size=options.batch_size,
                                   seed_type=options.seed,
                                   concat_files=options.concat)
        #FIXME
        val_gen = VolumeGenerator(input_files,
                                  label_files=label_files,
                                  batch_size=options.batch_size,
                                  seed_type=options.seed,
                                  concat_files=options.concat,
                                  load_files=True,
                                  include_labels=True)

        logging.info('Compiling model.')
        model.compile(util.get_weights(aug_gen.labels))

        logging.info('Training model.')
        model.train(aug_gen, val_gen, options.epochs)
        model.save()

    if options.predict:
        logging.info('Making predictions.')

        input_files = glob.glob(options.predict[0])
        seed_files = None if gen_seed else glob.glob(options.predict[1])
        label_files = glob.glob(options.predict[1]) if gen_seed else None
        save_path = options.predict[2]

        pred_gen = VolumeGenerator(input_files,
                                   seed_files=seed_files,
                                   label_files=label_files,
                                   batch_size=options.batch_size,
                                   seed_type=options.seed,
                                   concat_files=options.concat,
                                   include_labels=False)
        model.predict(pred_gen, save_path)

    if options.test:
        logging.info('Testing model.')

        input_files = glob.glob(options.test[0])
        seed_files = None if gen_seed else glob.glob(options.test[1])
        label_files = glob.glob(options.test[1]) if gen_seed else glob.glob(options.test[2])

        test_gen = VolumeGenerator(input_files,
                                   seed_files=seed_files,
                                   label_files=label_files,
                                   batch_size=options.batch_size,
                                   seed_type=options.seed,
                                   concat_files=options.concat,
                                   include_labels=True)
        metrics = model.test(test_gen)
        logging.info(metrics)

    end = time.time()
    logging.info('total time: {}s'.format(end - start))


def run(options):
    start = time.time()
    metrics = {}
    organ = 'all_brains' if options.organ[0] == 'brains' else options.organ[0]
    samples = constants.SAMPLES[(options.part-1)*10:options.part*10]

    for sample in samples:
        logging.info(sample)

        logging.info('Creating model.')
        shape = constants.SHAPE
        if options.seed:
            shape = tuple(list(shape[:-1]) + [shape[-1] + 1])
        if options.run == 'concat':
            #TODO
            pass
        if options.size == 'small':
            m = UNetSmall
        elif options.size == 'big':
            m = UNetBig
        else:
            m = UNet
        model = m(shape, name='unet_brains_{}_{}'.format(options.run, sample), filename=options.model_file)

        logging.info('Creating data generator.')

        if options.run == 'concat':
            #TODO
            pass
        else:
            concat_files = None

        if options.run == 'one-out':
            label_files = [file for file in glob.glob('data/labels/*/*_{}.nii.gz'.format(organ))
                           if not os.path.basename(file).startswith(sample)]
        elif options.run == 'single':
            label_files = glob.glob('data/labels/{}/{}_0_{}.nii.gz'.format(sample, sample, organ))
        elif options.run == 'concat':
            #TODO
            pass
        else:
            raise ValueError('Preset program not defined.')

        input_files = [file.replace('labels', 'raw').replace('_{}'.format(organ), '') for file in label_files]
        aug_gen = AugmentGenerator(input_files,
                                   label_files=label_files,
                                   batch_size=options.batch_size,
                                   seed_type=options.seed,
                                   concat_files=concat_files)
        #FIXME
        val_gen = VolumeGenerator(input_files,
                                  label_files=label_files,
                                  batch_size=options.batch_size,
                                  seed_type=options.seed,
                                  concat_files=concat_files,
                                  load_files=True,
                                  include_labels=True)

        logging.info('Compiling model.')
        model.compile(util.get_weights(aug_gen.labels))

        logging.info('Training model.')
        model.train(aug_gen, val_gen, options.epochs)

        logging.info('Saving model.')
        model.save()

        logging.info('Making predictions.')
        if options.run == 'one-out':
            predict_files = glob.glob('data/raw/{}/{}_*.nii.gz'.format(sample, sample))
        elif options.run == 'single':
            predict_files = [f for f in glob.glob('data/raw/{}/{}_*.nii.gz'.format(sample, sample))
                           if not os.path.basename(f).endswith('_0.nii.gz')]
        elif options.run == 'concat':
            #TODO
            pass
        else:
            raise ValueError('Preset program not defined.')

        pred_gen = VolumeGenerator(predict_files,
                                   batch_size=options.batch_size,
                                   seed_type=options.seed,
                                   concat_files=concat_files,
                                   include_labels=False)
        save_path = 'data/predict/{}/'.format(sample)
        os.makedirs(save_path, exist_ok=True)
        model.predict(pred_gen, save_path)

        if options.run == 'one-out':
            logging.info('Testing model.')
            test_files = glob.glob('data/raw/{}/{}_0.nii.gz'.format(sample, sample))
            label_files = glob.glob('data/labels/{}/{}_0_{}.nii.gz'.format(sample, sample, organ))

            test_gen = VolumeGenerator(test_files,
                                       label_files=label_files,
                                       batch_size=options.batch_size,
                                       seed_type=options.seed,
                                       concat_files=concat_files,
                                       include_labels=True)
            metrics[sample] = model.test(test_gen)

    if len(metrics) > 0:
        logging.info(metrics)

    end = time.time()
    logging.info('total time: {}s'.format(end - start))


if __name__ == '__main__':
    if options.run:
        run(options)
    else:
        main(options)
