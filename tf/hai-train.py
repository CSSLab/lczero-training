import train

import multiprocessing as mp
import tempfile
import yaml
import argparse
import os

if __name__ == "__main__":

    os.makedirs('full_weights', exist_ok = True)

    os.makedirs('configs/temp', exist_ok = True)

    with tempfile.NamedTemporaryFile(dir='configs/temp', mode = 'wt', delete=Fasle) as fp:

        print(fp.name)

        argparser = argparse.ArgumentParser(
            description= 'Tensorflow pipeline for training Leela Chess modified for haibrid server training.'
        )

        argparser.add_argument('elo', type=int,
            help='ELO range to train against')

        argparser.add_argument('gpu', type=int,
            help='gpu to use 0 or 1')

        n = argparser.parse_args()

        with open('configs/train_prototype.yaml') as f:
            conf = yaml.safe_load(f.read())

        conf['name'] = f"{n.elo}-64x6"
        conf['gpu'] = n.gpu
        conf['input'] = f'/datadrive/processed_games/{n.elo}/supervise-*/*.gz'


        fp.write(yaml.dump(conf))

        print(f"Doing run with {n.elo} on GPU: {n.gpu}")

        argparser = argparse.ArgumentParser(
            description= 'Dummy args')
        argparser.add_argument('elo', type=int,
            help='ELO range to train against')

        argparser.add_argument('gpu', type=int,
            help='gpu to use 0 or 1')
        argparser.add_argument('--cfg', type=argparse.FileType('r'),
            help='yaml configuration with training parameters', default = fp.name)
        argparser.add_argument('--output', type=str,
            help='file to store weights in', default = f"full_weights/{n.elo}-64x6.txt")

        #print(argparser.parse_args())

        mp.set_start_method('spawn')
        train.main(argparser.parse_args())
        mp.freeze_support()
