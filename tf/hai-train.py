import train

import multiprocessing as mp
import tempfile
import yaml
import argparse
import os
import os.path

if __name__ == "__main__":

    os.makedirs('full_weights', exist_ok = True)

    os.makedirs('configs/temp', exist_ok = True)

    with tempfile.NamedTemporaryFile(dir='configs/temp', mode = 'wt', delete=False) as fp:

        print(fp.name)

        argparser = argparse.ArgumentParser(
            description= 'Tensorflow pipeline for training Leela Chess modified for haibrid server training.'
        )

        argparser.add_argument('target', type=str,
            help='target path train against')

        argparser.add_argument('gpu', type=int,
            help='gpu to use 0 or 1')

        n = argparser.parse_args()

        with open('configs/train_prototype.yaml') as f:
            conf = yaml.safe_load(f.read())

        name = os.path.basename(n.target[:-1]) if n.target.endswith('/') else os.path.basename(n.target)

        conf['name'] = f"{name}-64x6"
        conf['gpu'] = n.gpu
        conf['dataset']['input'] = os.path.join(n.target, 'supervise-*/*.gz')


        fp.write(yaml.dump(conf))

        print(f"Doing run with {name} on GPU: {n.gpu}")

    argparser = argparse.ArgumentParser(
        description= 'Dummy args')
    argparser.add_argument('target', type=str,
        help='target path train against')

    argparser.add_argument('gpu', type=int,
        help='gpu to use 0 or 1')
    argparser.add_argument('--cfg', type=argparse.FileType('r'),
        help='yaml configuration with training parameters', default = fp.name)
    argparser.add_argument('--output', type=str,
        help='file to store weights in', default = f"full_weights/{name}-64x6.txt")


    c = argparser.parse_args()
    mp.set_start_method('spawn')
    train.main(c)
    mp.freeze_support()
