import os
import argparse
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

sets = {
    'train', 'val'
}
# 'ntu/xview', 'ntu/xsub'
benchmarks = {
    'ntu': ('ntu/xview', 'ntu/xsub'),
    'ntu120': ('ntu120/xset', 'ntu120/xsub'),
}

parts = { 'train', 'val' }

modality = {
    'joint', 'bone'
}



parser = argparse.ArgumentParser(description='Bone data generation for NTU60/NTU120')
parser.add_argument('--dataset', choices=['ntu', 'ntu120'], required=True)
args = parser.parse_args()
    
for benchmark in benchmarks[args.dataset]:
    for part in parts:
        for mod in modality:
            print(benchmark, part, mod)
            try:
                data = np.load('../data/{}/{}_data_{}.npy'.format(benchmark, part, mod))
                N, C, T, V, M = data.shape
                fp_sp = open_memmap(
                    '../data/{}/{}_data_{}_motion.npy'.format(benchmark, part, mod),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 3, T, V, M))
                for t in tqdm(range(T - 1)):
                    fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]
                fp_sp[:, :, T - 1, :, :] = 0
            except Exception as e:
                print(f'Run into error: {e}')
                print(f'Skipping ({benchmark} {part})')