# -*- coding: utf-8 -*-
import os
import pandas as pd
from glob import glob
from skimage import io


class BatchReader:
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.loaded_batch_idx = -1
        self.batch_of = {}
        self.img = None
        self.item_of = None
        self.sz = None

        self.list_path = os.path.join(self.work_dir, 'list_batches_of_filenames.txt')
        if not os.path.isfile(self.list_path):
            print('List of batches not found in ' + self.list_path)
            print('Making batches list ...')
            self.make_bathes_list()
            print('OK')

        self.read_list_to_dict()

    def get_mask_of(self, filename):
        if filename in self.batch_of:
            batch_idx = self.batch_of[filename]

            if batch_idx != self.loaded_batch_idx:
                self.load_batch(batch_idx)

            item_idx = self.item_of[filename]
            start = self.sz * item_idx
            end = self.sz * (item_idx + 1)
            img = self.img[:, start:end].copy()

            return img
        else:
            return None

    def load_batch(self, batch_idx):
        print('Loading batch #%i' % batch_idx)

        batch_img_path = os.path.join(self.work_dir, 'batch%06i-mask.png' % batch_idx)
        batch_txt_path = os.path.join(self.work_dir, 'batch%06i.txt' % batch_idx)

        self.img = io.imread(batch_img_path).astype(float) / 255.
        self.sz = self.img.shape[0]

        self.item_of = {}
        df = pd.read_csv(batch_txt_path)
        for i, row in df.iterrows():
            path = row['filenames']
            filename = os.path.split(path)[1]
            self.item_of[filename] = i

        self.loaded_batch_idx = batch_idx

    def read_list_to_dict(self):
        print('Reading batches list to dictionary from ' + self.list_path)
        batch_of = {}
        df = pd.read_csv(self.list_path)
        for i, row in df.iterrows():
            if i % 10000 == 0:
                print('%i / %i' % (i, df.shape[0]))

            batch_of[row['filename']] = row['batch']
        print('OK')

        self.batch_of = batch_of

    def make_bathes_list(self):
        filenames = []
        batches = []

        paths = glob(os.path.join(self.work_dir, 'batch*.txt'))

        for path in paths:
            if path.endswith('00.txt'):
                print(path)

            batch_idx = int(os.path.split(path)[1][5:11])

            df = pd.read_csv(path)
            for row in df.iterrows():
                filename = row[1]['filenames']
                filename = os.path.split(filename)[1]

                filenames.append(filename)
                batches.append(batch_idx)

        df = pd.DataFrame.from_dict({'filename': filenames, 'batch': batches})
        df.to_csv(os.path.join(self.work_dir, 'list_batches_of_filenames.txt'), index=0)
