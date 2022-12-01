# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Data module, batch and dataset for FilterBERT. """

import glob
import os

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class FilteringDataModule(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.num_workers = args.num_workers
        self.batch_size_train = 1
        self.batch_size_test = 1
        self.max_src_len = 512

    def collate(self, data):
        def truncate(x):
            srcs = x['srcs']
            for i in range(len(srcs)):
                srcs[i] = srcs[i][:self.max_src_len][:-1] + [srcs[i][-1]]
            x['srcs'] = srcs
            return x

        data = list(map(truncate, data))
        return FilteringBatch(data)

    def train_dataloader(self):
        dataset = FilteringDataset(
            data_dir=self.data_dir,
            split='train',
        )

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_train,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
        )

    def val_dataloader(self):
        dataset = FilteringDataset(
            data_dir=self.data_dir,
            split='valid',
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_test,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
        )

    def test_dataloader(self):
        dataset = FilteringDataset(
            data_dir=self.data_dir,
            split='test',
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_test,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
        )


class FilteringBatch:

    def __init__(self, data, max_len=512, pad_id=0):
        assert len(data) == 1, "Can only handle batch size of 1 (input becomes too big otherwise)"
        self.batch_size = len(data)
        self.max_len = max_len
        self.pad_id = pad_id
        self.srcs = torch.tensor(self.pad(data[0]['srcs']))
        self.tgts = torch.tensor(self.pad(data[0]['tgts']))
        self.tgts_segs = torch.tensor(self.pad(data[0]['tgts_segs']))
        self.filter_target = torch.tensor(data[0]['filter_target'])
        self.mask_srcs = 1 - (self.srcs == pad_id).to(torch.uint8)
        self.mask_tgts = 1 - (self.tgts == pad_id).to(torch.uint8)

    def pad(self, data):
        """ Pad `data` to same length with `pad_id`. """
        return [x + [self.pad_id] * (self.max_len - len(x)) for x in data]

    def __len__(self):
        return self.batch_size

    def to(self, *args, **kwargs):
        self.srcs = self.srcs.to(*args, **kwargs)
        self.tgts = self.tgts.to(*args, **kwargs)
        self.tgts_segs = self.tgts_segs.to(*args, **kwargs)
        self.filter_target = self.filter_target.to(*args, **kwargs)
        self.mask_srcs = self.mask_srcs.to(*args, **kwargs)
        self.mask_tgts = self.mask_tgts.to(*args, **kwargs)
        return self

    def pin_memory(self):
        self.srcs = self.srcs.pin_memory()
        self.tgts = self.tgts.pin_memory()
        self.tgts_segs = self.tgts_segs.pin_memory()
        self.filter_target = self.filter_target.pin_memory()
        self.mask_srcs = self.mask_srcs.pin_memory()
        self.mask_tgts = self.mask_tgts.pin_memory()
        return self


class FilteringDataset(Dataset):

    def __init__(self, data_dir, split='train'):
        data_files = sorted(glob.glob(os.path.join(data_dir, f'fomc.{split}.pt')))
        rouge_files = sorted(glob.glob(os.path.join(data_dir, f'fomc.{split}.rouge.pt')))
        assert len(data_files) == len(rouge_files), 'Number of data and ROUGE files does not match'
        self.data = []
        for data_file, rouge_file in zip(data_files, rouge_files):
            data_samples = torch.load(data_file)
            rouge_samples = torch.load(rouge_file)
            for data_sample, rouge_sample in zip(data_samples, rouge_samples):
                data_sample.update(rouge_sample)  # add information from rouge_sample
                self.data.append(data_sample)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
