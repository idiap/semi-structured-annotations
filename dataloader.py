# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Data module, batch and dataset. """

import glob
import os

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class SummarizationDataModule(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.data_model = 'bart' if args.model == 'bart' else 'bert'
        self.filter_model = args.filter_model
        self.num_workers = args.num_workers
        self.batch_size_train = args.batch_size
        self.batch_size_test = 1
        self.max_pos = 512 if self.data_model == 'bert' else 1024
        self.max_tgt_len = 512
        self.tgt_eos_id = 2

    def _truncate_bert(self, x):
        x['src'] = x['src'][:-1][:self.max_pos - 1] + x['src'][-1:]  # slicing notation works with empty inputs
        x['tgt'] = x['tgt'][:self.max_tgt_len][:-1] + [self.tgt_eos_id]
        x['src_segs'] = x['src_segs'][:self.max_pos]
        return x

    def _truncate_bart(self, x):
        x['src'] = x['src'][:self.max_pos][:-1] + x['src'][-1:]
        x['tgt'] = x['tgt'][:self.max_tgt_len][:-1] + x['tgt'][-1:]
        return x

    def collate(self, data):
        assert self.data_model in ['bert', 'bart'], f"Unknown data model: {self.data_model}"
        if self.data_model == 'bert':
            data = list(map(self._truncate_bert, data))
            return SummarizationBatch(data)
        else:
            return BartBatch(list(map(self._truncate_bart, data)))

    def train_dataloader(self):
        dataset = SummarizationDataset(
            data_dir=self.data_dir,
            filter_model=self.filter_model,
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
        dataset = SummarizationDataset(
            data_dir=self.data_dir,
            filter_model=self.filter_model,
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
        dataset = SummarizationDataset(
            data_dir=self.data_dir,
            filter_model=self.filter_model,
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


class SummarizationBatch:

    def __init__(self, data, pad_id=0):
        self.batch_size = len(data)
        self.pad_id = pad_id
        self.src = torch.tensor(self.pad([x['src'] for x in data]))
        self.tgt = torch.tensor(self.pad([x['tgt'] for x in data]))
        self.segs = torch.tensor(self.pad([x['src_segs'] for x in data]))
        self.mask_src = 1 - (self.src == 0).to(torch.uint8)
        self.mask_tgt = 1 - (self.tgt == 0).to(torch.uint8)
        self.refdoc = [x['name'] for x in data]

    def pad(self, data):
        """ Pad `data` to same length with `pad_id`. """
        max_len = max(len(x) for x in data)
        return [x + [self.pad_id] * (max_len - len(x)) for x in data]

    def __len__(self):
        return self.batch_size

    def to(self, *args, **kwargs):
        self.src = self.src.to(*args, **kwargs)
        self.tgt = self.tgt.to(*args, **kwargs)
        self.segs = self.segs.to(*args, **kwargs)
        self.mask_src = self.mask_src.to(*args, **kwargs)
        self.mask_tgt = self.mask_tgt.to(*args, **kwargs)
        return self

    def pin_memory(self):
        self.src = self.src.pin_memory()
        self.tgt = self.tgt.pin_memory()
        self.segs = self.segs.pin_memory()
        self.mask_src = self.mask_src.pin_memory()
        self.mask_tgt = self.mask_tgt.pin_memory()
        return self


class BartBatch:

    def __init__(self, data, pad_id=1):
        self.batch_size = len(data)
        self.pad_id = pad_id
        self.src = torch.tensor(self.pad([x['src'] for x in data]))
        self.tgt = torch.tensor(self.pad([x['tgt'] for x in data]))
        self.mask_src = 1 - (self.src == pad_id).to(torch.uint8)
        self.mask_tgt = 1 - (self.tgt == pad_id).to(torch.uint8)
        self.refdoc = [x['name'] for x in data]

    def pad(self, data):
        """ Pad `data` to same length with `pad_id`. """
        max_len = max(len(x) for x in data)
        return [x + [self.pad_id] * (max_len - len(x)) for x in data]

    def __len__(self):
        return self.batch_size

    def to(self, *args, **kwargs):
        self.src = self.src.to(*args, **kwargs)
        self.tgt = self.tgt.to(*args, **kwargs)
        self.mask_src = self.mask_src.to(*args, **kwargs)
        self.mask_tgt = self.mask_tgt.to(*args, **kwargs)
        return self

    def pin_memory(self):
        self.src = self.src.pin_memory()
        self.tgt = self.tgt.pin_memory()
        self.mask_src = self.mask_src.pin_memory()
        self.mask_tgt = self.mask_tgt.pin_memory()
        return self


class SummarizationDataset(Dataset):

    def __init__(self, data_dir, filter_model, split='train'):
        data_files = sorted(glob.glob(os.path.join(data_dir, f'fomc.{filter_model}.{split}.pt')))
        self.data = []
        for pt in data_files:
            self.data.extend(torch.load(pt))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
