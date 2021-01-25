# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import os
import sys
import json
from typing import Dict, Any

import torch
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader
from data_utils.label_map import DATA_META, GLOBAL_MAP, DATA_TYPE, DATA_SWAP, TASK_TYPE, generate_decoder_opt
from data_utils.utils import set_environment
from data_utils.mediqa2019_evaluator_allTasks_final import eval_mediqa_official
from mt_dnn.batcher import BatchGen

UNK_ID = 100
BOS_ID = 101

mediqa_name_list = ['mediqa', 'mediqa_url']


class DoubleTransfer(Dataset):
    """
    This class customizes BatchGen class to be able to read and return data sample by sample
    """

    def __init__(self, data, is_train=True,
                 dataset_name=None):
        self.is_train = is_train
        self.gpu = torch.cuda.is_available()
        self.dataset_name = dataset_name
        self.sequential_data = [sample for sample in data]
        if dataset_name in mediqa_name_list:
            self.q_dict = {}
            for sample in self.sequential_data:
                qid, aid = sample['uid'].split('____')
                if qid not in self.q_dict:
                    self.q_dict[qid] = []
                self.q_dict[qid].append(sample)
        if not self.is_train:
            self.data = data

    def __len__(self):
        return len(self.data)

    @staticmethod
    def load(path, is_train=True, maxlen=128, factor=1.0, pairwise=False,
             opt=None, dataset='mediqa'):
        prefix = dataset.split('_')[0]
        score_offset = opt['mediqa_score_offset'] if prefix in mediqa_name_list else 0.0

        with open(path, 'r', encoding='utf-8') as reader:
            data = []
            cnt = 0
            for line in reader:
                sample = json.loads(line)
                sample['factor'] = factor
                if is_train:
                    if pairwise and (len(sample['token_id'][0]) > maxlen or len(sample['token_id'][1]) > maxlen):
                        continue
                    if (not pairwise) and (len(sample['token_id']) > maxlen):
                        continue
                # if is_train:
                if prefix in mediqa_name_list and opt['mediqa_score'] == 'raw':
                    sample['label'] = float(sample['score'])
                if prefix == 'medquad' and opt['float_medquad']:
                    sample['label'] = sample['label'] * 4.0 + opt['mediqa_score_offset']  # to locate in range

                if score_offset != 0.0:
                    assert prefix in mediqa_name_list
                    sample['label'] += score_offset
                # pdb.set_trace()

                data.append(sample)
                cnt += 1
            # if dataset=='medquad':
            #     pdb.set_trace()
            print('Loaded {} samples out of {}'.format(len(data), cnt))
            return data

    def reset(self):
        if self.is_train:
            if self.dataset_name in mediqa_name_list:
                for qid in self.q_dict:
                    random.shuffle(self.q_dict[qid])
                q_pair_list = []
                for qid in self.q_dict:
                    for idx, sample in enumerate(self.q_dict[qid]):
                        next_idx = (idx + 1) % len(self.q_dict[qid])
                        q_pair_list.append((sample, self.q_dict[qid][next_idx]))
                random.shuffle(q_pair_list)
                self.data = []
                for pair in q_pair_list:
                    first_rank = int(pair[0]['rank'])
                    second_rank = int(pair[1]['rank'])
                    sam1 = {k: v for k, v in pair[0].items()}
                    sam2 = {k: v for k, v in pair[1].items()}
                    if first_rank < second_rank:
                        sam1['rank_label'] = 1
                        sam2['rank_label'] = 0
                    else:
                        sam1['rank_label'] = 0
                        sam2['rank_label'] = 1
                    self.data.extend([sam1, sam2])
            else:
                indices = list(range(len(self.sequential_data)))
                random.shuffle(indices)
                self.data = [self.sequential_data[i] for i in indices]
            # self.data = [self.data[i:i + self.batch_size] for i in range(0, len(self.data), self.batch_size)]
        # self.offset = 0

    def __getitem__(self, offset):
        return self.data[offset]


class CustomCollator(object):
    """
    Collate function for processing data and giving result
    """

    def __init__(self, data_loader_config: Dict[str, Any]):
        self.data_loader_config = data_loader_config
        self.pairwise_size = 1

    def __random_select__(self, arr):
        if self.data_loader_config['dropout_w'] > 0:
            return [UNK_ID if random.uniform(0, 1) < self.data_loader_config['dropout_w'] else e for e in arr]
        else:
            return arr

    def rebatch(self, batch):
        newbatch = []
        for sample in batch:
            size = len(sample['token_id'])
            self.pairwise_size = size
            assert size == len(sample['type_id'])
            for idx in range(0, size):
                token_id = sample['token_id'][idx]
                type_id = sample['type_id'][idx]
                uid = sample['ruid'][idx]
                olab = sample['olabel'][idx]
                newbatch.append({'uid': uid, 'token_id': token_id, 'type_id': type_id, 'label': sample['label'],
                                 'true_label': olab})
        return newbatch

    def __call__(self, batch):
        if self.data_loader_config['pairwise']:
            batch = self.rebatch(batch)
        batch_size = len(batch)
        # print('batch_size:',batch_size)
        batch_dict = {}
        tok_len = max(len(x['token_id']) for x in batch)
        hypothesis_len = max(len(x['type_id']) - sum(x['type_id']) for x in batch)
        token_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
        type_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
        masks = torch.LongTensor(batch_size, tok_len).fill_(0)
        if self.data_loader_config['data_type'] < 1:
            premise_masks = torch.ByteTensor(batch_size, tok_len).fill_(1)
            hypothesis_masks = torch.ByteTensor(batch_size, hypothesis_len).fill_(1)

        for i, sample in enumerate(batch):
            select_len = min(len(sample['token_id']), tok_len)
            tok = sample['token_id']
            if self.data_loader_config['is_train']:
                tok = self.__random_select__(tok)
            token_ids[i, :select_len] = torch.LongTensor(tok[:select_len])
            type_ids[i, :select_len] = torch.LongTensor(sample['type_id'][:select_len])
            masks[i, :select_len] = torch.LongTensor([1] * select_len)
            if self.data_loader_config['data_type'] < 1:
                hlen = len(sample['type_id']) - sum(sample['type_id'])
                hypothesis_masks[i, :hlen] = torch.LongTensor([0] * hlen)
                for j in range(hlen, select_len):
                    premise_masks[i, j] = 0
        if self.data_loader_config['data_type'] < 1:
            batch_info = {
                'token_id': 0,
                'segment_id': 1,
                'mask': 2,
                'premise_mask': 3,
                'hypothesis_mask': 4
            }
            batch_data = [token_ids, type_ids, masks, premise_masks, hypothesis_masks]
            current_idx = 5
            valid_input_len = 5
        else:
            batch_info = {
                'token_id': 0,
                'segment_id': 1,
                'mask': 2
            }
            batch_data = [token_ids, type_ids, masks]
            current_idx = 3
            valid_input_len = 3

        if self.data_loader_config['is_train']:
            labels = [sample['label'] for sample in batch]
            if self.data_loader_config['task_type'] > 0:
                batch_data.append(torch.FloatTensor(labels))
            else:
                batch_data.append(torch.LongTensor(labels))
            batch_info['label'] = current_idx
            current_idx += 1
            if 'rank_label' in batch[0]:
                rank_labels = [sample['rank_label'] for sample in batch]
                batch_data.append(torch.LongTensor(rank_labels))
                batch_info['rank_label'] = current_idx
                current_idx += 1

        # if self.gpu:
        #     for i, item in enumerate(batch_data):
        #         batch_data[i] = self.patch(item.pin_memory())

        # meta
        batch_info['uids'] = [sample['uid'] for sample in batch]
        batch_info['task_id'] = self.data_loader_config['task_id']
        batch_info['input_len'] = valid_input_len
        batch_info['pairwise'] = self.data_loader_config['pairwise']
        batch_info['pairwise_size'] = self.pairwise_size
        batch_info['task_type'] = self.data_loader_config['task_type']
        batch_info['dataset_name'] = self.data_loader_config['dataset_name']
        if not self.data_loader_config['is_train']:
            labels = [sample['label'] for sample in batch]
            batch_info['label'] = labels
            if self.data_loader_config['pairwise']:
                batch_info['true_label'] = [sample['true_label'] for sample in batch]
        return batch_info, batch_data


class DoubleTransfer_Batched(Dataset):
    """
    This utility uses BatchGen objects, but wraps a dataset class around the iterators so that we can use
    DataLoaders for training and evaluation
    """

    def __init__(self, data_config: Dict[str, Any], mode: str = 'train', if_batch_gen_util: bool = True):
        """
        :param data_config: The configuration required for data generation
        :param mode: One of 'train/dev/test'.
        :param if_batch_gen_util: Whether we will return BatchGen objects or our custom Dataset
        """
        self.data_config = data_config
        self.mode = mode
        self.tasks_config = {}
        self.tasks_class = {}
        self.process_config()
        self.if_batch_gen_util = if_batch_gen_util
        self.data = None
        if self.mode == 'train':
            self.data = self.prepare_train_data()
        elif self.mode == 'dev':
            self.data = self.prepare_validation_data()
        elif self.mode == 'test':
            self.data = self.prepare_test_data()

    def __len__(self):
        if self.data is None:
            return 0
        else:
            return len(self.data)

    def process_config(self):
        """
        This utility takes config and processes them to enable data pipeline building
        """
        if self.data_config['float_medquad']:
            TASK_TYPE['medquad'] = 1
            DATA_META['medquad'] = 1

        self.data_config['float_target'] = False

        if not 'batch_size_eval' in self.data_config.keys():
            self.data_config['batch_size_eval'] = None

        if self.data_config['batch_size_eval'] is None:
            self.data_config['batch_size_eval'] = self.data_config['batch_size']

        if not 'test_datasets' in self.data_config.keys():
            self.data_config['test_datasets'] = None

        if self.data_config['test_datasets'] is None:
            self.data_config['test_datasets'] = self.data_config['train_datasets']
        # args.train_datasets = args.train_datasets.split(',')
        # args.test_datasets = args.test_datasets.split(',')
        if len(self.data_config['train_datasets']) == 1:
            self.data_config['mtl_observe_datasets'] = self.data_config['train_datasets']
        self.data_config['external_datasets'] = self.data_config['external_datasets'].split(',') if self.data_config[
                                                                                    'external_datasets'] != '' else []
        self.data_config['train_datasets'] = self.data_config['train_datasets'] + self.data_config['external_datasets']

        # tasks_config = {}
        if os.path.exists(self.data_config['task_config_path']):
            with open(self.data_config['task_config_path'], 'r') as reader:
                self.tasks_config = json.loads(reader.read())

    def prepare_train_data(self):
        """
        This utility prepares data list for training process
        """
        train_data_list = []
        tasks = {}
        nclass_list = []
        decoder_opts = []
        dropout_list = []

        for dataset in self.data_config['train_datasets']:
            prefix = dataset.split('_')[0]
            if prefix in tasks: continue
            assert prefix in DATA_META
            assert prefix in DATA_TYPE
            data_type = DATA_TYPE[prefix]
            nclass = DATA_META[prefix]
            task_id = len(tasks)
            if self.data_config['mtl_opt'] > 0:
                task_id = self.tasks_class[nclass] if nclass in self.tasks_class else len(self.tasks_class)

            task_type = TASK_TYPE[prefix]
            pw_task = False

            dopt = generate_decoder_opt(prefix, self.data_config['answer_opt'])
            if task_id < len(decoder_opts):
                decoder_opts[task_id] = min(decoder_opts[task_id], dopt)
            else:
                decoder_opts.append(dopt)

            if prefix not in tasks:
                tasks[prefix] = len(tasks)
                if self.data_config['mtl_opt'] < 1: nclass_list.append(nclass)

            if (nclass not in self.tasks_class):
                self.tasks_class[nclass] = len(self.tasks_class)
                if self.data_config['mtl_opt'] > 0: nclass_list.append(nclass)

            dropout_p = self.data_config['dropout_p']
            if self.tasks_config and prefix in self.tasks_config:
                dropout_p = self.tasks_config[prefix]
            dropout_list.append(dropout_p)

            train_path = os.path.join(self.data_config['data_dir'], '{}_train.json'.format(dataset))
            # logger.info('Loading {} as task {}'.format(train_path, task_id))
            train_data = BatchGen(
                BatchGen.load(train_path, True, pairwise=pw_task, maxlen=self.data_config['max_seq_len'],
                              opt=self.data_config, dataset=dataset),
                batch_size=self.data_config['batch_size'],
                dropout_w=self.data_config['dropout_w'],
                gpu=self.data_config['cuda'],
                task_id=task_id,
                maxlen=self.data_config['max_seq_len'],
                pairwise=pw_task,
                data_type=data_type,
                task_type=task_type,
                dataset_name=dataset)
            train_data.reset()
            if self.if_batch_gen_util:
                train_data_list.append(train_data)
            else:
                # train_data.reset()
                train_iter = iter(train_data)
                while True:
                    try:
                        # get the next item
                        train_data_list.append(next(train_iter))
                        # do something with element
                    except StopIteration:
                        # if StopIteration is raised, break from loop
                        break
                # train_data_list.append(train_data)
        self.data_config['answer_opt'] = decoder_opts
        self.data_config['tasks_dropout_p'] = dropout_list
        self.data_config['label_size'] = nclass_list #','.join([str(l) for l in nclass_list])
        # logger.info(self.data_config['label_size'])
        return train_data_list

    def prepare_validation_data(self):
        """
        This utility prepares data list for training process
        """
        dev_data_list = []
        for dataset in self.data_config['test_datasets']:
            prefix = dataset.split('_')[0]
            task_id = self.tasks_class[DATA_META[prefix]] if self.data_config['mtl_opt'] > 0 else self.tasks_class[prefix]
            task_type = TASK_TYPE[prefix]

            pw_task = False

            assert prefix in DATA_TYPE
            data_type = DATA_TYPE[prefix]

            if self.data_config['predict_split'] is not None:
                dev_path = os.path.join(self.data_config['data_dir'], '{}_{}.json'.format(dataset,
                    self.data_config['predict_split']))
            else:
                dev_path = os.path.join(self.data_config['data_dir'], '{}_dev.json'.format(dataset))
            dev_data = None
            if os.path.exists(dev_path):
                dev_data = BatchGen(BatchGen.load(dev_path, False, pairwise=pw_task, maxlen=self.data_config['max_seq_len'],
                                                opt=self.data_config, dataset=dataset),
                                      batch_size=self.data_config['batch_size'],
                                      gpu=self.data_config['cuda'], is_train=False,
                                      task_id=task_id,
                                      maxlen=self.data_config['max_seq_len'],
                                      pairwise=pw_task,
                                      data_type=data_type,
                                      task_type=task_type,
                                      dataset_name=dataset)
            dev_data_list.append(dev_data)
        return dev_data_list


    def prepare_test_data(self):
        """
        This utility prepares test data that will enable us to test the trained model
        """
        test_data_list = []
        for dataset in self.data_config['test_datasets']:
            prefix = dataset.split('_')[0]
            task_id = self.tasks_class[DATA_META[prefix]] if self.data_config['mtl_opt'] > 0 else self.tasks_class[prefix]
            task_type = TASK_TYPE[prefix]

            pw_task = False

            assert prefix in DATA_TYPE
            data_type = DATA_TYPE[prefix]
            test_path = os.path.join(self.data_config['data_dir'], '{}_test.json'.format(dataset))
            test_data = None
            if os.path.exists(test_path):
                test_data = BatchGen(BatchGen.load(test_path, False, pairwise=pw_task,
                                                maxlen=self.data_config['max_seq_len'], opt=self.data_config, dataset=dataset),
                                      batch_size=self.data_config['batch_size'],
                                      gpu=self.data_config['cuda'], is_train=False,
                                      task_id=task_id,
                                      maxlen=self.data_config['max_seq_len'],
                                      pairwise=pw_task,
                                      data_type=data_type,
                                      task_type=task_type,
                                      dataset_name=dataset)
            test_data_list.append(test_data)
        return test_data_list

    def __getitem__(self, item):
       return self.data[item]

data_config = json.load(open('data_config.json', 'r'))
train_config = json.load(open('train_config.json', 'r'))
model_config = json.load(open('model_config.json', 'r'))
config = {}
config.update(data_config)
config.update(train_config)
config.update(model_config)
print(len(config))
if_batch_gen_util = False
ds = DoubleTransfer_Batched(config, if_batch_gen_util=if_batch_gen_util)
print(len(ds))
# for i in range(3):
#     print(ds[i])
# ds_iter = iter(ds[i])
# for j in range(5):
#     print(next(ds_iter))

data_loader = DataLoader(ds, batch_size=1)
for i, data in enumerate(data_loader):
    if i < 2:
        print(data)
    else:
        break
