# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import os
import sys
import json
from typing import Dict, Any

import torch
import random
import numpy as np

from torch.utils.data import Dataset
from data_utils.label_map import DATA_META, GLOBAL_MAP, DATA_TYPE, DATA_SWAP, TASK_TYPE, generate_decoder_opt
from data_utils.utils import set_environment
from data_utils.mediqa2019_evaluator_allTasks_final import eval_mediqa_official
from mt_dnn.batcher import BatchGen

UNK_ID=100
BOS_ID=101

mediqa_name_list = ['mediqa','mediqa_url']

class DoubleTransfer_Batched(Dataset):
    """
    This utility uses BatchGen objects, but wraps a dataset class around the iterators so that we can use
    DataLoaders for training and evaluation
    """

    def __init__(self, data_config: Dict[str, Any], mode: str ='train', if_batch_gen_util: bool = True):
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
for i in range(3):
    print(ds[i])
    # ds_iter = iter(ds[i])
    # for j in range(5):
    #     print(next(ds_iter))