from typing import Any, Dict

import torch
import numpy as np
import determined as det
from determined.pytorch import DataLoader, PyTorchTrial

from mt_dnn.batcher import BatchGen
from mt_dnn.model import MTDNNModel
from bert.modeling import BertModel
from bert.modeling import BertConfig
import os
import json

torch.manual_seed(42)
np.random.seed(42)

class DoubleTransferModel:
    """
    DoubleTransfer models in the class
    """

    def __init__(self, **config):



class DoubleTransferTrial(PyTorchTrial):

    def __init__(self, context: det.TrialContext) -> None:
        self.context = context
        self.data_config = self.context.get_data_config()

        self.learning_rate = context.get_hparam("LR")
        self._batch_size = context.get_per_slot_batch_size()
        self.binary = context.get_data_config()["binary"] #context.get_hparam("binary")
        self.drug_encoding = context.get_hparam("drug_encoding")
        self.target_encoding = context.get_hparam("target_encoding")

    def build_training_data_loader(self) -> DataLoader:
        if self.data_config['url_path'].lower() == "davis":
           X_drug, X_target, y = load_process_DAVIS('/data/DAVIS', binary=self.data_config['binary'], convert_to_log=self.data_config['convert_to_log'])
           # y = y.astype(float)
        elif self.data_config['url_path'].lower() == "kiba":
            X_drug, X_target, y = load_process_KIBA('/data/KIBA', binary=self.data_config['binary'])
        else:
            X_drug, X_target, y = process_BindingDB(
                self.data_config['url_path'],
                y=self.data_config['yy'],
                binary=self.data_config['binary'],
                convert_to_log=self.data_config['convert_to_log'])

        train, val, test = data_process(X_drug, X_target, y,
                                        self.context.get_hparam('drug_encoding'),
                                        self.context.get_hparam('target_encoding'),
                                        split_method='random',
                                        frac=self.data_config['frac'],
                                        random_seed=self.context.get_hparam("random_seed"))

        params = {'batch_size': self.context.get_per_slot_batch_size(),
                  'shuffle': self.data_config['shuffle'],
                  'num_workers': self.data_config['num_workers'],
                  'drop_last': self.data_config['drop_last']}

        if (self.context.get_hparam('drug_encoding') == "MPNN"):
            params['collate_fn'] = mpnn_collate_func

        training_generator = DataLoader(
            data_process_loader(train.index.values, train.Label.values, train,
                                **self.context.get_hparams()),
            **params)

        return training_generator

    def build_validation_data_loader(self) -> DataLoader:
        if self.data_config['url_path'].lower() == "davis":
           X_drug, X_target, y = load_process_DAVIS('/data/DAVIS', binary=self.data_config['binary'], convert_to_log=self.data_config['convert_to_log'])
        elif self.data_config['url_path'].lower() == "kiba":
            X_drug, X_target, y = load_process_KIBA('/data/KIBA', binary=self.data_config['binary'])
        else:
            X_drug, X_target, y = process_BindingDB(
                self.data_config['url_path'],
                y=self.data_config['yy'],
                binary=self.data_config['binary'],
                convert_to_log=self.data_config['convert_to_log'])
        train, val, test = data_process(X_drug, X_target, y,
                                        self.context.get_hparam('drug_encoding'),
                                        self.context.get_hparam('target_encoding'),
                                        split_method='random',
                                        frac=self.data_config['frac'],
                                        random_seed=self.context.get_hparam("random_seed"))

        params = {'batch_size': self.context.get_per_slot_batch_size(),
                  'shuffle': self.data_config['shuffle'],
                  'num_workers': self.data_config['num_workers'],
                  'drop_last': self.data_config['drop_last']}
        if (self.context.get_hparam('drug_encoding') == "MPNN"):
            params['collate_fn'] = mpnn_collate_func

        validation_generator = DataLoader(
            data_process_loader(val.index.values, val.Label.values, val,
                                **self.context.get_hparams()),
            **params)

        return validation_generator

    def batch_size(self) -> int:
        return self._batch_size

    def build_model(self) -> nn.Module:
        hp = self.context.get_hparams().copy()
        hp.pop('random_seed')
        hp.pop('global_batch_size')
        hp['batch_size'] = self.context.get_per_slot_batch_size()

        cls_list = [k for k in hp.keys() if 'cls_hidden' in k]
        hp.update(generate_classifier_params(hp))
        for k in cls_list:
            hp.pop(k)

        if hp['drug_encoding'].lower() == 'cnn' or hp['drug_encoding'].lower() == 'cnn_rnn':
            drug_list = [k for k in hp.keys() if 'cnn_drug' in k]
            hp.update(generate_cnn_params(hp, 'drug'))
            for k in drug_list:
                hp.pop(k)
        elif hp['drug_encoding'].lower() == 'mlp':
            mlp_list = [k for k in hp.keys() if 'mlp_drug' in k]
            hp.update(generate_mlp_params(hp, 'drug'))
            for k in mlp_list:
                hp.pop(k)

        if hp['target_encoding'].lower() == 'cnn' or hp['target_encoding'].lower() == 'cnn_rnn':
            target_list = [k for k in hp.keys() if 'cnn_target' in k]
            hp.update(generate_cnn_params(hp, 'target'))
            for k in target_list:
                hp.pop(k)
        elif hp['target_encoding'].lower() == 'mlp':
            mlp_list = [k for k in hp.keys() if 'mlp_target' in k]
            hp.update(generate_mlp_params(hp, 'target'))
            for k in mlp_list:
                hp.pop(k)

        config = generate_config(**hp)
        model_obj = DeepPurposeModel(**config)
        return model_obj.get_model()

    def optimizer(self, model: nn.Module) -> torch.optim.Optimizer:  # type: ignore
        #        return torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        return torch.optim.Adam(model.parameters(), lr=self.learning_rate)

    def losses(self, predictions: TorchData, labels: TorchData) -> TorchData:
        if self.binary:
            loss_fct = torch.nn.BCELoss()
            final = torch.nn.Sigmoid()
            if len(predictions.shape) > 1:
                pred = torch.squeeze(final(predictions), 1)
            else:
                pred = final(predictions)
            labels = labels.float()
            pred = pred.float()
            loss = loss_fct(pred, labels)
            return loss # type: ignore
        else:
            loss_fct = torch.nn.MSELoss()
            if len(predictions.shape) > 1:
                pred = torch.squeeze(predictions, 1)
            else:
                pred = predictions
            loss = loss_fct(pred, labels)
            return loss

    def train_batch(
        self, batch: TorchData, model: nn.Module, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        batch = cast(Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch)
        v_d, v_p, labels = batch

        if self.target_encoding == 'Transformer':
            v_p = v_p
        else:
            v_p = v_p.float()
        if self.drug_encoding == "MPNN" or self.drug_encoding == 'Transformer':
            v_d = v_d
        else:
            v_d = v_d.float()
        scores = model(v_d, v_p)

        if self.binary:
            final = torch.nn.Sigmoid()
            logits = torch.squeeze(final(scores)).detach().cpu().numpy()
            outputs = np.where(logits >= 0.5, 1, 0)
            apr = average_precision_score(labels.cpu().numpy(), logits)
            fsc = f1_score(labels.cpu().numpy(), outputs)
            fsc = fsc if not math.isnan(fsc) else 0
            apr = apr if not math.isnan(apr) else 0
            return {
                "loss": self.losses(scores, labels),
                "train_f1": fsc,
                "train_apr": apr
            }
        else:
            logits = torch.squeeze(scores).detach().cpu().numpy()
            mse = mean_squared_error(labels.cpu().numpy(), logits)
            r2, pval = pearsonr(labels.cpu().numpy(), logits)
            ci = concordance_index(labels.cpu().numpy(), logits)
            return {
                "loss": self.losses(scores, labels),
                "train_mse": mse,
                "train_r2": r2,
                "train_pval": pval,
                "train_CI": ci
            }

    def evaluate_full_dataset(self, data_loader: torch.utils.data.dataloader.DataLoader, model: torch.nn.modules.module.Module):
        all_logits = np.empty(0)
        all_labels = np.empty(0)
        all_outputs = np.empty(0)
        all_scores = np.empty(0)
        for v_d, v_p, labels in iter(data_loader):
            if self.target_encoding == 'Transformer':
                v_p = v_p
            else:
                v_p = self.context.to_device(v_p.float())
            if self.drug_encoding == "MPNN" or self.drug_encoding == 'Transformer':
                v_d = v_d
            else:
                v_d = self.context.to_device(v_d.float())

            scores = model(v_d, v_p)
            labels = labels.cpu().numpy()
            all_scores = np.append(all_scores, scores.cpu().numpy())

            if self.binary:
                final = torch.nn.Sigmoid()
                logits = torch.squeeze(final(scores)).detach().cpu().numpy()
                outputs = np.where(logits >= 0.5, 1, 0)
                all_outputs = np.append(all_outputs, outputs)
            else:
                logits = torch.squeeze(scores).detach().cpu().numpy()

            all_logits = np.append(all_logits, logits.flatten().tolist())
            all_labels = np.append(all_labels, labels.flatten().tolist())

        if self.binary:
            roc = roc_auc_score(all_labels, all_logits)
            apr = average_precision_score(all_labels, all_logits)
            fsc = f1_score(all_labels, all_outputs)
            fsc = fsc if not math.isnan(fsc) else 0
            apr = apr if not math.isnan(apr) else 0
            loss = log_loss(all_labels, all_outputs)
            return {
                "val_loss": loss,
                "val_f1": fsc,
                "val_apr": apr,
                "val_roc": roc
            }
        else:
            mse = mean_squared_error(all_labels, all_logits)
            r2, pval = pearsonr(all_labels, all_logits)
            ci = concordance_index(all_labels, all_logits)
            return {
                "val_loss": mse,
                "val_r2": r2,
                "val_pval": pval,
                "val_CI": ci
            }