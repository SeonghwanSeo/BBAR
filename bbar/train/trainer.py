from rdkit import Chem

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Batch as PyGBatch
from torch_scatter import scatter_log_softmax
from torch.utils.tensorboard import SummaryWriter

import os
import pickle
import logging
import time
from omegaconf import OmegaConf
import pandas as pd
import gc

from typing import Tuple, Union, List, Iterable, Dict
from collections import OrderedDict
from torch import LongTensor, BoolTensor, FloatTensor
from rdkit.Chem import Mol
from bbar.utils.typing import SMILES

from bbar.model import BlockConnectionPredictor
from bbar.transform import BlockGraphTransform
from bbar.fragmentation import BRICS_BlockLibrary
from .dataset import BBARDataset 
from bbar.utils.common import convert_to_SMILES

bce_with_logit_loss = nn.BCEWithLogitsLoss()
bce_with_logit_loss_sum = nn.BCEWithLogitsLoss(reduction='sum')
eps = 1e-6

class Trainer() :
    def __init__(self, args, run_dir: str) :
        self.args = args
        self.setup_trainer(args)
        self.setup_work_directory(run_dir)
        self.setup_library(args)
        self.setup_data(args)
        self.setup_model(args)

        # Variable for Training
        self.frequency_batch = self.frequency_distribution.repeat(self.train_batch_size, 1)
        self.Z_library = None

    def setup_trainer(self, args) :
        logging.info('Setup Trainer')
        self.device = 'cuda:0' if args.gpus > 0 else 'cpu'
        self.lr = args.lr
        self.max_step = args.max_step
        self.num_negative_samples = args.num_negative_samples
        self.num_validate = args.num_validate
        self.alpha = args.alpha

        self.num_workers = args.num_workers
        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size

        self.val_interval = args.val_interval
        self.save_interval = args.save_interval
        self.log_interval = args.log_interval
        self.print_interval = args.print_interval

    def setup_work_directory(self, run_dir) :
        logging.info('Setup Work Directory')
        self.save_dir = save_dir = os.path.join(run_dir, 'save')
        self.log_dir = log_dir = os.path.join(run_dir, 'log')
        os.mkdir(save_dir)
        os.mkdir(log_dir)
        self.tb_logger = SummaryWriter(log_dir = log_dir)

    def setup_library(self, args) :
        logging.info('Setup Library')
        self.library = library = BRICS_BlockLibrary(args.library_path, use_frequency=True)
        self.library_pygdata_list = [BlockGraphTransform.call(mol) for mol in library]
        self.library_pygbatch = PyGBatch.from_data_list(self.library_pygdata_list).to(self.device)
        self.frequency_distribution = self.library.frequency_distribution.to(self.device) ** args.alpha

    def setup_data(self, args) :
        logging.info('Setup Data')
        # Load Data
        dataframe = pd.read_csv(args.data_path, usecols=['SMILES'] + args.property)
        """
                                                      SMILES      mw   tpsa     logp      qed
        0  CCCCCCC1=NN2C(=N)/C(=C\c3cc(C)n(-c4ccc(C)cc4C)...  461.22  73.81  6.30055  0.38828
        1                 COCC[C@@H](C)C(=O)N(C)Cc1ccc(O)cc1  251.15  49.77  2.02320  0.84121
        2  C=CCn1c(S[C@H](C)c2nc3sc(C)c(C)c3c(=O)[nH]2)nn...  387.12  76.46  4.10964  0.50871
        ...

        >>> dataframe.loc[[0,1]]
                                                      SMILES      mw   tpsa     logp      qed
        0  CCCCCCC1=NN2C(=N)/C(=C\c3cc(C)n(-c4ccc(C)cc4C)...  461.22  73.81  6.30055  0.38828
        1                 COCC[C@@H](C)C(=O)N(C)Cc1ccc(O)cc1  251.15  49.77  2.02320  0.84121

        >>> dataframe.loc[[0,1]].values.tolist()
        [
            [CCCCCCC1=NN2C(=N)/C(=C\c3cc(C)n(-c4ccc(C)cc4C)..., 461.22, 73.81, 6.30055, 0.38828],
            [COCC[C@@H](C)C(=O)N(C)Cc1ccc(O)cc1, 251.15, 49.77, 2.02320, 0.84121],
        ]
        """
        data_pkl = None
        if args.data_pkl_path is not None :
            if os.path.exists(args.data_pkl_path) :
                with open(args.data_pkl_path, 'rb') as f :
                    data_pkl = pickle.load(f)

        # Load Split Index
        with open(args.split_path) as f :
            split_index = [l.strip().split(',') for l in f.readlines()]
        train_index = [int(idx) for label, idx in split_index if label == 'train']
        val_index = [int(idx) for label, idx in split_index if label == 'val']

        # Setup Dataset
        def construct_dataset(dataframe, data_pkl, index) :
            indexed_df = dataframe.loc[index]
            indexed_data = indexed_df.values.tolist()
            molecules = [convert_to_SMILES(row[0], isomericSmiles=False) for row in indexed_data]
            properties = [{key: val for key, val in zip(args.property, row[1:])}
                        for row in indexed_data]
            fragmented_molecules = [data_pkl[idx] for idx in index] if data_pkl is not None else None
            return BBARDataset(molecules, fragmented_molecules, properties, self.library)

        self.train_dataset = construct_dataset(dataframe, data_pkl, train_index)
        self.val_dataset = construct_dataset(dataframe, data_pkl, val_index)

        # Setup Dataloader
        self.train_dataloader = PyGDataLoader(self.train_dataset, args.train_batch_size, \
                                    num_workers = args.num_workers, shuffle = True,
                                    drop_last = True)
           
        self.val_dataloader = PyGDataLoader(self.val_dataset, args.val_batch_size, \
                                    num_workers = args.num_workers, shuffle = True,
                                    drop_last = True)
        
        # Setup Property Mean / Standard Deviation
        self.property_mean_std = OrderedDict()
        for desc in args.property :
            mean, std = dataframe[desc].mean(), dataframe[desc].std()
            self.property_mean_std[desc] = (mean, std)

        del [[dataframe]]
        gc.collect()

        logging.info(f'num of train data: {len(self.train_dataset)}')
        logging.info(f'num of val data: {len(self.val_dataset)}\n')

    def setup_model(self, args) :
        logging.info('Setup Model')
        model_config = OmegaConf.load(args.model_config)
        OmegaConf.resolve(model_config)
        model = BlockConnectionPredictor(model_config, self.property_mean_std)
        model.initialize_parameter()
        self.model = model.to(self.device)
        logging.info(f"number of parameters : {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}\n")

    def fit(self) :
        optimizer, schedular = self.setup_optimizer()
        self.global_step = 0
        self.global_epoch = 0
        self.min_loss = float('inf')
        self.model.train()
        logging.info('Train Start')
        optimizer.zero_grad()
        while self.global_step < self.max_step :
            self.global_epoch += 1
            for batch in self.train_dataloader :
                metrics = self.run_train_step(batch, optimizer)

                self.global_step += 1
                if self.global_step % self.log_interval == 0 :
                    self.log_metrics(metrics, 'TRAIN')
                if self.global_step % self.print_interval == 0 :
                    self.print_metrics(metrics, 'TRAIN')
                if self.global_step % self.save_interval == 0 :
                    save_path = os.path.join(self.save_dir, f'ckpt_{self.global_epoch}_{self.global_step}.tar')
                    self.model.save(save_path)
                if self.global_step == self.max_step :
                    val_loss = self.validate()
                    break
                if self.global_step % self.val_interval == 0 :
                    val_loss = self.validate()
                    schedular.step(val_loss)

        save_path = os.path.join(self.save_dir, 'last.tar')
        self.model.save(save_path)

    def setup_optimizer(self) :
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, \
                min_lr=1e-5)
        return optimizer, schedular

    @torch.no_grad()
    def validate(self) :
        self.model.eval()
        batch_library = self.library_pygbatch
        _, self.Z_library = self.model.building_block_embedding(batch_library)

        metrics_storage = self.get_metrics_storage()
        for _ in range(self.num_validate) :
            for batch in self.val_dataloader :
                metrics = self.run_val_step(batch)
                self.append_metrics(metrics_storage, metrics)
        metrics = self.aggregate_metrics(metrics_storage)
        loss = metrics['loss']
        self.log_metrics(metrics, 'VAL')
        self.print_metrics(metrics, 'VAL  ')

        self.Z_library = None
        self.model.train()

        if loss < self.min_loss :
            self.min_loss = loss
            save_path = os.path.join(self.save_dir, 'best.tar')
            self.model.save(save_path)
        return loss

    def run_train_step(self, batch, optimizer) :
        loss, metrics = self._step(batch, train=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        return metrics

    def run_val_step(self, batch) :
        loss, metrics = self._step(batch, train=False)
        return metrics

    def _step(self, batch, train: bool) :
        """
        Input: 
            batch: (pygbatch_core, condition)
            pygbatch_core: Graph of core molecule
            condition: Dictionary of Tensor
                ex) {'MW': Tensor(200.0, 215.2, 163.6), 'TPSA': Tensor(40.4, 130.5, 74.3)}

            answer:
                pygbatch_core['y_term']: BoolTensor    (N, )   if termination, then True
                pygbatch_core['y_block']: LongTensor   (N, )   block index
                pygbatch_core['y_atom']: BoolTensor    (V, )   for selected atom index, True

        Output:
            loss: FloatTensor (Scalar)
            metric: Dict[str, float]
        """

        pygbatch_core, condition = batch
        pygbatch_core, condition = self.to_device(pygbatch_core, condition)
        y_term, y_block, y_atom = pygbatch_core['y_term'], pygbatch_core['y_block'], pygbatch_core['y_atom']
        y_add = torch.logical_not(y_term)
        
        # Termination Prediction
        x_upd_core, Z_core = self.model.core_molecule_embedding(pygbatch_core, condition)
        logit_term = self.model.get_termination_logit(Z_core)               # (N,)
        
        loss_term = bce_with_logit_loss(logit_term, y_term.float())
        if y_add.sum().item() == 0 :
            loss = loss_term
            metrics = {'loss' : loss.item(), 'loss_term' : loss_term.item()}
            return loss, metrics

        # Block Predicton
        Z_core_add = Z_core[y_add]                                          # (N, Fz_core) -> (N_add, Fz_core)
        positive_sample = y_block[y_add]                                    # (N, ) -> (N_add,)

        ## Positive Sample
        Z_block = self.get_Z_block(positive_sample, train)                  # (N_add, Fz_block)
        p_block = self.model.get_block_priority(Z_core_add, Z_block)        # (N_add,)
        loss_block_pos = (p_block + eps).log().mean().neg()

        ## Negative Samples
        loss_block_neg = 0
        for negative_sample in self.get_negative_samples(positive_sample) :
            Z_block = self.get_Z_block(negative_sample, train)              # (N_add, Fz_block)
            p_block = self.model.get_block_priority(Z_core_add, Z_block)    # (N_add,)
            loss_block_neg += (1. - p_block + eps).log().mean().neg()
        loss_block_neg /= self.num_negative_samples

        # Atom Index Prediction
        """
        Loss Function: Same to CrossEntropyLoss
        It is hard to apply termination mask to PyGBatch, 
          masking is performed after calculate P_atom.
        """
        Z_block_ = torch.zeros_like(Z_core)
        Z_block_[y_add] = Z_block
        logit_atom = self.model.get_atom_logit(
                pygbatch_core, x_upd_core, Z_core, Z_block_
        )

        log_P_atom = scatter_log_softmax(logit_atom, pygbatch_core.batch, dim=-1)
        loss_atom = (log_P_atom * y_atom).sum().neg() / y_add.sum()    # (V, ) -> (V, ) -> scalar

        loss = loss_term + loss_block_pos + loss_block_neg + loss_atom
        metrics = {
            'loss' : loss.item(), 'loss_term' : loss_term.item(),
            'loss_block_pos' : loss_block_pos.item(), 'loss_block_neg' : loss_block_neg.item(),
            'loss_atom' : loss_atom.item()
        }
        return loss, metrics

    def get_Z_block(self, indexs: LongTensor, train: bool) -> FloatTensor :
        if (train is True) or (self.Z_library is None) :
            pygbatch_block = PyGBatch.from_data_list([
                self.library_pygdata_list[idx] for idx in indexs
            ])
            pygbatch_block = pygbatch_block.to(self.device)
            _, Z_block = self.model.building_block_embedding(pygbatch_block)
            return Z_block
        else :
            return self.Z_library[indexs]

    @torch.no_grad()
    def get_negative_samples(self, positive_sample: LongTensor) -> Iterable[LongTensor]:
        """
        positive_sample: LongTensor (N, )  # For Masking
        output: LongTensor (N, )
        """
        batch_size = positive_sample.size(0)
        if self.frequency_batch.size(0) < batch_size :
            freq = self.frequency_distribution.repeat(batch_size, 1)
        else :
            freq = self.frequency_batch[:batch_size]
        freq = freq.scatter(1, positive_sample.unsqueeze(1), 0)
        for _ in range(self.num_negative_samples) :
            neg_idxs = torch.multinomial(freq, 1, True).view(-1)
            yield neg_idxs.tolist()

    def log_metrics(self, metrics, prefix) :
        for key, value in metrics.items() :
            self.tb_logger.add_scalar(f'{prefix}/{key}', value, self.global_step)

    def print_metrics(self, metrics, prefix) :
        loss, tloss, bploss, bnloss, aloss = metrics['loss'], metrics['loss_term'], \
                            metrics['loss_block_pos'], metrics['loss_block_neg'], metrics['loss_atom']
        logging.info(
            f'STEP {self.global_step}\t'
            f'EPOCH {self.global_epoch}\t'
            f'{prefix}\t'
            f'loss: {loss:.3f}\t'
            f'tloss: {tloss:.3f}\t'
            f'ploss: {bploss:.3f}\t'
            f'nloss: {bnloss:.3f}\t'
            f'aloss: {aloss:.3f}\t'
        )

    def get_metrics_storage(self) -> Dict[str, List]:
        return {
            'loss_term': [],
            'loss_block_pos': [],
            'loss_block_neg': [],
            'loss_atom': [],
        }

    def clear_metrics_storage(self, metrics_storage) :
        for key in metrics_storage.keys () :
            metrics_storage[key] = []

    def append_metrics(self, metrics_storage, metrics) :
        metrics_storage['loss_term'].append(metrics['loss_term'])
        if 'loss_atom' in metrics :
            metrics_storage['loss_block_pos'].append(metrics['loss_block_pos'])
            metrics_storage['loss_block_neg'].append(metrics['loss_block_neg'])
            metrics_storage['loss_atom'].append(metrics['loss_atom'])

    def aggregate_metrics(self, metrics_storage) :
        loss_term = np.mean(metrics_storage['loss_term'])
        loss_block_pos = np.mean(metrics_storage['loss_block_pos']) 
        loss_block_neg = np.mean(metrics_storage['loss_block_neg']) 
        loss_atom = np.mean(metrics_storage['loss_atom']) 
        loss = loss_term + loss_block_pos + loss_block_neg + loss_atom
        return {
            'loss_term': loss_term,
            'loss_block_pos': loss_block_pos,
            'loss_block_neg': loss_block_neg,
            'loss_atom': loss_atom,
            'loss': loss,
        }

    def to_device(self, pygbatch, condition) :
        pygbatch = pygbatch.to(self.device)
        condition = {key: val.to(self.device) for key, val in condition.items()}
        return pygbatch, condition

