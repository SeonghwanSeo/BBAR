from rdkit import Chem

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from bbar.utils.typing import GraphVector, NodeVector, PropertyVector, LossScalar

from bbar.model import BlockConnectionPredictor
from bbar.transform import BlockGraphTransform
from bbar.fragmentation import BRICS_BlockLibrary
from .dataset import BBARDataset 
from bbar.utils.common import convert_to_SMILES

"""
The structure of Trainer is partially similar to 
    https://github.com/eladrich/pixel2style2pixel/training/coach.py
"""

class Trainer() :
    def __init__(self, args, run_dir: str) :
        self.args = args
        self.setup_trainer(args)
        self.setup_work_directory(run_dir)
        self.setup_library(args)
        self.setup_data(args)
        self.setup_model(args)

        # Variable for Training
        self.Z_library = None

    def setup_trainer(self, args) :
        logging.info('Setup Trainer')
        self.device = 'cuda:0' if args.gpus > 0 else 'cpu'
        self.lr = args.lr
        self.max_step = args.max_step
        self.num_negative_samples = args.num_negative_samples
        self.num_validate = args.num_validate
        self.alpha = args.alpha
        self.condition_noise = args.condition_noise

        self.lambda_term = args.lambda_term
        self.lambda_property = args.lambda_property
        self.lambda_block = args.lambda_block
        self.lambda_atom = args.lambda_atom

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
        self.model_config_path = os.path.join(run_dir, 'model_config.yaml')
        os.mkdir(save_dir)
        os.mkdir(log_dir)
        self.tb_logger = SummaryWriter(log_dir = log_dir)

    def setup_library(self, args) :
        logging.info('Setup Library')
        library_path = os.path.join(args.data_dir, 'library.csv')
        self.library = library = BRICS_BlockLibrary(library_path, use_frequency=True)
        self.library_frequency = self.library.frequency_distribution ** args.alpha
        self.library_pygdata_list = [BlockGraphTransform.call(mol) for mol in library]
        self.library_pygbatch = PyGBatch.from_data_list(self.library_pygdata_list)

    def setup_data(self, args) :
        logging.info('Setup Data')
        # Load Data
        data_path = os.path.join(args.data_dir, 'data.csv')
        dataframe = pd.read_csv(data_path, usecols=['SMILES'] + args.property)
        """
                                                      SMILES      mw   tpsa     logp      qed
        0  CCCCCCC1=NN2C(=N)/C(=C\c3cc(C)n(-c4ccc(C)cc4C)...  461.22  73.81  6.30055  0.38828
        1                 COCC[C@@H](C)C(=O)N(C)Cc1ccc(O)cc1  251.15  49.77  2.02320  0.84121
        2  C=CCn1c(S[C@H](C)c2nc3sc(C)c(C)c3c(=O)[nH]2)nn...  387.12  76.46  4.10964  0.50871
        ...
        >>> dataframe.loc[[0,1]].values.tolist()
        [
          [CCCCCCC1=NN2C(=N)/C(=C\c3cc(C)n(-c4ccc(C)cc4C)..., 461.22, 73.81, 6.30055, 0.38828],
          [COCC[C@@H](C)C(=O)N(C)Cc1ccc(O)cc1, 251.15, 49.77, 2.02320, 0.84121],
        ]
        """
        data_pkl_path = os.path.join(args.data_dir, 'data.pkl')
        if os.path.exists(data_pkl_path) :
            with open(data_pkl_path, 'rb') as f :
                data_pkl = pickle.load(open(data_pkl_path, 'rb'))
        else :
            data_pkl = None

        # Load Split Index
        split_path = os.path.join(args.data_dir, 'split.csv')
        with open(split_path) as f :
            split_index = [l.strip().split(',') for l in f.readlines()]
        train_index = [int(idx) for label, idx in split_index if label == 'train']
        val_index = [int(idx) for label, idx in split_index if label == 'val']

        # Setup Dataset
        def construct_dataset(dataframe, data_pkl, index, train: bool) :
            indexed_df = dataframe.loc[index]
            indexed_data = indexed_df.values.tolist()
            molecules = [row[0] for row in indexed_data]
            properties = [{key: val for key, val in zip(args.property, row[1:])}
                        for row in indexed_data]
            fragmented_molecules = [data_pkl[idx] for idx in index] if data_pkl is not None else None
            return BBARDataset(molecules, fragmented_molecules, properties, 
                    self.library, self.library_pygdata_list, self.library_frequency, self.num_negative_samples, 
                    train
            )

        self.train_dataset = construct_dataset(dataframe, data_pkl, train_index, train = True)
        self.val_dataset = construct_dataset(dataframe, data_pkl, val_index, train = False)

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
        OmegaConf.save(model_config, self.model_config_path)
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
                    self.log_metrics(metrics, prefix='TRAIN')
                if self.global_step % self.print_interval == 0 :
                    self.print_metrics(metrics, prefix='TRAIN')
                if self.global_step % self.save_interval == 0 :
                    save_path = os.path.join(self.save_dir, f'ckpt_{self.global_epoch}_{self.global_step}.tar')
                    self.model.save(save_path)

                if self.global_step % self.val_interval == 0 :
                    val_loss_dict = self.validate()
                    schedular.step(val_loss_dict['loss'])

                if self.global_step == self.max_step :
                    break

        self.validate()
        save_path = os.path.join(self.save_dir, 'last.tar')
        self.model.save(save_path)
        logging.info('Train Finish')

    def setup_optimizer(self) :
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, \
                min_lr=1e-5)
        return optimizer, schedular

    @torch.no_grad()
    def validate(self) :
        self.model.eval()
        batch_library = self.library_pygbatch.to(self.device)
        _, self.Z_library = self.model.building_block_embedding(batch_library)

        agg_loss_dict = []
        for _ in range(self.num_validate) :
            for batch in self.val_dataloader :
                loss_dict = self.run_val_step(batch)
                agg_loss_dict.append(loss_dict)
        loss_dict = self.aggregate_loss_dict(agg_loss_dict)

        loss = loss_dict['loss']
        if loss < self.min_loss :
            self.min_loss = loss
            save_path = os.path.join(self.save_dir, 'best.tar')
            self.model.save(save_path)
            self.print_metrics(loss_dict, prefix='VAL* ')
        else :
            self.print_metrics(loss_dict, prefix='VAL  ')
        self.log_metrics(loss_dict, prefix='VAL')

        self.Z_library = None
        self.library_pygbatch.to('cpu')
        self.model.train()

        return loss_dict

    def run_train_step(self, batch, optimizer) :
        metrics = self._step(batch, train=True)
        loss, loss_dict = self.calc_loss(metrics)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        return loss_dict

    def run_val_step(self, batch) :
        metrics = self._step(batch, train=False)
        loss, loss_dict = self.calc_loss(metrics)
        return loss_dict 

    def _step(self, batch, train: bool) :
        """
        Input: 
            pygbatch_core: Graph of core molecule
            condition: Dictionary of Tensor
                ex) {'MW': Tensor(200.0, 215.2, 163.6), 'TPSA': Tensor(40.4, 130.5, 74.3)}
            pos_block: Union[PyGBatch, LongTensor]
                - train: PyGBatch 
                - val: LongTensor
            neg_blocks: Union[List[PyGBatch], List[LongTensor]]
                - train: List[PyGBatch]
                - val: List[LongTensor]

            answer:
                pygbatch_core['y_term']: BoolTensor    (N, )   if termination, then True
                pygbatch_core['y_atom']: BoolTensor    (V, )   for selected atom index, True

        Output:
            metrics: Dict[str, LossScalar]
        """

        batch = self.to_device(batch)
        pygbatch_core, condition, pos_block, neg_blocks = batch
        y_term, y_atom = pygbatch_core['y_term'], pygbatch_core['y_atom']
        y_add = torch.logical_not(y_term)

        metrics: Dict[str, LossScalar] = {}

        # Standardize Property
        cond = self.standardize_condition(condition)
        
        # Core Molecule Embedding
        x_upd_core, Z_core = self.core_mol_embedding(pygbatch_core)

        # Property Prediction (Only for terminated one)
        if y_term.sum().item() > 0 and self.lambda_property > 0 :
            loss_property = self.property_prediction(Z_core, cond, y_term)
            metrics['loss_property'] = loss_property

        # Condition Embedding
        node2graph_core = pygbatch_core.batch
        x_upd_core, Z_core = self.condition_embedding(x_upd_core, Z_core, cond, node2graph_core)

        # Termination Prediction
        loss_term = self.termination_prediction(Z_core, y_term)
        metrics['loss_term'] = loss_term
        
        if y_add.sum().item() == 0 :
            return metrics

        # Block Prediction
        ## Positive Sample
        Z_pos_block = self.building_block_embedding(pos_block, train = train)
        loss_pos_block = self.building_block_priority(Z_core, Z_pos_block, y_add, 'positive')
        metrics['loss_pos_block'] = loss_pos_block

        ## Negative Samples
        loss_neg_block = 0.
        for neg_block in neg_blocks :
            Z_neg_block = self.building_block_embedding(neg_block, train = train)
            loss_neg_block += self.building_block_priority(Z_core, Z_neg_block, y_add, 'negative')
        loss_neg_block /= self.num_negative_samples
        metrics['loss_neg_block'] = loss_neg_block

        # Atom Index Prediction
        loss_atom = self.atom_prediction(pygbatch_core, x_upd_core, Z_core, Z_pos_block, y_atom)
        metrics['loss_atom'] = loss_atom

        return metrics
    
    def standardize_condition(self, condition: Dict[str, FloatTensor]) -> PropertyVector :
        cond = self.model.standardize_property(condition)
        return cond

    def core_mol_embedding(self, pygbatch_core: PyGBatch) -> Tuple[NodeVector, GraphVector]:
        x_upd_core, Z_core = self.model.core_molecule_embedding(pygbatch_core)
        return x_upd_core, Z_core

    def property_prediction(self, Z_core: GraphVector, cond: PropertyVector, \
                            y_term: BoolTensor) -> LossScalar:
        # For terminated datapoint, the property of core molecule is same to condition.
        y_property = cond[y_term]
        Z_core = Z_core[y_term]
        y_hat_property = self.model.get_property_prediction(Z_core)
        loss_property = F.mse_loss(y_hat_property, y_property, reduction='mean')
        return loss_property

    def condition_embedding(self, x_upd_core: NodeVector, Z_core: GraphVector,
                            cond: PropertyVector, node2graph_core: LongTensor
    ) -> Tuple[NodeVector, GraphVector] :
        # Embed condition information to node feature and graph feature
        if self.condition_noise > 0 :
            cond += torch.randn_like(cond) * self.condition_noise
        x_upd_core, Z_core = self.model.condition_embedding(x_upd_core, Z_core, \
                                                        cond, node2graph_core)
        return x_upd_core, Z_core
       
    def termination_prediction(self, Z_core: GraphVector, y_term: BoolTensor) -> LossScalar :
        logit_term = self.model.get_termination_logit(Z_core)
        loss_term = F.binary_cross_entropy_with_logits(logit_term, y_term.float())
        return loss_term

    def building_block_embedding(self, block: Union[PyGBatch, LongTensor], \
                                train: bool) -> GraphVector :
        if train :
            pygbatch_block = block
            _, Z_block = self.model.building_block_embedding(pygbatch_block)
        else :
            block_idx = block
            Z_block = self.Z_library[block_idx]
        return Z_block

    def building_block_priority(self, Z_core: GraphVector, Z_block: GraphVector,
                                y_add: BoolTensor, sample_type: str) -> LossScalar :
        p_block = self.model.get_block_priority(Z_core, Z_block)
        eps = 1e-6
        if sample_type == 'positive' :
            loss_block = ((p_block + eps).log() * y_add).sum().neg() / y_add.sum()
        else :
            loss_block = ((1 - p_block + eps).log() * y_add).sum().neg() / y_add.sum()
        return loss_block

    def atom_prediction(
        self, 
        pygbatch_core: PyGBatch, 
        x_upd_core: NodeVector, 
        Z_core: GraphVector, 
        Z_block: GraphVector,
        y_atom: BoolTensor,
    ) -> LossScalar :
        logit_atom = self.model.get_atom_logit(
                pygbatch_core, x_upd_core, Z_core, Z_block
        )
        log_P_atom = scatter_log_softmax(logit_atom, pygbatch_core.batch, dim=-1)
        loss_atom = (log_P_atom * y_atom).sum().neg() / y_atom.sum()    # CrossEntropyLoss
        return loss_atom

    def calc_loss(self, metrics) -> Tuple[LossScalar, Dict[str, float]]:
        loss: LossScalar = 0
        
        loss_term = metrics['loss_term']
        loss += loss_term * self.lambda_term

        if 'loss_property' in metrics :
            loss_property = metrics['loss_property']
            loss += loss_property * self.lambda_property

        if 'loss_pos_block' in metrics :
            loss_pos_block, loss_neg_block, loss_atom = \
                    metrics['loss_pos_block'], metrics['loss_neg_block'], metrics['loss_atom']
            loss += (loss_pos_block + loss_neg_block) * self.lambda_block
            loss += loss_atom * self.lambda_atom

        loss_dict = {key: loss.item() for key, loss in metrics.items()}
        loss_dict['loss'] = loss.item()

        return loss, loss_dict

    def aggregate_loss_dict(self, agg_loss_dict):
        sum_loss = {}
        num_loss = {}
        for loss_dict in agg_loss_dict:
            for key, loss in loss_dict.items():
                sum_loss[key] = sum_loss.setdefault(key, 0) + loss
                num_loss[key] = num_loss.setdefault(key, 0) + 1
        mean_loss_dict = {key: sum_loss[key] / num_loss[key] for key in sum_loss.keys()}
        return mean_loss_dict

    def log_metrics(self, metrics, prefix) :
        for key, value in metrics.items() :
            self.tb_logger.add_scalars(f'scalar/{key}', {prefix: value}, self.global_step)

    def print_metrics(self, metrics, prefix) :
        loss, tloss = metrics['loss'], metrics['loss_term']
        
        ploss = metrics.get('loss_property', float('NaN'))
        pbloss = metrics.get('loss_pos_block', float('NaN'))
        nbloss = metrics.get('loss_neg_block', float('NaN'))
        aloss = metrics.get('loss_atom', float('NaN'))

        logging.info(
            f'STEP {self.global_step}\t'
            f'EPOCH {self.global_epoch}\t'
            f'{prefix}\t'
            f'loss: {loss:.3f}\t'
            f'term: {tloss:.3f}\t'
            f'prop: {ploss:.3f}\t'
            f'pblock: {pbloss:.3f}\t'
            f'nblock: {nbloss:.3f}\t'
            f'atom: {aloss:.3f}\t'
        )

    def to_device(self, batch) :
        pygbatch_core, condition, pos_block, *neg_blocks = batch
        pygbatch_core = pygbatch_core.to(self.device)
        condition = {key: val.to(self.device) for key, val in condition.items()}
        pos_block = pos_block.to(self.device)
        neg_blocks = [block.to(self.device) for block in neg_blocks]
        return pygbatch_core, condition, pos_block, neg_blocks
