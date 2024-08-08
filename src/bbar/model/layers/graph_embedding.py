import torch
import torch.nn as nn

from typing import Optional, Tuple, Union
from torch import FloatTensor, LongTensor
from torch_geometric import nn as pyg_nn
from torch_geometric.data import Data as PyGData, Batch as PyGBatch
from torch_geometric.typing import Adj

from . import block
from bbar.utils.typing import NodeVector, EdgeVector, GlobalVector, GraphVector

class GraphEmbeddingModel(nn.Module):
    def __init__(
        self,
        node_input_dim: int, 
        edge_input_dim: int,
        global_input_dim: Optional[int] = 0, 
        hidden_dim: int = 128,
        graph_vector_dim: Optional[int] = 0, 
        n_block: int = 2, 
        dropout: float = 0.0,
    ) :
        super(GraphEmbeddingModel, self).__init__()

        global_input_dim = global_input_dim or 0
        graph_vector_dim = graph_vector_dim or 0

        self.node_embedding = block.Linear(
            input_dim = node_input_dim + global_input_dim,
            output_dim = hidden_dim,
            activation = 'SiLU'
        )
        self.edge_embedding = block.Linear(
            input_dim = edge_input_dim,
            output_dim = hidden_dim,
            activation = 'SiLU'
        )
        self.blocks = nn.ModuleList([
            block.ResidualBlock(
                node_dim = hidden_dim, edge_dim = hidden_dim,
                activation = 'SiLU', layer_norm = None, dropout = dropout)
            for _ in range(n_block)
        ])

        self.final_node_embedding = block.Linear(
            input_dim = hidden_dim + node_input_dim,
            output_dim = hidden_dim,
            activation = 'SiLU',
            dropout = dropout,
        )

        if graph_vector_dim > 0 :
            self.readout = block.Readout(
                node_dim = hidden_dim,
                hidden_dim = graph_vector_dim,
                output_dim = graph_vector_dim,
                global_input_dim = global_input_dim,
                activation = 'SiLU',
                dropout = dropout
            )
        else :
            self.readout = None 

    def forward(
        self,
        x_inp: NodeVector,
        edge_index: Adj,
        edge_attr: EdgeVector,
        global_x: Optional[GlobalVector] = None,
        node2graph: Optional[LongTensor] = None,
    ) -> Tuple[NodeVector, Optional[GraphVector]] :
        """
        Input :
            x_inp: input node feature of graph          (V, Fv)
            edge_index: edge index of graph             (2, E)
            edge_attr: input edge attr of graph         (E, Fe)
            global_x: input graph feature such as condition (optional)   
                                                        (N, Fc)
            node2graph: map node to graph (optional)    (V,)

        Output:
            x_upd: updated node feature                 (V, Fh)
            Z: latent vector of graph (graph vector)    (N, Fz)
                if graph_vector_dim is 0, Z is None
        """

        x = self.concat(x_inp, global_x, node2graph)

        x_emb = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        for convblock in self.blocks :
            x_emb = convblock(x_emb, edge_index, edge_attr, node2graph)

        x_emb = torch.cat([x_emb, x_inp], dim=-1)
        x_emb = self.final_node_embedding(x_emb)

        if self.readout is not None :
            Z = self.readout(x_emb, node2graph, global_x)
        else :
            Z = None

        return x_emb, Z

    def forward_batch(self, batch: Union[PyGBatch, PyGData]) -> Tuple[NodeVector, Optional[GraphVector]] :
        if isinstance(batch, PyGBatch) :
            node2graph = batch.batch
        else :
            node2graph = None
        
        global_x = batch.get('global_x', None)

        return self.forward(batch.x, batch.edge_index, batch.edge_attr, global_x, node2graph)

    def concat(self, x: NodeVector, global_x: Optional[GlobalVector], node2graph: LongTensor) -> FloatTensor :
        if global_x is not None :
            if node2graph is None :
                global_x = global_x.repeat(x.size(0), 1)
            else :
                global_x = global_x[node2graph]
            x = torch.cat([x, global_x], dim=-1)
        return x
