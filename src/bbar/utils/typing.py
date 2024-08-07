from torch import FloatTensor

"""
NodeVector      (V, F)
EdgeVector      (E, F)
GlobalVector    (N, F)
PropertyVector  (N, F)
GraphVector     (N, F)
Loss            (1,) scalar
"""
NodeVector = EdgeVector = GlobalVector = PropertyVector = GraphVector = \
        LossScalar = FloatTensor

SMILES = str

