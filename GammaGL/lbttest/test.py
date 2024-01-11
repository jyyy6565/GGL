import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from gammagl.datasets import MoleculeNet

dataset = MoleculeNet(root='./', name='ESOL')

print(dataset)


