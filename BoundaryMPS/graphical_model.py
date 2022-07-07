import torch
from scipy.linalg import sqrtm
import numpy as np


def identity_tensor(shape, dtype=torch.float64):
    tensor = torch.zeros(np.prod(shape), dtype=dtype)
    tensor[0] = tensor[-1] = 1
    return tensor.reshape(shape)


def boltzmann_matrix(beta, coupling, dtype=torch.float64):
    return torch.exp(beta * coupling * torch.tensor([[1, -1], [-1, 1]], dtype=dtype))


def self_sqrtm(m):
    u, s, v = torch.linalg.svd(m)
    return (u @ torch.diag(s.sqrt()), torch.diag(s.sqrt()) @ v)

class URand:
    def __init__(self, L, D, lam, seed, dtype=torch.float64) -> None:
        self.L = L
        self.n = L ** 2
        self.D = D
        self.dtype = dtype
        self.seed = seed
        self.lam = lam
        pass
    
    def construct_tn(self):
        torch.manual_seed(self.seed)
        tensor_network = []
        for i in range(self.L):
            tensor_network.append([])
            for j in range(self.L):
                size = [1 if i == 0 else self.D, 1 if j == 0 else self.D, 1 if i == self.L-1 else self.D, 1 if j == self.L-1 else self.D]
                tensor_network[i].append(torch.rand(size, dtype=self.dtype) * (1-self.lam) + self.lam)
        return tensor_network

class lattice2D_obc:
    def __init__(self, L, couplings, beta, dtype=torch.float64) -> None:
        self.L = L
        self.n = L ** 2
        self.couplings = couplings
        assert self.couplings.shape[0] == self.couplings.shape[1] == self.n
        self.edges = []
        for i, j in self.couplings.nonzero().tolist():
            if i < j:
                self.edges.append((i, j))
        self.beta = beta
        self.dtype = dtype
    
    def id_mapping(self, i, j):
        return i * self.L + j

    def construct_tn(self, gauge='self_sqrtm'):
        couplings_split_archived = {}.fromkeys(self.edges)
        for i in range(len(self.edges)):
            m, n = self.edges[i]
            if gauge == 'sqrtm':
                matrix = torch.from_numpy(sqrtm(np.exp(self.beta * self.couplings[m, n] * np.array([[1, -1], [-1, 1]], dtype=np.float64))))
                couplings_split_archived[self.edges[i]] = (matrix, matrix)
            elif gauge == 'self_sqrtm':
                bmatrix = boltzmann_matrix(self.beta, self.couplings[m, n], self.dtype)
                couplings_split_archived[self.edges[i]] = self_sqrtm(bmatrix)
            elif gauge == 'qr':
                couplings_split_archived[self.edges[i]] = torch.linalg.qr(boltzmann_matrix(self.beta, self.couplings[m, n], self.dtype))
            elif gauge == 'trivial':
                couplings_split_archived[self.edges[i]] = (boltzmann_matrix(self.beta, self.couplings[m, n], self.dtype), torch.eye(2, dtype=self.dtype))
            else:
                raise ValueError('Unkown gauge type.')

        trivial_component = torch.tensor([[1], [1]], dtype=self.dtype)
        tensor_network = [[] for i in range(self.L)]
        for i in range(self.L):
            for j in range(self.L):
                surroundings = [
                    (self.id_mapping(i-1, j), self.id_mapping(i, j)), 
                    (self.id_mapping(i, j-1), self.id_mapping(i, j)), 
                    (self.id_mapping(i+1, j), self.id_mapping(i, j)), 
                    (self.id_mapping(i, j+1), self.id_mapping(i, j))
                ]
                merge_to_lattice_tensors, merge_eqs, result_eq = [], [], ''
                for k in range(len(surroundings)):
                    m, n = surroundings[k]
                    if (m, n) in self.edges:
                        merge_to_lattice_tensors.append(couplings_split_archived[(m, n)][1])
                        merge_eqs.append(chr(122-k)+chr(97+k))
                    elif (n, m) in self.edges:
                        merge_to_lattice_tensors.append(couplings_split_archived[(n, m)][0])
                        merge_eqs.append(chr(97+k) + chr(122-k))
                    else:
                        merge_to_lattice_tensors.append(trivial_component)
                        merge_eqs.append(chr(97+k) + chr(122-k))
                    result_eq += chr(122-k)
                eq, tensors = ','.join(['abcd'] + merge_eqs) + '->' + result_eq, [identity_tensor((2,2,2,2))] + merge_to_lattice_tensors
                tensor_network[i].append(torch.einsum(eq, tensors))
        return tensor_network