import numpy as np

class kacward:
    """
    Kac-Ward exact Ising
    See Theorem 1 of https://arxiv.org/abs/1011.3494
    """

    def __init__(self, L, J, beta):
        self.L = L
        self.beta = beta
        self.phi = np.array([[0., np.pi / 2, -np.pi / 2, np.nan],
                             [-np.pi / 2, 0.0, np.nan, np.pi / 2],
                             [np.pi / 2, np.nan, 0.0, -np.pi / 2],
                             [np.nan, -np.pi / 2, np.pi / 2, 0]
                             ])

        K = np.ones((self.L ** 2, 4)) * self.beta
        for i in range(self.L ** 2):
            for j in range(4):
                site = self.neighborsite(i, j)
                if site is not None:
                    K[i, j] *= J[i, site].item()
        self.lnZ = self.kacward_solution(K)

    def logcosh(self, x):
        xp = np.abs(x)
        if xp < 12:
            return np.log(np.cosh(x))
        else:
            return xp - np.log(2.)

    def neighborsite(self, i, n):
        """
        The coordinate system is geometrically left->right, down -> up
              y|
               |
               |
               |________ x
              (0,0)
        So as a definition, l means x-1, r means x+1, u means y+1, and d means y-1
        """
        x = i % self.L
        y = i // self.L  # y denotes
        site = None
        # ludr :
        if n == 0:
            if x - 1 >= 0:
                site = (x - 1) + y * self.L
        elif n == 1:
            if y + 1 < self.L:
                site = x + (y + 1) * self.L
        elif n == 2:
            if y - 1 >= 0:
                site = x + (y - 1) * self.L
        elif n == 3:
            if x + 1 < self.L:
                site = (x + 1) + y * self.L
        return site

    # K: ludr
    def kacward_solution(self, K):
        V = self.L ** 2  # number of vertex
        E = 2 * (V - self.L)  # number of edge

        D = np.zeros((2 * E, 2 * E), np.complex128)
        ij = 0
        ijdict = {}
        for i in range(V):
            for j in range(4):
                if self.neighborsite(i, j) is not None:
                    D[ij, ij] = np.tanh(K[i, j])
                    ijdict[(i, j)] = ij  # mapping for (site, neighbor) to index
                    ij += 1

        A = np.zeros((2 * E, 2 * E), np.complex128)
        for i in range(V):
            for j in range(4):
                for l in range(4):
                    k = self.neighborsite(i, j)
                    if (not np.isnan(self.phi[j, l])) and (k is not None) and (self.neighborsite(k, l) is not None):
                        ij = ijdict[(i, j)]
                        kl = ijdict[(k, l)]
                        A[ij, kl] = np.exp(1J * self.phi[j, l] / 2.)

        res = V * np.log(2)
        for i in range(V):
            for j in [1, 3]:  # only u, r to avoid double counting
                if self.neighborsite(i, j) is not None:
                    res += self.logcosh(K[i, j])
        _, logdet = np.linalg.slogdet(np.eye(2 * E, 2 * E, dtype=np.float64) - A @ D)
        res += 0.5 * logdet

        return res