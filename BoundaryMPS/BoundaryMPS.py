import torch, logging, math

class BoundaryMPS:
    def __init__(self, op) -> None:
        self.op = op
        self.einsum_op = op.einsum
        self.qr_op = op.linalg.qr
        self.svd_op = op.linalg.svd
        self.permute_op = op.permute if op == torch else op.transpose
        self.norm_op = op.linalg.norm
        self.diag_op = op.diag
        pass

    def __call__(self, mpos, contraction_way, chi, cutoff):
        self.mpos = mpos
        self.length = len(mpos)
        self.width = len(mpos[0])
        logging.info(f'{self.length} {self.width} 2dTN {contraction_way}')
        for i in range(1, self.length):
            assert len(mpos[i]) == self.width
        if contraction_way == 'normal':
            return self.normal_bmps(chi, cutoff)
        elif contraction_way == 'three':
            return self.threeway_bmps(chi, cutoff)
        elif contraction_way == 'four':
            return self.fourway_bmps(chi, cutoff)
        else:
            raise ValueError('Unkown contraction strategy choice.')
    
    def contract(self, loc_i, loc_j):
        i0, i1 = loc_i
        j0, j1 = loc_j
        assert abs(i0-j0) + abs(i1-j1) == 1
        logging.info(f'{loc_i} {loc_j} {list(self.mpos[i0][i1].shape)} {list(self.mpos[j0][j1].shape)}')
        if i0 == j0 and i1 < j1:
            eq = "ijkl,albc->iajkbc"
            rshape = (self.mpos[i0][i1].shape[0] * self.mpos[j0][j1].shape[0], self.mpos[i0][i1].shape[1], self.mpos[i0][i1].shape[2] * self.mpos[j0][j1].shape[2], self.mpos[j0][j1].shape[3])
            merge_dim = max(rshape[0], rshape[2])
        elif i0 == j0 and i1 > j1:
            eq = "ijkl,abcj->aibckl"
            rshape = (self.mpos[i0][i1].shape[0] * self.mpos[j0][j1].shape[0], self.mpos[j0][j1].shape[1], self.mpos[i0][i1].shape[2] * self.mpos[j0][j1].shape[2], self.mpos[i0][i1].shape[3])
            merge_dim = max(rshape[0], rshape[2])
        elif i0 < j0 and i1 == j1:
            eq = "ijkl,kabc->ijablc"
            rshape = (self.mpos[i0][i1].shape[0], self.mpos[i0][i1].shape[1]*self.mpos[j0][j1].shape[1], self.mpos[j0][j1].shape[2], self.mpos[i0][i1].shape[3] * self.mpos[j0][j1].shape[3])
            merge_dim = max(rshape[1], rshape[3])
        elif i0 > j0 and i1 == j1:
            eq = "ijkl,abic->abjkcl"
            rshape = (self.mpos[j0][j1].shape[0], self.mpos[i0][i1].shape[1]*self.mpos[j0][j1].shape[1], self.mpos[i0][i1].shape[2], self.mpos[i0][i1].shape[3] * self.mpos[j0][j1].shape[3])
            merge_dim = max(rshape[1], rshape[3])
        self.mpos[i0][i1] = self.einsum_op(eq, self.mpos[i0][i1], self.mpos[j0][j1]).reshape(rshape)
        logging.info(f'{loc_i} {list(self.mpos[i0][i1].shape)}')
        return merge_dim

    def canonical(self, loc_i, loc_j):
        i0, i1 = loc_i
        j0, j1 = loc_j
        assert abs(i0-j0) + abs(i1-j1) == 1
        logging.info(f'{loc_i} {loc_j} {list(self.mpos[i0][i1].shape)} {list(self.mpos[j0][j1].shape)}')
        if i0 == j0 and i1 < j1:
            eq = "ijkl,albc->ijkabc"
            rshape = (self.mpos[i0][i1].shape[0]*self.mpos[i0][i1].shape[1]*self.mpos[i0][i1].shape[2], self.mpos[j0][j1].shape[0]*self.mpos[j0][j1].shape[2]*self.mpos[j0][j1].shape[3])
            rrshape = (
                (self.mpos[i0][i1].shape[0], self.mpos[i0][i1].shape[1], self.mpos[i0][i1].shape[2], -1),
                (-1, self.mpos[j0][j1].shape[0], self.mpos[j0][j1].shape[2], self.mpos[j0][j1].shape[3])
            )
            permute_dims = (
                (0, 1, 2, 3),
                (1, 0, 2, 3)
            )
        elif i0 == j0 and i1 > j1:
            eq = "ijkl,abcj->iklabc"
            rshape = (self.mpos[i0][i1].shape[0]*self.mpos[i0][i1].shape[2]*self.mpos[i0][i1].shape[3], self.mpos[j0][j1].shape[0]*self.mpos[j0][j1].shape[1]*self.mpos[j0][j1].shape[2])
            rrshape = (
                (self.mpos[i0][i1].shape[0], self.mpos[i0][i1].shape[2], self.mpos[i0][i1].shape[3], -1),
                (-1, self.mpos[j0][j1].shape[0], self.mpos[j0][j1].shape[1], self.mpos[j0][j1].shape[2])
            )
            permute_dims = (
                (0, 3, 1, 2),
                (1, 2, 3, 0)
            )
        elif i0 < j0 and i1 == j1:
            eq = "ijkl,kabc->ijlabc"
            rshape = (self.mpos[i0][i1].shape[0]*self.mpos[i0][i1].shape[1]*self.mpos[i0][i1].shape[3], self.mpos[j0][j1].shape[1]*self.mpos[j0][j1].shape[2]*self.mpos[j0][j1].shape[3])
            rrshape = (
                (self.mpos[i0][i1].shape[0], self.mpos[i0][i1].shape[1], self.mpos[i0][i1].shape[3], -1),
                (-1, self.mpos[j0][j1].shape[1], self.mpos[j0][j1].shape[2], self.mpos[j0][j1].shape[3])
            )
            permute_dims = (
                (0, 1, 3, 2),
                (0, 1, 2 ,3)
            )
        elif i0 > j0 and i1 == j1:
            eq = "ijkl,abic->jklabc"
            rshape = (self.mpos[i0][i1].shape[1]*self.mpos[i0][i1].shape[2]*self.mpos[i0][i1].shape[3], self.mpos[j0][j1].shape[0]*self.mpos[j0][j1].shape[1]*self.mpos[j0][j1].shape[3])
            rrshape = (
                (self.mpos[i0][i1].shape[1], self.mpos[i0][i1].shape[2], self.mpos[i0][i1].shape[3], -1),
                (-1, self.mpos[j0][j1].shape[0], self.mpos[j0][j1].shape[1], self.mpos[j0][j1].shape[3])
            )
            permute_dims = (
                (3, 0, 1, 2),
                (1, 2, 0, 3)
            )
        q, r = self.qr_op(self.einsum_op(eq, self.mpos[i0][i1], self.mpos[j0][j1]).reshape(rshape))
        self.mpos[i0][i1] = self.permute_op(q.reshape(rrshape[0]), permute_dims[0])
        self.mpos[j0][j1] = self.permute_op(r.reshape(rrshape[1]), permute_dims[1])
        norm = self.norm_op(self.mpos[j0][j1])
        self.mpos[j0][j1] /= norm
        logging.info(f'{loc_i} {list(self.mpos[i0][i1].shape)}')
        return norm.item() if self.op == torch else norm
    
    def rounding(self, loc_i, loc_j, chi, cutoff):
        i0, i1 = loc_i
        j0, j1 = loc_j
        assert abs(i0-j0) + abs(i1-j1) == 1
        logging.info(f'{loc_i} {loc_j} {list(self.mpos[i0][i1].shape)} {list(self.mpos[j0][j1].shape)}')
        if i0 == j0 and i1 < j1:
            eq = "ijkl,albc->ijkabc"
            rshape = (self.mpos[i0][i1].shape[0]*self.mpos[i0][i1].shape[1]*self.mpos[i0][i1].shape[2], self.mpos[j0][j1].shape[0]*self.mpos[j0][j1].shape[2]*self.mpos[j0][j1].shape[3])
            rrshape = (
                (self.mpos[i0][i1].shape[0], self.mpos[i0][i1].shape[1], self.mpos[i0][i1].shape[2], -1),
                (-1, self.mpos[j0][j1].shape[0], self.mpos[j0][j1].shape[2], self.mpos[j0][j1].shape[3])
            )
            permute_dims = (
                (0, 1, 2, 3),
                (1, 0, 2, 3)
            )
        elif i0 == j0 and i1 > j1:
            eq = "ijkl,abcj->iklabc"
            rshape = (self.mpos[i0][i1].shape[0]*self.mpos[i0][i1].shape[2]*self.mpos[i0][i1].shape[3], self.mpos[j0][j1].shape[0]*self.mpos[j0][j1].shape[1]*self.mpos[j0][j1].shape[2])
            rrshape = (
                (self.mpos[i0][i1].shape[0], self.mpos[i0][i1].shape[2], self.mpos[i0][i1].shape[3], -1),
                (-1, self.mpos[j0][j1].shape[0], self.mpos[j0][j1].shape[1], self.mpos[j0][j1].shape[2])
            )
            permute_dims = (
                (0, 3, 1, 2),
                (1, 2, 3, 0)
            )
        elif i0 < j0 and i1 == j1:
            eq = "ijkl,kabc->ijlabc"
            rshape = (self.mpos[i0][i1].shape[0]*self.mpos[i0][i1].shape[1]*self.mpos[i0][i1].shape[3], self.mpos[j0][j1].shape[1]*self.mpos[j0][j1].shape[2]*self.mpos[j0][j1].shape[3])
            rrshape = (
                (self.mpos[i0][i1].shape[0], self.mpos[i0][i1].shape[1], self.mpos[i0][i1].shape[3], -1),
                (-1, self.mpos[j0][j1].shape[1], self.mpos[j0][j1].shape[2], self.mpos[j0][j1].shape[3])
            )
            permute_dims = (
                (0, 1, 3, 2),
                (0, 1, 2 ,3)
            )
        elif i0 > j0 and i1 == j1:
            eq = "ijkl,abic->jklabc"
            rshape = (self.mpos[i0][i1].shape[1]*self.mpos[i0][i1].shape[2]*self.mpos[i0][i1].shape[3], self.mpos[j0][j1].shape[0]*self.mpos[j0][j1].shape[1]*self.mpos[j0][j1].shape[3])
            rrshape = (
                (self.mpos[i0][i1].shape[1], self.mpos[i0][i1].shape[2], self.mpos[i0][i1].shape[3], -1),
                (-1, self.mpos[j0][j1].shape[0], self.mpos[j0][j1].shape[1], self.mpos[j0][j1].shape[3])
            )
            permute_dims = (
                (3, 0, 1, 2),
                (1, 2, 0, 3)
            )
        u, s, v = self.svd_op(self.einsum_op(eq, self.mpos[i0][i1], self.mpos[j0][j1]).reshape(rshape))
        chi_new = max(min(len(s[s>cutoff]), chi), 1)
        logging.info(f'{chi_new}/{len(s)} bound will be saved, error {(s[chi_new:]).sum()}')
        self.error += (s[chi_new:]).sum().item() if self.op == torch else (s[chi_new:]).sum()
        self.mpos[i0][i1] = self.permute_op(u[:, :chi_new].reshape(rrshape[0]), permute_dims[0])
        self.mpos[j0][j1] = self.permute_op((self.diag_op(s[:chi_new]) @ v[:chi_new, :]).reshape(rrshape[1]), permute_dims[1])
        norm = self.norm_op(self.mpos[j0][j1])
        self.mpos[j0][j1] /= norm
        logging.info(f'{loc_i} {list(self.mpos[i0][i1].shape)}')
        return norm.item() if self.op == torch else norm
    
    def ladder_contraction(self, row_number, width_range):
        assert len(row_number) == 2
        r1, r2 = row_number
        assert r2 - r1 == 1
        for j in width_range:
            if j == width_range[0]:
                # print((r1, r2, j), self.mpos[r1][j].shape, self.mpos[r2][j].shape)
                assert self.mpos[r1][j].shape[0] == 1 and self.mpos[r1][j].shape[1] == 1
                assert self.mpos[r2][j].shape[1] == 1 and self.mpos[r2][j].shape[2] == 1
                self.mpos[r1][j] = self.mpos[r1][j].reshape(self.mpos[r1][j].shape[2], self.mpos[r1][j].shape[3])
                self.mpos[r2][j] = self.mpos[r2][j].reshape(self.mpos[r2][j].shape[0], self.mpos[r2][j].shape[3])
            elif j == width_range[-1]:
                assert self.mpos[r1][j].shape[0] == 1 and self.mpos[r1][j].shape[3] == 1
                assert self.mpos[r2][j].shape[2] == 1 and self.mpos[r2][j].shape[3] == 1
                self.mpos[r1][j] = self.mpos[r1][j].reshape(self.mpos[r1][j].shape[1], self.mpos[r1][j].shape[2])
                self.mpos[r2][j] = self.mpos[r2][j].reshape(self.mpos[r2][j].shape[0], self.mpos[r2][j].shape[1])
            else:
                assert self.mpos[r1][j].shape[0] == 1
                assert self.mpos[r2][j].shape[2] == 1
                self.mpos[r1][j] = self.mpos[r1][j].reshape((self.mpos[r1][j].shape[1], self.mpos[r1][j].shape[2], self.mpos[r1][j].shape[3]))
                self.mpos[r2][j] = self.mpos[r2][j].reshape((self.mpos[r2][j].shape[0], self.mpos[r2][j].shape[1], self.mpos[r2][j].shape[3]))
        ladder_result = self.einsum_op(
            'ij,ik->jk', self.mpos[r1][width_range[0]], self.mpos[r2][width_range[0]]
        )
        for j in width_range[1:-1]:
            ladder_result = self.einsum_op(
                'ij,ikl,kjm->lm', ladder_result, self.mpos[r1][j], self.mpos[r2][j]
            )
        ladder_result = self.einsum_op(
            'ij,ik,kj->', ladder_result, self.mpos[r1][width_range[-1]], self.mpos[r2][width_range[-1]]
        )
        
        return ladder_result
    
    def normal_bmps(self, chi, cutoff):
        self.exponent = 0.0
        self.error = 0.0
        length_up = self.length // 2 + self.length % 2
        for i in range(length_up-1):
            logging.info('up contraction:')
            merge_dims = []
            for j in range(self.width):
                merge_dims.append(self.contract((i+1, j), (i, j)))
            if max(merge_dims) > chi:
                logging.info('up canonical:')
                for j in range(self.width-1):
                    self.exponent += math.log10(self.canonical((i+1, j), (i+1, j+1)))
                logging.info('up rounding:')
                for j in range(self.width-1, 0, -1):
                    self.exponent += math.log10(self.rounding((i+1, j), (i+1, j-1), chi, cutoff))
        length_down = self.length // 2
        for i in range(self.length-1, self.length-length_down, -1):
            logging.info('down contraction:')
            merge_dims = []
            for j in range(self.width):
                merge_dims.append(self.contract((i-1, j), (i, j)))
            if max(merge_dims) > chi:
                logging.info('down canonical:')
                for j in range(self.width-1):
                    self.exponent += math.log10(self.canonical((i-1, j), (i-1, j+1)))
                logging.info('down rounding:')
                for j in range(self.width-1, 0, -1):
                    self.exponent += math.log10(self.rounding((i-1, j), (i-1, j-1), chi, cutoff))
        ladder_result = self.ladder_contraction((length_up-1, length_up), list(range(self.width)))
        self.significand = ladder_result
        self.result = (self.significand, self.exponent)
        
        return self.result, self.error
    
    def threeway_bmps(self, chi, cutoff):
        self.exponent = 0.0
        self.error = 0.0
        length_up = self.length // 2
        for i in range(length_up-1):
            logging.info('contraction:')
            contraction_path = sum([
                [((i+1, j), (i, j)) for j in range(i, self.width)],
                [((self.length-2-i, j), (self.length-i-1, j)) for j in range(i, self.width)],
                [((j, i+1), (j, i)) for j in range(i+1, self.length-i-1)] 
            ], start=[])
            for source, end in contraction_path:
                self.contract(source, end)
            logging.info('canonical:')
            canonical_path = sum([
                [((i+1, j), (i+1, j-1)) for j in range(self.width-1, i+1, -1)],
                [((j, i+1), (j+1, i+1)) for j in range(i+1, self.length-i-2)],
                [((self.length-2-i, j), (self.length-i-2, j+1)) for j in range(i+1, self.width-1)]
            ], start=[])
            for source, end in canonical_path:
                self.exponent += math.log10(self.canonical(source, end))
            logging.info('rounding:')
            rounding_path = sum([
                [((self.length-2-i, j), (self.length-i-2, j-1)) for j in range(self.width-1, i+1, -1)],
                [((j, i+1), (j-1, i+1)) for j in range(self.length-i-2, i+1, -1)],
                [((i+1, j), (i+1, j+1)) for j in range(i+1, self.width-1)],
            ], start=[])
            for source, end in rounding_path:
                self.exponent += math.log10(self.rounding(source, end, chi, cutoff))
        
        if self.length % 2 == 1:
            logging.info('extra contraction:')
            for j in range(length_up-1, self.width):
                self.contract((length_up, j), (length_up-1, j))
            logging.info('extra canonical:')
            for j in range(length_up-1, self.width-1):
                self.exponent += math.log10(self.canonical((length_up, j), (length_up, j+1)))
            logging.info('extra rounding:')
            for j in range(self.width-1, length_up-1, -1):
                self.exponent += math.log10(self.rounding((length_up, j), (length_up, j-1), chi, cutoff))
            ladder_result = self.ladder_contraction((length_up, length_up+1), list(range(length_up-1, self.width)))
        else:
            ladder_result = self.ladder_contraction((length_up-1, length_up), list(range(length_up-1, self.width)))
        self.significand = ladder_result
        self.result = (self.significand, self.exponent)
        return self.result, self.error
    
    def fourway_bmps(self, chi, cutoff):
        self.exponent = 0.0
        self.error = 0.0
        length_up = self.length // 2
        for i in range(length_up-1):
            logging.info('contraction:')
            contraction_path = sum([
                [((i+1, j), (i, j)) for j in range(i, self.width-i)],
                [((self.length-2-i, j), (self.length-i-1, j)) for j in range(i, self.width-i)],
                [((j, i+1), (j, i)) for j in range(i+1, self.length-i-1)],
                [((j, self.length-2-i), (j, self.length-1-i)) for j in range(i+1, self.length-i-1)]
            ], start=[])
            for source, end in contraction_path:
                self.contract(source, end)
            logging.info('canonical:')
            canonical_path = sum([
                [((i+1, j), (i+1, j-1)) for j in range(self.width-i-2, i+1, -1)],
                [((j, i+1), (j+1, i+1)) for j in range(i+1, self.length-i-2)],
                [((self.length-2-i, j), (self.length-i-2, j+1)) for j in range(i+1, self.width-i-2)],
                [((j, self.length-2-i), (j-1, self.length-2-i)) for j in range(self.length-2-i, i+2, -1)]
            ], start=[])
            for source, end in canonical_path:
                self.exponent += math.log10(self.canonical(source, end))
            logging.info('rounding:')
            rounding_path = sum([
                [((j, self.length-2-i), (j+1, self.length-2-i)) for j in range(i+2, self.length-i-2)],
                [((self.length-2-i, j), (self.length-i-2, j-1)) for j in range(self.width-i-2, i+1, -1)],
                [((j, i+1), (j-1, i+1)) for j in range(self.length-i-2, i+1, -1)],
                [((i+1, j), (i+1, j+1)) for j in range(i+1, self.width-i-2)],
            ], start=[])
            for source, end in rounding_path:
                self.exponent += math.log10(self.rounding(source, end, chi, cutoff))
        
        if self.length % 2 == 1:
            logging.info('extra contraction:')
            for j in range(length_up-1, self.width-(length_up-1)):
                self.contract((length_up, j), (length_up-1, j))
            logging.info('extra canonical:')
            for j in range(length_up-1, self.width-length_up):
                self.exponent += math.log10(self.canonical((length_up, j), (length_up, j+1)))
            logging.info('extra rounding:')
            for j in range(self.width-length_up, length_up-1, -1):
                self.exponent += math.log10(self.rounding((length_up, j), (length_up, j-1), chi, cutoff))
            ladder_result = self.ladder_contraction((length_up, length_up+1), list(range(length_up-1, self.width-(length_up-1))))
        else:
            ladder_result = self.ladder_contraction((length_up-1, length_up), list(range(length_up-1, self.width-(length_up-1))))
        self.significand = ladder_result
        self.result = (self.significand, self.exponent)
        return self.result, self.error