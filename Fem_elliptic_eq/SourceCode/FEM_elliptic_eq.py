import numpy as np
from scipy import sparse
from typing import Callable
from SourceCode.Domains import AbstractDomain
import abc


class AbstractEllipticFem:
    @abc.abstractmethod
    def get_final_left_part(self, f1: Callable, f2: Callable) -> Callable:
        raise NotImplementedError

    @abc.abstractmethod
    def get_final_right_part(
            self, basis_func: Callable, fin_el_func: Callable
    ) -> Callable:
        raise NotImplementedError

    @abc.abstractmethod
    def get_solution(self) -> np.array:
        raise NotImplementedError

    def apply_bound_conds(self, a_matr: sparse.dok_matrix, b_vec: np.array) -> tuple:
        for bound_ind in self.domain.bound_inds:
            _, non_zero_cols = a_matr[bound_ind, :].nonzero()
            for j in non_zero_cols:
                a_matr[bound_ind, j] = 0.0
            a_matr[bound_ind, bound_ind] = 1.0
            b_vec[bound_ind] = self.dirichlet_init_cond_func(
                *self.domain.points[bound_ind].get_val()
            )
        return a_matr, b_vec

    def calculate_solution(self) -> np.array:
        a_matr, b_vec = self.assemble()
        a_matr, b_vec = self.apply_bound_conds(a_matr, b_vec)
        a_matr = a_matr.tocsr()
        solution = sparse.linalg.spsolve(a_matr, b_vec)
        return solution

    def assemble(self) -> tuple:
        dtype = np.float64
        total_n_nodes = self.domain.n_points
        a_matr = sparse.dok_matrix((total_n_nodes, total_n_nodes), dtype=dtype)
        b_vec = np.zeros(total_n_nodes, dtype=dtype)
        nodes_in_fin_el = len(self.domain.finite_elms[0])
        for fin_el in self.domain.finite_elms:
            for i in range(nodes_in_fin_el):
                loc_func_i = fin_el.local_funcs[i]
                final_right_part = self.get_final_right_part(loc_func_i)
                int_val = fin_el.calculate_integral(final_right_part)
                b_vec[fin_el.points[i].ind] += int_val
                for j in range(nodes_in_fin_el):
                    loc_func_j = fin_el.local_funcs[j]
                    final_left_part = self.get_final_left_part(loc_func_i, loc_func_j)
                    int_val = fin_el.calculate_integral(final_left_part)
                    a_matr[fin_el.points[i].ind, fin_el.points[j].ind] += int_val
        return a_matr, b_vec


class FEM_elliptic_eq2D(AbstractEllipticFem):
    def __init__(
            self,
            domain: AbstractDomain,
            left_part_eq: Callable,
            right_part_eq: Callable,
            dirichlet_init_cond_func: Callable,
    ):
        super().__init__()
        self.domain = domain
        self.left_part_eq = left_part_eq
        self.right_part_eq = right_part_eq
        self.dirichlet_init_cond_func = dirichlet_init_cond_func

    def get_final_left_part(self, f1: Callable, f2: Callable) -> Callable:
        return lambda x, y: self.left_part_eq(x, y, f1, f2)

    def get_final_right_part(self, basis_func: Callable) -> Callable:
        return lambda x, y: self.right_part_eq(x, y) * basis_func["func_val"](x, y)

    def get_solution(self) -> np.array:
        solution = self.calculate_solution()
        solution = solution.reshape(self.domain.n_points_x, self.domain.n_points_y)
        return solution


class FEM_elliptic_eq1D(AbstractEllipticFem):
    def __init__(
            self,
            domain: AbstractDomain,
            left_part_eq: Callable,
            right_part_eq: Callable,
            dirichlet_init_cond_func: Callable,
    ):
        self.domain = domain
        self.left_part_eq = left_part_eq
        self.right_part_eq = right_part_eq
        self.dirichlet_init_cond_func = dirichlet_init_cond_func

    def get_final_left_part(self, f1: Callable, f2: Callable) -> Callable:
        return lambda x: self.left_part_eq(x, f1, f2)

    def get_final_right_part(self, basis_func: Callable) -> Callable:
        return lambda x: self.right_part_eq(x) * basis_func["func_val"](x)

    def get_solution(self) -> np.array:
        solution = self.calculate_solution()
        return solution
