from SourceCode.FEM_elliptic_eq import FEM_elliptic_eq2D, FEM_elliptic_eq1D
from SourceCode.Domains import Domain2DRectangle, Domain1D
from SourceCode.utilities import get_func
from math import pi, e
import numpy as np

def analyt_func(x,y):
    n: int = 10
    total_s: float = 0
    for i in range(1,n,2):
        for j in range(1,n,2):
            total_s += (-1)**((i+j)//2-1)/(i*j*(i*i+j*j))*np.cos(i*pi/2*x)*np.cos(j*pi/2*y)
    total_s = total_s * (8/(pi*pi))**2
    return total_s

left_part = lambda x, y, f1, f2: (get_func(f1,'x1',1)(x,y)*get_func(f2,'x1',1)(x,y)
                               +  get_func(f1,'x2',1)(x,y)*get_func(f2,'x2',1)(x,y))
right_part = lambda x, y: 1
dirichle_cond = lambda x, y:  0
xl = -1
xr = 1
yl = -1
yr = 1
n_points = 10
domain = Domain2DRectangle(n_points, n_points, xl, xr, yl, yr)
fem_obj = FEM_elliptic_eq2D(domain,
                            left_part,
                            right_part,
                            dirichle_cond)
appr_sol = fem_obj.get_solution()
x, y = domain.get_domain()
analyt_sol = analyt_func(x, y)
error = np.max(np.abs(appr_sol - analyt_sol))
print(error)


left_part = lambda x, f1, f2: get_func(f1,'x1',1)(x)*get_func(f2,'x1',1)(x)+get_func(f1)(x)*get_func(f2)(x)
right_part = lambda x: 0
dirichle_cond = lambda x: x
xl = 0
xr = 1
n_points = 10
domain = Domain1D(n_points, xl, xr)
fem_obj = FEM_elliptic_eq1D(domain,
                            left_part,
                            right_part,
                            dirichle_cond)
appr_sol = fem_obj.get_solution()
analyt_func = lambda x: e/(e*e-1)*np.exp(x) + e/(1-e*e)*np.exp(-x)
x = domain.get_domain()
analyt_sol = analyt_func(x)
error = np.max(np.abs(appr_sol - analyt_sol))
print(error)