# FEM_Galerkin_Python
This project provides an example how the Finite element method (here I use the one which based on Galerkin) and pure Galerkin method can be implemented on Python

Currently only solver of the elliptic equations in 1d and 2d cases was implemented. In future a set of programs will be expanded 
for parabolic eqs, hyperbolic eqs and 3d case. Also it is planned to add extra basic functions, current version 
has only linear ones.

Save for FEM soon the pure Galerking method implementation will also be added.

In Fem_ellipic_eq.Presentations.ipynb you may see how to use this program. 
Example 1. Given an equation $$ y' -y = 0$$
with boundary conditions: $$y(0)=0, y(1)=1, x \in [0,1]$$
true solution:
$$
true_sol(x)=\frac{e}{e^2-1}e^x+\frac{e}{1-e^2}e^x
$$
```
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
```