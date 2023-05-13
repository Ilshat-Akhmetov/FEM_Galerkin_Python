from SourceCode.Points import Point2D, Point1D
from scipy import integrate
from typing import Callable

lin_el_funcs = [
    {
        "func": lambda x, dx, start: 1 - (x + start) / dx,
        "func_deriv": lambda x, dx: -1 / dx,
    },
    {"func": lambda x, dx, start: (x + start) / dx, "func_deriv": lambda x, dx: 1 / dx},
]


class Finite_el_2D_rectangle:
    def __init__(
        self,
        base_node: Point2D,
        right_node: Point2D,
        top_node: Point2D,
        top_right_node: Point2D,
    ):
        self.dx = right_node.x - base_node.x
        self.dy = top_node.y - base_node.y
        self.start_coords = (base_node.x, base_node.y)
        self.points = [base_node, right_node, top_node, top_right_node]
        f_base = {"func_val": lambda x, y: lin_el_funcs[0]["func"](
            x, self.dx, self.start_coords[0]
        )
                                           * lin_el_funcs[0]["func"](y, self.dy, self.start_coords[1]), "x1": [
            lambda x, y: lin_el_funcs[0]["func_deriv"](x, self.dx)
                         * lin_el_funcs[0]["func"](y, self.dy, self.start_coords[1])
        ], "x2": [
            lambda x, y: lin_el_funcs[0]["func"](x, self.dx, self.start_coords[0])
                         * lin_el_funcs[0]["func_deriv"](y, self.dy)
        ]}

        f_right = {"func_val": lambda x, y: lin_el_funcs[1]["func"](
            x, self.dx, self.start_coords[0]
        )
                                            * lin_el_funcs[0]["func"](y, self.dy, self.start_coords[1]), "x1": [
            lambda x, y: lin_el_funcs[1]["func_deriv"](x, self.dx)
                         * lin_el_funcs[0]["func"](y, self.dy, self.start_coords[1])
        ], "x2": [
            lambda x, y: lin_el_funcs[1]["func"](x, self.dx, self.start_coords[0])
                         * lin_el_funcs[0]["func_deriv"](y, self.dy)
        ]}

        f_top = {"func_val": lambda x, y: lin_el_funcs[0]["func"](
            x, self.dx, self.start_coords[0]
        )
                                          * lin_el_funcs[1]["func"](y, self.dy, self.start_coords[1]), "x1": [
            lambda x, y: lin_el_funcs[0]["func_deriv"](x, self.dx)
                         * lin_el_funcs[1]["func"](y, self.dy, self.start_coords[1])
        ], "x2": [
            lambda x, y: lin_el_funcs[0]["func"](x, self.dx, self.start_coords[0])
                         * lin_el_funcs[1]["func_deriv"](y, self.dy)
        ]}

        f_top_right = {"func_val": lambda x, y: lin_el_funcs[1]["func"](
            x, self.dx, self.start_coords[0]
        )
                                                * lin_el_funcs[1]["func"](y, self.dy, self.start_coords[1]), "x1": [
            lambda x, y: lin_el_funcs[1]["func_deriv"](x, self.dx)
                         * lin_el_funcs[1]["func"](y, self.dy, self.start_coords[1])
        ], "x2": [
            lambda x, y: lin_el_funcs[1]["func"](x, self.dx, self.start_coords[0])
                         * lin_el_funcs[1]["func_deriv"](y, self.dy)
        ]}

        self.local_funcs = [f_base, f_right, f_top, f_top_right]

    def __len__(self):
        return 4

    def calculate_integral(self, f: Callable):
        f_from_start_coord = lambda x, y: f(
            x - self.start_coords[0], x - self.start_coords[1]
        )
        return integrate.dblquad(f_from_start_coord, 0, self.dx, 0, self.dy)[0]


class Finite_el_1D_2point_chord:
    def __init__(self, left_node: Point1D, right_node: Point1D):
        self.dx = right_node.x - left_node.x
        self.start_coords = [left_node.x]
        self.points = [left_node, right_node]
        f_l = {"func_val": lambda x: lin_el_funcs[0]["func"](
            x, self.dx, self.start_coords[0]
        ), "x1": [lambda x: lin_el_funcs[0]["func_deriv"](x, self.dx)]}
        f_r = {"func_val": lambda x: lin_el_funcs[1]["func"](
            x, self.dx, self.start_coords[0]
        ), "x1": [lambda x: lin_el_funcs[1]["func_deriv"](x, self.dx)]}
        self.local_funcs = [f_l, f_r]

    def __len__(self):
        return 2

    def calculate_integral(self, f: Callable):
        f_from_start_coords = lambda x: f(x - self.start_coords[0])
        return integrate.quad(f_from_start_coords, 0, self.dx)[0]
