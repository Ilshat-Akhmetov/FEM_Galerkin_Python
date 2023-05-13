from typing import Union, List
from SourceCode.Points import Point2D, Point1D
from SourceCode.FiniteElements import Finite_el_2D_rectangle, Finite_el_1D_2point_chord
import numpy as np
import abc


class AbstractDomain:
    @abc.abstractmethod
    def set_bound_nodes(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def numerate_nodes(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def create_finite_elms(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_domain(self) -> Union[List[np.array], np.array]:
        raise NotImplementedError


class Domain2DRectangle(AbstractDomain):
    def __init__(
            self,
            n_points_x: int,
            n_points_y: int,
            x_left: Union[int, float],
            x_right: Union[int, float],
            y_left: Union[int, float],
            y_right: Union[int, float],
    ):
        self.n_points_x = n_points_x
        self.n_points_y = n_points_y
        self.n_points = n_points_x * n_points_y
        self.x_right = x_right
        self.x_left = x_left
        self.y_right = y_right
        self.y_left = y_left
        self.points: List[Point2D] = []
        self.finite_elms = []
        self.bound_inds = []
        self.numerate_nodes()
        self.create_finite_elms()
        self.set_bound_nodes()
        print("count points: {} ".format(len(self.points)))
        print("count finite elements: {} ".format(len(self.finite_elms)))

    def set_bound_nodes(self) -> None:
        for node in self.points:
            if (
                    node.x == self.x_right
                    or node.x == self.x_left
                    or node.y == self.y_left
                    or node.y == self.y_right
            ):
                self.bound_inds.append(node.ind)

    def numerate_nodes(self) -> None:
        x_domain = np.linspace(self.x_left, self.x_right, self.n_points_x)
        y_domain = np.linspace(self.y_left, self.y_right, self.n_points_y)
        xm, ym = np.meshgrid(x_domain, y_domain)
        for i, (x, y) in enumerate(zip(xm.ravel(), ym.ravel())):
            self.points.append(Point2D(i, x, y))

    def create_finite_elms(self) -> None:
        for i in range(self.n_points - 1):
            if self.points[i].x != self.x_right and self.points[i].y != self.y_right:
                self.finite_elms.append(
                    Finite_el_2D_rectangle(
                        self.points[i],
                        self.points[i + 1],
                        self.points[i + self.n_points_x],
                        self.points[i + self.n_points_x + 1],
                    )
                )

    def get_domain(self) -> List[np.array]:
        arr = np.array([point.get_val() for point in self.points])
        x, y= arr[:, 0], arr[:, 1]
        x = x.reshape(self.n_points_x, self.n_points_y)
        y = y.reshape(self.n_points_x, self.n_points_y)
        return x, y



class Domain1D(AbstractDomain):
    def __init__(
            self,
            n_points: int,
            x_left: Union[int, float],
            x_right: Union[int, float],
    ):
        self.n_points = n_points
        self.x_left = x_left
        self.x_right = x_right
        self.points = []
        self.finite_elms = []
        self.x_domain = np.linspace(x_left, x_right, n_points)
        self.numerate_nodes()
        self.create_finite_elms()
        self.set_bound_nodes()
        print("count points: {} ".format(len(self.points)))
        print("count finite elements: {} ".format(len(self.finite_elms)))

    def numerate_nodes(self) -> None:
        for i, node_val in enumerate(self.x_domain):
            self.points.append(Point1D(ind=i, x=node_val))

    def set_bound_nodes(self) -> None:
        self.bound_inds = [0, self.n_points - 1]

    def create_finite_elms(self) -> None:
        for i in range(self.n_points - 1):
            self.finite_elms.append(
                Finite_el_1D_2point_chord(self.points[i], self.points[i + 1])
            )

    def get_domain(self) -> np.array:
        return np.array([point.get_val() for point in self.points]).ravel()
