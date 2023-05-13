from typing import Union

class Point2D:
    def __init__(self, ind: int, x: Union[float, int], y: Union[float, int]):
        self.ind = ind
        self.x = x
        self.y = y

    def get_val(self) -> tuple:
        return (self.x, self.y)

class Point1D:
    def __init__(self, ind: int, x: Union[float, int]):
        self.ind = ind
        self.x = x

    def get_val(self) -> tuple:
        return tuple([self.x])