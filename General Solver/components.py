import numpy as np
from sympy.physics.vector import dynamicsymbols
import sympy as sp

class Component:
    def __init__(self):
        pass

    def setup(self, i, t):
        self.t = t
        return i

    def potential(self, g):
        return sp.Integer(0)

    def kinetic(self):
        return sp.Integer(0)
    
    def get_symbols(self):
        return []

class FixedPoint(Component):
    # Unmoving object, acts as base point for other objects
    def __init__(self, location: list):
        self.location = np.array(location)

    def get_position(self):
        return self.location

class PointMass(Component):
    # A child of a connector or spring
    def __init__(self, parent, mass: float):
        self.parent = parent
        self.mass = mass

    def setup(self, i, t):
        self.t = t
        return i

    def get_position(self):
        return self.parent.get_position()

    def potential(self, g):
        return self.mass * g * self.get_position()[1]

    def kinetic(self):
        position = self.get_position()
        print(position)
        vel = [sp.diff(position[0], self.t), sp.diff(position[1], self.t)]
        print(vel)
        return 0.5 * self.mass * (vel[0]**2 + vel[1]**2)

class Connector(Component):
    # Allows dynamic movement of objects
    def __init__(self, parent, length):
        self.parent = parent
        self.length = length

    def setup(self, i, t):
        self.t = t
        self.q = dynamicsymbols(f'q{i}')
        self.dq = dynamicsymbols(f'q{i}', 1)
        return i + 1

    def get_position(self):
        return [self.parent.get_position()[0] + self.length * sp.sin(self.q), self.parent.get_position()[1] - self.length * sp.cos(self.q)]

    def get_symbols(self):
        return [self.q, self.dq]

    