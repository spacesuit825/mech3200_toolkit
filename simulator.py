import numpy as np
import sympy as sp
import components as comp
from sympy.physics.mechanics import LagrangesMethod

## A generalised solver for complex MDOF systems lagrangians ##

# Currently supports point masses and rigid connectors #

#TODO: 1. Add support for trolleys and springs
#TODO: 2. Add support for non-point masses (maybe rectangles)
#TODO: 3. Add support for rotational objects
#TODO: 4. Add GUI

class Sim:
    def __init__(self):
        self.objects = []
        self.t = sp.symbols('t')
        self.g = sp.symbols('g')

    def add_objects(self, objects):
        self.objects.extend(objects)

    def setup_scene(self):
        i = 0
        for object in self.objects:
            i = object.setup(i, self.t)
        return i

    def calculate_potential(self):
        U = 0 
        for i,object in enumerate(self.objects):
            Ui = object.potential(self.g) 
            print("Object {} has potential {}".format(i, Ui))
            U += Ui
        return sp.simplify(U)

    def calculate_kinetic(self):
        T = 0
        for i,object in enumerate(self.objects):
            Ti = object.kinetic()
            print("Object {} has kinetic Energy {}".format(i, Ti))
            T += Ti
        return sp.simplify(T)

    def calculate_lagrange(self):
        T = self.calculate_kinetic()
        U = self.calculate_potential()
        L = sp.simplify(T - U)
        return L

    def get_symbols(self):
        s = []
        for object in self.objects:
            if object.get_symbols():
                s.append(object.get_symbols())
        return s

# Instanciate the simulator
sim = Sim()

# Generate the objects (ex. double pendulum)
point = comp.FixedPoint(location = [0, 0])
connector1 = comp.Connector(parent = point, length = sp.symbols('L_1'))
mass1 = comp.PointMass(parent = connector1, mass = sp.symbols('m_1'))
connector2 = comp.Connector(parent = mass1, length = sp.symbols('L_2'))
mass2 = comp.PointMass(parent = connector2, mass = sp.symbols('m_2'))

# Add the objects to the scene
sim.add_objects([point, connector1, mass1, connector2, mass2])

# Perform setup
sim.setup_scene()

# Compute the Lagrangian
L = sim.calculate_lagrange()

s = sim.get_symbols()
#print(s)
L = L.replace(sp.sin, lambda *args: args[0]).replace(sp.cos, lambda *args: 1)

## Can generate the euler-lagrange equations but turning them into matrix form is difficult with sympy ##
# TODO: 1. Look into automatically solving natural frequencies, maybe numerically
LM = LagrangesMethod(L, [i[0] for i in s])

euler_lag = LM.form_lagranges_equations()

# Sympy's LagrangesMethod only produces a mass matrix, no stiffness matrix
mass_matrix = LM.mass_matrix
print('\n')
sp.pprint(mass_matrix)
sp.pprint(LM.forcing_full)

#print(L)


