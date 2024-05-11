import numpy as np
from Riemann_solver import HLLC_Riemann_Solver
from Voronoi_grid import Create_Voronoi
import astropy.units as u
import astropy.constants as c
import h5py

CUSTOM_SEED = 0
if CUSTOM_SEED != 0: 
    np.random.seed(CUSTOM_SEED)
    
units_m = 1.989e+33*u.g
units_l = 3.08568e+18*u.cm
units_v = 977792.222*u.cm/u.s
units_t = units_l/units_v
units_e = (units_v**2).cgs

GAMMA=5/3
Boxsize=10
density_0=(1.25e4*u.M_sun/(Boxsize*u.pc)**3).to(units_m/units_l**3)
velocity_radial_0= 0

X_H=1
mu = 4/(1+3*X_H+4*X_H)
temperature = 10*u.K
utherm_0=(temperature*c.k_B*3/2/mu/c.m_p).to(units_e)
X_H=1
mu = 4/(1+3*X_H+4*X_H)
#utherm_0 = pressure_0 / ( GAMMA - 1.0 ) / density_0
pressure_0 = utherm_0*( GAMMA - 1.0 ) * density_0

CellsPerDimension = 15
dx=Boxsize/CellsPerDimension
NumberOfCells = CellsPerDimension * CellsPerDimension * CellsPerDimension
pos_first, pos_last = 0.5 * dx, Boxsize - 0.5 * dx
Pos = np.random.uniform(pos_first, pos_last,size=(NumberOfCells,3))
## calculate distance from center
xPosFromCenter = (Pos[:,0] - 0.5 * Boxsize)
yPosFromCenter = (Pos[:,1] - 0.5 * Boxsize)
zPosFromCenter = (Pos[:,2] - 0.5 * Boxsize)
Radius = np.sqrt( xPosFromCenter**2 + yPosFromCenter**2 + zPosFromCenter**2 )

""" set up hydrodynamical quantitites """
## mass insetad of density
# Turbulence
density_tur = np.random.normal(density_0.value,0.1,NumberOfCells)
Mass = np.full(NumberOfCells, density_0.value*dx*dx*dx)+density_tur*dx**3
Density = np.full(NumberOfCells, density_0.value)+density_tur
Pressure = np.full(NumberOfCells, pressure_0.value)
## velocity
Velocity = np.zeros([NumberOfCells,3])
Velocity[:,0] = velocity_radial_0 * xPosFromCenter / Radius
Velocity[:,1] = velocity_radial_0 * yPosFromCenter / Radius
Velocity[:,2] = velocity_radial_0 * zPosFromCenter / Radius
## specific internal energy
Uthermal = np.full(NumberOfCells, utherm_0)

FilePath = './IC.hdf5'
IC = h5py.File(FilePath, 'w')
header = IC.create_group("Header")
header.attrs.create("Time", 0.0)
header.attrs.create("BoxSize", Boxsize)
header.attrs.create("Flag_Feedback", 1)

part0 = IC.create_group("Gas")
part0.create_dataset("ParticleIDs", data=np.arange(1, NumberOfCells+1) )
part0.create_dataset("Coordinates", data=Pos)
part0.create_dataset("Masses", data=Mass)
part0.create_dataset("density", data=Density)
part0.create_dataset("Velocities", data=Velocity)
part0.create_dataset("Pressure", data=Pressure)
IC.close()
print(pressure_0.value,density_0.value,)