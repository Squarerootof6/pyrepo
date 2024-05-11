""" @package ./examples/Noh_3d/create.py
Code that creates 3d Noh test problem initial conditions

created by Rainer Weinberger, last modified 24.02.2019
"""

""" load libraries """
import sys    # system specific calls
import numpy as np    # scientific computing package
import h5py    # hdf5 format
import astropy.units as u
import astropy.constants as c
import pandas as pd
import re

simulation_directory = str(sys.argv[1])
print("./: creating ICs in directory " + simulation_directory)

""" initial condition parameters """
FilePath = simulation_directory + '/IC.hdf5'

FloatType = np.float64  # double precision: np.float64, for single use np.float32
IntType = np.int32

## units
units_m =  (1*u.M_sun).to(u.g)
units_l =  (1*u.pc).to(u.cm)
units_t = (0.01*u.Myr).to(u.s)
units_v =  (units_l/units_t).to(u.cm/u.s)


## initial state

density_0 = (24*u.M_sun/u.pc**3).to(units_m/units_l**3)
n_H = (density_0/c.m_p).to(u.cm**(-3))
dM = FloatType(data['ReferenceGasPartMass'][0])*units_m
dx = ((dM/density_0)**(1/3)).to(units_l).value
#print('GAS:density_0=',density_0)
velocity_radial_0 = 0    ## radial inflow velocity
#pressure_0 = 0
gamma = 5./3.  ## note: this has to be consistent with the parameter settings for Arepo!
#utherm_0 = pressure_0 / ( gamma - 1.0 ) / density_0
#utherm_0 = ((100*u.K*c.k_B*n_H/density_0)/(gamma-1.0)).to(units_e/units_m).value
X_H=1
mu = 4/(1+3*X_H+4*X_H)
temperature = 10*u.K
utherm_0=(temperature*c.k_B*3/2/mu/c.m_p).to(units_e).value
""" set up grid: cartesian 3d grid """
Boxsize = FloatType(data['BoxSize'][0])
CellsPerDimension = IntType(Boxsize//dx)
NumberOfCells = CellsPerDimension * CellsPerDimension * CellsPerDimension
## spacing
dx = Boxsize / FloatType(CellsPerDimension)
## position of first and last cell
pos_first, pos_last = 0.5 * dx, Boxsize - 0.5 * dx

import inspect

def print_local_variables():
    current_frame = inspect.currentframe()
    caller_frame = inspect.getouterframes(current_frame, 2)
    local_vars = caller_frame[1].frame.f_locals
    
    for var_name, var_value in local_vars.items():
        if not var_name.startswith("__") and not inspect.isclass(var_value) and not inspect.ismodule(var_value) and not inspect.isfunction(var_value):
            print(f"{var_name}: {var_value}")

# 调用函数以打印本文件中定义的变量
print_local_variables()


## set up grid
'''Grid1d = np.linspace(pos_first, pos_last, CellsPerDimension, dtype=FloatType)
xx, yy, zz = np.meshgrid(Grid1d, Grid1d, Grid1d)
Pos = np.zeros([NumberOfCells, 3], dtype=FloatType)
Pos[:,0] = xx.reshape(NumberOfCells)
Pos[:,1] = yy.reshape(NumberOfCells)
Pos[:,2] = zz.reshape(NumberOfCells)'''

Pos = np.random.uniform(pos_first, pos_last,size=(NumberOfCells,3))
## calculate distance from center
xPosFromCenter = (Pos[:,0] - 0.5 * Boxsize)
yPosFromCenter = (Pos[:,1] - 0.5 * Boxsize)
zPosFromCenter = (Pos[:,2] - 0.5 * Boxsize)
Radius = np.sqrt( xPosFromCenter**2 + yPosFromCenter**2 + zPosFromCenter**2 )

""" set up hydrodynamical quantitites """
## mass insetad of density
Mass = np.full(NumberOfCells, density_0.value*dx*dx*dx, dtype=FloatType)
## velocity
Velocity = np.zeros([NumberOfCells,3], dtype=FloatType)
Velocity[:,0] = velocity_radial_0 * xPosFromCenter / Radius
Velocity[:,1] = velocity_radial_0 * yPosFromCenter / Radius
Velocity[:,2] = velocity_radial_0 * zPosFromCenter / Radius
## specific internal energy
Uthermal = np.full(NumberOfCells, utherm_0, dtype=FloatType)

StarPos = np.array([[0.5 * Boxsize,0.5 * Boxsize,0.5 * Boxsize]])
StarVelocity = np.zeros([1,3],dtype=FloatType)

""" write *.hdf5 file; minimum number of fields required by Arepo """
IC = h5py.File(FilePath, 'w')

## create hdf5 groups
header = IC.create_group("Header")
part0 = IC.create_group("PartType0")
part1 = IC.create_group("PartType4")
## header entries
# six types of particle in total
NumPart = np.array([NumberOfCells, 0, 0, 0, 1, 0], dtype=IntType)
header.attrs.create("NumPart_ThisFile", NumPart)
header.attrs.create("NumPart_Total", NumPart)
header.attrs.create("NumPart_Total_HighWord", np.zeros(6, dtype=IntType) )
header.attrs.create("MassTable", np.zeros(6, dtype=IntType) )
header.attrs.create("Time", 0.0)
header.attrs.create("Redshift", 0.0)
header.attrs.create("BoxSize", Boxsize)
header.attrs.create("NumFilesPerSnapshot", 1)
header.attrs.create("Omega0", 0.0)
header.attrs.create("OmegaB", 0.0)
header.attrs.create("OmegaLambda", 0.0)
header.attrs.create("HubbleParam", 1.0)
header.attrs.create("Flag_Sfr", 0)
header.attrs.create("Flag_Cooling", 0)
header.attrs.create("Flag_StellarAge", 0)
header.attrs.create("Flag_Metals", 0)
header.attrs.create("Flag_Feedback", 1)
if Pos.dtype == np.float64:
    header.attrs.create("Flag_DoublePrecision", 1)
else:
    header.attrs.create("Flag_DoublePrecision", 0)

## copy datasets
part0.create_dataset("ParticleIDs", data=np.arange(1, NumberOfCells+1) )
part0.create_dataset("Coordinates", data=Pos)
part0.create_dataset("Masses", data=Mass)
part0.create_dataset("Velocities", data=Velocity)
#part0.create_dataset("InternalEnergy", data=Uthermal)

#Type 4 particle
Part4CellsPerDimension = 0
Part4NumberOfCells = Part4CellsPerDimension**3+1
Part4MassPerCell = 50
#Part4MassPerCell = M_star.value
Part4Pos = np.zeros([Part4NumberOfCells, 3], dtype = FloatType)
Random4Pos = np.random.rand(Part4NumberOfCells, 3)*Boxsize
Part4Pos[:] = Random4Pos[:]
Part4Mass = np.full(Part4NumberOfCells, Part4MassPerCell, dtype=FloatType)
Part4Velocity = np.zeros([Part4NumberOfCells,3], dtype=FloatType)
MassiveStarMass = np.full(Part4NumberOfCells, 50, dtype=FloatType)
GFM_InitialMass = np.full(Part4NumberOfCells, Part4MassPerCell, dtype=FloatType)
GFM_Metallicity = np.full((Part4NumberOfCells,),1e-2, dtype=FloatType)
GFM_Metals = np.array(np.array([0.7381,0.2485,2.4e-3,7e-4,5.8e-3,1.3e-3,7e-4,7e-4,1.3e-3,5e-4]), dtype=FloatType)
GFM_StellarFormationTime = np.full((Part4NumberOfCells,), 1e-10, dtype=FloatType)
Part4Pos[0,:] = np.array([Boxsize/2,Boxsize/2,Boxsize/2])

'''
part1.create_dataset("ParticleIDs", data=np.array([NumberOfCells+1],dtype=np.uint32) )
part1.create_dataset("Coordinates", data=StarPos)
part1.create_dataset("Velocities", data=Part4Velocity)
part1.create_dataset("Masses", data=Part4Mass)
part1.create_dataset("MassiveStarMass", data=MassiveStarMass)
part1.create_dataset("GFM_InitialMass", data=GFM_InitialMass)
part1.create_dataset("GFM_Metallicity", data=GFM_Metallicity)
part1.create_dataset("GFM_Metals", data=GFM_Metals)
part1.create_dataset("GFM_StellarFormationTime", data=GFM_StellarFormationTime)
part1.create_dataset("MainSequenceFlag", data=np.array([2],dtype=IntType))'''

import arepo_processing.rigel_table_fitting as MM

def decide_metal_Q(mass,metal):
    UNIT_RATE = 3.168808781402895e+46
    pr  = MM.get_UV0(mass,metal)/UNIT_RATE*1e63/u.Gyr
    pr += MM.get_UV1(mass,metal)/UNIT_RATE*1e63/u.Gyr
    pr += MM.get_UV2(mass,metal)/UNIT_RATE*1e63/u.Gyr
    return pr.to(1/u.s)
def StromgenSphere(Q,n_H,alphab):
    Q48 = Q/(1e48/u.s)
    alpha0=alphab/(2.59*1e-13 *u.cm**3/u.s)
    n3=n_H/(1e3/u.cm**3)
    rs = 0.315*Q48**(1/3)*n3**(-2/3)*alpha0**(-1/3)
    ms =3.25*Q48*n3**(-1)*alpha0**(-1)
    return rs*u.pc,ms*u.M_sun
Q = decide_metal_Q(Part4MassPerCell,1e-2)
T4 = 15000*u.K/1e4/u.K
alphab=2.72e-13*T4**(-0.789)*u.cm**3/u.s
Rs,Ms = StromgenSphere(Q,n_H,alphab)
ci=11.4*T4**(1/2)*u.km/u.s
t = (FloatType(data['TimeMax'][0])*units_t).to(u.Myr)
ri=Rs*(1+(7/4*np.sqrt(4/3)*(ci*t)/Rs).value)**(4/7)
    
print('Stromgen Sphere:',Rs)
print('RT Bubble:',ri)
Mass = (dM).to(u.M_sun)
Resolution = Ms/Mass
print('Resolution:',np.median(Resolution))

## close file
print('Recommanded CPUs:{0}-{1}'.format(NumberOfCells/1e4,NumberOfCells/1e5))
IC.close()

import os
CoreNumber = int(min(64,NumberOfCells/1e4))
os.system('CoreNumber={:d}'.format(CoreNumber))
os.system('echo ${CoreNumber}')

