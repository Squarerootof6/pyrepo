import numpy as np
from Riemann_solver import HLLC_Riemann_Solver,W2U,U2W,W2F
from Voronoi_grid import Create_Voronoi
import astropy.units as u
import astropy.constants as c
import h5py
from scipy.spatial.transform import Rotation
import datetime
GAMMA=5/3
units_m = 1.989e+33*u.g
units_l = 3.08568e+18*u.cm
units_v = 977792.222*u.cm/u.s
units_t = units_l/units_v
units_e = (units_v**2).cgs
G = (c.G).to(units_l**3/units_m/units_t**2).value
def JACOBIANA(W):
    return np.array([[W[1],W[0],0],[0,W[1],1/W[0]],[0,GAMMA*W[2],W[1]]]) 
def Inverse_JACOBIANA(W):
    rho,v,P=W
    a=np.sqrt(GAMMA*P/rho)
    return 1/(v**2-a**2)*np.array([[(v**2-a**2)/v,-rho,1/v],[0,v,-1/rho],[0,-rho*a**2,v]]) 
def rotatx_align(n,rot):
    '''
    this equation aims at rotating the frame to align the x-axis to the direction of w.
    wi: velocity of cell i
    '''
    # 定义给定的向量
    target_vector = n

    # 计算旋转矩阵
    angle = np.arccos(np.dot(target_vector, [1, 0, 0]) / (np.linalg.norm(target_vector) * np.linalg.norm([1, 0, 0])))
    axis = np.cross([1, 0, 0], target_vector)/np.linalg.norm(np.cross([1, 0, 0], target_vector))
    rotation = Rotation.from_rotvec(angle * axis)
    rotated_vector = rotation.apply(rot)
    return rotated_vector,rotation
def Get_State(face,WL,WR,w):
    '''
    以cell_i中的某个面的法向为x轴
    '''
    new_v,rotator = rotatx_align(face.Areas/np.linalg.norm(face.Areas),WL[1:4])
    WpL=WL-np.array([0,*w,0])
    WpR=WR-np.array([0,*w,0])
    # 没有predict them forward in time by half a time-step
    #A = np.array([[W[1],W[0],0],[0,W[1],1/W[0]],[0,GAMMA*W[2],W[1]]])
    #Wpp = Wp+gradp(Wp)*(f-s)-A*gradp(Wp)*deltat/2
    WpppL = np.array([WpL[0],*new_v,WpL[4]])
    WpppR = np.array([WpR[0],*new_v,WpR[4]])
    return WpppL,WpppR,rotator
def determine_MeshSpeed(celli,face,V):
    wi=V
    wL = wi[celli.index];rL = celli.coordinate
    wR = wi[face.adj_index];rR = celli.voronoi_points[face.adj_index]
    f = face.centroid
    wprime = (wL-wR)*(f-(rR+rL)/2)/np.linalg.norm(rR-rL)**2*(rR-rL)
    w = (wR+wL)/2+wprime
    return w
def Gravitational_force(celli,Mass):
    ri = celli.voronoi_points - celli.coordinate
    dx =celli.Volume**(1/3)
    rik = np.linalg.norm(ri,axis=1)
    mask = np.where(np.logical_and(rik>0,rik<100*dx))
    ri = ri[mask]
    rik = rik[mask]
    scalar = G*Mass[mask]*rik**(-3/2)
    Fg = np.zeros(3)
    Fg[0] = np.sum(scalar*ri[:,0])
    Fg[1] = np.sum(scalar*ri[:,1])
    Fg[2] = np.sum(scalar*ri[:,2])
    return Fg
def Gravitational_Potential(celli,Mass):
    ri = celli.voronoi_points - celli.coordinate
    dx =celli.Volume**(1/3)
    rik = np.linalg.norm(ri,axis=1)
    mask = np.where(np.logical_and(rik>0,rik<100*dx))
    ri = ri[mask]
    rik = rik[mask]
    scalar = np.sum(G*Mass[mask]*rik**(-1/2))
    return scalar
def determine_dt(Ri,ci,vi=0,CFL=0.3):
    dt = np.min(CFL*Ri/(ci+vi))
    if np.isnan(dt):
        print('dt is nan',Ri,ci,vi)
        dt=0.01
    return dt
def refinement(celli):
    for face in celli.Faces:
        
    New = cells.voronoi.points[np.where(cells.W_Grid[:,-1]>0)]
    print('{} particles are removed...{} remains...'.format(len(cells.voronoi.points)-len(New),len(New)))
    return New
def run():
    IC = h5py.File('./IC.hdf5','r')
    points = IC['Gas/Coordinates'][:].T
    BoxSize = IC['Header'].attrs['BoxSize']
    cells = Create_Voronoi(BoxSize=BoxSize,points=points.T)
    cells.W_Grid[0] = IC['Gas/density'][:]
    cells.W_Grid[1] = IC['Gas/Velocities'][:,0]
    cells.W_Grid[2] = IC['Gas/Velocities'][:,1]
    cells.W_Grid[3] = IC['Gas/Velocities'][:,2]
    cells.W_Grid[4] = IC['Gas/Pressure'][:]
    
    cells.Mass = IC['Gas/Masses'][:]
    IC.close()
    
    dt=0.01
    Old_Wgrid = cells.W_Grid.copy() # [n*5]
    Old_Ugrid = W2U(Old_Wgrid)
    
    t=0;CFL=0.3;T=10
    SNAPSHOTNUM=20
    SNAPCOUNT=0
    REFINE = False
    
    while t<T:
        New_Wgrid = Old_Wgrid.copy()
        New_Ugrid = W2U(Old_Wgrid)
        
        #dt = CFL*BoxSize**3/len(Old_Ugrid)/np.max(np.linalg.norm((Old_Wgrid[:,1:4])))
        dt = determine_dt((3*cells.Mass/Old_Wgrid[0]/4/np.pi)**(1/3),np.sqrt(GAMMA*Old_Wgrid[4]/Old_Wgrid[0]),np.linalg.norm(Old_Wgrid[1:4].T,axis=1),CFL=CFL)
        print('t={},dt = {}\t'.format(t,dt)+str(datetime.datetime.now()))
        Cell_Vel = np.zeros_like(points)
        FLUX_GRID_RECORD = set()
        for i in range(points.shape[-1]):
            FLUX = np.array([0,0,0,0,0],dtype=np.float64)
            celli = cells[i]
            m = Old_Ugrid[0,i]*celli.Volume
            if celli.Volume<0.05*BoxSize/(points.shape[-1]):
                celli.alive=False
                print('skip one death')
                continue
            elif m>1.5*np.mean(cells.Mass):
                REFINE = True
                try:
                    new_points = np.hstack((new_points,[*(celli.coordinate+1e-4*celli.coordinate).reshape(3,-1)]))
                    New_W = np.hstack((New_W,Old_Wgrid[:,i]))
                    newmass = np.hstack((newmass,Old_Wgrid[0,i]*celli.Volume))
                except:
                    new_points = np.array([*(celli.coordinate+1e-4*celli.coordinate).reshape(3,-1)])
                    New_W = Old_Wgrid[:,i].reshape(5,-1)
                    newmass = Old_Wgrid[0,i]*celli.Volume
                
            Fg = Gravitational_force(celli,cells.Mass)
            PHI = Gravitational_Potential(celli,cells.Mass)
            Cell_Vel[:,i] = Old_Wgrid[1:4,i]
            DeltaE=0
            for face in celli.Faces:
                A = np.linalg.norm(face.Areas)
                dir = face.centroid+face.Areas
                dir1 = face.centroid - celli.coordinate
                pm = np.dot(dir,dir1)/np.abs(np.dot(dir,dir1))
                if face.Face_index in FLUX_GRID_RECORD:
                    F = cells.Flux_Grid[:,face.Face_index]
                    FLUX+= pm*A*F
                    continue
                else:
                    Mesh_Vel = determine_MeshSpeed(celli,face,Old_Wgrid[1:4].T)
                    WL,WR,rotator = Get_State(face,Old_Wgrid[:,i],Old_Wgrid[:,face.adj_index],Mesh_Vel)

                    F,WS = HLLC_Riemann_Solver(WL,WR,PHI)
                    #不太懂论文里这部分怎么算的
                    #Wlab = np.array([WS[0],*rotator.apply(WS[1:4],inverse=True),WS[-1]])+np.array([0,*Mesh_Vel,0])
                    #print(Wlab)
                    #F_bar = W2F(Wlab)-np.dot(W2U(Wlab),np.array([0,*Mesh_Vel,0]))
                    F_bar = np.array([F[0],*rotator.apply(F[1:4],inverse=True),F[-1]])+np.array([0,*Mesh_Vel,0])
                    FLUX+=pm*A*F_bar
                    cells.Flux_Grid[:,face.Face_index] = F
                    FLUX_GRID_RECORD.update([face.Face_index])
                #Fadj = Gravitational_force(cells[face.adj_index],Mass)
                #DeltaE += (Mass[i]-Mass[face.adj_index])*np.dot(celli.coordinate-face.adj_points,Fg-Fadj)
                DeltaE += (cells.Mass[i]-cells.Mass[face.adj_index])*(PHI - Gravitational_Potential(cells[face.adj_index],cells.Mass))
            New_Ugrid[:,i]-=dt*FLUX
            m_new = New_Ugrid[0,i]*celli.Volume
            m = Old_Ugrid[0][i]*celli.Volume
            
            New_Ugrid[1:4,i] -= dt/2*(m*Fg+m_new*Fg)
            #v_new = U2W(New_Ugrid[i])[1:4]
            #New_Ugrid[i][-1] -= dt/2*(m*np.dot(Old_Ugrid[i][1:4],Fg)+m_new*np.dot(v_new,Fg))
            New_Ugrid[-1,i] += -dt*m*np.dot(Cell_Vel[:,i],Fg)-DeltaE/2 #Springel 2013
            # n+1 的phi还没算，先用老的代替
        
        #更新网格生成点的坐标
        points += np.nan_to_num(Cell_Vel*dt)
        if REFINE:
            points = np.hstack((points,new_points))
            Old_Wgrid = np.hstack((U2W(New_Ugrid),New_W))
            Old_Ugrid = W2U(Old_Wgrid)
            cells.Mass = np.hstack((cells.Mass,newmass))
            print('add {} particles..'.format(new_points.shape[-1]))
        else:
            Old_Ugrid = New_Ugrid.copy()
            New_Wgrid = U2W(New_Ugrid)
            Old_Wgrid = New_Wgrid.copy()
        
        if t/T>SNAPCOUNT/SNAPSHOTNUM:
        #if True:
            cells.save('./output/',SNAPCOUNT)
            SNAPCOUNT+=1
        cells = Create_Voronoi(BoxSize=BoxSize,points=points.T)
        cells.W_Grid = New_Wgrid
        t+=dt
    
if __name__ == '__main__':
    import cProfile
    #cProfile.run('run()')
    run()