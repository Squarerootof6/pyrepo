import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c 
import astropy.units as u
from scipy.spatial import Voronoi
import pickle
boxsize=1

def Create_Voronoi(BoxSize=1,points=np.random.rand(10, 3),kwargs = {'incremental':False}):
    #p = np.vstack((points,np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]])*BoxSize))
    p=points
    vor = Voronoi(p,**kwargs)
    Cell = VoronoiMesh(vor)
    return Cell
class Load_VoronoiMesh_Data:
    def __init__(self, filepath):
        with open(filepath, "rb") as f:
            loaded_variable = pickle.load(f)
        for key, value in loaded_variable.items():
            setattr(self, key, value)
class VoronoiMesh:
    def __init__(self,voro,load=False):
        self.voronoi = voro
        if load:
            self.Flux_Grid = self.voronoi.Flux_Grid
            self.W_Grid = self.voronoi.W_Grid
            self.Cells = self.voronoi.Cells
            self.Mass = self.voronoi.Mass
            self.Active = self.voronoi.Active
        else:
            self.Flux_Grid = np.zeros((5,len(self.voronoi.ridge_vertices)))
            self.W_Grid = np.zeros((5,len(self.voronoi.points)))
            self.Cells = {}
            self.Mass = np.zeros(len(self.voronoi.points))
            self.Active = np.ones(len(self.voronoi.points))
    def __repr__(self):
        r = "["
        for p in self.voronoi.points: 
            r += str(p)
        return r[:-3] + "]"
    def __getitem__(self, index):
        if index in self.Cells.keys():
            return self.Cells[index]
        else:
            cell = Cell(index,self.voronoi.points[index])
            cell.Faces = self.find_face(index)
            cell.voronoi_points = self.voronoi.points
            cell.W = self.W_Grid[:,index]
            #Voronoi多面体是凸多面体，所有多边形行列式均为正，此处abs省去逆时针排列顶点的麻烦
            cell.Volume = np.sum(np.abs([face.det for face in cell.Faces]))/6
            if cell.Volume == 0:
                assert 'cell with 0 volume!'
            self.Cells.update({index:cell})
            return cell
    def find_face(self,i):
        pos = self.voronoi.points
        ridge_points_index =  np.argwhere(self.voronoi.ridge_points == i)[:,0]
        Faces = []
        for j in ridge_points_index:
            pos_index=self.voronoi.ridge_points[j]
            Faces.append(Polyface(Vertices=self.voronoi.vertices[self.voronoi.ridge_vertices[j]], Face_index = j,adj_points=pos[pos_index[pos_index!=i]],adj_index = pos_index[pos_index!=i][0]))
        return Faces
    def show_in_graph(self,ax,i):
        pos = self.voronoi.points
        ax.scatter(*self.voronoi.points.T,c='grey',alpha=0.5)
        ax.scatter(*pos[i],c='r',marker='*')
        Faces = self.find_face(i)
        for face in Faces:
            ptp = np.vstack((face.adj_points,face.adj_points[0]))
            vts = np.vstack((face.Vertices,face.Vertices[0]))
            ax.scatter(*ptp.T,ls='--' ,c='r')
            ax.plot(*vts.T,c='r')
            ax.scatter(*face.centroid,c='b',s=2)
            ax.quiver(*face.centroid, *(face.centroid+face.Areas), length=0.1, normalize=True, color='blue')
    def show_all_face(self,ax):
        ax.scatter(*self.voronoi.points.T,c='grey',alpha=0.5)
        for face in self.voronoi.ridge_vertices[:100]:
            Vertices = np.vstack((self.voronoi.vertices[face],self.voronoi.vertices[face][0]))
            #print(Vertices.T)
            ax.plot(*Vertices.T,c='r')
    def save(self,output_dir,i):
        f_vor = dict()
        for key in self.voronoi.__dict__.keys():
            if key[0] !='_':
                f_vor.update({key:self.voronoi.__dict__[key]})
        for key in self.__dict__.keys():
            if key != 'voronoi':
                f_vor.update({key:self.__dict__[key]})
        f_vor.update({'points':self.voronoi.points})
        import pickle
        with open(output_dir+'/snapshot{:03d}'.format(i), "wb") as f:
            pickle.dump(f_vor, f)
class Cell(VoronoiMesh):
    def __init__(self,index,pos):
        self.coordinate = pos
        self.index = index
        self.alive=True
        #self.Faces = self.find_face(index)
        #Voronoi多面体是凸多面体，所有多边形行列式均为正，此处abs省去逆时针排列顶点的麻烦
        #self.Volume = np.sum(np.abs([face.det for face in self.Faces]))/6
    def __repr__(self):
        return str(self.index)+str(self.coordinate)
        
        
class Polyface:
    def __init__(self,Vertices,Face_index,adj_points,adj_index):
        self.Vertices = Vertices
        self.Face_index=Face_index
        self.adj_index = adj_index
        self.lines = []
        self.adj_points = adj_points
        self.cal_centroid()
        self.cal_Areas()
        self.cal_det()
    def cal_centroid(self):
        self.centroid = np.mean(self.Vertices,axis=0)
    def cal_Areas(self):
        p0 = self.Vertices[0]
        self.Areas = np.zeros(3)
        for i in range(2,len(self.Vertices)):
            v1=self.Vertices[i]-p0
            v2=self.Vertices[i-1]-p0
            self.Areas +=np.cross(v1,v2)/2
    def cal_det(self):
        p0 = self.Vertices[0]
        self.det = 0 
        for i in range(3,len(self.Vertices)):
            self.det+=np.linalg.det(np.array([p0,self.Vertices[i-1],self.Vertices[i]]).T)
    def add(self, p):
        self.Vertices.append(p)
        if len(self.Vertices) > 1:
            self.lines.append([self.Vertices[-2], self.Vertices[-1]])
            
if __name__=='__main__':
    points=np.random.rand(10**3, 3)
    import time
    start = time.time()
    cells = Create_Voronoi(1,points)
    print(time.time()-start)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cells.show_in_graph(ax,12)
    #cells.show_all_face(ax)
    plt.show();
    