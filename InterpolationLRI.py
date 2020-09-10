
bl_info = {
    "name": "Interpolation by LRI",
    "author": "Prashant Domadiya",
    "version": (1, 1),
    "blender": (2, 82, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Compute Intrpolated Poses",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Development"}


LibPath='/home/prashant/anaconda3/envs/Blender282/lib/python3.7/site-packages/'
FilePath='/media/prashant/DATA/MyCodes/'

import os
from os.path import join
import sys
sys.path.append(LibPath)



import bpy
import bmesh
import numpy as np
from scipy import sparse as sp
from scipy.sparse import linalg as sl
from sksparse import cholmod as chmd
from scipy import linalg as ll
from functools import reduce,partial
from multiprocessing import Pool
import time
import itertools as itr
import shutil

##############################################################################################
#                      Functionas
##############################################################################################

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                   Display Mesh
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def CreateMesh(V,F,NPs):
    E=np.zeros(np.shape(V))
    F = [ [int(i) for i in thing] for thing in F]
    for i in range(NPs):
        E[:,3*i]=V[:,3*i]
        E[:,3*i+1]=-V[:,3*i+2]
        E[:,3*i+2]=V[:,3*i+1]
        me = bpy.data.meshes.new('MyMesh')
        ob = bpy.data.objects.new('Itr_LRI'+str(i), me)
        bpy.context.collection.objects.link(ob)
        me.from_pydata(E[:,3*i:3*i+3], [], F)
        me.update()



    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #               Face Transformation
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



def ConnectionMatrices(Nbr,NV,NVec):
    A=sp.lil_matrix((3*NVec,3*NV))
    tt=0
    t=0
    for n in Nbr:
        Nn=len(n)
        FF=np.reshape(3*np.array([n]*3)+np.array([[0],[1],[2]]),3*Nn,order='F')
        A[tt:tt+3*Nn,FF]=np.eye(3*Nn)
        A[tt:tt+3*Nn,3*t:3*t+3]=-np.array([[1.0,0,0],[0,1,0],[0,0,1]]*Nn)
        tt+=3*Nn
        t+=1
    return A

def CheckCoffs(M):
    
    if M>1.0:
        M=1.0
    elif M<-1:
        M=-1.0
    else:
        M=M/1
            
    return np.arccos(M)


def ComputeCofficients(Nrml,Nbr,Vrt,NVrt,NVec):
    FtrsVec=np.zeros([4*NVec,2])
    #Frm=np.zeros((3*NVec,3))
    Frm=np.zeros((NVec,9))
    x=0
    t=0
    
    for v in range(NVrt):
        NB=Nbr[v]
        Vec=np.zeros((len(NB),6))
        ProjVec=np.zeros((len(NB),6))
        FF=np.reshape(3*np.array([NB]*3)+np.array([[0],[1],[2]]),3*len(NB),order='F')
        Vec[:,0:3]=np.reshape(Vrt[FF,0],(len(NB),3))-Vrt[3*v:3*v+3,0]
        ProjVec[:,0:3]=Vec[:,0:3]-np.dot(np.reshape(Vec[:,0:3].dot(Nrml[v]),(len(NB),1)),np.reshape(Nrml[v],(1,3)))
        F=np.zeros((3,3))
        F[2]=Nrml[v]
        F[0]=ProjVec[0,0:3]/np.linalg.norm(ProjVec[0,0:3])
        F[1]=np.cross(F[2],F[0])/np.linalg.norm(np.cross(F[2],F[0]))

        Vec[:,3:6]=np.reshape(Vrt[FF,1],(len(NB),3))-Vrt[3*v:3*v+3,1]
        ProjVec[:,3:6]=Vec[:,3:6]-np.dot(np.reshape(Vec[:,3:6].dot(Nrml[v]),(len(NB),1)),np.reshape(Nrml[v],(1,3)))
        for i in range(len(NB)):
            for ps in range(2):
                Coff=np.dot(F,Vec[i,3*ps:3*ps+3])
                
                FtrsVec[t+4*i,ps]=CheckCoffs(Coff[0]/np.linalg.norm(ProjVec[i,3*ps:3*ps+3]))
                FtrsVec[t+4*i+1,ps]=CheckCoffs(Coff[1]/np.linalg.norm(ProjVec[i,3*ps:3*ps+3]))
                FtrsVec[t+4*i+2,ps]=CheckCoffs(Coff[2]/np.linalg.norm(Vec[i,3*ps:3*ps+3]))
                
            FtrsVec[t+4*i+3,0]=1.0
            FtrsVec[t+4*i+3,1]=np.linalg.norm(Vec[i,3:6])/np.linalg.norm(Vec[i,0:3])
            #Frm[3*x:3*x+3]=F.T
            Frm[x]=np.ravel(F.T)
            x+=1
        t+=4*len(NB)
    
    return FtrsVec,Frm




def ConverAnglToCoff(Ftrs):
    D=Ftrs[0:4]
    F=np.reshape(Ftrs[4:],(3,3))
    vec=Ftrs[3]*F.dot(np.cos(Ftrs[0:3]))
    return vec

def ReconstructPoses(D,Frms,Vref,NVec):
    Vdef=np.zeros(np.shape(Vref))
    L=Vref
    p=Pool()
    PrllInpt=np.concatenate((np.reshape(D,(NVec,4)),Frms),axis=1)
    PrllOut=p.map(ConverAnglToCoff,PrllInpt)
    for i in range(NVec):
        Vdef[3*i:3*i+3]=PrllOut[i]*np.linalg.norm(L[3*i:3*i+3])
    p.close()
    return Vdef

def ReadFaces(fileName):
    F=[]
    fl = open(fileName, 'r')
    for line in fl:
        words = line.split()
        l=len(words)
        tmp=[]
        for i in range(l):
            tmp.append(int(words[i]))
        F.append(tmp)
    return F



def WriteAsTxt(Name,Vec):
    with open(Name, 'w') as fl:
        for i in Vec:
            if str(type(i))=="<class 'list'>":
                for j in i:
                    fl.write(" %d" % j)
                fl.write("\n")
            else:
                fl.write(" %d" % i)
                
def ReadTxt(Name):
    Vec=[]
    fl = open(Name, 'r')
    NumLine=0
    for line in fl:
        words = line.split()
        l=len(words)
        tmp=[]
        for i in range(l):
            tmp.append(int(words[i]))
        Vec.append(tmp)
        NumLine+=1
    
    return Vec



#####################################################################################################
#           Deformation Transfer Tool
#####################################################################################################



class DTToolsPanel(bpy.types.Panel):
    bl_label = "Interpolation"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
 
    def draw(self, context):
        
        self.layout.label(text="Interpolation")
        self.layout.operator("get.seq",text='RefPose').seqType="ref"
        self.layout.operator("get.seq",text='DefPose').seqType="def"
        self.layout.prop(context.scene,"NumOfPoses")
        self.layout.prop(context.scene,"MinTime")
        self.layout.prop(context.scene,"MaxTime")
        self.layout.operator("itr.tools",text='Interpolate').seqType="Itr"


class GetSequence(bpy.types.Operator):
    bl_idname = "get.seq"
    bl_label = "Target Sequence"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global FilePath
        
        obj = bpy.context.active_object
        
        V=np.zeros([3*len(obj.data.vertices)])
        bpy.ops.object.editmode_toggle()
        obj = bpy.context.edit_object
        bm = bmesh.from_edit_mesh(obj.data)

        if self.seqType=='ref':
            Nbr=[]
            N=[]

            t=0
            for v in bm.verts:
                co_final= obj.matrix_world @ v.co
                V[3*t:3*t+3]=[co_final.x,co_final.z,-co_final.y]
                t+=1
                
                N.append(list(v.normal))
                tmp=[]
                for e in v.link_edges:
                    tmp.append(e.other_vert(v).index)    
                Nbr.append(tmp)
                    
            F=[]
            for f in obj.data.polygons:
                F.append(list(f.vertices))
            
            np.savetxt(FilePath+'normal.txt',np.array(N),delimiter=',')
            WriteAsTxt(FilePath+'vrtnbr.txt',Nbr)
            WriteAsTxt(FilePath+'face.txt',F)
        else:
            t=0
            for v in bm.verts:
                co_final= obj.matrix_world @ v.co
                V[3*t:3*t+3]=[co_final.x,co_final.z,-co_final.y]
                t+=1
        bpy.ops.object.editmode_toggle()        
        np.savetxt(FilePath+self.seqType+'_vrt.txt',V,delimiter=',')
        
        

        return{'FINISHED'}



class InterpolationTools(bpy.types.Operator):
    bl_idname = "itr.tools"
    bl_label = "DT Tools"
    seqType : bpy.props.StringProperty()
    def execute(self,context):
        global FilePath
        tmp=np.loadtxt(FilePath+'ref_vrt.txt',delimiter=',')
        Vrt=np.zeros((len(tmp),2))
        Vrt[:,0]=tmp
        Vrt[:,1]=np.loadtxt(FilePath+'def_vrt.txt',delimiter=',')
    
        #Vrt=np.loadtxt(path+'source_vrt.txt',delimiter=',')
        N=np.loadtxt(FilePath+'normal.txt',delimiter=',')
        Nbr=ReadTxt(FilePath+'vrtnbr.txt')
        F=ReadTxt(FilePath+'face.txt')
        
        
        Flen=[len(i) for i in Nbr]
        NVec=sum(Flen)
        NVrt=int(len(Vrt)/3)
        NPs=2
        strttime=time.time()
        X=(ConnectionMatrices(Nbr,NVrt,NVec)).tocsc()
        factor=chmd.cholesky(((X[:,:3*(NVrt-1)].transpose()).dot(X[:,:3*(NVrt-1)])).tocsc())
        
        D,Frms=ComputeCofficients(N,Nbr,Vrt,NVrt,NVec)
        print("Precomputation.....",time.time()-strttime)
        itrPoses=context.scene.NumOfPoses
        InlTm=context.scene.MinTime
        FnlTm=context.scene.MaxTime
       
        I=np.zeros((2,itrPoses))
        stpsz=(FnlTm-InlTm)/(itrPoses-1)
        for i in range(itrPoses):
            I[1,i]=InlTm+i*stpsz
            I[0,i]=1-I[1,i]
        

        H=D.dot(I) 
        Vref=X.dot(Vrt[:,0])
        Vdef=np.zeros((3*NVec,itrPoses))
        print('Interpolating....')
        strttime=time.time()
        for i in range(itrPoses):
            Vdef[:,i]=ReconstructPoses(H[:,i],Frms,Vref,NVec)
        Poses=np.zeros([3*NVrt,itrPoses])
        Poses[:3*(NVrt-1)]=factor(X[:,:3*(NVrt-1)].transpose().dot(Vdef))
        print("Genarating poses.....",time.time()-strttime)
        for i in range(itrPoses):
            tmp=np.reshape(Poses[:,i],(NVrt,3))
            CreateMesh(tmp-np.mean(tmp,axis=0),F,1)
        
        return {'FINISHED'}

def register():
    
    bpy.utils.register_class(DTToolsPanel)
    bpy.utils.register_class(GetSequence)
    bpy.utils.register_class(InterpolationTools)
    bpy.types.Scene.NumOfPoses=bpy.props.IntProperty(name="Num Of Poses",description="Pose",default=0)
    bpy.types.Scene.MinTime = bpy.props.FloatProperty(name = "tMin", description = "Minimum time", default=0.0)
    bpy.types.Scene.MaxTime = bpy.props.FloatProperty(name = "tMax", description = "Minimum time", default=1.0)
    
    
def unregister():
    
    bpy.utils.unregister_class(DTToolsPanel)
    bpy.utils.unregister_class(GetSequence)
    bpy.utils.unregister_class(InterpolationTools)
    del bpy.types.Scene.NumOfPoses
    del bpy.types.Scene.MinTime
    del bpy.types.Scene.MaxTime
 
if __name__ == "__main__":  # only for live edit.
    bpy.utils.register_module(__name__) 


