
bl_info = {
    "name": "Interpolation By DG",
    "author": "Prashant Domadiya",
    "version": (1, 1),
    "blender": (2, 82, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Compute Intrpolated Poses by Deformation Gradient",
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
import bmesh as bm
import numpy as np

from scipy import sparse as sp
from scipy.sparse import linalg as sl
from sksparse import cholmod as chmd
import scipy.linalg as scla
from functools import reduce,partial
from multiprocessing import Pool
import time
import itertools as it
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
        ob = bpy.data.objects.new('DG'+str(i), me)
        bpy.context.collection.objects.link(ob)
        me.from_pydata(E[:,3*i:3*i+3], [], F)
        me.update()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#               Face Transformation
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def ConnectionMatrices(fcs,NV,NF):
    A=sp.lil_matrix((6*NF,3*NV))
    t=0
    for f in fcs:
        NVF=len(f)
        FF=np.reshape(3*np.array([f]*3)+np.array([[0],[1],[2]]),3*len(f),order='F')
        A[6*t:6*t+6,FF]=np.array([[-1,0,0,1,0,0,0,0,0],[0,-1,0,0,1,0,0,0,0],[0,0,-1,0,0,1,0,0,0],
                                  [-1,0,0,0,0,0,1,0,0],[0,-1,0,0,0,0,0,1,0],[0,0,-1,0,0,0,0,0,1.0]])
        t+=1
    return A

def RotMatToAnglAxis(R):
    M=np.trace(R)-1
    if M>2:
        M=1.99
    elif M<-2:
        M=-1.99
    else:
        M=M/1
            
    angl=np.arccos(M/2)
    if np.sin(angl)==0:
        Axs=np.zeros(3)
    else:
        Axs=(1/(2*np.sin(angl)))*np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    return angl,Axs


def ComputeDeformation(F,Vrt,NVrt,NF):
    FtrsVec=np.zeros((4*NF,1))
    Axs=np.zeros(3*NF)
    EVecs=np.zeros((3*NF,3))
    t=0
    for f in F:
        RF=np.zeros((3,3))
        PF=np.zeros((3,3))
        FF=np.reshape(3*np.array([f]*3)+np.array([[0],[1],[2]]),3*len(f),order='F')

        
        RF[0:2]=np.reshape(Vrt[FF[3:],0],(2,3))-Vrt[FF[0:3],0]
        Nr=np.cross(RF[0],RF[1])
        Nr=Nr/np.linalg.norm(Nr)
        RF[2]=Nr


        PF[0:2]=np.reshape(Vrt[FF[3:],1],(2,3))-Vrt[FF[0:3],1]
        Nr=np.cross(PF[0],PF[1])
        Nr=Nr/np.linalg.norm(Nr)
        PF[2]=Nr

        
        Q=np.dot(PF.T,np.linalg.inv(RF.T))
        
        R,S=scla.polar(Q)
        
        Evl,EVecs[3*t:3*t+3]=np.linalg.eig(S)
        
        FtrsVec[4*t:4*t+3,0]=np.log(Evl)
        FtrsVec[4*t+3,0],Axs[3*t:3*t+3]=RotMatToAnglAxis(R)
        t+=1 
    return FtrsVec,Axs,EVecs

def Interpolate(D,Alphas):
    IP=D.dot(Alphas)
    return IP

def AnglAxisToRotMat(D):
    w=D[1:]
    C=np.cos(D[0])
    S=np.sin(D[0])     
    T=1-C
    R=np.array([[T*(w[0]**2)+C,T*w[0]*w[1]-S*w[2], T*w[0]*w[2]+S*w[1]],
                [T*w[0]*w[1]+S*w[2],T*(w[1]**2)+C,T*w[1]*w[2]-S*w[0]],
                [T*w[0]*w[2]-S*w[1],T*w[1]*w[2]+S*w[0],T*(w[2]**2)+C]])
    return R
def GetDefVec(EVecs,Ftrs):
    evl=np.diag(np.exp(Ftrs[0:3]))
    fcN=int(Ftrs[7])
    tmp=EVecs[3*fcN:3*fcN+3]
    Scl=(tmp.dot(evl)).dot(np.linalg.inv(tmp))
    
    w=Ftrs[4:7]
    angl=Ftrs[3]

    C=np.cos(angl)
    S=np.sin(angl)     
    T=1-C
    R=np.array([[T*(w[0]**2)+C,T*w[0]*w[1]-S*w[2], T*w[0]*w[2]+S*w[1]],
                [T*w[0]*w[1]+S*w[2],T*(w[1]**2)+C,T*w[1]*w[2]-S*w[0]],
                [T*w[0]*w[2]-S*w[1],T*w[1]*w[2]+S*w[0],T*(w[2]**2)+C]])
    Q=R.dot(Scl)
    return Q

def ReconstructPoses(D,Axs,EVecs,Vref,NF):
    Vdef=np.zeros(np.shape(Vref))
    p=Pool()
    func=partial(GetDefVec,EVecs)
    X=np.reshape(range(NF),(NF,1))
    PrllInpt=np.append(np.reshape(D,(NF,4)),np.reshape(Axs,(NF,3)),axis=1)
    PrllInpt=np.append(PrllInpt,X,axis=1)
    PrllOut=p.map(func,PrllInpt)
    for i in range(NF):
        Q=PrllOut[i]
        for j in range(2):
            Vdef[6*i+3*j:6*i+3*j+3]=Q.dot(Vref[6*i+3*j:6*i+3*j+3])
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

def ComuteReducedFaceList(F):
    bpy.ops.object.mode_set(mode="EDIT")
    obj = bpy.context.active_object.data
    m=bm.from_edit_mesh(obj)
    B=set()
    for vrt in m.verts:
        X=[]
        ASF=[]
        FV=set()
        Num_link_face=0
        for fc in vrt.link_faces:
            Num_link_face+=1
            if fc.index in B:
                ASF.append(fc.index)
                FV.update(F[fc.index])
            else:
                X.append(fc.index)
        if Num_link_face>2:
            for f in X:
                for t in it.combinations(F[f],2):
                    if len(set(list(t)).intersection(FV))==0:
                        ASF.append(f)
                        FV.update(F[f])      
                        break
            B.update(ASF)
        else:
            B.update(X)
                    
    bpy.ops.object.mode_set(mode ="OBJECT")
    return B

#####################################################################################################
#           Deformation Transfer Tool
#####################################################################################################

        

class ToolsPanel(bpy.types.Panel):
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
        self.layout.operator("dt.tools",text='Interpolate').seqType="itr"


class GetSequence(bpy.types.Operator):
    bl_idname = "get.seq"
    bl_label = "Target Sequence"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global FilePath
        
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object

        t=0
        V=np.zeros(3*len(obj.data.vertices))
        for v in obj.data.vertices:  
            co_final= obj.matrix_world @ v.co
            V[3*t:3*t+3]=np.array([co_final.x,co_final.z,-co_final.y])
            t+=1
        np.savetxt(FilePath+self.seqType+'_vrt.txt',V,delimiter=',')
        
        if self.seqType=='def':
            Fful=[]
            for f in obj.data.polygons:
                Fful.append(list(f.vertices))            
            FcList=ComuteReducedFaceList(Fful)
            
            F=[]
            t=0
            for f in obj.data.polygons:
                if t in FcList:
                    F.append(list(f.vertices))
                t+=1
        
            WriteAsTxt(FilePath+'faces.txt',F)
            WriteAsTxt(FilePath+'Fulfaces.txt',Fful)

        return{'FINISHED'}



class InterpolationTools(bpy.types.Operator):
    bl_idname = "dt.tools"
    bl_label = "DT Tools"
    seqType : bpy.props.StringProperty()
    def execute(self,context):
        global FilePath
        tmp=np.loadtxt(FilePath+'ref_vrt.txt',delimiter=',')
        Vrt=np.zeros((len(tmp),2))
        Vrt[:,0]=tmp
        Vrt[:,1]=np.loadtxt(FilePath+'def_vrt.txt',delimiter=',')
        F=ReadTxt(FilePath+'faces.txt')
        Fful=ReadTxt(FilePath+'Fulfaces.txt')
        
        NF=len(F)
        NVrt=int(len(Vrt)/3)
        NPs=len(Vrt.T)
        strttime=time.time()
        X=ConnectionMatrices(F,NVrt,NF)
        factor=chmd.cholesky(((X[:,:3*(NVrt-1)].transpose().dot(X[:,:3*(NVrt-1)]))).tocsc())
        
        D,Axs,EVecs=ComputeDeformation(F,Vrt,NVrt,NF)
        print("Precomputation.....",time.time()-strttime)
        #itrPoses=6
        #I=np.array([list(range(itrPoses))])
        #I=I/(itrPoses-1)
        itrPoses=context.scene.NumOfPoses
        InlTm=context.scene.MinTime
        FnlTm=context.scene.MaxTime
        stepSize=(FnlTm-InlTm)/(itrPoses-1)
        I=[]
        for i in range(itrPoses):
            I.append(InlTm+i*stepSize)
        I=np.array([I])

        H=D.dot(I)
        Vref=X.dot(Vrt[:,0])
        Vdef=np.zeros((6*NF,itrPoses))
        strttime=time.time()
        for i in range(itrPoses):
            Vdef[:,i]=ReconstructPoses(H[:,i],Axs,EVecs,Vref,NF)
        Poses=np.zeros([3*NVrt,itrPoses])
        Poses[:3*(NVrt-1)]=factor(X[:,:3*(NVrt-1)].transpose().dot(Vdef))
        print("Genarating poses.....",time.time()-strttime)
        for i in range(itrPoses):
            tmp=np.reshape(Poses[:,i],(NVrt,3))
            CreateMesh(tmp-np.mean(tmp,axis=0),Fful,1)
        return {'FINISHED'}

def register():
    
    bpy.utils.register_class(ToolsPanel)
    bpy.utils.register_class(GetSequence)
    bpy.utils.register_class(InterpolationTools)
    bpy.types.Scene.NumOfPoses=bpy.props.IntProperty(name="Num Of Poses",description="Pose",default=0)
    bpy.types.Scene.MinTime = bpy.props.FloatProperty(name = "tMin", description = "Minimum time", default=0.0)
    bpy.types.Scene.MaxTime = bpy.props.FloatProperty(name = "tMax", description = "Minimum time", default=1.0)
    
    
    
def unregister():
    
    bpy.utils.unregister_class(ToolsPanel)
    bpy.utils.unregister_class(GetSequence)
    bpy.utils.unregister_class(InterpolationTools)
    del bpy.types.Scene.NumOfPoses
    del bpy.types.Scene.MinTime
    del bpy.types.Scene.MaxTime
 
if __name__ == "__main__":  # only for live edit.
    bpy.utils.register_module(__name__) 


