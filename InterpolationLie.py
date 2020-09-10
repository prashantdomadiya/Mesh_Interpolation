
bl_info = {
    "name": "Interpolation By Lie",
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
        ob = bpy.data.objects.new('Itr_Lie'+str(i), me)
        bpy.context.collection.objects.link(ob)
        me.from_pydata(E[:,3*i:3*i+3], [], F)
        me.update()

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #               Face Transformation
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def ConnectionMatrices(fcs,NV,NF):
    A=sp.lil_matrix((3*NF,NV))
    t=0
    for f in fcs:
        A[3*t:3*t+3,f]=np.array([[0,0,0],[-1,1,0],[-1,0,1]])
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


def VecRotation(rotateTowardVec, targetVec):
    # Rotate 'targetVec' towards 'rotateTowardVec'
    w=np.cross(targetVec,rotateTowardVec)
    if np.linalg.norm(w)==0.0:
        if np.linalg.norm(targetVec)!=-targetVec[0]:
            R=np.eye(3)
        else:
            R=np.array([[-1,0,0],[0,-1,0],[0,0,1.0]])
    else:
        w=w/np.linalg.norm(w)
        Dot_prdct=np.dot(rotateTowardVec,targetVec)
        tmp=Dot_prdct/(np.linalg.norm(rotateTowardVec)*np.linalg.norm(targetVec))
        if tmp>1.0:
            theta=0.0
        else:
            theta=np.arccos(tmp)
        S=np.sin(theta)
        C=np.cos(theta)
        T=1-C
        R=np.array([[T*(w[0]**2)+C,T*w[0]*w[1]-S*w[2], T*w[0]*w[2]+S*w[1]],[T*w[0]*w[1]+S*w[2],T*(w[1]**2)+C,T*w[1]*w[2]-S*w[0]],[T*w[0]*w[2]-S*w[1],T*w[1]*w[2]+S*w[0],T*(w[2]**2)+C]])
    return R    

def CanonicalForm(T):
    C=np.array([np.sqrt(np.sum(T[1]**2)),0,0])
    R1=VecRotation(C,T[1])
    X=R1.dot(T[2])
    c=X[1]/(np.sqrt(np.sum(X[1:]**2)))
    s=-X[2]/(np.sqrt(np.sum(X[1:]**2)))
    R2=np.array([[1,0,0],[0,c,-s],[0,s,c]])
    return np.dot(R2,R1)

def GetScaleAndTransform(T1,T2):
    
    S=abs(T2[1,0]/T1[1,0])
    if T1[2,1]==0:
        T1[2,1]=0.01
    A=np.array([[1,(T2[2,0]-S*T1[2,0])/T1[2,1],0],[0,T2[2,1]/T1[2,1],0],[0,0,1]])
     
    return S,A


def GetDeformation(Tx,Ty):
    Rx=CanonicalForm(Tx-Tx[0])
    Ry=CanonicalForm(Ty-Ty[0])
    
    R=(Ry.T).dot(Rx)
    Scl,A=GetScaleAndTransform(np.dot(Rx,(Tx-Tx[0,:]).T).T,
                               np.dot(Ry,(Ty-Ty[0,:]).T).T)
    return Scl,A,Rx,R

def LogA(A):
    lgA=np.zeros(2)
    lgA[0]=np.log(A[1,1])
    if A[1,1]==1.0:
        lgA[1]=A[0,1]
    else:
        lgA[1]=(A[0,1]*lgA[0])/(A[1,1]-1)
    return lgA

def ComputeDeformation(F,Vrt,NVrt,NF):
    FtrsVec=np.zeros((4*NF,1))
    Axs=np.zeros((3*NF,1))
    RefRot=np.zeros((3*NF,3))
    t=0
    for f in F:
        RF=Vrt[f,0:3]
        DF=Vrt[f,3:6]
        Scl,A,Rx,R=GetDeformation(RF,DF)
        RefRot[3*t:3*t+3]=Rx
        FtrsVec[4*t,0],Axs[3*t:3*t+3,0]=RotMatToAnglAxis(R)
        FtrsVec[4*t+1:4*t+3,0]=LogA(A)
        FtrsVec[4*t+3,0]=np.log(Scl)
        t+=1 
    return FtrsVec,Axs,RefRot

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

def GetDefVec(RefRot,Ftrs):
    fcN=int(Ftrs[7])
    Rx=RefRot[3*fcN:3*fcN+3]
    
    Scl=np.exp(Ftrs[3])
    
    w=Ftrs[4:7]
    angl=Ftrs[0]
    C=np.cos(angl)
    S=np.sin(angl)     
    T=1-C
    R=np.array([[T*(w[0]**2)+C,T*w[0]*w[1]-S*w[2], T*w[0]*w[2]+S*w[1]],
                [T*w[0]*w[1]+S*w[2],T*(w[1]**2)+C,T*w[1]*w[2]-S*w[0]],
                [T*w[0]*w[2]-S*w[1],T*w[1]*w[2]+S*w[0],T*(w[2]**2)+C]])
    

    V=np.exp(Ftrs[1])
    if Ftrs[1]!=0.0:
        U=(V-1)*(Ftrs[2]/Ftrs[1])
    else:
        U=Ftrs[2]
    A=np.array([[1,U,0],[0,V,0],[0,0,1.0]])
    Q=(R.dot(Rx.T)).dot(A.dot(Scl*Rx))
    return Q


def ReconstructPoses(D,Axs,RefRot,Vref,NF):
    Vdef=np.zeros(np.shape(Vref))
    
    p=Pool()
    func=partial(GetDefVec,RefRot)
    X=np.reshape(range(NF),(NF,1))
    PrllInpt=np.append(np.reshape(D,(NF,4)),np.reshape(Axs,(NF,3)),axis=1)
    PrllInpt=np.append(PrllInpt,X,axis=1)
    PrllOut=p.map(func,PrllInpt)
    for i in range(NF):
        Q=PrllOut[i]
        Vdef[3*i:3*i+3]=Q.dot(Vref[3*i:3*i+3].T).T
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
        
        if self.seqType=='ref':
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
    bl_label = "Intr Tools"
    seqType : bpy.props.StringProperty()
    def execute(self,context):
        global FilePath
        tmp=np.loadtxt(FilePath+'ref_vrt.txt',delimiter=',')
        NVrt=len(tmp)//3
        Vrt=np.zeros((NVrt,6))
        Vrt[:,0:3]=np.reshape(tmp,(NVrt,3))
        Vrt[:,3:6]=np.reshape(np.loadtxt(FilePath+'def_vrt.txt',delimiter=','),(NVrt,3))
        F=ReadTxt(FilePath+'faces.txt')
        Fful=ReadTxt(FilePath+'Fulfaces.txt')
        
        NF=len(F)
        NPs=len(Vrt.T)//3
        strttime=time.time()
        X=ConnectionMatrices(F,NVrt,NF)
        factor=chmd.cholesky(((X[:,:(NVrt-1)].transpose().dot(X[:,:(NVrt-1)]))).tocsc())
        
        D,Axs,RefRot=ComputeDeformation(F,Vrt,NVrt,NF)
        print("Precomputation.....",time.time()-strttime)
        
        itrPoses=context.scene.NumOfPoses
        InlTm=context.scene.MinTime
        FnlTm=context.scene.MaxTime
        stepSize=(FnlTm-InlTm)/(itrPoses-1)
        I=[]
        for i in range(itrPoses):
            I.append(InlTm+i*stepSize)
        I=np.array([I])
        
        H=D.dot(I)
        Vref=X.dot(Vrt[:,0:3])
        Vdef=np.zeros((3*NF,3*itrPoses))
        strttime=time.time()
        for i in range(itrPoses):
            Vdef[:,3*i:3*i+3]=ReconstructPoses(H[:,i],Axs,RefRot,Vref,NF)
        Poses=np.zeros([NVrt,3*itrPoses])
        Poses[:(NVrt-1)]=factor(X[:,:(NVrt-1)].transpose().dot(Vdef))
        print("Genarating poses.....",time.time()-strttime)
        for i in range(itrPoses):
            CreateMesh(Poses[:,3*i:3*i+3]-np.mean(Poses[:,3*i:3*i+3],axis=0),Fful,1)
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


