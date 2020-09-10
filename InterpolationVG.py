

bl_info = {
    "name": "Interpolation",
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
import bmesh as bm
import numpy as np

from scipy import sparse as sp
from scipy.sparse import linalg as sl
from scipy.sparse.linalg import inv
from sksparse import cholmod as chmd
from functools import reduce,partial
from multiprocessing import Pool
import time
import itertools as it
import shutil

##############################################################################################
#                      Functionas
##############################################################################################

factor=1

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
def ConnectionMatrices(fcs,NV,NVec):
    B=sp.lil_matrix((NV,NVec),dtype=float)
    A=sp.lil_matrix((NV,NVec),dtype=float)
    tt=0
    for t in fcs:
        NVF=len(t)
        B[t,tt:tt+NVF]=1.0/NVF
        A[t,tt:tt+NVF]=np.eye(NVF)
        tt+=NVF
    return A-B





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


def ComputeDeformation(F,Vrt,NVrt,NVec):
    Axis=np.zeros([3*NVec])
    SclAngl=np.zeros([2*NVec,1])
    t=0
    
    for f in F:
        
        RVec=Vrt[0:3,f].T-np.mean(Vrt[0:3,f].T,axis=0)
        PVec=Vrt[3:6,f].T-np.mean(Vrt[3:6,f].T,axis=0)
        
        if len(f)>2:
            Nr=np.mean(np.cross(RVec,np.roll(RVec,-1,axis=0)),axis=0)
            Nr=Nr/np.linalg.norm(Nr)
            
            RF=np.zeros((3,3))
            Rscl=np.linalg.norm(RVec,axis=1)
            RF[2]=Nr

            
            PF=np.zeros((3,3))
            PNr=np.mean(np.cross(PVec,np.roll(PVec,-1,axis=0)),axis=0)
            PF[2]=PNr/np.linalg.norm(PNr)
            
            for vc in range(len(f)):
                RF[0]=RVec[vc]/np.linalg.norm(RVec[vc])
                RF[1]=np.cross(RF[2],RF[0])/np.linalg.norm(np.cross(RF[2],RF[0]))
                
                PF[0]=PVec[vc]/np.linalg.norm(PVec[vc])
                PF[1]=np.cross(PF[2],PF[0])/np.linalg.norm(np.cross(PF[2],PF[0]))
                
                R=np.dot(PF.T,RF)
                
                
                SclAngl[2*t+2*vc+1,0],Axis[3*t+3*vc:3*t+3*vc+3]=RotMatToAnglAxis(R)
                SclAngl[2*t+2*vc,0]=np.log(np.linalg.norm(PVec[vc])/Rscl[vc])
                
                
        else:
            W=np.cross(RVec[0],PVec[0])
            W=W/np.linalg.norm(W)
            angl=np.arccos(np.dot(RVec[0],PVec[0])/(np.linalg.norm(RVec[0])*np.linalg.norm(PVec[0])))
            for vc in range(len(f)):
                SclAngl[2*t+2*vc+1,0]=angl
                Axis[3*t+3*vc:3*t+3*vc+3]=W
                        
        t+=len(f)
        
    return SclAngl, Axis

def Interpolate(D,Alphas):
    IP=D.dot(Alphas)
    return IP

def GetDefVec(Ftrs):
    scl=np.exp(Ftrs[0])
    angl=Ftrs[1]
    w=Ftrs[2:5]
    Vref=Ftrs[5:]
    C=np.cos(angl)
    S=np.sin(angl)     
    T=1-C
    R=np.array([[T*(w[0]**2)+C,T*w[0]*w[1]-S*w[2], T*w[0]*w[2]+S*w[1]],
                [T*w[0]*w[1]+S*w[2],T*(w[1]**2)+C,T*w[1]*w[2]-S*w[0]],
                [T*w[0]*w[2]-S*w[1],T*w[1]*w[2]+S*w[0],T*(w[2]**2)+C]])
    Vdef=scl*(R.dot(Vref))
    return Vdef

def ReconstructPoses(D,Axis,Vref,NVec):
    Vdef=np.zeros(np.shape(Vref))
    p=Pool()
    PrllInpt=np.concatenate((np.reshape(D,(NVec,2)),np.reshape(Axis,(NVec,3)),Vref.T),axis=1)
    PrllOut=p.map(GetDefVec,PrllInpt)
    for i in range(NVec):
        Vdef[:,i]=PrllOut[i]
    p.close()
    return Vdef

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
#           Global Variable
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
        self.layout.operator("dt.tools",text='Interpolate').seqType="Itr"
        self.layout.prop(context.scene,"MyDirPath")
        self.layout.prop(context.scene,"PoseNum")
        self.layout.operator("write.dir",text='Save').seqType='DT'
        
class SaveObj(bpy.types.Operator):
    bl_idname = "write.dir"
    bl_label = "Write_in_dir"
    seqType : bpy.props.StringProperty()
    def execute(self,  context):
       
        Path=context.scene.MyDirPath
        
        PoseN=context.scene.PoseNum
        
        if not os.path.exists(Path):
            os.mkdir(tdir)

        Selected_Meshes=bpy.context.selected_objects
        bpy.context.view_layer.objects.active = Selected_Meshes[i]
        InputPath=Path+"/"
        for i in range(len(Selected_Meshes)):
            filepath=join(InputPath,'%05d.obj' % (PoseN+i))
            bpy.context.view_layer.objects.active = Selected_Meshes[i]
            obj = bpy.context.active_object
            with open(filepath, 'w') as f:
                f.write("# OBJ file\n")
                for v in obj.data.vertices:
                    co_final= obj.matrix_world @ v.co
                    f.write("v %.4f %.4f %.4f\n" % (co_final.x,co_final.z,-co_final.y))
                for p in obj.data.polygons:
                    f.write("f")
                    tmp=[a for a in p.vertices[:]]
                    for i in tmp:
                        f.write(" %d" % (i + 1))
                    f.write("\n")
        return{'FINISHED'}

class GetSequence(bpy.types.Operator):
    bl_idname = "get.seq"
    bl_label = "Target Sequence"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global FilePath
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object
        
        V=np.zeros([3,len(obj.data.vertices)])
        t=0
        for v in obj.data.vertices:  
            co_final= obj.matrix_world @ v.co
            V[:,t]=np.array([co_final.x,co_final.y,co_final.z])
            t+=1
        np.savetxt(FilePath+self.seqType+'_vrt.txt',V,delimiter=',')
        
        if self.seqType=='ref':
            Fful=[]
            if len(obj.data.polygons)!=0:
                if len(obj.data.polygons)!=0:
                    for f in obj.data.polygons:
                        Fful.append(list(f.vertices))
                    FcList=ComuteReducedFaceList(Fful)
                F=[]    
                k=0
                for f in obj.data.polygons:
                    if k in FcList:
                        F.append(list(f.vertices))
                    k+=1
            else:
                for e in obj.data.edges:
                    Fful.append(list(e.vertices))
                F=[f for f in Fful]

        
            WriteAsTxt(FilePath+'faces.txt',F)
            WriteAsTxt(FilePath+'Fullfaces.txt',Fful)
        return{'FINISHED'}



class InterpolationTools(bpy.types.Operator):
    bl_idname = "dt.tools"
    bl_label = "DT Tools"
    seqType : bpy.props.StringProperty()
    def execute(self,context):
        global FilePath, factor
        
        Vrt=np.loadtxt(FilePath+'ref_vrt.txt',delimiter=',')
        tmp=np.loadtxt(FilePath+'def_vrt.txt',delimiter=',')
        Vrt=np.concatenate((Vrt,tmp),axis=0)
        F=ReadTxt(FilePath+'faces.txt')
        Fful=ReadTxt(FilePath+'Fullfaces.txt')
        
        Flen=[len(i) for i in F]
        NVec=sum(Flen)
        NVrt=len(Vrt.T)
        NPs=len(Vrt)//3
        print('Computing Deformations....')
        strttime=time.time()
        AB=ConnectionMatrices(F,NVrt,NVec)
        print(NVrt)
        #factor=chmd.cholesky_AAt(AB[:(NVrt-1),:].tocsc())
        factor=chmd.cholesky((AB[:(NVrt-1),:].dot(AB[:(NVrt-1),:].transpose())).tocsc())
        SclAngl,Axis=ComputeDeformation(F,Vrt,NVrt,NVec)
        print("Precomputation.....",time.time()-strttime)
        itrPoses=context.scene.NumOfPoses
        InlTm=context.scene.MinTime
        FnlTm=context.scene.MaxTime
        stepSize=(FnlTm-InlTm)/(itrPoses-1)
        I=[]
        for i in range(itrPoses):
            I.append(InlTm+i*stepSize)
        I=np.array([I])  

        Vref=((AB.transpose()).dot(Vrt[0:3].T)).T
        H=SclAngl.dot(I)
        
        print('Interpolating....')
        Vdef=np.zeros((3*itrPoses,NVec))
        strttime=time.time()
        for i in range(itrPoses):
            Vdef[3*i:3*i+3]=ReconstructPoses(H[:,i],Axis,Vref,NVec)
        Poses=np.zeros([NVrt,3*itrPoses])
        Poses[:NVrt-1]=factor(AB[:(NVrt-1),:].dot(Vdef.T))
        print(time.time()-strttime)
        for i in range(itrPoses):
            CreateMesh(Poses[:,3*i:3*i+3]-np.mean(Poses[:,3*i:3*i+3],axis=0),Fful,1)
        
        return {'FINISHED'}

def register():
    
    bpy.utils.register_class(ToolsPanel)
    bpy.utils.register_class(GetSequence)
    bpy.utils.register_class(InterpolationTools)
    bpy.utils.register_class(SaveObj)
    bpy.types.Scene.MyDirPath=bpy.props.StringProperty(name="Dir Path", description="My directory",default="default")
    bpy.types.Scene.PoseNum=bpy.props.IntProperty(name="Pose Num",description="Pose",default=0)
    bpy.types.Scene.NumOfPoses=bpy.props.IntProperty(name="Num Of Poses",description="Pose",default=0)
    bpy.types.Scene.MinTime = bpy.props.FloatProperty(name = "tMin", description = "Minimum time", default=0.0)
    bpy.types.Scene.MaxTime = bpy.props.FloatProperty(name = "tMax", description = "Minimum time", default=1.0)
    
    
def unregister():
    
    bpy.utils.unregister_class(ToolsPanel)
    bpy.utils.unregister_class(GetSequence)
    bpy.utils.unregister_class(InterpolationTools)
    bpy.utils.unregister_class(SaveObj)
    del bpy.types.Scene.MyDirPath
    del bpy.types.Scene.PoseNum
    del bpy.types.Scene.NumOfPoses
    del bpy.types.Scene.MinTime
    del bpy.types.Scene.MaxTime
 
if __name__ == "__main__":  # only for live edit.
    bpy.utils.register_module(__name__) 


