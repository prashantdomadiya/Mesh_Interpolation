
bl_info = {
    "name": "Manifold Learning",
    "author": "Prashant Domadiya",
    "version": (1, 0),
    "blender": (2, 82, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Manifold Learning",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Development"}

LibPath='/home/prashant/anaconda3/envs/Blender282/lib/python3.8/site-packages/'
FilePath='/media/prashant/DATA/MyCodes/Interpolation/'

import sys
import os
from os.path import join
sys.path.append(LibPath)

import bpy
import bmesh as bm
import numpy as np


from scipy import sparse as sp
from sksparse import cholmod as chmd
from scipy.sparse.linalg import inv

#from functools import reduce,partial
#from multiprocessing import Pool
import time
import itertools as itr
import shutil


def ConnectionMatrices(fcs,NV,NVec,CnsVrt,FrVrt):
    
    NCvrt=len(CnsVrt)
    AB=sp.lil_matrix((NV-NCvrt,NVec))
    ABc=sp.lil_matrix((NCvrt,NVec))
    tt=0
    for f in fcs:
        NVF=len(f)
        tmp=np.eye(NVF)-np.ones(NVF)/NVF
        j=0
        for i in f:
            if i in CnsVrt:
                ABc[CnsVrt.index(i),tt:tt+NVF]=tmp[j]       
            else:
                AB[FrVrt.index(i),tt:tt+NVF]=tmp[j]
            j+=1
        tt+=NVF
    return AB,ABc


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

def ComputeDeformation(RR,DR,Rscl,Dscl,Ncl):
    Axis=np.zeros(3*Ncl)
    SclAngl=np.zeros((2*Ncl,1))
    
    for c in range(Ncl):
        R=DR[:,3*c:3*c+3].dot(RR[:,3*c:3*c+3].T)
        SclAngl[2*c+1,0],Axis[3*c:3*c+3]=RotMatToAnglAxis(R)
        SclAngl[2*c,0]=np.log(Dscl[c]/Rscl[c])   
        
    return SclAngl, Axis


def ComputeDeformationMrph(F,Vrt,NVrt,NVec):
    Axis=np.zeros([3*NVec])
    SclAngl=np.zeros([2*NVec,1])
    t=0
    for f in F:
        
        RVec=Vrt[0:3,f].T-np.mean(Vrt[0:3,f].T,axis=0)
        PVec=Vrt[3:6,f].T-np.mean(Vrt[3:6,f].T,axis=0)
        
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
                        
        t+=len(f)
        
    return SclAngl, Axis


def GetBoneDef(Ftrs,RR,Rscl, Axs,Ncl,NPs):
    Rd=np.zeros((3*NPs,3*Ncl))
    for i in range(Ncl):
        w=Axs[3*i:3*i+3]
        for p in range(NPs):
            scl=np.exp(Ftrs[2*i,p])
            angl=Ftrs[2*i+1,p]
            C=np.cos(angl)
            S=np.sin(angl)     
            T=1-C
            R=np.array([[T*(w[0]**2)+C,T*w[0]*w[1]-S*w[2], T*w[0]*w[2]+S*w[1]],
                        [T*w[0]*w[1]+S*w[2],T*(w[1]**2)+C,T*w[1]*w[2]-S*w[0]],
                        [T*w[0]*w[2]-S*w[1],T*w[1]*w[2]+S*w[0],T*(w[2]**2)+C]])
            Rd[3*p:3*p+3,3*i:3*i+3]=((Rscl[i]*scl)*(R.dot(RR[:,3*i:3*i+3])))
    return Rd

def GetBoneDef2(Ftrs,RR,Rscl, Axs,Ncl,NPs):
    Rd=np.zeros((3*NPs,3*Ncl))
    for i in range(Ncl):
        w=Axs[3*i:3*i+3]
        for p in range(NPs):
            scl=np.exp(Ftrs[2*i,p])
            angl=Ftrs[2*i+1,p]
            C=np.cos(angl)
            S=np.sin(angl)     
            T=1-C
            R=np.array([[T*(w[0]**2)+C,T*w[0]*w[1]-S*w[2], T*w[0]*w[2]+S*w[1]],
                        [T*w[0]*w[1]+S*w[2],T*(w[1]**2)+C,T*w[1]*w[2]-S*w[0]],
                        [T*w[0]*w[2]-S*w[1],T*w[1]*w[2]+S*w[0],T*(w[2]**2)+C]])
            Rd[3*p:3*p+3,3*i:3*i+3]=(Rscl[i]*scl)*(RR[:,3*i:3*i+3].dot(R))
    return Rd

def ReadCorrespondences(fileName):
    Cr=[]
    fl = open(fileName, 'r')
    j=0
    for line in fl:
        words = line.split()
        Cr.append(int(words[0]))
        j+=1
    return Cr


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
        ob = bpy.data.objects.new('VG'+str(i), me)
        bpy.context.scene.objects.link(ob)
        me.from_pydata(E[:,3*i:3*i+3], [], F)
        me.update()
        
def UpdateMesh(Vertz):
    Selected_Meshes=bpy.context.selected_objects
    for i in range(len(Selected_Meshes)):
        bpy.context.scene.objects.active = Selected_Meshes[i]
        obj = bpy.context.active_object
        j=0
        for v in obj.data.vertices:
            v.co=Vertz[j]#Vertz[3*j:3*j+3,i]
            j+=1
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #               Face Transformation
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def ItrParameters(self, context):
    global itrPoses,SclAngl,ftrs
    itrPoses=context.scene.NumOfPoses
    InlTm=context.scene.MinTime
    FnlTm=context.scene.MaxTime
    stepSize=(FnlTm-InlTm)/(itrPoses-1)
    I=[]
    for i in range(itrPoses):
        I.append(InlTm+i*stepSize)
    I=np.array([I])
    ftrs=SclAngl.dot(I)
    return

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
                for t in itr.combinations(F[f],2):
                    if len(set(list(t)).intersection(FV))==0:
                        ASF.append(f)
                        FV.update(F[f])      
                        break
            B.update(ASF)
        else:
            B.update(X)
                    
    bpy.ops.object.mode_set(mode ="OBJECT")
    return B

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


def ReadStringList(Name):
    Lst=[]
    fl = open(Name, 'r')
    NumLine=0
    for line in fl:
        words = line.split()
        Lst=Lst+words
    return Lst

##############################################################################################
#                      Global Variable
##############################################################################################
def LoadRigWeight(self, context):
    global ActiveModel,FilePath,path,Fful,NVec,MpsRot1,MpsScl1,MpsIdx1,Ncl,Cls,NV,RefMesh,Phi,Ainc,VecCls,AAt
    if len(RigList)!=0:
        ActiveModel=RigList[context.scene.Model1] #
        path=FilePath+ActiveModel+'/' #
        print('Loading Model '+ActiveModel)
        MpsRot1=np.loadtxt(path+ActiveModel+'_MpsRot.txt',delimiter=',') #
        MpsScl1=np.loadtxt(path+ActiveModel+'_MpsScl.txt',delimiter=',') #
        MpsIdx1=ReadTxt(path+ActiveModel+'_MpsIdx.txt')[0] #
        Fful=ReadTxt(path+ActiveModel+'_facz.txt')
        RefMesh=np.loadtxt(path+ActiveModel+'_RefMesh.txt',delimiter=',')#
        NVec=len(MpsScl1)
        NV=len(RefMesh)
        VecCls=ReadTxt(path+ActiveModel+'_VecCls.txt')
        Ncl=len(VecCls)
        Phi=np.loadtxt(path+ActiveModel+'_SegPhi.txt',delimiter=',')
        Ainc=sp.load_npz(path+ActiveModel+'_IncidentMtrx.npz')
    return


##############################################################################################

path=''
Phi=[]
Fful=[]
NVec=1
Ncl=1
NV=1
VecCls=[]
ActiveModel=''
# Morphing and DT
MpsRot1=[]
MpsScl1=[]
MpsIdx1=[]
MrphFtrs=[]
MrphAxs=[]
RefMesh=[]
# Interpolation
RR=[]
Rscl=[]
SclAngl=[]
Axs=[]
ftrs=[]
itrPoses=[]
SclAngl=[]
Ainc=0
AAt=0


if os.path.isfile(FilePath+'Riglist.txt'):
    RigList=ReadStringList(FilePath+'Riglist.txt')
else:
    RigList=[]

    
##############################################################################################
#                                   Tools
##############################################################################################
    
class ToolsPanel(bpy.types.Panel):
    bl_label = "VG Animation Tools Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
 
    def draw(self, context):     
        self.layout.prop(context.scene,"Model1")
        self.layout.label(text="Morphing")
        self.layout.operator("trgt.ms",text='target').seqType="trgt"
        self.layout.prop(context.scene,"Mrphtime")
        self.layout.operator("mrph.tool",text='Morph').seqType="mrph"

        self.layout.label(text="Deformation Transfer")
        self.layout.prop(context.scene,"FaceFilePath")
        self.layout.operator("dt.tool",text='Target').seqType="dt"
        self.layout.operator("get.dt",text='DefTrans').seqType="dt"
        
        self.layout.label(text="Interpolator")
        self.layout.operator("get.mesh",text='Ref Mesh').seqType="ref"
        self.layout.operator("get.mesh",text='Def Mesh').seqType="def"
        self.layout.prop(context.scene,"NumOfPoses")
        self.layout.prop(context.scene,"MinTime")
        self.layout.prop(context.scene,"MaxTime")
        self.layout.operator("itr.tool",text='Intr').seqType="itr"

        self.layout.label(text="Save As Obj")
        self.layout.prop(context.scene,"MyDirPath")
        self.layout.prop(context.scene,"PoseNum")
        self.layout.operator("write.dir",text='Save').seqType='sv'
#################################################################################################

class GetTargetMesh(bpy.types.Operator):
    bl_idname = "trgt.ms"
    bl_label = "Target Mesh"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global Fful,NV,NVec,RefMesh,MrphFtrs,MrphAxs
        
        obj = bpy.context.active_object
        
        Vrt=np.zeros((3,NV))
       
        t=0
        for v in obj.data.vertices:
            co_final= obj.matrix_world @ v.co
            Vrt[:,t]=np.array([co_final.x,co_final.z,-co_final.y])
            t+=1   
        Ms=np.append(RefMesh.T,Vrt,axis=0)
        
        MrphFtrs,MrphAxs=ComputeDeformationMrph(Fful,Ms,NV,NVec)
        
        return{'FINISHED'}
    
class Morph(bpy.types.Operator):
    bl_idname = "mrph.tool"
    bl_label = "Morph two Models"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global MrphFtrs,MrphAxs,NVec,Ncl,NV,MpsRot1,MpsScl1,MpsIdx1,Fful,Phi,RefMesh

        #NV=len(RefMsh.T)
        CnsVrt=[0]
        
        FrVrt=[i for i in range(NV) if i not in CnsVrt]
        
        AB,ABc=ConnectionMatrices(Fful,NV,NVec,CnsVrt,FrVrt)
        factor=chmd.cholesky_AAt(AB.tocsc())
        
        T=context.scene.Mrphtime
        DR=GetBoneDef2(T*MrphFtrs,MpsRot1,MpsScl1,MrphAxs,NVec,1)
        
        h=0
        j=0
        W=sp.lil_matrix((3*Ncl,NVec))
        for f in Fful:
            Vrc=(RefMesh[f]-np.mean(RefMesh[f],axis=0))
            for t in range(len(f)):
                W[3*MpsIdx1[j]:3*MpsIdx1[j]+3,h+t]=np.reshape(DR[:,3*(h+t):3*(h+t)+3].dot(Vrc[t]),(3,1))
            h+=len(f)
            j+=1
            
        tmp=((factor((AB.dot(W.transpose())).tocsc())).transpose()).toarray()
        
        
        Phi=np.zeros((3*Ncl,NV))
        for i in range(NV):
            if i not in CnsVrt:
                Phi[:,i]=tmp[:,FrVrt.index(i)]
        ###################
        #CreateMesh(X.T,Fm,itrPoses)
        
        return{'FINISHED'}


class GetTargetDT(bpy.types.Operator):
    bl_idname = "dt.tool"
    bl_label = "Morph two Models"
    seqType: bpy.props.StringProperty()
 
    def execute(self, context):
        global Ncl,MpsRot1,MpsScl1,MpsIdx1,Phi,Fful,ftrs,RR,Rscl,Axs,Ncl,itrPoses
        
        obj = bpy.context.active_object
        
        
        NV=len(obj.data.vertices)
        Vrt=np.zeros((NV,3))
        t=0
        for v in obj.data.vertices:
            co_final= obj.matrix_world @ v.co
            Vrt[t]=np.array([co_final.x,co_final.z,-co_final.y])
            t+=1   
        strttime=time.time()
        
        F=[]
        NVec=0
        for f in obj.data.polygons:
            F.append(list(f.vertices))
            NVec+=len(f.vertices)
        
        CnsVrt=[0]
        FrVrt=list(range(1,NV))
        AB,ABc=ConnectionMatrices(F,NV,NVec,CnsVrt,FrVrt)
        factor=chmd.cholesky_AAt(AB.tocsc())
        
        if context.scene.FaceFilePath!='Default':
            Cr=ReadCorrespondences(context.scene.FaceFilePath)
        else:
            Cr=list(range(len(F)))
           
        Flen=[]
        j=0
        for f in Fful:
            Flen.append(j)
            j+=len(f)
            
        j=0
        h=0
        W=sp.lil_matrix((3*Ncl,NVec))
        Vecs=np.zeros((NVec,3))
        for f in F:
            Vrc=(Vrt[f]-np.mean(Vrt[f],axis=0))
            for t in range(len(f)):
                W[3*MpsIdx1[Cr[j]]:3*MpsIdx1[Cr[j]]+3,h+t]=np.reshape(MpsRot1[:,3*(Flen[Cr[j]]+t):3*(Flen[Cr[j]]+t)+3].dot(MpsScl1[Flen[Cr[j]]+t]*Vrc[t]),(3,1))
            j+=1
            h+=len(f)
                      
        
        tmp=((factor((AB.dot(W.transpose())).tocsc())).transpose()).toarray()
        
        
        Phi=np.zeros((3*Ncl,NV))
        for i in range(NV):
            if i not in CnsVrt:
                Phi[:,i]=tmp[:,FrVrt.index(i)]
        
        Fm=[f for f in F]
        print("Preprocessing time for DT..", time.time()-strttime)
        
        return{'FINISHED'}

class DeformationTransfer(bpy.types.Operator):
    bl_idname = "get.dt"
    bl_label = "Load Skeleton"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global path, ActiveModel,NV, Ainc, Ncl,VecCls, Phi,Fful
        
        obj = bpy.context.active_object
        Sref=np.loadtxt(path+ActiveModel+'_Sr.txt',delimiter=',')
        Aref=np.loadtxt(path+ActiveModel+'_Ar.txt',delimiter=',')
        
        print("Number of Classes...",Ncl)
        SFrm=np.zeros([3,3*Ncl])
        Vrt=np.zeros((3,NV))
       
        t=0
        for v in obj.data.vertices:
            co_final= obj.matrix_world @ v.co
            Vrt[:,t]=np.array([co_final.x,co_final.z,-co_final.y])
            t+=1

        strttime=time.time()
        print('Computing segment Deformation')
        Vec=((Ainc.transpose()).dot(Vrt.T)).T
        Dptch=np.zeros([3,3*Ncl])
        for ci in range(Ncl):
            Vc=Vec[:,VecCls[ci]]
            Scl=np.linalg.norm(Vc,axis=0)
            Vc=Vc/Scl
            
            U,Lmda,V=np.linalg.svd((Sref[:,VecCls[ci]]).dot(Vc.T))
            tmp=(V.T).dot(U.T)
            Dptch[:,3*ci:3*ci+3]=(np.mean(Scl/Aref[VecCls[ci]]))*tmp
        X=Dptch.dot(Phi)
        print("DT time .....", time.time()-strttime)
        CreateMesh(X.T,Fful,1)
        
        return{'FINISHED'}    
        
################################################################################################    
class GetMeshSeq(bpy.types.Operator):
    bl_idname = "get.mesh"
    bl_label = "Load Skeleton"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global path, ActiveModel,SclAngl,Axs,Ncl,RR,Rscl,ftrs,Fful,VecCls,NV,Ainc
        
        obj = bpy.context.active_object
        Sref=np.loadtxt(path+ActiveModel+'_Sr.txt',delimiter=',')
        Aref=np.loadtxt(path+ActiveModel+'_Ar.txt',delimiter=',')
        
        print("Number of Classes...",Ncl)
        SFrm=np.zeros([3,3*Ncl])
        Vrt=np.zeros((3,NV))
       
        t=0
        for v in obj.data.vertices:
            co_final= obj.matrix_world @ v.co
            Vrt[:,t]=np.array([co_final.x,co_final.z,-co_final.y])
            t+=1


        print('Computing segment Deformation')
        Vec=((Ainc.transpose()).dot(Vrt.T)).T
        RSeg=np.zeros([3,3*Ncl])
        ASeg=np.zeros([Ncl])
        for ci in range(Ncl):
            Vc=Vec[:,VecCls[ci]]
            Scl=np.linalg.norm(Vc,axis=0)
            Vc=Vc/Scl
            
            U,Lmda,V=np.linalg.svd((Sref[:,VecCls[ci]]).dot(Vc.T))
            tmp=(V.T).dot(U.T)
            RSeg[:,3*ci:3*ci+3]=tmp
            ASeg[ci]=np.mean(Scl/Aref[VecCls[ci]])    
        
        if self.seqType=='def':
            SclAngl,Axs=ComputeDeformation(RR,RSeg,Rscl,ASeg,Ncl)
            ftrs=SclAngl.dot(np.array([[0.0,1.0]]))
        else:
            RR=1*RSeg
            Rscl=1*ASeg
        return{'FINISHED'}



class Iterpolate(bpy.types.Operator):
    bl_idname = "itr.tool"
    bl_label = "Compute Animation"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global Phi,Axs,Ncl,itrPoses,Fful,RR,Rscl,SclAngl,ftrs

        strttime=time.time()
        DR=GetBoneDef(ftrs,RR,Rscl,Axs,Ncl,itrPoses)
        X=DR.dot(Phi)
        print("Computing "+str(itrPoses)+" poses in "+str(time.time()-strttime)+" time")
        #CreateMesh(X.T,Fful,itrPoses)
        
        return{'FINISHED'}
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
        bpy.context.view_layer.objects.active = Selected_Meshes[len(Selected_Meshes)-1]
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

def register():
    bpy.utils.register_class(ToolsPanel)
    bpy.types.Scene.Model1=bpy.props.IntProperty(name="Model", description="Select Rigged Model", default=0,
                                                 min=0,max=(len(RigList)-1),options={'ANIMATABLE'}, update=LoadRigWeight)
    bpy.utils.register_class(GetTargetMesh)
    bpy.types.Scene.Mrphtime = bpy.props.FloatProperty(name = "mrph", description = "morphing value", default=0.0, min=0.0,max=1.0)
    bpy.utils.register_class(Morph)
    
    bpy.types.Scene.FaceFilePath=bpy.props.StringProperty(name="Face Path", description="My Faces", default="Default")
    bpy.utils.register_class(GetTargetDT)
    bpy.utils.register_class(DeformationTransfer)
    
    bpy.utils.register_class(GetMeshSeq)
    bpy.types.Scene.NumOfPoses=bpy.props.IntProperty(name="Num Of Poses",description="Pose",default=2,min=2,update=ItrParameters)
    bpy.types.Scene.MinTime = bpy.props.FloatProperty(name = "tMin", description = "Minimum time", default=0.0,update=ItrParameters)
    bpy.types.Scene.MaxTime = bpy.props.FloatProperty(name = "tMax", description = "Minimum time", default=1.0,update=ItrParameters)
    bpy.utils.register_class(Iterpolate)

    bpy.utils.register_class(SaveObj)
    bpy.types.Scene.MyDirPath=bpy.props.StringProperty(name="Dir Path", description="My directory",default="default")
    bpy.types.Scene.PoseNum=bpy.props.IntProperty(name="Pose Num",description="Pose",default=0)
    
def unregister():
    bpy.utils.unregister_class(ToolsPanel)
    bpy.utils.unregister_class(GetTargetMesh)
    bpy.utils.unregister_class(GetTargetDT)
    bpy.utils.unregister_class(DeformationTransfer)
    bpy.utils.unregister_class(Morph)
    bpy.utils.unregister_class(GetMeshSeq)
    bpy.utils.unregister_class(Iterpolate)
    del bpy.types.Scene.FaceFilePath
    del bpy.types.Scene.MyDirPath
    del bpy.types.Scene.Mrphtime
    del bpy.types.Scene.Model1
    del bpy.types.Scene.NumOfPoses
    del bpy.types.Scene.MinTime
    del bpy.types.Scene.MaxTime
    
    
    
if __name__ == "__main__":  
    bpy.utils.register_module(__name__) 


