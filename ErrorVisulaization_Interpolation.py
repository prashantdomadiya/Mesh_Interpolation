
bl_info = {
    "name": "Interpolatio Error Calculation",
    "author": "Prashant Domadiya",
    "version": (1, 1),
    "blender": (2, 82, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Compute MSE",
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
import bmesh
import numpy as np
import math as mt
import scipy.sparse as sp


##############################################################################################
#                      Global Variable
##############################################################################################
def CreateMesh(V,F,NPs):
    E=np.zeros(np.shape(V))
    F = [ [int(i) for i in thing] for thing in F]
    for i in range(NPs):
        E[:,3*i]=V[:,3*i]
        E[:,3*i+1]=-V[:,3*i+2]
        E[:,3*i+2]=V[:,3*i+1]
        me = bpy.data.meshes.new('MyMesh')
        ob = bpy.data.objects.new('VG'+str(i), me)
        bpy.context.collection.objects.link(ob)
        me.from_pydata(E[:,3*i:3*i+3], [], F)
        me.update()
        
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

def WriteQuanMaric(Name,QuantValue,QuantName):
    Ln=len(QuantName)
    with open(Name, 'w') as fl:
        for i in range(Ln):
            fl.write(" %s" % QuantName[i])
            fl.write(" %f" % QuantValue[i])
            fl.write("\n")

def ReadStringList(Name):
    Lst=[]
    fl = open(Name, 'r')
    NumLine=0
    for line in fl:
        words = line.split()
        Lst=Lst+words
    return Lst




def LoadRigWeight(self, context):
    global ActiveModel,FilePath,path,Ncl,Phi,Ainc,VecCls
    if len(RigList)!=0:
        ActiveModel=RigList[context.scene.ModelNo] #
        path=FilePath+ActiveModel+'/' #
        print('Loading Model '+ActiveModel)
        VecCls=ReadTxt(path+ActiveModel+'_VecCls.txt')
        Ncl=len(VecCls)
        Phi=np.loadtxt(path+ActiveModel+'_SegPhi.txt',delimiter=',')
        Ainc=sp.load_npz(path+ActiveModel+'_IncidentMtrx.npz').tocsc()
    return


path=''
Ncl=0
Phi=0
Ainc=0
VecCls=[]
ActiveModel=''
if os.path.isfile(FilePath+'Riglist.txt'):
    RigList=ReadStringList(FilePath+'Riglist.txt')
else:
    RigList=[]
##############################################################################################
#                                   Tools
##############################################################################################
    
class ToolsPanel(bpy.types.Panel):
    bl_label = "Error calculation Tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
 
    def draw(self, context):
        
        self.layout.prop(context.scene,"ModelNo")
        self.layout.label(text="Error")
        self.layout.operator("get.mesh",text='Mesh Seq').seqType="Mesh"
        self.layout.operator("err.cmpt",text='Error').seqType="error"
        self.layout.operator("clr.err",text='ColorOriantationErr').seqType="oriantation"
        self.layout.operator("clr.err",text='ColorAreaErr').seqType="area"
        

class GetMeshSeq(bpy.types.Operator):
    bl_idname = "get.mesh"
    bl_label = "Load Skeleton"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global path, ActiveModel,Ainc,Phi,Ncl,VecCls
        F=ReadTxt(path+ActiveModel+'_facz.txt')
        
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object
        Sref=np.loadtxt(path+ActiveModel+'_Sr.txt',delimiter=',')
        Aref=np.loadtxt(path+ActiveModel+'_Ar.txt',delimiter=',')
        
        NV=len(obj.data.vertices)
        NPs=len(Selected_Meshes)
        
        Vrt=np.zeros((NV,3))
        RefVrtErr=np.zeros((3*NPs,NV))
        DefVrtErr=np.zeros((3*NPs,NV))
        Dptch=np.zeros([3,3*Ncl])
        for i in range(NPs):
            bpy.context.view_layer.objects.active = Selected_Meshes[-i-1]
            obj = bpy.context.active_object
             
            t=0
            for v in obj.data.vertices:
                co_final= obj.matrix_world @ v.co
                Vrt[t]=np.array([co_final.x,co_final.z,-co_final.y])
                t+=1
            RefVrtErr[3*i:3*i+3]=Vrt.T
        
            Vec=((Ainc.transpose()).dot(Vrt)).T
            
            for ci in range(Ncl):
                Vc=Vec[:,VecCls[ci]]
                Scl=np.linalg.norm(Vc,axis=0)
                Vc=Vc/Scl
            
                U,Lmda,V=np.linalg.svd((Sref[:,VecCls[ci]]).dot(Vc.T))
                tmp=(V.T).dot(U.T)
                Dptch[:,3*ci:3*ci+3]=(np.mean(Scl/Aref[VecCls[ci]]))*tmp
            DefVrtErr[3*i:3*i+3]=Dptch.dot(Phi)
        
        np.savetxt(path+'RefVrtErr.txt',RefVrtErr,delimiter=',')
        np.savetxt(path+'DefVrtErr.txt',DefVrtErr,delimiter=',')
        
        
        return{'FINISHED'}



class ErrCompute(bpy.types.Operator):
    bl_idname = "err.cmpt"
    bl_label = "Compute Animation"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global path,ActiveModel,Ainc
        
        RefVrtErr=np.loadtxt(path+'RefVrtErr.txt',delimiter=',')
        EstVrtErr=np.loadtxt(path+'DefVrtErr.txt',delimiter=',')
        F=ReadTxt(path+ActiveModel+'_facz.txt')

        NF=len(F)
        NPs=len(RefVrtErr)//3
        Err=np.zeros((NPs,NF))
        SclErr=np.zeros((NPs,NF))
        for p in range(NPs):
            for i in range(len(F)):
                f=F[i]
                RVrt=RefVrtErr[3*p:3*p+3,f].T-np.mean(RefVrtErr[3*p:3*p+3,f],axis=1)
                #Rscl=np.mean(np.linalg.norm(RVrt,axis=1))
                N=np.cross(RVrt[0],RVrt[1])
                if np.linalg.norm(N)==0:
                    N=np.cross(RVrt[0],RVrt[2])
                Nr=N/np.linalg.norm(N)

                EVrt=EstVrtErr[3*p:3*p+3,f].T-np.mean(EstVrtErr[3*p:3*p+3,f],axis=1)
                #Escl=np.mean(np.linalg.norm(EVrt,axis=1))
                N=np.cross(EVrt[0],EVrt[1])
                if np.linalg.norm(N)==0:
                    N=np.cross(EVrt[0],EVrt[2])
                Ne=N/np.linalg.norm(N)

                RAr=0
                EAr=0
                X=list(range(len(f)))
                Y=list(range(1,len(f)))+[0]
                for j in range(len(f)):
                    RAr+=0.5*np.linalg.norm(np.cross(RVrt[X[j]],RVrt[Y[j]]))
                    EAr+=0.5*np.linalg.norm(np.cross(EVrt[X[j]],EVrt[Y[j]]))

                Err[p,i]=1-abs(Nr.dot(Ne))#np.linalg.norm(np.cross(Nr,Ne))#
                SclErr[p,i]=abs(1-(EAr/RAr))#abs(1-(Rscl/Escl))
        #SclErr=(SclErr-np.min(SclErr,axis=0))/np.max(SclErr,axis=0)

        Nmean=np.mean(Err,axis=0)
        Smean=np.mean(SclErr,axis=0)
        Nstd=np.std(Err,axis=0)
        Sstd=np.std(SclErr,axis=0)
       
        QuantValue=[np.mean(Nmean),np.std(Err),np.max(Nmean),np.max(Nstd),np.min(Nmean),np.min(Nstd),
                    np.argmax(Nmean),np.argmax(Nstd),np.argmin(Nmean),np.argmin(Nstd),
                    np.mean(Smean),np.std(SclErr),np.max(Smean),np.max(Sstd),np.min(Smean),np.min(Sstd),
                    np.argmax(Smean),np.argmax(Sstd),np.argmin(Smean),np.argmin(Sstd)]
        QuantName=['Orientation Mean','Orientation Standard deviation', 'Orientation Max Mean', 'Orientation Max Std',
                   'Orientation min Mean', 'Orientation min std', 'Orientation Max Mean Id',
                   'Orientation Max Std Id','Orientation Min Mean Id','Orientation Min std Id',
                   'Area Mean','Area Standard deviation', 'Area Max Mean', 'Area Max Std',
                   'Area min Mean', 'Area min std', 'Area Max Mean Id',
                   'Area Max Std Id','Area Min Mean Id','Area Min std Id']
        WriteQuanMaric(path+ActiveModel+'_OriAreaErr_QuantitativeValues.txt',QuantValue,QuantName)

        np.savetxt(path+ActiveModel+'_NormalErr.txt',Nmean,delimiter=',')
        np.savetxt(path+ActiveModel+'_AreaErr.txt',Smean,delimiter=',')
          
        return{'FINISHED'}

    
class ColorERR(bpy.types.Operator):
    bl_idname = "clr.err"
    bl_label = "Color Segmentations"
    seqType:bpy.props.StringProperty()
 
    def execute(self, context):
        global path,ActiveModel
        
        if self.seqType=='oriantation':
            MxErr=np.loadtxt(path+ActiveModel+'_NormalErr.txt',delimiter=',')
        else:
            MxErr=np.loadtxt(path+ActiveModel+'_AreaErr.txt',delimiter=',')

        mu=np.mean(MxErr)
        sd=np.std(MxErr)
        numSeg=8
        Ll=[0.0]+[(max(MxErr)-1)/i for i in range(numSeg,0,-1)]
        Ul=Ll[1:]+[max(MxErr)]
        Cls=[]
        for i in range(len(Ul)):
            tmp=[]
            for m in range(len(MxErr)):
                if ((MxErr[m]>=Ll[i]) and (MxErr[m]<=Ul[i])):
                    tmp.append(m)
            Cls.append(tmp)
        obj = bpy.context.active_object
        mesh=bpy.context.object.data
        numMtrl=len(Cls)
        t=0
        for c in Cls:
            Mtrl=bpy.data.materials.new('Material'+str(t))
            Mtrl.diffuse_color=[t/(numMtrl-1),1-(t/(numMtrl-1)),0.0,1]
            mesh.materials.append(Mtrl)
            for j in c:
                obj.data.polygons[j].material_index=t
            t+=1
        
        return{'FINISHED'}



def register():
    bpy.utils.register_class(ToolsPanel)
    bpy.types.Scene.ModelNo=bpy.props.IntProperty(name="Model", description="Select Rigged Model", default=0,
                                                min=0,max=(len(RigList)-1),options={'ANIMATABLE'}, update=LoadRigWeight)
    
    
    bpy.utils.register_class(GetMeshSeq)
    bpy.utils.register_class(ErrCompute)
    bpy.utils.register_class(ColorERR)
   
    
def unregister():
    bpy.utils.unregister_class(ToolsPanel)
    
    bpy.utils.unregister_class(GetMeshSeq)
    bpy.utils.unregister_class(ErrCompute)
    bpy.utils.unregister_class(ColorERR)
    del bpy.types.Scene.ModelNo
    
    
    
if __name__ == "__main__":  
    bpy.utils.register_module(__name__) 


