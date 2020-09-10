
bl_info = {
    "name": "Interpolation using Manifold Training",
    "author": "Prashant Domadiya",
    "version": (1, 1),
    "blender": (2, 81, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Rigg Model for Deep Learning",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Development"}

LibPath='/home/prashant/anaconda3/envs/Blender269/lib/python3.4/site-packages/'
FilePath='/media/prashant/DATA/MyCodes/OurSegSkinning/'

import os
from os.path import join
import sys
sys.path.append(LibPath)

import bpy
import bmesh as bm
import numpy as np
from scipy import sparse as sp
from sksparse import cholmod as chmd
import scipy.linalg as scla
from sklearn.cluster import KMeans
import time
import itertools as it
import shutil


def CreateMesh(V,F,NPs):
    E=np.zeros(np.shape(V))
    F = [ [int(i) for i in thing] for thing in F]
    for i in range(NPs):
        
        E[:,3*i]=V[:,3*i]
        E[:,3*i+1]=-V[:,3*i+2]
        E[:,3*i+2]=V[:,3*i+1]
       
        
        me = bpy.data.meshes.new('MyMesh')
        ob = bpy.data.objects.new('mymeshes'+sty(i), me)
        #scn = bpy.context.scene
        #scn.objects.link(ob)
        bpy.context.collection.objects.link(ob)
        
        me.from_pydata(E[:,3*i:3*i+3], [], F)
        me.update()

def ConnectionMatrices(fcs,NV,NVec,CnsVrt,FrVrt):
    
    NCvrt=len(CnsVrt)
    AB=sp.lil_matrix((NV-NCvrt,NVec),dtype=float)
    ABc=sp.lil_matrix((NCvrt,NVec),dtype=float)
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

def ComputeWeights(wghts,Vecs):
    Beta=2*0.1
    W=Beta*(wghts.T).dot(wghts)
    b=Beta*(wghts.T).dot(Vecs)
    Nw=len(wghts.T)
    Wt=np.array([0.5]*Nw)
    err=10000
    maxitr=100
    k=0
    while (err>1 and k<maxitr): 
        Wt=Wt-W.dot(Wt)+b
        for i in range(Nw):
            if Wt[i]<0:
                Wt[i]=0.0
        err=np.linalg.norm(Vecs-wghts.dot(Wt))
        k+=1
    return Wt

def WritteOBJ(vert,fc,filepath):
    filepath=filepath+'.obj'
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in vert:
            f.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
        for p in fc:
            f.write("f")
            for i in p:
                f.write(" %d" % (i + 1))
            f.write("\n")
def ReadOBJList(fname):
    file = open(fname, 'r')
    
    vertices=np.array([])
    faces = []
    for line in file:
        words = line.split()
        
        if len(words) == 0 or words[0].startswith('#'):
            pass
        elif words[0] == 'v':
            v=np.array([np.float32(words[1]), np.float32(words[2]), np.float32(words[3])])
            vertices =np.append(vertices,v)
        elif words[0] == 'f':
            l=len(words)
            tmp=[]
            for i in range(1,l):
                tmp.append(int(words[i])-1)
            faces.append(tmp)
    
    file.close()
    vertices=np.reshape(vertices,(-1,3))
    return vertices, faces

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

def WriteStringList(Name, Lst):
    with open(Name, 'w') as fl:
        for i in Lst:
            fl.write(" %s" % i)
            fl.write("\n")

def ReadStringList(Name):
    Lst=[]
    fl = open(Name, 'r')
    NumLine=0
    for line in fl:
        words = line.split()
        Lst=Lst+words
    return Lst


def GetNeighbours(G):
    Nbr=[[]]*len(G)
    for i in range(len(G)):
        g=G[i]
        for j in range(i+1,len(G)):
            gn=G[j]
            if len(set(g).intersection(set(gn)))!=0:
                Nbr[i]=Nbr[i]+[j]
                Nbr[j]=Nbr[j]+[i]
    return Nbr

def FindProxyBones(v,G):
    Pb=[]
    for i in range(len(G)):
        if v in G[i]:
            Pb.append(i)
    return Pb

def getIndexPositions(listOfElements,FcsIdx, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    indexPosList = []
    HalfindexPosList = []
    indexPos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            indexPos = listOfElements.index(element, indexPos)
            # Add the index position in list
            indexPosList.append(indexPos)
            if indexPos in FcsIdx:
                HalfindexPosList.append(indexPos)
                
            indexPos += 1
        except ValueError as e:
            break
    return indexPosList,HalfindexPosList
##############################################################################################
#                      Global Variable
##############################################################################################
def LoadRigWeight(self, context):
    global FilePath,ActiveModel,path
    if len(RigList)!=0:
        ActiveModel=RigList[context.scene.ModelVG]
        print('Active Model is....',ActiveModel)
        path=FilePath+ActiveModel+'/'
        
FilePath='/home/student/Documents/Skinning/OurSegSkinning/'
LBSFilePath='/home/student/Documents/Skinning/LBS/'

if os.path.isfile(FilePath+'Riglist.txt'):
    RigList=ReadStringList(FilePath+'Riglist.txt')
else:
    RigList=[]

if len(RigList)!=0:
    ActiveModel=RigList[0]
    print('Active Model is....',ActiveModel)
    path=FilePath+ActiveModel+'/'
else:
    print('Please Rigg the model first.....')

    
##############################################################################################
#                                   Tools
##############################################################################################
    
class TOOlS_PT_panel(bpy.types.Panel):
    bl_label = "Animation Tool"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
 
    def draw(self, context):
        self.layout.prop(context.scene,"RigName")
        self.layout.label(text="Segment the mesh")
        self.layout.operator("get.seq",text='Mesh Seq').seqType="mesh"
        self.layout.prop(context.scene,"NumOfCls")
        self.layout.operator("get.seg",text='Segment').seqType="segment"
        self.layout.operator("clr.mesh",text='Color').seqType="ColorMesh"
        self.layout.label(text="Training Tools")
        self.layout.operator("rigg.model",text='Train').seqType="rigg"

class GetMeshSeq(bpy.types.Operator):
    bl_idname = "get.seq"
    bl_label = "Target Sequence"
    seqType: bpy.props.StringProperty()
 
    def execute(self, context):
        global FilePath,path,ActiveModel,RigList
        ActiveModel=context.scene.RigName
        if os.path.exists(FilePath+ActiveModel)==False:
            os.mkdir(FilePath+ActiveModel)

        if (ActiveModel in RigList)==False:
            RigList.append(ActiveModel)
        WriteStringList(FilePath+'Riglist.txt',RigList)
        path=FilePath+ActiveModel+'/'
        
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object
        F=[]
        for f in obj.data.polygons:
            F.append(list(f.vertices))

        WriteAsTxt(path+ActiveModel+'_facz.txt',F)
        FcList=ComuteReducedFaceList(F)
        
        V=np.zeros([3*len(Selected_Meshes),len(obj.data.vertices)])
        F=[]
        
        NPs=len(Selected_Meshes)
        for i in range(len(Selected_Meshes)):
            bpy.context.view_layer.objects.active = Selected_Meshes[i]
            obj = bpy.context.active_object
             
            t=0
            for v in obj.data.vertices:
                co_final= obj.matrix_world @ v.co
                V[3*i:3*i+3,t]=np.array([co_final.x,co_final.z,-co_final.y])
                t+=1
            k=0
            for f in obj.data.polygons:
                if k in FcList:
                    if i==0:
                        F.append(list(f.vertices))
                k+=1
              
        np.savetxt(path+ActiveModel+'_vertz.txt',V,delimiter=',')
        WriteAsTxt(path+ActiveModel+'_Halffacz.txt',F)
        WriteAsTxt(path+ActiveModel+'_HalffaczIdx.txt',FcList)
        return{'FINISHED'}

    
class GetSegments(bpy.types.Operator):
    bl_idname = "get.seg"
    bl_label = "Segment Meshes"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global path,ActiveModel
        Vrt=np.loadtxt(path+ActiveModel+'_vertz.txt',delimiter=',')
        Fful=ReadTxt(path+ActiveModel+'_facz.txt')
        Fcs=ReadTxt(path+ActiveModel+'_Halffacz.txt')
        FcsIdx=ReadTxt(path+ActiveModel+'_HalffaczIdx.txt')[0]
        NF=len(Fful)
        NPs,NV=np.shape(Vrt)
        NPs=NPs//3
        RM=np.zeros((NF,9*(NPs-1)))
        strttime=time.time()
        t=0
        print('Computing Rotations....')
        for f in Fful:
            for ps in range(NPs):
                if ps==0:
                    RefFrm=Vrt[3*ps:3*ps+3,f].T-np.mean(Vrt[3*ps:3*ps+3,f].T,axis=0)
                    nrm=np.cross(RefFrm[0],RefFrm[1])/np.linalg.norm(np.cross(RefFrm[0],RefFrm[1]))
                    RefFrm=np.concatenate((RefFrm,np.reshape(nrm,(1,3))),axis=0)
                else:
                    DefFrm=Vrt[3*ps:3*ps+3,f].T-np.mean(Vrt[3*ps:3*ps+3,f].T,axis=0)
                    nrm=np.cross(DefFrm[0],DefFrm[1])/np.linalg.norm(np.cross(DefFrm[0],DefFrm[1]))
                    DefFrm=np.concatenate((DefFrm,np.reshape(nrm,(1,3))),axis=0)
                    
                    Q=np.dot(DefFrm.T,np.linalg.pinv(RefFrm.T))
                    
                    R,S=scla.polar(Q)
                    RM[t,9*(ps-1):9*(ps-1)+9]=np.ravel(R)    
            t+=1
        
        
        Ncl=context.scene.NumOfCls
        print('Classifying....',Ncl,'...classes')
        clustering = KMeans(n_clusters=Ncl).fit(RM)
        Y=list(clustering.labels_)
        
        Cls=[[]]*Ncl
        HalfCls=[[]]*Ncl
        t=0
        for i in list(Y):            
            Cls[i]=Cls[i]+[t]
            if t in FcsIdx:#Fful[t] in Fcs:#
                HalfCls[i]=HalfCls[i]+[t]
            t+=1
        
        print("Segmentation time ...", time.time()-strttime)
        WriteAsTxt(path+ActiveModel+"_ClusterKmeans.txt",Cls)
        WriteAsTxt(path+ActiveModel+"_HalfClusterKmeans.txt",HalfCls)
        return{'FINISHED'}

class ColorMesh(bpy.types.Operator):
    bl_idname = "clr.mesh"
    bl_label = "Color Segmentations"
    seqType: bpy.props.StringProperty()
 
    def execute(self, context):
        global path,ActiveModel
        
        Cls=ReadTxt(path+ActiveModel+'_ClusterKmeans.txt')
        
        Selected_Meshes=bpy.context.selected_objects
        bpy.context.view_layer.objects.active = Selected_Meshes[0]
        obj = bpy.context.active_object
        mesh=bpy.context.object.data
        t=0
        for c in Cls:
            Mtrl=bpy.data.materials.new('Material'+str(t))
            Mtrl.diffuse_color=np.random.uniform(0,1,4).tolist()
            mesh.materials.append(Mtrl)
            for j in c:
                obj.data.polygons[j].material_index=t
            t+=1
        
        return{'FINISHED'}

class Rigg(bpy.types.Operator):
    bl_idname = "rigg.model"
    bl_label = "Rigg Model"
    seqType: bpy.props.StringProperty()
 
    def execute(self, context):
        global path,ActiveModel,LBSpath
        #############################################################################################################################
        Vrt=np.loadtxt(path+ActiveModel+'_vertz.txt',delimiter=',')
        Fcs=ReadTxt(path+ActiveModel+'_Halffacz.txt')

        NPs,NV=np.shape(Vrt)
        NPs=NPs//3
        NVec=sum([len(f) for f in Fcs])
        
            

        CnsVrt=[0]
        print("Number of LBS Vrt=",len(CnsVrt))
        FrVrt=[]
        for i in range(NV):
            if i not in CnsVrt: 
                FrVrt.append(i)
        WriteAsTxt(path+ActiveModel+'_FrVrt.txt',FrVrt)



        strttime=time.time()
        print("Computing Incident Matrices.....")
        AB,ABc=ConnectionMatrices(Fcs,NV,NVec,CnsVrt,FrVrt)
        np.savetxt(path+ActiveModel+'_RefMesh.txt',Vrt[0:3,:].T,delimiter=',')
        factor=chmd.cholesky_AAt(AB.tocsc())
        Vec=((sp.vstack([ABc,AB]).transpose()).dot(Vrt.T)).T
        
        sp.save_npz(path+ActiveModel+'_IncidentMtrx.npz', sp.vstack([ABc,AB]))
        ############################################################################################################################
        RM=np.zeros((3*(NPs-1),3*NVec)) 
        AM=np.zeros(((NPs-1),NVec))
        RS=np.zeros((len(Fcs),9*(NPs-1)))
        
        h=0
        fn=0
        
        for f in Fcs:
            MrFrm=np.zeros((3,3*len(f)))
            MdFrm=np.zeros((3,3*len(f)))
            Sclr=np.zeros(len(f))
            
            for ps in range(NPs):
                Vc=Vec[3*ps:3*ps+3,h:h+len(f)]
                N=np.cross(Vc[:,0],Vc[:,1])/np.linalg.norm(np.cross(Vc[:,0],Vc[:,1]))
                B=np.cross(N,Vc,axis=0)
                B=B/np.linalg.norm(B,axis=0)
                Nr=np.cross(Vc,B,axis=0)
                Nr=Nr/np.linalg.norm(Nr,axis=0)
                for t in range(len(f)):
                    if ps==0:
                        Sclr[t]=np.linalg.norm(Vc[:,t])
                        MrFrm[:,3*t]=Vc[:,t]/Sclr[t]
                        MrFrm[:,3*t+1]=B[:,t]
                        MrFrm[:,3*t+2]=Nr[:,t]
                        
                    else:
                        Sc=np.linalg.norm(Vc[:,t])
                        MdFrm[:,3*t]=Vc[:,t]/Sc
                        MdFrm[:,3*t+1]=B[:,t]
                        MdFrm[:,3*t+2]=Nr[:,t]
                        tmp=MdFrm[:,3*t:3*t+3].dot(MrFrm[:,3*t:3*t+3].T)
                        RM[3*(ps-1):3*(ps-1)+3,3*h+3*t:3*h+3*t+3]=tmp
                        
                        AM[ps-1,h+t:h+t+1]=Sc/Sclr[t]
                
                if ps!=0:
                    Q=np.dot(MdFrm,np.linalg.pinv(MrFrm))
                    Rf,sf=scla.polar(Q)
                    RS[fn,9*(ps-1):9*(ps-1)+9]=np.ravel(Rf)
                
            fn+=1
            h+=len(f)
        Ncl=context.scene.NumOfCls
        print('Classifying....',Ncl,'...classes')
        clustering = KMeans(n_clusters=Ncl).fit(RS)
        Y=list(clustering.labels_)
        
        del RS
        
        HalfCls=[[]]*Ncl
        VecCls=[[]]*Ncl
        t=0
        h=0
        for i in list(Y):            
            HalfCls[i]=HalfCls[i]+[t]
            VecCls[i]=VecCls[i]+list(range(h,h+len(Fcs[t])))
            h+=len(Fcs[t])
            t+=1
        
        WriteAsTxt(path+ActiveModel+'_VecCls.txt',VecCls)
       
        ################################################################################################################################
        print('Computing segment Deformation')
        RSeg=np.zeros([3*(NPs-1),3*Ncl])
        
        ASeg=np.zeros([Ncl,(NPs-1)])
        Sref=np.empty((3,1))
        Aref=np.empty(1)
        for ci in range(Ncl):
            for p in range(NPs):
                Vc=Vec[3*p:3*p+3,VecCls[ci]]
                Scl=np.linalg.norm(Vc,axis=0)
                Vc=Vc/Scl
                if p==0:
                    Sr=1*Vc
                    Ar=1*Scl
                    Sref=np.append(Sref,Sr,axis=1)
                    Aref=np.append(Aref,Ar)
                else:
                    U,Lmda,V=np.linalg.svd((Sr).dot(Vc.T))
                    tmp=(V.T).dot(U.T)
                    RSeg[3*(p-1):3*(p-1)+3,3*ci:3*ci+3]=tmp
                    ASeg[ci,p-1]=np.mean(Scl/Ar)
                    
        np.savetxt(path+ActiveModel+'_Ar.txt',np.linalg.norm(Vec[0:3],axis=0),delimiter=',')        
        np.savetxt(path+ActiveModel+'_Sr.txt',Vec[0:3]/np.linalg.norm(Vec[0:3],axis=0),delimiter=',')
        
        ########################## Comute Mesh Frames ########################################### 
        
        
        
        print("Computing Mesh Frames.....")
        
        W=sp.lil_matrix((3*Ncl,NVec))
        MpsRot=np.zeros((3,3*NVec))
        MpsScl=np.zeros(NVec)
        

        h=0
        fn=0
        for f in Fcs:
            for t in range(len(f)):
                U,Lmda,V=np.linalg.svd((RM[:,3*h+3*t:3*h+3*t+3].T).dot(RSeg[:,3*Y[fn]:3*Y[fn]+3]))
                tmp=(V.T).dot(U.T)
                MpsRot[:,3*(h+t):3*(h+t)+3]=tmp
                Scl=np.mean(AM[:,t]/ASeg[Y[fn]])
                MpsScl[(h+t):(h+t)+1]=Scl
                W[3*Y[fn]:3*Y[fn]+3,h+t]=Scl*np.reshape((tmp).dot(Vec[0:3,h+t:h+t+1]),(3,1))
                
            h+=len(f)
            fn+=1
        np.savetxt(path+ActiveModel+'_MpsRot.txt',MpsRot,delimiter=',')
        np.savetxt(path+ActiveModel+'_MpsScl.txt',MpsScl,delimiter=',')
        WriteAsTxt(path+ActiveModel+'_MpsIdx.txt',Y)
        
        Phi=((factor((AB.dot(W.transpose())).tocsc())).transpose()).toarray()
        
        del W,factor,AB,ABc
        
        PhiSeg=np.zeros((3*Ncl,NV))
        for i in range(NV):
            if i not in CnsVrt:
                PhiSeg[:,i]=Phi[:,FrVrt.index(i)]
        print("Enveloping time..",time.time()-strttime)
        np.savetxt(path+ActiveModel+'_SegPhi.txt',PhiSeg,delimiter=',')
              
        return{'FINISHED'}


def register():
    bpy.utils.register_class(TOOlS_PT_panel)
    bpy.types.Scene.NumOfCls=bpy.props.IntProperty(name="classes", description="Set Number of segments", default=1,
                                                  min=1,options={'ANIMATABLE'})
    bpy.utils.register_class(GetMeshSeq)
    bpy.utils.register_class(ImportMeshes)
    bpy.utils.register_class(GetSegments)
    bpy.utils.register_class(ColorMesh)
    bpy.utils.register_class(Rigg)
    bpy.types.Scene.RigName=bpy.props.StringProperty(name="RigName", description="", default="Default")
    
def unregister():
    bpy.utils.unregister_class(TOOlS_PT_panel)
    bpy.utils.unregister_class(GetMeshSeq)
    bpy.utils.unregister_class(ImportMeshes)
    bpy.utils.unregister_class(GetSegments)
    bpy.utils.unregister_class(ColorMesh)
    bpy.utils.unregister_class(Rigg)
    del bpy.types.Scene.RigName
    del bpy.types.Scene.NumOfCls
    
if __name__ == "__main__":  
    bpy.utils.register_module(__name__) 


