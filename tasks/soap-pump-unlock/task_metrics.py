"""Geometric end-state scoring of captive head rise, release and stillness."""
import math
from itertools import product
import numpy as np

SIGNS=np.asarray(list(product([-1.,1.],repeat=3)))

def rz(a):
    c,s=math.cos(a),math.sin(a)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])

def boxpoints(x):
    assert x[3]==(0.,0.), 'Rotated boxes need an explicit metric transform.'
    return np.asarray(x[0])+SIGNS*np.asarray(x[1])/2

def cylpoints(x):
    c,r,h,_,axis=x
    assert axis=='Z'
    return np.array([np.asarray(c)+[r*math.cos(a),r*math.sin(a),z*h/2]
                     for a in np.linspace(0,2*math.pi,32,endpoint=False) for z in [-1,1]])

def points(sp):
    return np.concatenate([boxpoints(x) for x in sp.boxes]+[cylpoints(x) for x in sp.cylinders]+[m.points for m in sp.meshes])

def boxdist(p,x):
    return float(np.linalg.norm(np.maximum(np.abs(p-np.asarray(x[0]))-np.asarray(x[1])/2,0.)))

def cyldist(p,x):
    c,r,h,_,axis=x
    assert axis=='Z'
    q=p-np.asarray(c)
    return math.hypot(max(math.hypot(q[0],q[1])-r,0.),max(abs(q[2])-h/2,0.))

def polydist(q,p):
    e=np.roll(p,-1,axis=0)-p;v=q-p
    if np.min(e[:,0]*v[:,1]-e[:,1]*v[:,0])>=-1e-12:return 0.
    u=np.clip(np.sum(v*e,axis=1)/np.sum(e*e,axis=1),0,1)
    return float(np.min(np.linalg.norm(v-u[:,None]*e,axis=1)))

class Metric:
    def __init__(self,ns):
        self.lk=ns['labkit'];self.top=ns['TOP'];self.sp=ns['BOTTLE']
        self.lift=next(x for x in self.sp.joints if x.name=='lift')
        self.head=next(x for x in self.sp.joints if x.name=='head')
        self.rootpts=points(self.sp);self.liftpts=points(self.lift);self.headpts=points(self.head)
        self.walls=[]
        for m in self.sp.meshes:
            p=np.unique(m.points[:,:2],axis=0);c=p.mean(0)
            p=p[np.argsort(np.arctan2(p[:,1]-c[1],p[:,0]-c[0]))]
            self.walls.append((p,float(m.points[:,2].min()),float(m.points[:,2].max())))
        self.reset()

    def reset(self):self.hist=[];self.value=0.;self.detail={}

    def distance(self,p,d):
        q=d.rotation().T@(np.asarray(p)-np.asarray(d.pos))
        ds=[boxdist(q,x) for x in self.sp.boxes]+[cyldist(q,x) for x in self.sp.cylinders]
        ds += [math.hypot(polydist(q[:2],p),max(lo-q[2],q[2]-hi,0.)) for p,lo,hi in self.walls]
        ql=q-np.array([0.,0.,float(d.joints['lift'])])
        ds += [cyldist(ql,x) for x in self.lift.cylinders]
        qh=rz(-float(d.joints['head']))@ql
        ds += [boxdist(qh,x) for x in self.head.boxes]+[cyldist(qh,x) for x in self.head.cylinders]
        return min(ds)

    def update(self,env,d):
        rise=float(d.joints['lift']);angle=float(d.joints['head']);shift=np.array([0.,0.,rise])
        local=np.concatenate([self.rootpts,self.liftpts+shift,self.headpts@rz(angle).T+shift])
        world=local@d.rotation().T+np.asarray(d.pos)
        self.hist=(self.hist+[world])[-21:]
        still=False
        if len(self.hist)==21:
            h=np.stack(self.hist);ranges=np.ptp(h,axis=0);bound=np.linalg.norm(ranges,axis=-1)
            if float(bound.max())<.001:still=True
            elif float(ranges.max())<.001:
                bad=np.flatnonzero(bound>=.001)
                disp=h[:,None,bad,:]-h[None,:,bad,:]
                still=float(np.max(np.linalg.norm(disp,axis=-1)))<.001
        from yam_arena import runtime
        base=env.unwrapped;origin=self.lk._np(base.scene.env_origins[0]);tips=[]
        for name in ['left_robot','right_robot']:
            arm=base.scene[name];mid=runtime.pad_tips_world(arm)-origin
            g=runtime.link_poses(arm)['gripper'];ids=arm.find_joints(['joint7','joint8'],preserve_order=True)[0]
            j7,j8=np.abs(self.lk._np(arm.data.joint_pos)[0,ids])
            tips.extend([mid,mid-j7*g[:3,0],mid+j8*g[:3,0]])
        clear=min(self.distance(p,d) for p in tips)
        root=self.rootpts@d.rotation().T+np.asarray(d.pos)
        supported=abs(float(root[:,2].min())-self.top)<=.003
        upright=float(d.rotation()[2,2])>=math.cos(math.radians(5))
        valid=still and clear>=.080 and supported and upright and -.0005<=rise<=.0185
        self.value=(1. if rise>=.015 else .5 if rise>=.012 else 0.) if valid else 0.
        self.detail=dict(rise_mm=rise*1000,relative_turn_deg=math.degrees(angle),hands_mm=clear*1000,
                         still=bool(still),supported=bool(supported),upright=bool(upright))

    def score(self):return float(self.value)
