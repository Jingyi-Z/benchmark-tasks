"""Geometric end-state scoring, independent of sleeve opening angle or control phases."""
import math
import numpy as np
from itertools import product
SIGNS=np.asarray(list(product([-1.,1.],repeat=3)))
def boxpoints(spec):
 return np.asarray(spec[0])+SIGNS*np.asarray(spec[1])/2

def cylpoints(c,r,h,axis):
 k='XYZ'.index(axis);a=[j for j in range(3) if j!=k];out=[]
 for theta in np.linspace(0,2*math.pi,32,endpoint=False):
  for sign in [-1,1]:
   q=np.asarray(c,float).copy();q[k]+=h/2*sign;q[a[0]]+=r*math.cos(theta);q[a[1]]+=r*math.sin(theta);out.append(q)
 return np.asarray(out)
def boxdist(q,c,size):return float(np.linalg.norm(np.maximum(np.abs(q-np.asarray(c))-np.asarray(size)/2,0.)))
def cyldist(q,c,r,h,axis):
 p=q-np.asarray(c);k='XYZ'.index(axis);rad=math.sqrt(sum(p[j]**2 for j in range(3) if j!=k));return math.hypot(max(rad-r,0.),max(abs(p[k])-h/2,0.))
def polydist(q,poly):
 e=np.roll(poly,-1,axis=0)-poly;v=q-poly;cross=e[:,0]*v[:,1]-e[:,1]*v[:,0]
 if np.min(cross)>=-1e-12:return 0.
 u=np.clip(np.sum(v*e,axis=1)/np.sum(e*e,axis=1),0,1);return float(np.min(np.linalg.norm(v-u[:,None]*e,axis=1)))
def rotation_y(a):
 return np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
def extent(axis):return .0465*np.abs(axis)+.003*np.sqrt(np.maximum(0.,1-axis**2))
class Metric:
 def __init__(self,ns):
  self.ns=ns;self.lk=ns['labkit'];self.top=ns['TOP'];self.spec=ns['DRILL'];self.trspec=ns['TRAY']
  self.root=np.concatenate([boxpoints(x) for x in self.spec.boxes]+[cylpoints(c,r,h,axis) for c,r,h,_,axis in self.spec.cylinders])
  self.tray=np.concatenate([boxpoints(x) for x in self.trspec.boxes]);self.bit=cylpoints((0,0,0),.003,.093,'Y')
  self.sleeve=np.concatenate([m.points for m in ns['SLEEVE'].meshes]);self.polys=[]
  for j in range(16):
   a,b=2*math.pi*j/16,2*math.pi*(j+1)/16;self.polys.append(np.array([(r*math.cos(t),r*math.sin(t)) for r,t in [(.012,a),(.021,a),(.021,b),(.012,b)]]))
  self.reset()
 def reset(self):self.hist=[];self.value=0.;self.detail={}
 def shapes(self,d):
  pts=[self.root];a=np.array([0.,-.049,0.]);pts.append((self.sleeve-a)@rotation_y(float(d.joints['sleeve'])).T+a)
  for j in range(3):
   phi=j*2*math.pi/3;n=np.array([math.cos(phi),0,math.sin(phi)]);v=n*math.sin(self.ns['ALPHA'])+[0,math.cos(self.ns['ALPHA']),0]
   pts.append(self.ns['JAWS'][j].meshes[0].points+v*float(d.joints['jaw'+str(j)]))
  return np.concatenate(pts)
 def distance(self,p,d,b,tr):
  q=d.rotation().T@(p-np.asarray(d.pos));dist=min([boxdist(q,s[0],s[1]) for s in self.spec.boxes]+[cyldist(q,c,r,h,a) for c,r,h,_,a in self.spec.cylinders])
  qs=rotation_y(-float(d.joints['sleeve']))@q;dy=max(abs(qs[1]+.049)-.020,0.)
  dist=min(dist,min(math.hypot(polydist(qs[[0,2]],poly),dy) for poly in self.polys))
  for j in range(3):
   phi=j*2*math.pi/3;n=np.array([math.cos(phi),0,math.sin(phi)]);t=np.array([-math.sin(phi),0,math.cos(phi)]);v=n*math.sin(self.ns['ALPHA'])+[0,math.cos(self.ns['ALPHA']),0]
   rel=q-n*.00548-np.array([0.,-.056,0.])-v*float(d.joints['jaw'+str(j)])
   dist=min(dist,boxdist(np.array([n@rel,rel[1],t@rel]),(0,0,0),(.005,.024,.004)))
  qb=b.rotation().T@(p-np.asarray(b.pos));dist=min(dist,cyldist(qb,(0,0,0),.003,.093,'Y'))
  qt=tr.rotation().T@(p-np.asarray(tr.pos));return min(dist,min(boxdist(qt,s[0],s[1]) for s in self.trspec.boxes))
 def update(self,env,d,b,tr):
  dp=self.shapes(d)@d.rotation().T+np.asarray(d.pos);bp=self.bit@b.rotation().T+np.asarray(b.pos);tp=self.tray@tr.rotation().T+np.asarray(tr.pos);pts=np.concatenate([dp,bp,tp]);self.hist=(self.hist+[pts])[-21:]
  still=False
  if len(self.hist)==21:
   h=np.stack(self.hist);still=float(np.max(np.linalg.norm(h[:,None]-h[None,:],axis=-1)))<.001
  from yam_arena import runtime
  base=env.unwrapped;origin=self.lk._np(base.scene.env_origins[0]);tips=[]
  for name in ['left_robot','right_robot']:
   a=base.scene[name];mid=runtime.pad_tips_world(a)-origin;g=runtime.link_poses(a)['gripper'];ids=a.find_joints(['joint7','joint8'],preserve_order=True)[0];j7,j8=np.abs(self.lk._np(a.data.joint_pos)[0,ids]);tips.extend([mid,mid-j7*g[:3,0],mid+j8*g[:3,0]])
  clear=min(self.distance(p,d,b,tr) for p in tips)
  rel=d.rotation().T@(np.asarray(b.pos)-np.asarray(d.pos));axis=d.rotation().T@b.rotation()[:,1];rear=rel[1]+extent(axis)[1];outside=rear<=-.089
  bitlow=np.asarray(b.pos)-extent(b.rotation()[:,1]);drill_supported=abs(float(dp[:,2].min())-self.top)<=.003;bit_supported=self.top-.001<=float(bitlow[2])<=self.top+.025
  qr=tr.rotation().T@(np.asarray(b.pos)-np.asarray(tr.pos));he=extent(tr.rotation().T@b.rotation()[:,1]);lo,hi=qr-he,qr+he
  inside=lo[0]>=-.082 and hi[0]<=.082 and lo[1]>=-.052 and hi[1]<=.052 and abs(float(lo[2])-.003)<=.003 and hi[2]<=.019 and tr.rotation()[2,2]>=math.cos(math.radians(5)) and abs(float(tp[:,2].min())-self.top)<=.003
  self.value=(1. if inside else .5) if outside and drill_supported and bit_supported and clear>=.080 and still else 0.
  self.detail=dict(still=still,hands_mm=clear*1000,bit_outside=bool(outside),nose_gap_mm=(-.069-rear)*1000,drill_supported=bool(drill_supported),bit_supported=bool(bit_supported),in_tray=bool(inside))
 def score(self):return self.value
