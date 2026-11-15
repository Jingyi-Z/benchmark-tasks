"""Release a clamped round-shank drill bit using a loose handheld keyless drill and a parts tray."""
import sys,pathlib,math,json,os
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import labkit
from labkit import part,Joint,Mesh,on_step,on_reset
from yam_arena.task_file import TABLE_TOP
from yam_arena.demo import Waypoint
import yam_arena.demo as dm
from scipy.spatial import ConvexHull
SCENE='station';TOP=TABLE_TOP[SCENE];RUN_STEPS=1600;SUCCESS_HOLD_S=1.0
labkit.CONTACT_OFFSET=.0001
STEEL=(.60,.62,.64);BLACK=(.12,.13,.14);BODY=(.70,.54,.15)
ORIGIN=np.array([.32,.07,TOP+.032]);LEAD=.002;ALPHA=math.radians(20);GEAR=LEAD/(2*math.pi);GEAR_USD=LEAD/360.
def hull(points,col):
 v=np.asarray(points);return Mesh(v,ConvexHull(v).simplices,col,'hull')
def annulus():
 out=[]
 for j in range(16):
  a,b=2*math.pi*j/16,2*math.pi*(j+1)/16
  cross=[(r*math.cos(t),r*math.sin(t)) for r,t in [(.012,a),(.021,a),(.021,b),(.012,b)]]
  out.append(hull([[x,y,z] for y in [-.069,-.029] for x,z in cross],BLACK))
 return out
SLEEVE=Joint('sleeve',kind='hinge',axis='Y',anchor=(0,-.049,0),lower=-6*math.pi,upper=.1,stiffness=0.,damping=.002,friction=.15,mass=.10,meshes=annulus())
JAWS=[]
for j in range(3):
 phi=j*2*math.pi/3;n=np.array([math.cos(phi),0,math.sin(phi)]);t=np.array([-math.sin(phi),0,math.cos(phi)])
 centre=n*.00548+np.array([0.,-.056,0.]);points=[centre+n*(a*.0025)+t*(b*.002)+np.array([0.,c*.012,0.]) for a in [-1,1] for b in [-1,1] for c in [-1,1]]
 JAWS.append(Joint('jaw'+str(j),kind='slide',axis='Y',anchor=tuple(centre),lower=0.,upper=.006,stiffness=0.,damping=0.,mass=.012,meshes=[hull(points,STEEL)]))
# Loose handheld drill; native jaw constraints preserve the calibrated transmission.
DRILL=part('drill',pos=tuple(ORIGIN),fixed=False,mass=.734,friction=.5,self_collide=True,boxes=[((-.075,.026,0),(.15,.032,.035),BODY),((-.135,.029,0),(.04,.055,.06),BLACK)],cylinders=[((0,.027,0),.03,.112,BODY,'Y')],joints=[SLEEVE,*JAWS])
BIT=part('bit',pos=tuple(ORIGIN+[0.,-.0965,0.]),mass=math.pi*.003**2*.093*7850,friction=.5,cylinders=[((0,0,0),.003,.093,STEEL,'Y')])
TRAY=part('tray',pos=(.32,-.065,TOP),mass=.10,friction=.5,boxes=[((0,0,.0015),(.18,.12,.003),(.28,.36,.44)),((.0885,0,.0115),(.003,.12,.017),(.28,.36,.44)),((-.0885,0,.0115),(.003,.12,.017),(.28,.36,.44)),((0,.0585,.0115),(.174,.003,.017),(.28,.36,.44)),((0,-.0585,.0115),(.174,.003,.017),(.28,.36,.44))])
OBJECTS=[DRILL,BIT,TRAY]
_author0=labkit._author

def author(prim_path,cfg,*args,**kwargs):
 root=_author0(prim_path,cfg,*args,**kwargs)
 if cfg.part_name!='drill':return root
 from isaaclab.sim.utils.stage import get_current_stage
 from pxr import Gf,PhysxSchema,Sdf,UsdPhysics
 stage=get_current_stage();leader=f'{prim_path}/sleeve/sleeve_joint'
 for j in range(3):
  phi=j*2*math.pi/3;h=ALPHA/2
  quat=Gf.Quatf(math.cos(h),Gf.Vec3f(math.sin(phi)*math.sin(h),0.,-math.cos(phi)*math.sin(h)))
  path=f'{prim_path}/jaw{j}/jaw{j}_joint';prim=stage.GetPrimAtPath(path);joint=UsdPhysics.PrismaticJoint(prim)
  joint.GetLocalRot0Attr().Set(quat);joint.GetLocalRot1Attr().Set(quat)
  api=PhysxSchema.PhysxMimicJointAPI.Apply(prim,'rotX')
  api.CreateReferenceJointRel().SetTargets([Sdf.Path(leader)]);api.CreateReferenceJointAxisAttr().Set('rotY')
  api.CreateGearingAttr().Set(GEAR_USD);api.CreateOffsetAttr().Set(0.)
 return root
labkit._author=author

from task_metrics import Metric
_METRIC=Metric(globals());_step=0
_DEBUG=os.environ.get('CHUCK_DEBUG')=='1'
def jaw_clearance(d,b):
 # Diagnostic only: bound the cylinder against each jaw's inner plane over its
 # axial span. A negative value is conservative because the plane is unbounded
 # tangentially; None means no remaining bit segment overlaps that jaw span.
 c=d.rotation().T@(np.asarray(b.pos)-np.asarray(d.pos));u=d.rotation().T@b.rotation()[:,1]
 out=[];radial=[]
 for j in range(3):
  q=float(d.joints['jaw'+str(j)]);radial.append((.00298+q*math.sin(ALPHA)-.003)*1e6)
  lo=-.056+q*math.cos(ALPHA)-.012;hi=lo+.024
  if abs(u[1])<1e-8:out.append(None);continue
  ry=.003*math.sqrt(max(0.,1-u[1]**2));a,z=sorted(((lo-c[1]-ry)/u[1],(hi-c[1]+ry)/u[1]));a=max(a,-.0465);z=min(z,.0465)
  if z<a:out.append(None);continue
  phi=j*2*math.pi/3;n=np.array([math.cos(phi),0.,math.sin(phi)]);nu=float(n@u)
  support=float(n@c)+max(a*nu,z*nu)+.003*math.sqrt(max(0.,1-nu**2))
  out.append((.00298+q*math.sin(ALPHA)-support)*1e6)
 return dict(nominal_radial_gap_um=radial,jaw_span_plane_gap_um=out,axis_mismatch_deg=math.degrees(math.acos(float(np.clip(u[1],-1.,1.)))))
@on_reset
def reset(env):
 global _step
 _step=0;_METRIC.reset();_direct.clear()
@on_step
def observe(env):
 global _step
 _step+=1;art=env.unwrapped.scene['drill'];art.set_joint_position_target(art.data.default_joint_pos)
 d=labkit._part_state(env,labkit._SPECS['drill']);b=labkit._part_state(env,labkit._SPECS['bit']);tr=labkit._part_state(env,labkit._SPECS['tray'])
 _METRIC.update(env,d,b,tr)
 if _DEBUG and _step%10==0:print('CHUCKTASK '+json.dumps(dict(step=_step,body_pos=list(map(float,d.pos)),tray_pos=list(map(float,tr.pos)),tray_R=tr.rotation().tolist(),body_R=d.rotation().tolist(),bit_R=b.rotation().tolist(),bit_pos=list(map(float,b.pos)),sleeve_rad=float(d.joints['sleeve']),jaw_m=[float(d.joints['jaw'+str(j)]) for j in range(3)],placement_height_mm=(_direct.get('placement',(None,None,None))[2]*1000 if 'placement' in _direct else None),score=_METRIC.score(),detail=_METRIC.detail,clearance=jaw_clearance(d,b))),flush=True)
def score(objects):return _METRIC.score()
def success(objects):return score(objects)==1.
class Orient(float):
 def __new__(cls,r):v=float.__new__(cls,0.);v.rot=np.asarray(r);return v
old=dm.grip_rotation
def orient(yaw=0.):return yaw.rot if hasattr(yaw,'rot') else old(yaw)
dm.grip_rotation=orient
def frame(x,z):return np.column_stack((x,np.cross(z,x),z))
R_LEFT=frame(np.array([0.,-1.,0.]),np.array([math.sin(math.radians(40)),0.,-math.cos(math.radians(40))]))
R_RIGHT=frame(np.array([1.,0.,0.]),np.array([0.,math.sin(math.radians(75)),-math.cos(math.radians(75))]))
def ry(a):return np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
def tcp(p,r):return np.asarray(p)-r[:,2]*.013

def demo():
 from yam_arena import ik
 from yam_arena.constants import READY,LEFT_BASE_POS,RIGHT_BASE_POS
 homeL,rotL=ik.tcp_pose(np.asarray(READY,float),np.asarray(LEFT_BASE_POS,float));homeR,rotR=ik.tcp_pose(np.asarray(READY,float),np.asarray(RIGHT_BASE_POS,float))
 l=ORIGIN+[-.045,.026,.008];s=ORIGIN+[0,-.049,0];lift=np.array([0,-.040,.180])
 lp=tcp(l,R_LEFT);sp=tcp(s+lift,R_RIGHT);tw=ry(math.radians(-90))@R_RIGHT
 a=math.radians(45);R_TILT=np.array([[1.,0.,0.],[0.,math.cos(a),-math.sin(a)],[0.,math.sin(a),math.cos(a)]])@R_LEFT
 b=math.radians(30);R_TILT=R_TILT@np.array([[1.,0.,0.],[0.,math.cos(b),-math.sin(b)],[0.,math.sin(b),math.cos(b)]])
 def w(d,p,f,label,r):return Waypoint(d,p,f,label,Orient(r))
 left=[w(2.,lp-R_LEFT[:,2]*.070,-.046,'approach existing pistol grip',R_LEFT),
 w(1.,lp,-.046,'lower around grip',R_LEFT),w(.7,lp,0.,'grasp handle',R_LEFT),
 w(1.3,lp+lift,0.,'lift drill on existing grip',R_LEFT),
 w(43.,lp+lift,0.,'hold drill still while bit is grasped and extracted',R_LEFT),
 w(2.4,lp+[0.,.045,0.],0.,'place drill clear of tray',R_LEFT),
 w(.6,lp+[0.,.045,0.],-.046,'release drill',R_LEFT),
 w(.8,lp+[0.,.045,.020]-R_LEFT[:,2]*.070,-.046,'withdraw',R_LEFT),
 w(1.6,homeL,-.046,'park left',rotL),w(18.6,homeL,-.046,'settle',rotL)]
 right=[w(5.,sp-R_RIGHT[:,2]*.090,-.046,'wait clear while tool lifts',R_RIGHT),
 w(1.,sp-R_RIGHT[:,2]*.030,-.046,'approach42mm sleeve',R_RIGHT),
 w(.6,sp,-.046,'position pads across sleeve',R_RIGHT),
 w(.7,sp,0.,'grasp sleeve',R_RIGHT)]
 for quarter in range(6):
  right += [w(2.,tcp(s+lift,tw),0.,f'open sleeve quarter{quarter+1} of6',tw),
            w(.5,tcp(s+lift,tw),0.,'hold opened sleeve',tw)]
  if quarter<5:
   right += [w(.4,tcp(s+lift,tw),-.046,'release sleeve for regrip',tw),
             w(.5,tcp(s+lift,tw)-tw[:,2]*.070,-.046,'withdraw open pads before resetting wrist',tw),
             w(1.,sp-R_RIGHT[:,2]*.070,-.046,'reset wrist while clear of sleeve',R_RIGHT),
             w(.5,sp,-.046,'return open pads to sleeve',R_RIGHT),
             w(.5,sp,0.,'regrasp sleeve',R_RIGHT)]
 right += [w(.8,tcp(s+lift,tw),-.046,'release sleeve after opening540degrees',tw),
           w(.8,tcp(s+lift,tw)-tw[:,2]*.080,-.046,'withdraw from chuck',tw)]
 rp,rrot=_pick_pose(_BIT_NOM,_BIT_R_NOM)
 lay=_LAY
 hover=np.array([.320,-.065,TOP+.120])+.010*lay[:,1]-lay[:,2]*.004
 low=np.array([.320,-.065,TOP+.014])+.010*lay[:,1]-lay[:,2]*.004
 right += [w(1.6,rp-rrot[:,2]*.040,-.046,'bit_hover',rrot),
           w(1.5,rp,-.012,'bit_approach',rrot),w(1.5,rp,0.,'bit_close',rrot),
           w(1.,rp,0.,'bit_hold',rrot),w(2.,rp-_BIT_R_NOM[:,1]*.050,0.,'bit_extract',rrot),
           w(2.,rp-_BIT_R_NOM[:,1]*.080,0.,'bit_extra_extract',rrot),w(6.,rp-_BIT_R_NOM[:,1]*.080,0.,'bit_wait_drill',rrot),w(4.,hover,0.,'bit_transfer',lay),w(1.,hover,0.,'bit_over_tray',lay),
           w(3.,low,0.,'bit_lower',lay),w(1.,low,0.,'bit_rest',lay),
           w(1.,low,-.012,'bit_release',lay),w(2.,hover,-.012,'bit_rise',lay),
           w(1.,hover,-.046,'bit_open',lay),w(2.5,homeR,-.046,'bit_park',rotR),
           w(2.5,homeR,-.046,'bit_settle',rotR)]

 return dict(left=left,right=right)

# Fixed open-loop compensation measured during a prior development demo.
# Sleeve-prefix compensation is fixed; the later bit controller reads visible poses.
from scipy.spatial.transform import Rotation
_ORIG_TARGET=dm.ArmPlan.target
_S_NOM=ORIGIN+np.array([0.,-.089,.180])
_FIXED_POS=np.array([0.29917362332344055, 0.03413170203566551, 0.9129056334495544])
_FIXED_ROT=np.array([[0.9881658943773773, 0.00947351796692178, 0.15309609533040597], [0.0013888310283309557, 0.9974974392373379, -0.0706889607923028], [-0.15338263622868492, 0.0700650446860015, 0.9856797920696617]])
_FIXED_CENTRE=_FIXED_POS+_FIXED_ROT@np.array([0.,-.049,0.])

def track_sleeve(self,t):
 p,rot,f,label=_ORIG_TARGET(self,t)
 if self.base_pos[1]>0 or t<4.:return p,rot,f,label
 w=float(np.clip((t-4.)/1.5,0.,1.))*float(np.clip((39.8-t)/1.4,0.,1.))
 rb=Rotation.from_rotvec(Rotation.from_matrix(_FIXED_ROT).as_rotvec()*w).as_matrix()
 return _S_NOM+w*(_FIXED_CENTRE-_S_NOM)+rb@(np.asarray(p)-_S_NOM),rb@rot,f,label

dm.ArmPlan.target=track_sleeve

# Camera-observable bit/tray poses only. No object writes, joint-state grasp flags,
# contact forces, or kinematic attachment is used by this controller.
_LAY=frame(np.array([.5,math.sqrt(3)/2,0.]),np.array([0.,0.,-1.]))
_BIT_NOM=np.array([.2980050147,-.0633013397,.8820738792])
_BIT_R_NOM=np.array([[-.8435578,-.03364293,.5359836],[.1835908,.91984044,.3466814],[-.5046828,.39084743,-.7697621]])
from scipy.spatial.transform import Slerp
_CAM=json.loads((pathlib.Path(__file__).parent/'camera-replay-d16c.json').read_text(encoding='utf-8-sig'))
_CT=np.array([x['t'] for x in _CAM]);_CP=np.array([x['pos'] for x in _CAM]);_CR=Slerp(_CT,Rotation.from_matrix(np.array([x['rot'] for x in _CAM])))
_HAND_ENVELOPE=np.array(json.loads((pathlib.Path(__file__).parent/'hand-envelope.json').read_text(encoding='utf-8-sig')))
def _camera_replay(t):
 tt=float(np.clip(t,_CT[0],_CT[-1]));return np.array([np.interp(tt,_CT,_CP[:,i]) for i in range(3)]),_CR(tt).as_matrix()
_direct={}
def _pick_pose(pos,rot):
 axis=np.asarray(rot)[:,1];axis=axis/np.linalg.norm(axis)
 if axis[1]<0:axis=-axis
 gx=np.array([1.,0.,0.]);gx-=axis*np.dot(axis,gx);gx/=np.linalg.norm(gx)
 rr=np.column_stack([gx,-axis,np.cross(gx,-axis)])
 return np.asarray(pos)-axis*.010-rr[:,2]*.004,rr
def _blend(a,b,u):
 return Rotation.from_rotvec(float(u)*Rotation.from_matrix(b@a.T).as_rotvec()).as_matrix()@a
_before_direct=dm.ArmPlan.target
def direct_bit(self,t):
 p,rot,f,label=_before_direct(self,t)
 if self.base_pos[1]>0 or not label.startswith('bit_'):return p,rot,f,label
 e=labkit.env()
 if e is None:
  bp,br=_camera_replay(t);bp=bp-br[:,1]*.030*float(np.clip((t-46.)/2.,0.,1.));tp=np.array([.32,-.065,TOP]);tr=np.eye(3)
 else:
  bs=labkit._part_state(e,labkit._SPECS['bit']);ts=labkit._part_state(e,labkit._SPECS['tray'])
  bp,br=bs.pos,bs.rotation();tp,tr=ts.pos,ts.rotation()
 pick,pr=_pick_pose(bp,br);nom,nr=_pick_pose(_BIT_NOM,_BIT_R_NOM)
 rv=Rotation.from_matrix(pr@nr.T).as_rotvec();rv*=min(1.,math.radians(10)/max(np.linalg.norm(rv),1e-12));pr=Rotation.from_rotvec(rv).as_matrix()@nr
 if label in ('bit_hover','bit_approach','bit_close','bit_hold'):
  # Bound the pose correction; retain the planned continuous approach. Only the
  # final few millimetres follow visible drift, before the grasp freezes.
  u=float(np.clip((t-38.4)/1.6,0,1));delta=np.clip(pick-nom,-.008,.008)
  p=np.asarray(p)+u*delta;rot=_blend(rot,pr,u) if label!='bit_hover' else _blend(rot,pr,u)
  _direct['grasp']=(p.copy(),rot.copy(),br[:,1].copy())
 elif label=='bit_extract':
  if 'extract' not in _direct:_direct['extract']=_direct.get('grasp',(nom,nr,_BIT_R_NOM[:,1]))
  gp,gr,axis=_direct['extract'];axis=axis/np.linalg.norm(axis)
  if axis[1]<0:axis=-axis
  p=gp-axis*.050*float(np.clip((t-44.)/2.,0,1));rot=gr
  _direct['withdrawn']=(p.copy(),rot.copy())
 elif label in ('bit_extra_extract','bit_wait_drill'):
  if 'extra' not in _direct:
   hp,hr=_direct['withdrawn'];axis=br[:,1].copy();axis/=np.linalg.norm(axis)
   if axis[1]<0:axis=-axis
   _direct['extra']=(hp.copy(),hr.copy(),axis)
  hp,hr,axis=_direct['extra'];u=float(np.clip((t-46.)/2.,0.,1.));p=hp-axis*.030*u;rot=hr
  _direct['post_extract']=(p.copy(),rot.copy())
 else:
  if 'placement' not in _direct:
   hp,hr=_direct.get('post_extract',(nom-_BIT_R_NOM[:,1]*.080,nr))
   if e is not None:
    from yam_arena import runtime
    rb=e.unwrapped;g=runtime.link_poses(rb.scene['right_robot'])['gripper'];hr=g[:3,:3];hp=g[:3,3]-labkit._np(rb.scene.env_origins[0])+hr@np.array([0.,0.,.134])
   rel=hr.T@(np.asarray(bp)-hp);axis=hr.T@br[:,1];axis/=np.linalg.norm(axis)
   baseline=tr@_LAY;aa=baseline@axis;bb=-baseline[:,1];v=np.cross(aa,bb);c=float(aa@bb)
   if c<-.99:raise RuntimeError('Unexpected reversed held-bit axis; stop before placement')
   k=np.array([[0.,-v[2],v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])
   lay=(np.eye(3)+k+k@k/(1+c))@baseline
   lowest=float(np.min(((_HAND_ENVELOPE-rel)@lay.T)[:,2]));height=max(.014,.005-lowest)
   if height>.040:raise RuntimeError('No low release pose with tray-floor clearance')
   _direct['placement']=(rel,lay,height,hp.copy(),hr.copy())
  rel,lay,height,startp,startr=_direct['placement']
  hover=tp+tr@np.array([0.,0.,.120])-lay@rel
  low=tp+tr@np.array([0.,0.,height])-lay@rel
  if label=='bit_transfer':
   u=float(np.clip((t-54.)/4.,0,1));p=(1-u)*startp+u*hover;rot=_blend(startr,lay,u)
  elif label in ('bit_over_tray','bit_open'):p=hover;rot=lay
  elif label=='bit_lower':u=float(np.clip((t-59.)/3.,0,1));p=(1-u)*hover+u*low;rot=lay
  elif label in ('bit_rest','bit_release'):p=low;rot=lay
  elif label=='bit_rise':u=float(np.clip((t-64.)/2.,0,1));p=(1-u)*low+u*hover;rot=lay
  elif label in ('bit_park','bit_settle'):
   from yam_arena import ik
   from yam_arena.constants import READY,RIGHT_BASE_POS
   home,homeR=ik.tcp_pose(np.asarray(READY,float),np.asarray(RIGHT_BASE_POS,float));u=1. if label=='bit_settle' else float(np.clip((t-67.)/2.5,0,1));p=(1-u)*hover+u*home;rot=_blend(lay,homeR,u)

 return p,rot,f,label
dm.ArmPlan.target=direct_bit
