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
 _step=0;_METRIC.reset()
@on_step
def observe(env):
 global _step
 _step+=1;art=env.unwrapped.scene['drill'];art.set_joint_position_target(art.data.default_joint_pos)
 d=labkit._part_state(env,labkit._SPECS['drill']);b=labkit._part_state(env,labkit._SPECS['bit']);tr=labkit._part_state(env,labkit._SPECS['tray'])
 _METRIC.update(env,d,b,tr)
 if _DEBUG and _step%10==0:print('CHUCKTASK '+json.dumps(dict(step=_step,body_pos=list(map(float,d.pos)),tray_pos=list(map(float,tr.pos)),tray_R=tr.rotation().tolist(),body_R=d.rotation().tolist(),bit_R=b.rotation().tolist(),bit_pos=list(map(float,b.pos)),sleeve_rad=float(d.joints['sleeve']),jaw_m=[float(d.joints['jaw'+str(j)]) for j in range(3)],score=_METRIC.score(),detail=_METRIC.detail,clearance=jaw_clearance(d,b))),flush=True)
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
 l=ORIGIN+[-.070,.026,.016];s=ORIGIN+[0,-.049,0];lift=np.array([0,-.040,.180])
 lp=tcp(l,R_LEFT);sp=tcp(s+lift,R_RIGHT);tw=ry(math.radians(-90))@R_RIGHT
 a=math.radians(45);R_TILT=np.array([[1.,0.,0.],[0.,math.cos(a),-math.sin(a)],[0.,math.sin(a),math.cos(a)]])@R_LEFT
 b=math.radians(30);R_TILT=R_TILT@np.array([[1.,0.,0.],[0.,math.cos(b),-math.sin(b)],[0.,math.sin(b),math.cos(b)]])
 def w(d,p,f,label,r):return Waypoint(d,p,f,label,Orient(r))
 left=[w(2.,lp-R_LEFT[:,2]*.070,-.046,'approach existing pistol grip',R_LEFT),
 w(1.,lp,-.046,'lower around grip',R_LEFT),w(.7,lp,0.,'grasp handle',R_LEFT),
 w(1.3,lp+lift,0.,'lift drill on existing grip',R_LEFT),
 w(10.,lp+lift,0.,'counterhold until sleeve hand has withdrawn',R_LEFT),
 w(1.,tcp(l+lift+[0.,-.045,0.],R_LEFT),0.,'centre the open chuck over the tray',R_LEFT),
 w(2.5,tcp(l+lift+[0.,-.045,0.],R_TILT),0.,'tip the open chuck45deg nose down',R_TILT),
 w(3.,tcp(l+lift+[0.,-.045,0.],R_TILT),0.,'hold nose down while the loose bit slides into the tray',R_TILT),
 w(2.5,tcp(l+lift+[0.,-.045,0.],R_LEFT),0.,'return the empty drill to level',R_LEFT),
 w(2.4,lp+[0.,.045,0.],0.,'place drill on counter clear of bit',R_LEFT),
 w(.6,lp+[0.,.045,0.],-.046,'release drill on counter',R_LEFT),
 w(.8,lp+[0.,.045,.020]-R_LEFT[:,2]*.070,-.046,'withdraw from drill',R_LEFT),
 w(1.6,homeL,-.046,'park left',rotL),w(2.5,homeL,-.046,'settle released scene',rotL)]
 right=[w(5.,sp-R_RIGHT[:,2]*.090,-.046,'wait clear while tool lifts',R_RIGHT),w(1.,sp-R_RIGHT[:,2]*.030,-.046,'approach 42mm sleeve',R_RIGHT),w(.6,sp,-.046,'position pads across sleeve',R_RIGHT),w(.7,sp,0.,'grasp sleeve',R_RIGHT),w(2.,tcp(s+lift,tw),0.,'turn sleeve90deg open about spindle axis',tw),w(2.7,tcp(s+lift,tw),0.,'hold open while released bit falls into tray',tw),w(.8,tcp(s+lift,tw),-.046,'release sleeve',tw),w(.8,tcp(s+lift,tw)-tw[:,2]*.080,-.046,'withdraw from chuck',tw),w(1.4,homeR,-.046,'park right',rotR),w(16.9,homeR,-.046,'settle released scene',rotR)]
 return dict(left=left,right=right)

# Fixed open-loop compensation measured during a prior development demo.
# Neither demo() nor its waypoint controller reads simulator object state.
from scipy.spatial.transform import Rotation
_ORIG_TARGET=dm.ArmPlan.target
_S_NOM=ORIGIN+np.array([0.,-.089,.180])
_FIXED_POS=np.array([0.29303455352783203,0.05095786601305008,0.8994172215461731])
_FIXED_ROT=np.array([[0.962344451406542,-0.0036695201144600897,0.27180817218373415],[0.030448254080352788,0.9950715508058823,-0.09436901727717628],[-0.2701222902079559,0.09909158501371995,0.9577133161868852]])
_FIXED_CENTRE=_FIXED_POS+_FIXED_ROT@np.array([0.,-.049,0.])

def track_sleeve(self,t):
 p,rot,f,label=_ORIG_TARGET(self,t)
 if self.base_pos[1]>0 or t<4.:return p,rot,f,label
 w=float(np.clip((t-4.)/1.5,0.,1.))*float(np.clip((15.-t)/1.4,0.,1.))
 rb=Rotation.from_rotvec(Rotation.from_matrix(_FIXED_ROT).as_rotvec()*w).as_matrix()
 return _S_NOM+w*(_FIXED_CENTRE-_S_NOM)+rb@(np.asarray(p)-_S_NOM),rb@rot,f,label

dm.ArmPlan.target=track_sleeve
