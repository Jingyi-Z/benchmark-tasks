"""Unlock a representative reusable soap-pump down-lock.
Primary topology: US6065647A Fig2, radial projections, retaining shoulders and
vertical escape channels. Geometry/loads are estimates, not EuroFlow internals.
Elastic detent resistance is approximated by distributed rotational friction.
"""
import sys,pathlib,math,json,os
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import arena_assets_fix
import labkit
from labkit import part,Joint,Mesh,on_step,on_reset
from yam_arena.task_file import TABLE_TOP
from yam_arena.demo import Waypoint
import yam_arena.demo as dm
from scipy.spatial import ConvexHull
SCENE='station';TOP=TABLE_TOP[SCENE];RUN_STEPS=800;SUCCESS_HOLD_S=1.0
labkit.CONTACT_OFFSET=.0001
BLUE=(.34,.58,.72);WHITE=(.88,.88,.85)

def wedge(ri,ro,a,b,z0,z1):
 cross=[(r*math.cos(t),r*math.sin(t)) for r,t in [(ri,a),(ro,a),(ro,b),(ri,b)]]
 v=np.array([[x,y,z] for z in [z0,z1] for x,y in cross])
 return Mesh(v,ConvexHull(v).simplices,WHITE,'hull')

# Hollow collar wall; two physical retaining ledges leave channels at +/-X.
wall=[wedge(.012,.014,a,a+2*math.pi/32,.170,.194) for a in np.arange(32)*2*math.pi/32]
# Two uninterrupted shoulder faces prevent convex-hull seam snagging.
# The rails fit inside the28mm collar envelope (corner radius13.96mm).
rails=[((0.,sgn*.00685,.189),(.021,.0047,.002),WHITE) for sgn in [-1,1]]
lift=Joint('lift',kind='slide',axis='Z',anchor=(0,0,.18),lower=0.,upper=.018,
 initial=0.,target=.018,stiffness=200.,damping=1.,friction=0.,mass=.001,
 cylinders=[((0,0,.180),.003,.030,WHITE,'Z')])
head=Joint('head',kind='hinge',axis='Z',anchor=(0,0,.200),parent='lift',
 lower=-math.pi,upper=.2,initial=0.,target=0.,stiffness=0.,damping=.008,
 friction=0.,mass=.026,
 cylinders=[((0,0,.202),.0275,.008,WHITE,'Z'),
            ((0,.009,.187),.0015,.002,WHITE,'Z'),((0,-.009,.187),.0015,.002,WHITE,'Z')],
 boxes=[((.027,0,.202),(.040,.010,.008),WHITE)])
BOTTLE=part('pump',pos=(.32,0,TOP),fixed=False,mass=.55,friction=.45,self_collide=True,
 cylinders=[((0,0,.085),.035,.170,BLUE,'Z')],boxes=rails,meshes=wall,joints=[lift,head])
OBJECTS=[BOTTLE]

# The legacy jointFriction field is dimensionless. Author the native per-axis
# absolute friction efforts, confirmed in the installed 110.3.2 USD schema.
_author0=labkit._author
def author(prim_path,cfg,*args,**kwargs):
 root=_author0(prim_path,cfg,*args,**kwargs)
 if cfg.part_name!='pump':return root
 from isaaclab.sim.utils.stage import get_current_stage
 from pxr import PhysxSchema,Sdf
 stage=get_current_stage()
 for name,axis,effort in [('head','angular',.025),('lift','linear',.05)]:
  prim=stage.GetPrimAtPath(f'{prim_path}/{name}/{name}_joint')
  PhysxSchema.PhysxJointAPI.Apply(prim).CreateJointFrictionAttr(0.)
  prim.AddAppliedSchema('PhysxJointAxisAPI:'+axis)
  for attr in ['staticFrictionEffort','dynamicFrictionEffort']:
   prim.CreateAttribute('physxJointAxis:'+axis+':'+attr,Sdf.ValueTypeNames.Float).Set(effort)
  assert 'PhysxJointAxisAPI:'+axis in prim.GetAppliedSchemas()
  print('PUMPFRIC '+json.dumps(dict(joint=name,axis=axis,absolute_effort=effort,legacy_coefficient=0.)),flush=True)
 return root
labkit._author=author

from task_metrics import Metric
_METRIC=Metric(globals());_step=0
_DEBUG=os.environ.get('PUMP_DEBUG')=='1'

@on_reset
def reset(env):
 global _step;_step=0;_METRIC.reset()

@on_step
def update(env):
 global _step;_step+=1
 art=env.unwrapped.scene['pump']
 art.set_joint_position_target(art.data.default_joint_pos)
 d=labkit._part_state(env,BOTTLE);_METRIC.update(env,d)
 if _DEBUG and _step%10==0:
  print('PUMPTASK '+json.dumps(dict(step=_step,score=_METRIC.score(),detail=_METRIC.detail,
   body_pos=list(map(float,d.pos)),body_R=d.rotation().tolist())),flush=True)

def score(objects):return _METRIC.score()
def success(objects):return score(objects)==1.

class Orient(float):
 def __new__(cls,r):v=float.__new__(cls,0.);v.rot=np.asarray(r);return v
_old=dm.grip_rotation
dm.grip_rotation=lambda yaw=0.:yaw.rot if hasattr(yaw,'rot') else _old(yaw)

# Fixed, preflighted two-arm path; no object-state feedback.
"""Fixed robot path; no object-state controller feedback."""
import pathlib, math
import numpy as np
from yam_arena import ik
from yam_arena.constants import READY,FINGER_OPEN,LEFT_BASE_POS,RIGHT_BASE_POS
from yam_arena.demo import Waypoint

def rz(a):return np.array([[math.cos(a),-math.sin(a),0.],[math.sin(a),math.cos(a),0.],[0.,0.,1.]])
def ry(a):return np.array([[math.cos(a),0.,math.sin(a)],[0.,1.,0.],[-math.sin(a),0.,math.cos(a)]])
def frame(x,z):return np.column_stack((x,np.cross(z,x),z))
left=rz(math.radians(-82))*1.0@ry(math.radians(-86))@_old(0.)
right=frame(np.array([1.,0.,0.]),np.array([0.,1.,0.]))
turned=rz(-math.pi/2)@right
body=np.array([.32,0.,TOP+.09]);head=np.array([.32,0.,TOP+.202])
tcp=lambda p,r:np.asarray(p)-.013*r[:,2]
def w(t,p,f,label,r):return Waypoint(t,p,f,label,Orient(r))

def hand_demo():
 hl,rl=ik.tcp_pose(np.array(READY,float),np.array(LEFT_BASE_POS,float))
 hr,rr=ik.tcp_pose(np.array(READY,float),np.array(RIGHT_BASE_POS,float))
 lp=tcp(body,left);rp=tcp(head,right);rt=tcp(head,turned)
 return {'left':[
  w(2.,lp-left[:,2]*.075,-.046,'approach bottle body low',left),
  w(1.,lp,-.046,'seat around bottle body',left),
  w(.7,lp,-.026,'counterhold70mm bottle',left),
  w(9.,lp,-.026,'counterhold through head turn',left),
  w(.6,lp,-.046,'release bottle',left),
  w(1.,lp-left[:,2]*.08,-.046,'withdraw from body',left),
  w(2.,hl,-.046,'park left',rl)],
 'right':[
  w(3.7,rp-right[:,2]*.08,-.046,'wait before pump head',right),
  w(1.,rp,-.046,'approach55mm head',right),
  w(.7,rp,-.020,'grip pump head',right),
  w(3.,rt,-.020,'turn head90degrees aboutZ',turned),
  w(1.,rt,-.020,'hold angular position',turned),
  w(.6,rt,-.046,'release to let spring raise head',turned),
  w(1.,rt-turned[:,2]*.08,-.046,'withdraw from pump',turned),
  w(2.,hr,-.046,'park right',rr),
  w(2.,hr,-.046,'settle',rr)]}

demo=hand_demo
