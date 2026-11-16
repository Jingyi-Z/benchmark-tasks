"""Reviewer-requested full production scorer checks; CPU only, no native sim.

Object poses are prescribed static inputs. Only installed-hand runtime access
is stubbed; Metric.update executes geometry, removal, support, tray, six-point
hand clearance and the complete 21-sample stillness test unchanged.
"""
from pathlib import Path
import json,math,types,sys,hashlib,csv
import numpy as np
import torch
import task
from task_metrics import Metric,rotation_y,extent

HERE=Path(__file__).resolve().parent
def rz(a):
 c,s=math.cos(a),math.sin(a)
 return np.array([[c,-s,0],[s,c,0],[0,0,1.]])
class Part:
 def __init__(self,p,R=None,joints=None):
  self.pos=np.array(p,float);self.R=np.eye(3) if R is None else R;self.joints=joints or {}
 def rotation(self):return self.R

class Robot:
 def __init__(self,mid):
  self.mid=np.array(mid,float)
  self.data=types.SimpleNamespace(joint_pos=torch.tensor([[.012,.012]]))
 def find_joints(self,*args,**kwargs):return ([0,1],None)
class Scene(dict):env_origins=torch.zeros((1,3))
robots={'left_robot':Robot([.20,.50,1.30]),'right_robot':Robot([.20,-.50,1.30])}
env=types.SimpleNamespace(unwrapped=types.SimpleNamespace(scene=Scene(robots)))
runtime=types.SimpleNamespace(pad_tips_world=lambda robot:robot.mid.copy(),link_poses=lambda robot:{'gripper':np.eye(4)})
import yam_arena
yam_arena.runtime=runtime;sys.modules['yam_arena.runtime']=runtime

def make_drill(yaw,flip,m):
 d=Part([.32,.19,task.TOP],rz(math.radians(yaw))@rotation_y(math.radians(flip)),
        {'sleeve':-3*math.pi,**{f'jaw{i}':.003 for i in range(3)}})
 d.pos[2]=task.TOP-float((m.shapes(d)@d.R.T)[:,2].min())
 return d

tr=Part([.32,-.065,task.TOP]);rows=[]
for flip in (0,180):
 for yaw in range(0,360,45):
  for bit_yaw in (0,45,90,135):
   Rb=rz(math.radians(bit_yaw));half=extent(Rb[:,1])
   # Exact cylinder extents, independent of production tray-inclusion code.
   # Every extremal placement retains at least2mm margin to each inner wall.
   room=np.array([.082,.052])-half[:2]-.002
   assert min(room)>0
   for ix in (-1,0,1):
    for iy in (-1,0,1):
     b=Part(tr.pos+np.array([ix*room[0],iy*room[1],.006]),Rb)
     m=Metric(vars(task));d=make_drill(yaw,flip,m)
     for _ in range(21):m.update(env,d,b,tr)
     assert m.score()==1.,(yaw,flip,bit_yaw,ix,iy,m.detail)
     assert all(m.detail[k] for k in ('still','bit_outside','drill_supported','bit_supported','in_tray'))
     assert m.detail['hands_mm']>=80
     row=dict(drill_yaw_deg=yaw,drill_roll_deg=flip,bit_yaw_deg=bit_yaw,
              tray_x_mm=float(ix*room[0]*1000),tray_y_mm=float(iy*room[1]*1000),
              inner_wall_margin_mm=float(np.min(np.array([.082,.052])-np.abs([ix*room[0],iy*room[1]])-half[:2])*1000),
              floor_clearance_mm=3.,score=m.score(),detail=m.detail)
     rows.append(row)
  print(f'PASS yaw={yaw} roll={flip}:36/36 valid tray placements score1.0',flush=True)

# Meaningful guard counterexamples alongside the requested positive matrix.
controls=[]
for name in ('outside_tray','short_history','near_hand','moving_bit','not_removed'):
 m=Metric(vars(task));d=make_drill(0,0,m);b=Part(tr.pos+[0,0,.006],rz(math.pi/2))
 if name=='outside_tray':b.pos[0]+=.20
 if name=='near_hand':robots['left_robot'].mid=b.pos+[0,0,.004]
 if name=='not_removed':b.pos=d.pos+d.R@np.array([0.,-.049,0.]);b.R=d.R.copy()
 for i in range(20 if name=='short_history' else 21):
  if name=='moving_bit' and i==20:b.pos[0]+=.002
  m.update(env,d,b,tr)
 expected=.5 if name=='outside_tray' else 0.
 assert m.score()==expected,(name,m.score(),m.detail)
 if name=='not_removed':assert not m.detail['bit_outside']
 controls.append(dict(case=name,expected=expected,score=m.score(),detail=m.detail))
 robots['left_robot'].mid=np.array([.20,.50,1.30])

manifest={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ('task.py','task_metrics.py','labkit.py','arena_assets_fix.py','task.json','camera-replay-d16c.json','hand-envelope.json')}
summary=[]
for flip in (0,180):
 for yaw in range(0,360,45):
  rs=[r for r in rows if r['drill_roll_deg']==flip and r['drill_yaw_deg']==yaw]
  summary.append(dict(drill_yaw_deg=yaw,drill_roll_deg=flip,count=len(rs),minimum_score=min(r['score'] for r in rs),maximum_score=max(r['score'] for r in rs)))
report=dict(scope='CPU production Metric.update with prescribed static part poses,21observations per positive case,and only hand runtime stubbed. This is scorer validation,not576native runs or a dynamics/stability proof.',submitted_head='650061ff003499cacd36472f31e338a74441745f',production_source_sha256=manifest,orientation_count=16,placements_per_orientation=36,positive_cases=len(rows),all_positive_full_score=all(r['score']==1 for r in rows),min_hand_clearance_mm=min(r['detail']['hands_mm'] for r in rows),minimum_inner_wall_margin_mm=min(r['inner_wall_margin_mm'] for r in rows),summary=summary,guard_controls=controls,rows=rows)
(HERE/'tray-orientation-checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
(HERE/'tray-orientation-summary.json').write_text(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2)+'\n',encoding='utf-8')
with (HERE/'tray-orientation-checks.csv').open('w',newline='',encoding='utf-8') as f:
 writer=csv.DictWriter(f,fieldnames=[k for k in rows[0] if k!='detail']);writer.writeheader();writer.writerows({k:v for k,v in row.items() if k!='detail'} for row in rows)
print(json.dumps({k:v for k,v in report.items() if k not in ('rows','production_source_sha256')}),flush=True)
