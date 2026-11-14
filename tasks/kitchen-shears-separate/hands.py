"""Installed YAM pad endpoints and conservative hand-volume measurements."""
import math
import numpy as np

def _np(value):
    value=getattr(value,'torch',value)
    return value.detach().cpu().numpy().astype(float)

def measure(env):
    from yam_arena import runtime
    base=env.unwrapped; origin=_np(base.scene.env_origins[0])
    tips,centers,radii=[],[],[]
    for name in ('left_robot','right_robot'):
        robot=base.scene[name]
        mid=np.asarray(runtime.pad_tips_world(robot),dtype=float)-origin
        hand=np.asarray(runtime.link_poses(robot)['gripper'],dtype=float)
        root=hand[:3,3]-origin
        ids=robot.find_joints(['joint7'],preserve_order=True)[0]
        half=abs(float(_np(robot.data.joint_pos)[0,ids[0]]))
        tips.extend((mid+half*hand[:3,0],mid-half*hand[:3,0]))
        length=float(np.linalg.norm(mid-root)); n=max(2,int(math.ceil(length/.010))+1)
        for f in np.linspace(0,1,n):
            centers.append(root*(1-f)+mid*f)
            radii.append((1-f)*.040+f*(half+.008)+length/(2*(n-1)))
    return np.asarray(tips),np.asarray(centers),np.asarray(radii)
