"""Normal-fit trajectory spatial prior, never manually placed on test images."""
import numpy as np
from .config import PROTOCOL


def fit_roi(rows,geometry,classes,ids,fit):
    groups={}
    for i in np.flatnonzero(fit):
        for j,c in enumerate(classes[i]):
            if c<0:continue
            key=(int(c),rows[i]['clip'],int(ids[i,j]))
            groups.setdefault(key,[]).append(geometry[i,j,1:3])
    result={}
    for c in sorted(set(classes[classes>=0].tolist())):
        moving=[]; recordings=set(); tracks=0; stationary=0
        for (role,clip,ident),points in groups.items():
            if role!=c:continue
            points=np.array(points)
            if len(points)<PROTOCOL['roi_min_track_observations']:continue
            # Robust percentiles avoid a single jittered detector box defining motion.
            displacement=float(np.linalg.norm(np.quantile(points,.9,axis=0)-np.quantile(points,.1,axis=0)))
            if displacement>=PROTOCOL['roi_mobile_displacement']:
                moving.extend(points.tolist());recordings.add(clip);tracks+=1
            else:stationary+=1
        enabled=tracks>=PROTOCOL['roi_min_mobile_tracks'] and len(recordings)>=PROTOCOL['roi_min_mobile_recordings']
        rectangle=None
        if enabled:
            points=np.array(moving);q=PROTOCOL['roi_quantiles'];pad=PROTOCOL['roi_padding']
            lo=np.maximum(0,np.quantile(points,q[0],axis=0)-pad)
            hi=np.minimum(1,np.quantile(points,q[1],axis=0)+pad)
            rectangle=[*lo.tolist(),*hi.tolist()]
        result[str(c)]={'enabled':enabled,'normal_center_region':rectangle,'mobile_tracks':tracks,
                        'mobile_recordings':len(recordings),'stationary_tracks':stationary,
                        'fit_normal_only':True,'fallback':'retain role if no reliable mobile spatial prior'}
    return result


def acceptance(geometry,classes,roi):
    keep=classes>=0
    for key,rule in roi.items():
        if not rule['enabled']:continue
        x0,y0,x1,y1=rule['normal_center_region'];c=int(key)
        inside=(geometry[...,1]>=x0)&(geometry[...,1]<=x1)&(geometry[...,2]>=y0)&(geometry[...,2]<=y1)
        keep&=(classes!=c)|inside
    return keep


def apply_roi(geometry,classes,ids,crops,roi):
    keep=acceptance(geometry,classes,roi)
    out_c=classes.copy();out_i=ids.copy();out_g=geometry.copy();out_x=crops.copy()
    out_c[~keep]=-1;out_i[~keep]=-1;out_g[~keep]=0;out_x[~keep]=0
    return out_g,out_c,out_i,out_x,keep
