"""Frozen finite-input readouts with safe nonfinite missing-value handling."""

import numpy as np

def corr(x,y):
    if len(x)<3 or min(np.std(x),np.std(y))<1e-9:return 0.
    return float(np.corrcoef(x,y)[0,1])

def double_features(a,b,z,h):
 a,b,z,h=np.broadcast_arrays(a,b,z,h);have_s=np.isfinite(a)&np.isfinite(b);have=have_s&np.isfinite(z);have_h=have&np.isfinite(h)
 # Keep the availability masks above; replace every nonfinite value before
 # arithmetic so a missing infinity cannot overflow and survive as NaN * 0.
 a,b,z,h=[np.nan_to_num(v,nan=0.,posinf=0.,neginf=0.) for v in [a,b,z,h]];eps=.1
 raw=abs(z-a)-abs(z-b);null=abs(h-a)-abs(h-b);res=z-h
 retain=(z-h)*(b/(b*b+eps**2)-a/(a*a+eps**2))
 smooth=np.logaddexp(0,abs(z-a))-np.logaddexp(0,abs(z-b))
 reliability=abs(res)/(abs(res)+.3)
 d=np.stack([raw*have,(raw-null)*have_h,retain*have_h,smooth*have,raw*reliability*have_h],axis=-1)
 s=np.stack([(a-b)*have_s,(a*a-b*b)*have_s,(a-b)/(abs(a)+abs(b)+eps)*have_s],axis=-1)
 q=np.stack([have.astype(float),have_h.astype(float),reliability*have_h,abs(a-b)*have_s],axis=-1)
 return s,d,q
