cnt=True
for i in range(10):
    if cnt:
        x=1
        cnt=False
    else:
        print(x)
import numpy as np
stride,pad= conv_param['stride'],conv_param['pad']
N,c,hh,ww=x.shape
F,c,HH,WW=w.shape
x_pad=np.pad(x,(0,0),(0,0),(pad,pad),(pad,pad),mode='constant',constant_values=0)
new_h=1+(hh+2*pad-HH) //stride
new_w=1+(ww+2*pad-WW) //stride
out=np.zeros((N,F,new_h,new_w))
for i in range(N):
    for f in range(F):
        for j in range(new_h):
            for k in range(new_w):
                out[i,f,j,k]=np.sum(x_pad[i,:,j*stride:j*stride+HH,k*stride:k*stride+WW]*w[f,:])+b[f]