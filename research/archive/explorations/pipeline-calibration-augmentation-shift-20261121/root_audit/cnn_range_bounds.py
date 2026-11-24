"""Analytical pixel-cube range bound, evaluated in FP64; not a real-data certificate.

For output intervals [l_k,u_k], w_k=u_k-l_k, each half-view difference
d_k lies in [-w_k/2,w_k/2]. Thus span(d) <= half the sum of the two largest
w_k. Positive temperature multiplies that bound. This diagnostic is fixed
before inspecting the newly trained models and runs after serving timing.
No input-TV or posterior-TV upper bound is inferred from the models or images.
"""
from pathlib import Path
import datetime, hashlib, json, os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent.parent
HERE=ROOT/'augmentation'

class IndependentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU())
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(64*7*7,128),nn.ReLU(),nn.Linear(128,10))
    def forward(self,x):return self.head(self.features(x))

@torch.inference_mode()
def propagate(net):
    lower=torch.zeros((1,1,28,28),dtype=torch.float64)
    upper=torch.ones_like(lower)
    rows=[]
    for index,layer in enumerate([*net.features,*net.head]):
        if isinstance(layer,(nn.Conv2d,nn.Linear)):
            positive=layer.weight.clamp(min=0);negative=layer.weight.clamp(max=0)
            if isinstance(layer,nn.Conv2d):
                op=lambda x,w,b:F.conv2d(x,w,b,stride=layer.stride,padding=layer.padding,dilation=layer.dilation,groups=layer.groups)
            else:op=lambda x,w,b:F.linear(x,w,b)
            lo=op(lower,positive,layer.bias)+op(upper,negative,None)
            hi=op(upper,positive,layer.bias)+op(lower,negative,None)
            # Deliberately conservative numerical padding, not formal interval arithmetic.
            pad=1e-10*torch.maximum(torch.ones_like(lo),torch.maximum(abs(lo),abs(hi)))
            lower,upper=lo-pad,hi+pad
        elif isinstance(layer,nn.ReLU):lower,upper=lower.clamp(min=0),upper.clamp(min=0)
        elif isinstance(layer,nn.MaxPool2d):lower,upper=layer(lower),layer(upper)
        elif isinstance(layer,nn.Flatten):lower,upper=layer(lower),layer(upper)
        else:raise TypeError(type(layer))
        assert torch.all(lower<=upper)
        rows.append(dict(layer=index,type=type(layer).__name__,minimum=float(lower.min()),maximum=float(upper.max())))
    widths=(upper-lower).numpy()[0]
    return float(np.sort(widths)[-2:].sum()/2),lower.numpy()[0],upper.numpy()[0],rows

def main():
    assert (HERE/'TIMING_COMPLETE.json').exists(),'Run numerical audit after isolated benchmarks.'
    torch.set_num_threads(1)
    rows=[]
    temps=json.loads((HERE/'temperatures.json').read_text())['rows']
    for row in temps:
        path=HERE/'models'/f"{row['pair']}_{row['arm']}.pt"
        payload=torch.load(path,map_location='cpu',weights_only=True)
        net=IndependentCNN().double();net.load_state_dict(payload['state_dict']);net.eval()
        D,lower,upper,layers=propagate(net)
        rows.append(dict(pair=row['pair'],arm=row['arm'],model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            pixel_domain=[0.,1.],output_lower=lower.tolist(),output_upper=upper.tolist(),
            raw_span_D=D,raw_temperature_span_D=D*row['raw']['beta'],full_temperature_span_D=D*row['full']['beta'],
            layer_ranges=layers))
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS',models=rows,
        conditional_bound_only=True,natural_input_tv_upper_bound=None,natural_posterior_tv_upper_bound=None,
        real_data_certificate=False,model_forward_examples=0,
        qualification='Analytical interval-propagation bound evaluated in FP64 with outward numerical padding; not a formally rounded machine proof. Pixel cube includes transformed inputs. No natural-distribution defect bounds are supplied.')
    (ROOT/'root_audit/CNN_RANGE_BOUNDS.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(dict(status='PASS',models=len(rows),minimum_raw_D=min(r['raw_span_D'] for r in rows),
                         maximum_raw_D=max(r['raw_span_D'] for r in rows),real_data_certificate=False)))

if __name__=='__main__':main()
