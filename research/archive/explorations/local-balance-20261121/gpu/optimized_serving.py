"""Fixed-quota integer-index inference; no Boolean nonzero in the serving path."""
import numpy as np

def prepare_tables(torch,table):
    result=dict(table)
    values=table['gate_folded'].detach().cpu().numpy()
    _,rank=np.unique(values,return_inverse=True)
    result['gate_cell_rank']=torch.as_tensor(rank,dtype=torch.int64,device=table['gate_folded'].device)
    return result

def tie_ranks(torch,ties):
    by_tie=torch.argsort(ties,stable=True)
    ranks=torch.empty(len(ties),dtype=torch.int64,device=ties.device)
    ranks.scatter_(0,by_tie,torch.arange(len(ties),device=ties.device))
    return ranks

def confidence_indices(torch,first,ties):
    # Nonnegative finite IEEE FP32 bit patterns are ordered like their values.
    # Abs canonicalizes negative zero. The integer key also resolves all ties.
    bits=torch.abs(first).contiguous().view(torch.int32).to(torch.int64)
    key=(0x7f800000-bits)*(len(first)+1)+(len(first)-tie_ranks(torch,ties))
    return torch.topk(key,64,sorted=False).indices

def folded_indices(torch,x,ties,table):
    r=torch.linalg.vector_norm(x,dim=1)
    b=torch.bucketize(r,table['gate_edges'],right=False)
    flip=(x[:,1]<0)|((x[:,1]==0)&(x[:,0]<0))
    canonical=torch.where(flip[:,None],-x,x)
    phi=torch.remainder(torch.atan2(canonical[:,1],canonical[:,0]),np.pi)
    ab=torch.clamp((phi*8/np.pi).to(torch.int64),max=7)
    rank=table['gate_cell_rank'][8*b+ab]
    tie_rank=tie_ranks(torch,ties)
    # Integer key exactly reproduces descending cell score then ascending tie.
    key=rank*(len(x)+1)+(len(x)-tie_rank)
    return torch.topk(key,64,sorted=False).indices

def deploy(torch,net,x,ties,table,name,calls=None):
    def predict(z):
        if calls is not None:calls.append(len(z))
        return net(z).flatten()
    n=len(x)
    if name=='full_fused':
        both=predict(torch.cat((x,-x),dim=0))
        return .5*(both[:n]+both[n:]),None
    if name in ('folded_fused','folded_sequential'):
        indices=folded_indices(torch,x,ties,table)
        second=-torch.index_select(x,0,indices)
        if name=='folded_fused':
            both=predict(torch.cat((x,second),dim=0));out=both[:n].clone()
            out.index_copy_(0,indices,.5*(torch.index_select(out,0,indices)+both[n:]))
            return out,indices
    first=predict(x)
    if name=='raw':return first,None
    if name=='full_sequential':return .5*(first+predict(-x)),None
    if name!='folded_sequential':
        indices=confidence_indices(torch,first,ties)
        xx=torch.index_select(x,0,indices)
        if name=='confidence_radial':
            b=torch.bucketize(torch.linalg.vector_norm(xx,dim=1),table['angle_edges'],right=True)
            angle=table['angle_indices'][b].to(torch.float32)*(2*np.pi/128)
            c=torch.cos(angle);s=torch.sin(angle)
            second=torch.stack((c*xx[:,0]-s*xx[:,1],s*xx[:,0]+c*xx[:,1]),dim=1)
        else:second=-xx
    out=first.clone()
    out.index_copy_(0,indices,.5*(torch.index_select(first,0,indices)+predict(second)))
    return out,indices
