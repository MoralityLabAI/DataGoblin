"""Visit-specific activation interchange on the physically shared native Aleph A."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
from state03 import panel

LEGACY=Path(__file__).resolve().parents[2]/'aleph_rlt_placement'
sys.path.insert(0,str(LEGACY))


def actor(starts,ops):
    return torch.cat([torch.full((len(ops),1),260,dtype=torch.long),
                      torch.full((len(ops),1),262,dtype=torch.long),
                      starts[:,None],ops+32],1)


def at_visit(model,tokens,physical,visit,replace=None):
    """A direct block output is its next residual; count shared visits explicitly."""
    if physical not in range(1,11) or visit not in (1,2):
        raise ValueError('choose a shared physical core block and pass 1 or 2')
    calls=0; captured=[]
    def hook(_module,_args,out):
        nonlocal calls
        calls+=1
        if calls==visit:
            captured.append(out.detach().clone())
            if replace is not None: return replace(out)
    handle=model.blocks[physical].register_forward_hook(hook)
    try:
        logits=model(tokens,'direct',2,1)
    finally:
        handle.remove()
    assert calls==2 and len(captured)==1
    return logits,captured[0]


def replace_prefix(donor,cut,mode):
    end=3+cut
    def fn(out):
        if donor.shape[1]!=end or donor.shape[0]!=out.shape[0] or donor.shape[2]!=out.shape[2]:
            raise ValueError('donor prefix alignment mismatch')
        out=out.clone()
        if mode=='token': out[:,end-1]=donor[:,-1]
        elif mode=='prefix': out[:,:end]=donor
        else: raise ValueError(mode)
        return out
    return fn


@torch.no_grad()
def evaluate(model,physical,visit,n_words=4):
    if not 1<=n_words<=128: raise ValueError('n_words out of range')
    model.eval(); results={}
    for key,(rp,dp,a,u,idx,y) in panel().items():
        cut=int(key.split('/')[0]); k=n_words*4; n=n_words*12
        rp,dp,a=rp[:k],dp[:k],a[:k]
        u=u[:n]; idx=[v[:n] for v in idx]; y0,y1,yw=[v[:n] for v in y]
        rec=actor(a.repeat_interleave(3),torch.cat([rp.repeat_interleave(3,0),u],1))
        _,d=at_visit(model,actor(a,dp),physical,visit)
        base,r=at_visit(model,rec,physical,visit)
        base=base[:,-1,:4].argmax(1)
        results[key]={}
        for mode in ('token','prefix'):
            controls=dict(self=r[:,:3+cut],same=d[idx[0]],target=d[idx[1]],wrong=d[idx[2]])
            preds={name:at_visit(model,rec,physical,visit,replace_prefix(h,cut,mode))[0][:,-1,:4].argmax(1)
                   for name,h in controls.items()}
            assert torch.equal(preds['self'],base)
            row={}
            for name,mask0 in [('surviving',y0!=y1),('erased',y0==y1)]:
                for tag,mask in [('all',mask0),('base_correct',mask0&base.eq(y0))]:
                    row[f'{name}/{tag}']=dict(n=int(mask.sum()),wrong_oracle_target=int((mask&yw.eq(y1)).sum()),
                        **{name:{'target':int((mask&p.eq(y1)).sum()),'original':int((mask&p.eq(y0)).sum()),
                                 'changed':int((mask&p.ne(base)).sum())} for name,p in preds.items()})
            results[key][mode]=row
    return results


@torch.no_grad()
def smoke(model):
    model.eval(); tokens=actor(torch.tensor([0,1]),torch.tensor([[0,1,2,1],[1,0,2,0]]))
    out={}
    for visit in (1,2):
        z,h=at_visit(model,tokens,10,visit)
        zs,_=at_visit(model,tokens,10,visit,replace_prefix(h[:,:5],2,'prefix'))
        torch.testing.assert_close(z,zs,atol=0,rtol=0)
        zp,_=at_visit(model,tokens,10,visit,replace_prefix(h.flip(0)[:,:5],2,'token'))
        torch.testing.assert_close(z[:,:4],zp[:,:4],atol=1e-5,rtol=1e-5)
        out[str(visit)]={'self_max_delta':float((z-zs).abs().max()),'earlier_max_delta':float((z[:,:4]-zp[:,:4]).abs().max()),
                         'later_max_delta':float((z[:,4:]-zp[:,4:]).abs().max())}
        assert out[str(visit)]['later_max_delta']>0
    return out


def main():
    from study import dense
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path); p.add_argument('--sha256'); p.add_argument('--physical',type=int,default=10)
    p.add_argument('--visit',type=int,default=2);p.add_argument('--words',type=int,default=4)
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(137);m=dense(256)
    if a.checkpoint is None:
        print(json.dumps({'random_init_smoke':smoke(m)},indent=2));return
    if not a.sha256 or hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()!=a.sha256:
        raise ValueError('checkpoint SHA-256 mismatch')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if 'model' in ck: ck=ck['model']
    m.load_state_dict(ck,strict=True)
    print(json.dumps({'checkpoint_sha256':a.sha256,'physical':a.physical,'visit':a.visit,
                      'smoke':smoke(m),'results':evaluate(m,a.physical,a.visit,a.words)},indent=2))

if __name__=='__main__': main()
