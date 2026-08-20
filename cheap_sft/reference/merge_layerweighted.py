#!/usr/bin/env python3
"""Layer-weighted delta-scale merge, grounded in the measured weight geometry:
Coder DESCENDS from V2 (late-layer proj->1.0), its delta is ~1.8x v2's magnitude, and in EARLY
layers the coding delta is ANTI-aligned with v2's reasoning (cos<0) => it damaged reasoning/control.
So: W_new = V2 + alpha(L) * (Coder - V2), alpha ramps low (early, protect reasoning) -> high (late,
inject coding). embed/lm_head/norm/mtp/vision are frozen across all 3 -> copied from V2 verbatim.
"""
import os, re, json, gc, sys, argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SSD="/media/kylehessling/Extreme SSD"
V2_PATH=f"{SSD}/Qwopus3.6-27B-v2"       # base (reasoning) — also donates embed/lm_head/norm/mtp/vision
CO_PATH=f"{SSD}/Qwopus3.6-27B-Coder"    # coder — delta source
SHARD=5_000_000_000
NLAYERS=64
COPY_V2=re.compile(r"(embed_tokens|lm_head|^mtp\.|\.mtp\.|nextn|\.norm\.|vision|visual|image|video|patch_embed|merger)")
LAYER_RE=re.compile(r"\.layers\.(\d+)\.")

def load_index(p):
    return {k:os.path.join(p,v) for k,v in json.load(open(f"{p}/model.safetensors.index.json"))["weight_map"].items()}

class ShardWriter:
    def __init__(s,o): s.o,s.buf,s.cur,s.i,s.tot,s.wm=o,{},0,0,0,{}; os.makedirs(o,exist_ok=True)
    def _flush(s):
        if not s.buf: return
        s.i+=1; nm=f"model-{s.i:05d}.safetensors"; save_file(s.buf,os.path.join(s.o,nm),metadata={"format":"pt"})
        for k in s.buf: s.wm[k]=nm
        s.buf,s.cur={},0; gc.collect()
    def add(s,n,t): s.buf[n]=t.contiguous(); s.cur+=t.numel()*t.element_size(); s.tot+=1; (s._flush() if s.cur>=SHARD else None)
    def finalize(s):
        s._flush(); n=s.i
        for j in range(1,n+1):
            old=f"model-{j:05d}.safetensors"; new=f"model-{j:05d}-of-{n:05d}.safetensors"
            os.rename(os.path.join(s.o,old),os.path.join(s.o,new))
            for k,v in list(s.wm.items()):
                if v==old: s.wm[k]=new
        json.dump({"metadata":{"total_size":0},"weight_map":s.wm},open(f"{s.o}/model.safetensors.index.json","w"),indent=2)

def alpha(L, a_early, a_late):
    return a_early + (a_late-a_early)*(L/(NLAYERS-1))   # smooth linear ramp by depth

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--a-early",type=float,default=0.12)
    ap.add_argument("--a-late",type=float,default=0.48)
    ap.add_argument("--uniform",type=float,default=None,help="if set, constant alpha (control candidate)")
    a=ap.parse_args()
    iv,ic=load_index(V2_PATH),load_index(CO_PATH)
    assert set(iv)==set(ic),"key mismatch"
    H={}
    def get(i,n):
        p=i[n]
        if p not in H: H[p]=safe_open(p,"pt")
        return H[p].get_tensor(n)
    w=ShardWriter(a.out); copied=merged=0
    for i,n in enumerate(sorted(iv)):
        if COPY_V2.search(n):
            w.add(n,get(iv,n)); copied+=1
        else:
            m=LAYER_RE.search(n); L=int(m.group(1)) if m else NLAYERS//2
            al = a.uniform if a.uniform is not None else alpha(L,a.a_early,a.a_late)
            v2=get(iv,n).float(); co=get(ic,n).float()
            out=(v2 + al*(co-v2)).to(torch.bfloat16)
            w.add(n,out); merged+=1
        if i%150==0:
            print(f"  [{i}/{len(iv)}] copied={copied} merged={merged}",flush=True); H.clear(); gc.collect()
    w.finalize()
    sched = f"uniform {a.uniform}" if a.uniform is not None else f"ramp {a.a_early}->{a.a_late}"
    print(f"[merge] DONE ({sched}) copied={copied} merged={merged} -> {a.out}",flush=True)
    import shutil
    for fn in os.listdir(V2_PATH):
        if fn.endswith((".json",".model",".txt")) and "safetensors" not in fn: shutil.copy2(os.path.join(V2_PATH,fn),os.path.join(a.out,fn))
    print("[merge] copied config+tokenizer from V2",flush=True)

if __name__=="__main__": main()
