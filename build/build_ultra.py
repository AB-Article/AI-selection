import re,glob,pandas as pd
def _clean(c): return re.sub(r"<br>|<sup>|</sup>|[`*]"," ",c).strip()
def _num(s):
    m=re.search(r"-?\d+\.?\d*",str(s).replace(",",""))
    return float(m.group()) if m else None
METRIC={"mAPval50-95":("Object detection","COCO",0.0,"mAP50-95"),
        "mAPmask50-95":("Instance segmentation","COCO",0.0,"mask mAP50-95"),
        "acctop1":("Image classification","ImageNet-1k",0.001,"top1"),
        "mAPpose50-95":("Pose estimation","COCO-Pose",0.0,"mAP50-95"),
        "mAPtest50":("Oriented detection","DOTAv1",0.0,"mAP50")}
def norm(h):
    k=re.sub(r"\s+","",_clean(h))
    return k if k in METRIC else None
rows=[];seen=set()
for f in sorted(glob.glob("ultralytics-main/docs/en/models/*.md")+
                glob.glob("ultralytics-main/docs/en/tasks/*.md")):
    lines=open(f,encoding="utf-8",errors="ignore").read().splitlines()
    blocks=[];cur=[]
    for L in lines:
        if L.strip().startswith("|"): cur.append(L.strip())
        else:
            if len(cur)>2: blocks.append(cur)
            cur=[]
    if len(cur)>2: blocks.append(cur)
    for tb in blocks:
        hdr=tb[0].strip("|").split("|")
        dev={}
        for i,h in enumerate(hdr):
            c=re.sub(r"\s+"," ",_clean(h))
            if "CPU ONNX" in c: dev[i]=("CPU ONNX","ONNX Runtime")
            elif "T4 TensorRT10" in c: dev[i]=("NVIDIA T4","TensorRT10")
        mt={i:norm(h) for i,h in enumerate(hdr) if norm(h)}
        if not dev or not mt: continue
        for L in tb[2:]:
            cells=[c.strip() for c in L.strip("|").split("|")]
            if len(cells)!=len(hdr): continue
            name=re.sub(r"[`*]","",re.sub(r"\[([^\]]*)\].*",r"\1",cells[0])).strip()
            if not name: continue
            for di,(dname,rt) in dev.items():
                ms=_num(cells[di])
                if ms is None or ms<=0: continue
                for mi,mk in mt.items():
                    task,ds,ch,metric=METRIC[mk]; q=_num(cells[mi])
                    if q is None: continue
                    key=(task,dname,name)
                    if key in seen: continue
                    seen.add(key)
                    rows.append(dict(task=task,dataset=ds,model=name,device=dname,
                        runtime=rt,precision="fp32" if dname=="CPU ONNX" else "fp16",
                        cell=f"ultralytics | {dname} | {rt}",metric=metric,
                        Q=q/100.,chance=ch,latency_ms=ms,param_count=None,gmacs=None,
                        source="ultralytics"))
U=pd.DataFrame(rows)
U=U[~((U.task=="Object detection")&U.model.str.contains("-seg",na=False))]
U.to_csv("corpus_ultralytics.csv",index=False)
print(U.groupby(["task","device"]).size().to_string())
