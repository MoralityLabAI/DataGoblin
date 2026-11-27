// CORT-02: learned confidence with causal prefix features; development only.
import { rng, episode, rank, fit } from './gym.mjs';

export function rootKey(task) {
  if(task.family==="subset")return JSON.stringify({...task,values:[...task.values].sort((a,b)=>a-b)});
  return JSON.stringify(task);
}

export function split(seed, sizes) {
  const random=rng(seed), seen=new Set(), families=["recurrence","graph","subset"];
  return sizes.map(([n,shift])=>{
    const out=[];
    while(out.length<n){
      const e=episode(random,families[out.length%3],shift), key=rootKey(e.task);
      if(seen.has(key))continue;
      seen.add(key);out.push(e);
    }
    return out;
  });
}

export function observation(records){
  const r=rank(records,true), top=r[0], k=records.length;
  const good=records.filter(x=>x.answer!==null), exec=good.filter(x=>x.executed);
  const support=top?.members??[];
  const methods=new Set(exec.map(x=>x.method));
  const conflict=Number(r.length>1);
  return {answer:top?.answer??null,methods:top?.methods??0,x:[
    k/7,good.length/7,exec.length/7,(k-good.length)/7,methods.size/3,
    (top?.methods??0)/3,(top?.exec??0)/7,support.length/7,r.length/7,
    (top?.score??0)/3,((top?.score??0)-(r[1]?.score??0))/3,conflict,
    ...[0,1,2].map(m=>Number(support.some(x=>x.executed&&x.method===m))),
    ...[0,1,2].map(m=>Number(new Set(exec.filter(x=>x.method===m).map(x=>x.answer)).size>1)),
    Number(support.some(x=>!x.executed)),Number(support.some(x=>x.role>=4))
  ]};
}

export function sequences(es){
  return es.map(e=>e.records.map((_,i)=>{
    const o=observation(e.records.slice(0,i+1));
    return {...o,y:Number(o.answer!==null&&o.answer===e.truth)};
  }));
}

export function model(seed, recurrent, hidden=10, dim=20) {
  const random=rng(seed), init=n=>Array.from({length:n},()=>.16*(2*random()-1));
  return {recurrent,hidden,dim,w:init(hidden*dim),u:recurrent?init(hidden*hidden):Array(hidden*hidden).fill(0),
    b:Array(hidden).fill(0),v:init(hidden),c:[0]};
}

export function forward(net, seq, cutHistory=false){
  let prev=Array(net.hidden).fill(0);
  return seq.map(o=>{
    const h=net.b.map((b,j)=>{
      let z=b;
      for(let i=0;i<net.dim;i++)z+=net.w[j*net.dim+i]*o.x[i];
      if(net.recurrent&&!cutHistory)for(let i=0;i<net.hidden;i++)z+=net.u[j*net.hidden+i]*prev[i];
      return Math.tanh(z);
    });
    const logit=h.reduce((a,b,i)=>a+b*net.v[i],net.c[0]),p=1/(1+Math.exp(-logit));
    const state={h,prev,p};prev=h;return state;
  });
}

export function train(net, seqs, seed, epochs=14) {
  const random=rng(seed), keys=["w","u","b","v","c"];
  const accum=Object.fromEntries(keys.map(k=>[k,Array(net[k].length).fill(0)]));
  const momentum=Object.fromEntries(keys.map(k=>[k,Array(net[k].length).fill(0)]));
  let step=0;
  for(let epoch=0;epoch<epochs;epoch++){
    const order=seqs.map((_,i)=>i);
    for(let i=order.length-1;i>0;i--){const j=Math.floor(random()*(i+1));[order[i],order[j]]=[order[j],order[i]];}
    for(const ix of order){
      const seq=seqs[ix], states=forward(net,seq);
      const g=Object.fromEntries(keys.map(k=>[k,Array(net[k].length).fill(0)]));
      let next=Array(net.hidden).fill(0);
      for(let t=seq.length-1;t>=0;t--){
        const {h,prev,p}=states[t], x=seq[t].x, dy=(p-seq[t].y)/seq.length;
        g.c[0]+=dy;
        const delta=h.map((z,j)=>(dy*net.v[j]+next[j])*(1-z*z));
        for(let j=0;j<net.hidden;j++){
          g.v[j]+=dy*h[j];g.b[j]+=delta[j];
          for(let i=0;i<net.dim;i++)g.w[j*net.dim+i]+=delta[j]*x[i];
          if(net.recurrent)for(let i=0;i<net.hidden;i++)g.u[j*net.hidden+i]+=delta[j]*prev[i];
        }
        next=net.recurrent?Array.from({length:net.hidden},(_,i)=>delta.reduce((s,d,j)=>s+d*net.u[j*net.hidden+i],0)):Array(net.hidden).fill(0);
      }
      step++;
      // Adam; identical settings and training roots for feed-forward and recurrent controls.
      for(const key of keys){
        if(key==="u"&&!net.recurrent)continue;
        for(let i=0;i<g[key].length;i++){
          const val=g[key][i];
          momentum[key][i]=.9*momentum[key][i]+.1*val;
          accum[key][i]=.999*accum[key][i]+.001*val*val;
          const m=momentum[key][i]/(1-.9**step),v=accum[key][i]/(1-.999**step);
          net[key][i]-=.003*m/(Math.sqrt(v)+1e-8);
        }
      }
    }
  }
  return net;
}

export function predictor(policy, net, bins, cut=false){
  if(policy==="tabular")return (seq,records)=>seq.map((_,i)=>{
    // Match the CORT-01 lookup key exactly, including empty states.
    const r=rank(records.slice(0,i+1),true),t=r[0];
    const key=t?[Math.min(t.methods,3),Math.min(t.exec,4),Number(r.length>1)].join(":"):"empty";
    const b=bins[key];return b?(b.correct+1)/(b.n+2):0;
  });
  return seq=>forward(net,seq,cut).map(x=>x.p);
}

export function prepared(es,predict){
  return es.map(e=>{
    const seq=e.records.map((_,i)=>observation(e.records.slice(0,i+1)));
    return {truth:e.truth,seq,p:predict(seq,e.records),records:e.records};
  });
}

export function decide(row,threshold,cap=7){
  for(let i=0;i<cap;i++){
    const o=row.seq[i],p=row.p[i],k=i+1;
    if(k===cap || (k>=2&&o.answer!==null&&o.methods>=2&&p>=threshold))
      return {answer:o.answer,calls:k,p,early:k<cap};
  }
  throw Error("unreachable");
}

export function measure(rows,threshold){
  let correct=0,calls=0,early=0,falseEarly=0,brier=0;
  const decisions=rows.map(r=>{
    const d=decide(r,threshold),ok=d.answer!==null&&d.answer===r.truth;
    correct+=Number(ok);calls+=d.calls;early+=Number(d.early);falseEarly+=Number(d.early&&!ok);
    brier+=(d.p-Number(ok))**2;
    return {...d,ok};
  });
  return {n:rows.length,correct,calls,early,falseEarly,brier:brier/rows.length,decisions};
}

export function tune(rows,target=4){
  let best=null;
  for(let i=0;i<=101;i++){
    const threshold=i/100, m=measure(rows,threshold);
    if(m.calls>target*m.n)continue;
    if(!best||m.correct>best.correct||(m.correct===best.correct&&m.calls<best.calls))
      best={threshold,correct:m.correct,calls:m.calls};
  }
  return best; // Infeasible targets are reported; never silently widen the budget.
}

export function matched(rows,decisions,seed){
  const total=decisions.reduce((s,d)=>s+d.calls,0),n=rows.length,base=Math.floor(total/n),extra=total-base*n;
  const random=rng(seed),order=rows.map((_,i)=>i);
  for(let i=n-1;i>0;i--){const j=Math.floor(random()*(i+1));[order[i],order[j]]=[order[j],order[i]];}
  const plus=new Set(order.slice(0,extra));
  let correct=0,win=0,loss=0,calls=0;
  for(let i=0;i<n;i++){
    const k=base+Number(plus.has(i)),answer=rank(rows[i].records.slice(0,k),true)[0]?.answer??null;
    const ok=answer!==null&&answer===rows[i].truth;
    correct+=Number(ok);calls+=k;
    win+=Number(decisions[i].ok&&!ok);loss+=Number(!decisions[i].ok&&ok);
  }
  if(calls!==total)throw Error("matched call count mismatch");
  return {correct,calls,win,loss};
}

export function evaluate(es,predict,threshold,seed){
  const rows=prepared(es,predict),m=measure(rows,threshold);
  const control=matched(rows,m.decisions,seed);
  const {decisions,...summary}=m;
  return {...summary,matched:control};
}

export function campaign2(seed=27012026){
  const [tr,ca,id,ood]=split(seed,[[900,false],[450,false],[450,false],[450,true]]);
  const seqs=sequences(tr),bins=fit(tr);
  const ff=train(model(seed^31,false),seqs,seed^81), rnn=train(model(seed^31,true),seqs,seed^81);
  const policies=[["tabular",predictor("tabular",null,bins)],
    ["feedforward",predictor("ff",ff)],["recurrent",predictor("rnn",rnn)],
    ["history_cut",predictor("rnn",rnn,null,true)]];
  const out={seed,counts:[tr.length,ca.length,id.length,ood.length],policies:{}};
  for(const [name,pred] of policies){
    const cal=prepared(ca,pred),t=tune(cal);
    if(!t){out.policies[name]={feasible:false};continue;}
    out.policies[name]={threshold:t.threshold,calibration:{correct:t.correct,calls:t.calls},
      ordinary:evaluate(id,pred,t.threshold,seed^21),shift:evaluate(ood,pred,t.threshold,seed^21)};
  }
  return out;
}

export function tests(){
  const assert=(v,msg)=>{if(!v)throw Error(msg)};
  const [a,b,c,d]=split(88,[[12,false],[12,false],[12,false],[12,true]]);
  assert(new Set([...a,...b,...c,...d].map(e=>rootKey(e.task))).size===48,"root overlap");
  const seq=sequences(a),n=model(1,true),ff=model(1,false);
  assert(seq[0][0].x.length===n.dim,"feature shape");
  assert(forward(ff,seq[0]).every((s,i)=>s.p===forward(ff,seq[0],true)[i].p),"static history independence");
  const before=JSON.stringify(n.w);train(n,seq,11,2);
  assert(JSON.stringify(n.w)!==before&&n.u.some(x=>x!==0),"training update");
  const row=prepared(b,predictor("rnn",n))[0];
  for(let p of [0,.5,1.01]){
    const v=decide(row,p);
    assert(v.calls>=2&&v.calls<=7&&v.early===(v.calls<7),"stop accounting");
    assert(v.answer===row.seq[v.calls-1].answer,"prefix provenance");
  }
  const rr=prepared(c,predictor("rnn",n)), m=measure(rr,.3);
  assert(matched(rr,m.decisions,91).calls===m.calls,"matched budget");
  return "CORT-02 root custody, recurrent training, prefix provenance, stop accounting, and matched cost checks passed";
}
