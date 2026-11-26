// MTR-02A public development experiment, standard Node.js.
import { pathToFileURL } from "node:url";

export function mtr02a(seed=20261127,scale=1){
  const A=(x,m)=>{if(!x)throw Error(m);};
  A(Number.isSafeInteger(seed)&&Number.isInteger(scale)&&scale>=1&&scale<=4,"bad arguments");
  const seq=n=>Array.from({length:n},(_,i)=>i),sum=a=>a.reduce((x,y)=>x+y,0),dot=(a,b)=>sum(a.map((x,i)=>x*b[i]));
  const parity=x=>{let r=0;for(;x;x&=x-1)r^=1;return r;};
  const pop=x=>{let r=0;for(;x;x&=x-1)r++;return r;};
  const rng=key=>{let s=2166136261;for(const c of key)s=Math.imul(s^c.charCodeAt(0),16777619);s=s||1;return ()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return (s>>>0)/4294967296;};};
  const shuffle=(a,r)=>{a=[...a];for(let i=a.length-1;i>0;i--){let j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;};
  const rank=xs=>{const p=Array(4).fill(0);let n=0;for(let x of xs)for(let i=3;i>=0;i--)if(x>>i&1){if(p[i])x^=p[i];else{p[i]=x;n++;break;}}return n;};
  const keys=new Set(),full=4294967295,depths=[0,4,8,12,16],certificateCheck={count:0},policyNames=["oracle","two_ply","myopic","information","rank","rlt","damping","pooled"];
  function make(split,family,i){
    const id=`${seed}/${split}/${family}/${i}`,r=rng(id),ri=n=>Math.floor(r()*n),K=depths[i%5],budget=6+i%3,visits=6+i%3;
    let basis=[1,2,4,8];
    for(let k=0;k<20;k++){let a=ri(4),b=ri(4);if(a!==b)basis[a]^=basis[b];}
    A(rank(basis)===4,"invertible encoding");
    function encode(h){
      if(family==="lp")return h;
      if(family==="affine")return basis.reduce((s,a,j)=>s|(parity(a&h)<<j),0);
      return seq(4).reduce((s,j)=>s|((((h>>j)&1)^(j>=2?((h>>(j-1))&1)*((h>>(j-2))&1):0))<<j),0);
    }
    const encoded=seq(16).map(encode);A(new Set(encoded).size===16,"unique certificate");
    const ws=seq(32).map(i=>({h:i>>1,z:i&1,y:encoded[i>>1]}));
    const prior=ws.map(w=>BigInt(1+ri(3)+(w.z===0&&split==="ood"?2:0))),origin=ws.map(()=>2+ri(13)),report=ri(2),channels=[];
    let mass=origin.map(BigInt),den=16n;
    for(let k=0;k<K;k++){
      const a=ws.map(()=>1+ri(7)),b=ws.map(()=>9+ri(7));if(split==="ood"&&k%3===0){for(let j=0;j<32;j++)[a[j],b[j]]=[b[j],a[j]];}
      mass=mass.map((v,j)=>(den-v)*BigInt(a[j])+v*BigInt(b[j]));den*=16n;channels.push([a,b]);
    }
    const weights=mass.map((v,j)=>prior[j]*(report?v:den-v)),coarse=1+ri(15);
    function mask(fn){let m=0;for(let j=0;j<32;j++)if(fn(ws[j]))m=(m|(1<<j))>>>0;return m;}
    function valid(h,w){
      if(family==="lp"){
        const x=seq(4).map(j=>1+(h>>j&1)),b=seq(4).map(j=>1+(w.h>>j&1));
        return x.every((v,j)=>v>=0&&v<=b[j])&&sum(x)===sum(b);
      }
      if(family==="affine")return basis.every((a,j)=>parity(a&h)===(w.y>>j&1));
      return seq(4).every(j=>((h>>j&1)^(j>=2?((h>>(j-1))&1)*((h>>(j-2))&1):0))===(w.y>>j&1));
    }
    const certs=seq(16).map(h=>({kind:"full",h,reward:16,penalty:24,mask:mask(w=>{certificateCheck.count++;const ok=valid(h,w);A(ok===(h===w.h),"certificate cross-check");return ok;})}));
    for(let b=0;b<2;b++)certs.push({kind:"coarse",h:b,reward:4,penalty:10,mask:mask(w=>parity(coarse&w.y)===b)});
    const forms=shuffle(basis,r),actions=[];
    const add=(name,cost,requires,form=0,noisy=false,setup=false)=>{
      const m=setup?0:mask(w=>parity(form&w.y)^(noisy?w.z:0));
      actions.push({name,cost,requires,form,noisy,setup,mask:m});
    };
    add("prepare_a",1+(split==="ood"&&i%4===0?1:0),0,0,false,true);
    add("prepare_b",1+ri(2),0,0,false,true);
    add("a_0",1,1,forms[0]);add("a_1",1,1,forms[1]);add("a_2",1,1,forms[2],i%4===0);
    add("b_0",1,2,forms[2]);add("b_1",1,2,forms[3]);add("b_2",1+ri(2),2,forms[0]^forms[3],true);
    add("direct_0",2+ri(2),0,forms[3]);add("direct_1",2+ri(2),0,forms[1]^forms[2]);
    add("cheap_noise",1,0,coarse,true);
    if(split==="ood"&&i%2===0){actions[4].requires=3;actions[6].cost=2;}
    const sig=JSON.stringify([family,K,budget,basis,coarse,origin,report,channels,prior.map(String),actions]);
    A(!keys.has(sig),"duplicate content root");keys.add(sig);
    return {id,split,family,K,budget,visits,ws,weights,actions,certs,coarse};
  }
  function game(root){
    const {weights,actions,certs,budget,visits}=root,mc=new Map(),tc=new Map(),memo=new Map(),depthMemo=new Map();
    let oracleStates=0;
    const W=m=>{if(!mc.has(m)){let w=0n;for(let j=0;j<32;j++)if(m>>>j&1)w+=weights[j];mc.set(m,w);}return mc.get(m);};
    const norm=(v,m)=>Number(v)/Number(W(m));
    function terminal(m){
      if(tc.has(m))return tc.get(m);
      let best={v:0n,cert:null};
      for(const c of certs){const v=BigInt(c.reward+c.penalty)*W((m&c.mask)>>>0)-BigInt(c.penalty)*W(m);if(v>best.v)best={v,cert:c};}
      tc.set(m,best);return best;
    }
    function children(m,a){if(a.setup)return [[-1,m]];const m1=(m&a.mask)>>>0,m0=(m&~a.mask)>>>0;return [[0,m0],[1,m1]].filter(x=>x[1]);}
    function legal(m,u,b,t){if(!t)return [];return actions.flatMap((a,i)=>!(u>>i&1)&&(u&a.requires)===a.requires&&a.cost<=b&&(a.setup||children(m,a).length>1)?[i]:[]);}
    function next(s,i,m){return {m,u:s.u|(1<<i),b:s.b-actions[i].cost,t:s.t-1};}
    const initial={m:full,u:0,b:budget,t:visits};
    function value(s,look){
      const {m,u,b,t}=s;look=Math.min(look,t,b);
      if(!look)return {v:terminal(m).v,i:-1};
      const exact=look===Math.min(t,b),store=exact?memo:depthMemo,key=exact?`${m}/${u}`:`${m}/${u}/${look}`;
      if(store.has(key))return store.get(key);
      let best={v:terminal(m).v,i:-1};
      for(const i of legal(m,u,b,t)){const a=actions[i],v=children(m,a).reduce((v,[,cm])=>v+value(next(s,i,cm),look-1).v,-BigInt(a.cost)*W(m));if(v>best.v)best={v,i};}
      if(exact)oracleStates++;store.set(key,best);return best;
    }
    function qvals(s){return [-1,...legal(s.m,s.u,s.b,s.t)].map(i=>({i,v:i===-1?terminal(s.m).v:children(s.m,actions[i]).reduce((v,[,m])=>v+value(next(s,i,m),s.t-1).v,-BigInt(actions[i].cost)*W(s.m))}));}
    const maxCert=(m,k)=>Math.max(...certs.filter(c=>c.kind===k).map(c=>norm(W((m&c.mask)>>>0),m)));
    function entropy(m){
      const w=W(m);let H=0;for(let j=0;j<32;j++)if(m>>>j&1){const p=Number(weights[j])/Number(w);H-=p*Math.log2(p);}return H/5;
    }
    function token(s,last=-1,outcome=-1){
      return [pop(s.m)/32,entropy(s.m),maxCert(s.m,"full"),maxCert(s.m,"coarse"),s.b/8,s.t/8,Number(Boolean(s.u&1)),Number(Boolean(s.u&2)),outcome<0?0:outcome===0?-1:1,last<0?0:actions[last].cost/3];
    }
    function features(s,i){
      const {m,u,b,t}=s,T=norm(terminal(m).v,m)/16,H=entropy(m);
      if(i===-1)return [1,1,0,0,0,0,0,0,0,T,0,0,b/8,t/8,maxCert(m,"full"),maxCert(m,"coarse")];
      const a=actions[i],gs=children(m,a),v=gs.reduce((v,[,x])=>v+terminal(x).v,0n),gain=H-sum(gs.map(([,x])=>norm(W(x),m)*entropy(x)));
      const known=actions.filter((p,j)=>u>>j&1&&!p.setup&&!p.noisy).map(p=>p.form),rankGain=a.noisy||a.setup?0:rank([...known,a.form])-rank(known);
      const unlock=a.setup?actions.filter((p,j)=>!(u>>j&1)&&(p.requires&(1<<i))!==0&&(p.requires&(u|(1<<i)))===p.requires).length/4:0;
      return [1,0,+a.setup,+a.noisy,a.cost/3,+(a.requires===1),+(a.requires===2),gain,rankGain,norm(v,m)/16,unlock,pop(a.form)/4,b/8,t/8,maxCert(m,"full"),maxCert(m,"coarse")];
    }
    function observation(s,i,h){const a=actions[i],out=a.setup?-1:(a.mask>>>h)&1,cm=children(s.m,a).find(([o])=>o===out)[1];return [next(s,i,cm),out];}
    function examples(r){
      const out=[],seen=new Set();
      for(let path=0;path<2&&out.length<4;path++){
        let s=initial,hist=[token(s)],h=Math.floor(r()*32);
        for(let turn=0;turn<3&&out.length<4;turn++){
          const key=`${s.m}/${s.u}`;
          if(!seen.has(key)){
            seen.add(key);const labels=qvals(s);
            out.push({hist,ids:labels.map(x=>x.i),x:labels.map(x=>features(s,x.i)),y:labels.map(x=>norm(x.v,s.m)/16)});
          }
          const ls=legal(s.m,s.u,s.b,s.t);if(!ls.length)break;
          let i=path===0?value(s,s.t).i:ls[Math.floor(r()*ls.length)];if(i===-1)i=ls[Math.floor(r()*ls.length)];
          const [ns,o]=observation(s,i,h);s=ns;hist=[...hist,token(s,i,o)];
        }
      }return out;
    }
    function heuristic(s,kind){
      const ls=legal(s.m,s.u,s.b,s.t);if(!ls.length)return -1;
      let best=-1,bv=0;
      const T=norm(terminal(s.m).v,s.m);
      for(const i of ls){const a=actions[i],f=features(s,i);let v;
        if(kind==="information")v=f[7]/a.cost+0.12*f[10]/a.cost-0.015*a.cost;
        else v=f[8]/a.cost+0.32*f[10]/a.cost-0.02*a.cost;
        // Stop if a guaranteed full witness is already available, or the
        // immediate utility is too high for another paid query to improve it.
        if(T>=16-a.cost)v=-1;
        if(v>bv){bv=v;best=i;}
      }return best;
    }
    function run(policy,model=null){
      let utility=0n,cost=0n,checks=0n,pubs=0n,falsePubs=0n,fullPubs=0n,decisions=0;
      const choices=new Map();
      for(let h=0;h<32;h++){
        let s=initial,hist=[token(s)],spent=0,calls=0;
        while(true){
          let i,key=`${s.m}/${s.u}/${JSON.stringify(hist)}`;
          if(policy==="oracle")i=value(s,s.t).i;
          else if(policy==="two_ply")i=value(s,2).i;
          else if(policy==="myopic")i=value(s,1).i;
          else if(policy==="rank"||policy==="information")i=heuristic(s,policy);
          else if(choices.has(key))i=choices.get(key);
          else{const ids=[-1,...legal(s.m,s.u,s.b,s.t)],f=ids.map(i=>features(s,i)),scores=predict(model,hist,f).scores;let j=0;for(let k=1;k<ids.length;k++)if(scores[k]>scores[j])j=k;i=ids[j];choices.set(key,i);}
          decisions++;
          if(i===-1){
            const c=terminal(s.m).cert,ok=c&&Boolean(c.mask>>>h&1),w=weights[h];
            utility+=w*BigInt((c?(ok?c.reward:-c.penalty):0)-spent);cost+=w*BigInt(spent);checks+=w*BigInt(calls);
            if(c)pubs+=w;if(c&&!ok)falsePubs+=w;if(c?.kind==="full")fullPubs+=w;break;
          }
          A(legal(s.m,s.u,s.b,s.t).includes(i),"illegal action");
          const a=actions[i],[ns,o]=observation(s,i,h);A(ns.m>>>h&1,"lost true world");spent+=a.cost;if(!a.setup)calls++;s=ns;hist=[...hist,token(s,i,o)];
        }
      }
      const n=x=>norm(x,full);
      if(policy==="oracle")A(utility===value(initial,visits).v,"oracle replay mismatch");
      else A(utility<=value(initial,visits).v,"oracle exceeded");
      return {utility:n(utility),cost:n(cost),queries:n(checks),publication:n(pubs),falsePublication:n(falsePubs),fullPublication:n(fullPubs),decisions};
    }
    return {initial,value,examples,run,oracleStates:()=>oracleStates};
  }
  const H=10,F=16,P=2*H*H+H+F+H*F,modelSeeds=[19,37,73];
  const modes=["rlt","damping","pooled"];
  function init(mode,s){
    const r=rng(`${seed}/model/${s}`),p=Array.from({length:P},()=>0.15*(2*r()-1));return {mode,p,m:Array(P).fill(0),v:Array(P).fill(0),step:0};
  }
  function forward(model,hist){
    const p=model.p,stack=[];
    let c=Array(H).fill(0);
    for(let t=0;t<hist.length;t++){
      const x=hist[t],z=seq(H).map(j=>Math.tanh(p[2*H*H+j]+sum(seq(H).map(k=>p[j*H+k]*x[k]+p[H*H+j*H+k]*c[k]))));
      stack.push({x,c:[...c],z});
      if(model.mode==="rlt")c=z;
      else if(model.mode==="damping")c=c.map((v,j)=>0.5*v+0.5*x[j]);
      else c=c.map((v,j)=>(t*v+x[j])/(t+1));
    }
    return stack;
  }
  function predict(model,hist,x){
    const stack=forward(model,hist),h=stack.at(-1).z,p=model.p,off=2*H*H+H;
    const scores=x.map(f=>sum(f.map((v,k)=>v*(p[off+k]+sum(h.map((z,j)=>z*p[off+F+j*F+k]))))));
    return {stack,h,scores};
  }
  function update(model,e){
    const {p}=model,{stack,h,scores}=predict(model,e.hist,e.x),g=Array(P).fill(0),off=2*H*H+H,dh=Array(H).fill(0);
    let loss=0;
    for(let i=0;i<e.x.length;i++){
      const f=e.x[i],err=scores[i]-e.y[i],d=2*err/e.x.length;loss+=err*err/e.x.length;
      for(let k=0;k<F;k++){g[off+k]+=d*f[k];for(let j=0;j<H;j++){g[off+F+j*F+k]+=d*h[j]*f[k];dh[j]+=d*p[off+F+j*F+k]*f[k];}}
    }
    let back=dh;
    for(let t=stack.length-1;t>=0;t--){
      const {x,c,z}=stack[t],d=z.map((v,j)=>back[j]*(1-v*v)),prev=Array(H).fill(0);
      for(let j=0;j<H;j++){g[2*H*H+j]+=d[j];for(let k=0;k<H;k++){g[j*H+k]+=d[j]*x[k];g[H*H+j*H+k]+=d[j]*c[k];prev[k]+=d[j]*p[H*H+j*H+k];}}
      if(model.mode!=="rlt")break;back=prev;
    }
    model.step++;const a=0.002*Math.sqrt(1-Math.pow(.999,model.step))/(1-Math.pow(.9,model.step));
    const norm=Math.sqrt(sum(g.map(v=>v*v))),clip=Math.min(1,4/Math.max(1e-12,norm));
    for(let j=0;j<P;j++){const d=g[j]*clip;model.m[j]=.9*model.m[j]+.1*d;model.v[j]=.999*model.v[j]+.001*d*d;p[j]-=a*model.m[j]/(Math.sqrt(model.v[j])+1e-8);}
    A(Number.isFinite(loss),"nonfinite loss");return loss;
  }
  const rows=[],trainRoots=80*scale,ordinaryRoots=40*scale,oodRoots=40*scale;
  let oracleStates=0;
  for(let i=0;i<trainRoots;i++){
    const f=i%2===0?"lp":"affine",r=make("train",f,i),g=game(r),ex=g.examples(rng(r.id+"/explore"));
    A(ex.length<=4&&ex.length>=1,"presentation cap");for(const e of ex)rows.push({...e,root:r.id});oracleStates+=g.oracleStates();
  }
  const models=[],fit=[];
  for(const s of modelSeeds)for(const mode of modes){
    const model=init(mode,s),r=rng(`${seed}/shuffle/${s}`),epochs=14,losses=[];
    for(let epoch=0;epoch<epochs;epoch++){let loss=0;for(const i of shuffle(seq(rows.length),r))loss+=update(model,rows[i]);losses.push(loss/rows.length);}
    models.push(model);fit.push({mode,seed:s,params:P,steps:model.step,firstLoss:losses[0],lastLoss:losses.at(-1)});
  }
  const metrics=["utility","cost","queries","publication","falsePublication","fullPublication"];
  const aggregate=[],rootResults=[];
  for(const split of ["ordinary","ood"]){
    const N=split==="ordinary"?ordinaryRoots:oodRoots;
    for(let i=0;i<N;i++){
      const f=split==="ood"?"quadratic":i%2===0?"lp":"affine",r=make(split,f,i),g=game(r),results={};
      for(const p of policyNames.slice(0,5))results[p]=g.run(p);
      for(const p of modes){const v=models.filter(m=>m.mode===p).map(m=>g.run(p,m));results[p]=Object.fromEntries(metrics.map(k=>[k,sum(v.map(x=>x[k]))/v.length]));results[p].seedUtility=v.map(x=>x.utility);}
      rootResults.push({id:r.id,split,family:f,K:r.K,budget:r.budget,results});oracleStates+=g.oracleStates();
    }
  }
  const round=x=>Math.round(x*1e6)/1e6;
  function interval(x){const n=x.length,m=sum(x)/n,sd=Math.sqrt(sum(x.map(v=>(v-m)**2))/Math.max(1,n-1));return [round(m),round(m-1.96*sd/Math.sqrt(n)),round(m+1.96*sd/Math.sqrt(n))];}
  for(const split of ["ordinary","ood"]){
    const rr=rootResults.filter(r=>r.split===split),mean={};
    for(const p of policyNames)mean[p]=Object.fromEntries(metrics.map(k=>[k,round(sum(rr.map(r=>r.results[p][k]))/rr.length)]));
    const comparisons={};
    for(const p of ["damping","pooled","two_ply","rank"])comparisons[p]=Object.fromEntries(["utility","falsePublication","queries"].map(k=>[k,interval(rr.map(r=>r.results.rlt[k]-r.results[p][k]))]));
    aggregate.push({split,roots:rr.length,mean,comparisons,oracleGapOverTwoPly:interval(rr.map(r=>r.results.oracle.utility-r.results.two_ply.utility))});
  }
  const result={schema:"MTR-02A-public-development-v1",seed,scale,trainRoots,ordinaryRoots,oodRoots,uniqueContentRoots:keys.size,trainingPresentations:rows.length,
    exactCertificateChecks:certificateCheck.count,exactOracleReplays:ordinaryRoots+oodRoots,oracleStates,parameterCount:P,modelSeeds,fit,aggregate,rootResults};
  return {receipt:result,rows,weights:models.map((m,i)=>({mode:m.mode,seed:fit[i].seed,p:m.p}))};
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const flags = new Set(process.argv.slice(2).filter(x=>x.startsWith("--")));
  const args = process.argv.slice(2).filter(x=>!x.startsWith("--")).map(Number);
  const result = mtr02a(...args);
  console.log(JSON.stringify(flags.has("--weights")?result.weights:flags.has("--rows")?result.rows:result.receipt,null,2));
}
