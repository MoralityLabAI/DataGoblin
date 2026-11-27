// MTR-02G interruptible constructive portfolio, public development only.
import {pathToFileURL} from "node:url";
export function mtr02g(seed=20261127,scale=1){
  const A=(x,m)=>{if(!x)throw Error(m);},seq=n=>Array.from({length:n},(_,i)=>i),sum=a=>a.reduce((x,y)=>x+y,0),mean=a=>sum(a)/a.length;
  A(Number.isSafeInteger(seed)&&Number.isInteger(scale)&&scale>=1&&scale<=2,"bad configuration");
  const N=128,ALL=(1n<<128n)-1n,BITS=seq(N).map(i=>1n<<BigInt(i)),BI=new Map(BITS.map((b,i)=>[b,i])),STOP=Symbol("work"),names=["eager_route","eager_info","route","info","constructive_first","round_robin","budget_conditioned","mass_gap","random19","random37","random73","learned","certification"],signatures=new Set();
  const checks={witness:0,nonlinear:0,legalLists:0,childMasks:0,oracleReplays:0,policyReplays:0,rootBounds:0,backups:0,relaxedOracle:0,relaxedTighter:0};
  function rng(k){let s=2166136261;for(const c of k)s=Math.imul(s^c.charCodeAt(0),16777619);s||=1;return()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return(s>>>0)/4294967296;};}
  function shuffle(a,r){a=[...a];for(let i=a.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
  function make(split,i){
    const id=`${seed}/mtr02g/${split}/${i}`,r=rng(id),ri=n=>Math.floor(r()*n),family=split==="ood"?"cross_gate_cubic":i%2?"coupled_payload":"quadratic_tree",budget=7+i%3,perm=shuffle(seq(7),r),flip=ri(128);
    function witness(h){
      const b=perm.map(j=>(h^flip)>>j&1),p=b.slice(4),y=[p[0]^(family==="coupled_payload"?p[1]*p[2]:0),p[1],p[2]];
      const s0=b[0]^(family==="cross_gate_cubic"?p[0]*p[1]*p[2]:p[0]*p[1]),s1=b[1]^(s0*p[2]),s2=b[2]^(s1*p[0]),s3=b[3]^(family==="cross_gate_cubic"?s2*p[0]*p[1]:p[1]*p[2]);
      const branch=8*s0+4*s1+2*s2+s3,payload=4*y[0]+2*y[1]+y[2],coarse=y[0]^(y[1]*y[2]);
      return{branch,payload,coarse};
    }
    const worlds=seq(N).map(witness),w=seq(N).map(h=>1+ri(4)+(i%4===0&&worlds[h].branch<8?2:0));
    A(new Set(worlds.map(x=>8*x.branch+x.payload)).size===N,"nonbijective encoding");
    A(seq(7).some(j=>new Set(seq(N).map(h=>(worlds[h].branch>>3)^(worlds[h^(1<<j)].branch>>3))).size>1),"linear selector");checks.nonlinear++;
    const mask=f=>worlds.reduce((m,x,h)=>f(x,h)?m|BITS[h]:m,0n),actions=[];
    function add(kind,cost,l,p,gate,observe){
      const g=gate||mask(x=>l<0||x.branch>>(4-l)===p),outcomes=worlds.map(observe),parts=new Map();
      for(let h=0;h<N;h++)parts.set(outcomes[h],(parts.get(outcomes[h])||0n)|BITS[h]);
      actions.push({kind,cost,l,p,gate:g,outcomes,parts:[...parts].sort((a,b)=>a[0]-b[0])});
    }
    for(let l=0;l<4;l++)for(let p=0;p<1<<l;p++){
      add("single",1+Number(ri(4)===0)+Number(split==="ood"&&l===2&&p%3===0&&i%3===0),l,p,null,x=>x.branch>>(3-l)&1);
      if(l<3)add("pair",1+ri(3)+Number(split==="ood"&&p%2===0&&i%4===0),l,p,null,x=>x.branch>>(2-l)&3);
    }
    for(let p=0;p<16;p++){
      add("full",2+ri(3)+Number(split==="ood"&&p%3===0),4,p,null,x=>x.payload);
      add("partial",1+ri(2),4,p,null,x=>x.payload>>1);
    }
    add("global",1,-1,0,ALL,x=>x.coarse);
    add("global",1+ri(2),-1,0,ALL,x=>(x.branch>>3)^(x.payload&1));
    add("global",2+Number(split==="ood"&&i%3===0),-1,0,ALL,x=>x.payload>>1);
    add("global",1+ri(2),-1,0,ALL,x=>((x.branch>>3)^(x.branch>>1)^(x.payload>>2))&1);
    for(let c=0;c<2;c++){
      const gate=mask(x=>split==="ood"?((x.payload>>2)^(x.branch&1))===c:x.coarse===c);
      add("cross",2+ri(2),-1,0,gate,x=>split==="ood"?((x.branch>>1)&3)^(x.payload&3):(x.branch>>2)^(x.payload>>1));
    }
    A(actions.length===60,"action count");
    const rewards={full:[36,54],partial:[24,36],branch:[15,23],coarse:[8,12]},kinds=["full","partial","branch","coarse"],sizes=[128,64,16,2];
    const label=(kind,x,h)=>kind==="full"?h:kind==="partial"?4*x.branch+(x.payload>>1):kind==="branch"?x.branch:x.coarse;
    for(let k=0;k<kinds.length;k++)for(let j=0;j<sizes[k];j++){
      const m=mask((x,h)=>label(kinds[k],x,h)===j);
      for(let h=0;h<N;h++){A(Boolean(m&BITS[h])===(label(kinds[k],witness(h),h)===j),"certificate");checks.witness++;}
    }
    const sig=JSON.stringify([family,budget,perm,flip,w,actions.map(a=>[a.cost,a.gate.toString(),a.outcomes])]);A(!signatures.has(sig),"duplicate root");signatures.add(sig);
    return{id,split,family,budget,worlds,w,actions,rewards,kinds,sizes,label,total:sum(w)};
  }
  const key=s=>`${s.m.toString(16)}/${s.b}`;
  function meter(cap){return{cap,jobCap:cap,used:0,c:{},tick(k,n=1){const left=Math.min(this.cap,this.jobCap)-this.used,z=Math.min(n,left);A(Number.isInteger(n)&&n>=0&&z>=0,"meter");this.used+=z;this.c[k]=(this.c[k]||0)+z;if(z<n)throw STOP;},left(){return this.cap-this.used;}};}
  function index(r,M){
    const lists=[],masks=[],extras=[],charge=(k,n=1)=>M?.tick(k,n);
    for(let l=0;l<=4;l++){lists[l]=[];masks[l]=[];for(let p=0;p<1<<l;p++){charge("indexBuild",2);lists[l][p]=[];masks[l][p]=0n;}}
    for(let j=0;j<r.actions.length;j++){charge("indexBuild",3);const a=r.actions[j];if(a.l<0)extras.push(j);else{lists[a.l][a.p].push(j);masks[a.l][a.p]=a.gate;}}
    return{lists,masks,extras,entries:62+r.actions.length};
  }
  function indexed(s,x,M){
    const out=[],charge=(k,n=1)=>M?.tick(k,n);let p=0;
    for(let l=0;l<=4;l++){charge("indexRead");for(const j of x.lists[l][p]){charge("indexEmit");out.push(j);}if(l===4)break;charge("prefixTest");const z=s.m&x.masks[l+1][2*p];if(z===s.m)p*=2;else if(z===0n)p=2*p+1;else break;}
    charge("indexRead");for(const j of x.extras){charge("indexEmit");out.push(j);}return out;
  }
  function legal(r,s,ids,M){
    const out=[],charge=(k,n=1)=>M?.tick(k,n);if(s.b<=0)return out;
    for(const i of ids){charge("legal");const a=r.actions[i];if(a.cost>s.b||(s.m&a.gate)!==s.m)continue;const cs=[];
      for(const [o,p] of a.parts){charge("partition");const m=s.m&p;if(m)cs.push({o,s:{m,b:s.b-a.cost}});}if(cs.length>1)out.push({i,cost:a.cost,cs});}
    return out;
  }
  function replay(r,start,read){
    let value=0,cost=0,queries=0,pub=0,wrong=0,full=0;
    for(let h=0;h<N;h++){
      let s={m:ALL,b:r.budget},p=start,spent=0,q=0;
      while(true){
        const a=read(p);
        if(a.i<0){const good=a.kind==="abstain"||r.label(a.kind,r.worlds[h],h)===a.claim,[g,b]=a.kind==="abstain"?[0,0]:r.rewards[a.kind],w=r.w[h];value+=w*((good?g:-b)-spent);cost+=w*spent;queries+=w*q;pub+=w*Number(a.kind!=="abstain");wrong+=w*Number(!good);full+=w*Number(a.kind==="full");break;}
        const x=r.actions[a.i];A(x.cost<=s.b&&(s.m&x.gate)===s.m,"illegal replay");const o=x.outcomes[h],parts=x.parts.filter(([,m])=>Boolean(m&s.m)),part=parts.find(z=>z[0]===o),child=a.cs.find(z=>z.o===o);
        A(parts.length>1&&part&&child,"invalid observation");s={m:s.m&part[1],b:s.b-x.cost};A(Boolean(s.m&BITS[h]),"truth removed");p=child.p;spent+=x.cost;q++;
      }
    }
    return{raw:value,utility:value/r.total,cost:cost/r.total,queries:queries/r.total,publication:pub/r.total,falsePublication:wrong/r.total,fullPublication:full/r.total};
  }
  function relax(r,s,ls,stats,M){
    const charge=(k,n=1)=>M?.tick(k,n),t=stats(s.m);if(!ls.length)return t.best.v;
    let cells=[s.m],min=Infinity;for(const a of ls){charge("relax");min=Math.min(min,a.cost);}
    for(const a of r.actions){charge("relax");if(a.cost>s.b||!(a.gate&s.m))continue;
      const next=[];for(const c of cells)for(const [,p] of a.parts){charge("relax");const z=c&p;if(z)next.push(z);}
      cells=next;
      // Positive-mass worlds; singleton cells cannot refine further.
      if(cells.length===t.n)break;
    }
    let v=-min*t.mass;for(const c of cells){charge("relax");v+=stats(c).best.v;}
    return Math.max(t.best.v,v);
  }
  function oracle(r,audit){
    const sm=new Map(),memo=new Map(),states=[],idx=index(r,null);
    function stats(m){
      if(sm.has(m))return sm.get(m);let mass=0,n=0;const bins=r.sizes.map(n=>Array(n).fill(0));
      for(let h=0;h<N;h++)if(m&BITS[h]){mass+=r.w[h];n++;for(let k=0;k<4;k++)bins[k][r.label(r.kinds[k],r.worlds[h],h)]+=r.w[h];}
      let best={v:0,i:-1,kind:"abstain",claim:-1};
      for(let k=0;k<4;k++){const kind=r.kinds[k],[g,b]=r.rewards[kind];for(let j=0;j<bins[k].length;j++){const v=(g+b)*bins[k][j]-b*mass;if(v>best.v)best={v,i:-1,kind,claim:j};}}
      const z={mass,n,best};sm.set(m,z);return z;
    }
    function solve(s){
      const k=key(s);if(memo.has(k))return memo.get(k);const t=stats(s.m);let p=t.best;
      if(s.b>0){
        const ls=legal(r,s,seq(r.actions.length),null),zs=legal(r,s,indexed(s,idx,null),null);
        A(ls.length===zs.length,"indexed length");checks.legalLists++;
        for(let i=0;i<ls.length;i++){const a=ls[i],z=zs[i];A(a.i===z.i&&a.cost===z.cost&&a.cs.length===z.cs.length,"index action");for(let j=0;j<a.cs.length;j++){A(a.cs[j].o===z.cs[j].o&&a.cs[j].s.m===z.cs[j].s.m,"child mask");checks.childMasks++;}}
        for(const a of ls){if((36-a.cost)*t.mass<=p.v)continue;let v=-a.cost*t.mass;const cs=[];for(const c of a.cs){const q=solve(c.s);v+=q.v;cs.push({o:c.o,p:q});}if(v>p.v)p={v,i:a.i,cs};}
        if(ls.length)states.push({s,ls});
      }
      memo.set(k,p);return p;
    }
    const p=solve({m:ALL,b:r.budget}),out=replay(r,p,x=>x);A(out.raw===p.v,"oracle replay");checks.oracleReplays++;
    if(audit)for(const {s,ls} of shuffle(states,rng(r.id+"/relax-audit")).slice(0,64)){
      const t=stats(s.m),u=Math.max(t.best.v,...ls.map(a=>(36-a.cost)*t.mass)),h=relax(r,s,ls,stats,null),v=solve(s).v;A(v<=h&&h<=u,"invalid relaxed bound");checks.relaxedOracle++;if(h<u)checks.relaxedTighter++;
    }
    return{...out,states:memo.size};
  }
  function context(r,M){
    const sm=new Map(),lm=new Map(),pm=[new Map(),new Map(),new Map()],roll=[new Map(),new Map()];let idx=null;
    function stats(m){
      M.tick("lookup");if(sm.has(m))return sm.get(m);
      const bins=r.kinds.map(()=>new Map()),mx=[0,0,0,0],arg=[0,0,0,0];let mass=0,n=0;
      for(let z=m;z;z&=z-1n){
        M.tick("world",5);const h=BI.get(z&-z),w=r.w[h],x=r.worlds[h];mass+=w;n++;
        for(let k=0;k<4;k++){const j=r.label(r.kinds[k],x,h),v=(bins[k].get(j)||0)+w;bins[k].set(j,v);if(v>mx[k]||v===mx[k]&&j<arg[k]){mx[k]=v;arg[k]=j;}}
      }
      let best={v:0,i:-1,kind:"abstain",claim:-1};for(let k=0;k<4;k++){M.tick("claim");const kind=r.kinds[k],[g,b]=r.rewards[kind],v=(g+b)*mx[k]-b*mass;if(v>best.v)best={v,i:-1,kind,claim:arg[k]};}
      const t={mass,n,best};sm.set(m,t);return t;
    }
    function ls(s){
      M.tick("lookup");const k=key(s);if(lm.has(k))return lm.get(k);if(!idx)idx=index(r,M);const a=s.b>0?legal(r,s,indexed(s,idx,M),M):[];lm.set(k,a);return a;
    }
    function proposal(s,k){
      M.tick("lookup");const id=key(s),memo=pm[k];if(memo.has(id))return memo.get(id);const t=stats(s.m);let best=null,score=-Infinity;
      for(const a of ls(s)){
        M.tick("proposal");const kind=r.actions[a.i].kind;let v;
        if(k===0){v=Number(!["global","cross"].includes(kind))+Math.log2(a.cs.length)/a.cost;}
        else if(k===1){let mx=0;for(const c of a.cs){M.tick("proposal");mx=Math.max(mx,stats(c.s.m).mass);}v=(t.mass-mx)/t.mass/a.cost;}
        else{let q=-a.cost*t.mass;for(const c of a.cs){M.tick("proposal");q+=stats(c.s.m).best.v;}v=q/t.mass;}
        if(v>score){score=v;best=a;}
      }
      // The immediate stream is a one-step constructive strategy; it stops
      // where its preferred probe cannot improve on the current terminal.
      if(k===2&&score<=t.best.v/t.mass)best=null;
      memo.set(id,best);return best;
    }
    function rollout(s,k){
      M.tick("lookup");const id=key(s),memo=roll[k];if(memo.has(id))return memo.get(id);
      const t=stats(s.m);let best=t.best;const a=proposal(s,k);
      if(a){let v=-a.cost*t.mass;const cs=[];for(const c of a.cs){M.tick("plan");const p=rollout(c.s,k);v+=p.v;cs.push({o:c.o,p});}if(v>best.v)best={v,i:a.i,cs};}
      M.tick("plan");memo.set(id,best);return best;
    }
    return{stats,ls,proposal,rollout,info:()=>({statStates:sm.size,legalStates:lm.size,proposalStates:sum(pm.map(x=>x.size)),rolloutStates:sum(roll.map(x=>x.size)),indexEntries:idx?.entries||0})};
  }
  function run(r,cap,mode,models=null,train=false){
    const M=meter(cap),C=context(r,M),nodes=new Map(),rr=rng(r.id+"/"+mode),initial={m:ALL,b:r.budget},labels=[],trace=[];
    let root=null,fallback=null,fallbackU=36*r.total,stopped=false,actionJobs=0,boundJobs=0,abortedJobs=0,abortedWork=0,relaxWork=0,relaxTightens=0,relaxReduction=0,scans=0,reuse=0,eagerComplete=false;
    const snapshot=()=>({work:M.used,lower:(root?.L??fallback?.v??0)/r.total,upper:(root?.U??fallbackU)/r.total,actionJobs,boundJobs});
    function create(s,stage){
      M.tick("lookup");const k=key(s);if(nodes.has(k)||stage.has(k)){reuse++;return nodes.get(k)||stage.get(k);}
      const t=C.stats(s.m),ls=C.ls(s);let U=t.best.v,ua=null;
      for(const a of ls){M.tick("bound");const v=(36-a.cost)*t.mass;if(v>U){U=v;ua=a.i;}}
      const n={k,s,t,ls,L:t.best.v,U,ua,la:null,arcs:new Map(),parents:new Set(),relaxed:null};M.tick("graph");stage.set(k,n);return n;
    }
    function initialize(){
      fallback=C.stats(ALL).best;const stage=new Map(),n=create(initial,stage);fallbackU=n.U;M.tick("graph");nodes.set(n.k,n);root=n;
    }
    function refine(job){
      const n=job.n,stage=new Map(),overlay=new Map(),links=[],arcs=new Map(n.arcs);M.tick("graph",n.arcs.size+1);let relaxed=n.relaxed;
      const start=M.used;
      if(job.type==="action"){
        A(!arcs.has(job.a.i),"repeated expansion");const cs=[];
        for(const c of job.a.cs){M.tick("child");const z=create(c.s,stage);cs.push({o:c.o,n:z});links.push(z);}
        arcs.set(job.a.i,{i:job.a.i,cost:job.a.cost,cs});
      }else{
        A(relaxed===null,"repeated relaxation");const v=relax(r,n.s,n.ls,C.stats,M);A(v>=n.L,"relax below incumbent");relaxed=v;
      }
      const pending=new Set([n]);
      while(pending.size){
        let x=null;for(const z of pending){M.tick("graph");if(x===null||z.s.b<x.s.b)x=z;}pending.delete(x);
        const aa=x===n?arcs:x.arcs,rel=x===n?relaxed:x.relaxed;
        let L=x.t.best.v,U=x.t.best.v,la=null,ua=null;
        for(const a of x.ls){
          M.tick("backup");const arc=aa.get(a.i);let u=(36-a.cost)*x.t.mass;
          if(arc){let l=-a.cost*x.t.mass;u=l;for(const c of arc.cs){M.tick("backup");const z=overlay.get(c.n)||c.n;l+=z.L;u+=z.U;}if(l>L){L=l;la=arc;}}
          if(u>U){U=u;ua=a.i;}
        }
        if(rel!==null)U=Math.min(U,rel);
        A(L<=U&&L>=x.L&&U<=x.U,"nonmonotone backup");checks.backups++;
        overlay.set(x,{L,U,la,ua});if(L!==x.L||U!==x.U)for(const p of x.parents){M.tick("graph");pending.add(p);}
      }
      const previous=n.U;
      M.tick("graph",stage.size+links.length+overlay.size+2);
      for(const [k,z] of stage)nodes.set(k,z);for(const z of links)z.parents.add(n);
      n.arcs=arcs;n.relaxed=relaxed;for(const [x,z] of overlay)Object.assign(x,z);
      if(job.type==="action")actionJobs++;else{boundJobs++;relaxWork+=M.used-start;if(n.U<previous){relaxTightens++;relaxReduction+=(previous-n.U)/r.total;}}
    }
    function bestJob(stack,k){
      const seen=new Set();let best=null,v=-Infinity,rel=null,rbest=-Infinity;
      while(stack.length){
        M.tick("frontier");const n=stack.pop();if(seen.has(n.k))continue;seen.add(n.k);if(n.U<=n.L)continue;
        const gap=n.U-n.L;
        if(k===3&&n.relaxed===null){const score=gap/(1+n.t.n);if(score>rbest){rbest=score;rel={n,type:"bound",stream:3,gap};}}
        let a=null;
        if(k<3)a=C.proposal(n.s,k);
        else{for(const z of n.ls){M.tick("frontier");if(z.i===n.ua){a=z;break;}}}
        if(!a)continue;
        const arc=n.arcs.get(a.i);
        if(arc){for(const c of arc.cs){M.tick("frontier");stack.push(c.n);}}
        else{const u=(36-a.cost)*n.t.mass,score=Math.min(gap,Math.max(0,u-n.L));if(score>v){v=score;best={n,type:"action",a,stream:k,gap:score};}}
      }
      return{action:best,bound:rel};
    }
    function jobs(){
      const out=[],which=mode==="route"?[0,3]:mode==="info"?[1,3]:mode==="mass_gap"?[3]:[0,1,2,3];
      for(const k of which){
        const a=bestJob([root],k);if(a.action)out.push(a.action);
        // Single-stream controls use optimistic action fallback, without relaxation.
        if(k===3&&a.bound&&!["route","info","mass_gap"].includes(mode))out.push(a.bound);
      }
      return out;
    }
    function feat(j,q){
      M.tick("feature",22);const n=j.n,m=n.t.mass/r.total,L=n.L/n.t.mass/36,U=n.U/n.t.mass/36,c=j.type==="action"?j.a.cs.length:0,a=j.type==="action"?r.actions[j.a.i]:null;
      return[1,m,n.t.n/128,n.s.b/9,L,U,(n.U-n.L)/r.total/36,j.gap/r.total/36,root.L/r.total/36,(root.U-root.L)/r.total/36,c/8,a?a.cost/5:0,+(j.type==="bound"),+(j.stream===0),+(j.stream===1),+(j.stream===2),+(j.stream===3),+(a?.kind==="global"),+(a?.kind==="full"),n.arcs.size/13,n.ls.length/15,Math.min(1,Math.log1p(q)/Math.log1p(32768))];
    }
    const estimate=j=>j.type==="bound"?60+5*j.n.t.n:20+5*j.n.t.n+80*j.a.cs.length;
    function pick(fs){
      if(mode==="route"||mode==="info"||mode==="constructive_first"){const order=mode==="route"?[0,3]:mode==="info"?[1,3]:[0,1,2,3];for(const k of order)for(const j of fs){M.tick("priority");if(j.stream===k&&j.type==="action")return j;}return fs[0];}
      if(mode==="round_robin"){M.tick("priority",fs.length);return fs[(actionJobs+boundJobs)%fs.length];}
      if(mode.startsWith("random")){M.tick("priority");return fs[Math.floor(rr()*fs.length)];}
      let best=null,score=-Infinity;
      for(const j of fs){
        M.tick("priority");scans++;let v;
        if(mode==="mass_gap")v=j.gap;
        else if(mode==="budget_conditioned")v=j.gap/(1+estimate(j))*(estimate(j)>M.left()/2?0.1:1);
        else{
          const x=feat(j,M.left());M.tick("priority",44);const c=Math.exp(Math.min(16,Math.max(0,sum(x.map((z,i)=>z*models.cost[i]))))),p=Math.max(0,sum(x.map((z,i)=>z*models[mode==="learned"?"gain":"gap"][i])));
          v=p/(1+c)*(c>M.left()?0.05:1);
        }
        if(v>score||v===score&&best&&j.gap>best.gap){score=v;best=j;}
      }return best;
    }
    function read(p){return p.s?(p.la?{i:p.la.i,cs:p.la.cs.map(c=>({o:c.o,p:c.n}))}:p.t.best):p;}
    let pendingJob=null,jobStart=0;
    try{
      initialize();trace.push(snapshot());
      if(mode.startsWith("eager")){
        const p=C.rollout(initial,mode==="eager_route"?0:1);fallback=p;eagerComplete=true;trace.push({work:M.used,lower:p.v/r.total,upper:root.U/r.total,actionJobs:0,boundJobs:0});
      }else if(train){
        const checkpoints=new Set([0,2,5,10]),qv=[512,2048,8192,32768];
        for(let j=0;j<=10&&root.L<root.U;j++){
          const fs=jobs();A(fs.length,"empty training frontier");const job=fs[Math.floor(rr()*fs.length)],sample=checkpoints.has(j),quantum=sample?qv[(Number(r.id.split("/").at(-1))+[0,2,5,10].indexOf(j))%4]:10000000,x=sample?feat(job,quantum):null,l=root.L,g=root.U-root.L,start=M.used;
          let censored=false;M.jobCap=Math.min(cap,start+quantum);
          try{refine(job);}catch(e){if(e!==STOP)throw e;censored=true;abortedJobs++;abortedWork+=M.used-start;}finally{M.jobCap=cap;}
          if(sample)labels.push({root:r.id,type:job.type,stream:job.stream,x,quantum,work:M.used-start,censored,initialLower:l/r.total,initialGap:g/r.total,gain:(root.L-l)/r.total,gap:(g-(root.U-root.L))/r.total});
        }
      }else{
        let nextMark=cap/4;
        while(root.L<root.U){const fs=jobs();A(fs.length,"empty portfolio frontier");pendingJob=pick(fs);jobStart=M.used;refine(pendingJob);pendingJob=null;
          if(M.used>=nextMark){trace.push(snapshot());while(nextMark<=M.used)nextMark+=cap/4;}
        }
      }
    }catch(e){if(e!==STOP)throw e;stopped=true;if(pendingJob){abortedJobs++;abortedWork+=M.used-jobStart;}}
    A(fallback,"no terminal");if(train)return labels;
    const p=mode.startsWith("eager")?fallback:root||fallback,L=mode.startsWith("eager")?fallback.v:(root?.L??fallback.v),U=root?.U??fallbackU,out=replay(r,p,read);
    A(out.raw===L,"partial incumbent replay");checks.policyReplays++;
    trace.push({work:M.used,lower:L/r.total,upper:U/r.total,actionJobs,boundJobs});
    return{...out,lower:L/r.total,upper:U/r.total,certifiedGap:(U-L)/r.total,work:M.used,counts:M.c,actionJobs,boundJobs,abortedJobs,abortedWork,relaxWork,relaxTightens,relaxReduction,scans,reuse,uniqueNodes:nodes.size,eagerComplete,stopped,trace,...C.info()};
  }
  function ridge(rows,target){
    const d=rows[0].x.length,Z=seq(d).map(i=>seq(d+1).map(j=>i===j?0.10:0));
    for(const r of rows)for(let i=0;i<d;i++){for(let j=0;j<d;j++)Z[i][j]+=r.x[i]*r.x[j];Z[i][d]+=r.x[i]*r[target];}
    for(let i=0;i<d;i++){let p=i;for(let j=i+1;j<d;j++)if(Math.abs(Z[j][i])>Math.abs(Z[p][i]))p=j;[Z[i],Z[p]]=[Z[p],Z[i]];const v=Z[i][i];A(Math.abs(v)>1e-12,"ridge singular");for(let j=i;j<=d;j++)Z[i][j]/=v;
      for(let k=0;k<d;k++)if(k!==i){const v=Z[k][i];for(let j=i;j<=d;j++)Z[k][j]-=v*Z[i][j];}}
    return Z.map(x=>x[d]);
  }
  function fitCost(rows){
    const d=rows[0].x.length,w=Array(d).fill(0),predict=r=>sum(r.x.map((x,i)=>x*w[i]));w[0]=Math.log(Math.max(1,mean(rows.map(r=>r.work))));
    // Censored squared loss: interrupted jobs contribute only if prediction
    // falls below log of work already spent. Full observations use squared error.
    for(let epoch=0;epoch<500;epoch++){const g=Array(d).fill(0);for(const r of rows){const error=predict(r)-Math.log(Math.max(1,r.work));if(r.censored&&error>=0)continue;for(let i=0;i<d;i++)g[i]+=2*error*r.x[i]/rows.length;}for(let i=0;i<d;i++)w[i]-=.01*(g[i]+.002*w[i]);}
    const loss=mean(rows.map(r=>{const e=predict(r)-Math.log(Math.max(1,r.work));return r.censored?Math.min(0,e)**2:e**2;}));return{weights:w,loss};
  }
  const trainN=32*scale,evalN=24*scale,rows=[];
  for(let i=0;i<trainN;i++){const r=make("train",i),a=run(r,100000000,"training",null,true);A(a.length<=4,"presentation cap");rows.push(...a);}
  const cost=fitCost(rows),gain=ridge(rows,"gain"),gap=ridge(rows,"gap"),models={cost:cost.weights,gain,gap};
  const modelFit={costLoss:cost.loss,gainMSE:mean(rows.map(r=>(sum(r.x.map((x,i)=>x*gain[i]))-r.gain)**2)),gapMSE:mean(rows.map(r=>(sum(r.x.map((x,i)=>x*gap[i]))-r.gap)**2)),gainConstantMSE:mean(rows.map(r=>(r.gain-mean(rows.map(x=>x.gain)))**2)),gapConstantMSE:mean(rows.map(r=>(r.gap-mean(rows.map(x=>x.gap)))**2)),positiveGain:rows.filter(r=>r.gain>0).length,positiveGap:rows.filter(r=>r.gap>0).length,censored:rows.filter(r=>r.censored).length,boundSamples:rows.filter(r=>r.type==="bound").length};
  const cal=seq(8).map(i=>make("calibration",i)),calibration=[];let primary=262144,qualified=false;
  for(const b of [8192,16384,32768,65536,131072,262144]){
    const a=cal.map(r=>{const x=run(r,b,"route");return{id:r.id,actionJobs:x.actionJobs,utility:x.utility,work:x.work};}),v=a.map(x=>x.actionJobs).sort((a,b)=>a-b),med=(v[3]+v[4])/2,positive=v.filter(x=>x>0).length;
    calibration.push({budget:b,medianActionJobs:med,rootsWithAction:positive,roots:a});if(med>=8&&positive>=6){primary=b;qualified=true;break;}
  }
  const budgets=[primary/2,primary],rootResults=[];
  for(const split of ["ordinary","ood"])for(let i=0;i<evalN;i++){
    const r=make(split,i),truth=oracle(r,i<4),atBudget=[];
    for(const b of budgets){
      const arms={};for(const mode of names){const x=run(r,b,mode,models);A(x.work<=b&&x.lower<=truth.utility+1e-12&&x.upper>=truth.utility-1e-12,"root work/bound");checks.rootBounds++;arms[mode]={...x,oracleGap:truth.utility-x.utility,oracleTie:x.raw===truth.raw,closed:x.certifiedGap===0};}
      atBudget.push({budget:b,arms});
    }rootResults.push({id:r.id,split,family:r.family,credits:r.budget,oracle:truth,atBudget});
  }
  const round=x=>Math.round(x*1e6)/1e6;
  function ci(a){const m=mean(a),sd=Math.sqrt(sum(a.map(x=>(x-m)**2))/Math.max(1,a.length-1));return[round(m),round(m-1.96*sd/Math.sqrt(a.length)),round(m+1.96*sd/Math.sqrt(a.length))];}
  const fields=["utility","oracleGap","certifiedGap","work","actionJobs","boundJobs","abortedJobs","abortedWork","relaxWork","relaxTightens","relaxReduction","uniqueNodes","statStates","legalStates","proposalStates","rolloutStates","indexEntries","cost","queries","falsePublication","fullPublication"];
  const aggregate=["ordinary","ood"].map(split=>{
    const rr=rootResults.filter(r=>r.split===split),atBudget=[];
    for(let j=0;j<budgets.length;j++){
      const keys=names.filter(k=>!k.startsWith("random")).concat("random"),arms={};
      function val(r,k,f){return k==="random"?mean(["random19","random37","random73"].map(z=>Number(r.atBudget[j].arms[z][f]))):Number(r.atBudget[j].arms[k][f]);}
      for(const k of keys)arms[k]={...Object.fromEntries(fields.map(f=>[f,round(mean(rr.map(r=>val(r,k,f))))])),oracleTies:round(sum(rr.map(r=>val(r,k,"oracleTie")))),closed:round(sum(rr.map(r=>val(r,k,"closed")))),eagerCompleted:round(sum(rr.map(r=>val(r,k,"eagerComplete"))))};
      const comparisons={};for(const t of ["route","constructive_first","learned","certification"])for(const p of keys.filter(k=>k!==t&&!(t==="route"&&k!=="eager_route")&&!(t==="constructive_first"&&!["eager_route","eager_info","route","info","mass_gap"].includes(k))))comparisons[`${t}_vs_${p}`]={utility:ci(rr.map(r=>val(r,t,"utility")-val(r,p,"utility"))),gapReduction:ci(rr.map(r=>val(r,p,"certifiedGap")-val(r,t,"certifiedGap")))};
      atBudget.push({budget:budgets[j],arms,comparisons});
    }return{split,roots:rr.length,oracleUtility:round(mean(rr.map(r=>r.oracle.utility))),oracleStates:sum(rr.map(r=>r.oracle.states)),atBudget};
  });
  return{receipt:{schema:"MTR-02G-public-development-v1",seed,scale,trainRoots:trainN,calibrationRoots:8,ordinaryRoots:evalN,oodRoots:evalN,uniqueRoots:signatures.size,trainingRows:rows.length,features:gain.length,modelFit,checks,calibration,primary,qualified,budgets,names,aggregate,rootResults},models,rows};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
 const args=process.argv.slice(2).filter(x=>!x.startsWith("--")).map(Number),start=process.hrtime.bigint(),cpu=process.cpuUsage(),r=mtr02g(...args);
 if(process.argv.includes("--profile"))console.error(JSON.stringify({scope:"whole campaign; includes training/oracle/replay",elapsedMs:Number(process.hrtime.bigint()-start)/1e6,cpuMicros:process.cpuUsage(cpu),maxRSSKiB:process.resourceUsage().maxRSS}));
 console.log(JSON.stringify(process.argv.includes("--models")?r.models:process.argv.includes("--rows")?r.rows:r.receipt,null,2));
}
