// MTR-02I paired option intervention and route-anchored policy. Derived from pinned MTR-02H.
import {pathToFileURL} from "node:url";
export function mtr02i(seed=20261127,scale=1){
  const A=(x,m)=>{if(!x)throw Error(m);},seq=n=>Array.from({length:n},(_,i)=>i),sum=a=>a.reduce((x,y)=>x+y,0),mean=a=>sum(a)/a.length;
  A(Number.isSafeInteger(seed)&&Number.isInteger(scale)&&scale>=1&&scale<=2,"bad configuration");
  const N=128,ALL=(1n<<128n)-1n,BITS=seq(N).map(i=>1n<<BigInt(i)),BI=new Map(BITS.map((b,i)=>[b,i])),STOP=Symbol("work"),names=["route","route_first","bound_first","info","optimistic","random19","random37","random73","constant","permuted","learned","certification"],signatures=new Set();
  const checks={witness:0,nonlinear:0,legalLists:0,childMasks:0,oracleReplays:0,policyReplays:0,rootBounds:0,backups:0,massDifferential:0,proposalDifferential:0,cheapBoundOracle:0,freeBoundOracle:0};
  const primitiveAudit=[],pairedStates=[];checks.prefixComparisons=0;checks.pureBoundChecks=0;
  function rng(k){let s=2166136261;for(const c of k)s=Math.imul(s^c.charCodeAt(0),16777619);s||=1;return()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return(s>>>0)/4294967296;};}
  function shuffle(a,r){a=[...a];for(let i=a.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
  function make(split,i){
    const id=`${seed}/mtr02i/${split}/${i}`,r=rng(id),ri=n=>Math.floor(r()*n),family=split==="ood"?"cross_gate_cubic":i%2?"coupled_payload":"quadratic_tree",budget=7+i%3,perm=shuffle(seq(7),r),flip=ri(128);
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
    return{id,split,family,budget,worlds,w,actions,rewards,kinds,sizes,label,total:sum(w),minWeight:Math.min(...w),minCost:Math.min(...actions.map(a=>a.cost))};
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
  function cheapBound(r,s,ls,stats,mass,M){
    const charge=(k,n=1)=>M?.tick(k,n),t=stats(s.m);let U=t.best.v;
    for(const a of ls){
      charge("cheapBound");let v=-a.cost*t.mass;
      for(const c of a.cs){
        charge("cheapBound");const z=mass(c.s.m),W=z.mass;
        if(z.n===1){v+=36*W;continue;}
        let h=Math.max(24*W,36*W-90*r.minWeight);
        if(c.s.b>=r.minCost)h=Math.max(h,(36-r.minCost)*W);
        v+=h;
      }
      U=Math.max(U,v);
    }return U;
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
    if(audit)for(const {s,ls} of shuffle(states,rng(r.id+"/primitive-audit")).slice(0,64)){
      const t=stats(s.m),u=Math.max(t.best.v,...ls.map(a=>(36-a.cost)*t.mass)),v=solve(s).v;
      const mf=meter(10000000),ms=meter(10000000),F=context(r,mf,false),S=context(r,ms,true);
      const a=F.proposal(s,1),b=S.proposal(s,1);
      A((a?.i??-1)===(b?.i??-1),"information proposal differential");checks.proposalDifferential++;
      const fullProposalWork=mf.used,splitProposalWork=ms.used;
      for(const m of new Set([s.m,...ls.flatMap(a=>a.cs.map(c=>c.s.m))])){
        const d=S.mass(m),q=stats(m);A(d.mass===q.mass&&d.n===q.n,"mass differential");checks.massDifferential++;
        const x=S.stats(m),y=F.stats(m);A(x.mass===q.mass&&x.n===q.n&&x.best.v===q.best.v&&x.best.kind===q.best.kind&&x.best.claim===q.best.claim&&y.best.v===x.best.v,"terminal differential");
      }
      const mc=meter(10000000),mr=meter(10000000),C=context(r,mc,true),R=context(r,mr,true);
      const h=cheapBound(r,s,C.ls(s),C.stats,C.mass,mc),g=relax(r,s,R.ls(s),R.stats,mr);
      A(v<=h&&h<=u&&v<=g&&g<=u,"inadmissible bound");checks.cheapBoundOracle++;checks.freeBoundOracle++;
      primitiveAudit.push({root:r.id,split:r.split,mask:s.m.toString(16),credits:s.b,fullProposalWork,splitProposalWork,cheapWork:mc.used,freeWork:mr.used,original:u,cheap:h,free:g,oracle:v,mass:t.mass});
    }
    return{...out,states:memo.size};
  }
  function context(r,M,splitMass=true){
    const sm=new Map(),mm=new Map(),lm=new Map(),pm=[new Map(),new Map(),new Map()];let idx=null;
    function mass(m){
      if(!splitMass){const z=stats(m);return{mass:z.mass,n:z.n};}
      M.tick("lookup");if(mm.has(m))return mm.get(m);
      let w=0,n=0;for(let z=m;z;z&=z-1n){M.tick("massWorld");w+=r.w[BI.get(z&-z)];n++;}
      const t={mass:w,n};M.tick("insert");mm.set(m,t);return t;
    }
    function stats(m){
      M.tick("lookup");if(sm.has(m))return sm.get(m);
      const bins=r.kinds.map(()=>new Map()),mx=[0,0,0,0],arg=[0,0,0,0];let cached=null,w0=0,n=0;
      if(splitMass){M.tick("lookup");cached=mm.get(m)||null;}
      for(let z=m;z;z&=z-1n){
        M.tick("world",cached?4:5);const h=BI.get(z&-z),w=r.w[h],x=r.worlds[h];if(!cached){w0+=w;n++;}
        for(let k=0;k<4;k++){const j=r.label(r.kinds[k],x,h),v=(bins[k].get(j)||0)+w;bins[k].set(j,v);if(v>mx[k]||v===mx[k]&&j<arg[k]){mx[k]=v;arg[k]=j;}}
      }
      if(cached){w0=cached.mass;n=cached.n;}
      let best={v:0,i:-1,kind:"abstain",claim:-1};for(let k=0;k<4;k++){M.tick("claim");const kind=r.kinds[k],[g,b]=r.rewards[kind],v=(g+b)*mx[k]-b*w0;if(v>best.v)best={v,i:-1,kind,claim:arg[k]};}
      const t={mass:w0,n,best};if(splitMass&&!cached){M.tick("insert");mm.set(m,{mass:w0,n});}M.tick("insert");sm.set(m,t);return t;
    }
    function ls(s){
      M.tick("lookup");const k=key(s);if(lm.has(k))return lm.get(k);if(!idx)idx=index(r,M);const a=s.b>0?legal(r,s,indexed(s,idx,M),M):[];M.tick("insert");lm.set(k,a);return a;
    }
    function proposal(s,k){
      M.tick("lookup");const id=key(s),memo=pm[k];if(memo.has(id))return memo.get(id);const t=k===1?mass(s.m):stats(s.m);let best=null,score=-Infinity;
      for(const a of ls(s)){
        M.tick("proposal");const kind=r.actions[a.i].kind;let v;
        if(k===0)v=Number(!["global","cross"].includes(kind))+Math.log2(a.cs.length)/a.cost;
        else if(k===1){let mx=0;for(const c of a.cs){M.tick("proposal");mx=Math.max(mx,mass(c.s.m).mass);}v=(t.mass-mx)/t.mass/a.cost;}
        else{let q=-a.cost*t.mass;for(const c of a.cs){M.tick("proposal");q+=stats(c.s.m).best.v;}v=q/t.mass;}
        if(v>score){score=v;best=a;}
      }
      if(k===2&&score<=t.best.v/t.mass)best=null;
      M.tick("insert");memo.set(id,best);return best;
    }
    return{stats,mass,ls,proposal,cacheSnapshot:()=>({stat:[...sm],mass:[...mm],legal:[...lm],proposals:pm.map(x=>[...x]),index:idx}),info:()=>({statStates:sm.size,massStates:mm.size,legalStates:lm.size,proposalStates:sum(pm.map(x=>x.size)),indexEntries:idx?.entries||0})};
  }
  function run(r,cap,mode,models=null,intervention=null){
    const M=meter(cap),C=context(r,M,true),nodes=new Map(),rr=rng(r.id+"/"+mode),initial={m:ALL,b:r.budget},labels=[],trace=[];
    let root=null,fallback=null,fallbackU=36*r.total,stopped=false,actionJobs=0,boundJobs=0,abortedJobs=0,abortedWork=0,relaxWork=0,relaxTightens=0,relaxReduction=0,scans=0,reuse=0,acquisitionWork=0,acquisitionCalls=0,optionCount=0,completedOptions=0,censoredOptions=0,optionWork=0,routeStalls=0,alternative=0,boundBlockedAt=-1;
    const history=seq(5).map(()=>({n:0,gain:0,gap:0,work:0})),options=[];
    const snapshot=()=>({work:M.used,lower:(root?.L??fallback?.v??0)/r.total,upper:(root?.U??fallbackU)/r.total,actionJobs,boundJobs,completedOptions});
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
        A(relaxed===null,"repeated relaxation");const v=cheapBound(r,n.s,n.ls,C.stats,C.mass,M);A(v>=n.L,"relax below incumbent");relaxed=v;
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
        if(k===4&&n.relaxed===null){const score=gap/(1+n.t.n);if(score>rbest){rbest=score;rel={n,type:"bound",stream:3,gap};}}
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
    function acquire(k){
      const start=M.used;acquisitionCalls++;
      try{
        const a=bestJob([root],k);
        if(k===4)return a.bound;
        return a.action;
      }finally{acquisitionWork+=M.used-start;}
    }
    function getJob(k){
      const a=acquire(k);return a||(k===3||k===4?null:acquire(3));
    }
    function optionFeatures(k,quantum){
      // Interaction block for each candidate option; no oracle or cache peek.
      M.tick("feature",70);const q=Math.min(1,quantum/4096),h=history[0];
      const f=[1,root.L/r.total/36,root.U/r.total/36,(root.U-root.L)/r.total/36,root.s.b/9,q,Math.min(1,actionJobs/48),Math.min(1,boundJobs/16),Math.min(1,nodes.size/128),Math.min(1,routeStalls/4),Math.min(1,optionCount/8),h.n?h.gain/h.n/36:0,Math.min(1,root.ls.length/15),Math.min(1,acquisitionCalls/100)];
      return seq(5*f.length).map(i=>Math.floor(i/f.length)===k?f[i%f.length]:0);
    }
    function allowed(){return boundBlockedAt===actionJobs?[0,1,2,3]:[0,1,2,3,4];}
    function chooseOption(quantum){
      M.tick("priority");const ids=allowed();
      if(mode==="route")return 0;
      if(mode==="info")return 1;
      if(mode==="optimistic")return 3;
      if(mode==="bound_first")return optionCount===0?4:0;
      if(mode==="route_first"){if(routeStalls<2)return 0;const k=[1,2,3,4][alternative++%4];routeStalls=0;return ids.includes(k)?k:0;}
      if(mode.startsWith("random"))return ids[Math.floor(rr()*ids.length)];
      let best=0,score=.25;
      for(let k=0;k<5;k++){
        const x=optionFeatures(k,quantum);M.tick("priority",x.length);
        let v=0;if(k!==0){
          if(mode==="constant")v=models.constant[k];
          else v=sum(x.map((z,i)=>z*models[mode==="learned"?"gain":mode==="certification"?"gap":"permuted"][i]));
        }
        if(ids.includes(k)&&v>score){score=v;best=k;}
      }return best;
    }
    function fingerprint(){
      return JSON.stringify({work:M.used,counts:M.c,root:root.k,history,options,routeStalls,actionJobs,boundJobs,acquisitionCalls,boundBlockedAt,
        nodes:[...nodes].map(([k,n])=>({k,s:n.s,t:n.t,ls:n.ls,L:n.L,U:n.U,ua:n.ua,la:n.la?.i??null,relaxed:n.relaxed,
          arcs:[...n.arcs].map(([i,a])=>[i,a.cost,a.cs.map(c=>[c.o,c.n.k])]),parents:[...n.parents].map(p=>p.k)})),cache:C.cacheSnapshot()},(_,v)=>typeof v==="bigint"?v.toString(16):v);
    }
    function digest(s){let h=2166136261;for(let i=0;i<s.length;i++)h=Math.imul(h^s.charCodeAt(i),16777619);return(h>>>0).toString(16).padStart(8,"0");}
    function runOption(k,quantum,x=null){
      const l=root.L,g=root.U-root.L,start=M.used,a0=actionJobs,b0=boundJobs;let censored=false,jobs=0,ended=false,pending=false,jobStart=0;
      optionCount++;M.jobCap=Math.min(cap,start+quantum);
      try{
        for(let t=0;t<4&&root.L<root.U;t++){
          const j=getJob(k);if(!j){ended=true;break;}pending=true;jobStart=M.used;refine(j);pending=false;jobs++;
        }
      }catch(e){if(e!==STOP)throw e;censored=true;if(pending){abortedJobs++;abortedWork+=M.used-jobStart;}}
      finally{M.jobCap=cap;}
      const work=M.used-start,gain=(root.L-l)/r.total,gap=(g-(root.U-root.L))/r.total;
      A(work>0&&gain>=0&&gap>=0,"option receipt");
      if(censored)censoredOptions++;else if(jobs===4||root.L>=root.U||ended)completedOptions++;
      optionWork+=work;
      // Only public outcomes of this selected option enter the history.
      if(k===4){A(actionJobs===a0&&gain===0,"pure bound credit");checks.pureBoundChecks++;if(ended||jobs===0)boundBlockedAt=actionJobs;}
      const h=history[k];h.n++;h.work+=work;h.gain+=gain;h.gap+=gap;
      if(k===0)routeStalls=gain>0?0:routeStalls+1;
      const row={root:r.id,stream:k,quantum,work,censored,gain,gap,jobs,actionJobs:actionJobs-a0,boundJobs:boundJobs-b0,initialLower:l/r.total,initialGap:g/r.total};
      options.push(row);return x?{...row,x}:row;
    }

    function read(p){return p.s?(p.la?{i:p.la.i,cs:p.la.cs.map(c=>({o:c.o,p:c.n}))}:p.t.best):p;}
    try{
      initialize();trace.push(snapshot());
      if(intervention){
        if(intervention.prefix&&root.L<root.U)runOption(0,4096);
        const prefixState=fingerprint(),prefixWork=M.used,prefixLower=root.L/r.total,prefixGap=(root.U-root.L)/r.total;
        const x=optionFeatures(intervention.k,4096);
        // A continuation has a fixed work horizon including public feature acquisition.
        // Work for fingerprinting is evaluator-only, as is final replay.
        const end=prefixWork+8192;let first=null;
        if(root.L<root.U)first=runOption(intervention.k,Math.min(4096,end-M.used));
        while(M.used<end&&root.L<root.U)runOption(0,Math.min(4096,end-M.used));
        const out=replay(r,root,read);A(out.raw===root.L,"intervention replay");checks.policyReplays++;
        return{prefixState,prefixDigest:digest(prefixState),prefixWork,prefixLower,prefixGap,x,stream:intervention.k,
          first:first||{gain:0,gap:0,work:0,jobs:0,actionJobs:0,boundJobs:0,censored:false},delayedGain:out.utility-prefixLower,delayedGap:prefixGap-(root.U-root.L)/r.total,
          continuationWork:M.used-prefixWork,continuationJobs:sum(options.slice(intervention.prefix?1:0).map(o=>o.jobs)),censoredOptions:options.slice(intervention.prefix?1:0).filter(o=>o.censored).length,
          utility:out.utility,falsePublication:out.falsePublication};
      }
      let nextMark=cap/4;
      while(root.L<root.U&&M.left()>0){
        const q=Math.min(4096,M.left()),k=chooseOption(q);if(M.left()===0)break;
        const o=runOption(k,Math.min(4096,M.left()));
        if(M.used>=nextMark){trace.push(snapshot());while(nextMark<=M.used)nextMark+=cap/4;}
        if(o.jobs===0&&!o.censored&&k!==4)A(root.L===root.U,"no action job at open root");
      }
    }catch(e){if(e!==STOP)throw e;stopped=true;}
    A(fallback,"no terminal");
    const p=root||fallback,L=root?.L??fallback.v,U=root?.U??fallbackU,out=replay(r,p,read);
    A(out.raw===L,"partial incumbent replay");checks.policyReplays++;
    trace.push({work:M.used,lower:L/r.total,upper:U/r.total,actionJobs,boundJobs,completedOptions});
    return{...out,lower:L/r.total,upper:U/r.total,certifiedGap:(U-L)/r.total,work:M.used,counts:M.c,actionJobs,boundJobs,abortedJobs,abortedWork,relaxWork,relaxTightens,relaxReduction,scans,reuse,uniqueNodes:nodes.size,stopped,trace,acquisitionWork,acquisitionCalls,optionCount,completedOptions,censoredOptions,optionWork,options,boundBlockedAt,...C.info()};
  }
  function ridge(rows,target){
    const d=rows[0].x.length,Z=seq(d).map(i=>seq(d+1).map(j=>i===j?0.10:0));
    for(const r of rows)for(let i=0;i<d;i++){for(let j=0;j<d;j++)Z[i][j]+=r.x[i]*r.x[j];Z[i][d]+=r.x[i]*r[target];}
    for(let i=0;i<d;i++){let p=i;for(let j=i+1;j<d;j++)if(Math.abs(Z[j][i])>Math.abs(Z[p][i]))p=j;[Z[i],Z[p]]=[Z[p],Z[i]];const v=Z[i][i];A(Math.abs(v)>1e-12,"ridge singular");for(let j=i;j<=d;j++)Z[i][j]/=v;
      for(let k=0;k<d;k++)if(k!==i){const v=Z[k][i];for(let j=i;j<=d;j++)Z[k][j]-=v*Z[i][j];}}
    return Z.map(x=>x[d]);
  }
  const trainN=32*scale,evalN=24*scale,rows=[];
  for(let i=0;i<trainN;i++){
    const r=make("train",i);
    for(let prefix=0;prefix<2;prefix++){
      const a=seq(5).map(k=>run(r,100000000,"train",null,{prefix,k})),base=a[0];
      for(const o of a.slice(1)){A(o.prefixState===base.prefixState,"paired prefix graph/cache mismatch");checks.prefixComparisons++;}
      pairedStates.push({root:r.id,prefix,digest:base.prefixDigest,serializedBytes:base.prefixState.length,work:base.prefixWork,lower:base.prefixLower,gap:base.prefixGap});
      for(const {prefixState,...o} of a){
        A(o.continuationWork<=8192,"paired horizon");A(o.stream!==4||o.first.gain===0&&o.first.actionJobs===0,"pure bound intervention");
        rows.push({root:r.id,prefix,...o,gain:o.delayedGain-base.delayedGain,gap:o.delayedGap-base.delayedGap});
      }
    }
  }
  const gain=ridge(rows,"gain"),gap=ridge(rows,"gap"),constant=seq(5).map(k=>mean(rows.filter(o=>o.stream===k).map(o=>o.gain)));
  // Shuffle residuals within option ID: preserves option means while breaking
  // state-dependent residual correspondence. No evaluation data or labels.
  const permRows=rows.map(o=>({...o}));
  for(let k=0;k<5;k++){
    const ids=seq(rows.length).filter(i=>rows[i].stream===k),v=shuffle(ids.map(i=>rows[i].gain),rng(seed+"/mtr02i/permute/"+k));
    ids.forEach((i,j)=>{permRows[i].gain=v[j];});
  }
  const permuted=ridge(permRows,"gain"),models={gain,gap,constant,permuted};
  const mse=(w,k)=>mean(rows.map(r=>(sum(r.x.map((x,i)=>x*w[i]))-r[k])**2));
  const modelFit={gainMSE:mse(gain,"gain"),gapMSE:mse(gap,"gap"),permutedMSE:mse(permuted,"gain"),constantGainMSE:mean(rows.map(r=>(r.gain-constant[r.stream])**2)),gainConstantMSE:mean(rows.map(r=>(r.gain-mean(rows.map(x=>x.gain)))**2)),gapConstantMSE:mean(rows.map(r=>(r.gap-mean(rows.map(x=>x.gap)))**2)),positiveGain:rows.filter(r=>r.gain>0).length,positiveGap:rows.filter(r=>r.gap>0).length,censored:rows.filter(r=>r.first.censored).length,pairedDecisionSets:pairedStates.length,rows:rows.length};
  const cal=seq(8).map(i=>make("calibration",i)),calibration=[];let primary=131072,qualified=false;
  for(const b of [8192,16384,32768,65536,131072]){
    const a=cal.map(r=>{const x=run(r,b,"route",models);return{id:r.id,actionJobs:x.actionJobs,completedOptions:x.completedOptions,utility:x.utility,work:x.work};}),v=a.map(x=>x.actionJobs).sort((a,b)=>a-b),med=(v[3]+v[4])/2,positive=a.filter(x=>x.completedOptions>=2).length;
    calibration.push({budget:b,medianActionJobs:med,rootsWithTwoOptions:positive,roots:a});
    if(med>=8&&positive>=6){primary=b;qualified=true;break;}
  }
  const budgets=[primary/2,primary],rootResults=[];
  for(const split of ["ordinary","ood"])for(let i=0;i<evalN;i++){
    const r=make(split,i),truth=oracle(r,false),atBudget=[];
    for(const b of budgets){
      const arms={};for(const mode of names){const x=run(r,b,mode,models);A(x.work<=b&&x.lower<=truth.utility+1e-12&&x.upper>=truth.utility-1e-12,"root work/bound");checks.rootBounds++;arms[mode]={...x,oracleGap:truth.utility-x.utility,oracleTie:x.raw===truth.raw,closed:x.certifiedGap===0};}
      atBudget.push({budget:b,arms});
    }rootResults.push({id:r.id,split,family:r.family,credits:r.budget,oracle:truth,atBudget});
  }
  const round=x=>Math.round(x*1e6)/1e6;
  function ci(a){const m=mean(a),sd=Math.sqrt(sum(a.map(x=>(x-m)**2))/Math.max(1,a.length-1));return[round(m),round(m-1.96*sd/Math.sqrt(a.length)),round(m+1.96*sd/Math.sqrt(a.length))];}
  const fields=["utility","oracleGap","certifiedGap","work","actionJobs","boundJobs","abortedJobs","abortedWork","relaxWork","relaxTightens","relaxReduction","uniqueNodes","statStates","massStates","legalStates","proposalStates","indexEntries","cost","queries","falsePublication","fullPublication","acquisitionWork","acquisitionCalls","optionCount","completedOptions","censoredOptions","optionWork"];
  const aggregate=["ordinary","ood"].map(split=>{
    const rr=rootResults.filter(r=>r.split===split),atBudget=[];
    for(let j=0;j<budgets.length;j++){
      const keys=names.filter(k=>!k.startsWith("random")).concat("random"),arms={};
      function val(r,k,f){return k==="random"?mean(["random19","random37","random73"].map(z=>Number(r.atBudget[j].arms[z][f]))):Number(r.atBudget[j].arms[k][f]);}
      for(const k of keys)arms[k]={...Object.fromEntries(fields.map(f=>[f,round(mean(rr.map(r=>val(r,k,f))))])),oracleTies:round(sum(rr.map(r=>val(r,k,"oracleTie")))),closed:round(sum(rr.map(r=>val(r,k,"closed"))))};
      const comparisons={};for(const t of ["learned","certification","constant","permuted","bound_first"])for(const p of keys.filter(k=>k!==t&&!(t==="bound_first"&&k!=="route")))comparisons[`${t}_vs_${p}`]={utility:ci(rr.map(r=>val(r,t,"utility")-val(r,p,"utility"))),gapReduction:ci(rr.map(r=>val(r,p,"certifiedGap")-val(r,t,"certifiedGap")))};
      atBudget.push({budget:budgets[j],arms,comparisons});
    }return{split,roots:rr.length,oracleUtility:round(mean(rr.map(r=>r.oracle.utility))),oracleStates:sum(rr.map(r=>r.oracle.states)),atBudget};
  });
  return{receipt:{schema:"MTR-02I-public-development-v1",seed,scale,trainRoots:trainN,calibrationRoots:8,ordinaryRoots:evalN,oodRoots:evalN,uniqueRoots:signatures.size,trainingRows:rows.length,features:gain.length,modelFit,checks,calibration,primary,qualified,budgets,names,aggregate,rootResults,pairedStates},models,rows};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
 const args=process.argv.slice(2).filter(x=>!x.startsWith("--")).map(Number),r=mtr02i(...args);
 console.log(JSON.stringify(process.argv.includes("--models")?r.models:process.argv.includes("--rows")?r.rows:r.receipt));
}
