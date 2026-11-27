// MTR-02F indexed lazy search; public development only.
import {pathToFileURL} from "node:url";
export function mtr02f(seed=20261127,scale=1){
  const ok=(b,m)=>{if(!b)throw Error(m);},seq=n=>Array.from({length:n},(_,i)=>i),sum=a=>a.reduce((x,y)=>x+y,0),mean=a=>sum(a)/a.length;
  ok(Number.isSafeInteger(seed)&&Number.isInteger(scale)&&scale>=1&&scale<=2,"invalid configuration");
  const N=128,ALL=(1n<<128n)-1n,BITS=seq(N).map(i=>1n<<BigInt(i)),INDEX=new Map(BITS.map((b,i)=>[b,i])),stop=Symbol("work cap");
  const modes=["mass_gap","work_gap","random19","random37","random73","discovery","certification","eager","action_only","analytic"];
  const signatures=new Set(),counts={witnessChecks:0,nonlinearChecks:0,oracleReplays:0,policyReplays:0,rootBounds:0,backupChecks:0,legalComparisons:0,childComparisons:0,trainingJobs:0};
  function rng(k){let s=2166136261;for(const c of k)s=Math.imul(s^c.charCodeAt(0),16777619);s||=1;return()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return(s>>>0)/4294967296;};}
  function shuffle(a,r){a=[...a];for(let i=a.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
  function make(split,i){
    const id=`${seed}/mtr02f/${split}/${i}`,r=rng(id),ri=n=>Math.floor(r()*n),family=split==="ood"?"cubic":i%2?"quadratic_b":"quadratic_a",budget=7+i%3,perm=shuffle(seq(7),r),flip=ri(128);
    function witness(h){
      const b=perm.map(j=>(h^flip)>>j&1),p=b.slice(4);
      const s0=b[0]^(family==="cubic"?p[0]*p[1]*p[2]:family==="quadratic_a"?p[0]*p[1]:p[1]*p[2]),s1=b[1]^(s0*p[2]),s2=b[2]^(s1*p[0]),s3=b[3]^(family==="cubic"?s2*p[0]*p[1]:p[1]*p[2]);
      return{branch:8*s0+4*s1+2*s2+s3,payload:4*p[0]+2*p[1]+p[2],coarse:p[0]^(p[1]*p[2])};
    }
    const worlds=seq(N).map(witness),w=seq(N).map(h=>1+ri(4)+(i%4===0&&worlds[h].branch<8?2:0));
    ok(new Set(worlds.map(x=>8*x.branch+x.payload)).size===N,"encoding");
    ok(seq(7).some(j=>new Set(seq(N).map(h=>(worlds[h].branch>>3)^(worlds[h^(1<<j)].branch>>3))).size>1),"nonlinear");counts.nonlinearChecks++;
    const mask=f=>worlds.reduce((m,x,h)=>f(x,h)?m|BITS[h]:m,0n),actions=[];
    function add(kind,cost,l,p,observe){
      const gate=l<0?ALL:mask(x=>x.branch>>(4-l)===p),outcomes=worlds.map(observe),parts=new Map();
      for(let h=0;h<N;h++)parts.set(outcomes[h],(parts.get(outcomes[h])||0n)|BITS[h]);
      actions.push({kind,cost,l,p,gate,outcomes,parts:[...parts].sort((a,b)=>a[0]-b[0])});
    }
    for(let l=0;l<4;l++)for(let p=0;p<1<<l;p++){
      add("single",1+Number(ri(4)===0)+Number(split==="ood"&&l===2&&p%3===0&&i%3===0),l,p,x=>x.branch>>(3-l)&1);
      if(l<3)add("pair",1+ri(3)+Number(split==="ood"&&p%2===0&&i%4===0),l,p,x=>x.branch>>(2-l)&3);
    }
    for(let p=0;p<16;p++){
      add("full",2+ri(3)+Number(split==="ood"&&p%3===0),4,p,x=>x.payload);
      add("partial",1+ri(2),4,p,x=>x.payload>>1);
    }
    add("global",1,-1,0,x=>x.coarse);add("global",1+ri(2),-1,0,x=>(x.branch>>3)^(x.payload&1));
    add("global",2+Number(split==="ood"&&i%3===0),-1,0,x=>x.payload>>1);add("global",1+ri(2),-1,0,x=>((x.branch>>3)^(x.branch>>1)^(x.payload>>2))&1);
    ok(actions.length===58,"action bank");
    const reward={full:[36,54],partial:[24,36],branch:[15,23],coarse:[8,12]},groups={full:128,partial:64,branch:16,coarse:2};
    const pred=(kind,x,h)=>kind==="full"?h:kind==="partial"?4*x.branch+(x.payload>>1):kind==="branch"?x.branch:x.coarse;
    const claims=[];
    for(const [kind,n] of Object.entries(groups))for(let j=0;j<n;j++){
      const m=mask((x,h)=>pred(kind,x,h)===j);
      for(let h=0;h<N;h++){ok(Boolean(m&BITS[h])===(pred(kind,witness(h),h)===j),"claim check");counts.witnessChecks++;}
      claims.push({kind,j,m});
    }
    const signature=JSON.stringify([family,budget,perm,flip,w,actions.map(a=>[a.cost,a.gate.toString(),a.outcomes])]);ok(!signatures.has(signature),"duplicate root");signatures.add(signature);
    return{id,split,family,budget,worlds,w,actions,reward,claims,pred,total:sum(w)};
  }
  const key=s=>`${s.m.toString(16)}/${s.b}`;
  function meter(limit){return{limit,used:0,counts:{},tick(k,n=1){ok(Number.isInteger(n)&&n>=0,"bad charge");const z=Math.min(n,this.limit-this.used);this.used+=z;this.counts[k]=(this.counts[k]||0)+z;if(z!==n)throw stop;}};}
  function index(r,M){
    const charge=(k,n=1)=>M?.tick(k,n),table=[],masks=[],global=[];
    for(let l=0;l<=4;l++){table[l]=[];masks[l]=[];for(let p=0;p<1<<l;p++){charge("indexBuild",2);table[l][p]=[];masks[l][p]=0n;}}
    for(let j=0;j<r.actions.length;j++){
      charge("indexBuild",3);const a=r.actions[j];if(a.l<0)global.push(j);else{table[a.l][a.p].push(j);masks[a.l][a.p]=a.gate;}
    }
    return{table,masks,global,logicalEntries:62+r.actions.length};
  }
  function candidates(s,idx,M){
    const charge=(k,n=1)=>M?.tick(k,n),out=[];let p=0;
    for(let l=0;l<=4;l++){
      charge("indexRead");for(const j of idx.table[l][p]){charge("indexEmit");out.push(j);}
      if(l===4)break;charge("prefixTest");const z=s.m&idx.masks[l+1][2*p];
      if(z===s.m)p*=2;else if(z===0n)p=2*p+1;else break;
    }
    charge("indexRead");for(const j of idx.global){charge("indexEmit");out.push(j);}return out;
  }
  function legalScan(r,s,ids,M){
    const out=[],charge=(k,n=1)=>M?.tick(k,n);
    if(s.b>0)for(const i of ids){
      charge("legal");const a=r.actions[i];if(a.cost>s.b||(s.m&a.gate)!==s.m)continue;
      const cs=[];for(const [o,p] of a.parts){charge("partition");const m=s.m&p;if(m)cs.push({o,s:{m,b:s.b-a.cost}});}
      if(cs.length>1)out.push({i,cost:a.cost,cs});
    }return out;
  }
  function context(r,M){
    const stat=new Map(),leg=new Map(),roll=[new Map(),new Map()],baseMemo=new Map();let idx=null,baseEvaluations=0,rolloutEvaluations=0;
    function stats(m){
      M.tick("lookup");if(stat.has(m))return stat.get(m);const kinds=["full","partial","branch","coarse"],t=[new Map(),new Map(),new Map(),new Map()],mx=[0,0,0,0],arg=[0,0,0,0];let mass=0,n=0;
      for(let z=m;z;z&=z-1n){
        M.tick("world",5);const h=INDEX.get(z&-z),x=r.worlds[h],w=r.w[h],ids=[h,4*x.branch+(x.payload>>1),x.branch,x.coarse];mass+=w;n++;
        for(let j=0;j<4;j++){const id=ids[j],v=(t[j].get(id)||0)+w;t[j].set(id,v);if(v>mx[j]||v===mx[j]&&id<arg[j]){mx[j]=v;arg[j]=id;}}
      }
      let best={v:0,i:-1,kind:"abstain",claim:-1};
      for(let j=0;j<4;j++){M.tick("claim");const kind=kinds[j],[g,b]=r.reward[kind],v=(g+b)*mx[j]-b*mass;if(v>best.v)best={v,i:-1,kind,claim:arg[j]};}
      const z={mass,n,best};stat.set(m,z);return z;
    }
    function legal(s){
      M.tick("lookup");const k=key(s);if(leg.has(k))return leg.get(k);if(!idx)idx=index(r,M);
      const z=s.b>0?legalScan(r,s,candidates(s,idx,M),M):[];leg.set(k,z);return z;
    }
    function actionPlan(s,a,choose){
      M.tick("plan");let v=-a.cost*stats(s.m).mass;const cs=[];
      for(const c of a.cs){M.tick("plan");const p=choose(c.s);v+=p.v;cs.push({o:c.o,p});}return{v,i:a.i,cs};
    }
    function rollout(s,k){
      M.tick("lookup");const id=key(s),memo=roll[k];if(memo.has(id))return memo.get(id);M.tick("constructive");rolloutEvaluations++;
      const t=stats(s.m),ls=legal(s);let p=t.best,chosen=null,score=-Infinity;
      for(const a of ls){
        M.tick("constructive");let mx=0;
        for(const c of a.cs){M.tick("constructive");mx=Math.max(mx,stats(c.s.m).mass);}
        const info=(t.mass-mx)/t.mass/a.cost,route=Number(r.actions[a.i].kind!=="global")+Math.log2(a.cs.length)/a.cost;
        const v=k===0?route+0.01*info:info+0.0001*route;if(v>score){score=v;chosen=a;}
      }
      if(chosen){const q=actionPlan(s,chosen,z=>rollout(z,k));if(q.v>p.v)p=q;}memo.set(id,p);return p;
    }
    function base(s,progress=null){
      M.tick("lookup");const id=key(s);if(baseMemo.has(id)){const p=baseMemo.get(id);if(progress)progress(p);return p;}M.tick("library");baseEvaluations++;
      let p=stats(s.m).best;if(progress)progress(p);
      for(const a of legal(s)){M.tick("library");const q=actionPlan(s,a,z=>stats(z.m).best);if(q.v>p.v){p=q;if(progress)progress(p);}}
      for(let k=0;k<2;k++){const q=rollout(s,k);if(q.v>p.v){p=q;if(progress)progress(p);}}
      baseMemo.set(id,p);return p;
    }
    return{stats,legal,actionPlan,rollout,base,info:()=>({statStates:stat.size,legalStates:leg.size,rolloutStates:sum(roll.map(m=>m.size)),baseStates:baseMemo.size,baseEvaluations,rolloutEvaluations,indexEntries:idx?.logicalEntries||0})};
  }
  function validateTerminal(r,p,h){
    if(p.kind==="abstain")return{reward:0,published:0,wrong:0,full:0};
    const yes=r.pred(p.kind,r.worlds[h],h)===p.claim,[g,b]=r.reward[p.kind];
    return{reward:yes?g:-b,published:1,wrong:Number(!yes),full:Number(p.kind==="full")};
  }
  function replay(r,first,step){
    let value=0,cost=0,queries=0,pub=0,wrong=0,full=0;
    for(let h=0;h<N;h++){
      let s={m:ALL,b:r.budget},p=first,spent=0,q=0;
      while(true){
        const d=step(p),a=d.i<0?null:r.actions[d.i];
        if(!a){const x=validateTerminal(r,d,h),w=r.w[h];value+=w*(x.reward-spent);cost+=w*spent;queries+=w*q;pub+=w*x.published;wrong+=w*x.wrong;full+=w*x.full;break;}
        ok(a.cost<=s.b&&(s.m&a.gate)===s.m,"illegal replay");
        const o=a.outcomes[h],possible=a.parts.filter(([,m])=>Boolean(s.m&m)),z=possible.find(x=>x[0]===o);
        ok(z&&possible.length>1,"uninformative replay");s={m:s.m&z[1],b:s.b-a.cost};ok(Boolean(s.m&BITS[h]),"truth lost");
        const c=d.cs.find(x=>x.o===o);ok(c,"missing child");p=c.p;spent+=a.cost;q++;
      }
    }
    return{raw:value,utility:value/r.total,cost:cost/r.total,queries:queries/r.total,publication:pub/r.total,falsePublication:wrong/r.total,fullPublication:full/r.total};
  }
  function oracle(r){
    const memo=new Map(),sm=new Map(),idx=index(r,null);
    function stats(m){
      if(sm.has(m))return sm.get(m);let W=0;
      const a=[Array(128).fill(0),Array(64).fill(0),Array(16).fill(0),[0,0]],kinds=["full","partial","branch","coarse"];
      for(let h=0;h<N;h++)if(m&BITS[h]){const x=r.worlds[h],w=r.w[h];W+=w;a[0][h]+=w;a[1][4*x.branch+(x.payload>>1)]+=w;a[2][x.branch]+=w;a[3][x.coarse]+=w;}
      let best={v:0,i:-1,kind:"abstain",claim:-1};
      for(let k=0;k<4;k++){const kind=kinds[k],[g,b]=r.reward[kind];for(let j=0;j<a[k].length;j++){const v=(g+b)*a[k][j]-b*W;if(v>best.v)best={v,i:-1,kind,claim:j};}}
      const t={W,best};sm.set(m,t);return t;
    }
    function solve(s){
      const k=key(s);if(memo.has(k))return memo.get(k);const t=stats(s.m);let p=t.best;
      if(s.b>0){
        const ls=legalScan(r,s,seq(r.actions.length),null),is=legalScan(r,s,candidates(s,idx,null),null);ok(ls.length===is.length,"indexed legal length");counts.legalComparisons++;
        for(let j=0;j<ls.length;j++){const a=ls[j],b=is[j];ok(a.i===b.i&&a.cost===b.cost&&a.cs.length===b.cs.length,"indexed action");for(let c=0;c<a.cs.length;c++){ok(a.cs[c].o===b.cs[c].o&&a.cs[c].s.m===b.cs[c].s.m,"indexed child");counts.childComparisons++;}}
        for(const a of ls){if((36-a.cost)*t.W<=p.v)continue;let v=-a.cost*t.W;const cs=[];for(const c of a.cs){const q=solve(c.s);v+=q.v;cs.push({o:c.o,p:q});}if(v>p.v)p={v,i:a.i,cs};}
      }
      memo.set(k,p);return p;
    }
    const p=solve({m:ALL,b:r.budget}),z=replay(r,p,x=>x);ok(z.raw===p.v,"oracle replay");counts.oracleReplays++;
    return{...z,states:memo.size,statStates:sm.size};
  }
  function search(r,limit,mode,weights=null,training=false){
    const M=meter(limit),C=context(r,M),initial={m:ALL,b:r.budget},nodes=new Map(),random=rng(r.id+"/"+mode),labels=[];
    let root=null,fallback=null,fallbackU=36*r.total,actionJobs=0,constructiveJobs=0,aborted=false,reuses=0,scans=0;
    const eager=mode==="eager",constructive=!(eager||mode==="action_only");
    function create(s,stage,progress=null){
      M.tick("lookup");const k=key(s);if(nodes.has(k)||stage.has(k)){reuses++;return nodes.get(k)||stage.get(k);}
      const t=C.stats(s.m),ls=C.legal(s);
      let U=t.best.v;for(const a of ls){M.tick("bound");U=Math.max(U,(36-a.cost)*t.mass);}
      let base=t.best;if(eager)base=C.base(s,progress);ok(base.v<=U,"leaf bounds");M.tick("graph");
      const n={k,s,t,ls,base,L:base.v,U,arcs:new Map(),parents:new Set(),done:[eager,eager],lowerArc:null,upperAction:null};
      // Initial optimistic action, with deterministic action order.
      for(const a of ls){M.tick("bound");if((36-a.cost)*t.mass===U&&n.upperAction===null)n.upperAction=a.i;}
      stage.set(k,n);return n;
    }
    function initialize(){
      fallback=C.stats(ALL).best;const stage=new Map(),ls=C.legal(initial),t=C.stats(ALL);
      fallbackU=ls.reduce((v,a)=>{M.tick("bound");return Math.max(v,(36-a.cost)*t.mass);},t.best.v);
      const n=create(initial,stage,p=>{if(p.v>fallback.v)fallback=p;});
      M.tick("graph");nodes.set(n.k,n);root=n;
    }
    function refine(job){
      const n=job.n,stage=new Map(),overlay=new Map(),links=[],arcs=new Map(n.arcs);M.tick("graph",n.arcs.size+1);
      let base=n.base,done=[...n.done];
      if(job.type==="action"){
        ok(!arcs.has(job.a.i),"duplicate action job");const cs=[];
        for(const c of job.a.cs){M.tick("child");const x=create(c.s,stage);cs.push({o:c.o,n:x});links.push(x);}
        arcs.set(job.a.i,{i:job.a.i,cost:job.a.cost,cs});
      }else{
        ok(!done[job.which],"duplicate constructive job");const p=C.rollout(n.s,job.which);if(p.v>base.v)base=p;done[job.which]=true;
      }
      const pending=new Set([n]);
      while(pending.size){
        let x=null;for(const z of pending){M.tick("graph");if(x===null||z.s.b<x.s.b)x=z;}pending.delete(x);
        const aa=x===n?arcs:x.arcs,bb=x===n?base:x.base;let L=bb.v,U=x.t.best.v,la=null,ua=null;
        for(const a of x.ls){
          M.tick("backup");const arc=aa.get(a.i);let hi=(36-a.cost)*x.t.mass;
          if(arc){
            let lo=-a.cost*x.t.mass;hi=lo;
            for(const c of arc.cs){M.tick("backup");const z=overlay.get(c.n)||c.n;lo+=z.L;hi+=z.U;}
            if(lo>L){L=lo;la=arc;}
          }
          if(hi>U){U=hi;ua=a.i;}
        }
        ok(L<=U&&L>=x.L&&U<=x.U,"nonmonotone backup");counts.backupChecks++;
        overlay.set(x,{L,U,lowerArc:la,upperAction:ua});
        if(L!==x.L||U!==x.U)for(const p of x.parents){M.tick("graph");pending.add(p);}
      }
      M.tick("graph",stage.size+links.length+overlay.size+2);
      for(const [k,z] of stage)nodes.set(k,z);for(const z of links)z.parents.add(n);
      n.arcs=arcs;n.base=base;n.done=done;for(const [x,z] of overlay)Object.assign(x,z);
      if(job.type==="action")actionJobs++;else constructiveJobs++;
    }
    function frontier(){
      const found=[],seen=new Set(),stack=[root];
      while(stack.length){
        M.tick("frontier");const n=stack.pop();if(seen.has(n.k))continue;seen.add(n.k);if(n.U<=n.L)continue;
        if(constructive)for(let k=0;k<2;k++){M.tick("frontier");if(!n.done[k])found.push({n,type:"constructive",which:k,gap:n.U-n.L});}
        ok(n.upperAction!==null,"missing optimistic action");let a;
        for(const x of n.ls){M.tick("frontier");if(x.i===n.upperAction){a=x;break;}}ok(a,"action missing");
        const arc=n.arcs.get(a.i);
        if(arc){for(const c of arc.cs){M.tick("frontier");stack.push(c.n);}}
        else found.push({n,type:"action",a,gap:(36-a.cost)*n.t.mass-n.L});
      }
      return found;
    }
    function features(j){
      M.tick("feature",20);const n=j.n,mass=n.t.mass/r.total,gap=j.gap/r.total/36,b=n.s.b/9,lo=n.L/n.t.mass/36,hi=n.U/n.t.mass/36,term=n.t.best.v/n.t.mass/36,rg=(root.U-root.L)/r.total/36;
      const c=j.type==="action"?j.a.cs.length:0,kind=j.type==="action"?r.actions[j.a.i].kind:"constructive";
      return[1,mass,gap,(n.U-n.L)/n.t.mass/36,b,lo,hi,term,n.t.n/128,n.ls.length/13,rg,root.L/r.total/36,+ (j.type==="constructive"),j.type==="constructive"?j.which:0,c/8,j.type==="action"?j.a.cost/4:0,+(kind==="global"),+(kind==="full"),gap*b,gap/(1+c+n.t.n/8)];
    }
    function pick(fs,kind){
      let best=null,vbest=-Infinity,gbest=-Infinity;for(const j of fs){
        M.tick("priority");scans++;const n=j.n,g=j.gap/r.total;let v;
        if(kind==="mass_gap"||kind==="eager"||kind==="action_only")v=g;
        else if(kind==="work_gap")v=g/(j.type==="action"?1+j.a.cs.length+n.t.n/8:4+2*n.t.n+n.ls.length);
        else if(kind.startsWith("random"))v=random();
        else{const x=features(j),theta=weights[kind];M.tick("priority",x.length);v=Math.max(0,sum(x.map((z,i)=>z*theta[i])));}
        const tie=j.type==="action"?0:1,bt=best?.type==="action"?0:1;
        if(v>vbest||v===vbest&&(g>gbest||g===gbest&&tie<bt)){vbest=v;gbest=g;best=j;}
      }return best;
    }
    function step(p){
      // Evaluator follows the already committed graph; no policy-tree materialization.
      if(p.s){return p.lowerArc?{i:p.lowerArc.i,cs:p.lowerArc.cs.map(c=>({o:c.o,p:c.n}))}:p.base;}
      return p;
    }
    try{
      initialize();
      if(training){
        const checkpoints=new Set([0,2,5,10]),rr=rng(r.id+"/training");
        for(let k=0;k<=10&&root.L<root.U;k++){
          const fs=frontier();ok(fs.length,"empty training frontier");
          const j=rr()<.65?fs[Math.floor(rr()*fs.length)]:pick(fs,"mass_gap"),sample=checkpoints.has(k),x=sample?features(j):null,l=root.L,g=root.U-root.L,start=M.used;
          refine(j);counts.trainingJobs++;
          if(sample)labels.push({root:r.id,type:j.type,x,discovery:(root.L-l)/r.total*4096/(M.used-start),certification:(g-(root.U-root.L))/r.total*4096/(M.used-start),work:M.used-start});
        }
      }else while(root.L<root.U){const fs=frontier();ok(fs.length,"empty frontier");refine(pick(fs,mode));}
    }catch(e){if(e!==stop)throw e;aborted=true;}
    ok(fallback,"no terminal fallback");
    if(training)return labels;
    const L=root?root.L:fallback.v,U=root?root.U:fallbackU,out=replay(r,root||fallback,step);
    ok(out.raw===L,"incumbent replay mismatch");counts.policyReplays++;
    return{...out,lower:L/r.total,upper:U/r.total,certifiedGap:(U-L)/r.total,work:M.used,counts:M.counts,actionJobs,constructiveJobs,uniqueNodes:nodes.size,reuses,scans,aborted,...C.info()};
  }
  function analytic(r,limit){
    const M=meter(limit),C=context(r,M),initial={m:ALL,b:r.budget},bm=[new Map(),new Map(),new Map()],rm=[null,new Map(),new Map()];
    let best=null,U=36*r.total,stage="terminal",aborted=false;
    const accept=(p,k)=>{if(!best||p.v>best.v){best=p;stage=k;}};
    function B(s,k,progress=null){
      M.tick("lookup");const id=key(s);if(bm[k].has(id)){const p=bm[k].get(id);if(progress)progress(p);return p;}
      let p=C.base(s,progress);if(k>0){const q=B(s,k-1);if(q.v>p.v){p=q;if(progress)progress(p);}
        for(const a of C.legal(s)){M.tick("contingent");const q=C.actionPlan(s,a,z=>B(z,k-1));if(q.v>p.v){p=q;if(progress)progress(p);}}}
      bm[k].set(id,p);return p;
    }
    function replan(s,k){
      M.tick("lookup");const id=key(s);if(rm[k].has(id))return rm[k].get(id);let p=B(s,k);
      if(p.i>=0){let a;for(const z of C.legal(s)){M.tick("contingent");if(z.i===p.i){a=z;break;}}ok(a,"replan action");
        const q=C.actionPlan(s,a,z=>replan(z,k));if(q.v>p.v)p=q;}
      rm[k].set(id,p);return p;
    }
    try{
      best=C.stats(ALL).best;const t=C.stats(ALL);U=C.legal(initial).reduce((v,a)=>{M.tick("bound");return Math.max(v,(36-a.cost)*t.mass);},best.v);
      accept(C.base(initial,p=>accept(p,"base")),"base");
      for(let k=1;k<=2;k++){accept(B(initial,k,p=>accept(p,`B${k}`)),`B${k}`);accept(replan(initial,k),`replan${k}`);}
    }catch(e){if(e!==stop)throw e;aborted=true;}
    ok(best,"analytic fallback");const out=replay(r,best,x=>x);ok(out.raw===best.v,"analytic replay");counts.policyReplays++;
    return{...out,lower:best.v/r.total,upper:U/r.total,certifiedGap:(U-best.v)/r.total,work:M.used,counts:M.counts,actionJobs:0,constructiveJobs:0,uniqueNodes:0,reuses:0,scans:0,aborted,stage,contingentStates:sum(bm.map(x=>x.size)),...C.info()};
  }
  function fit(rows,target){
    const d=rows[0].x.length,A=seq(d).map(i=>seq(d+1).map(j=>i===j?0.05:0));
    for(const a of rows)for(let i=0;i<d;i++){for(let j=0;j<d;j++)A[i][j]+=a.x[i]*a.x[j];A[i][d]+=a.x[i]*a[target];}
    for(let i=0;i<d;i++){
      let p=i;for(let j=i+1;j<d;j++)if(Math.abs(A[j][i])>Math.abs(A[p][i]))p=j;[A[i],A[p]]=[A[p],A[i]];
      const z=A[i][i];ok(Math.abs(z)>1e-12,"ridge singularity");for(let j=i;j<=d;j++)A[i][j]/=z;
      for(let k=0;k<d;k++)if(k!==i){const z=A[k][i];for(let j=i;j<=d;j++)A[k][j]-=z*A[i][j];}
    }
    const weights=A.map(x=>x[d]),avg=mean(rows.map(x=>x[target]));
    return{weights,positive:rows.filter(x=>x[target]>0).length,mean:avg,mse:mean(rows.map(x=>(sum(x.x.map((z,i)=>z*weights[i]))-x[target])**2)),constantMSE:mean(rows.map(x=>(x[target]-avg)**2))};
  }
  const trainN=32*scale,evalN=24*scale,rows=[];
  for(let i=0;i<trainN;i++){const r=make("train",i),a=search(r,100000000,"training",null,true);ok(a.length<=4,"row cap");rows.push(...a);}
  const discovery=fit(rows,"discovery"),certification=fit(rows,"certification"),weights={discovery:discovery.weights,certification:certification.weights};
  const calRoots=seq(8).map(i=>make("calibration",i)),calibration=[];let primary=524288,qualified=false;
  for(const b of [32768,65536,131072,262144,524288]){
    const v=calRoots.map(r=>{const z=search(r,b,"mass_gap");return{root:r.id,actionJobs:z.actionJobs,constructiveJobs:z.constructiveJobs,work:z.work,utility:z.utility};}),a=v.map(x=>x.actionJobs).sort((x,y)=>x-y),median=(a[3]+a[4])/2,positive=a.filter(x=>x>0).length;
    calibration.push({budget:b,medianActionJobs:median,rootsWithActionJob:positive,roots:v});
    if(median>=8&&positive>=6){primary=b;qualified=true;break;}
  }
  const budgets=[primary/2,primary],rootResults=[];
  for(const split of ["ordinary","ood"])for(let i=0;i<evalN;i++){
    const r=make(split,i),truth=oracle(r),atBudget=[];
    for(const b of budgets){
      const arms={};for(const mode of modes){
        const a=mode==="analytic"?analytic(r,b):search(r,b,mode,weights);
        ok(a.work<=b&&a.lower<=truth.utility+1e-12&&a.upper>=truth.utility-1e-12,"root bounds/work");counts.rootBounds++;
        arms[mode]={...a,oracleGap:truth.utility-a.utility,oracleTie:a.raw===truth.raw,closed:a.certifiedGap===0};
      }atBudget.push({budget:b,arms});
    }rootResults.push({id:r.id,split,family:r.family,credits:r.budget,oracle:truth,atBudget});
  }
  const round=x=>Math.round(x*1e6)/1e6;
  function ci(x){const m=mean(x),sd=Math.sqrt(sum(x.map(z=>(z-m)**2))/Math.max(1,x.length-1));return[round(m),round(m-1.96*sd/Math.sqrt(x.length)),round(m+1.96*sd/Math.sqrt(x.length))];}
  const fields=["utility","oracleGap","certifiedGap","work","actionJobs","constructiveJobs","uniqueNodes","cost","queries","falsePublication","fullPublication","indexEntries","statStates","legalStates","rolloutStates"];
  const aggregate=["ordinary","ood"].map(split=>{
    const rr=rootResults.filter(r=>r.split===split),atBudget=[];
    for(let j=0;j<budgets.length;j++){
      const names=["mass_gap","work_gap","random","discovery","certification","eager","action_only","analytic"],arms={};
      function val(r,k,f){return k==="random"?mean(["random19","random37","random73"].map(z=>Number(r.atBudget[j].arms[z][f]))):Number(r.atBudget[j].arms[k][f]);}
      for(const k of names)arms[k]={...Object.fromEntries(fields.map(f=>[f,round(mean(rr.map(r=>val(r,k,f))))])),oracleTies:round(sum(rr.map(r=>val(r,k,"oracleTie")))),closed:round(sum(rr.map(r=>val(r,k,"closed"))))};
      const comparisons={};
      for(const t of ["mass_gap","discovery","certification"])for(const p of names.filter(k=>k!==t&&!(t==="mass_gap"&&["discovery","certification"].includes(k))))comparisons[`${t}_vs_${p}`]={utility:ci(rr.map(r=>val(r,t,"utility")-val(r,p,"utility"))),gapReduction:ci(rr.map(r=>val(r,p,"certifiedGap")-val(r,t,"certifiedGap")))};
      atBudget.push({budget:budgets[j],arms,comparisons});
    }return{split,roots:rr.length,oracleUtility:round(mean(rr.map(r=>r.oracle.utility))),oracleStates:sum(rr.map(r=>r.oracle.states)),atBudget};
  });
  return{receipt:{schema:"MTR-02F-public-development-v1",seed,scale,trainRoots:trainN,calibrationRoots:8,ordinaryRoots:evalN,oodRoots:evalN,uniqueRoots:signatures.size,trainingRows:rows.length,trainingActionRows:rows.filter(x=>x.type==="action").length,trainingConstructiveRows:rows.filter(x=>x.type==="constructive").length,features:weights.discovery.length,fit:{discovery,certification},counts,calibration,qualified,primary,budgets,modes,aggregate,rootResults},weights,rows};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
 const args=process.argv.slice(2).filter(x=>!x.startsWith("--")).map(Number);
 const start=process.hrtime.bigint(),cpu=process.cpuUsage(),r=mtr02f(...args);
 if(process.argv.includes("--profile"))console.error(JSON.stringify({scope:"whole campaign including training, oracle and replay; excludes JSON output",elapsedMs:Number(process.hrtime.bigint()-start)/1e6,cpuMicros:process.cpuUsage(cpu),maxRSSKiB:process.resourceUsage().maxRSS,rssBytes:process.memoryUsage().rss}));
 console.log(JSON.stringify(process.argv.includes("--weights")?r.weights:process.argv.includes("--rows")?r.rows:r.receipt,null,2));
}
