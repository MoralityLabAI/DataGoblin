// MTR-02E hard logical-work budgets; public development only.
import {pathToFileURL} from "node:url";
export function mtr02e(seed=20261127,scale=1){
  const assert=(b,m)=>{if(!b)throw Error(m);},seq=n=>Array.from({length:n},(_,i)=>i),sum=a=>a.reduce((x,y)=>x+y,0),mean=a=>sum(a)/a.length;
  assert(Number.isSafeInteger(seed)&&Number.isInteger(scale)&&scale>=1&&scale<=3,"invalid configuration");
  const N=128,ALL=(1n<<128n)-1n,BITS=seq(N).map(i=>1n<<BigInt(i)),INDEX=new Map(BITS.map((b,i)=>[b,i]));
  const budgets=[8192,32768,131072],modes=["mass_gap","work_gap","random19","random37","random73","discovery","certification","analytic"],exhausted=Symbol("budget"),signatures=new Set();
  let witnessChecks=0,nonlinearChecks=0,oracleReplays=0,policyReplays=0,boundChecks=0,backupChecks=0,trainingExpansions=0;
  function rng(k){let s=2166136261;for(const c of k)s=Math.imul(s^c.charCodeAt(0),16777619);s=s||1;return()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return(s>>>0)/4294967296;};}
  function shuffle(a,r){a=[...a];for(let i=a.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
  function make(split,i){
    const id=`${seed}/mtr02e/${split}/${i}`,r=rng(id),ri=n=>Math.floor(r()*n),family=split==="ood"?"cubic":i%2?"quadratic_b":"quadratic_a",budget=7+i%3,perm=shuffle(seq(7),r),flip=ri(128);
    function witness(h){
      const b=perm.map(j=>(h^flip)>>j&1),p=b.slice(4);
      const s0=b[0]^(family==="cubic"?p[0]*p[1]*p[2]:family==="quadratic_a"?p[0]*p[1]:p[1]*p[2]),s1=b[1]^(s0*p[2]),s2=b[2]^(s1*p[0]),s3=b[3]^(family==="cubic"?s2*p[0]*p[1]:p[1]*p[2]);
      return{branch:8*s0+4*s1+2*s2+s3,payload:4*p[0]+2*p[1]+p[2],coarse:p[0]^(p[1]*p[2])};
    }
    const worlds=seq(N).map(witness),w=seq(N).map(h=>1+ri(4)+(i%4===0&&worlds[h].branch<8?2:0));
    assert(new Set(worlds.map(x=>8*x.branch+x.payload)).size===N,"nonbijective witnesses");
    assert(seq(7).some(j=>new Set(seq(N).map(h=>(worlds[h].branch>>3)^(worlds[h^(1<<j)].branch>>3))).size>1),"affine selector");nonlinearChecks++;
    const mask=f=>worlds.reduce((m,x,h)=>f(x,h)?m|BITS[h]:m,0n),actions=[];
    function add(name,kind,cost,gate,observe){
      const outcomes=worlds.map(observe),parts=new Map();for(let h=0;h<N;h++)parts.set(outcomes[h],(parts.get(outcomes[h])||0n)|BITS[h]);
      actions.push({name,kind,cost,gate,outcomes,parts:[...parts.entries()].sort((a,b)=>a[0]-b[0])});
    }
    for(let l=0;l<4;l++)for(let p=0;p<1<<l;p++){
      const gate=mask(x=>(x.branch>>(4-l))===p);
      add(`single_${l}_${p}`,"single",1+Number(ri(4)===0)+Number(split==="ood"&&l===2&&p%3===0&&i%3===0),gate,x=>x.branch>>(3-l)&1);
      if(l<3)add(`pair_${l}_${p}`,"pair",1+ri(3)+Number(split==="ood"&&p%2===0&&i%4===0),gate,x=>x.branch>>(2-l)&3);
    }
    for(let p=0;p<16;p++){
      const gate=mask(x=>x.branch===p);
      add(`full_${p}`,"full",2+ri(3)+Number(split==="ood"&&p%3===0),gate,x=>x.payload);
      add(`partial_${p}`,"partial",1+ri(2),gate,x=>x.payload>>1);
    }
    add("coarse","global",1,ALL,x=>x.coarse);
    add("mixed","global",1+ri(2),ALL,x=>(x.branch>>3)^(x.payload&1));
    add("payload_pair","global",2+Number(split==="ood"&&i%3===0),ALL,x=>x.payload>>1);
    add("branch_parity","global",1+ri(2),ALL,x=>((x.branch>>3)^(x.branch>>1)^(x.payload>>2))&1);
    const reward={full:[36,54],partial:[24,36],branch:[15,23],coarse:[8,12]},groups={full:N,partial:64,branch:16,coarse:2};
    const pred=(kind,x,h)=>kind==="full"?h:kind==="partial"?4*x.branch+(x.payload>>1):kind==="branch"?x.branch:x.coarse;
    const claims=[];
    for(const [kind,n] of Object.entries(groups))for(let j=0;j<n;j++){
      const m=mask((x,h)=>pred(kind,x,h)===j);
      for(let h=0;h<N;h++){const v=witness(h);assert(Boolean(m&BITS[h])===(pred(kind,v,h)===j),"claim mismatch");witnessChecks++;}
      claims.push({kind,j,m,good:reward[kind][0],bad:reward[kind][1]});
    }
    const sig=JSON.stringify([family,budget,perm,flip,w,actions.map(a=>[a.cost,a.gate.toString(),a.outcomes])]);assert(!signatures.has(sig),"duplicate root");signatures.add(sig);
    return{id,split,family,budget,worlds,w,actions,reward,groups,pred,claims,total:sum(w)};
  }
  function key(m,b){return `${m.toString(16)}/${b}`;}
  function meter(limit){
    return{limit,used:0,counts:{},tick(k,n=1){assert(Number.isInteger(n)&&n>=0,"bad meter charge");const d=Math.min(n,this.limit-this.used);this.used+=d;this.counts[k]=(this.counts[k]||0)+d;if(d!==n)throw exhausted;}};
  }
  function context(r,M){
    const{w,worlds,actions,reward,groups,pred}=r,statsMemo=new Map(),legalMemo=new Map(),baseMemo=new Map(),hMemo=[new Map(),new Map()];
    let baseEvaluations=0,heuristicEvaluations=0;
    function stats(m){
      M.tick("lookup");if(statsMemo.has(m))return statsMemo.get(m);
      const tally=[new Map(),new Map(),new Map(),new Map()],kinds=["full","partial","branch","coarse"],max=[0,0,0,0],arg=[0,0,0,0];let W=0,n=0;
      for(let z=m;z;z&=z-1n){
        M.tick("world",5);const h=INDEX.get(z&-z),v=worlds[h],p=w[h],ids=[h,4*v.branch+(v.payload>>1),v.branch,v.coarse];W+=p;n++;
        for(let j=0;j<4;j++){const id=ids[j],a=(tally[j].get(id)||0)+p;tally[j].set(id,a);if(a>max[j]||a===max[j]&&id<arg[j]){max[j]=a;arg[j]=id;}}
      }
      assert(W>0,"empty evidence state");
      let best={v:0,i:-1,kind:"abstain",claim:-1};
      for(let j=0;j<4;j++){
        M.tick("claim");const kind=kinds[j],[g,b]=reward[kind],v=(g+b)*max[j]-b*W;if(v>best.v)best={v,i:-1,kind,claim:arg[j]};
      }
      const a={W,n,best};statsMemo.set(m,a);return a;
    }
    function legal(s){
      M.tick("lookup");const k=key(s.m,s.b);if(legalMemo.has(k))return legalMemo.get(k);const out=[];
      if(s.b>0)for(let i=0;i<actions.length;i++){
        M.tick("legal");const a=actions[i];if(a.cost>s.b||(s.m&a.gate)!==s.m)continue;
        const cs=[];for(const [o,p] of a.parts){M.tick("partition");const m=s.m&p;if(m)cs.push({o,s:{m,b:s.b-a.cost}});}
        if(cs.length>1)out.push({i,cost:a.cost,cs});
      }
      legalMemo.set(k,out);return out;
    }
    function upper(s){
      const t=stats(s.m),ls=legal(s);let min=Infinity;for(const a of ls){M.tick("bound");min=Math.min(min,a.cost);}
      return min===Infinity?t.best.v:Math.max(t.best.v,(36-min)*t.W);
    }
    function actionPlan(s,a,choose){
      M.tick("plan");const t=stats(s.m);let v=-a.cost*t.W;const cs=[];
      for(const c of a.cs){M.tick("plan");const p=choose(c.s);v+=p.v;cs.push({o:c.o,p});}
      return{v,i:a.i,cs};
    }
    function heuristic(s,mode){
      M.tick("lookup");const k=key(s.m,s.b),memo=hMemo[mode];if(memo.has(k))return memo.get(k);heuristicEvaluations++;M.tick("library");
      const t=stats(s.m),ls=legal(s);let best=t.best,chosen=null,score=-Infinity;
      for(const a of ls){
        M.tick("library");let max=0;for(const c of a.cs){M.tick("library");max=Math.max(max,stats(c.s.m).W);}
        const kind=actions[a.i].kind,info=(t.W-max)/t.W/a.cost,route=(kind==="global"?0:1)+Math.log2(a.cs.length)/a.cost;
        const v=mode===0?route+0.01*info:info+0.0001*route;
        if(v>score){score=v;chosen=a;}
      }
      if(chosen){const q=actionPlan(s,chosen,z=>heuristic(z,mode));if(q.v>best.v)best=q;}
      memo.set(k,best);return best;
    }
    function base(s,progress=null){
      M.tick("lookup");const k=key(s.m,s.b);if(baseMemo.has(k)){const a=baseMemo.get(k);if(progress)progress(a);return a;}M.tick("library");baseEvaluations++;
      let best=stats(s.m).best;if(progress)progress(best);
      const ls=legal(s);
      for(const a of ls){M.tick("library");const q=actionPlan(s,a,z=>stats(z.m).best);if(q.v>best.v){best=q;if(progress)progress(best);}}
      for(let mode=0;mode<2;mode++){const q=heuristic(s,mode);if(q.v>best.v){best=q;if(progress)progress(best);}}
      baseMemo.set(k,best);return best;
    }
    return{stats,legal,upper,base,actionPlan,counters:()=>({statsStates:statsMemo.size,legalStates:legalMemo.size,baseEvaluations,heuristicEvaluations,baseStates:baseMemo.size})};
  }
  function oracle(r){
    const statsMemo=new Map(),memo=new Map();
    function stats(m){
      if(statsMemo.has(m))return statsMemo.get(m);
      let W=0;const full=[],part=Array(64).fill(0),branch=Array(16).fill(0),coarse=[0,0];
      for(let h=0;h<N;h++)if(m&BITS[h]){const p=r.w[h],x=r.worlds[h];W+=p;full.push([h,p]);part[4*x.branch+(x.payload>>1)]+=p;branch[x.branch]+=p;coarse[x.coarse]+=p;}
      let best={v:0,i:-1,kind:"abstain",claim:-1};
      const cats=[["full",full],["partial",part.map((v,j)=>[j,v])],["branch",branch.map((v,j)=>[j,v])],["coarse",coarse.map((v,j)=>[j,v])]];
      for(const [kind,a] of cats){const [g,b]=r.reward[kind];for(const [j,mass] of a){const v=(g+b)*mass-b*W;if(v>best.v)best={v,i:-1,kind,claim:j};}}
      const t={W,best};statsMemo.set(m,t);return t;
    }
    function solve(m,b){
      const k=key(m,b);if(memo.has(k))return memo.get(k);
      const t=stats(m);let best=t.best;
      for(let i=0;i<r.actions.length;i++){
        const a=r.actions[i];if(a.cost>b||(a.gate&m)!==m||(36-a.cost)*t.W<=best.v)continue;
        const cs=[];for(const [o,p] of a.parts){const z=m&p;if(z)cs.push({o,m:z});}if(cs.length<2)continue;
        let v=-a.cost*t.W;const children=[];
        for(const c of cs){const p=solve(c.m,b-a.cost);v+=p.v;children.push({o:c.o,p});}
        if(v>best.v)best={v,i,cs:children};
      }
      memo.set(k,best);return best;
    }
    const plan=solve(ALL,r.budget),out=replay(r,plan);assert(out.raw===plan.v,"independent oracle replay");oracleReplays++;return{...out,states:memo.size,terminalStates:statsMemo.size};
  }
  function replay(r,plan){
    let value=0,cost=0,queries=0,published=0,falseMass=0,fullMass=0;
    for(let h=0;h<N;h++){
      let m=ALL,b=r.budget,p=plan,spent=0,q=0;
      while(p.i>=0){
        const a=r.actions[p.i];assert(a.cost<=b&&(m&a.gate)===m,"infeasible replay action");
        const o=a.outcomes[h],possible=a.parts.filter(([,mask])=>Boolean(mask&m));assert(possible.length>1,"redundant replay");
        const child=p.cs.find(c=>c.o===o),obs=possible.find(x=>x[0]===o);assert(child&&obs,"missing policy branch");
        m&=obs[1];assert(Boolean(m&BITS[h]),"lost true world");b-=a.cost;spent+=a.cost;q++;p=child.p;
      }
      const ok=p.kind==="abstain"||r.pred(p.kind,r.worlds[h],h)===p.claim,[g,bad]=p.kind==="abstain"?[0,0]:r.reward[p.kind],w=r.w[h];
      value+=w*((ok?g:-bad)-spent);cost+=w*spent;queries+=w*q;published+=w*Number(p.kind!=="abstain");falseMass+=w*Number(!ok);fullMass+=w*Number(p.kind==="full");
    }
    return{raw:value,utility:value/r.total,cost:cost/r.total,queries:queries/r.total,publication:published/r.total,falsePublication:falseMass/r.total,fullPublication:fullMass/r.total};
  }
  function search(r,limit,mode,weights=null,training=false){
    const M=meter(limit),C=context(r,M),initial={m:ALL,b:r.budget},nodes=new Map(),random=rng(r.id+"/"+mode),labels=[];
    let root=null,fallback=null,initialUpper=36*r.total,expansions=0,scans=0,reuses=0,aborted=false;
    function create(s,stage){
      M.tick("lookup");const k=key(s.m,s.b);
      if(nodes.has(k)||stage.has(k)){reuses++;return nodes.get(k)||stage.get(k);}
      M.tick("graph");const b=C.base(s),u=C.upper(s),t=C.stats(s.m),ls=C.legal(s);assert(b.v<=u,"leaf bound");
      const n={k,s,L:b.v,U:u,base:b,t,ls,arcs:null,lowerArc:null,upperArc:null,parents:new Set()};stage.set(k,n);return n;
    }
    function initialize(){
      const t=C.stats(ALL);fallback=t.best;initialUpper=C.upper(initial);
      const b=C.base(initial,p=>{if(p.v>fallback.v)fallback=p;});
      const n={k:key(ALL,r.budget),s:initial,L:b.v,U:initialUpper,base:b,t,ls:C.legal(initial),arcs:null,lowerArc:null,upperArc:null,parents:new Set()};
      M.tick("graph");nodes.set(n.k,n);root=n;
    }
    function expand(n){
      assert(n.arcs===null&&n.U>n.L,"invalid expansion");const stage=new Map(),links=[],overlay=new Map();
      const arcs=[];
      for(const a of n.ls){
        M.tick("graph");const cs=[];
        for(const c of a.cs){M.tick("child");const z=create(c.s,stage);cs.push({o:c.o,n:z});links.push(z);}
        arcs.push({i:a.i,cost:a.cost,cs});
      }
      const pending=new Set([n]);
      while(pending.size){
        let x=null;for(const z of pending){M.tick("graph");if(x===null||z.s.b<x.s.b)x=z;}pending.delete(x);
        const list=x===n?arcs:x.arcs;assert(list,"missing expanded ancestor");
        let L=x.base.v,U=x.t.best.v,la=null,ua=null;
        for(const a of list){
          M.tick("backup");let l=-a.cost*x.t.W,u=l;
          for(const c of a.cs){M.tick("backup");const z=overlay.get(c.n)||c.n;l+=z.L;u+=z.U;}
          if(l>L){L=l;la=a;}if(u>U){U=u;ua=a;}
        }
        assert(L<=U&&L>=x.L&&U<=x.U,"nonmonotone backup");backupChecks++;overlay.set(x,{L,U,lowerArc:la,upperArc:ua});
        if(L!==x.L||U!==x.U)for(const p of x.parents){M.tick("graph");pending.add(p);}
      }
      // Charge every planned graph insertion, link and value mutation before
      // committing; if interrupted, no reached-state or incumbent mutation.
      M.tick("graph",stage.size+links.length+overlay.size+1);
      for(const [k,z] of stage)nodes.set(k,z);
      for(const z of links)z.parents.add(n);n.arcs=arcs;
      for(const [x,z] of overlay)Object.assign(x,z);expansions++;
    }
    function frontier(){
      const found=[],seen=new Set(),stack=[root];
      while(stack.length){
        M.tick("frontier");const n=stack.pop();if(seen.has(n.k))continue;seen.add(n.k);
        if(n.U<=n.L)continue;
        if(n.arcs===null){found.push(n);continue;}
        assert(n.upperArc,"open expanded node without upper action");
        for(const c of n.upperArc.cs){M.tick("frontier");stack.push(c.n);}
      }
      return found;
    }
    function features(n){
      M.tick("feature",16);const mass=n.t.W/r.total,lg=(n.U-n.L)/n.t.W/36,gap=(n.U-n.L)/r.total/36,b=n.s.b/9,
        lo=n.L/n.t.W/36,hi=n.U/n.t.W/36,term=n.t.best.v/n.t.W/36,rootGap=(root.U-root.L)/r.total/36;
      let cs=0;for(const a of n.ls){M.tick("feature");cs+=a.cs.length;}
      const la=n.ls.length/12,child=cs/48;
      return[1,mass,lg,gap,b,lo,hi,term,n.t.n/128,la,child,rootGap,root.L/r.total/36,gap*b,gap/(1+cs),mass*lo];
    }
    function pick(fs,kind){
      let best=null,score=-Infinity,bg=-Infinity;
      for(const n of fs){
        M.tick("priority");scans++;const gap=(n.U-n.L)/r.total;let v;
        if(kind==="mass_gap")v=gap;
        else if(kind==="work_gap"){
          let c=0;for(const a of n.ls){M.tick("priority");c+=a.cs.length;}v=gap/(1+c+n.t.n/8);
        }else if(kind.startsWith("random"))v=random();
        else{const x=features(n),theta=weights[kind];M.tick("priority",x.length);v=Math.max(0,sum(x.map((a,j)=>a*theta[j])));}
        if(v>score||v===score&&(gap>bg||gap===bg&&best!==null&&n.k<best.k)){score=v;bg=gap;best=n;}
      }
      return best;
    }
    function plan(n,memo=new Map()){
      if(memo.has(n.k))return memo.get(n.k);
      const p=n.lowerArc?{v:n.L,i:n.lowerArc.i,cs:n.lowerArc.cs.map(c=>({o:c.o,p:plan(c.n,memo)}))}:n.base;
      memo.set(n.k,p);return p;
    }
    try{
      initialize();
      if(training){
        const checkpoints=new Set([0,2,5,10]),rr=rng(r.id+"/training");
        for(let step=0;step<=10&&root.L<root.U;step++){
          const fs=frontier();assert(fs.length,"empty frontier");const n=rr()<0.5?fs[Math.floor(rr()*fs.length)]:pick(fs,"mass_gap"),sample=checkpoints.has(step),x=sample?features(n):null,l=root.L,g=root.U-root.L,start=M.used;
          expand(n);trainingExpansions++;
          if(sample)labels.push({root:r.id,x,discovery:(root.L-l)/r.total*4096/(M.used-start),certification:(g-(root.U-root.L))/r.total*4096/(M.used-start),work:M.used-start});
        }
      }else while(root.L<root.U){const fs=frontier();assert(fs.length>0,"empty unresolved frontier");expand(pick(fs,mode));}
    }catch(e){if(e!==exhausted)throw e;aborted=true;}
    assert(fallback,"initial terminal failed");
    if(training)return labels;
    const p=root?plan(root):fallback,L=root?root.L:fallback.v,U=root?root.U:initialUpper,out=replay(r,p);
    assert(out.raw===L,"incumbent replay differs");policyReplays++;
    return{...out,lower:L/r.total,upper:U/r.total,certifiedGap:(U-L)/r.total,work:M.used,counts:M.counts,expansions,scans,reuses,uniqueNodes:nodes.size,aborted,...C.counters()};
  }
  function analytic(r,limit){
    const M=meter(limit),C=context(r,M),initial={m:ALL,b:r.budget},Bm=[new Map(),new Map(),new Map()],Rm=[null,new Map(),new Map()];
    let best=null,U=36*r.total,stage="terminal",aborted=false,complete=0;
    const accept=(p,k)=>{if(!best||p.v>best.v){best=p;stage=k;}};
    function B(s,k,progress=null){
      M.tick("lookup");const id=key(s.m,s.b);if(Bm[k].has(id)){const p=Bm[k].get(id);if(progress)progress(p);return p;}
      let p=C.base(s,progress);
      if(k>0){
        const q=B(s,k-1);if(q.v>p.v){p=q;if(progress)progress(p);}
        for(const a of C.legal(s)){
          M.tick("contingent");const z=C.actionPlan(s,a,t=>B(t,k-1));if(z.v>p.v){p=z;if(progress)progress(p);}
        }
      }
      Bm[k].set(id,p);return p;
    }
    function replan(s,k){
      M.tick("lookup");const id=key(s.m,s.b),memo=Rm[k];if(memo.has(id))return memo.get(id);
      let p=B(s,k);
      if(p.i>=0){
        const a=C.legal(s).find(x=>x.i===p.i);assert(a,"analytic first action");
        const q=C.actionPlan(s,a,t=>replan(t,k));if(q.v>p.v)p=q;
      }
      memo.set(id,p);return p;
    }
    try{
      best=C.stats(ALL).best;U=C.upper(initial);accept(C.base(initial,p=>accept(p,"base")),"base");complete=1;
      for(let k=1;k<=2;k++){accept(B(initial,k,p=>accept(p,`B${k}`)),`B${k}`);complete=2*k;accept(replan(initial,k),`replan${k}`);complete=2*k+1;}
    }catch(e){if(e!==exhausted)throw e;aborted=true;}
    assert(best,"analytic terminal failed");const out=replay(r,best);assert(out.raw===best.v,"analytic replay");policyReplays++;
    return{...out,lower:best.v/r.total,upper:U/r.total,certifiedGap:(U-best.v)/r.total,work:M.used,counts:M.counts,expansions:0,scans:0,reuses:0,uniqueNodes:0,aborted,stage,complete,contingentStates:sum(Bm.map(m=>m.size)),...C.counters()};
  }
  function fit(rows,target){
    const d=rows[0].x.length,A=seq(d).map(i=>seq(d+1).map(j=>i===j?0.05:0));
    for(const a of rows)for(let i=0;i<d;i++){for(let j=0;j<d;j++)A[i][j]+=a.x[i]*a.x[j];A[i][d]+=a.x[i]*a[target];}
    for(let i=0;i<d;i++){
      let p=i;for(let j=i+1;j<d;j++)if(Math.abs(A[j][i])>Math.abs(A[p][i]))p=j;[A[i],A[p]]=[A[p],A[i]];
      const z=A[i][i];assert(Math.abs(z)>1e-12,"singular");for(let j=i;j<=d;j++)A[i][j]/=z;
      for(let k=0;k<d;k++)if(k!==i){const z=A[k][i];for(let j=i;j<=d;j++)A[k][j]-=z*A[i][j];}
    }
    const w=A.map(a=>a[d]),avg=mean(rows.map(a=>a[target]));
    return{weights:w,mean:avg,positive:rows.filter(a=>a[target]>0).length,mse:mean(rows.map(a=>(sum(a.x.map((x,i)=>x*w[i]))-a[target])**2)),constantMSE:mean(rows.map(a=>(a[target]-avg)**2))};
  }
  const trainN=48*scale,evalN=24*scale,rows=[];
  for(let i=0;i<trainN;i++){const r=make("train",i),a=search(r,100000000,"training",null,true);assert(a.length<=4,"root cap");rows.push(...a);}
  const discovery=fit(rows,"discovery"),certification=fit(rows,"certification"),weights={discovery:discovery.weights,certification:certification.weights},rootResults=[];
  for(const split of ["ordinary","ood"])for(let i=0;i<evalN;i++){
    const r=make(split,i),truth=oracle(r),atBudget=[];
    for(const b of budgets){
      const arms={};
      for(const mode of modes){
        const a=mode==="analytic"?analytic(r,b):search(r,b,mode,weights);
        assert(a.work<=b&&a.lower<=truth.utility+1e-12&&a.upper>=truth.utility-1e-12,"root bound/work");boundChecks++;
        arms[mode]={...a,oracleGap:truth.utility-a.utility,oracleTie:a.raw===truth.raw,closed:a.certifiedGap===0};
      }
      atBudget.push({budget:b,arms});
    }
    rootResults.push({id:r.id,split,family:r.family,credits:r.budget,oracle:truth,atBudget});
  }
  const round=x=>Math.round(x*1e6)/1e6;
  function ci(x){const m=mean(x),sd=Math.sqrt(sum(x.map(z=>(z-m)**2))/Math.max(1,x.length-1));return[round(m),round(m-1.96*sd/Math.sqrt(x.length)),round(m+1.96*sd/Math.sqrt(x.length))];}
  const aggregate=["ordinary","ood"].map(split=>{
    const rr=rootResults.filter(r=>r.split===split),atBudget=[];
    for(let j=0;j<budgets.length;j++){
      const names=["mass_gap","work_gap","random","discovery","certification","analytic"],arms={};
      function rowValue(r,k,f){return k==="random"?mean(["random19","random37","random73"].map(z=>r.atBudget[j].arms[z][f])):r.atBudget[j].arms[k][f];}
      for(const k of names){
        arms[k]={...Object.fromEntries(["utility","oracleGap","certifiedGap","work","expansions","uniqueNodes","cost","queries","falsePublication","fullPublication"].map(f=>[f,round(mean(rr.map(r=>rowValue(r,k,f))))])),oracleTies:round(sum(rr.map(r=>rowValue(r,k,"oracleTie")))),closed:round(sum(rr.map(r=>rowValue(r,k,"closed"))))};
      }
      const comparisons={};
      for(const target of ["discovery","certification"])for(const k of ["mass_gap","work_gap","random","analytic"]){
        comparisons[`${target}_vs_${k}`]={utility:ci(rr.map(r=>rowValue(r,target,"utility")-rowValue(r,k,"utility"))),gapReduction:ci(rr.map(r=>rowValue(r,k,"certifiedGap")-rowValue(r,target,"certifiedGap")))};
      }
      atBudget.push({budget:budgets[j],arms,comparisons});
    }
    return{split,roots:rr.length,oracleUtility:round(mean(rr.map(r=>r.oracle.utility))),oracleStates:sum(rr.map(r=>r.oracle.states)),atBudget};
  });
  return{receipt:{schema:"MTR-02E-public-development-v1",seed,scale,trainRoots:trainN,ordinaryRoots:evalN,oodRoots:evalN,uniqueRoots:signatures.size,trainingRows:rows.length,features:weights.discovery.length,fit:{discovery,certification},witnessChecks,nonlinearChecks,oracleReplays,policyReplays,boundChecks,backupChecks,trainingExpansions,budgets,modes,aggregate,rootResults},weights,rows};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
 const args=process.argv.slice(2).filter(x=>!x.startsWith("--")).map(Number),r=mtr02e(...args);
 console.log(JSON.stringify(process.argv.includes("--weights")?r.weights:process.argv.includes("--rows")?r.rows:r.receipt,null,2));
}
