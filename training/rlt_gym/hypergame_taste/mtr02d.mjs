// MTR-02D public development certified anytime research search.
import {pathToFileURL} from "node:url";
export function mtr02d(seed=20261127,scale=1){
  const check=(b,m)=>{if(!b)throw Error(m);},seq=n=>Array.from({length:n},(_,i)=>i),sum=a=>a.reduce((x,y)=>x+y,0),mean=a=>sum(a)/a.length;
  check(Number.isSafeInteger(seed)&&Number.isInteger(scale)&&scale>=1&&scale<=3,"invalid run");
  const N=64,A=17,all=(1n<<64n)-1n,credits=[0,8,32,128],schedulers=["optimistic","mass_gap","depth_first","random","learned"],rootSignatures=new Set();
  let witnessChecks=0,nonlinearChecks=0,oracleReplayChecks=0,searchReplayChecks=0,boundChecks=0,partitionChecks=0;
  function rng(key){let s=2166136261;for(const c of key)s=Math.imul(s^c.charCodeAt(0),16777619);s=s||1;return()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return(s>>>0)/4294967296;};}
  function shuffle(a,r){a=[...a];for(let i=a.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
  function bits(m){let n=0;for(;m;m&=m-1n)n++;return n;}
  function make(split,i){
    const id=`${seed}/mtr02d/${split}/${i}`,r=rng(id),ri=n=>Math.floor(r()*n),family=split==="ood"?"cubic":i%2?"quadratic_b":"quadratic_a",budget=5+i%3,perm=shuffle(seq(6),r),flip=ri(64);
    function witness(h){
      const b=perm.map(j=>((h^flip)>>j)&1),p=b.slice(3);
      const s0=b[0]^(family==="cubic"?p[0]*p[1]*p[2]:family==="quadratic_a"?p[0]*p[1]:p[1]*p[2]);
      const s1=b[1]^(s0*p[2]),s2=b[2]^(s1*p[0]);
      const branch=4*s0+2*s1+s2,payload=4*p[0]+2*p[1]+p[2];
      return{branch,payload,coarse:p[0]^(p[1]*p[2])};
    }
    const ws=seq(N).map(witness),pri=seq(N).map(h=>1+ri(4)+(i%4===0&&ws[h].branch<4?2:0));
    check(new Set(ws.map(w=>8*w.branch+w.payload)).size===N,"witness encoding");
    check(seq(6).some(j=>new Set(seq(N).map(h=>(ws[h].branch>>2)^(ws[h^(1<<j)].branch>>2))).size>1),"nonlinear selector");nonlinearChecks++;
    const actions=[];
    function add(name,cost,req,observe){
      const outcomes=seq(N).map(h=>observe(ws[h])),m=new Map();
      for(let h=0;h<N;h++)m.set(outcomes[h],(m.get(outcomes[h])||0n)|(1n<<BigInt(h)));
      actions.push({name,cost,req,outcomes,parts:[...m.entries()].sort((a,b)=>a[0]-b[0])});
    }
    add("selector0",1+(split==="ood"&&i%4===0?1:0),0,w=>w.branch>>2);
    for(let s=0;s<2;s++)add(`selector1_${s}`,1,1,w=>w.branch>>2===s?(w.branch>>1)&1:(w.coarse^(s&1)));
    for(let s=0;s<4;s++)add(`selector2_${s}`,1+(split==="ood"&&s%2===0&&i%3===1?1:0),1<<(1+(s>>1)),w=>w.branch>>1===s?w.branch&1:(w.coarse^(s&1)));
    for(let s=0;s<8;s++)add(`certificate_${s}`,2+(i%5===0&&s%2===0?1:0)+(split==="ood"&&s%3===0?1:0),1<<(3+(s>>1)),w=>w.branch===s?w.payload:split==="ood"?((w.payload>>1)&1)^w.coarse:w.coarse);
    add("global_coarse",1,0,w=>w.coarse);
    add("global_payload",2+(split==="ood"&&i%3===0?1:0),0,w=>w.payload>>1);
    check(actions.length===A,"action count");
    const claims=[];
    function claim(kind,j,good,bad,pred){
      let m=0n;for(let h=0;h<N;h++)if(pred(ws[h],h))m|=1n<<BigInt(h);
      for(let h=0;h<N;h++){
        const w=witness(h);let v=kind==="full"?h===j:kind==="branch"?w.branch===j:kind==="partial"?2*w.branch+(w.payload>>2)===j:w.coarse===j;
        check(v===Boolean(m>>BigInt(h)&1n),"claim verifier mismatch");witnessChecks++;
      }
      claims.push({kind,j,good,bad,m});
    }
    for(let h=0;h<N;h++)claim("full",h,30,45,(_,x)=>x===h);
    for(let j=0;j<8;j++)claim("branch",j,12,18,w=>w.branch===j);
    for(let j=0;j<16;j++)claim("partial",j,18,27,w=>2*w.branch+(w.payload>>2)===j);
    for(let j=0;j<2;j++)claim("coarse",j,7,11,w=>w.coarse===j);
    const sig=JSON.stringify([family,budget,perm,flip,pri,actions.map(a=>[a.cost,a.req,a.outcomes])]);
    check(!rootSignatures.has(sig),"duplicate content root");rootSignatures.add(sig);
    const paths=[];
    for(let j=0;j<8;j++){
      const path=[0,1+(j>>2),3+(j>>1),7+j];
      for(let k=1;k<=4;k++){
        const s=path.slice(0,k);
        paths.push(s,[15,...s],[16,...s],[...s,15],[...s,16],[15,16,...s],[16,15,...s]);
      }
    }
    paths.push([15],[16],[15,16],[16,15]);
    const library=[...new Map(paths.map(p=>[p.reduce((m,j)=>m|(1<<j),0),p])).values()];
    return{id,split,family,budget,pri,ws,actions,claims,library};
  }
  function engine(r){
    const {actions,claims,pri,budget,library}=r,initial={m:all,u:0,b:budget},wm=new Map(),tm=new Map(),em=new Map(),states=new Map(),partitionMemo=new Map([[0,[all]]]),costMemo=new Map([[0,0]]);
    const key=s=>`${s.m.toString(16)}/${s.u}`;
    const W=m=>{if(!wm.has(m)){let n=0;for(let h=0;h<N;h++)if(m>>BigInt(h)&1n)n+=pri[h];wm.set(m,n);}return wm.get(m);};
    const rootMass=W(all);
    function term(m){
      if(tm.has(m))return tm.get(m);
      const mass=W(m),weights=Array(90).fill(0);
      // Claims occupy contiguous full, branch, partial and coarse ranges.
      for(let h=0;h<N;h++)if(m>>BigInt(h)&1n){
        const w=r.ws[h],p=pri[h];weights[h]+=p;weights[64+w.branch]+=p;weights[72+2*w.branch+(w.payload>>2)]+=p;weights[88+w.coarse]+=p;
      }
      let best={v:0,c:-1};
      for(let i=0;i<claims.length;i++){const c=claims[i],v=(c.good+c.bad)*weights[i]-c.bad*mass;if(v>best.v)best={v,c:i};}
      tm.set(m,best);return best;
    }
    const legal=s=>seq(A).filter(i=>!(s.u>>>i&1)&&(s.u&actions[i].req)===actions[i].req&&actions[i].cost<=s.b);
    const children=(s,i)=>actions[i].parts.map(([o,m])=>[o,{m:s.m&m,u:s.u|(1<<i),b:s.b-actions[i].cost}]).filter(x=>x[1].m!==0n);
    function exact(s){
      const k=key(s);if(em.has(k))return em.get(k);
      let best={v:term(s.m).v,i:-1};
      if(best.v<30*W(s.m))for(const i of legal(s)){
        const upper=(30-actions[i].cost)*W(s.m);if(upper<=best.v)continue;
        const v=sum(children(s,i).map(([,z])=>exact(z).v))-actions[i].cost*W(s.m);
        if(v>best.v)best={v,i};
      }
      em.set(k,best);states.set(k,s);return best;
    }
    function parts(mask){
      if(partitionMemo.has(mask))return partitionMemo.get(mask);
      const l=mask&-mask,i=31-Math.clz32(l),a=parts(mask^l).flatMap(m=>actions[i].parts.map(([,z])=>m&z).filter(z=>z!==0n));
      check(sum(a.map(W))===rootMass,"observation partition mass");partitionChecks++;partitionMemo.set(mask,a);return a;
    }
    function countCost(mask){
      if(costMemo.has(mask))return costMemo.get(mask);const l=mask&-mask,v=countCost(mask^l)+actions[31-Math.clz32(l)].cost;costMemo.set(mask,v);return v;
    }
    function libraryValue(s,work){
      work.libraryCalls++;let out={v:term(s.m).v,path:[]},seen=new Set(),mass=W(s.m);
      // The same underlying set can be reached from several library paths.
      // Deduplicate the remaining set per state and preserve one legal order.
      for(const path of library){
        work.libraryScans++;
        const p=path.filter(i=>!(s.u>>>i&1));if(!p.length)continue;
        const mask=p.reduce((m,i)=>m|(1<<i),0);if(seen.has(mask))continue;
        const cost=countCost(mask);if(cost>s.b)continue;
        let u=s.u,valid=true;
        for(const i of p){if((u&actions[i].req)!==actions[i].req){valid=false;break;}u|=1<<i;}
        if(!valid)continue;seen.add(mask);work.libraryEvaluations++;
        let v=-cost*mass;for(const q of parts(mask)){work.partitionCells++;const z=s.m&q;if(z)v+=term(z).v;}
        if(v>out.v)out={v,path:p};
      }
      return out;
    }
    const newWork=()=>({expansions:0,libraryCalls:0,libraryScans:0,libraryEvaluations:0,partitionCells:0,childTransitions:0,backups:0,priorityCalls:0,frontierVisits:0,reusedNodes:0,uniqueNodes:0});
    function play(s,path,h){
      let spent=0,queries=0;
      for(const i of path){
        check(legal(s).includes(i),"infeasible incumbent suffix");
        const o=actions[i].outcomes[h],z=children(s,i).find(([a])=>a===o);
        check(z&&Boolean(z[1].m>>BigInt(h)&1n),"inconsistent incumbent observation");
        s=z[1];spent+=actions[i].cost;queries++;
      }
      return{s,spent,queries};
    }
    function terminalReward(s,h){
      const c=term(s.m).c;if(c<0)return{reward:0,published:0,wrong:0,full:0};
      const x=claims[c],ok=Boolean(x.m>>BigInt(h)&1n),w=r.ws[h];
      const v=x.kind==="full"?h===x.j:x.kind==="branch"?w.branch===x.j:x.kind==="partial"?2*w.branch+(w.payload>>2)===x.j:w.coarse===x.j;
      check(ok===v,"replay witness mismatch");return{reward:ok?x.good:-x.bad,published:1,wrong:Number(!ok),full:Number(x.kind==="full")};
    }
    function replay(decide){
      let value=0,cost=0,queries=0,pub=0,wrong=0,full=0;
      for(let h=0;h<N;h++){
        let s=initial,spent=0,q=0;
        while(true){
          const d=decide(s);
          if(d.path!==undefined){const z=play(s,d.path,h);s=z.s;spent+=z.spent;q+=z.queries;break;}
          if(d.i<0)break;
          const z=play(s,[d.i],h);s=z.s;spent+=z.spent;q++;
        }
        const x=terminalReward(s,h);value+=pri[h]*(x.reward-spent);cost+=pri[h]*spent;queries+=pri[h]*q;pub+=pri[h]*x.published;wrong+=pri[h]*x.wrong;full+=pri[h]*x.full;
      }
      return{raw:value,utility:value/rootMass,cost:cost/rootMass,queries:queries/rootMass,publication:pub/rootMass,falsePublication:wrong/rootMass,fullPublication:full/rootMass};
    }
    function baseline(){
      const work=newWork(),memo=[new Map(),new Map(),new Map()];
      function B(s,k){
        const id=key(s);if(memo[k].has(id))return memo[k].get(id);
        let a;
        if(k===0)a={...libraryValue(s,work),i:-1};
        else {
          const base=B(s,k-1);a={v:base.v,path:base.path,i:base.i};
          // B_{k-1} policy is attainable; its first action may be reused.
          for(const i of legal(s)){
            const cs=children(s,i);work.childTransitions+=cs.length;const v=sum(cs.map(([,z])=>B(z,k-1).v))-actions[i].cost*W(s.m);
            if(v>a.v)a={v,i};
          }
        }
        memo[k].set(id,a);return a;
      }
      const rows=[];
      for(let k=0;k<=2;k++){
        const v=B(initial,k).v,prior={...work},out=replay(s=>B(s,k));
        check(out.raw>=v,"contingent policy below bound");const truth=exact(initial).v;check(v<=truth&&out.raw<=truth,"contingent upper");
        rows.push({k,lower:v/rootMass,...out,rootPlanningCumulative:prior,includingReplanCumulative:{...work}});
      }
      return rows;
    }
    function upper(s){
      const t=term(s.m).v,ls=legal(s);if(!ls.length)return t;
      return Math.max(t,(30-Math.min(...ls.map(i=>actions[i].cost)))*W(s.m));
    }
    function features(s,lower,U){
      const m=W(s.m),t=term(s.m).v,ls=legal(s),d=bits(BigInt(s.u));let mx=0,H=0;
      for(let h=0;h<N;h++)if(s.m>>BigInt(h)&1n){const p=pri[h]/m;mx=Math.max(mx,p);H-=p*Math.log2(p);}
      const b=s.b/8,frac=bits(s.m)/64,low=lower/m/30,gap=(U-lower)/m/30,mass=m/rootMass,avail=ls.length/A,terminal=t/m/30;
      return[1,b,frac,low,gap,mass,avail,terminal,mx,H/6,d/8,Number(Boolean(s.u&1)),Number(Boolean(s.u&(1<<15))),Number(Boolean(s.u&(1<<16))),gap*b,gap*frac,b*avail,mass*gap];
    }
    function examples(rng){
      exact(initial);
      const work=newWork(),ss=[...states.values()].filter(s=>s.b>0&&upper(s)>term(s.m).v);
      const early=shuffle(ss.filter(s=>bits(BigInt(s.u))<=2),rng),late=shuffle(ss.filter(s=>bits(BigInt(s.u))>2),rng);
      const chosen=[initial,...early.filter(s=>key(s)!==key(initial)).slice(0,1),...late.slice(0,2)];
      return chosen.map(s=>{const L=libraryValue(s,work).v,U=upper(s),v=exact(s).v;check(L<=v&&v<=U,"training bound");return{x:features(s,L,U),y:(v-L)/W(s.m)/30};});
    }
    function search(kind,theta){
      const work=newWork(),nodes=new Map(),random=rng(r.id+"/scheduler/random");
      function get(s){
        const k=key(s);
        if(nodes.has(k)){work.reusedNodes++;return nodes.get(k);}
        const base=libraryValue(s,work),U=upper(s);check(base.v<=U,"initial bound inversion");
        const n={key:k,s,L:base.v,U,base,lowerAction:null,upperAction:null,arcs:null,parents:new Set()};nodes.set(k,n);work.uniqueNodes++;return n;
      }
      const root=get(initial);
      function backup(n){
        if(!n.arcs)return false;
        work.backups++;
        const oldL=n.L,oldU=n.U;let l=n.base.v,u=term(n.s.m).v,la=null,ua=null;
        for(const a of n.arcs){
          const lo=-a.cost*W(n.s.m)+sum(a.cs.map(z=>z.L)),hi=-a.cost*W(n.s.m)+sum(a.cs.map(z=>z.U));
          if(lo>l){l=lo;la=a;}
          if(hi>u){u=hi;ua=a;}
        }
        // The independent leaf upper bound stays valid after expansion.
        // All backed-up action upper bounds are <= it in this model.
        check(l<=u&&l>=oldL&&u<=oldU,"nonmonotone bound backup");
        n.L=l;n.U=u;n.lowerAction=la;n.upperAction=ua;
        return l!==oldL||u!==oldU;
      }
      function expand(n){
        check(n.arcs===null&&n.U>n.L,"invalid expansion");work.expansions++;
        n.arcs=legal(n.s).map(i=>{
          const cs=children(n.s,i).map(([,s])=>get(s));work.childTransitions+=cs.length;
          for(const c of cs)c.parents.add(n);
          return{i,cost:actions[i].cost,cs};
        });
        const pending=new Set([n]);
        // Topological order: every parent has fewer used actions.
        while(pending.size){
          let x=null;for(const q of pending)if(x===null||bits(BigInt(q.s.u))>bits(BigInt(x.s.u)))x=q;
          pending.delete(x);if(backup(x))for(const p of x.parents)pending.add(p);
        }
      }
      function frontier(){
        const found=[],seen=new Set(),stack=[root];
        while(stack.length){
          const n=stack.pop();if(seen.has(n.key))continue;seen.add(n.key);work.frontierVisits++;
          if(n.U<=n.L)continue;
          if(n.arcs===null){found.push(n);continue;}
          check(n.upperAction!==null,"unresolved node lacks upper action");stack.push(...n.upperAction.cs);
        }
        return found;
      }
      function choose(fs){
        let best=null,value=-Infinity;
        for(const n of fs){
          work.priorityCalls++;
          const m=W(n.s.m),gap=n.U-n.L;let v;
          if(kind==="optimistic")v=gap/m;
          else if(kind==="mass_gap")v=gap/rootMass;
          else if(kind==="depth_first")v=bits(BigInt(n.s.u))+gap/m/100;
          else if(kind==="random")v=random();
          else{const f=features(n.s,n.L,n.U);v=Math.max(0,Math.min(gap/m/30,sum(f.map((x,i)=>x*theta[i]))))*m/rootMass;}
          if(v>value||(v===value&&best!==null&&n.key<best.key)){value=v;best=n;}
        }
        return best;
      }
      const snapshots=[];
      for(const b of credits){
        while(work.expansions<b&&root.L<root.U){const fs=frontier();check(fs.length>0,"empty unresolved frontier");expand(choose(fs));}
        const out=replay(s=>{
          const n=nodes.get(key(s));check(n!==undefined,"incumbent state not reached");
          return n.lowerAction===null?{path:n.base.path}:{i:n.lowerAction.i};
        });
        check(out.raw===root.L,"incumbent replay differs from certified lower bound");searchReplayChecks++;
        const V=exact(initial).v;check(root.L<=V&&V<=root.U,"oracle outside search bounds");boundChecks++;
        snapshots.push({budget:b,lower:root.L/rootMass,upper:root.U/rootMass,oracleGap:(V-root.L)/rootMass,certifiedGap:(root.U-root.L)/rootMass,...out,work:{...work}});
      }
      return snapshots;
    }
    function oracle(){const a=exact(initial),p=replay(s=>exact(s));check(p.raw===a.v,"exact replay");oracleReplayChecks++;return{...p,states:em.size};}
    return{oracle,baseline,examples,search};
  }
  function fit(rows){
    const d=rows[0].x.length,A=seq(d).map(i=>seq(d+1).map(j=>j===i?0.02:0));
    for(const {x,y} of rows)for(let i=0;i<d;i++){for(let j=0;j<d;j++)A[i][j]+=x[i]*x[j];A[i][d]+=x[i]*y;}
    for(let i=0;i<d;i++){
      let p=i;for(let j=i+1;j<d;j++)if(Math.abs(A[j][i])>Math.abs(A[p][i]))p=j;
      [A[i],A[p]]=[A[p],A[i]];const a=A[i][i];check(Math.abs(a)>1e-10,"singular ridge");
      for(let j=i;j<=d;j++)A[i][j]/=a;
      for(let k=0;k<d;k++)if(k!==i){const v=A[k][i];for(let j=i;j<=d;j++)A[k][j]-=v*A[i][j];}
    }
    return A.map(x=>x[d]);
  }
  const trainN=32*scale,evalN=16*scale,rows=[],rootResults=[];
  for(let i=0;i<trainN;i++){const r=make("train",i),g=engine(r),ex=g.examples(rng(r.id+"/examples"));check(ex.length<=4,"presentation cap");rows.push(...ex.map(x=>({root:r.id,...x})));}
  const theta=fit(rows),mse=mean(rows.map(x=>(sum(x.x.map((v,i)=>v*theta[i]))-x.y)**2)),constantMSE=mean(rows.map(x=>(x.y-mean(rows.map(x=>x.y)))**2));
  for(const split of ["ordinary","ood"])for(let i=0;i<evalN;i++){
    const r=make(split,i),g=engine(r),oracle=g.oracle(),baseline=g.baseline(),search={};
    for(const kind of schedulers)search[kind]=g.search(kind,theta);
    rootResults.push({id:r.id,split,family:r.family,budget:r.budget,oracle,baseline,search});
  }
  const round=x=>Math.round(x*1e6)/1e6;
  function ci(a){const m=mean(a),sd=Math.sqrt(sum(a.map(x=>(x-m)**2))/Math.max(1,a.length-1));return[round(m),round(m-1.96*sd/Math.sqrt(a.length)),round(m+1.96*sd/Math.sqrt(a.length))];}
  const aggregate=["ordinary","ood"].map(split=>{
    const rr=rootResults.filter(r=>r.split===split),base=seq(3).map(k=>({k,utility:round(mean(rr.map(r=>r.baseline[k].utility))),lower:round(mean(rr.map(r=>r.baseline[k].lower)))})),atBudget=[];
    for(let j=0;j<credits.length;j++){
      const policies={};
      for(const p of schedulers){const a=rr.map(r=>r.search[p][j]);policies[p]={...Object.fromEntries(["utility","lower","upper","oracleGap","certifiedGap","cost","queries","falsePublication","fullPublication"].map(k=>[k,round(mean(a.map(x=>x[k])))])),work:Object.fromEntries(Object.keys(a[0].work).map(k=>[k,round(mean(a.map(x=>x.work[k])))])),oracleTies:a.filter(x=>x.oracleGap===0).length,closed:a.filter(x=>x.certifiedGap===0).length};}
      const comparisons=Object.fromEntries(schedulers.filter(p=>p!=="learned").map(p=>[p,ci(rr.map(r=>r.search.learned[j].utility-r.search[p][j].utility))]));
      atBudget.push({budget:credits[j],policies,comparisons});
    }
    return{split,roots:rr.length,oracleUtility:round(mean(rr.map(r=>r.oracle.utility))),oracleStates:sum(rr.map(r=>r.oracle.states)),baseline:base,atBudget};
  });
  return{receipt:{schema:"MTR-02D-public-development-v1",seed,scale,trainRoots:trainN,ordinaryRoots:evalN,oodRoots:evalN,uniqueRoots:rootSignatures.size,trainingRows:rows.length,learnedParameters:theta.length,trainingMSE:mse,constantMSE,witnessChecks,nonlinearChecks,oracleReplayChecks,searchReplayChecks,boundChecks,partitionChecks,credits,schedulers,aggregate,rootResults},weights:theta,rows};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
 const args=process.argv.slice(2).filter(x=>!x.startsWith("--")).map(Number),r=mtr02d(...args);
 console.log(JSON.stringify(process.argv.includes("--weights")?r.weights:process.argv.includes("--rows")?r.rows:r.receipt,null,2));
}
