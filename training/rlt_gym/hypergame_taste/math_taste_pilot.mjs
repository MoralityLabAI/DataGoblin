// Public finite-certificate research taste pilot; standard Node.js only.
import { pathToFileURL } from "node:url";

export function mathTastePilot(seed=20261127, perCell=16) {
  const check=(v,m)=>{if(!v)throw Error(m);};
  check(Number.isSafeInteger(seed)&&Number.isInteger(perCell)&&perCell>0&&perCell<=1000,"bad run configuration");
  const seq=n=>Array.from({length:n},(_,i)=>i),sum=a=>a.reduce((x,y)=>x+y,0);
  const unique=a=>[...new Map(a.map(x=>[JSON.stringify(x),x])).values()];
  function rngFor(key){
    let s=2166136261;
    for(const c of key)s=Math.imul(s^c.charCodeAt(0),16777619);
    s=s||1;
    return n=>{s^=s<<13;s^=s>>>17;s^=s<<5;return (s>>>0)%n;};
  }
  function shuffle(a,r){a=[...a];for(let i=a.length-1;i>0;i--){const j=r(i+1);[a[i],a[j]]=[a[j],a[i]];}return a;}
  function worlds(f){const a=[];for(let t=0;a.length<8&&t<1000;t++){const x=f();if(!a.some(y=>JSON.stringify(y)===JSON.stringify(x)))a.push(x);}check(a.length===8,"world generation");return a;}
  const q=(name,cost,observe)=>({name,cost,observe});
  function regrets([R,C],[a,b,c,d]){
    const u=R[0]*c+R[1]*(d-c),v=R[2]*c+R[3]*(d-c),z=C[0]*a+C[2]*(b-a),w=C[1]*a+C[3]*(b-a);
    return [Math.max(u,v)*b-a*u-(b-a)*v,Math.max(z,w)*d-c*z-(d-c)*w,10*b*d];
  }
  const nashOK=(g,p)=>{const [a,b,c,d]=p,[x,y,z]=regrets(g,p);return b>0&&d>0&&a>=0&&a<=b&&c>=0&&c<=d&&5*Math.max(x,y)<=z;};
  function rational(a,b){if(b<0){a=-a;b=-b;}let x=Math.abs(a),y=b;while(y){[x,y]=[y,x%y];}return [a/x,b/x];}
  function equilibrium(g){
    for(const a of [0,1])for(const c of [0,1]){const p=[a,1,c,1],r=regrets(g,p);if(r[0]===0&&r[1]===0)return p;}
    const [R,C]=g,p=[...rational(C[3]-C[2],C[0]-C[1]-C[2]+C[3]),...rational(R[3]-R[1],R[0]-R[1]-R[2]+R[3])],r=regrets(g,p);
    check(nashOK(g,p)&&r[0]===0&&r[1]===0,"2x2 equilibrium");return p;
  }
  function nash(r){
    const ws=worlds(()=>[seq(4).map(()=>r(11)),seq(4).map(()=>r(11))]),ps=[[0,1],[1,2],[1,1]];
    const cs=unique([...ps.flatMap(x=>ps.map(y=>[...x,...y])),...ws.map(equilibrium)]);
    for(const g of ws)for(const p of cs){
      const [R,C]=g,[a,b,c,d]=p,x=a/b,y=c/d,j=[x*y,x*(1-y),(1-x)*y,(1-x)*(1-y)];
      const v=sum(R.map((v,i)=>v*j[i]))/10,w=sum(C.map((v,i)=>v*j[i]))/10;
      const alt=Math.max((R[0]*y+R[1]*(1-y))/10-v,(R[2]*y+R[3]*(1-y))/10-v,(C[0]*x+C[2]*(1-x))/10-w,(C[1]*x+C[3]*(1-x))/10-w);
      check(nashOK(g,p)===(alt<=.2+1e-12),"Nash differential");
    }
    return {ws,cs,ok:nashOK,qs:[
      q("row_preference_0",1,g=>Math.sign(g[0][0]-g[0][2])),q("row_preference_1",1,g=>Math.sign(g[0][1]-g[0][3])),
      q("col_preference_0",1,g=>Math.sign(g[1][0]-g[1][1])),q("col_preference_1",1,g=>Math.sign(g[1][2]-g[1][3])),
      q("row_payoffs",2,g=>g[0]),q("col_payoffs",2,g=>g[1])]};
  }
  const pairs=seq(4).flatMap(i=>seq(4).filter(j=>j>i).map(j=>[i,j])),rects=pairs.flatMap(a=>pairs.map(b=>[a,b]));
  const rectOK=(M,[a,b])=>{const v=a.flatMap(i=>b.map(j=>M[4*i+j]));return v.every(x=>x===v[0]);};
  function rank(M){
    const A=seq(4).map(i=>M.slice(4*i,4*i+4));let r=0;
    for(let j=0;j<4&&r<4;j++){
      const p=A.findIndex((row,i)=>i>=r&&row[j]!==0);if(p<0)continue;[A[p],A[r]]=[A[r],A[p]];
      for(let i=r+1;i<4;i++){const a=A[i][j],b=A[r][j];if(a)for(let k=j;k<4;k++)A[i][k]=b*A[i][k]-a*A[r][k];}
      r++;
    }return r;
  }
  function rectangle(r){
    const ws=worlds(()=>{const M=seq(16).map(()=>r(2)),[a,b]=rects[r(rects.length)],v=r(2);for(const i of a)for(const j of b)M[4*i+j]=v;return M;});
    return {ws,cs:rects,ok:rectOK,qs:[
      ...seq(4).map(i=>q("row_sum_"+i,1,M=>sum(M.slice(4*i,4*i+4)))),
      ...seq(4).map(j=>q("column_sum_"+j,1,M=>sum(seq(4).map(i=>M[4*i+j])))),
      ...seq(4).map(i=>q("read_row_"+i,2,M=>M.slice(4*i,4*i+4))),q("real_rank",2,rank)]};
  }
  const edges=seq(5).flatMap(i=>seq(5).filter(j=>j>i).map(j=>[i,j]));
  const perm=a=>a.length?a.flatMap((x,i)=>perm(a.filter((_,j)=>i!==j)).map(b=>[x,...b])):[[]];
  const eidx=(a,b)=>edges.findIndex(([i,j])=>i===Math.min(a,b)&&j===Math.max(a,b));
  const maps=perm(seq(5)).map(p=>edges.map(([i,j])=>eidx(p[i],p[j])));
  const remap=(m,p)=>edges.reduce((s,_,i)=>s|((m>>i&1)<<p[i]),0),cache=new Map();
  function canon(m){if(!cache.has(m)){const a=maps.map(p=>remap(m,p)),c=Math.min(...a);for(const x of a)cache.set(x,c);}return cache.get(m);}
  const pop=m=>edges.filter((_,i)=>m>>i&1).length;
  const deg=m=>seq(5).map(v=>edges.filter(([a,b],i)=>(a===v||b===v)&&(m>>i&1)).length).sort((a,b)=>a-b);
  const tri=m=>sum(seq(5).flatMap(i=>seq(5).filter(j=>j>i).map(j=>seq(5).filter(k=>k>j&&[eidx(i,j),eidx(i,k),eidx(j,k)].every(e=>m>>e&1)).length)));
  const cards=m=>edges.flatMap((_,i)=>m>>i&1?[m^(1<<i)]:[]);
  const deck=m=>cards(m).map(canon).sort((a,b)=>a-b);
  const catalog=seq(1024).filter(m=>[4,5,6].includes(pop(m))&&canon(m)===m);
  for(const m of catalog)for(const p of maps)check(canon(remap(m,p))===m,"relabeling");
  function graph(r){
    const m=4+r(3),ws=shuffle(catalog.filter(g=>pop(g)===m),r).slice(0,8);
    return {ws,cs:ws,ok:(a,b)=>canon(a)===canon(b),qs:[
      q("degree_sequence",1,deg),q("triangle_count",1,tri),
      q("card_degree_multiset",2,m=>cards(m).map(deg).map(x=>JSON.stringify(x)).sort()),q("full_edge_deck",3,deck)]};
  }
  const lpOK=([a,c,b],[x,y,u,v,w])=>[x,y,u,v,w].every(z=>z>=0)&&x<=2&&y<=2&&x+y<=b&&u+w>=a&&v+w>=c&&a*x+c*y===2*u+2*v+b*w;
  function lpCert([a,c,b]){const x=a>=c?Math.min(2,b):Math.max(0,b-2),y=b-x,w=b<=2?Math.max(a,c):Math.min(a,c);return [x,y,Math.max(0,a-w),Math.max(0,c-w),w];}
  function lp(r){
    const ws=worlds(()=>[1+r(6),1+r(6),1+r(3)]);
    for(const g of ws){const [a,c,b]=g,p=lpCert(g),brute=Math.max(...seq(3).flatMap(x=>seq(3).filter(y=>x+y<=b).map(y=>a*x+c*y)));check(lpOK(g,p)&&a*p[0]+c*p[1]===brute,"LP differential");}
    return {ws,cs:unique(ws.map(lpCert)),ok:lpOK,qs:[
      q("objective_order",1,g=>Math.sign(g[0]-g[1])),q("objective_sum",1,g=>g[0]+g[1]),q("rhs",1,g=>g[2]),
      q("first_coefficient",1,g=>g[0]),q("second_coefficient",1,g=>g[1]),q("full_objective",2,g=>g.slice(0,2))]};
  }
  const families={nash,rectangle,graph,lp},depths=[0,2,4,8,12,16],budgets=[1,2,3],policies=["exact","two_ply","myopic","native_exact","no_probe","fixed_order"];
  const receipt={schema:"math-taste-pilot-v1",seed,perCell,scope:"public synthetic development; no neural policy",depths,budgets,roots:0,policyRootEvaluations:0,exactReplayChecks:0,certificateChecks:0,trainingRows:0,cohorts:[]};
  const rows=[],stats=new Map();
  function root(f,K,i){
    const id=`${seed}/${f}/${K}/${i}`,r=rngFor(id),p=families[f](r),n=p.ws.length,prior=seq(n).map(()=>BigInt(1+r(4))),origin=seq(n).map(()=>2+r(13)),report=r(2),channels=[];
    let mass=origin.map(BigInt),scale=16n;
    for(let k=0;k<K;k++){
      const low=seq(n).map(()=>1+r(7)),high=seq(n).map(()=>9+r(7)),invert=i%4===0&&k%3===0,a=invert?high:low,b=invert?low:high;
      mass=mass.map((v,j)=>(scale-v)*BigInt(a[j])+v*BigInt(b[j]));scale*=16n;channels.push([a,b]);
    }
    const weights=mass.map((v,j)=>prior[j]*(report?v:scale-v)),cs=shuffle(p.cs,r);
    const masks=cs.map(c=>p.ws.reduce((m,w,j)=>{receipt.certificateChecks++;return m|(p.ok(w,c)?1<<j:0);},0)),full=(1<<n)-1;
    check(masks.reduce((a,b)=>a|b,0)===full,"missing certificate");
    return {id,f,K,ws:p.ws,cs,masks,full,weights,prior,origin,channels,report,native:Math.min(3,cs.length),
      qs:p.qs.map(q=>({name:q.name,cost:q.cost,outcomes:p.ws.map(w=>JSON.stringify(q.observe(w)))}))};
  }
  function evaluate(r,B){
    const {weights,masks,qs,full,native}=r,weightsByMask=seq(full+1).map(m=>weights.reduce((s,w,i)=>s+(m>>i&1?w:0n),0n)),total=weightsByMask[full],memo=new Map();
    function groups(m,q){const g=new Map();q.outcomes.forEach((o,i)=>{if(m>>i&1)g.set(o,(g.get(o)||0)|(1<<i));});return [...g.entries()];}
    function terminal(m,restricted){
      let best={kind:"abstain",i:-1,v:0n};
      for(let i=0;i<(restricted?native:masks.length);i++){const v=25n*weightsByMask[m&masks[i]]-15n*weightsByMask[m];if(v>best.v)best={kind:"publish",i,v};}return best;
    }
    function solve(m,b,look,restricted){
      const key=`${m}/${b}/${look}/${restricted}`;if(memo.has(key))return memo.get(key);
      let best=terminal(m,restricted);
      if(look)for(let i=0;i<qs.length;i++){const q=qs[i];if(q.cost>b)continue;const g=groups(m,q);if(g.length<2)continue;
        const v=g.reduce((s,[,child])=>s+solve(child,b-q.cost,look-1,restricted).v,-BigInt(q.cost)*weightsByMask[m]);
        if(v>best.v)best={kind:"probe",i,v};
      }memo.set(key,best);return best;
    }
    function act(p,m,b){
      if(p==="no_probe")return terminal(m,false);
      if(p==="fixed_order"){const i=qs.findIndex(q=>q.cost<=b&&groups(m,q).length>1);return i<0?terminal(m,false):{kind:"probe",i};}
      return solve(m,b,p==="myopic"?1:p==="two_ply"?2:b,p==="native_exact");
    }
    function replay(p){
      let utility=0n,cost=0n,checks=0n,published=0n,falsePublished=0n;
      for(let h=0;h<weights.length;h++){
        let m=full,b=B,spent=0,k=0;
        while(true){
          const a=act(p,m,b);
          if(a.kind==="probe"){const q=qs[a.i],g=groups(m,q).find(([o])=>o===q.outcomes[h]);check(q.cost<=b&&g&&(g[1]>>h&1)&&g[1]!==m,"invalid probe");m=g[1];b-=q.cost;spent+=q.cost;k++;continue;}
          const pub=a.kind==="publish",ok=pub&&Boolean(masks[a.i]>>h&1),w=weights[h];
          utility+=w*BigInt((pub?(ok?10:-15):0)-spent);cost+=w*BigInt(spent);checks+=w*BigInt(k);
          if(pub)published+=w;if(pub&&!ok)falsePublished+=w;break;
        }
      }
      const n=x=>Number(x)/Number(total);
      return {raw:utility,utility:n(utility),cost:n(cost),checks:n(checks),publish:n(published),falsePublish:n(falsePublished)};
    }
    const results=Object.fromEntries(policies.map(p=>[p,replay(p)]));
    check(results.exact.raw===solve(full,B,B,false).v&&results.native_exact.raw===solve(full,B,B,true).v,"DP versus hidden-world replay");
    receipt.exactReplayChecks+=2;
    for(const p of policies)check(results.exact.raw>=results[p].raw,"oracle domination");
    if(B===3){
      const queue=[{m:full,b:B,evidence:[]}],seen=new Set();
      while(queue.length&&seen.size<4){
        const {m,b,evidence}=queue.shift(),key=`${m}/${b}`;if(seen.has(key))continue;seen.add(key);
        const a=act("exact",m,b);
        rows.push({root:r.id,split:"public_development",evidence_class:"exact_finite_synthetic",family:r.f,K:r.K,budget:b,
          worlds:r.ws,candidates:r.cs,native_count:native,prior:r.prior.map(String),origin:r.origin,channels:r.channels,report:r.report,
          probes:qs,evidence,posterior_masses:weights.map((w,i)=>m>>i&1?String(w):"0"),
          target:{kind:a.kind,index:a.i,utility_numerator:String(a.v),denominator:String(weightsByMask[m])}});
        if(a.kind==="probe")for(const [o,child] of groups(m,qs[a.i]))queue.push({m:child,b:b-qs[a.i].cost,evidence:[...evidence,[qs[a.i].name,o]]});
      }
    }
    return results;
  }
  for(const f of Object.keys(families))for(const K of depths)for(let i=0;i<perCell;i++){
    const r=root(f,K,i);receipt.roots++;
    for(const B of budgets){
      const a=evaluate(r,B);receipt.policyRootEvaluations+=policies.length;
      for(const key of [`${f}/${K}/${B}`,`${f}/all/${B}`,`all/all/${B}`]){
        if(!stats.has(key))stats.set(key,{n:0,metrics:Object.fromEntries(policies.map(p=>[p,{utility:0,cost:0,checks:0,publish:0,falsePublish:0,d:0,d2:0}]))});
        const s=stats.get(key);s.n++;
        for(const p of policies){const v=a[p],m=s.metrics[p],d=a.exact.utility-v.utility;for(const k of ["utility","cost","checks","publish","falsePublish"])m[k]+=v[k];m.d+=d;m.d2+=d*d;}
      }
    }
  }
  const caps=new Map();for(const row of rows)caps.set(row.root,(caps.get(row.root)||0)+1);
  check(caps.size===receipt.roots&&[...caps.values()].every(n=>n<=4),"root cap");
  receipt.trainingRows=rows.length;
  const round=x=>Math.round(x*1e6)/1e6;
  for(const [key,s] of stats){
    const [family,K,B]=key.split("/"),metrics={};
    for(const [p,m] of Object.entries(s.metrics)){
      const mean=m.d/s.n,sd=Math.sqrt(Math.max(0,(m.d2-s.n*mean*mean)/Math.max(1,s.n-1)));
      metrics[p]={...Object.fromEntries(["utility","cost","checks","publish","falsePublish"].map(k=>[k,round(m[k]/s.n)])),oracleGap:round(mean),pairedNormal95:[round(mean-1.96*sd/Math.sqrt(s.n)),round(mean+1.96*sd/Math.sqrt(s.n))]};
    }receipt.cohorts.push({family,K,budget:Number(B),roots:s.n,metrics});
  }
  return {receipt,rows};
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2).filter(x => x !== "--rows");
  const result = mathTastePilot(...args.map(Number));
  if (process.argv.includes("--rows")) {
    for (const row of result.rows) console.log(JSON.stringify(row));
  } else console.log(JSON.stringify(result.receipt, null, 2));
}
