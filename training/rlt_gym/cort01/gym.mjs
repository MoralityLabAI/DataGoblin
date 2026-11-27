// CORT-01 clean-room synthetic proxy. Node 18+; no external dependencies.
export function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export function makeTask(rand, family) {
  const pick = n => Math.floor(rand() * n);
  if (family === "recurrence") {
    return {family, x: pick(23), a: 1 + pick(8), b: pick(19), m: 23 + pick(20), n: 2 + pick(18)};
  }
  if (family === "graph") {
    const n = 5 + pick(5), edges = [];
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      if (i !== j && rand() < .24) edges.push([i,j]);
    }
    return {family, n, edges, start: pick(n)};
  }
  return {family, values: Array.from({length: 7 + pick(5)}, () => 1 + pick(10)), target: 5 + pick(24)};
}

export function solve(task, method) {
  if (task.family === "recurrence") {
    const {x,a,b,m,n} = task;
    if (method === 0) {
      let y = x;
      for (let i = 0; i < n; i++) y = (a*y+b)%m;
      return y;
    }
    if (method === 1) {
      const rec = k => k === 0 ? x : (a*rec(k-1)+b)%m;
      return rec(n);
    }
    let p = a%m, q = b%m, u = 1, v = 0, k = n;
    while (k) {
      if (k%2) { v = (p*v+q)%m; u = p*u%m; }
      q = (p*q+q)%m; p = p*p%m; k = Math.floor(k/2);
    }
    return (u*x+v)%m;
  }
  if (task.family === "graph") {
    const {n,edges,start} = task;
    if (method === 2) {
      const r = Array.from({length:n}, (_,i) => Array.from({length:n}, (_,j) => i===j));
      for (const [i,j] of edges) r[i][j]=true;
      for (let k=0;k<n;k++) for (let i=0;i<n;i++) for (let j=0;j<n;j++) {
        r[i][j] ||= r[i][k] && r[k][j];
      }
      return r[start].filter(Boolean).length;
    }
    if (method === 1) {
      const seen = new Set();
      function dfs(i) {
        if (seen.has(i)) return;
        seen.add(i);
        for (const [u,v] of edges) if (u===i) dfs(v);
      }
      dfs(start); return seen.size;
    }
    const seen = new Set([start]), queue=[start];
    for (let i=0;i<queue.length;i++) for (const [u,v] of edges) {
      if (u===queue[i] && !seen.has(v)) { seen.add(v); queue.push(v); }
    }
    return seen.size;
  }
  const {values,target} = task;
  if (method === 1) {
    let count=0;
    for (let mask=0;mask<2**values.length;mask++) {
      let sum=0;
      for (let i=0;i<values.length;i++) if (mask & (1<<i)) sum+=values[i];
      count += Number(sum===target);
    }
    return count;
  }
  if (method === 2) {
    const memo = new Map();
    function rec(i,s) {
      if (s<0) return 0;
      if (i===values.length) return Number(s===0);
      const key=i+":"+s;
      if (!memo.has(key)) memo.set(key,rec(i+1,s)+rec(i+1,s-values[i]));
      return memo.get(key);
    }
    return rec(0,target);
  }
  const dp=Array(target+1).fill(0); dp[0]=1;
  for (const v of values) for (let s=target;s>=v;s--) dp[s]+=dp[s-v];
  return dp[target];
}

export function oracle(task) {
  // Independent specifications: direct BigInt expansion, fixed-point set closure,
  // and recursive exhaustive include/exclude with no memoization.
  if (task.family==="recurrence") {
    const {x,a,b,m,n}=task, A=BigInt(a), B=BigInt(b);
    let geometric=0n;
    for(let k=0;k<n;k++) geometric+=A**BigInt(k);
    return Number((A**BigInt(n)*BigInt(x)+B*geometric)%BigInt(m));
  }
  if(task.family==="graph"){
    let seen=new Set([task.start]);
    for(let k=0;k<task.n;k++){
      const next=new Set(seen);
      for(const [u,v] of task.edges) if(seen.has(u)) next.add(v);
      seen=next;
    }
    return seen.size;
  }
  function visit(i,sum){
    if(i===task.values.length) return Number(sum===task.target);
    return visit(i+1,sum)+visit(i+1,sum+task.values[i]);
  }
  return visit(0,0);
}

export function misread(task) {
  if (task.family==="recurrence") return {...task,n:task.n-1};
  if (task.family==="graph") return {...task,edges:task.edges.map(([u,v])=>[v,u])};
  return {...task,target:task.target+1};
}

export function episode(rand, family, shift=false) {
  const task=makeTask(rand,family), truth=oracle(task);
  const altered=misread(task);
  // Shared interpretation errors and method-local errors are separate latent variables.
  const shared=rand() < (shift?.32:.16);
  const local=Array.from({length:3},()=>rand() < (shift?.24:.15));
  const methods=[0,1,0,2,2,1,0];
  const records=methods.map((method,i)=>{
    const executed=i!==3;
    const bad=(shared && (i<4 || rand()<.45)) || local[method] || rand()<.07;
    const source=bad?altered:task;
    const failure=rand() < (executed?.10:.04);
    return {role:i, method, executed:executed&&!failure,
      answer:failure?null:solve(source,method)};
  });
  return {family,task,truth,records};
}

export function rank(records, grouped=false, executableOnly=false) {
  const valid=records.filter(r=>r.answer!==null && (!executableOnly || r.executed));
  const clusters=new Map();
  for(const r of valid){
    if(!clusters.has(r.answer)) clusters.set(r.answer,[]);
    clusters.get(r.answer).push(r);
  }
  return [...clusters].map(([answer,members])=>{
    const methods=new Set(members.filter(r=>r.executed).map(r=>r.method));
    const exec=members.filter(r=>r.executed).length;
    // One vote per method per answer, split its vote across competing answers.
    const score=grouped?[...methods].reduce((s,m)=>s+1/new Set(
      valid.filter(r=>r.executed&&r.method===m).map(r=>r.answer)).size,0)
      +.25*Number(members.some(r=>!r.executed)):members.length;
    return {answer,members,methods:methods.size,exec,score};
  }).sort((a,b)=>b.score-a.score || b.exec-a.exec || a.answer-b.answer);
}

export function features(records) {
  const r=rank(records,true), top=r[0];
  if(!top) return {key:"empty",answer:null,methods:0};
  const conflict=r.length>1?1:0;
  const key=[Math.min(top.methods,3),Math.min(top.exec,4),conflict].join(":");
  return {key,answer:top.answer,methods:top.methods};
}

export function fit(episodes) {
  const bins={};
  for(const e of episodes) for(let k=2;k<=7;k++){
    const f=features(e.records.slice(0,k));
    const b=bins[f.key]??={n:0,correct:0};
    b.n++; b.correct+=Number(f.answer===e.truth);
  }
  return bins;
}

export function choose(records, policy, budget, calibration={}) {
  if(!Number.isInteger(budget)||budget<1||budget>7) throw Error("budget must be 1..7");
  if(!["single","majority","fixed","group_full","calibrated"].includes(policy)) throw Error("unknown policy");
  const cap=policy==="single"?1:budget;
  for(let k=1;k<=cap;k++){
    const seen=records.slice(0,k);
    if(policy==="fixed" && k>=2){
      const pal=rank(seen,false,true), top=pal[0];
      const contradiction=seen.some(r=>!r.executed&&r.answer!==null&&r.answer!==top?.answer);
      if(top?.exec>=2&&pal.length===1&&(!contradiction||top.exec>=3))
        return {answer:top.answer,calls:k,reason:"execution_consensus"};
    }
    if(policy==="calibrated"&&k>=2){
      const f=features(seen), b=calibration[f.key];
      const p=b?(b.correct+1)/(b.n+2):0;
      if(f.methods>=2&&b?.n>=40&&p>=.90)
        return {answer:f.answer,calls:k,reason:"calibrated_consensus",p};
    }
    if(k===cap){
      const grouped=policy==="calibrated"||policy==="group_full";
      const top=rank(seen,grouped)[0];
      return {answer:top?.answer??null,calls:k,reason:"budget"};
    }
  }
  throw Error("unreachable");
}

export function campaign(seed=20261127, trainN=600, evalN=300) {
  const families=["recurrence","graph","subset"];
  const trainRand=rng(seed), evalRand=rng(seed^0x93d37fa1);
  const train=Array.from({length:trainN},(_,i)=>episode(trainRand,families[i%3]));
  const calibration=fit(train);
  const summary={};
  for(const shift of [false,true]){
    const es=Array.from({length:evalN},(_,i)=>episode(evalRand,families[i%3],shift));
    const rows=[];
    for(const budget of [1,4,7]){
      const controls=["majority","fixed","group_full","calibrated"];
      for(const policy of budget===1?["single"]:controls){
        let correct=0,calls=0,stops=0,falseStops=0;
        for(const e of es){
          const d=choose(e.records,policy,budget,calibration);
          const ok=d.answer===e.truth;
          correct+=Number(ok);calls+=d.calls;
          if(d.calls<budget&&d.reason!=="budget"){stops++;falseStops+=Number(!ok);}
        }
        rows.push({policy,budget,n:es.length,correct,calls,stops,falseStops});
      }
    }
    summary[shift?"shift":"ordinary"]=rows;
  }
  return {scope:"synthetic public development replay; no model inference",trainN,evalN,summary};
}

export function selfTest(){
  const assert=(x,msg)=>{if(!x)throw Error(msg)};
  const rand=rng(19);
  for(let i=0;i<180;i++){
    const t=makeTask(rand,["recurrence","graph","subset"][i%3]);
    for(let m=0;m<3;m++) assert(solve(t,m)===oracle(t),"solver/oracle mismatch");
  }
  const rec=[{answer:7,executed:true,method:0},{answer:7,executed:true,method:0},
    {answer:4,executed:true,method:1},{answer:4,executed:true,method:2}];
  assert(choose(rec,"fixed",4).calls===2,"fixed consensus must stop");
  assert(choose(rec,"group_full",4).answer===4,"correlated duplicates overweighted");
  assert(choose(rec,"calibrated",4,{}).calls===4,"unseen calibration must abstain from stopping");
  const fail=Array.from({length:7},()=>({answer:null,executed:false,method:0}));
  assert(choose(fail,"fixed",7).answer===null,"all failures");
  const first=campaign(19,60,30), second=campaign(19,60,30);
  assert(JSON.stringify(first)===JSON.stringify(second),"determinism");
  for(const rows of Object.values(first.summary))for(const r of rows)
    assert(r.calls<=r.n*r.budget && r.falseStops<=r.stops,"budget accounting");
  return "540 solver cross-checks; consensus, deduplication, abstention, failures, determinism and budget checks passed";
}
