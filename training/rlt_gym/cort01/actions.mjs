// CORT-03 public synthetic action-value pilot; no external inference.
import { rng, episode, rank } from './gym.mjs';
import { rootKey } from './learned.mjs';

export const ACTIONS=[
  {
    "id": 0,
    "name": "query_m0",
    "cost": 1,
    "slot": 0,
    "method": 0
  },
  {
    "id": 1,
    "name": "query_m1",
    "cost": 1,
    "slot": 1,
    "method": 1
  },
  {
    "id": 2,
    "name": "repeat_m0",
    "cost": 1,
    "slot": 2,
    "method": 0
  },
  {
    "id": 3,
    "name": "query_m2",
    "cost": 1,
    "slot": 4,
    "method": 2
  },
  {
    "id": 4,
    "name": "fresh_requery",
    "cost": 2,
    "slot": 5,
    "method": 1
  },
  {
    "id": 5,
    "name": "residue3",
    "cost": 1,
    "mod": 3
  },
  {
    "id": 6,
    "name": "residue5",
    "cost": 1,
    "mod": 5
  },
  {
    "id": 7,
    "name": "stop",
    "cost": 0
  }
];
export const LAMBDA=0.035;

export function partition(seed,sizes){
  const rand=rng(seed),seen=new Set(),families=["recurrence","graph","subset"];
  return sizes.map(([n,shift])=>{
    const out=[];
    while(out.length<n){
      const e=episode(rand,families[out.length%3],shift),key=rootKey(e.task);
      if(seen.has(key))continue;
      seen.add(key);out.push(e);
    }
    return out;
  });
}

export function initial(family,cap){
  return {family,cap,spent:0,used:[],records:[],checks:[]};
}

export function viable(state){
  return rank(state.records,true).filter(c=>state.checks.every(
    check=>((c.answer%check.mod)===(check.answer%check.mod))===check.pass
  ));
}

export function current(state){return viable(state)[0]?.answer??null;}

export function legal(state){
  const remaining=state.cap-state.spent,answer=current(state);
  const a=ACTIONS.filter(x=>x.id!==7&&!state.used.includes(x.id)&&x.cost<=remaining
    &&(x.mod===undefined||answer!==null)&&(x.id!==4||state.records.length>0)).map(x=>x.id);
  if(answer!==null||a.length===0)a.push(7);
  return a;
}

export function step(world,state,id){
  if(!legal(state).includes(id)||id===7)throw Error("invalid acquisition");
  const a=ACTIONS[id],next={...state,spent:state.spent+a.cost,used:[...state.used,id],
    records:[...state.records],checks:[...state.checks]};
  if(a.mod!==undefined){
    const answer=current(state);
    next.checks.push({mod:a.mod,answer,pass:answer%a.mod===world.truth%a.mod});
  }else{
    const r=world.records[a.slot];
    next.records.push({role:a.id,method:a.method,executed:r.executed,answer:r.answer});
  }
  return next;
}

export function heuristic(state,policy){
  const choices=legal(state),available=id=>choices.includes(id);
  if(choices.length===1&&choices[0]===7)return 7;
  const candidates=viable(state),top=candidates[0], methods=new Set(state.records
    .filter(r=>r.answer!==null&&r.executed).map(r=>r.method));
  const distinct=top?.methods??0;
  const passes=state.checks.filter(c=>c.pass).length;
  if(policy==="fixed_query"||policy==="fixed_audit"){
    const order=policy==="fixed_query"?[0,1,3,2,4]:[0,1,5,3,6,2,4];
    return order.find(available)??7;
  }
  if(policy==="heuristic_stop"&&available(7)&&distinct>=2&&passes>=1&&candidates.length===1)return 7;
  const fresh=[0,1,3].find(id=>available(id)&&!methods.has(ACTIONS[id].method));
  const audit=[5,6].find(available);
  // Try a new method before duplicate votes. Independent checks become useful
  // when there is a candidate and multiple observations, or no fresh method.
  if((!top||state.records.length<2)&&fresh!==undefined)return fresh;
  if(top&&audit!==undefined&&(state.records.length>=2||fresh===undefined))return audit;
  if(fresh!==undefined)return fresh;
  if(available(4)&&(candidates.length!==1||state.checks.some(c=>!c.pass)))return 4;
  return [0,1,3,2,4,5,6].find(available)??7;
}

export function rollout(world,state,policy){
  while(true){
    const id=policy(state);
    if(id===7){
      if(!legal(state).includes(7))throw Error("illegal stop");
      return state;
    }
    state=step(world,state,id);
  }
}

export function vector(state){
  const candidates=viable(state),top=candidates[0],all=rank(state.records,true);
  const methods=new Set(state.records.filter(r=>r.executed&&r.answer!==null).map(r=>r.method));
  const pass=state.checks.filter(c=>c.pass).length,fail=state.checks.length-pass;
  return [1,(state.cap-state.spent)/7,state.spent/7,state.cap/7,
    state.records.length/5,state.records.filter(r=>r.answer===null).length/5,
    methods.size/3,candidates.length/5,(top?.methods??0)/3,(top?.score??0)/4,
    ((top?.score??0)-(candidates[1]?.score??0))/4,
    Number(top!==undefined),Number(candidates.length>1),pass/2,fail/2,
    (all.length-candidates.length)/5,
    ...[0,1,2].map(m=>Number(top?.members.some(r=>r.method===m&&r.executed)??false)),
    ...[0,1,2,3,4,5,6].map(id=>Number(state.used.includes(id))),
    Number(state.family==="graph"),Number(state.family==="subset"),
    Number(pass>0)*(top?.methods??0)/3,Number(fail>0)*Number(top===undefined)];
}

export function newFit(d){
  return {a:Array.from({length:d},()=>Array(d).fill(0)),b:Array(d).fill(0),n:0};
}

export function addFit(f,x,y){
  f.n++;
  for(let i=0;i<x.length;i++){
    f.b[i]+=x[i]*y;
    for(let j=0;j<=i;j++)f.a[i][j]+=x[i]*x[j];
  }
}

export function finishFit(f,ridge=4){
  const n=f.b.length,a=f.a.map((row,i)=>row.map((v,j)=>j<=i?v:f.a[j][i]));
  const b=[...f.b];
  for(let i=0;i<n;i++)a[i][i]+=i===0?0.1:ridge;
  for(let k=0;k<n;k++){
    const p=a[k][k];if(!(p>0))throw Error("singular fit");
    for(let i=k+1;i<n;i++){
      const scale=a[i][k]/p;
      for(let j=k;j<n;j++)a[i][j]-=scale*a[k][j];
      b[i]-=scale*b[k];
    }
  }
  const w=Array(n).fill(0);
  for(let i=n-1;i>=0;i--){
    let v=b[i];
    for(let j=i+1;j<n;j++)v-=a[i][j]*w[j];
    w[i]=v/a[i][i];
  }
  return {w,n:f.n};
}

export function label(world,state,id){
  const end=id===7?state:rollout(world,step(world,state,id),s=>heuristic(s,"heuristic_stop"));
  return Number(current(end)===world.truth)-LAMBDA*(end.spent-state.spent);
}

export function trainQ(worlds,seed){
  const rand=rng(seed),d=vector(initial("graph",4)).length;
  const fits=ACTIONS.map(()=>newFit(d));
  let states=0;
  for(const world of worlds)for(const cap of [4,6]){
    // Two behavior policies. Counterfactual labels use the same fixed continuation,
    // not a per-root oracle-maximized future.
    for(const behavior of ["random","heuristic"]){
      let state=initial(world.family,cap);
      while(true){
        const actions=legal(state),x=vector(state);
        states++;
        for(const id of actions)addFit(fits[id],x,label(world,state,id));
        const acquisitions=actions.filter(id=>id!==7);
        if(acquisitions.length===0)break;
        const id=behavior==="random"?acquisitions[Math.floor(rand()*acquisitions.length)]
          :heuristic(state,"heuristic_full");
        if(id===7)break;
        state=step(world,state,id);
      }
    }
  }
  return {models:fits.map(f=>finishFit(f)),states};
}

export function select(state,fit,full=false){
  const a=legal(state),choices=full&&a.some(x=>x!==7)?a.filter(x=>x!==7):a,x=vector(state);
  let id=choices[0],best=-Infinity;
  for(const action of choices){
    const m=fit.models[action];
    const q=m.n?m.w.reduce((s,v,i)=>s+v*x[i],0):-Infinity;
    if(q>best){best=q;id=action;}
  }
  return id;
}

export function evaluate(worlds,cap,policy){
  let correct=0,spent=0,queries=0,checks=0,repair=0,early=0,falseEarly=0;
  const ok=[],hist=Array(7).fill(0);
  for(const e of worlds){
    const end=rollout(e,initial(e.family,cap),policy);
    if(end.spent>cap)throw Error("cap exceeded");
    const good=current(end)===e.truth;
    correct+=Number(good);spent+=end.spent;
    queries+=end.records.length;checks+=end.checks.length;repair+=Number(end.used.includes(4));
    early+=Number(end.spent<cap);falseEarly+=Number(end.spent<cap&&!good);
    for(const id of end.used)hist[id]++;
    ok.push(good);
  }
  return {n:worlds.length,correct,spent,queries,checks,repair,early,falseEarly,
    utility:(correct-LAMBDA*spent)/worlds.length,hist,ok};
}

export function compare(a,b){
  return {win:a.ok.filter((ok,i)=>ok&&!b.ok[i]).length,loss:a.ok.filter((ok,i)=>!ok&&b.ok[i]).length};
}

export function campaign3(seed=2026112703){
  const [train,id,shift]=partition(seed,[[650,false],[320,false],[320,true]]);
  const fit=trainQ(train,seed^0x5812ab);
  const policies={
    fixed_query:s=>heuristic(s,"fixed_query"),
    fixed_audit:s=>heuristic(s,"fixed_audit"),
    heuristic_full:s=>heuristic(s,"heuristic_full"),
    heuristic_stop:s=>heuristic(s,"heuristic_stop"),
    learned_full:s=>select(s,fit,true),
    learned:s=>select(s,fit)
  };
  const out={seed,trainingStates:fit.states,fitCounts:fit.models.map(m=>m.n),summary:{}};
  for(const [cohort,es] of [["ordinary",id],["shift",shift]])for(const cap of [4,6]){
    const entries=Object.entries(policies).map(([name,policy])=>[name,evaluate(es,cap,policy)]);
    const baseline=entries.find(([n])=>n==="heuristic_stop")[1],full=entries.find(([n])=>n==="heuristic_full")[1];
    out.summary[`${cohort}_${cap}`]=Object.fromEntries(entries.map(([name,m])=>{
      const {ok,...row}=m;return [name,{...row,vsStop:compare(m,baseline),vsFull:compare(m,full)}];
    }));
  }
  return out;
}

export function tests3(){
  const assert=(b,m)=>{if(!b)throw Error(m)};
  const [tr,id,od]=partition(32,[[15,false],[15,false],[15,true]]);
  assert(new Set([...tr,...id,...od].map(e=>rootKey(e.task))).size===45,"duplicate root");
  const state=initial("graph",4);
  assert(!legal(state).includes(5)&&!legal(state).includes(4)&&!legal(state).includes(7),"initial gate");
  const mock={truth:8,records:Array.from({length:7},()=>({executed:true,answer:7}))};
  let s=step(mock,state,0);s=step(mock,s,5);
  assert(s.checks[0].pass===false&&current(s)===null,"failed check excludes candidate");
  assert(Object.keys(s.checks[0]).sort().join(",")==="answer,mod,pass","audit leaked hidden answer");
  assert(!legal(s).includes(5)&&!legal(s).includes(6),"repeat or empty audit");
  assert(state.spent===0&&state.records.length===0,"state mutation");
  const fit=trainQ(tr,10);
  assert(fit.models.every(m=>m.w.every(Number.isFinite)&&m.n>0),"invalid fitted coefficients");
  for(const e of [...id,...od])for(const cap of [4,6]){
    const end=rollout(e,initial(e.family,cap),s=>select(s,fit));
    assert(end.spent<=cap&&new Set(end.used).size===end.used.length,"budget or repeat");
    const answer=current(end);
    assert(answer===null||end.records.some(r=>r.answer===answer),"answer provenance");
    if(answer!==null)assert(end.checks.every(c=>((answer%c.mod)===(c.answer%c.mod))===c.pass),"audit contradiction");
    assert(end.spent===end.used.reduce((n,id)=>n+ACTIONS[id].cost,0),"cost receipt");
  }
  // Same visible state implies same policy action regardless of hidden root.
  assert(select(state,fit)===select({...state},fit),"policy not deterministic");
  return "CORT-03 custody, audit semantics, observation boundary, fit, action legality, cost receipts, candidate provenance passed";
}
