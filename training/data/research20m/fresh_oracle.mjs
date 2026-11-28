import {mkdirSync,writeFileSync} from "node:fs";
import {join,resolve} from "node:path";
import {createHash} from "node:crypto";
import {fileURLToPath} from "node:url";

function requireThat(x,msg){if(!x)throw Error(msg);}

function rng(seed){let s=seed>>>0||1;return(n)=>{s^=s<<13;s^=s>>>17;s^=s<<5;return(s>>>0)%n;};}

function pickBits(mask,n){return Array.from({length:n},(_,i)=>i).filter(i=>mask&(1<<i));}

function sum(a){return a.reduce((x,y)=>x+y,0);}

function evidenceProblem(r){
  const n=6+r(3),budget=5+r(12),items=Array.from({length:n},(_,i)=>({cost:1+r(5),value:1+r(12),current:r(7)!==0,authorized:r(9)!==0,requires:i&&r(3)===0?[r(i)]:[]}));
  const conflicts=[];for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)if(r(12)===0)conflicts.push([i,j]);
  return {budget,items,conflicts};
}

function evidenceSolve(p){
  let best={mask:0,cost:0,value:0};
  for(let m=0;m<1<<p.items.length;m++){
    const ids=pickBits(m,p.items.length),cost=sum(ids.map(i=>p.items[i].cost)),value=sum(ids.map(i=>p.items[i].value));
    if(cost>p.budget||ids.some(i=>!p.items[i].current||!p.items[i].authorized||p.items[i].requires.some(j=>!(m&(1<<j))))||p.conflicts.some(([i,j])=>(m&(1<<i))&&(m&(1<<j))))continue;
    if(value>best.value||(value===best.value&&m<best.mask))best={mask:m,cost,value};
  }
  let alternative={mask:0,cost:0,value:0};
  function dfs(i,ids,cost,value){
    if(cost>p.budget)return;
    if(i===p.items.length){const mask=sum(ids.map(j=>2**j));if(value>alternative.value||(value===alternative.value&&mask<alternative.mask))alternative={mask,cost,value};return;}
    dfs(i+1,ids,cost,value);
    const item=p.items[i];
    if(item.current&&item.authorized&&item.requires.every(j=>ids.includes(j))&&!p.conflicts.some(([a,b])=>(a===i&&ids.includes(b))||(b===i&&ids.includes(a))))dfs(i+1,[...ids,i],cost+item.cost,value+item.value);
  }
  dfs(0,[],0,0);requireThat(JSON.stringify(best)===JSON.stringify(alternative),"evidence oracle mismatch");
  return {selected:pickBits(best.mask,p.items.length),cost:best.cost,value:best.value,certificate:"All selected receipts are current and authorized; dependencies, conflict pairs and total cost satisfy the contract. Independent enumeration and incremental branch search agree on the maximum value and tie break."};
}

function experimentProblem(r){
  const n=4+r(3),m=3+r(2),nt=2+r(2);
  return {masses:Array.from({length:n},()=>1+r(9)),utility:Array.from({length:n},()=>Array.from({length:m},()=>r(25)-8)),tests:Array.from({length:nt},()=>({cost:r(6),observation:Array.from({length:n},()=>r(3))}))};
}

function experimentSolve(p){
  const n=p.masses.length,m=p.utility[0].length,den=sum(p.masses);
  function group(indices){
    const values=Array.from({length:m},(_,a)=>sum(indices.map(i=>p.masses[i]*p.utility[i][a])));
    let a=0;for(let j=1;j<m;j++)if(values[j]>values[a])a=j;
    return {action:a,weighted_utility:values[a]};
  }
  const base=group(Array.from({length:n},(_,i)=>i));
  const scores=[{test:-1,numerator:base.weighted_utility,policy:{none:base.action}}];
  for(let t=0;t<p.tests.length;t++){
    const test=p.tests[t],obs=[...new Set(test.observation)].sort((a,b)=>a-b),policy={},terms=[];
    for(const o of obs){const g=group(test.observation.flatMap((v,i)=>v===o?[i]:[]));policy[o]=g.action;terms.push(g.weighted_utility);}
    const numerator=sum(terms)-test.cost*den;
    let exhaustive=-Infinity;
    for(let code=0;code<m**obs.length;code++){
      let c=code;const choices={};for(const o of obs){choices[o]=c%m;c=Math.floor(c/m);}
      const v=sum(p.masses.map((w,i)=>w*p.utility[i][choices[test.observation[i]]]))-test.cost*den;
      exhaustive=Math.max(exhaustive,v);
    }
    requireThat(numerator===exhaustive,"experiment policy mismatch");
    scores.push({test:t,numerator,policy});
  }
  let winner=scores[0];for(const v of scores)if(v.numerator>winner.numerator)winner=v;
  return {denominator:den,baseline_action:base.action,options:scores,selected_test:winner.test,net_value_numerator:winner.numerator,increment_over_no_test:winner.numerator-base.weighted_utility,certificate:"For each observable outcome, maximize the unnormalized weighted utility. Subtract test cost times total mass before comparing options. Exhaustive contingent-action enumeration confirms each option. Value is conditional on this finite model."};
}

function proofProblem(r){
  const n=5+r(3),rules=[];
  for(let f=2;f<n;f++){const a=r(f),b=r(f);rules.push({premises:[...new Set([a,b])].sort((x,y)=>x-y),conclusion:f,cost:1+r(7)});}
  for(let i=0;i<3+r(3);i++){const f=2+r(n-2);rules.push({premises:[...new Set([r(f),r(f)])].sort((x,y)=>x-y),conclusion:f,cost:1+r(7)});}
  return {facts:n,initial:[0,1],goal:n-1,rules,budget:4+r(16)};
}

function proofSolve(p){
  const init=sum(p.initial.map(i=>1<<i)),goal=1<<p.goal;let best=null;
  for(let mask=0;mask<1<<p.rules.length;mask++){
    const ids=pickBits(mask,p.rules.length),cost=sum(ids.map(i=>p.rules[i].cost));
    if(best&&cost>=best.cost)continue;
    let have=init,changed=true,order=[];
    while(changed){changed=false;for(const i of ids){const q=p.rules[i],bit=1<<q.conclusion;if(!(have&bit)&&q.premises.every(j=>have&(1<<j))){have|=bit;order.push(i);changed=true;}}}
    if(have&goal)best={mask,cost,order};
  }
  requireThat(best,"generated proof unreachable");
  const dist=Array(1<<p.facts).fill(Infinity);dist[init]=0;
  const done=new Set();
  while(true){
    let state=-1;for(let i=0;i<dist.length;i++)if(!done.has(i)&&Number.isFinite(dist[i])&&(state<0||dist[i]<dist[state]))state=i;
    if(state<0)break;
    if(state&goal){requireThat(dist[state]===best.cost,"proof Dijkstra mismatch");break;}
    done.add(state);
    for(const q of p.rules){const bit=1<<q.conclusion;if(!(state&bit)&&q.premises.every(j=>state&(1<<j)))dist[state|bit]=Math.min(dist[state|bit],dist[state]+q.cost);}
  }
  let have=init;for(const i of best.order){const q=p.rules[i];requireThat(q.premises.every(j=>have&(1<<j)),"bad proof order");have|=1<<q.conclusion;}
  requireThat(have&goal,"bad witness");
  return {rule_order:best.order,min_cost:best.cost,budget_feasible:best.cost<=p.budget,certificate:"Replay rules in order, checking that every premise is already present. The least-cost subset enumeration agrees with shortest-path search over reachable fact sets. This proves only the finite Horn task."};
}

function observationProblem(r){
  const n=5+r(4),channels=3+r(3),property=Array.from({length:n},(_,i)=>i===0?0:i===1?1:r(2));
  return {property,channels:Array.from({length:channels},()=>({cost:1+r(8),values:Array.from({length:n},()=>r(3))}))};
}

function observationSolve(p){
  const n=p.property.length,cn=p.channels.length;
  let best=null;const cross=[];
  for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)if(p.property[i]!==p.property[j])cross.push([i,j]);
  for(let mask=0;mask<1<<cn;mask++){
    const ids=pickBits(mask,cn),cost=sum(ids.map(i=>p.channels[i].cost));
    if(cross.every(([a,b])=>ids.some(i=>p.channels[i].values[a]!==p.channels[i].values[b]))&&(!best||cost<best.cost))best={mask,cost};
  }
  let alt=null;for(let mask=0;mask<1<<cn;mask++){
    const ids=pickBits(mask,cn),fibers=new Map();let ok=true;
    for(let j=0;j<n;j++){const key=JSON.stringify(ids.map(i=>p.channels[i].values[j]));if(fibers.has(key)&&fibers.get(key)!==p.property[j]){ok=false;break;}fibers.set(key,p.property[j]);}
    const cost=sum(ids.map(i=>p.channels[i].cost));if(ok&&(!alt||cost<alt.cost))alt={mask,cost};
  }
  requireThat(JSON.stringify(best)===JSON.stringify(alt),"fiber oracle mismatch");
  if(!best){const witness=cross.find(([a,b])=>p.channels.every(c=>c.values[a]===c.values[b]));requireThat(witness,"missing impossibility witness");return {identifiable:false,indistinguishable_pair:witness,certificate:"Even all channels assign identical records to this pair, although the property differs. No subset can identify it."};}
  return {identifiable:true,channels:pickBits(best.mask,cn),min_cost:best.cost,certificate:"Every opposite-property state pair is separated by a selected channel. Independent observation-fiber enumeration confirms feasibility and minimum cost."};
}

function pairedProblem(r){
  const n=7+r(6);return {paired_differences:Array.from({length:n},()=>r(13)-6),alpha_numerator:1,alpha_denominator:20};
}

function pairedSolve(p){
  const d=p.paired_differences,observed=sum(d);let extreme=0;
  for(let mask=0;mask<1<<d.length;mask++)if(Math.abs(sum(d.map((v,i)=>mask&(1<<i)?v:-v)))>=Math.abs(observed))extreme++;
  let dist=new Map([[0,1]]);
  for(const v of d){const next=new Map();for(const [s,c] of dist)for(const w of [s-v,s+v])next.set(w,(next.get(w)||0)+c);dist=next;}
  const alt=sum([...dist].filter(([s])=>Math.abs(s)>=Math.abs(observed)).map(([,c])=>c));
  requireThat(extreme===alt&&sum([...dist.values()])===2**d.length,"sign-flip oracle mismatch");
  const total=2**d.length;
  return {observed_sum:observed,extreme_assignments:extreme,total_assignments:total,significant:extreme*p.alpha_denominator<=total*p.alpha_numerator,certificate:"Enumerate all paired sign assignments, including ties at the observed absolute statistic. A convolution of the exact signed-sum distribution gives the same tail count. Valid inference requires the registered sign-exchangeability assumption; this does not establish a mechanism or generalization."};
}

function generate(perFamily){
  requireThat(Number.isInteger(perFamily)&&perFamily>=1&&perFamily<=4096,"bounded root count");
  const defs=[
    ["evidence_selection","research_decision","DISTILL-22 / GSF retention",evidenceProblem,evidenceSolve,"Select maximum-value admissible evidence within the budget. A selected item must have a current authorized receipt and all prerequisites; conflicting pairs cannot coexist. On equal value choose the smallest numeric inclusion bitmask."],
    ["experiment_value","research_decision","MTR / GAL / CRD",experimentProblem,experimentSolve,"Choose no experiment or one test, then choose an action conditional on the test observation. Compute exact expected net utility using integer prior masses; subtract test cost. Break ties by no test, then lowest test index; actions tie by lowest index."],
    ["proof_ledger","formal_mathematics","COG-03 / EvalMath COG08",proofProblem,proofSolve,"Find a minimum-cost forward Horn proof of the goal from the initial facts. Each rule is charged when used, reusable facts are free, and a rule requires all premises. Report an executable rule order and whether the given budget suffices."],
    ["observation_fibers","quantum_information","QSM / PGC restricted observation",observationProblem,observationSolve,"Find a minimum-cost subset of observation channels that identifies the binary property on all latent states. If impossible, exhibit opposite-property states indistinguishable even with every channel. Tie-break by the smallest inclusion bitmask."],
    ["paired_audit","experiment_audit","SCCG-10 / matched-intervention audits",pairedProblem,pairedSolve,"Assuming paired sign-exchangeability under the registered null, compute the exact two-sided sign-flip tail probability for the absolute sum and whether it meets alpha. Report the numerator and denominator, including ties."]
  ];
  const all=[],seen=new Set();
  for(let f=0;f<defs.length;f++){
    const [family,lane,inspired,make,solve,task]=defs[f];
    for(let i=0;i<perFamily;i++){
      const random=rng((0x6e6f7632+Math.imul(f+1,0x9e3779b1)+Math.imul(i+1,0x85ebca6b))>>>0),p=make(random),key=JSON.stringify([family,p]);
      requireThat(!seen.has(key),"duplicate structural input");seen.add(key);
      const target=solve(p);
      all.push({schema:"dg.research20m.fresh-oracle.v1",canonical_root:"dg-r20m-v1/"+family+"/"+i,split:"train_candidate",lane,source_kind:"original_synthetic_finite_oracle",inspired_by:inspired,training_ready:false,prompt:task+"\nProblem: "+JSON.stringify(p),target,verification:"two algorithmic checks agree; shared implementation/runtime, not independent native verifier",custody:"new synthetic candidate; cross-corpus family overlap and rights review still required"});
    }
  }
  return all;
}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const args=process.argv.slice(2);
  if(args.length!==4||args[0]!=="--per-family"||args[2]!=="--out")throw Error("usage: node fresh_oracle.mjs --per-family 128 --out /new/custody/directory");
  const rows=generate(Number(args[1]));
  const out=resolve(args[3]);mkdirSync(out,{recursive:false,mode:0o700});
  const body=rows.map(r=>JSON.stringify(r)).join("\n")+"\n";
  const digest=createHash("sha256").update(body).digest("hex");
  const counts={};for(const r of rows)counts[r.lane]=(counts[r.lane]||0)+1;
  writeFileSync(join(out,"fresh_oracle_candidates.jsonl"),body,{flag:"wx",mode:0o600});
  writeFileSync(join(out,"RECEIPT.json"),JSON.stringify({schema:"dg.research20m.fresh-oracle-receipt.v1",rows:rows.length,counts,sha256:digest,training_ready:false,accepted_unique_aleph_tokens:null},null,2)+"\n",{flag:"wx",mode:0o600});
  console.log(JSON.stringify({rows:rows.length,counts,sha256:digest,training_ready:false}));
}
export {generate};
