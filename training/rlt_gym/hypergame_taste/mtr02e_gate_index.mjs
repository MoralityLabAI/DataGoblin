// Post-hoc exact prefix-gate lookup audit for MTR-02E.
import {readFileSync} from "node:fs";
import {pathToFileURL} from "node:url";
export function mtr02eGateAudit(receipt){
 const assert=(b,m)=>{if(!b)throw Error(m);},seq=n=>Array.from({length:n},(_,i)=>i),sum=a=>a.reduce((x,y)=>x+y,0),mean=a=>sum(a)/a.length;
 const N=128,BITS=seq(N).map(i=>1n<<BigInt(i)),ALL=(1n<<128n)-1n;
 function rng(k){let s=2166136261;for(const c of k)s=Math.imul(s^c.charCodeAt(0),16777619);s||=1;return()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return(s>>>0)/4294967296;};}
 function shuffle(a,r){a=[...a];for(let i=a.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
 function root(row){
  const {id,split,family,credits}=row,i=Number(id.split("/").at(-1)),r=rng(id),ri=n=>Math.floor(r()*n),perm=shuffle(seq(7),r),flip=ri(128);
  const worlds=seq(N).map(h=>{
   const b=perm.map(j=>(h^flip)>>j&1),p=b.slice(4);
   const s0=b[0]^(family==="cubic"?p[0]*p[1]*p[2]:family==="quadratic_a"?p[0]*p[1]:p[1]*p[2]),s1=b[1]^(s0*p[2]),s2=b[2]^(s1*p[0]),s3=b[3]^(family==="cubic"?s2*p[0]*p[1]:p[1]*p[2]);
   return{branch:8*s0+4*s1+2*s2+s3,payload:4*p[0]+2*p[1]+p[2],coarse:p[0]^(p[1]*p[2])};
  });
  const w=seq(N).map(h=>1+ri(4)+(i%4===0&&worlds[h].branch<8?2:0));
  const masks=seq(5).map(l=>seq(1<<l).map(p=>worlds.reduce((m,x,h)=>x.branch>>(4-l)===p?m|BITS[h]:m,0n)));
  const index=seq(5).map(l=>seq(1<<l).map(()=>[])),global=[],actions=[];
  function add(cost,l,p,observe){
   const gate=l<0?ALL:masks[l][p],outcomes=new Map();
   for(let h=0;h<N;h++){const o=observe(worlds[h]);outcomes.set(o,(outcomes.get(o)||0n)|BITS[h]);}
   const j=actions.length;actions.push({cost,gate,parts:[...outcomes].sort((a,b)=>a[0]-b[0])});
   if(l<0)global.push(j);else index[l][p].push(j);
  }
  for(let l=0;l<4;l++)for(let p=0;p<1<<l;p++){
   add(1+Number(ri(4)===0)+Number(split==="ood"&&l===2&&p%3===0&&i%3===0),l,p,x=>x.branch>>(3-l)&1);
   if(l<3)add(1+ri(3)+Number(split==="ood"&&p%2===0&&i%4===0),l,p,x=>x.branch>>(2-l)&3);
  }
  for(let p=0;p<16;p++){
   add(2+ri(3)+Number(split==="ood"&&p%3===0),4,p,x=>x.payload);
   add(1+ri(2),4,p,x=>x.payload>>1);
  }
  add(1,-1,0,x=>x.coarse);add(1+ri(2),-1,0,x=>(x.branch>>3)^(x.payload&1));add(2+Number(split==="ood"&&i%3===0),-1,0,x=>x.payload>>1);add(1+ri(2),-1,0,x=>((x.branch>>3)^(x.branch>>1)^(x.payload>>2))&1);
  assert(actions.length===58,"action bank");
  return{row,worlds,w,masks,index,global,actions,credits};
 }
 // The only eligible prefix gates are ancestors of the longest prefix
 // common to every witness in m. Prefix masks form a binary partition.
 // Actions are generated in increasing level/prefix order, then globals.
 // Appending along that path preserves the original candidate tie order.
 function candidates(m,r,c){
  const out=[];let p=0;
  for(let l=0;l<=4;l++){
   c.indexReads++;for(const j of r.index[l][p]){out.push(j);c.indexEmits++;}
   if(l===4)break;
   c.prefixTests++;const z=m&r.masks[l+1][2*p];
   if(z===m)p*=2;else if(z===0n)p=2*p+1;else break;
  }
  c.indexReads++;for(const j of r.global){out.push(j);c.indexEmits++;}
  return out;
 }
 const rows=[];
 for(const row of receipt.rootResults){
  const r=root(row),memo=new Map(),stats=new Map(),c={states:0,positiveBudgetStates:0,fullScans:0,indexedScans:0,prefixTests:0,indexReads:0,indexEmits:0,fullPartitionTests:0,indexedPartitionTests:0,eligibleActions:0,legalActions:0,childComparisons:0,maxCandidateCount:0,oracleMatches:0};
  function terminal(m){
   if(stats.has(m))return stats.get(m);let mass=0,mx=0;const part=Array(64).fill(0),branch=Array(16).fill(0),coarse=[0,0];
   for(let h=0;h<N;h++)if(m&BITS[h]){const w=r.w[h],x=r.worlds[h];mass+=w;mx=Math.max(mx,w);part[4*x.branch+(x.payload>>1)]+=w;branch[x.branch]+=w;coarse[x.coarse]+=w;}
   const v=Math.max(0,90*mx-54*mass,60*Math.max(...part)-36*mass,38*Math.max(...branch)-23*mass,20*Math.max(...coarse)-12*mass),z={v,mass};stats.set(m,z);return z;
  }
  function legal(m,b,ids,indexed){
   const result=[];
   for(const j of ids){if(indexed)c.indexedScans++;else c.fullScans++;const a=r.actions[j];if(a.cost>b||(m&a.gate)!==m)continue;const cs=[];
    for(const [o,p] of a.parts){if(indexed)c.indexedPartitionTests++;else c.fullPartitionTests++;const z=m&p;if(z)cs.push([o,z]);}
    if(cs.length>1)result.push({j,cost:a.cost,cs});
   }return result;
  }
  function solve(m,b){
   const key=`${m.toString(16)}/${b}`;if(memo.has(key))return memo.get(key);
   c.states++;const t=terminal(m);let v=t.v;
   if(b>0){
    c.positiveBudgetStates++;const ids=candidates(m,r,c);c.maxCandidateCount=Math.max(c.maxCandidateCount,ids.length);c.eligibleActions+=ids.length;
    const a=legal(m,b,seq(r.actions.length),false),z=legal(m,b,ids,true);assert(a.length===z.length,"legal count mismatch");c.legalActions+=a.length;
    for(let j=0;j<a.length;j++){
     assert(a[j].j===z[j].j&&a[j].cost===z[j].cost&&a[j].cs.length===z[j].cs.length,"legal action mismatch");
     for(let l=0;l<a[j].cs.length;l++){assert(a[j].cs[l][0]===z[j].cs[l][0]&&a[j].cs[l][1]===z[j].cs[l][1],"child mismatch");c.childComparisons++;}
    }
    for(const a of z){
     if((36-a.cost)*t.mass<=v)continue;
     const x=-a.cost*t.mass+sum(a.cs.map(([,m])=>solve(m,b-a.cost)));if(x>v)v=x;
    }
   }memo.set(key,v);return v;
  }
  const raw=solve(ALL,r.credits);assert(raw===row.oracle.raw,"indexed oracle mismatch");c.oracleMatches++;
  rows.push({id:row.id,split:row.split,raw,utility:raw/sum(r.w),...c});
 }
 const fields=Object.keys(rows[0]).filter(k=>!["id","split","raw","utility","maxCandidateCount"].includes(k));
 const aggregate=["ordinary","ood","all"].map(split=>{
  const a=rows.filter(r=>split==="all"||r.split===split),totals=Object.fromEntries(fields.map(k=>[k,sum(a.map(r=>r[k]))]));
  const scanAndIndex=totals.indexedScans+totals.prefixTests+totals.indexReads+totals.indexEmits;
  const fullWithPartition=totals.fullScans+totals.fullPartitionTests,indexedWithPartition=scanAndIndex+totals.indexedPartitionTests;
  return{split,roots:a.length,...totals,maxCandidateCount:Math.max(...a.map(r=>r.maxCandidateCount)),meanCandidates:totals.eligibleActions/totals.positiveBudgetStates,scanReduction:1-totals.indexedScans/totals.fullScans,scanAndIndexReduction:1-scanAndIndex/totals.fullScans,scanPartitionAndIndexReduction:1-indexedWithPartition/fullWithPartition};
 });
 return{schema:"MTR-02E-gate-index-posthoc-v1",scope:"same public evaluation roots; exhaustive legal-action equivalence over visited exact-oracle states; no scheduler rerun or timing",aggregate,rows};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
 const r=JSON.parse(readFileSync(new URL("mtr02e_receipt.json",import.meta.url),"utf8"));
 console.log(JSON.stringify(mtr02eGateAudit(r),null,2));
}
