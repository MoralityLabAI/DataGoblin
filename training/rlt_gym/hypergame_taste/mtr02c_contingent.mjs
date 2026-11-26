// Post-hoc contingent planner audit on the existing public MTR-02C roots.
import {readFileSync} from "node:fs";
import {pathToFileURL} from "node:url";
export function mtr02cContingent(receipt){
 const assert=(b,m)=>{if(!b)throw Error(m)},seq=n=>Array.from({length:n},(_,i)=>i),sum=a=>a.reduce((a,b)=>a+b,0),mean=a=>sum(a)/a.length;
 function rng(key){let s=2166136261;for(const c of key)s=Math.imul(s^c.charCodeAt(0),16777619);s=s||1;return()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return(s>>>0)/4294967296;};}
 function shuffle(a,r){a=[...a];for(let i=a.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
 const N=32,A=12,L=1<<A,all=0xffffffff,rows=[];let boundChecks=0,replayChecks=0,partitionChecks=0;
 for(const row of receipt.rootResults){
  const {id,split,family,budget}=row,i=Number(id.split("/").at(-1)),r=rng(id),ri=n=>Math.floor(r()*n),perm=shuffle(seq(5),r),flip=ri(32);
  function truth(h){const b=perm.map(j=>((h^flip)>>j)&1),s=b[0]^(family==="cubic"?b[1]*b[2]*b[3]:family==="quadratic_payload"?b[2]*b[3]:b[1]*b[2]),y=[b[1]^(family==="quadratic_payload"||family==="cubic"?b[3]*b[4]:0),b[2],b[3],b[4]];return{s,y,q:y[0]^(y[1]*y[2]),t:y[1]^(y[2]*y[3])};}
  const w=seq(N).map(truth),weights=seq(N).map(h=>1+ri(4)+(i%4===0&&w[h].s===1?2:0)),actions=[];
  function add(cost,req,setup,fn){actions.push({cost,req,setup,mask:fn?seq(N).reduce((m,h)=>fn(w[h])?(m|(1<<h))>>>0:m,0):0});}
  add(1+(split==="ood"&&i%4===0?1:0),0,false,x=>x.s);
  add(1+(i%7===0?1:0),0,true,null);add(1+(split==="ood"&&i%3===0?1:0),0,true,null);
  for(let s=0;s<2;s++)for(let j=0;j<4;j++)add(1+(split==="ood"&&j===3&&i%3===1?1:0),1<<(s+1),false,x=>x.s===s?x.y[j]:(split==="ood"&&j>=2?x.t:x.q)^(j&1));
  add(1+(i%5===0?1:0),0,false,x=>x.q);
  const massMemo=new Map(),terminalMemo=new Map(),exactMemo=new Map(),openMemo=new Map(),cont=[new Map(),new Map(),new Map()],partMemo=new Map([[0,[all]]]);
  function mass(m){if(massMemo.has(m))return massMemo.get(m);const v=sum(seq(N).map(h=>(m>>>h&1)*weights[h]));massMemo.set(m,v);return v;}
  function terminal(m){
   if(terminalMemo.has(m))return terminalMemo.get(m);
   const M=mass(m),branches=Array(8).fill(0),coarse=[0,0];let best={value:0,kind:"abstain",claim:-1};
   function offer(value,kind,claim){if(value>best.value)best={value,kind,claim};}
   for(let h=0;h<N;h++)if(m>>>h&1){offer(60*weights[h]-36*M,"full",h);const x=w[h];branches[4*x.s+2*x.y[0]+x.y[1]]+=weights[h];coarse[x.q]+=weights[h];}
   for(let j=0;j<8;j++)offer(30*branches[j]-18*M,"branch",j);
   for(let j=0;j<2;j++)offer(25*coarse[j]-15*M,"coarse",j);
   terminalMemo.set(m,best);return best;
  }
  const key=(m,u)=>`${m}/${u}`,costs=new Uint16Array(L),need=new Uint16Array(L);
  for(let u=1;u<L;u++){const l=u&-u,j=31-Math.clz32(l);costs[u]=costs[u^l]+actions[j].cost;need[u]=need[u^l]|actions[j].req;}
  function legal(m,u,b){return seq(A).filter(j=>{const a=actions[j];return !(u>>>j&1)&&(u&a.req)===a.req&&a.cost<=b&&(a.setup||((m&a.mask)>>>0)!==0&&((m&~a.mask)>>>0)!==0);});}
  function children(m,j){const a=actions[j];return a.setup?[m]:[(m&~a.mask)>>>0,(m&a.mask)>>>0].filter(Boolean);}
  function exact(m,u,b){
   const k=key(m,u);if(exactMemo.has(k))return exactMemo.get(k);
   let out={value:terminal(m).value,action:-1};
   for(const j of legal(m,u,b)){const a=actions[j],v=sum(children(m,j).map(x=>exact(x,u|(1<<j),b-a.cost).value))-a.cost*mass(m);if(v>out.value)out={value:v,action:j};}
   exactMemo.set(k,out);return out;
  }
  function partitions(u){
   u&=~6;if(partMemo.has(u))return partMemo.get(u);
   const l=u&-u,j=31-Math.clz32(l),v=partitions(u^l).flatMap(m=>children(m,j));
   assert(sum(v.map(m=>mass(m)))===mass(all),"partition mass");partitionChecks++;partMemo.set(u,v);return v;
  }
  function open(m,u,b){
   const k=key(m,u);if(openMemo.has(k))return openMemo.get(k);
   const first=legal(m,u,b),f=first.reduce((z,j)=>z|(1<<j),0),avail=(L-1)^u,M=mass(m);let best={value:terminal(m).value,action:-1};
   if(f)for(let sub=avail;sub;sub=(sub-1)&avail){
    if(!(f&sub)||costs[sub]>b||need[sub]&~(sub|u))continue;
    let v=-costs[sub]*M;for(const p of partitions(sub)){const x=(p&m)>>>0;if(x)v+=terminal(x).value;}
    const j=31-Math.clz32((f&sub)&-(f&sub));if(v>best.value||(v===best.value&&best.action>=0&&j<best.action))best={value:v,action:j};
   }
   openMemo.set(k,best);return best;
  }
  function look(m,u,b,k){
   if(k===0)return open(m,u,b);const id=key(m,u),memo=cont[k];if(memo.has(id))return memo.get(id);
   let out={value:terminal(m).value,action:-1};
   for(const j of legal(m,u,b)){const a=actions[j],v=sum(children(m,j).map(x=>look(x,u|(1<<j),b-a.cost,k-1).value))-a.cost*mass(m);if(v>out.value)out={value:v,action:j};}
   assert(out.value>=look(m,u,b,k-1).value&&out.value<=exact(m,u,b).value,"contingent bound");boundChecks++;memo.set(id,out);return out;
  }
  function replay(kind){
   let util=0,cost=0,queries=0,falseMass=0;
   for(let h=0;h<N;h++){
    let m=all,u=0,b=budget,spent=0,qs=0;
    while(true){
     const out=kind==="exact"?exact(m,u,b):look(m,u,b,kind),j=out.action;
     if(j<0){const c=terminal(m),x=w[h],good=c.kind==="full"?c.claim===h:c.kind==="branch"?c.claim===4*x.s+2*x.y[0]+x.y[1]:c.kind==="coarse"?c.claim===x.q:true,gain=c.kind==="full"?(good?24:-36):c.kind==="branch"?(good?12:-18):c.kind==="coarse"?(good?10:-15):0;util+=weights[h]*(gain-spent);cost+=weights[h]*spent;queries+=weights[h]*qs;if(!good)falseMass+=weights[h];break;}
     const a=actions[j];assert(legal(m,u,b).includes(j),"illegal action");m=a.setup?m:(a.mask>>>h&1)?(m&a.mask)>>>0:(m&~a.mask)>>>0;assert(m>>>h&1,"latent truth lost");b-=a.cost;spent+=a.cost;qs+=Number(!a.setup);u|=1<<j;
    }
   }
   const total=mass(all),bound=kind==="exact"?exact(all,0,budget).value:look(all,0,budget,kind).value;
   assert(util>=bound&&util<=exact(all,0,budget).value,"policy bound");replayChecks++;
   return{utility:util/total,lowerBound:bound/total,cost:cost/total,queries:queries/total,falsePublication:falseMass/total};
  }
  const result={exact:replay("exact"),open:replay(0),contingent1:replay(1),contingent2:replay(2)};
  assert(Math.abs(result.exact.utility-row.result.oracle.utility)<1e-10,"oracle root mismatch");
  assert(Math.abs(result.open.utility-row.result.analytic.utility)<1e-10,"analytic root mismatch");
  rows.push({id,split,budget,result,states:{exact:exactMemo.size,open:openMemo.size,contingent1:cont[1].size,contingent2:cont[2].size}});
 }
 return{schema:"MTR-02C-contingent-posthoc-v1",sourceReceiptSchema:receipt.schema,scope:"post-hoc development control; no training; regenerated same public roots",boundChecks,replayChecks,partitionChecks,rows,aggregate:["ordinary","ood"].map(split=>{const a=rows.filter(r=>r.split===split),out={split,roots:a.length};for(const k of ["exact","open","contingent1","contingent2"]){out[k]=Object.fromEntries(["utility","lowerBound","cost","queries","falsePublication"].map(m=>[m,mean(a.map(r=>r.result[k][m]))]));out[k].oracleTies=a.filter(r=>Math.abs(r.result[k].utility-r.result.exact.utility)<1e-10).length;}return out;})};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
 const r=JSON.parse(readFileSync(new URL("mtr02c_receipt.json",import.meta.url),"utf8"));
 console.log(JSON.stringify(mtr02cContingent(r),null,2));
}
