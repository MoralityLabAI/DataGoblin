import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
export function mtr02bAffine(receipt){
  const assert=(v,m)=>{if(!v)throw Error(m);},sum=a=>a.reduce((x,y)=>x+y,0),seq=n=>Array.from({length:n},(_,i)=>i);
  function rng(key){let s=2166136261;for(const c of key)s=Math.imul(s^c.charCodeAt(0),16777619);s=s||1;return ()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return (s>>>0)/4294967296;};}
  function shuffle(a,r){a=[...a];for(let i=a.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
  const parity=x=>{let p=0;for(;x;x&=x-1)p^=1;return p;};
  const result={schema:"MTR-02B-affine-audit-v1",scope:"post-hoc independent algebraic oracle and costed-basis control on existing public roots",roots:[],oracleChecks:0,stateChecks:0,transitionChecks:0,planWitnessChecks:0};
  for(const row of receipt.rootResults){
    const {id,split,family,d,budget}=row,i=Number(id.split("/").at(-1)),r=rng(id),ri=n=>Math.floor(r()*n),D=d+1,n=2**D,rhs=1<<D,coeff=rhs-1;
    let basis=seq(d).map(j=>1<<j);for(let k=0;k<4*d;k++){const a=ri(d),b=ri(d);if(a!==b)basis[a]^=basis[b];}
    const encode=h=>family==="lp"?h:family==="affine"?sum(basis.map((a,j)=>parity(a&h)<<j)):sum(seq(d).map(j=>((h>>j&1)^(j>=2?(h>>(j-1)&1)*(h>>(j-2)&1):0))<<j));
    const worlds=seq(n).map(j=>({h:j>>1,v:encode(j>>1)|((j&1)<<d),w:1+ri(4)+(split==="ood"&&(j&1)===0?2:0)}));
    const forms=shuffle(basis,r),a=[];
    function add(cost,req,form=0,noisy=0,setup=0){a.push({cost,req,form:form|(noisy<<d),setup});}
    add(1+(split==="ood"&&i%3===0?1:0),0,0,0,1);add(1+ri(2),0,0,0,1);
    add(1,1,forms[0]);add(1,1,forms[1]);add(1,1,forms[2],i%4===0?1:0);
    add(1,2,forms[2]);add(1,2,forms[3]);add(1,2,d===5?forms[4]:forms[0]^forms[3]);
    add(2+ri(2),0,forms[d-1]);add(2+ri(2),0,forms[0]^forms[1]);add(1,0,forms[0],1);
    if(split==="ood"&&i%2===0){a[4].req=3;a[6].cost=2;}
    const empty=Array(D).fill(0),bits=D+1;
    function insert(B,v){
      const C=[...B];for(let j=D-1;j>=0;j--)if(C[j]&&(v>>j&1))v^=C[j];
      if(!(v&coeff))return v&rhs?null:C;
      let j=D-1;while(!(v>>j&1))j--;
      for(let k=0;k<D;k++)if(C[k]>>j&1)C[k]^=v;C[j]=v;
      return C;
    }
    const pack=B=>sum(B.map((x,j)=>x*2**(bits*j))),zero=B=>B.every(x=>x===0);
    const fibreCache=new Map(),memo=new Map(),posteriorToBasis=new Map(),basisToPosterior=new Map();
    function fibre(B){
      const k=pack(B);if(fibreCache.has(k))return fibreCache.get(k);
      const w=worlds.filter(x=>B.every(v=>parity((v&coeff)&x.v)===Number(Boolean(v&rhs))));
      const count=B.filter(Boolean).length;assert(w.length===2**(D-count),"affine fibre size");
      const mass=sum(w.map(x=>x.w)),m=w.reduce((m,x)=>m+(1n<<BigInt((x.h<<1)|(x.v>>d))),0n).toString();
      assert(!posteriorToBasis.has(m)||posteriorToBasis.get(m)===k,"distinct RREF for same posterior");
      assert(!basisToPosterior.has(k)||basisToPosterior.get(k)===m,"packed collision");
      posteriorToBasis.set(m,k);basisToPosterior.set(k,m);
      const hMass=Array(2**d).fill(0);for(const x of w)hMass[x.h]+=x.w;
      const terminal=Math.max(0,50*Math.max(...hMass)-30*mass),f={w,mass,terminal};fibreCache.set(k,f);result.stateChecks++;
      return f;
    }
    function solve(B,u,b,t){
      const k=`${pack(B)}/${u}/${b}/${t}`;if(memo.has(k))return memo.get(k);
      const f=fibre(B);let best=f.terminal;
      if(b>0&&t>0)for(let j=0;j<a.length;j++){
        const p=a[j];if((u>>j&1)||(u&p.req)!==p.req||p.cost>b)continue;
        let children;
        if(p.setup)children=[B];else{
          const x=insert(B,p.form),y=insert(B,p.form|rhs);
          if(!x||!y)continue;children=[x,y];
        }
        const cc=children.map(fibre);assert(sum(cc.map(x=>x.mass))===f.mass,"mass conservation");
        if(!p.setup){
          for(let z=0;z<2;z++)assert(cc[z].w.every(x=>parity(x.v&p.form)===z),"syndrome outcome");
        }
        result.transitionChecks++;
        const v=-p.cost*f.mass+sum(children.map(C=>solve(C,u|(1<<j),b-p.cost,t-1)));
        if(v>best)best=v;
      }
      memo.set(k,best);return best;
    }
    assert(zero(empty),"empty basis");const total=fibre(empty).mass,utility=solve(empty,0,budget,budget)/total;
    assert(Math.abs(utility-row.result.oracle.utility)<1e-10,"independent affine DP differs from committed bitset oracle");result.oracleChecks++;
    const plannerMemo=new Map();
    function plan(B,u,b){
      // The all-zero-syndrome basis is enough for rank and safe recovery.
      const k=`${pack(B)}/${u}/${b}`;if(plannerMemo.has(k))return plannerMemo.get(k);
      const rk=B.filter(Boolean).length;
      if(seq(d).every(j=>insert(B,1<<j).filter(Boolean).length===rk))return {cost:0,actions:[]};
      let best={cost:Infinity,actions:[]};
      for(let j=0;j<a.length;j++){
        const p=a[j];if((u>>j&1)||(u&p.req)!==p.req||p.cost>b)continue;
        const C=p.setup?B:insert(B,p.form);if(!p.setup&&pack(C)===pack(B))continue;
        const next=plan(C,u|(1<<j),b-p.cost);
        if(p.cost+next.cost<best.cost)best={cost:p.cost+next.cost,actions:[j,...next.actions]};
      }
      plannerMemo.set(k,best);return best;
    }
    const best=plan(empty,0,budget),safeUtility=Number.isFinite(best.cost)?20-best.cost:0;
    if(Number.isFinite(best.cost))for(const w of worlds){
      let B=empty,u=0,cost=0;
      for(const j of best.actions){
        const p=a[j];assert(!(u>>j&1)&&(u&p.req)===p.req,"safe planner prerequisite");u|=1<<j;cost+=p.cost;
        if(!p.setup)B=insert(B,p.form|(parity(w.v&p.form)?rhs:0));
      }
      assert(cost===best.cost&&cost<=budget&&fibre(B).w.every(x=>x.h===w.h),"safe plan witness");result.planWitnessChecks++;
    }
    assert(safeUtility<=utility+1e-10,"safe policy exceeds oracle");
    result.roots.push({id,split,d,budget,oracleUtility:utility,safeUtility,planCost:Number.isFinite(best.cost)?best.cost:null,planActions:best.actions,
      posteriorMaskBits:n,canonicalBasisBits:D*bits,affineStates:memo.size,distinctFibres:fibreCache.size,plannerStates:plannerMemo.size});
  }
  result.aggregate=["ordinary","ood"].map(split=>{const a=result.roots.filter(x=>x.split===split);return {split,roots:a.length,oracleUtility:sum(a.map(x=>x.oracleUtility))/a.length,safeUtility:sum(a.map(x=>x.safeUtility))/a.length,exactTies:a.filter(x=>Math.abs(x.safeUtility-x.oracleUtility)<1e-10).length,affineStates:sum(a.map(x=>x.affineStates)),distinctFibres:sum(a.map(x=>x.distinctFibres)),plannerStates:sum(a.map(x=>x.plannerStates))};});
  return result;
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
 const r=JSON.parse(readFileSync(new URL("mtr02b_receipt.json",import.meta.url),"utf8"));
 console.log(JSON.stringify(mtr02bAffine(r),null,2));
}
