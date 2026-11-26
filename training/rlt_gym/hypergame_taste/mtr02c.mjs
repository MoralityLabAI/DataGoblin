// MTR-02C public-development nonlinear residual and cache experiment.
import { pathToFileURL } from "node:url";
export function mtr02c(seed=20261127,scale=1){
  const check=(b,m)=>{if(!b)throw Error(m);},seq=n=>Array.from({length:n},(_,i)=>i),sum=a=>a.reduce((x,y)=>x+y,0),mean=a=>sum(a)/a.length;
  check(Number.isSafeInteger(seed)&&Number.isInteger(scale)&&scale>=1&&scale<=3,"invalid run");
  const N=32,A=12,L=1<<A,F=24,H=8,I=32+A+32+A*35+2;
  const widths=[H],seeds=[19,37,73],modes=["state","rlt","damping","pooled","direct"],all=0xffffffff;
  function rng(key){let s=2166136261;for(const c of key)s=Math.imul(s^c.charCodeAt(0),16777619);s=s||1;return ()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return(s>>>0)/4294967296;};}
  function shuffle(a,r){a=[...a];for(let i=a.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
  function pop(a){let n=0;while(a){a=(a&(a-1))>>>0;n++;}return n;}
  const keys=new Set();let certChecks=0,nonlinearChecks=0,lowerBoundChecks=0,oracleReplays=0,oracleStates=0;
  function make(split,i){
    const id=`${seed}/mtr02c/${split}/${i}`,r=rng(id),ri=n=>Math.floor(r()*n),family=split==="ood"?"cubic":i%2?"quadratic_payload":"quadratic_selector",budget=6+i%3;
    const perm=shuffle(seq(5),r),flip=ri(32),mask=f=>seq(N).reduce((m,h)=>f(h)?(m|(1<<h))>>>0:m,0);
    function truth(h){
      const b=perm.map(j=>((h^flip)>>j)&1);
      const s=b[0]^(family==="cubic"?b[1]*b[2]*b[3]:family==="quadratic_payload"?b[2]*b[3]:b[1]*b[2]);
      const y=[b[1]^(family==="quadratic_payload"||family==="cubic"?b[3]*b[4]:0),b[2],b[3],b[4]];
      const q=y[0]^(y[1]*y[2]),t=y[1]^(y[2]*y[3]);return{s,y,q,t};
    }
    const worlds=seq(N).map(truth);
    check(new Set(worlds.map(w=>JSON.stringify([w.s,...w.y]))).size===N,"nonbijective witness encoding");
    const scout=mask(h=>worlds[h].s);
    // Every affine Boolean function on the original input has constant
    // derivative in each coordinate. Exhibit a changing derivative.
    check(seq(5).some(j=>new Set(seq(N).map(h=>worlds[h].s^worlds[h^(1<<j)].s)).size>1),"selector is affine");nonlinearChecks++;
    const pri=seq(N).map(h=>1+ri(4)+(i%4===0&&worlds[h].s===1?2:0));
    const p=[];const add=(name,cost,req,kind,fn)=>p.push({name,cost,req,kind,m:fn?mask(fn):0});
    add("scout",1+(split==="ood"&&i%4===0?1:0),0,0,h=>worlds[h].s);
    add("prepare_A",1+(i%7===0?1:0),0,1,null);add("prepare_B",1+(split==="ood"&&i%3===0?1:0),0,1,null);
    for(let s=0;s<2;s++)for(let j=0;j<4;j++){
      add(`${s?"B":"A"}_${j}`,1+(split==="ood"&&j===3&&i%3===1?1:0),1<<(s+1),2+s,h=>{
        const w=worlds[h];
        return w.s===s?w.y[j]:(split==="ood"&&j>=2?w.t:w.q)^(j&1);
      });
    }
    add("coarse",1+(i%5===0?1:0),0,4,h=>worlds[h].q);
    const cert=[];
    function witness(kind,value,good,bad,predicate){
      const m=mask(predicate);
      for(let h=0;h<N;h++){
        const w=truth(h);let v;
        if(kind==="full")v=h===value;
        else if(kind==="branch")v=(4*w.s+2*w.y[0]+w.y[1])===value;
        else v=w.q===value;
        check(v===Boolean(m>>>h&1),"certificate predicate mismatch");certChecks++;
      }
      cert.push({kind,value,good,bad,m});
    }
    for(let h=0;h<N;h++)witness("full",h,24,36,x=>x===h);
    for(let j=0;j<8;j++)witness("branch",j,12,18,h=>4*worlds[h].s+2*worlds[h].y[0]+worlds[h].y[1]===j);
    for(let j=0;j<2;j++)witness("coarse",j,10,15,h=>worlds[h].q===j);
    const signature=JSON.stringify([family,budget,perm,flip,pri,p.map(a=>[a.cost,a.req,a.kind,a.m])]);
    check(!keys.has(signature),"duplicate root");keys.add(signature);
    return{id,split,family,budget,pri,p,cert,scout};
  }
  function game(root){
    const{p,pri,cert,budget}=root,initial={m:all,u:0,b:budget,t:budget},mass=new Map(),term=new Map(),memo=new Map(),states=new Map(),plans=new Map();
    const W=m=>{m>>>=0;if(!mass.has(m)){let v=0;for(let j=0;j<N;j++)if(m>>>j&1)v+=pri[j];mass.set(m,v);}return mass.get(m);};
    const key=s=>`${s.m}/${s.u}/${s.b}/${s.t}`;
    function terminal(m){
      if(term.has(m))return term.get(m);let best={v:0,c:-1};
      for(let i=0;i<cert.length;i++){const c=cert[i],v=(c.good+c.bad)*W((m&c.m)>>>0)-c.bad*W(m);if(v>best.v)best={v,c:i};}
      term.set(m,best);return best;
    }
    function children(s,i){const a=p[i];return a.kind===1?[[-1,s.m]]:[[0,(s.m&~a.m)>>>0],[1,(s.m&a.m)>>>0]].filter(x=>x[1]);}
    function legal(s){return s.t>0?p.flatMap((a,i)=>!(s.u>>>i&1)&&(s.u&a.req)===a.req&&a.cost<=s.b&&(a.kind===1||children(s,i).length===2)?[i]:[]):[];}
    const next=(s,i,m)=>({m,u:s.u|(1<<i),b:s.b-p[i].cost,t:s.t-1});
    function solve(s){
      if(s.b<=0||s.t<=0)return{v:terminal(s.m).v,i:-1};
      const k=key(s);if(memo.has(k))return memo.get(k);let best={v:terminal(s.m).v,i:-1};
      for(const i of legal(s)){
        const v=children(s,i).reduce((v,[,m])=>v+solve(next(s,i,m)).v,-p[i].cost*W(s.m));
        if(v>best.v)best={v,i};
      }
      memo.set(k,best);states.set(k,s);return best;
    }
    const cost=new Uint16Array(L),need=new Uint16Array(L),counts=new Uint8Array(L),part=new Map([[0,[all]]]);
    for(let u=1;u<L;u++){const l=u&-u,j=31-Math.clz32(l),z=u^l;cost[u]=cost[z]+p[j].cost;need[u]=need[z]|p[j].req;counts[u]=counts[z]+1;}
    function partitions(u){
      // Setup actions have no observation. Collapse their partition keys.
      u&=~6;
      if(part.has(u))return part.get(u);
      const l=u&-u,j=31-Math.clz32(l),m=p[j].m,v=partitions(u^l).flatMap(x=>[(x&m)>>>0,(x&~m)>>>0].filter(Boolean));
      part.set(u,v);return v;
    }
    function plan(s){
      const k=key(s);if(plans.has(k))return plans.get(k);
      const ids=[-1,...legal(s)],v=[terminal(s.m).v,...ids.slice(1).map(()=>-Infinity)],first=ids.slice(1).reduce((m,i)=>m|(1<<i),0),available=(L-1)^s.u,W0=W(s.m);
      if(first)for(let u=available;u;u=(u-1)&available){
        if(!(u&first)||cost[u]>s.b||counts[u]>s.t||((need[u]&~(u|s.u))!==0))continue;
        let val=-cost[u]*W0;
        for(const m of partitions(u)){const z=(s.m&m)>>>0;if(z)val+=terminal(z).v;}
        for(let j=1;j<ids.length;j++)if((u>>>ids[j])&1&&val>v[j])v[j]=val;
      }
      const exact=ids.map(i=>i===-1?terminal(s.m).v:children(s,i).reduce((v,[,m])=>v+solve(next(s,i,m)).v,-p[i].cost*W0));
      for(let j=0;j<ids.length;j++){check(Number.isFinite(v[j])&&v[j]<=exact[j],"nonadaptive lower bound violated");lowerBoundChecks++;}
      const r={ids,base:v.map(x=>x/W0/24),target:exact.map(x=>x/W0/24)};
      plans.set(k,r);return r;
    }
    const pub=Array(I).fill(0);for(let j=0;j<N;j++)pub[32+A+j]=pri[j]/6;
    for(let j=0;j<A;j++){const off=32+A+32+j*35,a=p[j];pub[off]=a.cost/3;pub[off+1]=a.req/4;pub[off+2]=a.kind/4;for(let h=0;h<N;h++)pub[off+3+h]=a.m>>>h&1;}
    function token(s){const x=[...pub];for(let j=0;j<N;j++)x[j]=s.m>>>j&1;for(let j=0;j<A;j++)x[32+j]=s.u>>>j&1;x[I-2]=s.b/8;x[I-1]=s.t/8;return x;}
    function entropy(m){const w=W(m);let v=0;for(let h=0;h<N;h++)if(m>>>h&1){const z=pri[h]/w;v-=z*Math.log2(z);}return v;}
    function features(s,i,base){
      const v=Array(F).fill(0),w=W(s.m);v[0]=1;v[1]=Number(i<0);v[2]=s.b/8;v[3]=s.t/8;v[4]=base;v[5]=pop(s.m)/32;v[6]=terminal(s.m).v/w/24;
      if(i>=0){
        const a=p[i],c=children(s,i);v[7]=a.cost/3;v[8]=Number(a.kind===1);v[9]=Number(i===0);v[10]=Number(a.kind===4);v[11]=Number(a.kind===2);v[12]=Number(a.kind===3);
        v[13]=Number(Boolean(a.req&2));v[14]=Number(Boolean(a.req&4));v[15]=(entropy(s.m)-sum(c.map(([,m])=>W(m)/w*entropy(m))))/5;
        v[16]=sum(c.map(([,m])=>terminal(m).v))/w/24;v[17]=i/A;v[18]=Number(Boolean(s.u&2));v[19]=Number(Boolean(s.u&4));
        v[20]=a.kind===1?p.filter((z,j)=>!(s.u>>>j&1)&&Boolean(z.req&(1<<i))).length/4:0;
        v[21]=c.length===2?W(c[0][1])/w:1;v[22]=base-v[6];v[23]=Number(Boolean(s.u&1));
      }return v;
    }
    function step(s,i,h){
      const a=p[i],out=a.kind===1?-1:a.m>>>h&1,m=a.kind===1?s.m:out?(s.m&a.m)>>>0:(s.m&~a.m)>>>0;
      check(m>>>h&1,"truth lost");return next(s,i,m);
    }
    function examples(r){
      const rows=[],seen=new Set();
      function record(s,hist){const k=key(s);if(seen.has(k)||rows.length>=4)return;seen.add(k);const a=plan(s);rows.push({root:root.id,hist,ids:a.ids,base:a.base,y:a.target,x:a.ids.map((i,j)=>features(s,i,a.base[j]))});}
      for(let path=0;path<3&&rows.length<4;path++){
        let s=initial,h=Math.floor(r()*N),hist=[token(s)];
        for(let j=0;j<=5&&rows.length<4;j++){
          if(j===0||j===2||j===4||(path>0&&j===1))record(s,hist);
          const ls=legal(s);if(!ls.length)break;const a=plan(s);
          let i=path===0?solve(s).i:path===1?a.ids[a.base.indexOf(Math.max(...a.base))]:ls[Math.floor(r()*ls.length)];
          if(i<0)i=ls[Math.floor(r()*ls.length)];s=step(s,i,h);hist=[...hist,token(s)];
        }
      }return rows;
    }
    function heuristic(s,kind){
      let best=-1,v=kind==="myopic"?terminal(s.m).v:0;const w=W(s.m);
      for(const i of legal(s)){const a=p[i],cs=children(s,i),q=kind==="myopic"?sum(cs.map(([,m])=>terminal(m).v))-a.cost*w:
        (a.kind===1?0.2:(entropy(s.m)-sum(cs.map(([,m])=>W(m)/w*entropy(m)))))/a.cost-0.001*a.cost;
        if(q>v){v=q;best=i;}}
      return best;
    }
    function run(kind,model){
      let utility=0,cost=0,queries=0,pubMass=0,falseMass=0,fullMass=0;const w=W(all),cache=new Map();
      for(let h=0;h<N;h++){
        let s=initial,c=model?Array(H).fill(0):null,t=0,spent=0,q=0;
        while(true){
          let i;
          if(kind==="oracle")i=solve(s).i;
          else if(kind==="analytic"){const a=plan(s);i=a.ids[a.base.indexOf(Math.max(...a.base))];}
          else if(kind==="myopic"||kind==="information")i=heuristic(s,kind);
          else{
            const stateless=model.mode==="state"||model.mode==="direct",k=key(s)+(stateless?"":`/${t}/${c.join(",")}`);
            if(cache.has(k)){const a=cache.get(k);i=a.i;c=a.c;}
            else{
              const a=plan(s),z=advance(model,token(s),c,t),scores=predict(model,z.z,a.ids.map((i,j)=>features(s,i,a.base[j])),a.base,a.ids);
              i=a.ids[scores.indexOf(Math.max(...scores))];c=z.next;cache.set(k,{i,c});
            }
          }
          if(i<0){const a=terminal(s.m),v=a.c<0?null:cert[a.c],ok=v&&Boolean(v.m>>>h&1);
            utility+=pri[h]*((v?(ok?v.good:-v.bad):0)-spent);cost+=pri[h]*spent;queries+=pri[h]*q;if(v){pubMass+=pri[h];if(!ok)falseMass+=pri[h];if(v.kind==="full")fullMass+=pri[h];}break;}
          const a=p[i];check(s.t>0&&a.cost<=s.b&&!(s.u>>>i&1)&&(s.u&a.req)===a.req,"illegal replay");
          const z=step(s,i,h);check(a.kind===1||z.m!==s.m,"uninformative replay");spent+=a.cost;q+=Number(a.kind!==1);s=z;t++;
        }
      }
      if(kind==="oracle"){check(utility===solve(initial).v,"oracle replay mismatch");oracleReplays++;}
      else check(utility<=solve(initial).v,"policy beats oracle");
      return{utility:utility/w,cost:cost/w,queries:queries/w,publication:pubMass/w,falsePublication:falseMass/w,fullPublication:fullMass/w};
    }
    function cacheExperiment(){
      solve(initial);
      const r=rng(root.id+"/trace"),a=shuffle([...states.values()].filter(s=>s.u&&s.b>0&&s.t>0),r).slice(0,180);
      check(a.length>=80,"insufficient distinct states");
      const hot=[a.slice(0,20),a.slice(20,40)],cold=a.slice(40),trace=[];
      for(let phase=0;phase<2;phase++)for(let j=0;j<480;j++){
        const z=r(),pool=z<.8?hot[phase]:cold,index=z<.8?Math.floor(r()**2*pool.length):Math.floor(r()*pool.length);trace.push({phase,s:pool[index]});
      }
      // Independently implemented recursive value evaluator, with a fresh
      // private per-query memo. Charge actual expansions from each distinct
      // state on every simulated miss; this is not a timing experiment.
      const work=new Map();
      function fresh(s){
        const local=new Map();let expanded=0;
        function rec(m,u,b,t){
          let v=terminal(m).v;if(!b||!t)return v;
          const k=`${m}/${u}/${b}/${t}`;if(local.has(k))return local.get(k);
          for(let j=0;j<A;j++){
            const p0=p[j];if((u>>>j&1)||(u&p0.req)!==p0.req||p0.cost>b)continue;
            const a=p0.kind===1?[m]:[(m&p0.m)>>>0,(m&~p0.m)>>>0];
            if(a.some(x=>!x))continue;
            const q=sum(a.map(x=>rec(x,u|(1<<j),b-p0.cost,t-1)))-p0.cost*W(m);if(q>v)v=q;
          }
          local.set(k,v);expanded++;return v;
        }
        return{v:rec(s.m,s.u,s.b,s.t),expanded};
      }
      for(const {s} of trace){const k=key(s);if(!work.has(k)){const q=fresh(s);check(q.v===solve(s).v,"cache independent oracle");work.set(k,q);}}
      const result=[];
      for(const bytes of [0,1024,4096,16384])for(const mode of bytes===0?["none"]:["lru","frequency","work","utility"]){
        const useSketch=mode==="frequency"||mode==="utility",sketchBytes=useSketch?bytes/4:0,cap=Math.floor((bytes-sketchBytes)/32),buf=new ArrayBuffer(bytes),entries=new Uint32Array(buf,0,cap*8),sketch=useSketch?new Uint16Array(buf,cap*32,sketchBytes/2):null,sw=sketch?sketch.length/2:0;
        const phaseRows=seq(2).map(phase=>({phase,queries:0,hits:0,misses:0,expanded:0,keyComparisons:0,victimScans:0,metadataOps:0,admissions:0,evictions:0,verified:0}));
        const indices=(m,u)=>{const z=(Math.imul(m^u,0x9e3779b1)>>>0),v=(Math.imul((m>>>16)^u^m,0x85ebca6b)>>>0);return[z%sw,sw+v%sw];};
        function frequency(m,u,out){const [a,b]=indices(m,u);out.metadataOps+=2;return Math.min(sketch[a],sketch[b]);}
        function score(m,u,work,last,out){if(mode==="lru")return last;if(mode==="work")return work;const f=frequency(m,u,out);return mode==="frequency"?f:f*Math.max(1,work);}
        let clock=0,live=0,peak=0;
        for(const {phase,s} of trace){
          const out=phaseRows[phase],m=s.m>>>0,u=(s.u|(s.b<<12)|(s.t<<16))>>>0,k=key(s),expect=work.get(k);
          clock++;out.queries++;
          if(sketch){
            if(clock%128===0)for(let j=0;j<sketch.length;j++){sketch[j]>>>=1;out.metadataOps+=2;}
            for(const j of indices(m,u)){sketch[j]=Math.min(65535,sketch[j]+1);out.metadataOps+=2;}
          }
          let slot=-1,empty=-1;
          for(let j=0;j<cap;j++){const o=8*j;out.metadataOps++;if(!entries[o+5]){if(empty<0)empty=j;continue;}out.keyComparisons++;out.metadataOps+=2;if(entries[o]===m&&entries[o+1]===u){slot=j;break;}}
          let v;
          if(slot>=0){const o=slot*8;out.hits++;v=entries[o+2];entries[o+4]=clock;out.metadataOps+=2;}
          else{
            out.misses++;out.expanded+=expect.expanded;v=expect.v;
            let victim=-1,min=Infinity,old=Infinity;
            if(empty<0)for(let j=0;j<cap;j++){
              const o=j*8;out.victimScans++;out.metadataOps+=4;const q=score(entries[o],entries[o+1],entries[o+3],entries[o+4],out);
              if(q<min||q===min&&entries[o+4]<old){min=q;old=entries[o+4];victim=j;}
            }
            const admit=empty>=0||victim>=0&&(mode==="lru"||score(m,u,expect.expanded,clock,out)>=min),j=empty>=0?empty:victim;
            if(admit){const o=j*8;if(entries[o+5])out.evictions++;else live++;entries[o]=m;entries[o+1]=u;entries[o+2]=v;entries[o+3]=expect.expanded;entries[o+4]=clock;entries[o+5]=1;out.metadataOps+=7;out.admissions++;peak=Math.max(peak,live*32);}
          }
          check(v===solve(s).v,"cache answer changed");out.verified++;
        }
        result.push({root:root.id,split:root.split,bytes,mode,capacity:cap,sketchBytes,peakLiveEntryBytes:peak,distinctStates:work.size,phases:phaseRows});
      }
      return result;
    }
    function sameStatePairs(models){
      const pairs=[];
      for(const h of [0,7,15,31]){
        // Setup A and B commute; find an additional informative commuting pair.
        const a=step(initial,1,h),b=step(initial,2,h),ab=step(a,2,h),ba=step(b,1,h);
        check(key(ab)===key(ba),"commuting setups");const rootToken=token(initial);
        const h1=[rootToken,token(a),token(ab)],h2=[rootToken,token(b),token(ba)];pairs.push([ab,h1,h2]);
        const ls=legal(ab).filter(i=>p[i].kind!==1);
        outer:for(let i=0;i<ls.length;i++)for(let j=i+1;j<ls.length;j++){
          const x=step(ab,ls[i],h),y=step(ab,ls[j],h);if(!legal(x).includes(ls[j])||!legal(y).includes(ls[i]))continue;
          const xy=step(x,ls[j],h),yx=step(y,ls[i],h);check(key(xy)===key(yx),"probe commutation");
          pairs.push([xy,[...h1,token(x),token(xy)],[...h1,token(y),token(yx)]]);break outer;
        }
      }
      const out={pairs:pairs.length,metrics:{}};
      for(const mode of modes){
        let disagreements=0,gap=0,n=0;
        for(const model of models.filter(x=>x.mode===mode))for(const [s,h1,h2] of pairs){
          const a=plan(s),x=a.ids.map((i,j)=>features(s,i,a.base[j])),q1=predict(model,context(model,h1).z,x,a.base,a.ids),q2=predict(model,context(model,h2).z,x,a.base,a.ids);
          disagreements+=Number(q1.indexOf(Math.max(...q1))!==q2.indexOf(Math.max(...q2)));gap+=Math.max(...q1.map((v,i)=>Math.abs(v-q2[i])))*24;n++;
        }
        out.metrics[mode]={disagreement:disagreements/n,maxQDelta:gap/n};
      }return out;
    }
    return{examples,run,cacheExperiment,sameStatePairs,expansions:()=>memo.size,plannedStates:()=>plans.size};
  }
  const P=H*I+H*H+H+F+H*F,rec=H*I,bias=rec+H*H,off=bias+H;
  function project(x){
    const z=Array(H).fill(0);for(let i=0;i<x.length;i++)z[i%H]+=x[i]*(Math.floor(i/H)%2?-1:1);return z.map(v=>Math.tanh(v/Math.sqrt(I/H)));
  }
  function init(mode,s){
    const r=rng(`${seed}/mtr02c/model/${s}`),p=seq(P).map(()=>.07*(2*r()-1));
    return{mode,seed:s,p,m:Array(P).fill(0),v:Array(P).fill(0),step:0};
  }
  function advance(model,x,c,t){
    const p=model.p,stateless=model.mode==="state"||model.mode==="direct",raw=stateless?project(x):c,z=Array(H).fill(0);
    for(let j=0;j<H;j++){let a=p[bias+j];for(let k=0;k<I;k++)a+=p[j*I+k]*x[k];for(let k=0;k<H;k++)a+=p[rec+j*H+k]*raw[k];z[j]=Math.tanh(a);}
    let next=z;
    if(model.mode==="damping"||model.mode==="pooled"){const q=project(x);next=c.map((v,j)=>model.mode==="damping"?.5*v+.5*q[j]:(t*v+q[j])/(t+1));}
    return{x,raw,z,next};
  }
  function context(model,hist){
    const stateless=model.mode==="state"||model.mode==="direct";if(stateless)hist=[hist[hist.length-1]];
    let c=Array(H).fill(0),stack=[];
    for(let t=0;t<hist.length;t++){const a=advance(model,hist[t],c,t);stack.push(a);c=a.next;}return{stack,z:stack[stack.length-1].z};
  }
  function rawScores(model,z,x){
    const p=model.p,b=Array(F);
    for(let k=0;k<F;k++){let v=p[off+k];for(let j=0;j<H;j++)v+=z[j]*p[off+F+j*F+k];b[k]=v;}
    return x.map(f=>{let v=0;for(let k=0;k<F;k++)v+=f[k]*b[k];return v;});
  }
  function predict(model,z,x,base,ids){
    return rawScores(model,z,x).map((v,j)=>ids[j]<0?base[j]:model.mode==="direct"?v:base[j]+Math.max(0,v));
  }
  function update(model,e){
    const p=model.p,{stack,z}=context(model,e.hist),pred=rawScores(model,z,e.x),g=Array(P).fill(0),dh=Array(H).fill(0);let loss=0,n=0;
    for(let i=0;i<e.x.length;i++){
      if(e.ids[i]<0)continue;n++;const y=model.mode==="direct"?e.y[i]:e.y[i]-e.base[i],err=pred[i]-y,d=2*err/Math.max(1,e.x.length-1),f=e.x[i];loss+=err*err;
      for(let k=0;k<F;k++){g[off+k]+=d*f[k];for(let j=0;j<H;j++){g[off+F+j*F+k]+=d*z[j]*f[k];dh[j]+=d*p[off+F+j*F+k]*f[k];}}
    }
    let back=dh;
    for(let t=stack.length-1;t>=0;t--){
      const a=stack[t],d=a.z.map((v,j)=>back[j]*(1-v*v)),prev=Array(H).fill(0);
      for(let j=0;j<H;j++){g[bias+j]+=d[j];for(let k=0;k<I;k++)g[j*I+k]+=d[j]*a.x[k];for(let k=0;k<H;k++){g[rec+j*H+k]+=d[j]*a.raw[k];prev[k]+=d[j]*p[rec+j*H+k];}}
      if(model.mode!=="rlt")break;back=prev;
    }
    model.step++;const rate=.001*Math.sqrt(1-.999**model.step)/(1-.9**model.step),clip=Math.min(1,4/Math.max(1e-12,Math.sqrt(sum(g.map(v=>v*v)))));
    for(let j=0;j<P;j++){const v=clip*g[j];model.m[j]=.9*model.m[j]+.1*v;model.v[j]=.999*model.v[j]+.001*v*v;p[j]-=rate*model.m[j]/(Math.sqrt(model.v[j])+1e-8);}
    check(Number.isFinite(loss),"nonfinite training");return loss/Math.max(1,n);
  }
  const trainN=48*scale,evalN=24*scale,rows=[],models=[],fit=[];let plannedStates=0;
  for(let i=0;i<trainN;i++){const r=make("train",i),g=game(r),ex=g.examples(rng(r.id+"/teacher"));check(ex.length>=1&&ex.length<=4,"presentation cap");rows.push(...ex);oracleStates+=g.expansions();plannedStates+=g.plannedStates();}
  for(const s of seeds)for(const mode of modes){
    const model=init(mode,s),r=rng(`${seed}/mtr02c/shuffle/${s}`),loss=[];
    for(let epoch=0;epoch<9;epoch++){let l=0;for(const i of shuffle(seq(rows.length),r))l+=update(model,rows[i]);loss.push(l/rows.length);}
    models.push(model);fit.push({mode,seed:s,parameters:P,steps:model.step,firstLoss:loss[0],lastLoss:loss[loss.length-1]});
  }
  const metrics=["utility","cost","queries","publication","falsePublication","fullPublication"],rootResults=[],cacheRows=[];
  for(const split of ["ordinary","ood"])for(let i=0;i<evalN;i++){
    const r=make(split,i),g=game(r),result={};for(const mode of ["oracle","analytic","myopic","information"])result[mode]=g.run(mode,null);
    for(const mode of modes){const a=models.filter(m=>m.mode===mode).map(m=>g.run(mode,m));result[mode]={...Object.fromEntries(metrics.map(k=>[k,mean(a.map(v=>v[k]))])),seedUtility:a.map(v=>v.utility)};}
    const pairs=g.sameStatePairs(models);rootResults.push({id:r.id,split,family:r.family,budget:r.budget,result,pairs});
    if(i<3)cacheRows.push(...g.cacheExperiment());oracleStates+=g.expansions();plannedStates+=g.plannedStates();
  }
  const round=x=>Math.round(x*1e6)/1e6;
  function ci(a){const m=mean(a),sd=Math.sqrt(sum(a.map(x=>(x-m)**2))/Math.max(1,a.length-1));return[round(m),round(m-1.96*sd/Math.sqrt(a.length)),round(m+1.96*sd/Math.sqrt(a.length))];}
  const aggregate=["ordinary","ood"].map(split=>{
    const r=rootResults.filter(x=>x.split===split),means={},comparisons={},pairs={};
    for(const k of Object.keys(r[0].result))means[k]=Object.fromEntries(metrics.map(m=>[m,round(mean(r.map(x=>x.result[k][m])))]));
    for(const k of ["analytic","state","damping","pooled","direct"])comparisons[k]=Object.fromEntries(["utility","queries","falsePublication"].map(m=>[m,ci(r.map(x=>x.result.rlt[m]-x.result[k][m]))]));
    for(const k of modes)pairs[k]=Object.fromEntries(["disagreement","maxQDelta"].map(m=>[m,round(mean(r.map(x=>x.pairs.metrics[k][m])))]));
    return{split,roots:r.length,means,comparisons,pairs,sameStatePairs:sum(r.map(x=>x.pairs.pairs)),oracleAnalyticGap:ci(r.map(x=>x.result.oracle.utility-x.result.analytic.utility))};
  });
  const cacheAggregate=[];
  for(const bytes of [0,1024,4096,16384])for(const mode of bytes===0?["none"]:["lru","frequency","work","utility"]){
    const rs=cacheRows.filter(x=>x.bytes===bytes&&x.mode===mode),a={bytes,mode,roots:rs.length,capacity:rs[0].capacity,sketchBytes:rs[0].sketchBytes,phases:[]};
    for(let phase=0;phase<2;phase++){const q=rs.map(x=>x.phases[phase]),p0={phase};for(const k of Object.keys(q[0]).filter(k=>k!=="phase"))p0[k]=sum(q.map(x=>x[k]));a.phases.push(p0);}
    cacheAggregate.push(a);
  }
  const receipt={schema:"MTR-02C-public-development-v1",seed,scale,trainRoots:trainN,ordinaryRoots:evalN,oodRoots:evalN,uniqueRoots:keys.size,trainingRows:rows.length,certificateChecks:certChecks,nonlinearChecks,lowerBoundChecks,oracleReplays,oracleStates,plannedStates,widths,parameters:P,modelSeeds:seeds,fit,aggregate,cacheAggregate,cacheRows,rootResults};
  return{receipt,weights:models.map(m=>({mode:m.mode,seed:m.seed,p:m.p})),rows};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
 const a=process.argv.slice(2).filter(x=>!x.startsWith("--")).map(Number),r=mtr02c(...a);
 console.log(JSON.stringify(process.argv.includes("--weights")?r.weights:process.argv.includes("--rows")?r.rows:r.receipt,null,2));
}
