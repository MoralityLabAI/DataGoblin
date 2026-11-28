import {readFileSync,writeFileSync,mkdirSync,realpathSync} from "node:fs";
import {resolve,dirname,join,sep} from "node:path";
import {fileURLToPath} from "node:url";
import {execFileSync} from "node:child_process";
import {createHash} from "node:crypto";
import {gunzipSync} from "node:zlib";

function normalizeMtr(rows, letter) {
  const counts={g:128,h:128,i:320};
  if(!Array.isArray(rows)||rows.length!==counts[letter]) throw Error("source count");
  const roots=new Map();
  const allowed={g:["root","type","stream","x","quantum","work","censored","initialLower","initialGap","gain","gap"],h:["root","stream","quantum","work","censored","gain","gap","jobs","actionJobs","boundJobs","initialLower","initialGap","x"],i:["root","prefix","prefixDigest","prefixWork","prefixLower","prefixGap","x","stream","first","delayedGain","delayedGap","continuationWork","continuationJobs","censoredOptions","utility","falsePublication","gain","gap"]};
  const keys=allowed[letter];
  const seen=new Set();
  for(const row of rows){
    if(!row||Object.keys(row).some(k=>!keys.includes(k))||!keys.every(k=>k in row))throw Error("schema drift "+letter);
    if(typeof row.root!=="string"||!new RegExp("^20261127/mtr02"+letter+"/train/[0-9]+$").test(row.root))throw Error("nontrain root");
    if(!Array.isArray(row.x)||!row.x.every(Number.isFinite))throw Error("bad features");
    const state=letter==="i"?row.prefix:null;
    const key=JSON.stringify([row.root,state,row.stream,letter==="g"?row.type:null,letter==="i"?null:row.quantum]);
    if(seen.has(key))throw Error("duplicate option "+key);
    seen.add(key);
    const action={stream:row.stream,...(letter==="g"?{type:row.type}:{}),...(letter!=="i"?{quantum:row.quantum}:{quantum:row.first.quantum})};
    if(!Number.isInteger(row.stream)||!Number.isInteger(action.quantum)||action.quantum<=0)throw Error("bad action");
    const observation={features:row.x,...(letter==="i"?{prefix:row.prefix,prefix_digest:row.prefixDigest,prefix_work:row.prefixWork,lower:row.prefixLower,gap:row.prefixGap}:{lower:row.initialLower,gap:row.initialGap})};
    const excluded=new Set(["root","x","type","stream","quantum","initialLower","initialGap","prefix","prefixDigest","prefixWork","prefixLower","prefixGap"]);
    const outcome=Object.fromEntries(Object.entries(row).filter(([k])=>!excluded.has(k)));
    if(letter==="i"&&(row.first.root!==row.root||row.first.stream!==row.stream))throw Error("first action identity");
    if(!roots.has(row.root))roots.set(row.root,[]);
    roots.get(row.root).push({observation,action,outcome});
  }
  if(roots.size!==32||[...roots.values()].some(x=>x.length!==(letter==="i"?10:4)))throw Error("root roster differs");
  return [...roots.entries()].sort(([a],[b])=>a.localeCompare(b)).map(([root,options])=>({schema:"dg.research.observed-options.v1",canonical_root:root,split:"train_candidate",program:"MTR-02"+letter.toUpperCase(),source_kind:"finite_research_game_observations",feature_contract:"source-native numeric feature order; review before language rendering",claim_scope:"reported finite-game outcomes; not independent natural-language proof",training_ready:false,options}));
}

const sha = b => createHash("sha256").update(b).digest("hex");
function git(cwd,args) {
  return execFileSync("git", args, {cwd,maxBuffer:128*1024*1024,stdio:["ignore","pipe","pipe"]});
}
function readGit(cwd, commit, path, expectedBlob) {
  const identity=commit+":"+path;
  const blob=git(cwd,["rev-parse",identity]).toString().trim();
  if(expectedBlob&&blob!==expectedBlob)throw Error("Git blob mismatch: "+path);
  const bytes=git(cwd,["cat-file","blob",blob]);
  return {blob,bytes};
}
function canonical(value) {
  if(value===null||typeof value!=="object")return JSON.stringify(value);
  if(Array.isArray(value))return "["+value.map(canonical).join(",")+"]";
  return "{"+Object.keys(value).sort().map(k=>JSON.stringify(k)+":"+canonical(value[k])).join(",")+"}";
}
function candidateRows(buf,format) {
  const raw=buf.toString("utf8");
  if(format==="json-array")return JSON.parse(raw);
  if(format==="jsonl.gz")return raw.split(/\r?\n/).filter(x=>x.trim()).map(line=>JSON.parse(line));
  throw Error("unrecognized format");
}
function main() {
  const args=process.argv.slice(2);
  if(args.length!==4||args[0]!=="--repos"||args[2]!=="--out")throw Error("usage: node stage_recent.mjs --repos local_repos.json --out /absolute/custody/new-dir");
  const repos=JSON.parse(readFileSync(args[1],"utf8"));
  const out=resolve(args[3]);
  const plan=JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)),"sources.json"),"utf8"));
  const homes=Object.values(repos).map(p=>realpathSync(p));
  const ownRoot=git(dirname(fileURLToPath(import.meta.url)),["rev-parse","--show-toplevel"]).toString().trim();
  const dest=join(realpathSync(dirname(out)),out.split(sep).pop());
  if([...homes,ownRoot].some(p=>dest===p||dest.startsWith(p+sep)))throw Error("use a new custody directory outside every source checkout");
  mkdirSync(dest,{recursive:false,mode:0o700});
  const ledger=[],global=new Set(),mtr=[];
  const result={schema:"dg.research20m.stage-receipt.v1",training_ready:false,status:"candidate_only",accepted_unique_aleph_tokens:null,reported_target:plan.target_unique_aleph_tokens,source_files:ledger,errors:[]};
  for(const e of plan.exports) {
    try {
      const cwd=repos[e.repository];if(!cwd)throw Error("missing repo mapping");
      if(e.manifest)readGit(cwd,e.commit,e.manifest,e.manifest_git_blob);
      const {blob,bytes}=readGit(cwd,e.commit,e.path,e.git_blob);
      const decoded=e.format==="jsonl.gz"?gunzipSync(bytes,{maxOutputLength:128*1024*1024}):bytes;
      if(e.sha256&&sha(e.hash_scope==="decompressed"?decoded:bytes)!==e.sha256)throw Error("source SHA256 mismatch");
      const rows=candidateRows(decoded,e.format);
      if(!Array.isArray(rows)||rows.length!==e.reported_rows||rows.some(r=>!r||typeof r!=="object"||Array.isArray(r)))throw Error("row count or schema differs");
      const keys=new Set(),within=new Set();let crossDuplicate=0;
      const rootFields=new Map(),splitFields=new Map();
      for(const row of rows) {
        for(const k of Object.keys(row)){
          keys.add(k);
          if(/^(root|root_id|canonical_root|source_root|task_id)$/i.test(k)&&typeof row[k]==="string"){
            if(!rootFields.has(k))rootFields.set(k,new Set());rootFields.get(k).add(row[k]);
          }
          if(/^(split|partition|role)$/i.test(k)&&typeof row[k]==="string"){
            if(!splitFields.has(k))splitFields.set(k,new Set());splitFields.get(k).add(row[k]);
          }
        }
        const hash=sha(canonical(row));
        if(global.has(hash)&&!within.has(hash))crossDuplicate++;
        within.add(hash);
      }
      let normalized=null;
      if(e.kind==="paired_observed_outcomes"){
        const letter=e.namespace.slice(-1);normalized=normalizeMtr(rows,letter);
        for(const r of normalized)mtr.push({...r,source:{repository:e.repository,commit:e.commit,path:e.path,blob}});
      }
      const receipt={repository:e.repository,commit:e.commit,path:e.path,git_blob:blob,raw_sha256:sha(bytes),decoded_sha256:sha(decoded),decoded_bytes:decoded.length,rows:rows.length,exact_distinct_rows:within.size,cross_file_duplicate_rows:crossDuplicate,top_level_keys:[...keys].sort(),observed_root_counts:Object.fromEntries([...rootFields].map(([k,v])=>[k,v.size])),observed_split_values:Object.fromEntries([...splitFields].map(([k,v])=>[k,[...v].sort()])),normalized_root_rows:normalized?.length??null,training_ready:false};
      for(const h of within)global.add(h);
      ledger.push(receipt);
    } catch(err) {
      result.errors.push({repository:e.repository,path:e.path,error:String(err.message||err)});
    }
  }
  const mtrText=mtr.map(r=>JSON.stringify(r)).join("\n")+(mtr.length?"\n":"");
  writeFileSync(join(dest,"mtr_grouped_candidates.jsonl"),mtrText,{flag:"wx",mode:0o600});
  result.mtr_grouped_candidate_roots=mtr.length;
  result.mtr_sha256=sha(mtrText);
  result.read_rows=ledger.reduce((a,x)=>a+x.rows,0);
  result.exact_distinct_raw_rows=global.size;
  result.limitations=["Raw row hashes do not prove unique task roots or no semantic overlap.","Feature/target separation and lineage review remain mandatory; grouped MTR rows are not training-ready.","No rights, prior pilot duplicate, held-family, or tokenizer admission is asserted.","Raw compressed candidate payloads stay in the pinned Git object stores; only MTR grouped candidates are written locally."];
  writeFileSync(join(dest,"STAGE_RECEIPT.json"),JSON.stringify(result,null,2)+"\n",{flag:"wx",mode:0o600});
  console.log(JSON.stringify({status:result.status,read_rows:result.read_rows,distinct_raw_rows:global.size,mtr_roots:mtr.length,errors:result.errors.length,training_ready:false}));
  if(result.errors.length)process.exitCode=1;
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))main();
export {normalizeMtr};
