"""Generate V19 from pinned native V18 source."""
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
def sub(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a))
 s=s.replace(a,b)
sub('// ALG-DESCENT-18: budget-aware certified refinement; C++17.','// ALG-DESCENT-19: exact goal-support restriction; C++17.')
sub('U32 s=0,serial=0;U16 ri=', 'U32 s=0,serial=0,support=0;U16 ri=')
sub('RSTORE,RDSCAN,NC','RSTORE,RDSCAN,FSCAN,SHEAD,ISCAN,NC')
sub('"residual_entries","recomputed_scans"};','"residual_entries","recomputed_scans","ff_scans","support_heads","index_scans"};')
sub('std::array<U32,N> desc{};','std::array<U32,N> desc{},prem_union{};')
sub('g(gg),arm(a==0?0:2),policy(a),direct(a!=0),defer(a!=0),packed(true)','g(gg),arm(0),policy(a),direct(false),defer(false),packed(true)')
sub('log(l){if(arm==2)', 'log(l){if(policy>=2){for(int i=0;i<g.nr;i++){prem_union[g.rules[i].head]|=g.rules[i].prem;c[ISCAN]++;}}if(arm==2)')
sub('void distance(U32 s,Residual& q,U32 dirty)', '''U32 live(U32 s){U32 todo=g.goalbit&~s,cone=0;while(todo){int h=31-__builtin_clz(todo);U32 b=1u<<h;todo^=b;cone|=b;c[SHEAD]++;todo|=prem_union[h]&~s&~cone;}return cone;}
U32 support(Node& n,bool for_expansion){if(policy==1)return g.relevant&~n.s;if(policy==3){if(!n.support)n.support=live(n.s);return n.support;}if(policy==2&&for_expansion)return live(n.s);return ((1u<<g.n)-1)&~n.s;}
void distance(U32 s,Residual& q,U32 dirty)''')
sub('distance(n.s,q,(1u<<g.n)-1);n.b=', 'distance(n.s,q,support(n,false));n.b=')
sub('std::array<int,N> v{},chosen{};for(int h=3;h<g.n;h++)if(!(n.s&(1u<<h)))','std::array<int,N> v{},chosen{};U32 eligible=support(n,false);for(int h=3;h<g.n;h++)if(eligible&(1u<<h))')
sub('int a=g.rules[i].cost;for(U32 p=', 'c[FSCAN]++;int a=g.rules[i].cost;for(U32 p=')
sub('auto w=c[PROBES]+c[DSCAN]+c[BSCAN]+c[MSCAN];','auto w=c[PROBES]+c[DSCAN]+c[FSCAN]+c[ISCAN];')
sub('c[PROBES]+=g.nr;for(int h=3;h<g.n;h++)if(!(n.s&(1u<<h))){int best=INF,rule=-1;for(int i=g.begin[h];i<g.begin[h+1];i++)if((n.s&g.rules[i].prem)==g.rules[i].prem&&g.rules[i].cost<best){best=g.rules[i].cost;rule=i;}','U32 eligible=support(nodes[id],true);for(int h=3;h<g.n;h++)if(eligible&(1u<<h)){int best=INF,rule=-1;for(int i=g.begin[h];i<g.begin[h+1];i++){c[PROBES]++;if((n.s&g.rules[i].prem)==g.rules[i].prem&&g.rules[i].cost<best){best=g.rules[i].cost;rule=i;}}')
sub('std::array<Result,6> first{};std::array<std::vector<U64>,6> times;std::array<int,6> order{0,1,2,3,4,5};','std::array<Result,4> first{};std::array<std::vector<U64>,4> times;std::array<int,4> order{0,1,2,3};')
sub('for(int arm=0;arm<6;arm++)','for(int arm=0;arm<4;arm++)')
Path(sys.argv[2]).write_text(s)
