"""Generate V18 from pinned V17 native.cpp (ff9711045ac44bc5648edbb60e1979a06fdb8a2d)."""
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
def sub(a,b):
 global s
 assert s.count(a)==1,a
 s=s.replace(a,b)
sub('// ALG-DESCENT-17: factorial storage optimization; C++17.','// ALG-DESCENT-18: budget-aware certified refinement; C++17.')
sub('bool coarse=false,done=false,open=false;','bool coarse=false,done=false,open=false;U8 rounds=0;')
sub('constexpr std::array<int,6> budgets{0,4,16,32,64,128};','constexpr std::array<int,6> budgets{0,4,16,32,64,128};\nconstexpr std::array<int,3> workcaps{256,1024,4096};')
sub('std::array<Snap,6> snaps{};','std::array<Snap,6> snaps{};std::array<Snap,3> work{};')
sub('int arm;bool direct,defer,packed;','int arm,policy;bool direct,defer,packed;')
sub('g(gg),arm(a<2?0:2),direct(a>=2&&((a-2)&1)),defer(a>=2&&((a-2)&2)),packed(a==1||(a>=2&&((a-2)&4)))','g(gg),arm(a==0?0:2),policy(a),direct(a!=0),defer(a!=0),packed(true)')
sub('c[CUTS]++;c[REFINE]++;','n.rounds++;c[CUTS]++;c[REFINE]++;')
sub('void pop(){std::pop_heap(heap.begin(),heap.end());heap.pop_back();}', '''void pop(){std::pop_heap(heap.begin(),heap.end());heap.pop_back();}
bool want(const Node& n)const{if(!arm||n.done)return false;int slack=U-n.best-n.b;switch(policy){case 1:return true;case 2:return n.rounds<1;case 3:return n.rounds<2;case 4:return slack<=2;case 5:return slack<=4&&n.rounds<2;default:return false;}}
void work_snapshot(){if(!log)return;auto w=c[PROBES]+c[DSCAN]+c[BSCAN]+c[MSCAN];bool needed=false;for(int cap:workcaps)needed|=w<=U64(cap);if(!needed)return;int L=U;for(const auto& n:nodes)if(n.open)L=std::min(L,int(n.best)+n.b);for(size_t j=0;j<workcaps.size();j++)if(w<=U64(workcaps[j]))res.work[j]={workcaps[j],L,U,opens,incumbent,c};}
''')
sub('admit(root);for(int k=0;k<6;k++)','admit(root);work_snapshot();for(int k=0;k<6;k++)')
sub('bool need=arm&&!n.done;','bool need=want(n);')
sub('else push(id);continue;}event','else push(id);work_snapshot();continue;}event')
sub('else admit(j);}}\nint L=U;','else admit(j);}work_snapshot();}\nint L=U;')
sub('std::array<Result,10> first{};std::array<std::vector<U64>,10> times;std::array<int,10> order{0,1,2,3,4,5,6,7,8,9};','std::array<Result,6> first{};std::array<std::vector<U64>,6> times;std::array<int,6> order{0,1,2,3,4,5};')
sub('for(int arm=0;arm<10;arm++)','for(int arm=0;arm<6;arm++)')
sub('auto t=times[arm];','for(auto& s:r.work){std::cout<<"W "<<s.cap<<\' \'<<s.L<<\' \'<<s.U<<\' \'<<s.open;for(U64 v:s.c)std::cout<<\' \'<<v;pathout(s.proof);std::cout<<\'\\n\';}auto t=times[arm];')
Path(sys.argv[2]).write_text(s)
