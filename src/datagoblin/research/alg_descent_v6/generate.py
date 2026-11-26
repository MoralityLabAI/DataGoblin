"""Generate v6 translation unit from v5 common definitions and v6 code."""
from pathlib import Path
import sys
here = Path(__file__).resolve().parent
base = next(p for p in (here.parent/'alg_descent_v5/native.cpp',here.parent/'v5/native.cpp') if p.exists())
s = base.read_text().split('struct Arm{U32 cap;bool utility;};')[0]
old = 'struct Node{Key poly;std::vector<Action> actions;};'
assert s.count(old)==1
s=s.replace(old,'struct LegacyNode{Key poly;std::vector<Action> actions;};\nstruct Node{Key poly;U16 rep=0;std::vector<Action> actions;};')
Path(sys.argv[1]).write_text(s+(here/'benchmark.inc').read_text())
