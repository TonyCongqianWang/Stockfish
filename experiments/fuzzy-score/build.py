#!/usr/bin/env python3
"""Build A/B/C and separate trace binaries in an isolated local clone."""
from pathlib import Path
import argparse
import subprocess

BASE = "5d03b6e578a5eedc822e4563bdbb89e762fc2a59"
p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
p.add_argument("--out", type=Path, required=True)
p.add_argument("--net", type=Path, required=True)
p.add_argument("--jobs", default="2")
args = p.parse_args()
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
root = out / "build"
if root.exists():
    raise SystemExit("Output build directory already exists; use a fresh --out")
subprocess.run(["git", "clone", "--shared", str(args.repo.resolve()), str(root)], check=True)
a = subprocess.check_output(["git", "-C", str(root), "show", BASE + ":src/search.cpp"], text=True)
b = a.replace("    int  failHighRecovery   = 0;\n", "").replace(
    "            if (!pvIdx)\n                failHighRecovery = std::max(0, failHighRecovery - 2);\n", ""
).replace(" - failHighRecovery", "").replace(
    "            // Gradually increase depth after reduced depth search\n"
    "            if (failedHighCnt > 0 && !pvIdx)\n"
    "                failHighRecovery = (failedHighCnt + 1) / 2 + 2;\n\n", ""
)
assert "failHighRecovery" not in b
c = (args.repo / "src/search.cpp").read_text()
# Use the same C header/layout for all variants; A's bench must still be 1442480.
(root / "src/search.h").write_text((args.repo / "src/search.h").read_text())
(root / "src/nn-bddd9d57eb87.nnue").symlink_to(args.net.resolve())

def build(label, source):
    (root / "src/search.cpp").write_text(source)
    (out / (label + ".cpp")).write_text(source)
    with (out / ("build-" + label + ".log")).open("w") as f:
        subprocess.run(["make", "-j" + args.jobs, "build", "ARCH=x86-64-avx2",
                        "EXE=" + str(out / label)], cwd=root / "src",
                       stdout=f, stderr=subprocess.STDOUT, check=True)

def instrument(s):
    s=s.replace('                bestValue           = search<Root>', '                u64 attemptNodes = nodes;\n                bestValue           = search<Root>')
    marker='                // If search has been stopped, we break immediately.'
    log='''                sync_cout << "info string ASP " << rootDepth << " " << adjustedDepth
                          << " " << alpha << " " << beta << " " << bestValue
                          << " " << u64(nodes) - attemptNodes << " " << failedHighCnt
                          << " " << avg << " " << ss->moveCount << " " << rootMoves.size()
                          << " " << threads.stop << " " << UCIEngine::move(rootMoves[pvIdx].pv[0], rootPos.is_chess960())
                          << sync_endl;

'''
    assert marker in s
    s=s.replace(marker,log+marker)
    marker='        const bool forgottenMate ='
    s=s.replace(marker,'''        sync_cout << "info string ITER " << rootDepth << " " << rootMoves[0].score
                  << " " << rootMoves[0].uciScore << " " << rootMoves[0].averageScore
                  << " " << rootMoves[0].meanSquaredScore << " " << rootMoves[0].inexactLower
                  << " " << rootMoves[0].inexactUpper << " " << threads.stop
                  << " " << u64(nodes) << " " << UCIEngine::move(rootMoves[0].pv[0], rootPos.is_chess960()) << sync_endl;

'''+marker)
    return s
for label, source in [("A", a), ("B", b), ("C", c)]:
    print("Building", label, flush=True)
    build(label + "-release", source)
    build(label + "-trace", instrument(source))
(root / "src/search.cpp").write_text(c)
