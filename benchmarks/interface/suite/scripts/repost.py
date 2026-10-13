#!/usr/bin/env python3
"""repost.py <casesDir> -- re-run the final post.py on every finished case, so that every
result.json of the archive was produced by the same post-processor (the first 183 cases were
post-processed by the design job's unmodified post.py, which lacks the added measures).
Solver output is not touched."""
import os, sys, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import post  # noqa: E402

root = sys.argv[1]
n = bad = 0
t0 = time.time()
for cid in sorted(os.listdir(root)):
    case = os.path.join(root, cid)
    if not os.path.exists(os.path.join(case, 'result.json')):
        continue
    try:
        res = post.analyse(case)
        open(os.path.join(case, 'log.post'), 'w').write(post.summary(res) + '\n')
        n += 1
    except Exception:
        bad += 1
        print('FAILED', cid, traceback.format_exc().splitlines()[-1], flush=True)
print(f'{n} cases re-posted, {bad} failed, {time.time() - t0:.0f} s')
