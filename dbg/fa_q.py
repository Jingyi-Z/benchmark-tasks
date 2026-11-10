"""fa_q.py FILE T0 T1 [FIELDS...]: rows of a debug log between plan times T0 and T1, chosen key=value fields."""
import re, sys
f, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
keys = sys.argv[4:] or ["rod", "boat", "lid"]
for l in open(f):
    m = re.search(r" t=([0-9.]+)", l)
    if not m or not t0 <= float(m.group(1)) <= t1:
        continue
    out = [m.group(1)]
    for k in keys:
        if k == "R":  # the right hand's two fingertips
            tp = re.search(r"tips=(\[.*\])", l).group(1)
            out.append("R" + ",".join(re.findall(r"\[[^\[\]]*\]", tp)[2:]))
        elif k == "L":
            tp = re.search(r"tips=(\[.*\])", l).group(1)
            out.append("L" + ",".join(re.findall(r"\[[^\[\]]*\]", tp)[:2]))
        else:
            mm = re.search(rf" {k}=(\[[^\]]*\]|\S+)", l)
            out.append(f"{k}={mm.group(1) if mm else '?'}")
    print(" ".join(out))
