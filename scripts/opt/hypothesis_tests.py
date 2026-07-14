#!/usr/bin/env python3
"""Statistical hypothesis tests for LMC barrier comparisons."""
import json, math

def welch_t(x, y):
    n1, n2 = len(x), len(y)
    m1, m2 = sum(x)/n1, sum(y)/n2
    v1 = sum((xi-m1)**2 for xi in x)/(n1-1) if n1>1 else 0
    v2 = sum((yi-m2)**2 for yi in y)/(n2-1) if n2>1 else 0
    se = math.sqrt(v1/n1 + v2/n2)
    if se < 1e-10: return 0, 1.0, 1.0
    t = (m1 - m2) / se
    df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1)) if n1>1 and n2>1 else 1
    from math import erf, sqrt
    def ncdf(z): return 0.5*(1+erf(z/sqrt(2)))
    p = 2 * (1 - ncdf(abs(t)))
    return t, p, df

def cohens_d(x, y):
    n1, n2 = len(x), len(y)
    m1, m2 = sum(x)/n1, sum(y)/n2
    v1 = sum((xi-m1)**2 for xi in x)/(n1-1) if n1>1 else 0
    v2 = sum((yi-m2)**2 for yi in y)/(n2-1) if n2>1 else 0
    sp = math.sqrt(((n1-1)*v1 + (n2-1)*v2)/(n1+n2-2))
    if sp < 1e-10: return 0
    return (m1 - m2) / sp

# Data
cross_std_code = [0.0467, 0.0681, 0.0441]
cross_std_med  = [0.0663, 0.0347, 0.0522]
cross_high_code = [0.1349, 0.1437, 0.0748]
cross_high_med  = [0.3395, 0.0938, 0.2516]
within_code    = [0.0483, 0.0482, 0.0487]
within_med     = [0.1093, 0.1571, 0.1734]
within_math    = [0.1232, 0.0449, 0.0940]
within_general = [0.0689, 0.0624, 0.0802]
gauss_code     = [0.0034, 0.0037, 0.0045, 0.0106, 0.0136]
within_med_high = [0.2069, 0.0722, 0.1516, 0.3000, 0.1826, 0.3706, 1.2131, 0.3827, 0.0799, 0.0807]

print("="*65)
print("HYPOTHESIS TESTS — LMC Barrier Analysis")
print("="*65)

tests = [
    ("H1: code within = cross std?", within_code, cross_std_code),
    ("H2: med within = cross std?",  within_med, cross_std_med),
    ("H3: code within = med within?", within_code, within_med),
    ("H4: gaussian = zero?", gauss_code, [0.0]*len(gauss_code)),
    ("H5: code high = code std?", cross_high_code, cross_std_code),
    ("H6: med high = med std?", cross_high_med, cross_std_med),
]

for name, x, y in tests:
    t, p, df = welch_t(x, y)
    mx, my = sum(x)/len(x), sum(y)/len(y)
    sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
    print(f"\n{name}")
    print(f"  means: {mx:.4f} vs {my:.4f}")
    print(f"  t({df:.1f})={t:.3f}, p={p:.4f} {sig}")

print("\n" + "="*65)
print("EFFECT SIZES (Cohen d)")
print("="*65)
effects = [
    ("Code vs Med within", within_med, within_code),
    ("Med within vs cross", within_med, cross_std_med),
    ("Gaussian vs Training", cross_std_code, gauss_code[:3]),
    ("Code high vs std", cross_high_code, cross_std_code),
]
for name, x, y in effects:
    d = cohens_d(x, y)
    mag = "large" if abs(d)>0.8 else ("medium" if abs(d)>0.5 else "small")
    print(f"  {name}: d={d:.2f} ({mag})")
