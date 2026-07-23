#!/usr/bin/env python3
"""Simplify monotone (positive-unate) Gene-Protein-Reaction rules, in pure Python.

Pipeline:  parse -> minimal SOP (DNF + absorption) -> algebraic factoring.
Cubes are int bitmasks (bit i == variable i): subset = (a & b) == a, union = a | b.
Output is inverter-free by construction and boolean-EQUIVALENT to the input, so replacing a
reaction's GPR with its factored form leaves flux/knockout semantics -- and strain designs --
unchanged, while shrinking the GPR gadget built by extend_model_gpr.

`factor_auto(node, budget)` guards the only source of DNF blow-up (an AND of large ORs) by
AND-splitting over-budget conjuncts (exact, near-optimal since complexes sit on ~disjoint genes).
`simplify_model_gprs(model)` is the entry point used by extend_model_gpr.
"""
import re
import logging

# popcount: C-level int.bit_count() on Python 3.10+, else the bin().count fallback
_popcount = getattr(int, 'bit_count', None) or (lambda c: bin(c).count('1'))


def tokenize(s):
    for m in re.finditer(r'\(|\)|\*|\+|[^\s()*+]+', s):
        yield m.group()


def parse(s):
    """Parse a GPR string into an AST.

    Accepts both ``and``/``or`` and ``*``/``+`` operators, and is robust to any gene id,
    including digit-leading or dotted names.
    """
    toks = list(tokenize(s)); pos = 0
    def peek(): return toks[pos] if pos < len(toks) else None
    def eat():
        nonlocal pos; t = toks[pos]; pos += 1; return t
    def p_or():
        n = [p_and()]
        while peek() in ('or', '+'): eat(); n.append(p_and())
        return ('OR', n) if len(n) > 1 else n[0]
    def p_and():
        n = [p_atom()]
        while peek() in ('and', '*'): eat(); n.append(p_atom())
        return ('AND', n) if len(n) > 1 else n[0]
    def p_atom():
        if peek() == '(': eat(); e = p_or(); eat(); return e
        return ('VAR', eat())
    return p_or()


# ---- variable <-> bit mapping (reset per rule via simplify_gpr_string) ----
VMAP = {}; VINV = []
def bit(v):
    i = VMAP.get(v)
    if i is None:
        i = len(VINV); VMAP[v] = i; VINV.append(v)
    return 1 << i
def _lits_of(mask):
    out = []
    while mask:
        l = mask & -mask; out.append(('VAR', VINV[l.bit_length() - 1])); mask ^= l
    return out


# ---- cover algebra (cubes = ints) ----
def absorb(cubes):
    uniq = set(cubes)
    buckets = {}
    for c in uniq:
        buckets.setdefault(_popcount(c), []).append(c)
    keep = []
    for pc in sorted(buckets):
        smaller = keep[:]
        for c in buckets[pc]:
            if not any((k & c) == k for k in smaller):
                keep.append(c)
    return keep


def to_dnf(node):
    t = node[0]
    if t == 'VAR':   return [bit(node[1])]
    if t == 'CONST': return [] if not node[1] else [0]
    if t == 'OR':
        cov = []
        for ch in node[1]: cov += to_dnf(ch)
        return absorb(cov)
    if t == 'AND':
        cov = [0]
        for ch in node[1]:
            sub = to_dnf(ch)
            cov = absorb([a | b for a in cov for b in sub])
        return cov
    raise ValueError(t)


def common(cubes):
    it = iter(cubes); c = next(it)
    for x in it: c &= x
    return c


def lit_counts(F):
    cnt = {}
    for c in F:
        m = c
        while m:
            l = m & -m; cnt[l] = cnt.get(l, 0) + 1; m ^= l
    return cnt


def one_kernel(F, l):
    Q = [c & ~l for c in F if c & l]
    cc = common(Q)
    if cc: Q = [c & ~cc for c in Q]
    Q = absorb(Q)
    cnt = lit_counts(Q)
    reps = [x for x, n in cnt.items() if n >= 2]
    if not reps: return Q
    return one_kernel(Q, max(reps, key=lambda x: cnt[x]))


def candidate_divisors(F):
    F = absorb(F)
    if len(F) < 2: return []
    cnt = lit_counts(F)
    reps = sorted((x for x, n in cnt.items() if n >= 2), key=lambda x: -cnt[x])
    seen = set(); out = []
    for l in reps:
        K = tuple(sorted(one_kernel(F, l)))
        if len(K) >= 2 and K not in seen:
            seen.add(K); out.append(list(K))
    return out


def divide(F, D):
    """Exact algebraic division: (Q, R) with D*Q disjoint-union R == F (correctness guaranteed
    regardless of divisor quality -- a quotient cube is accepted only if D*Q stays inside F)."""
    Fs = set(F); quo = None
    for d in D:
        vd = {c & ~d for c in F if (c & d) == d}
        quo = vd if quo is None else (quo & vd)
        if not quo: return [], list(F)
    Q = list(quo)
    DQ = {dc | qc for dc in D for qc in Q}
    if not DQ <= Fs: return [], list(F)
    return Q, list(Fs - DQ)


def factor(F):
    F = absorb(F)
    if not F:     return ('CONST', False)
    if F == [0]:  return ('CONST', True)
    if len(F) == 1:
        lits = _lits_of(F[0])
        return lits[0] if len(lits) == 1 else ('AND', lits)
    cc = common(F)
    if cc:
        rem = [c & ~cc for c in F]
        return ('AND', _lits_of(cc) + [factor(rem)])
    best = None
    for D in candidate_divisors(F):
        Q, R = divide(F, D)
        if not Q or len(D) >= len(F) or len(Q) >= len(F):
            continue
        clean = 1 if not R else 0
        pulled = sum(_popcount(c) for c in D)
        cand = (clean, pulled, D, Q, R)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        return ('OR', [factor([c]) for c in F])
    _, _, D, Q, R = best
    dq = ('AND', [factor(D), factor(Q)])
    return dq if not R else ('OR', [dq, factor(R)])


def est_cubes(node):
    """Upper bound on DNF cube count (product across ANDs, sum across ORs); cheap, no expansion."""
    t = node[0]
    if t == 'VAR':   return 1
    if t == 'CONST': return 1
    if t == 'OR':    return sum(est_cubes(c) for c in node[1])
    if t == 'AND':
        p = 1
        for c in node[1]:
            p *= est_cubes(c)
            if p > 1 << 62: return p
        return p


_WARN = []
def factor_auto(node, budget=50000):
    """Global factoring within budget; AND-split above it. Never splits an OR unless one single
    OR-block alone exceeds budget (logged as a last resort -- raise the budget to avoid)."""
    if node[0] == 'VAR':
        return node
    if est_cubes(node) <= budget:
        return factor(to_dnf(node))
    if node[0] == 'AND':
        return ('AND', [factor_auto(c, budget) for c in node[1]])
    _WARN.append("OR-block of ~%d cubes exceeds budget %d; split anyway." % (est_cubes(node), budget))
    return ('OR', [factor_auto(c, budget) for c in node[1]])


def leaves(n):
    if n[0] == 'VAR':   return 1
    if n[0] == 'CONST': return 0
    return sum(leaves(c) for c in n[1])


def selfcheck(tree, node, budget):
    """Equivalence check without expanding the (possibly exploding) whole function: every
    within-budget subtree is compared by minimal cover; AND/OR composition is exact."""
    if node[0] == 'VAR':
        return True
    if est_cubes(tree) <= budget:
        return set(to_dnf(tree)) == set(to_dnf(node))
    if tree[0] != node[0] or len(tree[1]) != len(node[1]):
        return False
    return all(selfcheck(tc, nc, budget) for tc, nc in zip(tree[1], node[1]))


# ---- entry points ----
def _to_gpr_string(n):
    if n[0] == 'VAR':
        return n[1]
    if n[0] == 'CONST':
        return ''  # tautology -> no gene requirement
    if n[0] == 'AND':
        return ' and '.join(('(%s)' % _to_gpr_string(c)) if c[0] == 'OR' else _to_gpr_string(c) for c in n[1])
    return ' or '.join(('(%s)' % _to_gpr_string(c)) if c[0] == 'AND' else _to_gpr_string(c) for c in n[1])


def simplify_gpr_string(rule, budget=50000):
    """Return a leaf-minimized, boolean-equivalent monotone GPR string ('' passes through)."""
    if not rule or not rule.strip():
        return rule
    VMAP.clear(); VINV.clear(); _WARN.clear()
    return _to_gpr_string(factor_auto(parse(rule), budget))


def simplify_model_gprs(model, budget=50000):
    """In place: replace each reaction's gene_reaction_rule with a leaf-minimized equivalent.

    Monotone AND/OR boolean-equivalence => flux/knockout semantics (and strain designs) unchanged;
    only the GPR gadget built by extend_model_gpr shrinks. Any per-rule failure keeps the original.
    """
    n = nchg = 0
    for r in model.reactions:
        s = r.gene_reaction_rule
        if not s:
            continue
        n += 1
        try:
            new = simplify_gpr_string(s, budget)
            if new and new != s:
                r.gene_reaction_rule = new; nchg += 1
        except Exception as e:
            logging.warning('gpr_simplify: kept original GPR for %s (%s)' % (r.id, type(e).__name__))
    logging.info('  GPR rule simplification: %d rules, %d rewritten.' % (n, nchg))
