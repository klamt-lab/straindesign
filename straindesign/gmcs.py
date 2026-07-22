"""gmcs.py — gene-MCS via a lean, flux-free G-space MILP built on SD's own machinery (PROTOTYPE).

Standalone `compute_gmcs`. Reconstructs gMCSpy's intervention/G idea INSIDE SD's exact reaction-MCS
certificate: the flux-infeasibility certificate + indicator coupling stay over the real (compressed)
reactions (SD leanness, no gadget), while the decision/objective layer is the gene interventions
(option (a): reaction z_r kept as gene-gated auxiliaries; enumeration over gene z_g via idx_z).

Pipeline: 0 (opt) reversibility; 1 compress; 2 simplify GPR; 3 G (Berge transversals);
4 SDProblem over compressed model (ko_cost = GPR reactions); 5 surgery: add gene z_g (+ aux y_I),
G-coupling z_r <= sum_I applied(I), rewrite cost rows onto z_g, idx_z -> gene columns.

PROTOTYPE — validate vs the gadget (e_coli_core / iML1515-cone). verify_sd is bypassed; correctness
is judged by matching the gadget's gene-MCS SET.
"""
import logging
import numpy as np
from scipy import sparse
from . import gpr_bitmask as _gb
from .strainDesignProblem import SDProblem
from .strainDesignMILP import SDMILP
from .solver_interface import MILP_LP
from .names import SOLVER, KOCOST, MAX_COST, SEED, MILP_THREADS, OPTIMAL


# ---------------- step 3: interventions (minimal blocking gene sets = DNF transversals) ------------
def _minimal_hitting_sets(cubes, max_size):
    cs = sorted({frozenset(c) for c in cubes}, key=len)
    minc = []
    for c in cs:
        if not any(m <= c for m in minc):
            minc.append(c)
    hitters = [frozenset()]
    for cube in minc:
        nxt = []
        for h in hitters:
            if h & cube:
                nxt.append(h)
            else:
                for g in cube:
                    cand = h | {g}
                    if len(cand) <= max_size:
                        nxt.append(cand)
        nxt = sorted(set(nxt), key=len)
        keep = []
        for h in nxt:
            if not any(k <= h for k in keep):
                keep.append(h)
        hitters = keep
    return hitters


def _minimize(sets, max_ko):
    ss = sorted({s for s in sets if len(s) <= max_ko}, key=len)
    keep = []
    for s in ss:
        if not any(k <= s for k in keep):
            keep.append(s)
    return keep


def _tree_interventions(node, max_ko):
    """Minimal blocking gene-sets (interventions) computed RECURSIVELY on the monotone tree -- no DNF
    expansion. gene->{g}; AND->union of children (block any); OR->cartesian (block every), pruned <=max_ko."""
    if node[0] == 'VAR':
        return [frozenset([node[1]])]
    if node[0] == 'AND':
        out = []
        for ch in node[1]:
            out += _tree_interventions(ch, max_ko)
        return _minimize(out, max_ko)
    # OR: must block every child -> cartesian union
    result = [frozenset()]
    for ch in node[1]:
        ci = _tree_interventions(ch, max_ko)
        nxt = [a | b for a in result for b in ci if len(a | b) <= max_ko]
        result = _minimize(nxt, max_ko)
        if not result:
            return []
    return result


def build_gspace(model, max_ko=3):
    interv = {}
    for r in model.reactions:
        if not r.gene_reaction_rule:
            continue
        iv = _tree_interventions(_gb.parse(r.gene_reaction_rule), max_ko)
        if iv:
            interv[r.id] = iv
    logging.info('  [gmcs] G-space: %d GPR reactions, %d interventions'
                 % (len(interv), sum(len(v) for v in interv.values())))
    return interv


# ---------------- steps 4-5: lean MILP over reactions + gene layer on top --------------------------
class GMCSMILP(SDMILP):
    def __init__(self, model, sd_modules, interventions, max_ko, **kwargs):
        import time as _t
        _t0 = _t.time(); SDProblem.__init__(self, model, sd_modules, **kwargs)   # step 4
        logging.info('  [gmcs] .. SDProblem.__init__: %.1fs' % (_t.time() - _t0))
        _t0 = _t.time(); self._gmcs_surgery(model, interventions, max_ko)        # step 5
        logging.info('  [gmcs] .. surgery: %.1fs' % (_t.time() - _t0))
        _t0 = _t.time()
        MILP_LP.__init__(self, c=self.c, A_ineq=self.A_ineq, b_ineq=self.b_ineq, A_eq=self.A_eq,
                         b_eq=self.b_eq, lb=self.lb, ub=self.ub, vtype=self.vtype,
                         indic_constr=self.indic_constr, M=self.M, solver=self.solver,
                         seed=self.seed, milp_threads=self.milp_threads)
        logging.info('  [gmcs] .. MILP_LP(solver build): %.1fs' % (_t.time() - _t0))

    def verify_sd(self, sols):                                       # prototype: trust MILP
        return [True] * sols.shape[0]

    def _gmcs_surgery(self, model, interventions, max_ko):
        ncol0 = self.A_ineq.shape[1]
        rxn_col = {r.id: i for i, r in enumerate(model.reactions)}   # reaction z col = reaction index
        genes = sorted({g for iv in interventions.values() for I in iv for g in I})
        gcol = {g: ncol0 + i for i, g in enumerate(genes)}
        ng = len(genes)
        ycol, ny = {}, 0
        for rid, ivs in interventions.items():
            for k, I in enumerate(ivs):
                if len(I) >= 2:
                    ycol[(rid, k)] = ncol0 + ng + ny; ny += 1
        ntot = ncol0 + ng + ny

        def applied(rid, k, I):
            return gcol[next(iter(I))] if len(I) == 1 else ycol[(rid, k)]

        # a reaction with a z but NO <=max_ko intervention cannot be gene-blocked -> forbid its cut
        # (else its z_r is ungated and the certificate cuts biomass for free -> empty gene-MCS)
        for i, r in enumerate(model.reactions):
            if self.ub[i] > 0 and r.id not in interventions:
                self.ub[i] = 0.0
        # build new inequality rows as SPARSE COO (dense np.zeros(ntot) per row was the 131s bottleneck)
        ri, ci, dat, bnew, nr = [], [], [], [], 0
        def _addrow(entries, b):
            nonlocal nr
            for c, v in entries:
                ri.append(nr); ci.append(c); dat.append(v)
            bnew.append(float(b)); nr += 1
        # AND linearisation for multi-gene interventions: y - z_g <= 0 ; sum z_g - y <= |I|-1
        for rid, ivs in interventions.items():
            for k, I in enumerate(ivs):
                if len(I) >= 2:
                    y = ycol[(rid, k)]
                    for g in I:
                        _addrow([(y, 1.0), (gcol[g], -1.0)], 0.0)
                    _addrow([(gcol[g], 1.0) for g in I] + [(y, -1.0)], len(I) - 1.0)
        # OPTION (b): single intervention -> trigger on gene/aux directly (reaction z dropped);
        #             multiple -> keep reaction z as blocked_r with z_r <= sum_I applied(I)
        trigger = {}
        for rid, ivs in interventions.items():
            if rid not in rxn_col:
                continue
            zr = rxn_col[rid]
            if len(ivs) == 1:
                trigger[zr] = applied(rid, 0, ivs[0]); self.ub[zr] = 0.0
            else:
                trigger[zr] = zr
                _addrow([(zr, 1.0)] + [(applied(rid, k, I), -1.0) for k, I in enumerate(ivs)], 0.0)
        self.indic_constr.binv = [trigger.get(int(b), int(b)) for b in self.indic_constr.binv]
        new_A = sparse.csr_matrix((dat, (ri, ci)), shape=(nr, ntot)) if nr else sparse.csr_matrix((0, ntot))
        # widen old A_ineq to ntot cols; rewrite cost rows 0/1 onto genes (clear via lil row, not O(ntot))
        A = sparse.hstack([self.A_ineq.tocsr(), sparse.csr_matrix((self.A_ineq.shape[0], ng + ny))]).tolil()
        for _rn in (self.idx_row_maxcost, self.idx_row_mincost):
            A.rows[_rn] = []; A.data[_rn] = []
        for g in genes:
            A[self.idx_row_maxcost, gcol[g]] = -1.0
            A[self.idx_row_mincost, gcol[g]] = 1.0
        self.b_ineq[self.idx_row_maxcost] = 0.0
        self.b_ineq[self.idx_row_mincost] = float(max_ko)
        self.A_ineq = sparse.vstack([A.tocsr(), new_A]).tocsr()
        self.b_ineq = list(self.b_ineq) + bnew
        self.A_eq = sparse.hstack([self.A_eq, sparse.csc_matrix((self.A_eq.shape[0], ng + ny))]).tocsc()
        self.lb = list(self.lb) + [0.0] * (ng + ny)
        self.ub = list(self.ub) + [1.0] * (ng + ny)
        self.c = [0.0] * ntot
        for g in genes:
            self.c[gcol[g]] = 1.0                                    # minimise distinct genes
        self.vtype = self.vtype + 'B' * (ng + ny)
        self.indic_constr.A = sparse.hstack(
            [self.indic_constr.A, sparse.csc_matrix((self.indic_constr.A.shape[0], ng + ny))]).tocsr()
        self.idx_z = [gcol[g] for g in genes]
        self.num_z = ng
        self.cost = [0.0] * ntot
        for g in genes:
            self.cost[gcol[g]] = 1.0
        self._gene_by_col = {gcol[g]: g for g in genes}
        self._gene_cols = [gcol[g] for g in genes]
        logging.info('  [gmcs] MILP: %d cols (+%d gene z, +%d aux y), %d GPR reactions'
                     % (ntot, ng, ny, len(rxn_col)))

    def enumerate_gmcs(self, max_solutions=np.inf, batch=500):
        """Minimal gene-MCS via batched pool populate (reusing populateZ over the gene idx_z) +
        integer cuts. populateZ returns z as (nsol x num_z) with column k == genes[k] (idx_z order)."""
        import time as _t
        genes = [self._gene_by_col[c] for c in self._gene_cols]     # idx_z order
        seen, sols = set(), []
        while len(sols) < max_solutions:
            _t0 = _t.time()
            z, status = self.populateZ(int(min(batch, max_solutions)) if np.isfinite(max_solutions) else batch)
            logging.info('  [gmcs] populateZ: %d in pool, status=%s, %.1fs (total %d)'
                         % (z.shape[0], status, _t.time() - _t0, len(sols)))
            if status not in (OPTIMAL,) or z.shape[0] == 0:
                break
            new = 0
            for j in range(z.shape[0]):
                idx = z[j].indices.tolist()
                on = frozenset(genes[k] for k in idx)
                if not on or on in seen:
                    continue
                seen.add(on); sols.append(on); new += 1
                cut = np.zeros(self.A_ineq.shape[1])
                for k in idx:
                    cut[self._gene_cols[k]] = 1.0
                self.add_ineq_constraints(sparse.csr_matrix(cut), [float(len(on) - 1)])
            if new == 0:
                break
        return sols


# ---------------- orchestrator ---------------------------------------------------------------------
def compute_gmcs(model, sd_modules, max_ko=3, reversibility=False, compress=True,
                 solver=None, seed=None, milp_threads=None, max_solutions=np.inf):
    import time as _t
    from .networktools import copy_model_suppressed
    m = copy_model_suppressed(model)  # cheap copy (~0.3s): no solver rebuild. Compression is stub-safe
                                      # (compress_model_coupled runs under suppress_lp_context); SDProblem
                                      # builds its MILP from stoichiometry (empty-objective primal), so no
                                      # populated optlang solver is ever needed on the copy.
    if reversibility:                                               # step 0
        _t0 = _t.time()
        from .speedy_fva import fast_reversibility
        rev = fast_reversibility(m, solver=solver)
        for r in m.reactions:
            cf, cr = rev[r.id]
            if not cf and float(r._upper_bound) > 0: r._upper_bound = 0.0
            if not cr and float(r._lower_bound) < 0: r._lower_bound = 0.0
        logging.info('  [gmcs] step0 reversibility: %.1fs' % (_t.time() - _t0))
    if compress:                                                   # step 1  (True/'full' | 'coupled')
        _t0 = _t.time()
        from .compression import compress_model, compress_model_coupled
        from .networktools import compress_modules
        if compress == 'coupled':
            # single coupled-lumping pass (no parallel/conservation cycles-to-convergence): cheaper,
            # captures most of the reaction reduction from flux coupling.
            rmap = compress_model_coupled(m, propagate_gpr=True)
            cmp_map = [{"reac_map_exp": rmap, "parallel": False}]
        else:
            cmp_map = compress_model(m, set(), propagate_gpr=True)  # full routine
        sd_modules = compress_modules(sd_modules, cmp_map)          # remap module ids/coeffs to lumped model
        logging.info('  [gmcs] step1 compress (%s): %.1fs' % (compress, _t.time() - _t0))
    _t0 = _t.time(); _gb.simplify_model_gprs(m)                     # step 2
    logging.info('  [gmcs] step2 simplify GPR: %.1fs' % (_t.time() - _t0))
    _t0 = _t.time(); interv = build_gspace(m, max_ko=max_ko)        # step 3
    logging.info('  [gmcs] step3 build_gspace: %.1fs' % (_t.time() - _t0))
    gpr_reacs = {r.id: 1.0 for r in m.reactions if r.gene_reaction_rule}
    _t0 = _t.time()
    prob = GMCSMILP(m, sd_modules, interv, max_ko,                  # steps 4-5
                    **{KOCOST: gpr_reacs, MAX_COST: float(max_ko), SOLVER: solver,
                       SEED: seed, MILP_THREADS: milp_threads})
    logging.info('  [gmcs] step4-5 MILP build: %.1fs' % (_t.time() - _t0))
    _t0 = _t.time(); sols = prob.enumerate_gmcs(max_solutions=max_solutions)
    logging.info('  [gmcs] enumerate/solve: %.1fs' % (_t.time() - _t0))
    logging.info('  [gmcs] %d gene-MCS' % len(sols))
    return sols
