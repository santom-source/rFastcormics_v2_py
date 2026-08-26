"""Python port of rFASTCORMICS_v2 (sysbiolux/rFASTCORMICS,
"rFASTCORMICS for RNA-seq data/rFASTCORMICS_v2").

Covers:
    scripts/rFASTCORMICS/rFastcormics.m
    scripts/rFASTCORMICS/discretizeFPKM.m         (+ createFitR.m / createFitL.m)
    scripts/rFASTCORMICS/mapExpressionToModel.m   (+ GPRrulesMapper.m)
    scripts/rFASTCORMICS/constrainModelOnMedium.m (+ findOrganicExRxns.m)
    scripts/rFASTCORMICS/fixIrr.m
    scripts/FASTCORE/fastcc.m  (method='original'), fastcore.m, findSparseMode.m,
                               LP3.m, LP7.m, LP10.m
plus the COBRA helpers it calls (findExcRxns, findTransRxns, findRxnsFromMets,
removeRxns, removeUnusedGenes, changeObjective, optimizeCbModel,
buildOptProblemFromModel).

Things to remember:

- The core runs on plain numpy / scipy.sparse arrays mirroring the COBRA fields
  (S, lb, ub, c, b, csense, rxns, mets, genes, rules, ...) rather than on cobrapy
  objects. cobrapy differs from the COBRA Toolbox on bounds handling,
  reversibility, orphan-metabolite pruning and GPR representation, and those
  differences change the result silently. from_cobrapy / to_cobrapy at the edges.
- Indices are 0-based here. MATLAB's find / intersect / setdiff / union all return
  sorted vectors, and so do the numpy equivalents used here, so FASTCORE walks the
  search in the same order.
- `basis` (warm start) is accepted but never reused, same as MATLAB: LP3.m line 38
  and LP7.m line 49 both reset `basis = []` before the solve, so it is dead code
  there too.
- Default solver is gurobipy with COBRA's defaults (FeasibilityTol = OptimalityTol
  = 1e-6, Method = -1), because the MATLAB reference used Gurobi. A HiGHS backend
  via scipy.optimize.linprog is there for machines without a licence; FASTCORE
  picks one vertex out of many optimal ones, so HiGHS drifts by ~1% of reactions.
"""

from __future__ import annotations

import copy
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import scipy.sparse as sp

__all__ = [
    "CobraModel",
    "LPProblem",
    "set_default_solver",
    "solve_cobra_lp",
    "build_opt_problem_from_model",
    "fix_irr",
    "fastcc",
    "fastcore",
    "find_sparse_mode",
    "LP3",
    "LP7",
    "LP10",
    "find_exc_rxns",
    "find_trans_rxns",
    "find_rxns_from_mets",
    "find_organic_ex_rxns",
    "constrain_model_on_medium",
    "remove_rxns",
    "remove_unused_genes",
    "change_objective",
    "optimize_cb_model",
    "gpr_rules_mapper",
    "map_expression_to_model",
    "ksdensity",
    "fit_gauss1",
    "discretize_fpkm",
    "r_fastcormics",
    "from_cobrapy",
    "to_cobrapy",
]


# ======================================================================================
# 0.  Model container
# ======================================================================================

#: model fields that are indexed by reaction (n x 1)
RXN_FIELDS = (
    "rxns", "lb", "ub", "c", "rev", "rules", "grRules", "subSystems", "rxnNames",
    "rxnNotes", "rxnECNumbers", "rxnReferences", "rxnKEGGID", "rxnConfidenceScores",
    "rxnCOG", "rxnKeggOrthology", "rxnReconMap",
)

#: model fields that are indexed by metabolite (m x 1)
MET_FIELDS = (
    "mets", "b", "csense", "metNames", "metFormulas", "metCharges", "metSmiles",
    "metHMDBID", "metInChIString", "metKEGGID", "metPubChemID", "metCHEBIID",
    "metPdMap", "metReconMap",
)


@dataclass
class CobraModel:
    """A minimal, COBRA-Toolbox-faithful model container."""

    S: sp.spmatrix
    rxns: np.ndarray
    mets: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    c: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None
    csense: Optional[np.ndarray] = None
    osense: int = -1
    genes: Optional[np.ndarray] = None
    rules: Optional[np.ndarray] = None
    grRules: Optional[np.ndarray] = None
    rxnGeneMat: Optional[sp.spmatrix] = None
    subSystems: Optional[np.ndarray] = None
    metFormulas: Optional[np.ndarray] = None
    metNames: Optional[np.ndarray] = None
    rxnNames: Optional[np.ndarray] = None
    rev: Optional[np.ndarray] = None
    extra: dict = field(default_factory=dict)

    # -- construction -------------------------------------------------------------
    def __post_init__(self):
        self.S = sp.csc_matrix(self.S, dtype=float)
        self.rxns = np.asarray(self.rxns, dtype=object).ravel()
        self.mets = np.asarray(self.mets, dtype=object).ravel()
        self.lb = np.asarray(self.lb, dtype=float).ravel()
        self.ub = np.asarray(self.ub, dtype=float).ravel()
        n, m = self.S.shape[1], self.S.shape[0]
        if self.c is None:
            self.c = np.zeros(n)
        self.c = np.asarray(self.c, dtype=float).ravel()
        if self.b is None:
            self.b = np.zeros(m)
        self.b = np.asarray(self.b, dtype=float).ravel()
        if self.csense is None:
            self.csense = np.array(["E"] * m, dtype=object)
        self.csense = np.asarray(self.csense, dtype=object).ravel()
        for f in ("genes", "rules", "grRules", "subSystems", "metFormulas",
                  "metNames", "rxnNames", "rev"):
            v = getattr(self, f)
            if v is not None:
                setattr(self, f, np.asarray(v, dtype=object if f != "rev" else float).ravel())

    # -- convenience --------------------------------------------------------------
    @property
    def n_rxns(self) -> int:
        return self.S.shape[1]

    @property
    def n_mets(self) -> int:
        return self.S.shape[0]

    def copy(self) -> "CobraModel":
        return copy.deepcopy(self)

    def rxn_index(self, names) -> np.ndarray:
        """0-based indices of `names` in model.rxns (order of `names`, missing dropped)."""
        pos = {r: i for i, r in enumerate(self.rxns)}
        return np.array([pos[x] for x in _as_list(names) if x in pos], dtype=int)

    def __repr__(self):  # pragma: no cover - cosmetic
        return (f"<CobraModel {self.n_mets} mets x {self.n_rxns} rxns"
                f"{'' if self.genes is None else f', {len(self.genes)} genes'}>")


def _as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, np.ndarray):
        return list(x.ravel())
    return list(x)


# ======================================================================================
# 1.  LP layer  (solveCobraLP equivalent)
# ======================================================================================

@dataclass
class LPProblem:
    """Mirror of the COBRA `LPproblem` struct."""
    A: sp.spmatrix
    b: np.ndarray
    c: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    csense: np.ndarray          # array of 'E' / 'L' / 'G'
    osense: int = 1             # 1 = minimise, -1 = maximise

    def copy(self) -> "LPProblem":
        return LPProblem(self.A.copy(), self.b.copy(), self.c.copy(),
                         self.lb.copy(), self.ub.copy(), self.csense.copy(),
                         self.osense)


_DEFAULT_SOLVER = "gurobi"
FEAS_TOL = 1e-6      # COBRA getCobraSolverParams('LP','feasTol')
OPT_TOL = 1e-6       # COBRA getCobraSolverParams('LP','optTol')


def set_default_solver(name: str) -> None:
    """`'gurobi'` (default, matches the MATLAB reference run) or `'highs'`."""
    global _DEFAULT_SOLVER
    name = name.lower()
    if name not in ("gurobi", "highs"):
        raise ValueError("solver must be 'gurobi' or 'highs'")
    _DEFAULT_SOLVER = name


def get_default_solver() -> str:
    return _DEFAULT_SOLVER


def solve_cobra_lp(problem: LPProblem, solver: Optional[str] = None,
                   feas_tol: float = FEAS_TOL, opt_tol: float = OPT_TOL) -> dict:
    """Solve an LP; returns ``{'full': x, 'obj': f, 'stat': s, 'origStat': ...}``.

    ``stat``: 1 = optimal, 0 = infeasible, 2 = unbounded, -1 = other
    (identical encoding to ``solveCobraLP.m``).
    """
    solver = (solver or _DEFAULT_SOLVER).lower()
    if solver == "gurobi":
        return _solve_gurobi(problem, feas_tol, opt_tol)
    return _solve_highs(problem, feas_tol, opt_tol)


def _solve_gurobi(p: LPProblem, feas_tol: float, opt_tol: float) -> dict:
    import gurobipy as gp
    from gurobipy import GRB

    env = _gurobi_env()
    m = gp.Model(env=env)
    m.Params.OutputFlag = 0
    m.Params.FeasibilityTol = feas_tol
    m.Params.OptimalityTol = opt_tol
    m.Params.Method = -1                      # COBRA default (automatic)

    n = p.A.shape[1]
    lb = np.where(np.isneginf(p.lb), -GRB.INFINITY, p.lb)
    ub = np.where(np.isposinf(p.ub), GRB.INFINITY, p.ub)
    x = m.addMVar(shape=n, lb=lb, ub=ub, obj=np.asarray(p.c, float))
    m.ModelSense = GRB.MINIMIZE if p.osense == 1 else GRB.MAXIMIZE

    A = sp.csr_matrix(p.A)
    cs = np.asarray(p.csense, dtype=object)
    b = np.asarray(p.b, float)
    for sense, op in (("E", "=="), ("L", "<="), ("G", ">=")):
        idx = np.flatnonzero(cs == sense)
        if idx.size == 0:
            continue
        sub = A[idx, :]
        if op == "==":
            m.addConstr(sub @ x == b[idx])
        elif op == "<=":
            m.addConstr(sub @ x <= b[idx])
        else:
            m.addConstr(sub @ x >= b[idx])
    m.optimize()

    status = m.Status
    out = {"origStat": status, "stat": -1, "full": None, "obj": None}
    if status == GRB.OPTIMAL:
        out["stat"] = 1
        out["full"] = np.asarray(x.X, float)
        out["obj"] = m.ObjVal
    elif status == GRB.INFEASIBLE:
        out["stat"] = 0
    elif status == GRB.UNBOUNDED:
        out["stat"] = 2
    elif status == GRB.INF_OR_UNBD:
        # solveCobraLP.m: drop the objective and re-solve to disambiguate
        m.setObjective(0.0)
        m.optimize()
        out["stat"] = 2 if m.Status == GRB.OPTIMAL else 0
    m.dispose()
    return out


_GRB_ENV = None


def _gurobi_env():
    global _GRB_ENV
    if _GRB_ENV is None:
        import gurobipy as gp
        _GRB_ENV = gp.Env(params={"OutputFlag": 0})
    return _GRB_ENV


def _solve_highs(p: LPProblem, feas_tol: float, opt_tol: float) -> dict:
    from scipy.optimize import linprog

    cs = np.asarray(p.csense, dtype=object)
    A = sp.csr_matrix(p.A)
    b = np.asarray(p.b, float)

    eq = np.flatnonzero(cs == "E")
    le = np.flatnonzero(cs == "L")
    ge = np.flatnonzero(cs == "G")

    A_ub_parts, b_ub_parts = [], []
    if le.size:
        A_ub_parts.append(A[le, :])
        b_ub_parts.append(b[le])
    if ge.size:
        A_ub_parts.append(-A[ge, :])
        b_ub_parts.append(-b[ge])
    A_ub = sp.vstack(A_ub_parts, format="csr") if A_ub_parts else None
    b_ub = np.concatenate(b_ub_parts) if b_ub_parts else None
    A_eq = A[eq, :] if eq.size else None
    b_eq = b[eq] if eq.size else None

    c = np.asarray(p.c, float) * (1 if p.osense == 1 else -1)
    bounds = list(zip(np.where(np.isneginf(p.lb), None, p.lb),
                      np.where(np.isposinf(p.ub), None, p.ub)))
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method="highs",
                  options={"primal_feasibility_tolerance": feas_tol,
                           "dual_feasibility_tolerance": opt_tol})
    out = {"origStat": res.status, "stat": -1, "full": None, "obj": None}
    if res.status == 0:
        out["stat"] = 1
        out["full"] = np.asarray(res.x, float)
        out["obj"] = res.fun * (1 if p.osense == 1 else -1)
    elif res.status == 2:
        out["stat"] = 0
    elif res.status == 3:
        out["stat"] = 2
    return out


def build_opt_problem_from_model(model: CobraModel) -> LPProblem:
    """`buildOptProblemFromModel.m` for a model without C/E coupling fields."""
    return LPProblem(A=sp.csc_matrix(model.S).copy(),
                     b=model.b.astype(float).copy(),
                     c=model.c.astype(float).copy(),
                     lb=model.lb.astype(float).copy(),
                     ub=model.ub.astype(float).copy(),
                     csense=np.asarray(model.csense, dtype=object).copy(),
                     osense=1)


# ======================================================================================
# 2.  COBRA Toolbox helpers
# ======================================================================================

def fix_irr(model: CobraModel) -> CobraModel:
    """`fixIrr.m` - turn backwards-irreversible reactions into forward ones."""
    model = model.copy()
    n = model.n_rxns
    rev = np.zeros(n)
    rev[(model.lb < 0) & (model.ub > 0)] = 1
    irr = ((model.lb >= 0) & (model.ub > 0)) | ((model.ub <= 0) & (model.lb < 0))
    rev[irr] = 0
    model.rev = rev

    fake_irr = (model.ub <= 0) & (model.lb < 0)
    if fake_irr.any():
        S = sp.csc_matrix(model.S).copy().tolil()
        idx = np.flatnonzero(fake_irr)
        S[:, idx] = -S[:, idx]
        model.S = sp.csc_matrix(S)
        ub_new = -model.lb[fake_irr]
        model.ub[fake_irr] = ub_new
        model.lb[fake_irr] = 0.0
    return model


def find_exc_rxns(model: CobraModel, incl_obj_flag: bool = False,
                  irrev_flag: bool = False):
    """`findExcRxns.m` -> (selExc, selUpt) boolean vectors."""
    S = sp.csc_matrix(model.S)
    nnz_per_col = np.asarray((S != 0).sum(axis=0)).ravel()
    if not irrev_flag:
        n_minus1 = np.asarray((S == -1).sum(axis=0)).ravel()
        n_plus1 = np.asarray((S == 1).sum(axis=0)).ravel()
        sel_exc = ((n_minus1 == 1) & (nnz_per_col == 1)) | ((n_plus1 == 1) & (nnz_per_col == 1))
        if model.c is not None:
            sel_exc[model.c != 0] = bool(incl_obj_flag)
        sel_upt = (model.lb < 0) & sel_exc
    else:
        n_abs1 = np.asarray((abs(S) == 1).sum(axis=0)).ravel()
        sel_exc = (n_abs1 == 1) & (nnz_per_col == 1)
        if model.c is not None:
            sel_exc[model.c != 0] = bool(incl_obj_flag)
        n_plus1 = np.asarray((S == 1).sum(axis=0)).ravel()
        sel_upt = (n_plus1 == 1) & (nnz_per_col == 1)
    return sel_exc, sel_upt


_MET_COMP_RE = re.compile(r"^(?P<met>.*)\[(?P<comp>[^\[])\]$")


def parse_met_names(met_names) -> tuple[list, list]:
    """`parseMetNames.m` - split `name[c]` into base name and compartment."""
    base, comp = [], []
    for mn in _as_list(met_names):
        mo = _MET_COMP_RE.match(str(mn))
        if mo:
            base.append(mo.group("met"))
            comp.append(mo.group("comp"))
        else:
            mo2 = re.match(r"^(?P<met>.*)\((?P<comp>[^\[])\)$", str(mn))
            if mo2:
                base.append(mo2.group("met"))
                comp.append(mo2.group("comp"))
            else:
                base.append(str(mn))
                comp.append("")
    return base, comp


def find_trans_rxns(model: CobraModel, incl_exc: bool = False,
                    rxn_inds: Optional[np.ndarray] = None,
                    incl_obj_as_exc: bool = False, irrev_flag: bool = False):
    """`findTransRxns.m` - reactions touching more than one compartment."""
    n = model.n_rxns
    rxn_inds = np.arange(n) if rxn_inds is None else np.asarray(rxn_inds, int)
    if incl_exc:
        is_exc0, _ = find_exc_rxns(model, incl_obj_as_exc, irrev_flag)
        is_exc = is_exc0[rxn_inds]
    else:
        is_exc = np.zeros(rxn_inds.size, bool)

    _, comp_symbols = parse_met_names(model.mets)
    comp_symbols = np.asarray(comp_symbols, dtype=object)
    S = sp.csc_matrix(model.S)
    is_non_exch_trans = np.zeros(rxn_inds.size, bool)
    for i, j in enumerate(rxn_inds):
        rows = S.indices[S.indptr[j]:S.indptr[j + 1]]
        data = S.data[S.indptr[j]:S.indptr[j + 1]]
        rows = rows[data != 0]
        if len(np.unique(comp_symbols[rows])) > 1:
            is_non_exch_trans[i] = True

    trans_bool = is_non_exch_trans | is_exc
    trans_rxns = model.rxns[rxn_inds][trans_bool]
    non_trans_rxns = np.array(sorted(set(model.rxns[rxn_inds]) - set(trans_rxns)), dtype=object)
    return trans_rxns, non_trans_rxns, trans_bool


def find_rxns_from_mets(model: CobraModel, met_list) -> np.ndarray:
    """`findRxnsFromMets.m` (default options) - reactions touching any of `met_list`."""
    met_set = set(_as_list(met_list))
    index = np.array([m in met_set for m in model.mets])
    if not index.any():
        return np.array([], dtype=object)
    S = sp.csr_matrix(model.S)
    totals = np.asarray((S[index, :] != 0).sum(axis=0)).ravel()
    return model.rxns[totals > 0]


def find_organic_ex_rxns(model: CobraModel, biomass_reaction, function_to_keep):
    """`findOrganicExRxns.m` -> (exOrgaRxns, ExOrgaInd).

    Note the MATLAB quirk faithfully reproduced here: `exOrgaRxns` is the list of
    **all** considered exchange reactions, while `ExOrgaInd` holds only the indices
    of the *organic* ones.
    """
    S = sp.csc_matrix(model.S)
    abs_col_sum = np.asarray(abs(S).sum(axis=0)).ravel()
    ex_rxns_ind = np.flatnonzero(abs_col_sum == 1)

    biomass_id = np.flatnonzero(np.isin(model.rxns, _as_list(biomass_reaction)))
    ftk = _as_list(function_to_keep)

    if biomass_id.size == 0 and len(ftk) == 0:
        warnings.warn("No biomass set for the model, please verify if the medium "
                      "constraints do not affect the biomass production")
    elif biomass_id.size == 0:
        warnings.warn("No biomass set for the model, please verify if the medium "
                      "constraints do not affect the biomass production")
        function_id = np.flatnonzero(np.isin(model.rxns, ftk))
        ex_rxns_ind = np.setdiff1d(ex_rxns_ind, function_id)
    else:
        function_id = np.flatnonzero(np.isin(model.rxns, ftk))
        function_id = np.union1d(biomass_id, function_id)
        ex_rxns_ind = np.setdiff1d(ex_rxns_ind, function_id)

    exchange_mets = []
    for j in ex_rxns_ind:
        rows = S.indices[S.indptr[j]:S.indptr[j + 1]]
        data = S.data[S.indptr[j]:S.indptr[j + 1]]
        rows = rows[data != 0]
        exchange_mets.append(int(rows[0]))
    exchange_mets = np.asarray(exchange_mets, dtype=int)

    ex_orga_rxns = model.rxns[ex_rxns_ind]

    if model.metFormulas is None:
        raise ValueError("model.metFormulas is required by findOrganicExRxns")
    formulas = np.asarray([("" if f is None else str(f))
                           for f in model.metFormulas], dtype=object)[exchange_mets]

    has = lambda letter: np.array([letter in f for f in formulas])
    ex_mets_x, ex_mets_y = has("X"), has("Y")
    ex_mets_c, ex_mets_r = has("C"), has("R")
    ex_known_inorganic = np.isin(formulas, np.array(["Ca", "Cl", "Co", "Cu"], dtype=object))

    if ex_mets_x.any():
        print("Warning metabolites with X in their Formulas these inputs are closed")
    if ex_mets_y.any():
        print("Warning metabolites with Y in their Formulas these inputs are closed")

    is_organic = (ex_mets_c | ex_mets_x | ex_mets_y | ex_mets_r) & ~ex_known_inorganic
    ex_orga_ind = ex_rxns_ind[is_organic]
    return ex_orga_rxns, ex_orga_ind


def constrain_model_on_medium(model: CobraModel, medium_mets, not_medium_constrained,
                              biomass_reaction, function_to_keep) -> CobraModel:
    """`constrainModelOnMedium.m`."""
    model = model.copy()
    nmc = _as_list(not_medium_constrained)
    if len(nmc) > 0:
        nmc_idx = np.flatnonzero(np.isin(model.rxns, nmc))
        lb_keep = model.lb[nmc_idx].copy()
        ub_keep = model.ub[nmc_idx].copy()
    else:
        nmc_idx = np.array([], dtype=int)

    ex_rxns, ex_orga_ind = find_organic_ex_rxns(model, biomass_reaction, function_to_keep)

    med_rxns = find_rxns_from_mets(model, medium_mets)
    not_constrained_names = np.intersect1d(np.asarray(med_rxns, dtype=object),
                                           np.asarray(ex_rxns, dtype=object))
    not_constrained = np.flatnonzero(np.isin(model.rxns, not_constrained_names))
    lb2 = model.lb[not_constrained].copy()
    ub2 = model.ub[not_constrained].copy()

    if ex_orga_ind.size:
        sub = sp.csc_matrix(model.S)[:, ex_orga_ind]
        # close uptake of carbon sources
        cols_neg = np.unique(sp.find(sub < 0)[1])
        if cols_neg.size:
            model.lb[ex_orga_ind[cols_neg]] = 0.0
        # close secretion of carbon sources
        cols_pos = np.unique(sp.find(sub > 0)[1])
        if cols_pos.size:
            model.ub[ex_orga_ind[cols_pos]] = 0.0

    if nmc_idx.size:
        model.lb[nmc_idx] = lb_keep
        model.ub[nmc_idx] = ub_keep
    if not_constrained.size:
        model.lb[not_constrained] = lb2
        model.ub[not_constrained] = ub2
    return model


def _subset_fields(model: CobraModel, keep_rxns: np.ndarray, keep_mets: np.ndarray) -> CobraModel:
    out = model.copy()
    out.S = sp.csc_matrix(sp.csc_matrix(model.S)[keep_mets, :][:, keep_rxns])
    for f in RXN_FIELDS:
        v = getattr(out, f, None)
        if v is not None and isinstance(v, np.ndarray) and v.size == model.n_rxns:
            setattr(out, f, v[keep_rxns])
    for f in MET_FIELDS:
        v = getattr(out, f, None)
        if v is not None and isinstance(v, np.ndarray) and v.size == model.n_mets:
            setattr(out, f, v[keep_mets])
    if out.rxnGeneMat is not None:
        out.rxnGeneMat = sp.csr_matrix(sp.csr_matrix(model.rxnGeneMat)[keep_rxns, :])
    for k, v in list(out.extra.items()):
        if isinstance(v, np.ndarray) and v.size == model.n_rxns:
            out.extra[k] = v[keep_rxns]
        elif isinstance(v, np.ndarray) and v.size == model.n_mets:
            out.extra[k] = v[keep_mets]
    return out


def remove_rxns(model: CobraModel, rxn_remove_list, met_remove_method: str = "exclusive",
                met_flag: bool = True):
    """`removeRxns.m` (parameter defaults: metFlag = true, metRemoveMethod = 'exclusive').

    Returns ``(modelOut, metRemovedList)``.
    """
    remove = set(_as_list(rxn_remove_list))
    select_rxns = np.array([r not in remove for r in model.rxns])
    keep_rxns = np.flatnonzero(select_rxns)
    removed_rxns = np.flatnonzero(~select_rxns)

    if met_flag:
        S = sp.csc_matrix(model.S)
        nnz_removed = np.asarray((S[:, removed_rxns] != 0).sum(axis=1)).ravel() if removed_rxns.size else np.zeros(model.n_mets)
        nnz_kept = np.asarray((S[:, keep_rxns] != 0).sum(axis=1)).ravel() if keep_rxns.size else np.zeros(model.n_mets)
        if met_remove_method == "exclusive":
            remove_mets = (nnz_removed > 0) & (nnz_kept == 0)
        elif met_remove_method == "inclusive":
            remove_mets = nnz_removed > 0
        else:
            raise ValueError(met_remove_method)
        met_removed_list = model.mets[remove_mets]
        keep_mets = np.flatnonzero(~remove_mets)
    else:
        met_removed_list = np.array([], dtype=object)
        keep_mets = np.arange(model.n_mets)

    return _subset_fields(model, keep_rxns, keep_mets), met_removed_list


_RULE_X_RE = re.compile(r"x\((\d+)\)")


def remove_unused_genes(model: CobraModel) -> CobraModel:
    """`removeUnusedGenes.m` - drop genes no longer referenced by any rule."""
    if model.genes is None or model.rules is None:
        return model
    model = model.copy()
    used = np.zeros(len(model.genes), bool)
    for rule in model.rules:
        if rule is None:
            continue
        for g in _RULE_X_RE.findall(str(rule)):
            used[int(g) - 1] = True
    if used.all():
        return model
    old2new = -np.ones(len(model.genes), int)
    old2new[used] = np.arange(int(used.sum()))

    def renumber(rule):
        if rule is None or str(rule) == "":
            return rule
        return _RULE_X_RE.sub(lambda mo: f"x({old2new[int(mo.group(1)) - 1] + 1})", str(rule))

    model.rules = np.array([renumber(r) for r in model.rules], dtype=object)
    model.genes = model.genes[used]
    if model.rxnGeneMat is not None:
        model.rxnGeneMat = sp.csr_matrix(sp.csr_matrix(model.rxnGeneMat)[:, used])
    return model


def change_objective(model: CobraModel, rxn_name_list, objective_coeff=1.0) -> CobraModel:
    """`changeObjective.m`."""
    model = model.copy()
    model.c = np.zeros(model.n_rxns)
    idx = np.flatnonzero(np.isin(model.rxns, _as_list(rxn_name_list)))
    model.c[idx] = objective_coeff
    return model


def optimize_cb_model(model: CobraModel, osense_str: str = "max",
                      solver: Optional[str] = None) -> dict:
    """`optimizeCbModel.m` restricted to what rFASTCORMICS_v2 needs (the value of `.f`)."""
    lp = build_opt_problem_from_model(model)
    lp.osense = -1 if osense_str.lower().startswith("max") else 1
    sol = solve_cobra_lp(lp, solver=solver)
    out = {"stat": sol["stat"], "origStat": sol["origStat"]}
    if sol["stat"] == 1:
        out["x"] = sol["full"]
        out["f"] = float(np.dot(model.c, sol["full"]))
    else:
        out["x"] = None
        out["f"] = np.nan
    return out


# ======================================================================================
# 3.  FASTCORE family
# ======================================================================================

def LP7(J, model: CobraModel, lp: LPProblem, epsilon: float, basis=None,
        solver: Optional[str] = None):
    """`LP7.m` - maximise the number of reactions in J carrying flux >= epsilon."""
    J = np.atleast_1d(np.asarray(J, int))
    nj = J.size
    m2, n2 = lp.A.shape

    f = -np.concatenate([np.zeros(n2), np.ones(nj)])

    Aeq = sp.hstack([lp.A, sp.csr_matrix((m2, nj))], format="csr")
    beq = lp.b

    Ij = sp.csr_matrix((-np.ones(nj), (np.arange(nj), J)), shape=(nj, n2))
    Aineq = sp.hstack([Ij, sp.identity(nj, format="csr")], format="csr")
    bineq = np.zeros(nj)

    lb = np.concatenate([lp.lb, np.zeros(nj)])
    ub = np.concatenate([lp.ub, np.full(nj, epsilon)])

    prob = LPProblem(A=sp.vstack([Aeq, Aineq], format="csr"),
                     b=np.concatenate([beq, bineq]),
                     c=f, lb=lb, ub=ub,
                     csense=np.concatenate([np.asarray(lp.csense, dtype=object),
                                            np.array(["L"] * nj, dtype=object)]),
                     osense=1)
    sol = solve_cobra_lp(prob, solver=solver)
    if sol["stat"] != 1:
        warnings.warn(f"LP7: solution may not be optimal (stat={sol['stat']})")
    x = sol["full"]
    v = x[:n2] if x is not None else np.full(n2, np.nan)
    return v, None            # basis is never reused (see module docstring)


def LP3(J, model: CobraModel, lp: LPProblem, basis=None, solver: Optional[str] = None):
    """`LP3.m` - maximise the flux through the single reaction J."""
    J = np.atleast_1d(np.asarray(J, int))
    n = model.S.shape[1]
    m2, n2 = lp.A.shape

    lp = lp.copy()
    Ir = model.ub <= 0
    if Ir.any():
        idx = np.flatnonzero(Ir)
        A = sp.lil_matrix(lp.A)
        A[:, idx] = -A[:, idx]
        lp.A = sp.csr_matrix(A)
        tmp = lp.ub[idx].copy()
        lp.ub[idx] = -lp.lb[idx]
        lp.lb[idx] = -tmp

    f = np.zeros(n2)
    f[J] = -1.0
    prob = LPProblem(A=lp.A, b=lp.b, c=f, lb=lp.lb, ub=lp.ub,
                     csense=lp.csense, osense=1)
    sol = solve_cobra_lp(prob, solver=solver)
    x = sol["full"]
    v = (x[:n] if x is not None else np.full(n, np.nan)).copy()
    if Ir.any():
        v[Ir] = -v[Ir]
    return v, None


def LP10(K, P, v, lp: LPProblem, epsilon: float, adaptive_scaling_flag: int = 0,
         non_pen=None, solver: Optional[str] = None):
    """`LP10.m` - keep K active while minimising the L1 activity of P."""
    K = np.atleast_1d(np.asarray(K, int))
    P = np.atleast_1d(np.asarray(P, int)) if P is not None and len(P) else np.array([], int)
    if P.size == 0 or K.size == 0:
        return np.array([])

    non_pen = np.array([], int) if non_pen is None else np.atleast_1d(np.asarray(non_pen, int))
    np_, nk = P.size, K.size
    m2, n2 = lp.A.shape

    def _build(bineq_k_value, scaling):
        f = np.concatenate([np.zeros(n2), np.ones(np_)])
        if non_pen.size:
            not_pen = np.flatnonzero(np.isin(P, non_pen))
            if not_pen.size:
                f[n2 + not_pen] = 0.0
        Aeq = sp.hstack([lp.A, sp.csr_matrix((m2, np_))], format="csr")
        Ip = sp.csr_matrix((np.ones(np_), (np.arange(np_), P)), shape=(np_, n2))
        Ik = sp.csr_matrix((np.ones(nk), (np.arange(nk), K)), shape=(nk, n2))
        I_np = sp.identity(np_, format="csr")
        Aineq = sp.vstack([
            sp.hstack([Ip, -I_np], format="csr"),
            sp.hstack([-Ip, -I_np], format="csr"),
            sp.hstack([-Ik, sp.csr_matrix((nk, np_))], format="csr"),
        ], format="csr")
        bineq = np.concatenate([np.zeros(2 * np_), -np.ones(nk) * bineq_k_value])
        lb = np.concatenate([lp.lb, np.zeros(np_)])
        ub = np.concatenate([lp.ub, np.maximum(np.abs(lp.ub[P]), np.abs(lp.lb[P]))])
        return LPProblem(A=sp.vstack([Aeq, Aineq], format="csr"),
                         b=np.concatenate([lp.b, bineq]) * scaling,
                         c=f, lb=lb * scaling, ub=ub * scaling,
                         csense=np.concatenate([np.asarray(lp.csense, dtype=object),
                                                np.array(["L"] * (2 * np_ + nk), dtype=object)]),
                         osense=1)

    sol = None
    if adaptive_scaling_flag == 1:
        min_bound = np.min(v[K]) * 10.0
        sol = solve_cobra_lp(_build(min_bound, 1.0), solver=solver)
        if sol["stat"] != 1:
            print("LP10: Adaptive scaling factor did not work, switching to fixed "
                  "scaling factor of 1e4")
            adaptive_scaling_flag = 0
            sol = None

    if sol is None:
        scaling = 1e4
        sol = solve_cobra_lp(_build(epsilon, scaling), solver=solver)
        if sol["stat"] != 1:
            warnings.warn("solution.stat~=1. Scaling factor in LP10 is 10000, which "
                          "may cause numerical instability for solveCobraLP.")

    x = sol["full"]
    return x[:n2] if x is not None else np.full(n2, np.nan)


def find_sparse_mode(J, P, singleton: bool, model: CobraModel, lp: LPProblem,
                     epsilon: float, adaptive_scaling_flag: int = 0, basis=None,
                     non_pen=None, solver: Optional[str] = None):
    """`findSparseMode.m`."""
    J = np.asarray(J, int)
    if J.size == 0:
        return np.array([], int), basis

    if singleton:
        v, basis = LP7(J[:1], model, lp, epsilon, basis, solver=solver)
    else:
        v, basis = LP7(J, model, lp, epsilon, basis, solver=solver)

    K = np.intersect1d(J, np.flatnonzero(v >= 0.99 * epsilon))
    if K.size == 0:
        return np.array([], int), basis

    v = LP10(K, P, v, lp, epsilon, adaptive_scaling_flag, non_pen, solver=solver)
    supp = np.flatnonzero(np.abs(v) >= 0.99 * epsilon)
    return supp, basis


def fastcc(model: CobraModel, epsilon: Optional[float] = None, print_level: int = 1,
           mode_flag: int = 0, method: str = "original", solver: Optional[str] = None):
    """`fastcc.m` (method='original') -> 0-based indices of flux-consistent reactions."""
    if method != "original":
        raise NotImplementedError("only method='original' is ported")
    if epsilon is None:
        epsilon = FEAS_TOL * 100
    if epsilon < FEAS_TOL:
        raise ValueError(f"fastcc will not work with epsilon = {epsilon} < feastol = {FEAS_TOL}")

    n = model.n_rxns
    N = np.arange(n)
    lp = build_opt_problem_from_model(model)

    Ir = (model.lb < 0) & (model.ub <= 0)
    if Ir.any():
        idx = np.flatnonzero(Ir)
        A = sp.lil_matrix(lp.A)
        A[:, idx] = -A[:, idx]
        lp.A = sp.csr_matrix(A)
        tmp = lp.ub[idx].copy()
        lp.ub[idx] = -lp.lb[idx]
        lp.lb[idx] = -tmp

    I = np.flatnonzero(model.lb >= 0)
    J = np.intersect1d(N, I)
    if print_level > 0:
        print(f"{n:6d}\tTotal reactions")
        print(f"{n - I.size:6d}\tReversible reactions.")
        print(f"{I.size:6d}\tIrreversible reactions.")

    v, basis = LP7(J, model, lp, epsilon, solver=solver)
    A_set = np.flatnonzero(np.abs(v) >= 0.99 * epsilon)
    if print_level > 1:
        print(f"{A_set.size:6d}\tFlux consistent reactions, without flipping.")

    incI = np.setdiff1d(J, A_set)
    J = np.setdiff1d(np.setdiff1d(N, A_set), incI)
    if print_level > 1:
        print(f"{J.size:6d}\tFlux inconsistent reactions, without flipping.")

    flipped = False
    singleton = False
    while J.size:
        if singleton:
            Ji = J[:1]
            v, basis = LP3(Ji, model, lp, basis, solver=solver)
        else:
            Ji = J
            v, basis = LP7(Ji, model, lp, epsilon, basis, solver=solver)
        supp = np.flatnonzero(np.abs(v) >= 0.99 * epsilon)
        nA1 = A_set.size
        A_set = np.union1d(A_set, supp)
        if print_level > 1 and A_set.size > nA1:
            print(f"{A_set.size:6d}\tFlux consistent reactions.")

        if np.intersect1d(J, A_set).size:
            J = np.setdiff1d(J, A_set)
            if print_level > 1:
                print(f"{J.size:6d}\tFlux inconsistent reversible reactions left to flip.")
            flipped = False
        else:
            JiRev = np.setdiff1d(Ji, I)
            if flipped or JiRev.size == 0:
                flipped = False
                if singleton:
                    J = np.setdiff1d(J, Ji)
                    if print_level > 2:
                        print(f"{model.rxns[Ji][0]}\tis flux inconsistent.")
                else:
                    singleton = True
            else:
                A = sp.lil_matrix(lp.A)
                A[:, JiRev] = -A[:, JiRev]
                lp.A = sp.csr_matrix(A)
                tmp = lp.ub[JiRev].copy()
                lp.ub[JiRev] = -lp.lb[JiRev]
                lp.lb[JiRev] = -tmp
                flipped = True

    if A_set.size == n and print_level > 0:
        print("fastcc.m: The input model is entirely flux consistent.")
    orientation = np.ones(n)
    orientation[Ir] = -1
    return A_set, orientation, None


def fastcore(model: CobraModel, core_rxn_ind, epsilon: Optional[float] = None,
             print_level: int = 0, adaptive_scaling_flag: int = 0, non_pen=None,
             solver: Optional[str] = None):
    """`fastcore.m` -> (tissueModel, coreRxnBool)."""
    if epsilon is None:
        epsilon = FEAS_TOL * 100
    non_pen = np.array([], int) if non_pen is None else np.atleast_1d(np.asarray(non_pen, int))

    model_orig = model
    n_rxns = model.n_rxns
    lp = build_opt_problem_from_model(model)

    Ir = (model.lb < 0) & (model.ub <= 0)
    if Ir.any():
        idx = np.flatnonzero(Ir)
        A = sp.lil_matrix(lp.A)
        A[:, idx] = -A[:, idx]
        lp.A = sp.csr_matrix(A)
        tmp = lp.ub[idx].copy()
        lp.ub[idx] = -lp.lb[idx]
        lp.lb[idx] = -tmp

    irrev_rxns = np.flatnonzero(model.lb >= 0)
    core_rxn_ind = np.unique(np.asarray(core_rxn_ind, int))

    flipped = False
    singleton = False

    J = np.intersect1d(core_rxn_ind, irrev_rxns)
    if print_level > 0:
        print(f"|J|={J.size}  ", end="")

    nb_rxns = np.arange(n_rxns)
    P = np.setdiff1d(nb_rxns, core_rxn_ind)

    supp, basis = find_sparse_mode(J, P, singleton, model, lp, epsilon,
                                   adaptive_scaling_flag, None, non_pen, solver=solver)
    missing = np.setdiff1d(J, supp)
    if missing.size:
        warnings.warn("fastcore.m Error: Global network is not flux consistent, "
                      "ignoring the following irreversible core reactions:\n"
                      + ", ".join(map(str, model.rxns[missing])))
        core_rxn_ind = np.setdiff1d(core_rxn_ind, missing)

    A_set = supp
    if print_level > 0:
        print(f"|A|={A_set.size}")

    J = np.setdiff1d(core_rxn_ind, A_set)
    if print_level > 0:
        print(f"|J|={J.size}  ", end="")

    while J.size:
        P = np.setdiff1d(P, A_set)
        supp, basis = find_sparse_mode(J, P, singleton, model, lp, epsilon,
                                       adaptive_scaling_flag, basis, non_pen, solver=solver)
        A_set = np.union1d(A_set, supp)
        if print_level > 0:
            print(f"|A|={A_set.size}")

        if np.intersect1d(J, A_set).size:
            J = np.setdiff1d(J, A_set)
            if print_level > 0:
                print(f"|J|={J.size}  ", end="")
            flipped = False
        else:
            JiRev = np.setdiff1d(J[:1], irrev_rxns) if singleton else np.setdiff1d(J, irrev_rxns)
            if flipped or JiRev.size == 0:
                if singleton:
                    warnings.warn("fastcore.m: Global network is not flux consistent, "
                                  "ignoring corresponding core reaction:\n"
                                  + ", ".join(map(str, model.rxns[J])))
                    J = np.array([], int)
                else:
                    flipped = False
                    singleton = True
            else:
                A = sp.lil_matrix(lp.A)
                A[:, JiRev] = -A[:, JiRev]
                lp.A = sp.csr_matrix(A)
                tmp = lp.ub[JiRev].copy()
                lp.ub[JiRev] = -lp.lb[JiRev]
                lp.lb[JiRev] = -tmp
                flipped = True
                if print_level > 0:
                    print("(flip)  ", end="")

    if print_level > 0:
        print(f"|A|={A_set.size}")

    core_rxn_bool = np.zeros(n_rxns, bool)
    core_rxn_bool[A_set] = True

    rxn_remove_list = np.setdiff1d(np.asarray(model.rxns, dtype=object),
                                   np.asarray(model.rxns[A_set], dtype=object))
    tissue_model, _ = remove_rxns(model, rxn_remove_list, met_remove_method="exclusive")
    tissue_model = remove_unused_genes(tissue_model)
    return tissue_model, core_rxn_bool


# ======================================================================================
# 4.  Expression mapping
# ======================================================================================

class _GPRValue:
    """Wrapper implementing MATLAB's overloaded `and` = min, `or` = max."""
    __slots__ = ("a",)

    def __init__(self, a):
        self.a = a

    def __and__(self, other):
        return _GPRValue(np.minimum(self.a, other.a))

    def __or__(self, other):
        return _GPRValue(np.maximum(self.a, other.a))


class _GPRVars:
    __slots__ = ("X",)

    def __init__(self, X):
        self.X = X

    def __getitem__(self, i):
        return _GPRValue(self.X[i])


_RULE_TO_PY = re.compile(r"x\((\d+)(?:\s*,\s*:)?\)")


def gpr_rules_mapper(rule: str, mapped_to_genes: np.ndarray) -> np.ndarray:
    """`GPRrulesMapper.m` - evaluate a COBRA `rules` string with min/max semantics.

    `mapped_to_genes` is (nGenes x nSamples); returns a vector of length nSamples.
    """
    n_samples = mapped_to_genes.shape[1]
    if rule is None or str(rule).strip() == "":
        return np.zeros(n_samples)
    expr = _RULE_TO_PY.sub(lambda mo: f"x[{int(mo.group(1)) - 1}]", str(rule))
    val = eval(expr, {"__builtins__": {}}, {"x": _GPRVars(mapped_to_genes)})
    if isinstance(val, _GPRValue):
        return np.asarray(val.a, float)
    return np.asarray(val, float)


def map_expression_to_model(model: CobraModel, data: np.ndarray, dico,
                            rownames, process_transcripts: int = 0,
                            verbose: bool = True) -> np.ndarray:
    """`mapExpressionToModel.m` -> mapping matrix (nRxns x nSamples)."""
    import pandas as pd

    if isinstance(dico, pd.DataFrame):
        dico_arr = dico.to_numpy(dtype=object)
    else:
        dico_arr = np.asarray(dico, dtype=object)
        if dico_arr.ndim == 1:
            dico_arr = dico_arr.reshape(-1, 1)

    rownames = np.asarray(_as_list(rownames), dtype=object)
    if any((r is None or str(r) == "") for r in rownames):
        raise ValueError("rownames contains empty entries, please check")

    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # --- pick the dico column with the most matches in rownames ----------------
    rn_set = set(map(str, rownames))
    counts = np.array([sum(1 for v in dico_arr[:, j] if str(v) in rn_set)
                       for j in range(dico_arr.shape[1])])
    col = int(np.flatnonzero(counts == counts.max())[0])

    # --- MATLAB intersect(dico(:,col), rownames): sorted unique, first occurrence
    dico_col = np.array([str(v) for v in dico_arr[:, col]], dtype=object)
    rn_str = np.array([str(v) for v in rownames], dtype=object)
    first_d, first_r = {}, {}
    for i, v in enumerate(dico_col):
        first_d.setdefault(v, i)
    for i, v in enumerate(rn_str):
        first_r.setdefault(v, i)
    common = sorted(set(first_d) & set(first_r))
    if not common:
        raise ValueError("rownames does not match the dataIds in the dico")
    idico = np.array([first_d[v] for v in common], dtype=int)
    irownames = np.array([first_r[v] for v in common], dtype=int)

    mapped = data[irownames, :]

    # mapped2 = [rownames(irownames), dico(idico, :)]
    mapped2 = np.empty((idico.size, 1 + dico_arr.shape[1]), dtype=object)
    mapped2[:, 0] = rn_str[irownames]
    for j in range(dico_arr.shape[1]):
        mapped2[:, j + 1] = np.array([str(v) for v in dico_arr[idico, j]], dtype=object)

    genes = np.array([str(g) for g in (model.genes if model.genes is not None else [])],
                     dtype=object)
    if process_transcripts == 1 and any("." in g for g in genes):
        if verbose:
            print("Transcripts will temporarily be removed to allow matching with the "
                  "discretized data.")
        genes = np.array([re.sub(r"\.[0-9]*", "", g) for g in genes], dtype=object)

    # value -> row indices of mapped2 (search over ALL columns, like ismember on a matrix)
    lookup: dict = {}
    for r in range(mapped2.shape[0]):
        for cidx in range(mapped2.shape[1]):
            lookup.setdefault(mapped2[r, cidx], set()).add(r)

    mapped_to_genes = np.zeros((genes.size, mapped.shape[1]))
    genes_matched = 0
    for i, g in enumerate(genes):
        match = sorted(lookup.get(g, ()))
        if len(match) == 1:
            mapped_to_genes[i, :] = mapped[match[0], :]
        elif len(match) > 1:
            mapped_to_genes[i, :] = mapped[np.asarray(match), :].max(axis=0)
        if match:
            genes_matched += 1
    if verbose:
        print(f"{genes_matched} of {genes.size} genes of the model matched")

    mapping = np.zeros((model.n_rxns, mapped.shape[1]))
    rules = model.rules if model.rules is not None else np.array([""] * model.n_rxns,
                                                                 dtype=object)
    for k, rule in enumerate(rules):
        if rule is None or str(rule) == "":
            continue
        mapping[k, :] = gpr_rules_mapper(rule, mapped_to_genes)
    return mapping


# ======================================================================================
# 5.  Discretisation
# ======================================================================================

def ksdensity(x: np.ndarray, npoints: int = 100):
    """MATLAB `ksdensity(x)` with default options (normal kernel, unbounded support).

    Reproduces statslib's default bandwidth
    ``bw = (median(|x - median(x)|)/0.6745) * (4/(3n))^(1/5)``
    and default grid ``linspace(min(x) - 3bw, max(x) + 3bw, 100)``.
    """
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    med = np.median(x)
    sig = np.median(np.abs(x - med)) / 0.6745
    if sig <= 0:
        sig = x.max() - x.min()
    bw = sig * (4.0 / (3.0 * n)) ** 0.2 if sig > 0 else 1.0

    xi = np.linspace(x.min() - 3.0 * bw, x.max() + 3.0 * bw, npoints)
    # kernel density; chunked to bound memory for large n
    f = np.zeros(npoints)
    chunk = 200_000
    for start in range(0, n, chunk):
        xs = x[start:start + chunk]
        z = (xi[None, :] - xs[:, None]) / bw
        f += np.exp(-0.5 * z * z).sum(axis=0)
    f /= (n * bw * np.sqrt(2.0 * np.pi))
    return f, xi, bw


def fit_gauss1(x: np.ndarray, y: np.ndarray, start_point, lower=(-np.inf, -np.inf, 0.0),
               ftol: float = 1e-10, xtol: float = 1e-10):
    """MATLAB `fit(x, y, 'gauss1')` with `Lower`/`StartPoint` as in createFitR/createFitL.

    Model: ``a1 * exp(-((x - b1)/c1)^2)``.  Returns ``(a1, b1, c1)``.
    """
    from scipy.optimize import least_squares

    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)         # prepareCurveData
    x, y = x[ok], y[ok]

    def resid(p):
        a1, b1, c1 = p
        c1 = c1 if c1 != 0 else 1e-300
        return a1 * np.exp(-(((x - b1) / c1) ** 2)) - y

    lo = np.array(lower, float)
    hi = np.array([np.inf, np.inf, np.inf])
    p0 = np.array(start_point, float)
    p0 = np.clip(p0, lo + np.where(np.isfinite(lo), 1e-12, 0.0), hi)
    res = least_squares(resid, p0, bounds=(lo, hi), method="trf",
                        ftol=ftol, xtol=xtol, gtol=1e-12, max_nfev=6000)
    return tuple(res.x)


#: StartPoint values hard-coded in createFitR.m / createFitL.m
_START_R = (0.215494818454649, 5.36184523579772, 1.10452062242243)
_START_L = (0.0442312785349232, -0.515420615759673, 2.77989514663167)


def _hybrid_and_rest(prob: np.ndarray, peak_idx: int):
    """Mirror the right flank of the density around `peak_idx` (0-based)."""
    npts = prob.size
    max_r_side = prob[peak_idx:]
    max_l_side = max_r_side[::-1]
    hybrid = np.concatenate([max_l_side[:-1], max_r_side])

    hybrid_curve = np.zeros(npts)
    if hybrid.size > npts:
        hybrid_curve[:] = hybrid[-npts:]
    else:
        hybrid_curve[npts - hybrid.size:] = hybrid

    rest = prob - hybrid_curve
    rest[rest < 0.0001] = 0.0
    return hybrid_curve, rest


def discretize_fpkm(fpkm, colnames=None, figflag: int = 0, verbose: bool = True,
                    return_details: bool = False):
    """`discretizeFPKM.m` -> (discretized, scaledExpression[, details]).

    ``discretized``: 1 = expressed, 0 = unknown, -1 = not expressed.
    """
    fpkm = np.asarray(fpkm, float)
    if fpkm.ndim == 1:
        fpkm = fpkm.reshape(-1, 1)
    n_genes, n_samples = fpkm.shape

    with np.errstate(divide="ignore", invalid="ignore"):
        signal = np.log2(fpkm)
    signal[np.isinf(signal)] = -1e6

    discretized_keep = np.zeros((n_samples, n_genes))
    scaled = np.zeros((n_genes, n_samples))
    details = []

    for j in range(n_samples):
        signal_sample = signal[:, j]
        signal_sample = signal_sample[signal_sample > -1e6]

        prob, xi, bw = ksdensity(signal_sample)

        # -- METHOD 1: right-side peak (MATLAB indices 50:100 -> python 49:100)
        seg_max = prob[49:100].max()
        peak_idx_r = int(np.flatnonzero(prob == seg_max)[0])
        hyb_r, rest_r = _hybrid_and_rest(prob, peak_idx_r)
        a_r, b_r, c_r = fit_gauss1(xi, hyb_r, _START_R)
        a_lr, b_lr, c_lr = fit_gauss1(xi, rest_r, _START_L)
        sigma1_r = c_r / np.sqrt(2.0)
        mu1_r, mu2_r = b_r, b_lr
        ue_r = max(-3.0, (mu2_r - mu1_r) / sigma1_r)

        # -- METHOD 2: global peak
        peak_idx_g = int(np.flatnonzero(prob == prob.max())[0])
        hyb_g, rest_g = _hybrid_and_rest(prob, peak_idx_g)
        a_g, b_g, c_g = fit_gauss1(xi, hyb_g, _START_R)
        a_lg, b_lg, c_lg = fit_gauss1(xi, rest_g, _START_L)
        sigma1_g = c_g / np.sqrt(2.0)
        mu1_g, mu2_g = b_g, b_lg
        ue_g = max(-3.0, (mu2_g - mu1_g) / sigma1_g)

        # -- select method
        if ue_r < 0:
            mu1, mu2, sigma1, ue = mu1_r, mu2_r, sigma1_r, ue_r
        else:
            mu1, mu2, sigma1, ue = mu1_g, mu2_g, sigma1_g, ue_g

        if ue_g > 0 and verbose:
            print("Your distribution was left skewed.")

        z = (signal[:, j] - mu1) / sigma1
        scaled[:, j] = z

        d = np.zeros(n_genes)
        d[z >= 0.0] = 1.0                 # expressed  (exp_threshold = 0)
        d[z <= ue] = -1.0                 # not expressed
        d[(z < 0.0) & (z > ue)] = 0.0     # unknown
        discretized_keep[j, :] = d

        details.append(dict(sample=None if colnames is None else colnames[j],
                            mu1=mu1, mu2=mu2, sigma1=sigma1, ue=ue,
                            ue_r=ue_r, ue_g=ue_g, method=1 if ue_r < 0 else 2,
                            bandwidth=bw))

    discretized = discretized_keep.T
    if return_details:
        return discretized, scaled, details
    return discretized, scaled


# ======================================================================================
# 6.  rFASTCORMICS_v2 main routine
# ======================================================================================

def r_fastcormics(model: CobraModel, discretized, rownames, dico,
                  biomass_reaction_name: str, consensus_proportion: float = 0.9,
                  epsilon: float = 1e-4, optional_settings: Optional[dict] = None,
                  filling_medium_flag: int = 1, adaptive_scaling_flag: int = 0,
                  solver: Optional[str] = None, verbose: bool = True):
    """`rFastcormics.m`.

    Returns ``(contextSpecificModel, retainedRxns, indicesCompletedCoreOrig)`` where the
    two index arrays are **0-based** indices into the reactions of the *input* model
    (MATLAB returns the same sets 1-based).
    """
    def say(*a):
        if verbose:
            print(*a)

    discretized = np.asarray(discretized)
    if discretized.ndim == 1:
        discretized = discretized.reshape(-1, 1)
    rownames = np.asarray(_as_list(rownames), dtype=object)
    optional_settings = {} if optional_settings is None else dict(optional_settings)

    if not (0 < consensus_proportion <= 1):
        raise ValueError("consensusProportion must be in (0, 1]")
    if biomass_reaction_name not in set(model.rxns):
        raise ValueError(f"The objective function that you defined "
                         f"({biomass_reaction_name}) is not part of your input model")
    if discretized.shape[0] != rownames.size:
        raise ValueError(f"Row mismatch: discretized ({discretized.shape[0]}) vs "
                         f"rownames ({rownames.size}).")

    # ---- .func handling ------------------------------------------------------
    if "func" in optional_settings:
        function_keep = _as_list(optional_settings["func"])
        if biomass_reaction_name not in function_keep:
            function_keep = function_keep + [biomass_reaction_name]
            say(f"The reaction {biomass_reaction_name} has been added to the set of "
                f"reactions that will be forced to carry a flux (.func reactions).")
    else:
        function_keep = [biomass_reaction_name]
        say(f"The reaction {biomass_reaction_name} has been added to the set of "
            f"reactions that will be forced to carry a flux (.func reactions).")

    orig_model = model
    model = model.copy()

    # ---- correcting the format of the model ---------------------------------
    model = fix_irr(model)

    # ---- consistent model ----------------------------------------------------
    consistent_rxns, _, _ = fastcc(model, epsilon, 0, 0, "original", solver=solver)
    drop = np.setdiff1d(np.arange(model.n_rxns), consistent_rxns)
    consistent_model, _ = remove_rxns(model, model.rxns[drop])

    # ---- medium constraints --------------------------------------------------
    if "medium" in optional_settings:
        not_medium_constrained = _as_list(optional_settings.get("notMediumConstrained", []))
        medium_mets = _as_list(optional_settings["medium"])

        _, upt_rxns_bool = find_exc_rxns(consistent_model)
        upt_rxns = consistent_model.rxns[upt_rxns_bool]
        medium_rxns = find_rxns_from_mets(consistent_model, medium_mets)
        upt_medium_rxns = np.intersect1d(np.asarray(upt_rxns, dtype=object),
                                         np.asarray(medium_rxns, dtype=object))
        say(f"Number of uptake reactions associated with medium metabolites: "
            f"{upt_medium_rxns.size}.")

        medium_constrained_model = constrain_model_on_medium(
            consistent_model, medium_mets, not_medium_constrained,
            biomass_reaction_name, function_keep)
    elif not optional_settings:
        medium_constrained_model = consistent_model.copy()
        warnings.warn("No optional settings detected.")
        upt_medium_rxns = np.array([], dtype=object)
    else:
        medium_constrained_model = consistent_model.copy()
        warnings.warn("No given medium.")
        upt_medium_rxns = np.array([], dtype=object)

    # ---- mapping -------------------------------------------------------------
    mapping = map_expression_to_model(medium_constrained_model, discretized, dico,
                                      rownames, 1, verbose=verbose)
    if mapping.sum() == 0:
        raise RuntimeError("No expressed genes were mapped, check again.")

    # ---- core ----------------------------------------------------------------
    number_of_samples = discretized.shape[1]
    initial_core = np.flatnonzero((mapping == 1).sum(axis=1)
                                  >= consensus_proportion * number_of_samples)

    trans_rxns, _, _ = find_trans_rxns(medium_constrained_model)
    trans_ids = np.flatnonzero(np.isin(medium_constrained_model.rxns, trans_rxns))
    core_without_trans = np.setdiff1d(initial_core, trans_ids)
    say(f"Number of core reactions (after mapping, transporters removed, without "
        f".func reactions): {core_without_trans.size}")

    found_function_keep = np.flatnonzero(np.isin(medium_constrained_model.rxns, function_keep))
    if found_function_keep.size == 0:
        warnings.warn("No reactions from .func set were found in the model.")
    elif found_function_keep.size != len(function_keep):
        warnings.warn("Part of the reactions from .func set were not found in the model.")
    completed_core = np.union1d(core_without_trans, found_function_keep)

    rxn_names_completed_core = medium_constrained_model.rxns[completed_core]
    indices_completed_core_orig = np.flatnonzero(
        np.isin(orig_model.rxns, rxn_names_completed_core))

    # ---- remove reactions controlled by unexpressed genes --------------------
    not_expressed = np.flatnonzero((mapping == -1).sum(axis=1)
                                   >= consensus_proportion * number_of_samples)
    say(f"Number of non expressed reactions (after mapping): {not_expressed.size}.")
    medium_constrained_model.lb[not_expressed] = 0.0
    medium_constrained_model.ub[not_expressed] = 0.0

    consistent_after, _, _ = fastcc(medium_constrained_model, epsilon, 0, 0,
                                    "original", solver=solver)
    drop2 = np.setdiff1d(np.arange(medium_constrained_model.n_rxns), consistent_after)
    consistent_medium_constrained_model, _ = remove_rxns(
        medium_constrained_model, medium_constrained_model.rxns[drop2])

    # ---- medium sufficiency --------------------------------------------------
    if biomass_reaction_name in set(consistent_medium_constrained_model.rxns):
        consistent_medium_constrained_model = change_objective(
            consistent_medium_constrained_model, biomass_reaction_name)
        fba = optimize_cb_model(consistent_medium_constrained_model, "max", solver=solver)
        if fba["f"] is not None and not np.isnan(fba["f"]) and fba["f"] >= 1e-4:
            if "medium" in optional_settings:
                say(f"Model still contains {biomass_reaction_name} after application of "
                    f"medium constraints, and FBA result is not null.\nMedium is "
                    f"sufficient. Continuing with fastcore.")
            need_medium_filling = False
        else:
            say(f"Model still contains {biomass_reaction_name} after application of "
                f"medium constraints, but FBA result is {fba['f']}. Medium is not "
                f"sufficient.")
            need_medium_filling = True
    else:
        say(f"Model lost {biomass_reaction_name} after application of medium constraints.")
        need_medium_filling = True

    if need_medium_filling and "medium" in optional_settings:
        _, upt_after_bool = find_exc_rxns(consistent_medium_constrained_model)
        upt_after = consistent_medium_constrained_model.rxns[upt_after_bool]
        not_present = np.setdiff1d(upt_medium_rxns, np.asarray(upt_after, dtype=object))
        if not_present.size:
            say("The following uptake reactions:\n" + "\n".join(map(str, not_present)) +
                "\nwere initially not included in the consistent medium constraint "
                "model, even though they were associated with medium metabolites."
                "\nIn the case of medium filling, their inclusion in the model will "
                "not be penalized.")

    correct_indices_completed_core = np.flatnonzero(
        np.isin(consistent_medium_constrained_model.rxns, rxn_names_completed_core))

    # ---- fastcore ------------------------------------------------------------
    context_specific_model, retained_bool = fastcore(
        consistent_medium_constrained_model, correct_indices_completed_core, 1e-4, 0,
        adaptive_scaling_flag, solver=solver)
    keep_rxns = np.flatnonzero(retained_bool)
    retained_rxns = np.flatnonzero(
        np.isin(orig_model.rxns, consistent_medium_constrained_model.rxns[keep_rxns]))

    # ---- .func check ---------------------------------------------------------
    ctx_rxn_set = set(context_specific_model.rxns)
    missing_function_keep = [r for r in function_keep if r not in ctx_rxn_set]
    if missing_function_keep:
        say("The following .func reactions are missing:\n"
            + "\n".join(missing_function_keep)
            + "\n and will be added to the context-specific model through a second "
              "fastcore run.")
        function_keep_flag = True
    else:
        function_keep_flag = False

    # ---- medium filling / .func filling --------------------------------------
    if (filling_medium_flag == 1 and need_medium_filling) or function_keep_flag:
        common, idx_cons, idx_ctx = np.intersect1d(
            np.asarray(consistent_model.rxns, dtype=object),
            np.asarray(context_specific_model.rxns, dtype=object),
            return_indices=True)
        consistent_model = consistent_model.copy()
        consistent_model.lb[idx_cons] = context_specific_model.lb[idx_ctx]
        consistent_model.ub[idx_cons] = context_specific_model.ub[idx_ctx]

        filling_core_rxns = np.union1d(
            np.asarray(context_specific_model.rxns, dtype=object),
            np.asarray([biomass_reaction_name], dtype=object))

        if filling_medium_flag == 1 and need_medium_filling:
            say("Proceeding to medium filling.")
            _, upt_ctx_bool = find_exc_rxns(context_specific_model)
            upt_ctx = context_specific_model.rxns[upt_ctx_bool]
            not_present_spe = np.setdiff1d(upt_medium_rxns, np.asarray(upt_ctx, dtype=object))
            if not_present_spe.size and "medium" in optional_settings:
                say("The inclusion of the following uptake reactions:\n"
                    + "\n".join(map(str, not_present_spe))
                    + "\nwill not be penalized to fill the model as they are initially "
                      "associated with medium metabolites.")
        else:
            not_present_spe = np.array([], dtype=object)

        if function_keep_flag:
            say("Proceeding to .func filling.")
            filling_core_rxns = np.union1d(
                filling_core_rxns, np.asarray(missing_function_keep, dtype=object))

        indices_filling_core = np.flatnonzero(np.isin(consistent_model.rxns, filling_core_rxns))
        indices_non_pen = np.flatnonzero(np.isin(consistent_model.rxns, not_present_spe))

        context_specific_model, retained_filled_bool = fastcore(
            consistent_model, indices_filling_core, 1e-4, 0, adaptive_scaling_flag,
            non_pen=indices_non_pen, solver=solver)
        keep_rxns = np.flatnonzero(retained_filled_bool)
        retained_filled = np.flatnonzero(
            np.isin(orig_model.rxns, consistent_model.rxns[keep_rxns]))
        supplementary = np.setdiff1d(retained_filled, retained_rxns)
        say("The following reactions have been added to the model during the second "
            "fastcore run in order to fill it:\n"
            + "\n".join(map(str, orig_model.rxns[supplementary])))
        retained_rxns = retained_filled

    # ---- report additional uptakes ------------------------------------------
    if optional_settings.get("medium") is not None and len(_as_list(optional_settings["medium"])):
        _, upt_bool_spe = find_exc_rxns(context_specific_model)
        upt_spe = context_specific_model.rxns[upt_bool_spe]
        additional_medium = np.setdiff1d(np.asarray(upt_spe, dtype=object), upt_medium_rxns)
        say("Model additional uptakes (not provided by the medium):\n"
            + "\n".join(map(str, additional_medium)))

    say("rFastcormics is done.")
    return context_specific_model, retained_rxns, indices_completed_core_orig


# ======================================================================================
# 7.  cobrapy bridge
# ======================================================================================

def to_cobrapy(model: CobraModel, model_id: str = "context_specific_model"):
    """Convert a :class:`CobraModel` into a ``cobra.Model``."""
    import cobra
    from cobra import Metabolite, Reaction

    cm = cobra.Model(model_id)
    mets = []
    for i, mid in enumerate(model.mets):
        met = Metabolite(str(mid))
        if model.metNames is not None:
            met.name = str(model.metNames[i])
        if model.metFormulas is not None and model.metFormulas[i]:
            met.formula = str(model.metFormulas[i])
        mo = _MET_COMP_RE.match(str(mid))
        met.compartment = mo.group("comp") if mo else ""
        mets.append(met)
    cm.add_metabolites(mets)

    S = sp.csc_matrix(model.S)
    rxns = []
    for j, rid in enumerate(model.rxns):
        r = Reaction(str(rid))
        r.lower_bound = float(model.lb[j])
        r.upper_bound = float(model.ub[j])
        if model.rxnNames is not None:
            r.name = str(model.rxnNames[j])
        if model.subSystems is not None:
            r.subsystem = str(model.subSystems[j])
        rxns.append(r)
    cm.add_reactions(rxns)
    for j, rid in enumerate(model.rxns):
        rows = S.indices[S.indptr[j]:S.indptr[j + 1]]
        vals = S.data[S.indptr[j]:S.indptr[j + 1]]
        cm.reactions.get_by_id(str(rid)).add_metabolites(
            {cm.metabolites[int(i)]: float(v) for i, v in zip(rows, vals)})
    if model.grRules is not None:
        for j, rid in enumerate(model.rxns):
            gr = "" if model.grRules[j] is None else str(model.grRules[j])
            if gr:
                cm.reactions.get_by_id(str(rid)).gene_reaction_rule = gr
    obj = np.flatnonzero(model.c != 0)
    if obj.size:
        cm.objective = {cm.reactions.get_by_id(str(model.rxns[j])): float(model.c[j])
                        for j in obj}
    return cm


def from_cobrapy(cm) -> CobraModel:
    """Convert a ``cobra.Model`` into a :class:`CobraModel` (COBRA-Toolbox layout)."""
    import cobra  # noqa: F401

    rxns = np.array([r.id for r in cm.reactions], dtype=object)
    mets = np.array([m.id for m in cm.metabolites], dtype=object)
    met_pos = {m.id: i for i, m in enumerate(cm.metabolites)}
    rows, cols, vals = [], [], []
    for j, r in enumerate(cm.reactions):
        for m, v in r.metabolites.items():
            rows.append(met_pos[m.id]); cols.append(j); vals.append(float(v))
    S = sp.csc_matrix((vals, (rows, cols)), shape=(len(mets), len(rxns)))

    genes = np.array([g.id for g in cm.genes], dtype=object)
    gene_pos = {g: i + 1 for i, g in enumerate(genes)}

    def gr_to_rule(gr: str) -> str:
        if not gr:
            return ""
        out = gr.replace(" and ", " & ").replace(" or ", " | ")
        for g in sorted(gene_pos, key=len, reverse=True):
            out = re.sub(rf"(?<![\w.\-]){re.escape(g)}(?![\w.\-])", f"x({gene_pos[g]})", out)
        return out

    obj_c = np.array([float(r.objective_coefficient) for r in cm.reactions])
    return CobraModel(
        S=S, rxns=rxns, mets=mets,
        lb=np.array([r.lower_bound for r in cm.reactions], float),
        ub=np.array([r.upper_bound for r in cm.reactions], float),
        c=obj_c,
        b=np.zeros(len(mets)),
        csense=np.array(["E"] * len(mets), dtype=object),
        genes=genes,
        rules=np.array([gr_to_rule(r.gene_reaction_rule) for r in cm.reactions], dtype=object),
        grRules=np.array([r.gene_reaction_rule for r in cm.reactions], dtype=object),
        subSystems=np.array([r.subsystem for r in cm.reactions], dtype=object),
        metFormulas=np.array([(m.formula or "") for m in cm.metabolites], dtype=object),
        metNames=np.array([m.name for m in cm.metabolites], dtype=object),
        rxnNames=np.array([r.name for r in cm.reactions], dtype=object),
    )
