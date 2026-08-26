# rFASTCORMICS_v2 in Python

Port of `rFASTCORMICS for RNA-seq data/rFASTCORMICS_v2/` from
[sysbiolux/rFASTCORMICS](https://github.com/sysbiolux/rFASTCORMICS), function by function,
checked against the official `BRCA_example`.

## Files

| | |
|---|---|
| `rfastcormics_v2.py` | the port: `fastcc`, `fastcore`, `LP3/LP7/LP10`, `findSparseMode`, `discretizeFPKM`, `mapExpressionToModel`, `constrainModelOnMedium`, `rFastcormics` |
| `matlab_io.py` | reading the MATLAB workspace, including the MCOS parsing needed for the `dico` table |
| `01_extract_matlab_data.ipynb` | pull model / FPKM / dico / settings / MATLAB results out of `workspaceExample.mat` into `data/` |
| `02_discretization.ipynb` | `discretizeFPKM` vs MATLAB, cell by cell |
| `03_run_brca_example.ipynb` | the three cases from `tutorialScript.m`, compared with MATLAB's `retainedRxns` |
| `04_downstream_analysis.ipynb` | FBA / pFBA / FVA / deletions via cobrapy, plus a check against MATLAB's FBA and gene deletion results |
| `05_pipeGEM_vs_MATLAB_comparison.ipynb` | notes on how pipeGEM's rFASTCORMICS differs from v2 |
| `extract_brca_data.py`, `run_brca_example.py` | CLI versions of 01 and 03 |
| `test_rfastcormics.py` | port of `scripts/testrFastcormics.m`, plus a case that triggers the second fastcore run |

## Setup

```bash
pip install numpy scipy pandas matplotlib cobra gurobipy jupyterlab
```

Gurobi is the default backend because the MATLAB reference was produced with
`changeCobraSolver('gurobi')`; it is configured with COBRA's defaults
(`FeasibilityTol = OptimalityTol = 1e-6`, `Method = -1`). Note the licence bundled with
`pip install gurobipy` caps models at 2000 variables and the LPs here have ~16000, so a real
licence is needed. Without one, `rf.set_default_solver("highs")` falls back to
`scipy.optimize.linprog`.

Run the notebooks in order. `01` looks for `workspaceExample.mat` in `./BRCA_example/`.

## Usage

```python
import rfastcormics_v2 as rf

rf.set_default_solver("gurobi")

# model is an rf.CobraModel; rf.from_cobrapy(cobra_model) converts one
discretized, scaled = rf.discretize_fpkm(fpkm, colnames)

context_model, retained_rxns, core_rxns = rf.r_fastcormics(
    model, discretized, rownames, dico,
    biomass_reaction_name="biomass_reaction",
    consensus_proportion=0.9, epsilon=1e-4,
    optional_settings={"func": [...], "medium": [...], "notMediumConstrained": [...]},
    filling_medium_flag=1, adaptive_scaling_flag=0)

cobra_model = rf.to_cobrapy(context_model)
```

`retained_rxns` and `core_rxns` are 0-based indices; MATLAB returns the same sets 1-based.

For downstream work use cobrapy - only the minimal FBA that rFASTCORMICS needs internally is
reimplemented here (`rf.optimize_cb_model`):

```python
from cobra.flux_analysis import (pfba, flux_variability_analysis,
                                 single_gene_deletion, find_essential_genes)

ctx = rf.to_cobrapy(context_model); ctx.solver = "gurobi"
ctx.optimize()
flux_variability_analysis(ctx, fraction_of_optimum=0.9)
single_gene_deletion(ctx)
```

Gene IDs are ENTREZ with a transcript suffix (`8639.1`); cobrapy escapes those correctly, and
`data/dico.csv` maps them to symbols.

## MATLAB -> Python

| MATLAB | Python |
|---|---|
| `rFastcormics.m` | `rf.r_fastcormics` |
| `discretizeFPKM.m` + `createFitR/L.m` | `rf.discretize_fpkm` (`rf.ksdensity`, `rf.fit_gauss1`) |
| `mapExpressionToModel.m` + `GPRrulesMapper.m` | `rf.map_expression_to_model`, `rf.gpr_rules_mapper` |
| `constrainModelOnMedium.m` / `findOrganicExRxns.m` | `rf.constrain_model_on_medium`, `rf.find_organic_ex_rxns` |
| `fixIrr.m` | `rf.fix_irr` |
| `fastcc.m` / `fastcore.m` / `findSparseMode.m` | `rf.fastcc`, `rf.fastcore`, `rf.find_sparse_mode` |
| `LP3/LP7/LP10.m` | `rf.LP3`, `rf.LP7`, `rf.LP10` |
| `findExcRxns`, `findTransRxns`, `findRxnsFromMets`, `removeRxns`, `removeUnusedGenes`, `changeObjective`, `optimizeCbModel`, `buildOptProblemFromModel` | same names, snake_case |

### MATLAB quirks kept on purpose

- `basis` is reset to `[]` before use in `LP3.m` / `LP7.m`, so warm starts never actually
  happen. Not reimplemented here either.
- `findOrganicExRxns.m` returns *all* candidate exchange reactions as `exOrgaRxns`, not just the
  organic ones. `constrainModelOnMedium` relies on that.
- `fastcore.m` clears the whole of `J` when it gets stuck in singleton mode, while `fastcc.m`
  drops only the one reaction. The commented-out `error()` above it shows this used to be a hard
  abort in the original Vlassis code.
- MATLAB's `intersect` / `setdiff` / `union` / `find` return sorted results, so numpy's
  equivalents keep FASTCORE walking the same path.
- `mapExpressionToModel`: `intersect` takes the first occurrence, several probes mapping to one
  model gene take the max, GPR is `and = min` / `or = max` with `&` binding tighter than `|`.
- `processTranscripts=1` strips `.n` from model gene names before matching.

## How close it gets

On `BRCA_example` (Recon 2.04, 10601 rxns / 5837 mets / 2248 genes, 25 TCGA samples):

- `fastcc` gives exactly the same consistent set as MATLAB - 10600 reactions, same one dropped
  (`r12989`).
- `discretizeFPKM` differs on 14 of 584200 cells (99.9976%), all sitting on the `ue` threshold,
  from MATLAB's `fit` stopping at `TolFun = 1e-6`. The consensus gene sets are identical, and so
  is the reaction x sample mapping matrix, so it never propagates.
- Core (1449 after removing transporters and adding `.func`) and the 509 non-expressed knockouts
  match the MATLAB log. Medium is judged sufficient in both, so only one fastcore run.
- `removeRxns` + `removeUnusedGenes` reproduce `BCRAmodel` exactly (2584 / 2134 / 1290) when
  given MATLAB's own reaction list.
- `testrFastcormics.m` passes reaction for reaction.
- FBA on MATLAB's own `BCRAmodel` matches to 8 decimals (50.67169641), and single gene deletion
  reproduces `grRatioBCRA` for all 1108 genes - see notebook 04, section 9, which also documents the
  `strmatch` prefix-matching quirk in COBRA's `singleGeneDeletion.m`.

The one thing that drifts is the final reaction set from `fastcore`. LP7 and LP10 are degenerate:
unique optimal value, many optimal vertices, and which reactions end up in the support depends on
which vertex the solver returns. Gurobi lands around 2578 vs MATLAB's 2584 (jaccard 0.976),
HiGHS around 2575 (0.970). The differences are matched pairs - `3DPHBH1`/`3DPHBH2`,
`ACGALK`/`ACGALK2`, `CHSTEROLt1`/`CHSTEROLt2` - i.e. equivalent alternatives. MATLAB drifts the
same way if you change its solver. Compare growth rate and essential genes rather than reaction
membership (notebook 04, section 9c).

## Not ported

- `fastcc(..., method='nonconvex')` - v2 only uses `'original'`.
- The `figflag` plotting inside `discretizeFPKM` - notebook 02 redraws the equivalent figures.
- Models with coupling constraints (`model.C` / `d` / `E`); `buildOptProblemFromModel` only
  handles the uncoupled case, which is what the BRCA example is.
- Most of the `arguments` block validation, apart from the checks that actually fire.
