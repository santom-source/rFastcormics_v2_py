"""Command-line version of notebook 03 (handy for long runs / servers).

    python run_brca_example.py [gurobi|highs] [main|bis|ter]

Requires `data/` produced by 01_extract_matlab_data.ipynb (or extract_brca_data.py).
"""
from __future__ import annotations

import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

import rfastcormics_v2 as rf


def load_model(path="data/model.pkl") -> rf.CobraModel:
    m = pickle.load(open(path, "rb"))
    return rf.CobraModel(
        S=m["S"], rxns=m["rxns"], mets=m["mets"], lb=m["lb"], ub=m["ub"], c=m["c"],
        b=m["b"], csense=m["csense"], osense=m["osense"], genes=m["genes"],
        rules=m["rules"], grRules=m["grRules"], rxnGeneMat=m["rxnGeneMat"],
        subSystems=m["subSystems"], metFormulas=m["metFormulas"],
        metNames=m["metNames"], rxnNames=m["rxnNames"])


def main(solver="gurobi", case="main", use_matlab_discretized=False):
    rf.set_default_solver(solver)

    model = load_model()
    exp = np.load("data/expression.npz", allow_pickle=True)
    ref = np.load("data/matlab_reference.npz", allow_pickle=True)
    dico = pd.read_csv("data/dico.csv", dtype=str)
    settings = json.load(open("data/optional_settings.json"))

    # --- tutorialScript.m: consistency check + objective ---------------------
    t0 = time.time()
    consistent_rxns, _, _ = rf.fastcc(model, 1e-4, 1)
    drop = np.setdiff1d(np.arange(model.n_rxns), consistent_rxns)
    consistent_model, _ = rf.remove_rxns(model, model.rxns[drop])
    consistent_model = rf.change_objective(consistent_model, "biomass_reaction")
    print("fastcc %.1f s ->" % (time.time() - t0), consistent_model, flush=True)
    print("fastcc identical to MATLAB:",
          np.array_equal(consistent_rxns, np.sort(ref["consistentRxnsBool"] - 1)))

    # --- discretisation -------------------------------------------------------
    if use_matlab_discretized:
        discretized = exp["discretized_matlab"]
    else:
        discretized = rf.discretize_fpkm(exp["fpkm"], exp["colnames"], verbose=False)[0]

    opt = {"main": settings["optionalSettings"],
           "bis": settings["optionalSettingsBis"],
           "ter": None}[case]
    key = {"main": "retainedRxns", "bis": "retainedRxnsBis", "ter": "retainedRxnsTer"}[case]

    t0 = time.time()
    ctx, retained, core = rf.r_fastcormics(
        consistent_model, discretized, exp["rownames"], dico, "biomass_reaction",
        0.9, 1e-4, opt)
    print("rFastcormics %.1f s" % (time.time() - t0), flush=True)

    ml = np.sort(ref[key] - 1)
    py = np.sort(retained)
    inter = np.intersect1d(py, ml)
    print(f"\n=== {case} | solver={solver} ===")
    print(f"python {py.size} rxns | matlab {ml.size} rxns | identical={np.array_equal(py, ml)}")
    print(f"intersection {inter.size} | jaccard {inter.size / np.union1d(py, ml).size:.4f}")
    print("python-only:", list(consistent_model.rxns[np.setdiff1d(py, ml)]))
    print("matlab-only:", list(consistent_model.rxns[np.setdiff1d(ml, py)]))
    np.save(f"data/retained_{case}_{solver}.npy", py)
    return ctx, py, ml


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gurobi",
         sys.argv[2] if len(sys.argv) > 2 else "main")
