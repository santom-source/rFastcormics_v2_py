"""Extract everything needed from the MATLAB BRCA example into portable files.

Run once:

    python extract_brca_data.py "<path to>/rFASTCORMICS_v2/BRCA_example"

Produces (in ./data):
    model.pkl              generic model (COBRA Toolbox field layout)
    expression.npz         fpkm, rownames, colnames, MATLAB `discretized`
    dico.csv               ENTREZ / SYMBOL mapping table
    optional_settings.json func / medium / notMediumConstrained
    matlab_reference.npz   MATLAB outputs used as ground truth
"""

from __future__ import annotations

import json
import pathlib
import pickle
import sys

import numpy as np
import scipy.io as sio

import matlab_io as mio

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "data"


def main(brca_dir: str) -> None:
    brca = pathlib.Path(brca_dir)
    ws = brca / "workspaceExample.mat"
    OUT.mkdir(exist_ok=True)

    d = sio.loadmat(ws, struct_as_record=False)

    # ---- generic model ------------------------------------------------------
    model = mio.struct_to_model_dict(d["model"][0, 0])
    with open(OUT / "model.pkl", "wb") as fh:
        pickle.dump(model, fh, protocol=4)
    print("model:", model["S"].shape, len(model["rxns"]), "rxns,",
          len(model["genes"]), "genes")

    # ---- expression ---------------------------------------------------------
    fpkm = np.asarray(d["fpkm"], float)
    rownames = mio.cellstr(d["rownames"])
    colnames = mio.cellstr(d["colnames"])
    discretized_matlab = np.asarray(d["discretized"], float)
    np.savez_compressed(OUT / "expression.npz", fpkm=fpkm, rownames=rownames,
                        colnames=colnames, discretized_matlab=discretized_matlab)
    print("fpkm:", fpkm.shape, "| discretized(MATLAB):", discretized_matlab.shape)

    # ---- dico (MATLAB table) ------------------------------------------------
    dico = mio.read_matlab_table(ws)
    dico.to_csv(OUT / "dico.csv", index=False)
    print("dico:", dico.shape, list(dico.columns))

    # ---- optional settings --------------------------------------------------
    opt = d["optionalSettings"][0, 0]
    settings = {f: [str(x) for x in mio.cellstr(getattr(opt, f))]
                for f in opt._fieldnames}
    optb = d["optionalSettingsBis"][0, 0]
    settings_bis = {f: [str(x) for x in mio.cellstr(getattr(optb, f))]
                    for f in optb._fieldnames}
    with open(OUT / "optional_settings.json", "w") as fh:
        json.dump({"optionalSettings": settings,
                   "optionalSettingsBis": settings_bis}, fh, indent=1)
    print("optionalSettings:", {k: len(v) for k, v in settings.items()})

    # ---- MATLAB reference outputs ------------------------------------------
    ref = dict(
        consistentRxnsBool=np.asarray(d["consistentRxnsBool"], int).ravel(),
        retainedRxns=np.asarray(d["retainedRxns"], int).ravel(),
        retainedRxnsBis=np.asarray(d["retainedRxnsBis"], int).ravel(),
        retainedRxnsTer=np.asarray(d["retainedRxnsTer"], int).ravel(),
        BCRAmodel_rxns=mio.cellstr(d["BCRAmodel"][0, 0].rxns),
        BCRAmodel_genes=mio.cellstr(d["BCRAmodel"][0, 0].genes),
        BCRAmodelBis_rxns=mio.cellstr(d["BCRAmodelBis"][0, 0].rxns),
        BCRAmodelTer_rxns=mio.cellstr(d["BCRAmodelTer"][0, 0].rxns),
        biomassRxn=str(np.asarray(d["biomassRxn"]).ravel()[0]),
        consensusProportion=float(np.asarray(d["consensusProportion"]).ravel()[0]),
        epsilon=float(np.asarray(d["epsilon"]).ravel()[0]),
        # --- downstream analysis reference (tutorialScript.m sections "FBA" / "gene deletions")
        fba_f_control=float(np.asarray(d["fbaControl"][0, 0].f).ravel()[0]),
        fba_f_BRCA=float(np.asarray(d["fbaBCRA"][0, 0].f).ravel()[0]),
        grRateWT_control=float(np.asarray(d["grRateWTControl"]).ravel()[0]),
        grRateWT_BRCA=float(np.asarray(d["grRateWTBCRA"]).ravel()[0]),
        grRatio_control=np.asarray(d["grRatioControl"], float).ravel(),
        grRateKO_control=np.asarray(d["grRateKOControl"], float).ravel(),
        grRatio_BRCA=np.asarray(d["grRatioBCRA"], float).ravel(),
        grRateKO_BRCA=np.asarray(d["grRateKOBCRA"], float).ravel(),
        # MATLAB's own context-specific model, with its bounds - lets you validate the
        # downstream analysis independently of any FASTCORE difference
        BCRAmodel_lb=np.asarray(d["BCRAmodel"][0, 0].lb, float).ravel(),
        BCRAmodel_ub=np.asarray(d["BCRAmodel"][0, 0].ub, float).ravel(),
    )
    np.savez_compressed(OUT / "matlab_reference.npz", **ref)
    print("reference retainedRxns:", ref["retainedRxns"].size,
          "| Bis:", ref["retainedRxnsBis"].size,
          "| Ter:", ref["retainedRxnsTer"].size)
    print("written to", OUT)


if __name__ == "__main__":
    default = ("/tmp/rFASTCORMICS/rFASTCORMICS for RNA-seq data/"
               "rFASTCORMICS_v2/BRCA_example")
    main(sys.argv[1] if len(sys.argv) > 1 else default)
