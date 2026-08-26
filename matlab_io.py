"""Reading the MATLAB workspace that ships with rFASTCORMICS_v2.

Two awkward bits:

1. The COBRA model struct comes back as nested mat_struct + cell-of-char and needs
   flattening into numpy object arrays.
2. `dico` is a MATLAB `table`, i.e. a classdef (MCOS) object. loadmat only returns a
   MatlabOpaque placeholder because the payload lives in the *subsystem* section of
   the MAT file. Parsed by hand here: bytes 116-123 of the header give the subsystem
   offset -> a compressed element -> inflate it and you get a miniature MAT file
   holding the FileWrapper__ cell array with the table's columns and column names.

MAT-file v7 only (which is what the repo uses).
"""

from __future__ import annotations

import io
import struct

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.io.matlab._mio5 import MatFile5Reader


# --------------------------------------------------------------------------------------
# generic struct helpers
# --------------------------------------------------------------------------------------

def cellstr(a) -> np.ndarray:
    """MATLAB cell array of char -> 1-D numpy object array of ``str``."""
    if a is None:
        return np.array([], dtype=object)
    a = np.asarray(a, dtype=object).ravel()
    out = np.empty(a.size, dtype=object)
    for i, v in enumerate(a):
        if isinstance(v, np.ndarray):
            v = v.ravel()
            out[i] = "" if v.size == 0 else str(v[0])
        elif v is None:
            out[i] = ""
        else:
            out[i] = str(v)
    return out


def struct_to_model_dict(ms) -> dict:
    """`scipy` ``mat_struct`` of a COBRA model -> plain dict of numpy arrays."""
    d = {}
    for f in ms._fieldnames:
        v = getattr(ms, f)
        if f == "S":
            d["S"] = sp.csc_matrix(v)
        elif f == "rxnGeneMat":
            d["rxnGeneMat"] = sp.csr_matrix(v)
        elif f in ("lb", "ub", "c", "b", "rev", "metCharges", "rxnConfidenceScores"):
            d[f] = np.asarray(v, dtype=float).ravel()
        elif f == "osense":
            d[f] = int(np.asarray(v).ravel()[0])
        elif f == "csense":
            d[f] = cellstr(v) if np.asarray(v).dtype == object else \
                np.asarray([str(x) for x in np.asarray(v).ravel()], dtype=object)
        elif f == "subSystems":
            # can be a cell of cells
            arr = np.asarray(v, dtype=object).ravel()
            out = np.empty(arr.size, dtype=object)
            for i, x in enumerate(arr):
                xr = np.asarray(x, dtype=object).ravel()
                if xr.size == 0:
                    out[i] = ""
                else:
                    y = xr[0]
                    out[i] = str(np.asarray(y).ravel()[0]) if isinstance(y, np.ndarray) and np.asarray(y).size else str(y)
            d[f] = out
        else:
            arr = np.asarray(v, dtype=object)
            if arr.dtype == object:
                d[f] = cellstr(arr)
            else:
                d[f] = np.asarray(v).ravel()
    return d


# --------------------------------------------------------------------------------------
# MATLAB table (MCOS) extraction
# --------------------------------------------------------------------------------------

def _read_subsystem(path: str):
    """Return the ``FileWrapper__`` cell array stored in the MAT-file subsystem."""
    raw = open(path, "rb").read()
    subsys_offset = struct.unpack("<q", raw[116:124])[0]
    if subsys_offset <= 0:
        raise ValueError("this MAT-file has no subsystem section (no MCOS objects)")
    mdtype, nbytes = struct.unpack("<II", raw[subsys_offset:subsys_offset + 8])
    element = raw[subsys_offset:subsys_offset + 8 + nbytes]

    # wrap the single element in a valid MAT-file and let scipy inflate it
    d = sio.loadmat(io.BytesIO(raw[:128] + element), struct_as_record=False)
    fw = d["__function_workspace__"].tobytes()

    # the function workspace is itself a MAT-file: 4 byte version/endian marker,
    # 4 padding bytes, then the data elements
    buf = b" " * 124 + fw[:4] + fw[8:]
    stream = io.BytesIO(buf)
    reader = MatFile5Reader(stream, struct_as_record=False)
    reader.initialize_read()
    stream.seek(128)
    hdr, _ = reader.read_var_header()
    arr = reader.read_var_array(hdr, process=True)
    mcos = arr[0, 0].MCOS
    return np.asarray(mcos["arr"][0], dtype=object).ravel()


def read_matlab_table(path: str) -> "object":
    """Read the (single) MATLAB ``table`` stored in `path` as a pandas DataFrame.

    The MCOS ``FileWrapper__`` cell holds, among the class metadata,
    * a 1 x nCols cell of column data (each entry a cell array of char), and
    * a 1 x nCols cell of column names.
    """
    import pandas as pd

    cells = _read_subsystem(path)

    col_names, col_data = None, None
    for c in cells:
        c = np.asarray(c, dtype=object)
        if c.dtype != object or c.size == 0:
            continue
        flat = c.ravel()
        # a cell of char scalars -> candidate for the variable-name list
        if all(isinstance(x, np.ndarray) and x.dtype.kind in "US" for x in flat):
            if col_names is None or c.size > len(col_names):
                col_names = [str(np.asarray(x).ravel()[0]) for x in flat]
        # a cell of cells -> candidate for the column data
        elif all(isinstance(x, np.ndarray) and x.dtype == object and x.size > 1
                 for x in flat):
            if col_data is None or flat[0].size > col_data[0].size:
                col_data = [np.asarray(x, dtype=object).ravel() for x in flat]

    if col_data is None:
        raise ValueError("could not locate the table column data in the MCOS payload")
    cols = [cellstr(c) for c in col_data]
    if not col_names or len(col_names) != len(cols):
        col_names = [f"Var{i + 1}" for i in range(len(cols))]
    return pd.DataFrame({n: c for n, c in zip(col_names, cols)})
