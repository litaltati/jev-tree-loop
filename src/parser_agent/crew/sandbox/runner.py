"""Runs inside the chart sandbox: loads the rows into `df`, runs the agent's code.

The code finds `df` (a pandas DataFrame of the query result), `pd`, `plt` and
`OUT` — the path the chart must be saved to. A figure left open without a
savefig call is saved for it: forgetting that line should not cost a retry.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
# Defaults that keep long labels inside the picture, whatever the code does.
matplotlib.rcParams.update({"figure.autolayout": True, "savefig.bbox": "tight", "savefig.dpi": 110})
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUT = "/job/out/chart.png"

with open("/job/data.json", encoding="utf-8") as handle:
    data = json.load(handle)
df = pd.DataFrame(data["rows"], columns=data["columns"])

with open("/job/code.py", encoding="utf-8") as handle:
    code = handle.read()

namespace = {"df": df, "pd": pd, "plt": plt, "OUT": OUT, "__name__": "__main__"}
exec(compile(code, "chart_code.py", "exec"), namespace)

if not os.path.exists(OUT) and plt.get_fignums():
    plt.savefig(OUT, dpi=110, bbox_inches="tight")
if not os.path.exists(OUT):
    raise SystemExit("The code drew no chart: nothing was saved to OUT")
