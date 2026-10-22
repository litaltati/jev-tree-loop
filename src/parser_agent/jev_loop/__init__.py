"""The Jev loop: Jev picks the next agent, the agents share one state.

The Jev tree goes first (jev_tree/): its rows and notes start the state, or it
says which step comes first.

    question ─▶ Jev: what next? ─┬─ sql         ─▶ the SQL agent fills state.data ──┐
                     ▲           ├─ visualizer  ─▶ the visualizer fills state.chart ─┤
                     └───────────┴──────────────────────────────────────────────────┘
                                 └─ exit ─▶ the answer analyst writes the answer from the state

Every agent reads the question, the conversation and what the others found
from the same LoopState, so nothing has to wait for the question or be handed
back by hand. Code limits what Jev may choose (no chart without rows, no third
SQL run, nothing after the chart but the answer), so the loop always ends.
Jev sees what was found only as a description — table, columns, how many rows
— never the figures.
"""
