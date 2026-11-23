# Dressed-state printing — short design

**Intent.** Print a dressed state as a sum of bare product states, for notebooks and papers, the way
`dressed_state_decomp` in `two_qubit_gate_for_zero_pi_qubit/ham_data.py` does in Python.

**Goals.** One printing function, physics (0-based) indices in and out, plain text or LaTeX.
**Non-goals.** No change to `decompose`, `dressed_index` or `bare_labels` (they stay 1-based); no nested
relabeling (a later `depth` keyword if wanted).

**Approaches.** (1) A separate string function (chosen: the user wants it apart from `decompose`);
(2) a result type for `decompose` with `show` methods (rejected: changes `decompose`); (3) a heatmap (rejected:
printing only).

**Design.** `dressed_state_string(sys, [node,] n; thresh=1e-3, digits=3, label=nothing, latex=false) -> String`.
`n` counts kept dressed states from 0; bare labels count each child's kept states from 0; the docstring says so first.
Terms with |c|² ≥ `thresh`, by descending weight, amplitudes rounded to `digits`; real amplitudes joined with their
signs, complex ones in parentheses (a complex amplitude whose imaginary part rounds to zero prints as real).
Text `|ψ₃⟩ = 0.981|0,1⟩ − 0.172|1,0⟩`; LaTeX `\widetilde{|\psi_{3}\rangle} = 0.981\,|0,1\rangle - 0.172\,|1,0\rangle`.
`label` replaces `ψ₃` (`\psi_{3}`). No term above `thresh` gives a right side naming the threshold.
