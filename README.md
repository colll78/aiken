# Flamegraph evidence for aiken-lang/aiken#1389

pprof (SIGPROF, 497 Hz) flamegraphs of `aiken check -m
'validation_machine_v1.{static_rules_prove_a_network_mismatch_is_an_exact_no_op}'`
on the Midgard on-chain project — one expensive test whose transitive call
graph covers ~1,240 hoisted function variants.

- `before-main-6aa5105-one-heavy-test.svg.gz` — aiken `main` @ 6aa5105
  (~85 s). Dominated by `Term::var_occurrences` (per-binder occurrence scans
  in the UPLC shrinker) with `substitute_var` and the hoisting worklists
  behind it.
- `after-tracker-and-hoisting-fixes.svg.gz` — mid-series (after the capped,
  argument-first, tracker-backed scans and hoisting fixes; before the module
  constant cache). The scan pyramid is gone; remaining time is the traversal
  itself, module-constant recompilation (fixed in a later commit) and the
  frontend.

Decompress with `gunzip` and open in a browser.
