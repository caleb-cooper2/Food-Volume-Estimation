# Benchmark summary

Generated 2026-08-13 15:46 from `benchmark_simplefood45_llm_results_opus5.csv` (84 scored results)

```
RQ2 - volume MAPE vs ground truth (success < 20%):
  llm                      28.8%  (n=84)  -> FAIL

RQ3 - NLP text vs image-only, paired per image (success: text reduces error):
  no paired nlp/notext arms in this results file

==============================================================================================================================
approach              arm                  n  V-MAPE%  V-bias%  M-MAPE%  lat_s    overhead        tilt   side_left  side_right
------------------------------------------------------------------------------------------------------------------------------
llm                   checkerboard        42     32.1     20.6        -    8.3           -           -           -           -
llm                   sizeprior_notext    42     25.5     17.3        -    8.3           -           -           -           -
==============================================================================================================================

======================================================
approach              tilt         n  V-MAPE%  V-bias%
------------------------------------------------------
llm                   mild        54     32.2     24.4
llm                   oblique     24     23.2     11.1
======================================================

```
