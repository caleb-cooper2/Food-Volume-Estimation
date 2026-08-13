# Benchmark summary

Generated 2026-08-13 15:55 from `benchmark_simplefood45_llm_results_qwen3vl.csv` (84 scored results)

```
RQ2 - volume MAPE vs ground truth (success < 20%):
  llm                      53.1%  (n=84)  -> FAIL

RQ3 - NLP text vs image-only, paired per image (success: text reduces error):
  no paired nlp/notext arms in this results file

==============================================================================================================================
approach              arm                  n  V-MAPE%  V-bias%  M-MAPE%  lat_s    overhead        tilt   side_left  side_right
------------------------------------------------------------------------------------------------------------------------------
llm                   checkerboard        42     56.5      5.1        -    5.6           -           -           -           -
llm                   sizeprior_notext    42     49.7     -0.1        -    4.4           -           -           -           -
==============================================================================================================================

======================================================
approach              tilt         n  V-MAPE%  V-bias%
------------------------------------------------------
llm                   mild        54     56.6      7.1
llm                   oblique     24     27.0    -26.2
======================================================

```
