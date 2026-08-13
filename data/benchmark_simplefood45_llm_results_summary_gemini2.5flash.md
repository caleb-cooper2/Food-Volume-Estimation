# Benchmark summary

Generated 2026-08-13 15:33 from `benchmark_simplefood45_llm_results.csv` (84 scored results)

```
RQ2 - volume MAPE vs ground truth (success < 20%):
  llm                      26.8%  (n=84)  -> FAIL

RQ3 - NLP text vs image-only, paired per image (success: text reduces error):
  no paired nlp/notext arms in this results file

==============================================================================================================================
approach              arm                  n  V-MAPE%  V-bias%  M-MAPE%  lat_s    overhead        tilt   side_left  side_right
------------------------------------------------------------------------------------------------------------------------------
llm                   checkerboard        42     22.9     -6.8        -    3.6           -           -           -           -
llm                   sizeprior_notext    42     30.7      1.9        -    3.4           -           -           -           -
==============================================================================================================================

======================================================
approach              tilt         n  V-MAPE%  V-bias%
------------------------------------------------------
llm                   mild        54     28.4      2.5
llm                   oblique     24     20.2    -18.8
======================================================

```
