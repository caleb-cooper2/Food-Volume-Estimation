# Benchmark summary

Generated 2026-09-08 10:17 from `benchmark_simplefood45_llm_results_gemini3.1pro.csv` (50 scored results)

```
RQ2 - volume MAPE vs ground truth (success < 20%):
  llm                      18.6%  (n=50)  -> PASS

RQ3 - NLP text vs image-only, paired per image (success: text reduces error):
  no paired nlp/notext arms in this results file

==============================================================================================================================
approach              arm                  n  V-MAPE%  V-bias%  M-MAPE%  lat_s    overhead        tilt   side_left  side_right
------------------------------------------------------------------------------------------------------------------------------
llm                   checkerboard        25     17.4     -3.0        -   15.5           -           -           -           -
llm                   sizeprior_notext    25     19.8     -2.9        -   16.1           -           -           -           -
==============================================================================================================================

======================================================
approach              tilt         n  V-MAPE%  V-bias%
------------------------------------------------------
llm                   mild        32     16.2     -9.0
llm                   oblique     16     21.2      9.9
======================================================

```
