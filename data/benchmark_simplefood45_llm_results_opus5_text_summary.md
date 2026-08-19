# Benchmark summary

Generated 2026-08-20 10:15 from `benchmark_simplefood45_llm_results_opus5_text.csv` (67 scored results)

```
RQ2 - volume MAPE vs ground truth (success < 20%):
  llm                      25.8%  (n=67)  -> FAIL

RQ3 - NLP text vs image-only, paired per image (success: text reduces error):
  sizeprior     +3.1pp over 33 pairs, 11/33 better, sign p=0.080  -> FAIL

==============================================================================================================================
approach              arm                  n  V-MAPE%  V-bias%  M-MAPE%  lat_s    overhead        tilt   side_left  side_right
------------------------------------------------------------------------------------------------------------------------------
llm                   sizeprior_nlp       34     27.1     20.1     42.4   10.3           -           -           -           -
llm                   sizeprior_notext    33     24.5     14.6        -    9.0           -           -           -           -
==============================================================================================================================

======================================================
approach              tilt         n  V-MAPE%  V-bias%
------------------------------------------------------
llm                   mild        46     26.0     16.5
llm                   oblique     19     18.6     11.7
======================================================

```
