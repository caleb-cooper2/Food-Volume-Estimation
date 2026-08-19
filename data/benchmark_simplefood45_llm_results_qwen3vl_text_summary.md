# Benchmark summary

Generated 2026-08-20 10:13 from `benchmark_simplefood45_llm_results_qwen3vl_text.csv` (84 scored results)

```
RQ2 - volume MAPE vs ground truth (success < 20%):
  llm                      49.4%  (n=84)  -> FAIL

RQ3 - NLP text vs image-only, paired per image (success: text reduces error):
  sizeprior    -13.4pp over 42 pairs, 17/42 better, sign p=0.280  -> PASS

==============================================================================================================================
approach              arm                  n  V-MAPE%  V-bias%  M-MAPE%  lat_s    overhead        tilt   side_left  side_right
------------------------------------------------------------------------------------------------------------------------------
llm                   sizeprior_nlp       42     42.8     -5.5     60.1    6.7           -           -           -           -
llm                   sizeprior_notext    42     56.1      6.5        -    5.6           -           -           -           -
==============================================================================================================================

======================================================
approach              tilt         n  V-MAPE%  V-bias%
------------------------------------------------------
llm                   mild        54     49.5      5.5
llm                   oblique     24     30.3    -29.6
======================================================

```
