# Benchmark summary

Generated 2026-08-20 10:23 from `benchmark_simplefood45_llm_results_gemini3.1pro_text.csv` (72 scored results)

```
RQ2 - volume MAPE vs ground truth (success < 20%):
  llm                      18.7%  (n=72)  -> PASS

RQ3 - NLP text vs image-only, paired per image (success: text reduces error):
  sizeprior     -1.8pp over 36 pairs, 17/36 better, sign p=0.868  -> PASS

==============================================================================================================================
approach              arm                  n  V-MAPE%  V-bias%  M-MAPE%  lat_s    overhead        tilt   side_left  side_right
------------------------------------------------------------------------------------------------------------------------------
llm                   sizeprior_nlp       36     17.8     -2.7     21.2   16.6           -           -           -           -
llm                   sizeprior_notext    36     19.6     -2.5        -   16.1           -           -           -           -
==============================================================================================================================

======================================================
approach              tilt         n  V-MAPE%  V-bias%
------------------------------------------------------
llm                   mild        48     19.3     -9.0
llm                   oblique     22     16.7      8.9
======================================================

```
