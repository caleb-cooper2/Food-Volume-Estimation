# Benchmark summary

Generated 2026-08-20 10:00 from `benchmark_simplefood45_llm_results_gemini2.5_text.csv` (84 scored results)

```
RQ2 - volume MAPE vs ground truth (success < 20%):
  llm                      26.9%  (n=84)  -> FAIL

RQ3 - NLP text vs image-only, paired per image (success: text reduces error):
  sizeprior     +2.3pp over 42 pairs, 14/42 better, sign p=0.044  -> FAIL

==============================================================================================================================
approach              arm                  n  V-MAPE%  V-bias%  M-MAPE%  lat_s    overhead        tilt   side_left  side_right
------------------------------------------------------------------------------------------------------------------------------
llm                   sizeprior_nlp       42     28.1     -5.3     45.1    3.1           -           -           -           -
llm                   sizeprior_notext    42     25.8     -4.7        -    3.0           -           -           -           -
==============================================================================================================================

======================================================
approach              tilt         n  V-MAPE%  V-bias%
------------------------------------------------------
llm                   mild        54     29.2     -2.3
llm                   oblique     24     17.1    -15.6
======================================================

```
