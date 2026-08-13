# Benchmark summary

Generated 2026-08-13 15:07 from `benchmark_results_custom.csv` (660 scored results)

```
RQ2 - volume MAPE vs ground truth (success < 20%):
  deep-learning            56.0%  (n=120)  -> FAIL
  monocular-geometric      56.4%  (n=480)  -> FAIL
  multi-view              103.2%  (n=60)  -> FAIL

RQ3 - NLP text vs image-only, paired per image (success: text reduces error):
  utensil       +0.1pp over 120 pairs, 49/120 better, sign p=0.055  -> FAIL
  sizeprior     -0.4pp over 120 pairs, 58/120 better, sign p=0.784  -> PASS

======================================================================
approach                 n  V-MAPE%  V-bias%  M-MAPE%  R2(mass)  lat_s
----------------------------------------------------------------------
deep-learning          120     56.0     -5.1     56.0     -0.31    3.5
monocular-geometric    480     56.4     36.2     54.1     -2.57   15.8
multi-view              60    103.2     89.7    103.2    -15.16   24.3
======================================================================

==============================================================================================================================
approach              arm                  n  V-MAPE%  V-bias%  M-MAPE%  lat_s    overhead        tilt   side_left  side_right
------------------------------------------------------------------------------------------------------------------------------
deep-learning         none               120     56.0     -5.1        -    3.5        61.5        60.4        54.6        47.6
monocular-geometric   sizeprior_nlp      120     49.1     13.3     40.3   19.9        51.2        41.4        48.5        55.4
monocular-geometric   sizeprior_notext   120     49.5     13.5        -   13.0        52.2        42.0        48.4        55.4
monocular-geometric   utensil_nlp        120     63.5     59.0     63.4   17.3        56.7        54.5        63.8        79.1
monocular-geometric   utensil_notext     120     63.4     58.8        -   12.9        56.2        54.5        64.0        79.0
multi-view            sizeprior_notext    30     43.1     16.9        -   24.4           -           -           -           -
multi-view            utensil_notext      30    163.3    162.5        -   24.2           -           -           -           -
==============================================================================================================================

```
