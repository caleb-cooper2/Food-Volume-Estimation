# Benchmark summary

Generated 2026-09-25 10:49 from `benchmark_results_custom_v3.csv` (1364 scored results)

```
RQ2 - volume MAPE vs ground truth (success < 20%):
  deep-learning            55.9%  (n=248)  -> FAIL
  monocular-geometric      96.6%  (n=992)  -> FAIL
  multi-view              104.1%  (n=124)  -> FAIL

RQ3 - NLP text vs image-only, paired per image (success: text reduces error):
  utensil       -5.6pp over 248 pairs, 98/248 better, sign p=0.001  -> PASS
  sizeprior     -3.3pp over 248 pairs, 102/248 better, sign p=0.006  -> PASS

======================================================================
approach                 n  V-MAPE%  V-bias%  M-MAPE%  R2(mass)  lat_s
----------------------------------------------------------------------
deep-learning          248     55.9     -8.9     55.6     -0.35    2.1
monocular-geometric    992     96.6     56.9     92.6     -1.78    2.2
multi-view             124    104.1     87.3    106.0     -6.71   14.6
======================================================================

==============================================================================================================================
approach              arm                  n  V-MAPE%  V-bias%  M-MAPE%  lat_s    overhead        tilt   side_left  side_right
------------------------------------------------------------------------------------------------------------------------------
deep-learning         none               248     55.9     -8.9        -    2.1        58.4        56.5        54.7        53.8
monocular-geometric   sizeprior_nlp      248     80.1     21.2     69.5    2.3        72.3        79.9        85.5        82.6
monocular-geometric   sizeprior_notext   248     83.4     26.1        -    2.2        76.6        79.4        85.6        91.9
monocular-geometric   utensil_nlp        248    108.7     86.3    103.6    2.5        85.3       128.9       114.9       105.9
monocular-geometric   utensil_notext     248    114.3     94.1        -    1.7        93.3       129.8       114.8       119.5
multi-view            sizeprior_notext    62     41.9      8.9        -   14.8           -           -           -           -
multi-view            utensil_notext      62    166.3    165.8        -   14.5           -           -           -           -
==============================================================================================================================

```
