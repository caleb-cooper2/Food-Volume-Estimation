# Benchmark summary

Generated 2026-09-15 13:58 from `benchmark_results_custom_v2.csv` (1364 scored results)

```
RQ2 - volume MAPE vs ground truth (success < 20%):
  deep-learning            56.5%  (n=248)  -> FAIL
  monocular-geometric     141.8%  (n=992)  -> FAIL
  multi-view              105.6%  (n=124)  -> FAIL

RQ3 - NLP text vs image-only, paired per image (success: text reduces error):
  utensil     -150.7pp over 248 pairs, 120/248 better, sign p=0.657  -> PASS
  sizeprior     -1.7pp over 248 pairs, 114/248 better, sign p=0.228  -> PASS

======================================================================
approach                 n  V-MAPE%  V-bias%  M-MAPE%  R2(mass)  lat_s
----------------------------------------------------------------------
deep-learning          248     56.5     -8.1     56.2     -0.34    3.7
monocular-geometric    992    141.8    101.3    141.4    -35.39   11.9
multi-view             124    105.6     76.7    106.6     -6.52   22.2
======================================================================

==============================================================================================================================
approach              arm                  n  V-MAPE%  V-bias%  M-MAPE%  lat_s    overhead        tilt   side_left  side_right
------------------------------------------------------------------------------------------------------------------------------
deep-learning         none               248     56.5     -8.1        -    3.7        59.6        57.9        54.8        53.5
monocular-geometric   sizeprior_nlp      248     84.1     25.5     73.2   16.2        77.6        81.6        85.4        91.9
monocular-geometric   sizeprior_notext   248     85.9     23.3        -    9.1        80.3        83.2        86.9        93.1
monocular-geometric   utensil_nlp        248    123.3    102.3    119.7   13.9       129.4       129.1       115.3       119.3
monocular-geometric   utensil_notext     248    274.0    254.1        -    8.6       168.3       589.4       167.4       170.8
multi-view            sizeprior_notext    62     47.4      3.0        -   22.6           -           -           -           -
multi-view            utensil_notext      62    163.7    150.4        -   21.8           -           -           -           -
==============================================================================================================================

```
