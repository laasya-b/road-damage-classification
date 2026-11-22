# P1: data and evaluation

Data: RDD2022 (Japan, India, Norway), train folders only (the official test images have no labels).
Patches: one square crop per box (10% margin, resized to 224x224) for classes D00, D10, D20, D40.
Split: by source photo, about 70/15/15 within each country, seed 0. Training capped at 1,500 patches per class; dev and test are not capped.
The patch images are on Google Drive (too large for GitHub); samples/ has 3 example patches per class.

Scripts: count_labels.py, make_patches.py, make_splits.py (run in that order).
Files: boxes.csv (every labeled box), meta.csv (one row per patch), splits.csv (split column), splits_counts_*.csv.

## All labels found in the XML files (boxes, before filtering)
~~~
country  India  Japan  Norway    All
label                               
D00       1555   4049    8570  14174
D01        179      0       0    179
D0w0         1      0       0      1
D10         68   3979    1730   5777
D11         45      0       0     45
D20       2021   6199     468   8688
D40       3187   2243     461   5891
D43         57    736       0    793
D44       1062   3995       0   5057
D50         28   3553       0   3581
All       8203  24754   11229  44186
~~~
Only D00, D10, D20, D40 are used. D01, D0w0, D11, D43, D44, D50 are excluded.

## Patches per class and split
~~~
split   dev  test  train
label                   
D00    2148  2041   1500
D10     900   826   1500
D20    1292  1279   1500
D40     821   804   1500
~~~

## Patches per class, split, and country
~~~
split     dev               test              train             
country India Japan Norway India Japan Norway India Japan Norway
label                                                           
D00       230   590   1328   228   619   1194   154   416    930
D10        15   612    273    13   559    254    20  1030    450
D20       335   893     64   284   916     79   332  1091     77
D40       393   367     61   405   330     69   840   538    122
~~~
