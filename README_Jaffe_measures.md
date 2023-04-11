# "Jaffe measures"
The script `./tdapatents/b_jaffe_measures.py` is for computing the "Jaffe measures", as described in Appendix G of the paper [EHIO]](#EHIO).
This script is structured similary as `./tdapatents/tdapatents_script.py` 
except that it fixes several parameters that can be modified in `./tdapatents/tdapatents_script.py`

* transpose = False
* log = False
* sum_to_one = True
* metric = "cosine"

in order to get the correct Jaffe measure (cosine distance in percentage-share version of data) for firms.
With such choices, the output will go into (a subfolder of) `cos_pca2d_sumonemerg` (this is shared with the results for `./tdapatents/tdapatents_script.py`). 

Side-note: the original Jaffe paper considers cosine **proximity** (cosine, as-is). Here, we are computing cosine **dissimilarity** = 1 - proximity.

Usage example:
``` 
python b_jaffe_measures.py ma --data 1 --mode 0 --from_year 1976 --to_year 2005 -o outputfolder -w 5 -s 1
```
for a 5 year moving window shifted 1 year at a time.
Note: `-o` is also an option in `./tdapatents/tdapatents_script.py` to specify the main folder into which to output the results. This may help in organizing the output results.

## Main output

`./tdapatents/b_jaffe_measures.py` does not perform any Mapper analysis. Instead, the main output is a file `..._jaffemeasures.csv` which summarizes the "Jaffe measures" of firms.

This output is a csv with the following columns in addition to the column of firms:
"yfinal--yfinal_neighbor", "yfinal--yfinal_neighborname", "yfinal--yfinal_popmean",
"ymean--ymean_neighbor", "ymean--ymean_neighborname", "ymean--ymean_popmean"

A rough explanation of the codes in the column naming is the following:

1. Data specifiers:
    * yfinal = data in "final" year. For example, with 5 year moving window, the "final year" would be (the sum of) years 2001~2005, which is labeled as year 2001.

    * ymean = data taken as mean location of each firm. Again, with a 5 year moving window, this would be the mean location of the 5 year moving sums.

2. Comparison target specifiers:
    * neighbor = nearest neighbor distance in Jaffe distance; 
    
      neighborname = actual firm name of that nearest neighbor

    * popmean = population mean (Jaffe distance to)
    

In detail, we explain each column below.

1. "yfinal--yfinal_neighbor", and "yfinal--yfinal_neighborname"

   This column contains the Jaffe measure (and name of nearest neighbor)
   between each firm in the final year and its nearest neighbor in that same year.

2. "yfinal--yfinal_popmean",

    This column contains the Jaffe measure 
    between each firm and the population mean, where each firm is represented by its final year (taking into account moving window)

3. "ymean--ymean_neighbor", and "ymean--ymean_neighborname", 

    This column contains the Jaffe measure (and name of nearest neighbor)
    between each firm and its nearest neighbor, where each firm is represented by its mean (of sums in moving windows) over the years under consideration.

4. "ymean--ymean_popmean"

    This column contains the Jaffe measure 
    between each firm and the population mean, where each firm is represented by its mean
    (of sums in moving windows) over the years under consideration.
    
    (Thus the population mean is computed as a double mean: first to get the representation of each firm, then to get the mean of all firms)
    
    
Note: in this analysis, each firm represented its percentage-share version (sums-to-one) and so taking means (year mean and/or popmean) is always with respect to percentage-share. Note that Jaffe's paper also starts with percentage-share version of the data, so this is not a point of discrepancy.

This is different from first taking the raw data mean location and then normalizing to percentage-share.

## Additional outputs

* `..._firmmeans.csv`: data computed for "ymean" = data taken as mean location of each firm.

* `..._finalyeardistances.csv`: Jaffe measures for firms, using the final year data.

* `..._firmmeansdistances.csv`: Jaffe measures for firms, using the "ymean" data.

# References
<a id="EHIO">[EHIO]</a> 
Escolar, E. G., Hiraoka, Y., Igami, M., & Ozcan, Y. (2019). Mapping firms' locations in technological space: A topological analysis of patent statistics. arXiv preprint arXiv:1909.00257.
