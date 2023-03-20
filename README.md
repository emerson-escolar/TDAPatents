# TDAPatents

This repository contains code for the topological analysis of patent data, as described in the paper
"[Mapping Firms' Locations in Technological Space: A Topological Analysis of Patent Statistics](https://arxiv.org/abs/1909.00257)."


# Installation













# Usage - The main script

First, the script is in divided into three sub-commands, depending on how it is supposed to treat a time series data.

* accumulate

    Add everything together. 

* merge

    Take the data over different years and consider it as one dataset (each point is a firm-year).

* ma (merge-accumulate)

    Accumulate (add) data over a moving time window, then consider the entire panel dataset.
    **This is the method adopted in the paper**.

## Detailed options
### Data choice options
* --data {0,1}

  Choose data location: 0 (folder 180901_csv) or 1 (folder 200110_csv) (default: 1)
  
* --mode, -m {0,1,2} 

  Choose data type to read: 0 or 1 or 2 (both) (default: 0)

* --from_year, -f STARTING_YEAR
  
  starting year to do analysis. (default=1976)
  
* --to_year, -g ENDING_YEAR

  ending year (inclusive) to do analysis (default=2005)

### Processing options

* --keep_zeros, -z 

    By default, we drop zero columns in the data, as they cause problems in computations.
    With this option enabled, forcibly preserve zero columns in data. **Do not use**. Disabled by default.
    
*  --cos_trans, -c 

    Use cosine distances between patents classes to apply a pre-transformation to data. **Not relevant to the paper**.

* --transpose 

    Apply transpose as transformation on data. When enabled, consider patent classes instead of firms as entities/points.
    **Not relevant to the paper**.

* --log, -l

    Enable this to apply log transformation $x \mapsto \log(x) + 1$ on data.
        
* --sum_to_one
    
    Normalize data after other transformations to sum to one.

* --metric, -d METRIC

    Choose a metric. 'euclidean' or 'correlation' or 'cityblock' or 'cosine' or 'bloom' (Bloom et al.'s Mahalanobis normed tech closeness) (default: 'correlation')

### Mapper parameters
    
* --mds

    With this option enabled, use MDS instead of PCA  as filter function.
    
    
* --dimension DIM

    dimension for filter function (default: 2)

* --numbers, -n LIST_OF_NUMBERS 

    number(s) of cover elements in each axis (default=5,10,15,20)

* --overlaps, -p LIST_OF_NUMBERS

    overlap(s) of cover elements. Express as decimal between 0 and 1. (default=0.5)

* --clusterer

    clustering method. Choose one from 'HC_single', 'HC_complete', 'HC_average', 'HC_weighted', 'OPTICS'.
    (default='HC_single')
    
* --heuristic", 

    gap heuristic method, for hierarchical clustering (HC) type clustering methods only.
    (choose one from 'firstgap', 'midgap', 'lastgap', 'sil') (default='firstgap')

### output choices
    group_output = common_parser.add_argument_group("Output options")
    group_output.add_argument("--interactive", action="store_true", help="interactive plot of lens.")
    group_output.add_argument("--clustermap", action="store_true", help="Do clustermap.")
    group_output.add_argument("--no_dump_raw", action="store_true", help="Skip dumping raw data.")
    group_output.add_argument("--kclusters", help="number(s) of k-Medoids and k-Means to compute and append to cyjs output. Note that k-Means ignores --metric and always computes with Euclidean distances.", type=int, nargs="+", default=None)

    group_output.add_argument("--char_limit", help="limit chars for firms and patent class names", type=int, default=None)
    group_output.add_argument("--no_mapper", action="store_true", help="Skip Mapper computation entirely.")

    group_output.add_argument("--output", "-o", help="output base folder.", type=str)














