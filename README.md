# TDAPatents

This repository contains code for the topological analysis of patent data, as described in the paper
"[Mapping Firms' Locations in Technological Space: A Topological Analysis of Patent Statistics](https://arxiv.org/abs/1909.00257)."


# Installation













# Usage - The main script

First, the script is in divided into three sub-commands, depending on how it is supposed to treat a time series data.

* accumulate: Add everything together. 

* merge: Take the data over different years and consider it as one dataset (each point is a firm-year).

* ma: "merge-accumulate". Accumulate (add) data over a moving time window, then consider the entire panel dataset.
    **This is the method adopted in the paper**.
    
Thus, commands used for the paper starts with
`python analysis_merged_pca.py ma`.
In fact, running the previous command should already produce some output in a folder called
`cos_pca2d_logmerg` corresponding to the topological analysis with the default settings.
(See [here](#Detailed-options) for information about the settings one can tweak for the analysis.)

To see help for the many options to produce the analysis, input:
`python analysis_merged_pca.py ma --help`

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
    The effect is to look at each patent class as different and evolving over the years. Enabling this changes the interpretation of the results explained in this document, as the basic entities (points) being considered are now 
    class(-years), instead of firm(-years).
    **Not relevant to the paper**.

* --log, -l

    Enable this to apply log transformation $x \mapsto \log(x) + 1$ on data.
        
* --sum_to_one
    
    Normalize data after other transformations to sum to one.

* --metric, -d METRIC

    Choose a metric. 'euclidean' or 'correlation' or 'cityblock' or 'cosine' or 'bloom' (default: 'correlation')
    
    Notes: Setting the option `--metric bloom` will use Bloom et al.'s Mahalanobis normed technological closeness, transformed to a dissimilarity measure. However, using `--sum_to_one --metric bloom` more closely corresponds to the use in their paper, as they define this closeness measure on "patent shares".
    
    Using `--sum_to_one --metric cosine` gives Jaffe's measure of closeness (normalized uncentered covariance) transformed to a dissimilarity measure. 
    
### Mapper parameters
    
* --mds

    With this option enabled, use MDS instead of PCA  as filter (lens) function.
    
    
* --dimension DIM

    dimension for filter function (default: 2)

* --numbers, -n LIST_OF_NUMBERS 

    number(s) of cover elements in each axis (default=5,10,15,20)

* --overlaps, -p LIST_OF_NUMBERS

    overlap(s) of cover elements. Express as decimal between 0 and 1. (default=0.5)

* --clusterer CLUSTERING_METHOD

    clustering method. Choose one from 'HC_single', 'HC_complete', 'HC_average', 'HC_weighted', 'OPTICS'.
    (default='HC_single')
    
* --heuristic HEURISTIC

    Choose a gap heuristic method, for hierarchical clustering (HC) type clustering methods only.
    (choose one from 'firstgap', 'midgap', 'lastgap', 'sil') (default='firstgap')

### output choices

* --interactive

    If enabled (and a compatible matplotlib backend is available), output an interactive plot of filter function.
    
* --clustermap

    If enabled, do clustermap.
        
* --no_dump_raw
  
    By default, the script will dump processed raw data (data sent to Mapper) in Apache Parquet format, if the dump does not exist yet. Disable this using this option. Note: "processed raw data" means data after all data choices (start and end year, window, shift) and processing (keep or drop zeros, cosine transform, transpose, log, sum-to-one), but before Mapper analysis. So it is actually not entirely raw.
    
* --kclusters k1 k2 ...

    Perform both k-medoids and k-means clustering on the processed raw data with number of clusters k=k1, k=k2, and so on (independently of each other). The cluster information will be added into the cytoscape output. Note that this computation does not use any information from Mapper. k-Means always ignores --metric and always computes with Euclidean distances. (default=None)

* --char_limit LIMIT

    Limit names for firms and patent classes to LIMIT characters (roughly). 
    This may be useful for "transpose" analysis, as patent class names are very long.
    (default=None)
    
* --no_mapper

    If enabled, skip Mapper computation entirely.

* --output, -o FOLDER

    Set the base folder for outputs as FOLDER














