# TDAPatents

This repository contains code for the topological analysis of patent data, as described in the paper
"[Mapping Firms' Locations in Technological Space: A Topological Analysis of Patent Statistics](https://arxiv.org/abs/1909.00257)" [[EHIO]](#EHIO).


# Setup

Two options:

1. Install the scripts in your local python environment (or virtual environment). For this, installation of additional python modules is needed. 

   1. (Optional, but recommended) Set up a virtual environment.
   
   2. Download contents of the repository.
   
       * If you cloned the repository, do not forget to do `git submodule init` and `git submodule update` to get the submodules.
   
   3. Run:
   
       ```
       pip install -r requirements.txt
       ```
       
       to install required additional python modules.

    In this case, the main script is called by running  `python ./tdapatents/tdapatents_script.py` in the folder where the data is contained.
    If the data and script files are place in different folders, change `./tdapatents/tdapatents_script.py` appropriately to point to the script file.

2. Use the single-file executable which contains all the needed python modules and scripts. This executable was created using [pyinstaller](https://pyinstaller.org/en/stable/).

    In this case, the main script is called by running  `tdapatents_script` in the folder where the data is contained.



# Usage - The main script

As noted [above](#Setup), the main script is called by either running 

1. `python tdapatents_script.py` or

2. `tdapatents_script`

Next, the main script contains three sub-commands, 
depending on how it is supposed to treat the time series patent data.
For more details on the data, please consult the paper [[EHIO]](EHIO).

* accumulate: Add the patent counts over all time slices (years). Each firm is represented by a point (vector).

* merge: Take the data over different years and consider it as one dataset. 
Each combination of a firm & year gives a point.

* ma: "merge-accumulate". Accumulate (add) data over a moving time window, then consider the entire panel dataset. Each combination of a firm  & year gives a point, but the data for each firm-year is accumulated from the time window. **This is the method adopted in the paper**.
    
For example, commands used for the paper starts with 
`python tdapatents_script.py ma` or 
`tdapatents_script ma`, depending on your [setup](#Setup).
In fact, running the previous command should already produce some output in a folder called
`cos_pca2d_logmerg` corresponding to the topological analysis with the default settings.
(See [here](#Detailed-options) for information about the settings one can tweak for the analysis.)

To see help for the many options to produce the analysis, input:
`python tdapatents_script.py ma --help` or 
`tdapatents_script ma --help`

## Replication
For the code samples below, replace `SCRIPT` by either 
`python tdapatents_script.py` or
`tdapatents_script`
depending on your [setup](#Setup)

1. For the "main figure":
    ```
    SCRIPT ma -l -d cosine -w 5 -s 1 -n 20
    ```
    This should create a folder `cos_pca2d_logmerg\D1m0` which contains the output.
    Inside that folder, one finds the file `**_pca2d.png` which contains the 2d pca dimension reduction result, 
    and folder `n20_o0.5_HC_single_firstgap` containing the Mapper results.
    Inside the folder are:
    
    * a html file, containing an interactive visualization of the Mapper graph
    
    * a cyjs file, containing the Mapper graph, for use in the network visualization software [Cytoscape](https://cytoscape.org/)
    
    * a text file `**_mapper_stats.txt` containing some summary statistics of the Mapper graph. This includes the number of nodes and edges, connected components, degree distribution, and "how many firms are included in only 1 node, 2 nodes, etc".  Note that there are firms contained in 0 nodes. This is because of two factors: (a) the list of firms is based on "firm\_rank\_name\_industry.csv" that appears to contain firms not in the actual data set and (b) in the actual data set there are firms with all-zero data.
    
    * a csv file `**_derived_stats.csv` containing statistics of firms derived from their locations in the Mapper graph. 
    
    Note: the file and folder names of outputs describe the options used for their computation.
    In this case, we used "cosine distance", "pca" for the filter function, and "log" preprocessing, under merge-accumulate mode, so the base folder is `cos_pca2d_logmerg`. Next, we are using data set 1 and mode 0, giving `D1m0`. For the mapper results, we are using n = 20, overlap 50%, hierarchical clustering (HC) with single linkage rule and firstgap heuristic, giving the folder name `n20_o0.5_HC_single_firstgap`. Using different options will place outputs in appropriately-named folders.
    
2. For sensitivity to different options

    * Different numbers of cover elements:
    ```
    SCRIPT ma -l -d cosine -w 5 -s 1 -n 15 25
    ```
    
    * Different overlap percentages (30% and 70%):
    ```
    SCRIPT ma -l -d cosine -w 5 -s 1 -n 20 -p 0.3 0.7
    ```
    
    * Example with 3D-PCA as filter function:
    ```
    SCRIPT ma -l -d cosine -w 5 -s 1 -n 20 --dimension 3
    ```
    
    * Example with 2D-MDS as filter function:
    ```
    SCRIPT ma -l -d cosine -w 5 -s 1 -n 20 --mds
    ```
    
    * Example changing clustering method used for Mapper
    ```
    SCRIPT ma -l -d cosine -w 5 -s 1 -n 20 --clusterer HC_average
    ```
    
    * Example changing distance used for Mapper
    ```
    SCRIPT ma -l -d euclidean -w 5 -s 1 -n 20 
    ```
    
3. R&D with M&A Patents

    ```
    SCRIPT ma -l -d cosine -w 5 -s 1 -n 20 -m 2
    ```
    
4. With global clustering 

    ```
    SCRIPT ma -l -d cosine -w 5 -s 1 -n 20 -kclusters 21
    ```
    



    
## Details on the Mapper output - html version

At the top of the page, there is a "COLOR FUNCTION" dropdown. 
Select the color function to be used:  "years" or "total patent size" or "sector" information.
See the file `firm_rank_name_industry.csv` in the data to find the sector labels.

Click on "[+] CLUSTER DETAILS" to show details about each Mapper node (= a cluster). 
There is a "MEMBER DISTRIBUTION" histogram that tells us roughly the histogram of colorings of its members.

With the current colormap we are using, it goes:

    blue (low) - green, yellow (middle), orange  - red (high)

So for example with a sector dummy coloring, 
nodes containing mostly firms in that sector should show up red,
while nodes containing no firms in that sector show up blue.

## Details on the Mapper output - cyjs version

This file is in a format for use with the software [Cytoscape](https://cytoscape.org/).
Also contains the following data:
for each Mapper nodes, the average lens position of its members.
One can use this information to fix the locations of the mapper nodes, 
and make it comparable to the filter function (for example 2D PCA).

    
## Details on "derived_stats"

This file contains firm measures derived from Mapper output.
Currently, this includes information about flares as defined in the paper [[EHIO]](#EHIO) and 
various centrality measures (degree centrality, harmonic centrality, and closeness centrality), 
with choice of processing (aggregation) to go from mapper nodes to firm-years to firms: 
(as a list of centralities of the nodes containing the firm-years of a firm, mean of that list, min of that list, max of that list)
In more detail, for each Mapper node $v$, let $C(v)$ be its centrality measure 
(degree centrality, harmonic centrality, or closeness centrality). 
For each firm $i$, list above is simply the list 
$$
    C(i) = \left[ C(v) \mid i \in v \right].
$$
   

## Detailed options
### Data choice options
* --data {0,1}

  Choose data location: 0 (folder 180901_csv) or 1 (folder 200110_csv) (default: 1)
  
* --mode, -m {0,1,2} 

  Choose data type to read: 0 (R&D patents only) or 1 (M&A patents only) or 2 (both) (default: 0)

* --from_year, -f STARTING_YEAR
  
  starting year to do analysis. (default=1976)
  
* --to_year, -g ENDING_YEAR

  ending year (inclusive) to do analysis (default=2005)
  
### Additional data choice options for Merge-Accumulate (ma)

* --window, -w WINDOW_SIZE

    window size (default=5)
    
* --shift, -s, WINDOW_SHIFT

    window shift/step (default=5)

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
    
    The 'HC_' type arguments means to use Hierarchical Clustering, 
    with the following part ('single', 'complete', 'average', or 'weighted') being the choice of method for determining linkages. 
    This is passed onto the option `method` in [scipy.cluster.hierarchy.linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html). 
    See the link [scipy.cluster.hierarchy.linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) for more information on the different methods.
    Note that [scipy.cluster.hierarchy.linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) also supports the methods ‘centroid’, ‘median’, and ‘ward’, but these are only correctly defined for Euclidean metric, so I did not include them.
    
* --heuristic HEURISTIC

    Choose a gap heuristic method, for hierarchical clustering (HC) type clustering methods only.
    (choose one from 'firstgap', 'midgap', 'lastgap', 'sil') (default='firstgap')
    
    Recall that hierarchical clustering returns a dendrogram (tree). 
    We need to choose a level where to cut the tree to get the number of clusters. 
    The "first gap heuristic" (`--heuristic firstgap`) 
    is the method introduced in the original Mapper paper [[SMC]](#SMC).) 
    Other options here: 'midgap', 'lastgap', 'sil'. 
    The 'gap' based ones are obvious modifications of the firstgap heuristic. 
    'sil' means to use the silhouette score (can be quite slow).

### output choices

* --interactive

    If enabled (and a compatible matplotlib backend is available), output an interactive plot of filter function.
        
* --no_dump_raw
  
    By default, the script will dump processed raw data (data sent to Mapper) in Apache Parquet format, if the dump does not exist yet. Disable this using this option. Note: "processed raw data" means data after all data choices (start and end year, window, shift) and processing (keep or drop zeros, cosine transform, transpose, log, sum-to-one), but before Mapper analysis. So it is actually not entirely raw.
    
* --clustermap

    If enabled, do [clustermap](https://seaborn.pydata.org/generated/seaborn.clustermap.html) on the 
    **entire** processed raw data set. This is **not** directly related to the clustering used to compute Mapper, which performs clustering only locally.
    
* --kclusters k1 k2 ...

    Perform both k-medoids and k-means clustering on the processed raw data with number of clusters k=k1, k=k2, and so on (independently of each other). The cluster information will be added into the cytoscape output and also to csv files. This computation does not use any information from Mapper. k-Means always ignores --metric and always computes with Euclidean distances. The implementation contains some randomness, so results may vary per run! (default=None)
        

* --char_limit LIMIT

    Limit names for firms and patent classes to LIMIT characters (roughly). 
    This may be useful for "transpose" analysis, as patent class names are very long.
    (default=None)
    
* --no_mapper

    If enabled, skip Mapper computation entirely.

* --output, -o FOLDER

    Set the base folder for outputs as FOLDER



# References
<a id="EHIO">[EHIO]</a> 
Escolar, E. G., Hiraoka, Y., Igami, M., & Ozcan, Y. (2019). Mapping firms' locations in technological space: A topological analysis of patent statistics. arXiv preprint arXiv:1909.00257.

<a id="CYTOSCAPE">[CYTOSCAPE]</a>
Shannon, P., Markiel, A., Ozier, O., Baliga, N. S., Wang, J. T., Ramage, D., Amin, N., Schwikowski, B., Ideker, T. (2003). Cytoscape: a software environment for integrated models of biomolecular interaction networks. Genome research, 13(11), 2498-2504.

<a id="SMC">[SMC]</a>
Singh, G., Mémoli, F., & Carlsson, G. E. (2007). Topological methods for the analysis of high dimensional data sets and 3d object recognition. PBG@ Eurographics, 2, 091-100.


