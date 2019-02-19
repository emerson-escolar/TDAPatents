import numpy as np
import sklearn
import scipy
import pathlib
import argparse
import seaborn

import kmapper as km
import sklearn.manifold as skm
import sklearn.decomposition as skd

from patent_data import PatentData
from mapper_analyzer import MapperAnalyzer



def do_mapper(args, bigdata, verbosity):
    if args.accumulate:
        labels,data,cf,rgb_colors = bigdata.get_accumulated_data(args.from_year,args.to_year, do_log=args.log, do_transform=args.cos_trans, drop_zero=(not args.keep_zeros))
    else:
        labels,data,cf,years_data,rgb_colors = bigdata.get_merged_data(args.from_year, args.to_year, do_log=args.log, do_transform=args.cos_trans, drop_zero=(not args.keep_zeros))

    list_p_sizes = list(cf.flatten())

    if (data.shape[0] == 0):
        print("Warning: year {:d} to {:d} has no nonzero data! Skipping..".format(args.from_year,args.to_year))
        return

    if (data.shape[0] == 1):
        print("Warning: year {:d} to {:d} has only one nonzero firm data! Skipping..".format(args.from_year,args.to_year))
        return

    proc = MapperAnalyzer(data, cf,
                          labels=labels, lens= None, lens_name="pca2d", metric=args.metric,
                          verbosity=verbosity)
    mainfolder =  proc.get_main_folder()
    print(mainfolder)

    # hierarchical heatmap!
    output_fname = mainfolder.joinpath(proc.get_data_fullname()   + "_cluster.png")

    g = seaborn.clustermap(proc.data, metric=proc.metric, col_cluster=False, yticklabels=True, cmap="Reds", figsize=(20,40))
    g.savefig(str(output_fname), dpi=75)
    # end hierarchical heatmap!

    pca = skd.PCA(n_components=2)
    proc.lens = pca.fit_transform(data)



    def color_averager(list_of_triples):
        ans = np.mean(np.array(list_of_triples), axis=0)
        return "rgb({:d},{:d},{:d})".format(int(ans[0]),int(ans[1]),int(ans[2]))

    more_data = {'color': rgb_colors, 'p_sizes': list_p_sizes,
                 'ave_p_size': list_p_sizes, 'max_p_size': list_p_sizes}
    more_transforms = {'color': color_averager, 'ave_p_size' : np.mean, 'max_p_size' : np.max}

    if not args.accumulate:

        more_data['ave_year'] = years_data
        more_transforms['ave_year'] = np.mean

        more_data['unique_members'] = [x[:-6] for x in list(data.index)]
        more_transforms['unique_members'] = (lambda x:list(set(x)))

    for cub in args.numbers:
        for overlap in args.overlaps:
            proc.do_analysis(cub, overlap, more_data, more_transforms)




def main():
    parser = argparse.ArgumentParser(description="Program for performing Mapper analysis on patent data.")
    parser.add_argument("--verbose", "-v", action="store_true", help="verbose.")
    parser.add_argument("--log", "-l", action="store_true", help="do log.")

    parser.add_argument("--keep_zeros", "-z", action="store_true", help="preserve zeros columns in data. Do not use. Otherwise, drop zero columns.")
    parser.add_argument("--cos_trans", "-c", action="store_true", help="use cosine distances to transform data.")
    parser.add_argument("--mode", "-m", help="mode choice: 0 or 1 or 2 (default: 0; both modes: 2).", type=int, default=0, choices=[0,1,2])
    parser.add_argument("--metric", "-d", help="metric choice: 'euclidean' or 'correlation' (default: 'correlation').", type=str, default='correlation', choices=['euclidean', 'correlation'])

    parser.add_argument("--numbers", "-n", help="number(s) of cover elements in each axis.", type=int, nargs="+", default=[5,10,15,20])

    parser.add_argument("--overlaps", "-p", help="overlap(s) of cover elements. Express as decimal between 0 and 1.", type=float, nargs="+", default=[0.5])

    parser.add_argument("--from_year", "-f", help="starting year to do analysis.", type=int,default=1976)
    parser.add_argument("--to_year", "-g", help="ending year (inclusive) to do analysis.", type=int,default=2005)


    parser.add_argument("--accumulate", "-a", action="store_true", help="add all years data. Operation order: cos-dist-transform (if enabled), then accumulate, then log (if enabled)")


    args = parser.parse_args()
    # class_translator = "180901_csv/patent_classes.csv"
    class_translator = None

    if args.mode == 0:
        bigdata = PatentData(extra_data_desc="m0",
                             patent_folder_name="180901_csv/reshape_wide_byyear_mode0",
                             patent_fname_format="reshape_wide_year{:d}_mode0.csv",
                             cosdis_folder_name="180901_csv/cosine_distance_byyear",
                             cosdis_fname_format="cosine_distance_year{:d}.csv",
                             translator_fname="180901_csv/190204_firm_rank_name_industry.csv",
                             translator_func=(lambda x: ("firm_rank_{:s}").format(x)),
                             patent_class_translator_fname=class_translator)
    elif args.mode == 1:
        bigdata = PatentData(extra_data_desc="m1",
                             patent_folder_name="180901_csv/reshape_wide_byyear_mode1",
                             patent_fname_format="reshape_wide_year{:d}_mode1.csv",
                             cosdis_folder_name="180901_csv/cosine_distance_byyear",
                             cosdis_fname_format="cosine_distance_year{:d}.csv",
                             translator_fname="180901_csv/190204_firm_rank_name_industry.csv",
                             translator_func=(lambda x: ("firm_rank_{:s}").format(x)),
                             patent_class_translator_fname=class_translator)
    elif args.mode == 2:
        bigdata = PatentData(extra_data_desc="allm",
                             patent_folder_name="180901_csv/reshape_wide_byyear_allmodes",
                             patent_fname_format="reshape_wide_year{:d}_allmodes.csv",
                             cosdis_folder_name="180901_csv/cosine_distance_byyear",
                             cosdis_fname_format="cosine_distance_year{:d}.csv",
                             translator_fname="180901_csv/190204_firm_rank_name_industry.csv",
                             translator_func=(lambda x: ("firm_rank_{:s}").format(x)),
                             patent_class_translator_fname=class_translator)

    verbosity = 0
    if args.verbose:
        verbosity = 2

    do_mapper(args, bigdata, verbosity)







if __name__ == "__main__":
    main()
