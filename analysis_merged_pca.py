import numpy as np
import sklearn
import argparse
import sys

import kmapper as km
import sklearn.manifold as skm
import sklearn.decomposition as skd

import scipy.spatial.distance

from a_patent_data import PatentData
from a_mapper_analyzer import MapperAnalyzer
from a_utilities import color_averager, is_empty_data

import mappertools.text_dump as tdump


def get_common_parser():
    common_parser = argparse.ArgumentParser(add_help = False)
    common_parser.add_argument("--verbose", "-v", action="store_true", help="verbose.")

    common_parser.add_argument("--keep_zeros", "-z", action="store_true", help="preserve zeros columns in data. Do not use. Otherwise, drop zero columns.")

    common_parser.add_argument("--log", "-l", action="store_true", help="do log.")
    common_parser.add_argument("--cos_trans", "-c", action="store_true", help="use cosine distances to transform data.")
    common_parser.add_argument("--sum_to_one", action="store_true", help="normalize data, after other transformations, to sum to one.")

    common_parser.add_argument("--data", help="data choice: 0 (folder 180901_csv) or 1 (folder 200110_csv) (default: 1).", type=int, default=1, choices=[0,1])

    common_parser.add_argument("--mode", "-m", help="mode choice: 0 or 1 or 2 (default: 0; both modes: 2).", type=int, default=0, choices=[0,1,2])
    common_parser.add_argument("--metric", "-d", help="metric choice: 'euclidean' or 'correlation' or 'cityblock' or 'cosine' or 'bloom' (Bloom et al.'s Mahalanobis normed tech closeness) (default: 'correlation').", type=str, default='correlation', choices=['euclidean', 'correlation', 'cityblock', 'cosine', 'bloom'])

    common_parser.add_argument("--numbers", "-n", help="number(s) of cover elements in each axis.", type=int, nargs="+", default=[5,10,15,20])

    common_parser.add_argument("--overlaps", "-p", help="overlap(s) of cover elements. Express as decimal between 0 and 1.", type=float, nargs="+", default=[0.5])

    common_parser.add_argument("--heuristic", help="gap heuristic method.", type=str, default='firstgap', choices=['firstgap', 'midgap', 'lastgap', 'db', 'sil'])

    common_parser.add_argument("--from_year", "-f", help="starting year to do analysis.", type=int,default=1976)
    common_parser.add_argument("--to_year", "-g", help="ending year (inclusive) to do analysis.", type=int,default=2005)

    common_parser.add_argument("--mds", help="use MDS instead, as filter function.", action="store_true")

    common_parser.add_argument("--dimension", help="dimension for filter: positive integer (default: 2).", type=int, default=2)
    common_parser.add_argument("--interactive", action="store_true", help="interactive plot of lens.")


    common_parser.add_argument("--char_limit", help="limit number of characters to use for firms and patent classes", type=int, default=None)

    return common_parser


def get_parser():
    parser = argparse.ArgumentParser(description="Program for performing Mapper analysis on patent data.")
    parser.set_defaults(verbosity=0, procedure=None)
    subparsers = parser.add_subparsers(help="choose mode of operation:")
    common_parser = get_common_parser()

    # ACCUMULATE
    accum_parser = subparsers.add_parser("accumulate",
                                         help="accumulate all chosen years. Operation order: cos-dist-transform (if enabled), then accumulate, then log (if enabled)",
                                         parents=[common_parser])
    accum_parser.set_defaults(procedure="accumulate")

    # MERGE
    merge_parser = subparsers.add_parser("merge",
                                         help="merge all chosen years",
                                         parents=[common_parser])
    merge_parser.set_defaults(procedure="merge")

    # MERGE-ACCUMULATE
    merge_accum_parser = subparsers.add_parser("ma",
                                               help="accumulate over window, then merge across shifted windows.",
                                                    parents=[common_parser])
    merge_accum_parser.set_defaults(procedure="merge_accumulate")
    merge_accum_parser.add_argument("--window", "-w", type=int, help="window size (default=5)",default=5)
    merge_accum_parser.add_argument("--shift", "-s", type=int, help="window shift (default=5)",default=5)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.procedure == None:
        parser.print_help()
        exit()

    if args.data == 0:
        data_name = "D0"
        base_folder = "180901_csv/"
    elif args.data == 1:
        data_name = "D1"
        base_folder = "200110_csv/"
    else:
        exit()

    class_translator = base_folder + "patent_classes.csv"
    # class_translator = None


    if args.mode == 0:
        bigdata = PatentData(extra_data_desc=data_name+"m0",
                             patent_folder_name=base_folder + "reshape_wide_byyear_mode0",
                             patent_fname_format="reshape_wide_year{:d}_mode0.csv",
                             firm_translator_fname=base_folder + "firm_rank_name_industry.csv",
                             firm_translator_func=(lambda x: ("firm_rank_{:s}").format(x)),
                             class_translator_fname=class_translator,
                             char_limit=args.char_limit)
    elif args.mode == 1:
        bigdata = PatentData(extra_data_desc=data_name+"m1",
                             patent_folder_name=base_folder + "reshape_wide_byyear_mode1",
                             patent_fname_format="reshape_wide_year{:d}_mode1.csv",
                             firm_translator_fname=base_folder + "firm_rank_name_industry.csv",
                             firm_translator_func=(lambda x: ("firm_rank_{:s}").format(x)),
                             class_translator_fname=class_translator,
                             char_limit=args.char_limit)
    elif args.mode == 2:
        bigdata = PatentData(extra_data_desc=data_name+"allm",
                             patent_folder_name=base_folder + "reshape_wide_byyear_allmodes",
                             patent_fname_format="reshape_wide_year{:d}_allmodes.csv",
                             firm_translator_fname=base_folder + "firm_rank_name_industry.csv",
                             firm_translator_func=(lambda x: ("firm_rank_{:s}").format(x)),
                             class_translator_fname=class_translator,
                             char_limit=args.char_limit)

    bigdata.init_transform(cosdis_folder_name = (base_folder + "cosine_distance_byyear"),
                           cosdis_fname_format = "cosine_distance_year{:d}.csv")

    do_mapper(args, bigdata, verbosity=(2 if args.verbose else 0))




def do_mapper(args, bigdata, verbosity):
    years_data = None
    if args.procedure == "accumulate":
        labels,data,cf,rgb_colors = bigdata.get_accumulated_data(args.from_year, args.to_year,
                                                                 drop_zero=(not args.keep_zeros), do_transform=args.cos_trans,
                                                                 do_log=args.log, sum_to_one=args.sum_to_one)
    elif args.procedure == "merge":
        labels,data,cf,years_data,rgb_colors = bigdata.get_merged_data(args.from_year, args.to_year,
                                                                       drop_zero=(not args.keep_zeros), do_transform=args.cos_trans,
                                                                       do_log=args.log, sum_to_one=args.sum_to_one)
    elif args.procedure == "merge_accumulate":
        labels,data,cf,years_data,rgb_colors = bigdata.get_merged_accumulated_data(args.from_year, args.to_year,
                                                                                   args.window, args.shift,
                                                                                   drop_zero=(not args.keep_zeros), do_transform=args.cos_trans,
                                                                                   do_log=args.log, sum_to_one=args.sum_to_one)
    print(data.columns.values)
    if is_empty_data(data, args.from_year, args.to_year): return

    if args.sum_to_one:
        summed = data.sum(axis=1)
        assert np.allclose(summed.loc[summed!=0],1)

    firms = list(bigdata.firm_translator.values())
    # prepare mapper data and lens
    if args.mds == True:
        lens_name = "mds{}d".format(args.dimension)
    else:
        lens_name = "pca{}d".format(args.dimension)

    if years_data is not None:
        proc = MapperAnalyzer(data, firms, years_data,
                              labels=labels, lens= None, lens_name=lens_name,metric=args.metric,
                              verbose=verbosity)
    else:
        proc = MapperAnalyzer(data, firms, cf,
                              labels=labels, lens= None, lens_name=lens_name,metric=args.metric,
                              verbose=verbosity)

    if args.mds:
        if proc.metric == "precomputed":
            dists = proc.distance_matrix
        else:
            X = scipy.spatial.distance.pdist(data, metric=args.metric)
            dists = scipy.spatial.distance.squareform(X)

        proc.lens = skm.MDS(n_components=args.dimension, dissimilarity="precomputed").fit_transform(dists)
    else:
        proc.lens = skd.PCA(n_components=args.dimension).fit_transform(data)

    if True:
        proc.plot_lens(np.array(rgb_colors)/255., show=args.interactive)

    # do clustermap
    if False:
        proc.do_clustermap()

    # prepare additional data
    list_p_sizes = list(cf.flatten())
    more_data = {'members': list(proc.data.index),
                 'color': rgb_colors, 'p_sizes': list_p_sizes,
                 'ave_p_size': list_p_sizes, 'max_p_size': list_p_sizes}
    more_transforms = {'color': color_averager,
                       'ave_p_size' : np.mean, 'max_p_size' : np.max}

    query_data = 'members'
    if args.procedure == "merge" or args.procedure == "merge_accumulate":
        more_data['ave_year'] = years_data
        more_transforms['ave_year'] = np.mean
        more_data['unique_members'] = [x[:-6] for x in list(data.index)]
        more_transforms['unique_members'] = (lambda x:list(set(x)))

        query_data = 'unique_members'

    # do mapper analysis
    for n_cubes in args.numbers:
        for overlap in args.overlaps:
            if overlap <= 0 or overlap >= 1:
                print("Overlap: {} invalid; skipping.".format(overlap),file=sys.stderr)
                continue
            graph = proc.do_basic_analysis(n_cubes, overlap, args.heuristic)

            nxgraph = tdump.kmapper_to_nxmapper(graph,
                                                more_data, more_data,
                                                more_transforms, more_transforms,
                                                counts=True, weights=True,
                                                cen_flares=False)
            output_folder = proc.get_output_folder(n_cubes, overlap, args.heuristic)
            fullname = proc.get_fullname(n_cubes, overlap, args.heuristic)

            if False:
                nx.write_gpickle(nxgraph, output_folder.joinpath(fullname + ".gpickle"))

            proc.do_advanced_outputs(nxgraph, output_folder, fullname, query_data)






if __name__ == "__main__":
    main()
