import numpy as np
import sklearn
import argparse
import sys

import kmapper as km
import sklearn.manifold as skm
import sklearn.decomposition as skd

import scipy.spatial.distance

from a_patent_data import PatentData
from a_analyzer import Analyzer
from a_utilities import color_averager, is_empty_data

import mappertools.outputs.text_dump as tdump


def get_common_parser():
    common_parser = argparse.ArgumentParser(add_help = False)
    common_parser.add_argument("--verbose", "-v", action="store_true", help="verbose.")

    # Data choice
    group_data_choice = common_parser.add_argument_group("Data choice")
    group_data_choice.add_argument("--data", help="data choice: 0 (folder 180901_csv) or 1 (folder 200110_csv) (default: 1).", type=int, default=1, choices=[0,1])
    group_data_choice.add_argument("--mode", "-m", help="mode choice: 0 or 1 or 2 (default: 0; both modes: 2).", type=int, default=0, choices=[0,1,2])

    group_data_choice.add_argument("--from_year", "-f", help="starting year to do analysis.", type=int,default=1976)
    group_data_choice.add_argument("--to_year", "-g", help="ending year (inclusive) to do analysis.", type=int,default=2005)

    # Processing
    group_processing = common_parser.add_argument_group("Processing")
    group_processing.add_argument("--keep_zeros", "-z", action="store_true", help="preserve zeros columns in data. Do not use. Otherwise, drop zero columns.")
    group_processing.add_argument("--cos_trans", "-c", action="store_true", help="use cosine distances to transform data.")
    group_processing.add_argument("--transpose", action="store_true", help="do transpose. When enabled, consider patent classes, instead of firms, as entities/points.")

    group_processing.add_argument("--log", "-l", action="store_true", help="do log.")
    group_processing.add_argument("--sum_to_one", action="store_true", help="normalize data, after other transformations, to sum to one.")

    group_processing.add_argument("--metric", "-d", help="metric choice: 'euclidean' or 'correlation' or 'cityblock' or 'cosine' or 'bloom' (Bloom et al.'s Mahalanobis normed tech closeness) (default: 'correlation').", type=str, default='correlation', choices=['euclidean', 'correlation', 'cityblock', 'cosine', 'bloom'])

    # Mapper parameters
    group_mapper_params = common_parser.add_argument_group("Mapper parameters")
    group_mapper_params.add_argument("--mds", help="use MDS instead, as filter function.", action="store_true")
    group_mapper_params.add_argument("--dimension", help="dimension for filter: positive integer (default: 2).", type=int, default=2)

    group_mapper_params.add_argument("--numbers", "-n", help="number(s) of cover elements in each axis.", type=int, nargs="+", default=[5,10,15,20])
    group_mapper_params.add_argument("--overlaps", "-p", help="overlap(s) of cover elements. Express as decimal between 0 and 1.", type=float, nargs="+", default=[0.5])

    group_mapper_params.add_argument("--clusterer", help="clustering method.", type=str, default='HC_single', choices=['HC_single', 'HC_complete', 'HC_average', 'HC_weighted', 'OPTICS'])
    group_mapper_params.add_argument("--heuristic", help="gap heuristic method, for hierarchical clustering (HC) type clustering methods only.", type=str, default='firstgap', choices=['firstgap', 'midgap', 'lastgap', 'sil'])

    # output choices
    group_output = common_parser.add_argument_group("Output options")
    group_output.add_argument("--interactive", action="store_true", help="interactive plot of lens.")
    group_output.add_argument("--clustermap", action="store_true", help="Do clustermap.")
    group_output.add_argument("--no_dump_raw", action="store_true", help="Skip dumping raw data.")
    group_output.add_argument("--kclusters", help="number(s) of k-Medoids and k-Means to compute and append to cyjs output. Note that k-Means ignores --metric and always computes with Euclidean distances.", type=int, nargs="+", default=None)

    group_output.add_argument("--char_limit", help="limit chars for firms and patent class names", type=int, default=None)
    group_output.add_argument("--no_mapper", action="store_true", help="Skip Mapper computation entirely.")

    group_output.add_argument("--output", "-o", help="output base folder.", type=str)

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
    common_parser_copy = get_common_parser()
    matching_groups = (g for g in common_parser_copy._action_groups if g.title == 'Data choice')
    group = next(matching_groups, None) or common_parser_copy
    group.add_argument("--window", "-w", type=int, help="window size (default=5)",default=5)
    group.add_argument("--shift", "-s", type=int, help="window shift (default=5)",default=5)

    merge_accum_parser = subparsers.add_parser("ma",
                                               help="accumulate over window, then merge across shifted windows.",
                                               parents=[common_parser_copy])
    merge_accum_parser.set_defaults(procedure="merge_accumulate")

    return parser



def get_labels_and_data(args):
    if args.data == 0:
        data_name = "D0"
        base_folder = "180901_csv/"
    elif args.data == 1:
        data_name = "D1"
        base_folder = "200110_csv/"
    else:
        sys.exit()

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

    if args.procedure == "accumulate":
        labels,data = bigdata.get_accumulated_data(args.from_year, args.to_year,
                                                   drop_zero=(not args.keep_zeros), do_transform=args.cos_trans, do_transpose=args.transpose,
                                                   do_log=args.log, sum_to_one=args.sum_to_one)
    elif args.procedure == "merge":
        labels,data = bigdata.get_merged_data(args.from_year, args.to_year,
                                              drop_zero=(not args.keep_zeros), do_transform=args.cos_trans, do_transpose=args.transpose,
                                              do_log=args.log, sum_to_one=args.sum_to_one)
    elif args.procedure == "merge_accumulate":
        labels,data = bigdata.get_merged_accumulated_data(args.from_year, args.to_year,
                                                          args.window, args.shift,
                                                          drop_zero=(not args.keep_zeros),do_transform=args.cos_trans,do_transpose=args.transpose,
                                                          do_log=args.log, sum_to_one=args.sum_to_one)

    return labels, data


def do_mapper(args, labels, data, verbosity):
    if is_empty_data(data, args.from_year, args.to_year): return

    if args.sum_to_one:
        summed = data.sum(axis=1)
        assert np.allclose(summed.loc[summed!=0],1)

    # prepare mapper data and lens
    if args.mds == True:
        lens_name = "mds{}d".format(args.dimension)
    else:
        lens_name = "pca{}d".format(args.dimension)

    proc = Analyzer(data, labels=labels,
                          lens= None, lens_name=lens_name, metric=args.metric,
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

    # initialize analyzer
    proc.initialize(base_path=args.output)

    # Other outputs
    if True: proc.plot_lens(np.array(labels.rgb_colors)/255., show=args.interactive)
    if args.clustermap: proc.do_clustermap()
    if not args.no_dump_raw: proc.dump_data()

    if args.kclusters:
        kClusters = proc.dataframe_kClusters(args.kclusters, dump_summary=True, dump_aggregates=True)

    # Early end
    if args.no_mapper:
        sys.exit()

    # Additional data
    list_p_sizes = list(labels.p_sizes.flatten())
    more_data = {'members': list(proc.data.index),
                 'color': labels.rgb_colors, 'p_sizes': list_p_sizes,
                 'ave_p_size': list_p_sizes, 'max_p_size': list_p_sizes,
                 'ave_lensX': proc.lens[:,0], 'ave_lensY': proc.lens[:,1]}
    more_transforms = {'color': color_averager,
                       'ave_p_size' : np.mean, 'max_p_size' : np.max,
                       'ave_lensX': np.mean, 'ave_lensY': np.mean}

    if labels.years_data is not None:
        more_data['ave_year'] = labels.years_data
        more_transforms['ave_year'] = np.mean

    firm_query_string = "members"
    if labels.intemporal_index is not None:
        more_data['unique_members'] = list(labels.intemporal_index)
        more_transforms['unique_members'] = (lambda x:list(set(x)))
        firm_query_string = "unique_members"

    if args.kclusters:
        for col in kClusters.columns:
            more_data[col] = kClusters[col].to_list()
    # end additional data

    # process clustering options
    clusterer_dict = {"clusterer_arg": args.clusterer, "clusterer_HC_heuristic" : args.heuristic}

    if args.clusterer[:3] == "HC_":
        clusterer_dict["clusterer_name"] = args.clusterer + "_" + args.heuristic
    else:
        clusterer_dict["clusterer_name"] = args.clusterer

    # end process clustering options


    # do mapper analysis
    for n_cubes in args.numbers:
        for overlap in args.overlaps:
            if overlap <= 0 or overlap >= 1:
                print("Overlap: {} invalid; skipping.".format(overlap),file=sys.stderr)
                continue
            graph = proc.compute_mapper_graph(n_cubes, overlap, clusterer_dict)

            nxgraph = tdump.kmapper_to_nxmapper(graph,
                                                more_data, more_data,
                                                more_transforms, more_transforms,
                                                counts=True, weights=True,
                                                cen_flares=False)

            output_folder = proc.get_output_folder(n_cubes, overlap, clusterer_dict["clusterer_name"])
            fullname = proc.get_fullname(n_cubes, overlap, clusterer_dict["clusterer_name"])

            nxgraph.graph["name"] = fullname

            # extract average node positions
            pos = []
            flip_y = True
            if flip_y:
                for i in nxgraph.nodes:
                    pos.append([nxgraph.nodes[i]["ave_lensX"], -nxgraph.nodes[i]["ave_lensY"]])
            else:
                for i in nxgraph.nodes:
                    pos.append([nxgraph.nodes[i]["ave_lensX"], nxgraph.nodes[i]["ave_lensY"]])

            # Output cyjs
            output_fname = output_folder.joinpath(fullname + ".cyjs")
            tdump.cytoscapejson_dump(nxgraph, output_fname, 80, np.array(pos))

            # Output flares & other stats for each firm.
            proc.do_derived_stats_csv(nxgraph, output_folder, fullname, firm_query_string)

            proc.do_mapper_stats_txt(nxgraph, output_folder, fullname, firm_query_string)



def main(raw_args):
    parser = get_parser()
    args = parser.parse_args(raw_args)

    if args.procedure == None:
        parser.print_help()
        sys.exit()

    labels, data = get_labels_and_data(args)
    do_mapper(args, labels, data, verbosity=(2 if args.verbose else 0))

if __name__ == "__main__":
    main(sys.argv[1:])
