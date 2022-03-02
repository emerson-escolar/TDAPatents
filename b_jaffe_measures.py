import numpy as np
import pandas as pd
import sklearn
import argparse
import sys

import kmapper as km
import sklearn.manifold as skm
import sklearn.decomposition as skd
import sklearn.neighbors as skn

import scipy.spatial.distance as ssd

from a_patent_data import PatentData
from a_analyzer import Analyzer
from a_utilities import color_averager, is_empty_data

import mappertools.outputs.text_dump as tdump

from analysis_merged_pca import get_labels_and_data

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

    # output choices
    group_output = common_parser.add_argument_group("Output options")
    group_output.add_argument("--clustermap", action="store_true", help="Do clustermap.")
    group_output.add_argument("--no_dump_raw", action="store_true", help="Skip dumping raw data.")
    # group_output.add_argument("--interactive", action="store_true", help="interactive plot of lens.")
    group_output.add_argument("--kclusters", help="number(s) of k-Medoids and k-Means to compute and append to cyjs output. Note that k-Means ignores --metric and always computes with Euclidean distances.", type=int, nargs="+", default=None)

    group_output.add_argument("--char_limit", help="limit chars for firms and patent class names", type=int, default=None)
    # group_output.add_argument("--no_mapper", action="store_true", help="Skip Mapper computation entirely.")
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



def do_final_year_analysis(proc, data):
    ans_columns = ["yfinal--yfinal_neighbor", "yfinal--yfinal_neighborname", "yfinal--yfinal_popmean"]
    ans = pd.DataFrame(index=proc.labels.unique_members, columns=ans_columns)

    # extract final year
    final_year = max(proc.labels.years_data)
    final_year_bool = (proc.labels.years_data == final_year)
    final_year_data = data[final_year_bool]
    final_year_firms = proc.labels.intemporal_index[final_year_bool]

    # final year distances
    final_year_distances = ssd.squareform(ssd.pdist(final_year_data,metric="cosine"))

    # output distances
    main_folder = proc.get_main_folder()
    name = "{:s}_{:s}_finalyeardistances.csv".format(proc.labels.transforms_name, proc.labels.data_name)
    output_fname = main_folder.joinpath(name)
    pd.DataFrame(final_year_distances, index=final_year_firms, columns=final_year_firms).to_csv(str(output_fname))

    # final year pop mean
    final_year_popmean = np.mean(final_year_data, axis=0)

    # final year nn
    nbrs = skn.NearestNeighbors(n_neighbors=2, algorithm='auto', metric="precomputed").fit(final_year_distances)
    final_year_nn_distances, final_year_nn_indices = nbrs.kneighbors(n_neighbors=1)

    for i, firm in enumerate(final_year_firms):
        ans.loc[firm, "yfinal--yfinal_neighbor"] = final_year_nn_distances[i][0]
        ans.loc[firm, "yfinal--yfinal_neighborname"] = final_year_firms[final_year_nn_indices[i][0]]

        ans.loc[firm, "yfinal--yfinal_popmean"] = ssd.cosine(final_year_data.iloc[i,:], final_year_popmean)

    return ans


def do_year_mean_analysis(proc, data):
    ans_columns = ["ymean--ymean_neighbor", "ymean--ymean_neighborname", "ymean--ymean_popmean"]
    ans = pd.DataFrame(index=proc.labels.unique_members, columns=ans_columns)

    # create means for each firm
    firm_mean_data = pd.DataFrame(columns = data.columns)
    for firm in proc.labels.unique_members:
        firm_bool = proc.labels.intemporal_index == firm
        firm_data = np.mean(data[firm_bool])
        firm_mean_data.loc[firm] = firm_data
    # remove all-zero rows (firms):
    firm_mean_data.dropna(axis=0, how="all", inplace=True)

    # output year means
    main_folder = proc.get_main_folder()
    name = "{:s}_{:s}_firmmeans.csv".format(proc.labels.transforms_name, proc.labels.data_name)
    output_fname = main_folder.joinpath(name)
    firm_mean_data.to_csv(str(output_fname))

    # yearmean distances
    firm_mean_distances = ssd.squareform(ssd.pdist(firm_mean_data,metric="cosine"))

    # output distances
    name = "{:s}_{:s}_firmmeansdistances.csv".format(proc.labels.transforms_name, proc.labels.data_name)
    output_fname = main_folder.joinpath(name)
    pd.DataFrame(firm_mean_distances, index=firm_mean_data.index, columns=firm_mean_data.index).to_csv(str(output_fname))

    # firm mean pop mean
    firm_mean_popmean = np.mean(firm_mean_data, axis=0)

    # firm mean nn
    nbrs = skn.NearestNeighbors(n_neighbors=2, algorithm='auto', metric="precomputed").fit(firm_mean_distances)
    firm_mean_nn_distances, firm_mean_nn_indices = nbrs.kneighbors(n_neighbors=1)

    for i, firm in enumerate(firm_mean_data.index):
        ans.loc[firm, "ymean--ymean_neighbor"] = firm_mean_nn_distances[i][0]
        ans.loc[firm, "ymean--ymean_neighborname"] = firm_mean_data.index[firm_mean_nn_indices[i][0]]

        ans.loc[firm, "ymean--ymean_popmean"] = ssd.cosine(firm_mean_data.loc[firm,:], firm_mean_popmean)


    return ans


def do_jaffe_measures(args, labels, data, verbosity):
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
                          lens=None, lens_name=lens_name, metric=args.metric,
                          verbose=verbosity)

    if args.mds:
        if proc.metric == "precomputed":
            dists = proc.distance_matrix
        else:
            X = ssd.pdist(data, metric=args.metric)
            dists = ssd.squareform(X)
        proc.lens = skm.MDS(n_components=args.dimension, dissimilarity="precomputed").fit_transform(dists)
    else:
        proc.lens = skd.PCA(n_components=args.dimension).fit_transform(data)

    # initialize analyzer
    proc.initialize(base_path=args.output)

    # Other outputs
    if args.clustermap: proc.do_clustermap()
    if not args.no_dump_raw: proc.dump_data()
    if True: proc.plot_lens(np.array(labels.rgb_colors)/255., show=False)
    if args.kclusters:
        kClusters = proc.dataframe_kClusters(args.kclusters, dump_summary=True, dump_aggregates=True)


    ans = do_final_year_analysis(proc, data)
    ans = ans.join(do_year_mean_analysis(proc,data), how="outer")
    print(ans)

    main_folder = proc.get_main_folder()
    name = "{:s}_{:s}_jaffemeasures.csv".format(proc.labels.transforms_name, proc.labels.data_name)
    output_fname = main_folder.joinpath(name)
    ans.to_csv(str(output_fname))



def main(raw_args):
    parser = get_parser()
    args = parser.parse_args(raw_args)

    if args.procedure == None:
        parser.print_help()
        exit()

    # fix choices for Jaffe measure
    args.transpose = False
    args.log = False
    args.sum_to_one = True
    args.metric = "cosine"

    labels, data = get_labels_and_data(args)

    args.mds = False
    args.dimension = 2

    do_jaffe_measures(args, labels, data, verbosity=(2 if args.verbose else 0))

if __name__ == "__main__":
    main(sys.argv[1:])
