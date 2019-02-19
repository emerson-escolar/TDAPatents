import numpy as np
import sklearn
import scipy
import pathlib
import argparse

import kmapper as km
import sklearn.manifold as skm
import sklearn.decomposition as skd

import mappertools.linkage_mapper as lk
import mappertools.visualization as qs
import mappertools.text_dump as tdump

from patent_data import PatentData

class MapperAnalyzer(object):
    def __init__(self, data, color_function, labels, lens, lens_name,
                 metric, ts_output=False, verbosity=0):
        # data is pandas dataframe
        self.data = data
        self.color_function = color_function
        self.lens = lens

        self.lens_name = lens_name
        self.labels = labels

        self.metric = metric
        self.mapper = km.KeplerMapper(verbose=verbosity)

        self.ts_output = ts_output

        ## Determine main output folder below:
        cur_path = pathlib.Path.cwd()
        if ts_output == False:
            # if not organize into time series, each data (year!) gets its own folder..
            metric_lens_transforms_data = "{:s}_{:s}_{:s}_{:s}".format(self.metric[:3],
                                                                       self.lens_name,
                                                                       self.labels.transforms_name,
                                                                       self.labels.data_name)
            self.main_folder = cur_path.joinpath(metric_lens_transforms_data)
        else:
            # otherwise, do not specify data_name in main folder, so its shared
            # over different years to which we apply the same metric, lens, and transforms
            metric_lens_transforms = "ts_{:s}_{:s}_{:s}".format(self.metric[:3],
                                                                self.lens_name,
                                                                self.labels.transforms_name)
            self.main_folder = cur_path.joinpath(metric_lens_transforms)

            # further subdivide by mode..
            self.main_folder = self.main_folder.joinpath(labels.extra_desc)
        self.main_folder.mkdir(parents=True, exist_ok=True)


    def get_main_folder(self):
        return self.main_folder

    def get_param_name(self):
        metric_lens = "{:s}_{:s}".format(self.metric[:3], self.lens_name)
        return metric_lens



    def get_fullname(self, cubes, overlap):
        part1 = "{:s}_{:s}_{:s}".format(self.get_param_name(), self.labels.transforms_name, self.labels.data_name)
        mapper_choices = "_n{:s}_o{:.2f}_l".format(str(cubes), overlap)

        return part1 + mapper_choices


    def do_analysis(self, n_cubes, overlap):
        graph = self.mapper.map(self.lens, self.data.values,
                                clusterer = lk.LinkageMapper(metric=self.metric),
                                coverer=km.Cover(nr_cubes=n_cubes, overlap_perc=overlap))

        if self.ts_output == False:
            # from main_folder, separate by n_cubes
            output_folder = self.get_main_folder().joinpath("n_cube" + str(n_cubes))
        else:
            # from main_folder, separate by n_cubes and overlaps
            output_folder = self.get_main_folder().joinpath("n" + str(n_cubes) + "_o" + str(overlap))

        output_folder.mkdir(parents=True, exist_ok=True)
        fullname = self.get_fullname(n_cubes, overlap)
        output_fname = output_folder.joinpath(fullname + ".html")

        self.mapper.visualize(graph, color_function=self.color_function,
                              path_html = str(output_fname),
                              title = fullname,
                              custom_tooltips=np.array(self.data.index))


        output_fname = output_folder.joinpath(fullname + ".txt")
        ofile = open(str(output_fname), 'w')
        tdump.kmapper_text_dump(graph, ofile, list(self.data.index))
        ofile.close()

        output_fname = output_folder.joinpath(fullname+"clus_ave.txt")
        ofile = open(str(output_fname), 'w')
        tdump.kmapper_dump_cluster_averages(self.data, graph, ofile)
        ofile.close()





def do_mapper(args, bigdata, verbosity):
    for year in args.years:
        data, cf, labels = bigdata.get_data(year, do_log=args.log, do_transform=args.cos_trans,
                                        drop_zero=(not args.keep_zeros))
        if (data.shape[0] == 0):
            print("Warning: year {:s} has no nonzero data! Skipping..".format(str(year)))
            continue

        if (data.shape[0] == 1):
            print("Warning: year {:s} has only one nonzero firm data! Skipping..".format(str(year)))
            continue


        proc = MapperAnalyzer(data, cf, labels=labels, lens= None, lens_name="pca2d", metric=args.metric,
                              ts_output=args.time_series, verbosity=verbosity)
        print(proc.get_main_folder())

        name = "{:s}_{:s}_{:s}".format(proc.metric[:3], proc.labels.transforms_name, proc.labels.data_name)


        pca = skd.PCA(n_components=2)
        proc.lens = pca.fit_transform(data)


        for cub in args.numbers:
            for overlap in args.overlaps:
                proc.do_analysis(cub, overlap)




def main():
    parser = argparse.ArgumentParser(description="Program for performing Mapper analysis on patent data.")
    parser.add_argument("--verbose", "-v", action="store_true", help="verbose.")
    parser.add_argument("--log", "-l", action="store_true", help="do log.")
    parser.add_argument("--time_series", "-t", action="store_true", help="output instead as one folder containing one time series for each parameter combination. Otherwise, have one folder for each year.")

    parser.add_argument("--accumulate", "-a", action="store_true", help="instead of year-by-year, add all years data after transformations.")

    parser.add_argument("--keep_zeros", "-z", action="store_true", help="preserve zeros columns in data. Do not use. Otherwise, drop zero columns.")
    parser.add_argument("--cos_trans", "-c", action="store_true", help="use cosine distances to transform data.")
    parser.add_argument("--mode", "-m", help="mode choice: 0 or 1 or 2 (default: 0; both modes: 2).", type=int, default=0, choices=[0,1,2])
    parser.add_argument("--metric", "-d", help="metric choice: 'euclidean' or 'correlation' (default: 'correlation').", type=str, default='correlation', choices=['euclidean', 'correlation'])

    parser.add_argument("--numbers", "-n", help="number(s) of cover elements in each axis.", type=int, nargs="+", default=[5,10,15,20])

    parser.add_argument("--overlaps", "-p", help="overlap(s) of cover elements. Express as decimal between 0 and 1.", type=float, nargs="+", default=[0.5])

    parser.add_argument("--years", "-y", help="years to do analysis.", type=int, nargs="+")





    args = parser.parse_args()

    if args.mode == 0:
        bigdata = PatentData(extra_data_desc="m0",
                             patent_folder_name="180901_csv/reshape_wide_byyear_mode0",
                             patent_fname_format="reshape_wide_year{:d}_mode0.csv",
                             cosdis_folder_name="180901_csv/cosine_distance_byyear",
                             cosdis_fname_format="cosine_distance_year{:d}.csv",
                             translator_fname="180901_csv/firm_rank_name.csv",
                             translator_func=(lambda x: ("firm_rank_{:s}").format(x)))
    elif args.mode == 1:
        bigdata = PatentData(extra_data_desc="m1",
                             patent_folder_name="180901_csv/reshape_wide_byyear_mode1",
                             patent_fname_format="reshape_wide_year{:d}_mode1.csv",
                             cosdis_folder_name="180901_csv/cosine_distance_byyear",
                             cosdis_fname_format="cosine_distance_year{:d}.csv",
                             translator_fname="180901_csv/firm_rank_name.csv",
                             translator_func=(lambda x: ("firm_rank_{:s}").format(x)))
    elif args.mode == 2:
        bigdata = PatentData(extra_data_desc="allm",
                             patent_folder_name="180901_csv/reshape_wide_byyear_allmodes",
                             patent_fname_format="reshape_wide_year{:d}_allmodes.csv",
                             cosdis_folder_name="180901_csv/cosine_distance_byyear",
                             cosdis_fname_format="cosine_distance_year{:d}.csv",
                             translator_fname="180901_csv/firm_rank_name.csv",
                             translator_func=(lambda x: ("firm_rank_{:s}").format(x)))

    verbosity = 0
    if args.verbose:
        verbosity = 2

    do_mapper(args, bigdata, verbosity)







if __name__ == "__main__":
    main()
