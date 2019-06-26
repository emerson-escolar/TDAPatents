import pathlib
import numpy as np
import kmapper as km
import seaborn

import mappertools.linkage_mapper as lk
import mappertools.text_dump as tdump
import mappertools.covers as cvs
import mappertools.features.flare_balls as flare_balls

import networkx as nx


class MapperAnalyzer(object):
    """
    Class to handle logic of managing parameter choices and outputs.
    """

    def __init__(self, data, unique_members,
                 mapper_cf, labels, lens, lens_name,
                 metric, verbose=0):
        # data is pandas dataframe
        self.data = data
        self.unique_members = unique_members
        self.mapper_cf = mapper_cf
        self.lens = lens

        self.lens_name = lens_name
        self.labels = labels

        self.metric = metric
        self.mapper = km.KeplerMapper(verbose=verbose)
        self.verbose = verbose

        ## Determine main output folder below:
        cur_path = pathlib.Path.cwd()
        ts_output = True
        if ts_output == True:
            # otherwise, do not specify data_name in main folder, so its shared
            # over different years to which we apply the same metric, lens, and transforms
            metric_lens_transforms = "{:s}_{:s}_{:s}".format(self.metric[:3],
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

    def get_data_fullname(self):
        return "{:s}_{:s}_{:s}".format(self.get_param_name(),
                                       self.labels.transforms_name,
                                       self.labels.data_name)


    def get_fullname(self, cubes, overlap, heuristic=None):
        part1 = self.get_data_fullname()

        mapper_choices = "_n{:s}_o{:.2f}".format(str(cubes), overlap)
        if heuristic: mapper_choices += ("_" + heuristic)

        return part1 + mapper_choices


    def get_output_folder(self, n_cubes, overlap, heuristic):
        # from main_folder, separate by n_cubes and overlaps and heuristic
        output_folder = self.get_main_folder().joinpath("n"+str(n_cubes)+"_o" + str(overlap)+ "_" + heuristic)
        output_folder.mkdir(parents=True, exist_ok=True)
        return output_folder


    def do_basic_analysis(self, n_cubes, overlap, heuristic='firstgap', html_output=True):
        graph = self.mapper.map(self.lens, self.data.values,
                                clusterer = lk.LinkageMapper(metric=self.metric,
                                                             heuristic=heuristic,
                                                             verbose=self.verbose),
                                cover=km.Cover(n_cubes=n_cubes, perc_overlap=overlap))

        output_folder = self.get_output_folder(n_cubes,overlap,heuristic)
        fullname = self.get_fullname(n_cubes, overlap, heuristic)

        if html_output:
            output_fname = output_folder.joinpath(fullname + ".html")
            self.mapper.visualize(graph, color_function=self.mapper_cf,
                                  path_html = str(output_fname),
                                  title = fullname,
                                  custom_tooltips=np.array(self.data.index))
        return graph


    def do_advanced_outputs(self, nxgraph, output_folder, fullname, query_data='unique_members'):
        # output_fname = output_folder.joinpath(fullname + ".txt")
        # ofile = open(str(output_fname), 'w')
        # tdump.kmapper_text_dump(graph, ofile, list(self.data.index))
        # ofile.close()

        # output_fname = output_folder.joinpath(fullname+"clus_ave.txt")
        # ofile = open(str(output_fname), 'w')
        # tdump.kmapper_dump_cluster_averages(self.data, graph, ofile)
        # ofile.close()

        if True:
            output_fname = output_folder.joinpath(fullname + ".cyjs")
            tdump.cytoscapejson_dump(nxgraph, output_fname)

        if True:
            output_fname = output_folder.joinpath(fullname + "_flare_stats.csv")
            flare_k = flare_balls.compute_all_summary(nxgraph, self.unique_members, query_data=query_data, verbose=self.verbose)
            flare_k.to_csv(output_fname)
        return






    def do_clustermap(self,cmap="Reds"):
        output_fname = self.get_main_folder().joinpath(self.get_data_fullname()+ "_cluster.png")
        g = seaborn.clustermap(self.data, metric=self.metric,
                               col_cluster=False, yticklabels=True,
                               cmap=cmap, figsize=(20,40))
        g.savefig(str(output_fname), dpi=75)
        g.fig.clear()
