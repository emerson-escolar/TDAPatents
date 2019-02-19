import pathlib
import numpy as np
import kmapper as km
import seaborn

import mappertools.linkage_mapper as lk
import mappertools.text_dump as tdump


class MapperAnalyzer(object):
    def __init__(self, data, mapper_cf, labels, lens, lens_name,
                 metric, verbosity=0):
        # data is pandas dataframe
        self.data = data
        self.mapper_cf = mapper_cf
        self.lens = lens

        self.lens_name = lens_name
        self.labels = labels

        self.metric = metric
        self.mapper = km.KeplerMapper(verbose=verbosity)

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
        return "{:s}_{:s}_{:s}".format(self.get_param_name(), self.labels.transforms_name, self.labels.data_name)

    def get_fullname(self, cubes, overlap):
        part1 = self.get_data_fullname()

        mapper_choices = "_n{:s}_o{:.2f}_l".format(str(cubes), overlap)

        return part1 + mapper_choices


    def do_analysis(self, n_cubes, overlap, more_data, more_transforms):
        graph = self.mapper.map(self.lens, self.data.values,
                                clusterer = lk.LinkageMapper(metric=self.metric),
                                coverer=km.Cover(nr_cubes=n_cubes, overlap_perc=overlap))

        # from main_folder, separate by n_cubes and overlaps
        output_folder = self.get_main_folder().joinpath("n"+str(n_cubes)+"_o" + str(overlap))

        output_folder.mkdir(parents=True, exist_ok=True)
        fullname = self.get_fullname(n_cubes, overlap)

        output_fname = output_folder.joinpath(fullname + ".html")
        self.mapper.visualize(graph, color_function=self.mapper_cf,
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


        output_fname = output_folder.joinpath(fullname + ".cyjs")
        extra_data = {'members': list(self.data.index)}
        extra_transforms = {}
        extra_data.update(more_data)
        extra_transforms.update(more_transforms)

        tdump.cytoscapejson_dump(graph, str(output_fname),
                                 extra_data, extra_data,
                                 extra_transforms, extra_transforms)

    def do_clustermap(self,cmap="Reds"):
        output_fname = self.get_main_folder().joinpath(self.get_data_fullname()+ "_cluster.png")
        g = seaborn.clustermap(self.data, metric=self.metric,
                               col_cluster=False, yticklabels=True,
                               cmap=cmap, figsize=(20,40))
        g.savefig(str(output_fname), dpi=75)
