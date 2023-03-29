import pathlib
import numpy as np
import kmapper as km
import seaborn

import mappertools.mapper.covers as cvs
import mappertools.mapper.distances as mdists
import mappertools.mapper.linkage_mapper as lk
import mappertools.outputs.text_dump as tdump
import mappertools.features.flare_balls as flare_balls
import mappertools.features.core as mfc
import mappertools.features.mapper_stats as mfs
import mappertools.mapper.clustering as mclust

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import networkx as nx
import networkx.algorithms as nxa
import networkx.algorithms.centrality as nxc

import pandas
import scipy.spatial.distance
import scipy.cluster.hierarchy

import sklearn.cluster as skc

# pl_jet = [[0.0, 'rgb(0, 0, 127)'],
#           [0.1, 'rgb(0, 0, 241)'],
#           [0.2, 'rgb(0, 76, 255)'],
#           [0.3, 'rgb(0, 176, 255)'],
#           [0.4, 'rgb(41, 255, 205)'],
#           [0.5, 'rgb(124, 255, 121)'],
#           [0.6, 'rgb(205, 255, 41)'],
#           [0.7, 'rgb(255, 196, 0)'],
#           [0.8, 'rgb(255, 103, 0)'],
#           [0.9, 'rgb(241, 7, 0)'],
#           [1.0, 'rgb(127, 0, 0)']]
import turbo_colormap as turbo

pl_turbo = []
for i in np.linspace(0,1, num=11, endpoint=True):
    f_color = turbo.interpolate(turbo.turbo_colormap_data, i)
    pl_turbo.append([i, "rgb({:.0f},{:.0f},{:.0f})".format(f_color[0]*255, f_color[1]*255, f_color[2]*255)])

km.kmapper.colorscale_default = pl_turbo
km.visuals.colorscale_default = pl_turbo

# change the sizes of nodes and edges
def _my_size_node(member_ids):
    return 5 * int(np.log(len(member_ids) + 1) + 1)
km.visuals._size_node = _my_size_node

# current bug in kmapper makes the following pointless:
# TODO: file a bug report to kmapper
def _my_size_link_width(graph, node_id, linked_node_id):
    return 1
km.visuals._size_link_width = _my_size_link_width




class Analyzer(object):
    """
    Class to handle parameter choices and outputs for various analyses.
    """

    def __init__(self, data, labels,
                 lens, lens_name,
                 metric, verbose=0):
        # data is pandas dataframe
        self.data = data
        self.labels = labels

        self.lens = lens
        self.lens_name = lens_name

        self.metric_name = metric
        self.metric = metric
        self.__handle_custom_metric()

        self.mapper = km.KeplerMapper(verbose=verbose)
        self.verbose = verbose

        self.main_folder = None

    def initialize(self, base_path=None):
        ## Determine main output folder below:
        if base_path is None:
            cur_path = pathlib.Path.cwd()
        else:
            cur_path = pathlib.Path(base_path)

        ts_output = True
        if ts_output == True:
            # otherwise, do not specify data_name in main folder, so its shared
            # over different years to which we apply the same metric, lens, and transforms
            metric_lens_transforms = "{:s}_{:s}_{:s}".format(self.metric_name[:3],
                                                             self.lens_name,
                                                             self.labels.transforms_name)
            self.main_folder = cur_path.joinpath(metric_lens_transforms)

            # further subdivide by mode..
            self.main_folder = self.main_folder.joinpath(self.labels.extra_desc)
        self.main_folder.mkdir(parents=True, exist_ok=True)


    def __handle_custom_metric(self):
        if self.metric == "bloom":
            self.metric_name = self.metric
            self.distance_matrix = mdists.flipped_bloom_mahalanobis_dissimilarity(self.data)
            self.metric = "precomputed"

    def get_main_folder(self):
        if self.main_folder is None:
            raise RuntimeError("Analyzer not initialized!")

        return self.main_folder


    def get_param_name(self):
        metric_lens = "{:s}_{:s}".format(self.metric_name[:3], self.lens_name)
        return metric_lens

    def get_data_fullname(self):
        return "{:s}_{:s}_{:s}".format(self.get_param_name(),
                                       self.labels.transforms_name,
                                       self.labels.data_name)


    def get_fullname(self, cubes, overlap, clusterer_name=None):
        part1 = self.get_data_fullname()

        mapper_choices = "_n{:s}_o{:.2f}".format(str(cubes), overlap)
        if clusterer_name: mapper_choices += ("_" + clusterer_name)

        return part1 + mapper_choices


    def get_output_folder(self, n_cubes, overlap, clusterer_name):
        # from main_folder, separate by n_cubes and overlaps and clusterer_name
        output_folder = self.get_main_folder().joinpath("n"+str(n_cubes)+"_o" + str(overlap)+ "_" + clusterer_name)
        output_folder.mkdir(parents=True, exist_ok=True)
        return output_folder

    def get_mapper_colorpair(self):
        # print(self.labels.sectors_data)

        color_function_name = []
        color_function_values = np.array([])

        if self.labels.years_data is not None:
            color_function_name = ["years","psizes"]
            color_function_values = np.array([list(self.labels.years_data), list(self.labels.p_sizes)]).T
        else:
            color_function_name = ["psizes"]
            color_function_values = self.labels.p_sizes.reshape(-1,1)

        if self.labels.sectors_data is not None:
            sectors_data = self.labels.sectors_data
            for col in sectors_data.columns:
                color_function_name.append(col)

                color_function_values = np.concatenate(
                    (color_function_values, sectors_data.loc[:,col].to_numpy().reshape(-1,1)),
                                axis=1 )

        return color_function_name, color_function_values


    def compute_mapper_graph(self, n_cubes, overlap, clusterer_dict, html_output=True):
        clusterer = None
        if clusterer_dict["clusterer_arg"][:3] == "HC_":
            clusterer = lk.HeuristicHierarchical(metric = self.metric,
                                                 method = clusterer_dict["clusterer_arg"][3:],
                                                 heuristic = clusterer_dict["clusterer_HC_heuristic"],
                                                 verbose=self.verbose,
                                                 bins="doane")
        elif clusterer_dict["clusterer_arg"] == "OPTICS":
            print("Clustering using: sklearn OPTICS")
            clusterer = skc.OPTICS(metric = self.metric)


        if self.metric == "precomputed":
            graph = self.mapper.map(self.lens, X = self.distance_matrix,
                                    precomputed = True,
                                    clusterer = clusterer,
                                    cover=km.Cover(n_cubes=n_cubes, perc_overlap=overlap))
        else:
            graph = self.mapper.map(self.lens, self.data.values,
                                    clusterer = clusterer,
                                    cover=km.Cover(n_cubes=n_cubes, perc_overlap=overlap))

        output_folder = self.get_output_folder(n_cubes, overlap, clusterer_dict["clusterer_name"])
        fullname = self.get_fullname(n_cubes, overlap, clusterer_dict["clusterer_name"])

        if html_output:
            output_fname = output_folder.joinpath(fullname + ".html")
            color_function_name, color_values = self.get_mapper_colorpair()
            self.mapper.visualize(graph,
                                  color_function_name = color_function_name,
                                  color_values = color_values,
                                  include_searchbar = True,
                                  path_html = str(output_fname),
                                  title = fullname,
                                  custom_tooltips=np.array(self.data.index))
        return graph


    # def do_advanced_outputs(self, nxgraph, output_folder, fullname):
    #     output_fname = output_folder.joinpath(fullname + ".txt")
    #     ofile = open(str(output_fname), 'w')
    #     tdump.kmapper_text_dump(graph, ofile, list(self.data.index))
    #     ofile.close()

    #     output_fname = output_folder.joinpath(fullname+"clus_ave.txt")
    #     ofile = open(str(output_fname), 'w')
    #     tdump.kmapper_dump_cluster_averages(self.data, graph, ofile)
    #     ofile.close()
    #     pass

    # def do_flare_csv(self, nxgraph, output_folder, fullname, flare_query_string):
    #     output_fname = output_folder.joinpath(fullname + "_flare_stats.csv")
    #     flare_k = flare_balls.compute_all_summary(nxgraph, entities=self.labels.unique_members,
    #                                               query_data=flare_query_string, verbose=self.verbose, keep_missing=True)
    #     flare_k.to_csv(output_fname)

    #     return

    def do_mapper_stats_txt(self, nxgraph, output_folder, fullname, query_string):
        """
        query_string is the node attribute key containing 'unique members' (names of entities) of each node.
        """
        output_fname = output_folder.joinpath(fullname + "_mapper_stats.txt")

        def write_data(f, desc, data, star=False):
            if star:
                f.write("_" * len(desc) + "\n")
            f.write(desc)
            f.write(str(data))
            f.write("\n")
            if star:
                f.write("=" * len(desc) + "\n")

        with open(output_fname, 'w') as f:
            f.write(nx.info(nxgraph))
            f.write("\n")

            write_data(f, "Number of connected components: ", nx.number_connected_components(nxgraph))

            deg_dist = pandas.Series(nx.degree_histogram(nxgraph))
            dist_as_string = deg_dist.to_string(header=True, index=True)
            write_data(f, "degree (of nodes) distribution:\n", dist_as_string, True)

            # write_data(f, "Small-world sigma: ", nxa.smallworld.sigma(nxgraph))
            # write_data(f, "Small-world omega: ", nxa.smallworld.omega(nxgraph))


        with open(output_fname, 'a') as f:
            f.write("\n\n" + "#" * 70 + "\n")
            f.write("Entity-based stats:\n(entity = firms in firm_rank_name_industry.csv; or patent classes)\n")
            f.write("\n")

            write_data(f, "Number of entities (inc. all-zero data): ", len(self.labels.unique_members))

            cnode_dist = mfs.containing_node_distribution(nxgraph, self.labels.unique_members, query_string)
            dist_as_string = cnode_dist.to_string(header=True, index=True)
            write_data(f, "num_containing_nodes (of entities) frequency distribution:\n", dist_as_string, True)

        return



    def do_derived_stats_csv(self, nxgraph, output_folder, fullname, query_string):
        """
        query_string is the node attribute key containing 'unique members' (names of entities) of each node.
        """

        output_fname = output_folder.joinpath(fullname + "_derived_stats.csv")
        derived_stats = flare_balls.compute_all_summary(nxgraph, entities=self.labels.unique_members,
                                                        query_data=query_string, verbose=self.verbose, keep_missing=True)

        # centralities
        def eigenvector_centrality(G):
            return nxc.eigenvector_centrality(G, max_iter=10000)

        centrality_functions = [nxc.degree_centrality, nxc.harmonic_centrality, nxc.closeness_centrality,
                                eigenvector_centrality, nxc.betweenness_centrality]
        aggregation_functions = [np.mean, np.min, np.max]

        stats = mfs.compute_centrality_measures(nxgraph, self.labels.unique_members,
                                                centrality_functions, aggregation_functions,
                                                query_string)

        derived_stats = derived_stats.join(stats, how="outer")


        # containing nodes
        stats = mfs.compute_entity_membership(nxgraph, self.labels.unique_members, query_string)
        derived_stats = derived_stats.join(stats, how="outer")




        derived_stats.to_csv(output_fname)

        return



    def do_clustermap(self, cmap="Reds", overwrite=False):
        output_fname = self.get_main_folder().joinpath(self.get_data_fullname()+ "_cluster.png")

        if not overwrite and output_fname.exists():
            print("{} exists! Skipping clustermap plot".format(str(output_fname)))
            return

        if self.metric == "precomputed":
            g = seaborn.clustermap(self.data,
                                   col_cluster=False,
                                   row_linkage = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(self.distance_matrix, force='tovector'),metric="precomputed"),
                                   yticklabels=True, cmap=cmap, figsize=(20,40))
        else:
            g = seaborn.clustermap(self.data, metric=self.metric,
                                   col_cluster=False, yticklabels=True,
                                   cmap=cmap, figsize=(20,40))
        g.savefig(str(output_fname), dpi=75)
        g.fig.clear()


    def plot_lens(self, rgb_colors=None, show=False, overwrite=False):
        main_folder = self.get_main_folder()
        name = "{:s}_{:s}_{:s}.png".format(self.labels.transforms_name,
                                           self.labels.data_name,
                                           self.lens_name)
        output_fname = main_folder.joinpath(name)

        allow_writing = True
        if not overwrite and output_fname.exists():
            print("{} exists! Skipping lens plot".format(str(output_fname)))
            allow_writing = False

        if not show and not allow_writing:
            return

        if self.lens.shape[1] == 2:
            plt.figure(figsize=(8,8))
            plt.scatter(self.lens[:,0], self.lens[:,1],c=rgb_colors)

            if allow_writing: plt.savefig(str(output_fname))
            if show: plt.show()

        elif self.lens.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # ax = fig.gca(projection='3d')
            # ax.set_aspect("equal")
            ax.scatter(self.lens[:,0], self.lens[:,1], self.lens[:,2], c=rgb_colors)

            if allow_writing: plt.savefig(str(output_fname))
            if show: plt.show()


    def dump_data(self, overwrite=False):
        main_folder = self.get_main_folder()
        name = "{:s}_{:s}.parquet".format(self.labels.transforms_name,
                                          self.labels.data_name)
        output_fname = main_folder.joinpath(name)

        name = "{:s}_{:s}_labels.json".format(self.labels.transforms_name, self.labels.data_name)
        labels_fname = main_folder.joinpath(name)

        if not overwrite:
            if output_fname.exists():
                print("{} exists! Skipping annotated data dump.".format(str(output_fname)))
                return
            if labels_fname.exists():
                print("{} exists! Skipping annotated data dump.".format(str(labels_fname)))
                return

        # self.data.to_parquet(str(output_fname), engine='pyarrow',index=True)
        self.data.to_parquet(str(output_fname), engine='auto',index=True)
        self.labels.to_json_fname(str(labels_fname))


    def dataframe_kClusters(self, k_list, dump_summary=False, dump_aggregates=False):
        ans = pandas.DataFrame(index = self.data.index)

        # precompute distance matrix for kMedoids
        if self.metric != "precomputed":
            distance_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(self.data, metric=self.metric))
        else:
            distance_matrix = self.distance_matrix

        for k in k_list:
            ans = self.__do_kMedoids(k, distance_matrix, ans, dump=dump_aggregates)
            ans = self.__do_kMeans(k, ans, dump=dump_aggregates)

        if dump_summary:
            name = "{:s}_{:s}_{:s}_kclusters.csv".format(self.metric_name,
                                                         self.labels.transforms_name,
                                                         self.labels.data_name)
            output_fname = self.get_main_folder().joinpath(name)
            ans.to_csv(output_fname)

        return ans


    def __do_kMedoids(self, k, distance_matrix, ans, dump=False):
        prefix = "k{}Med".format(k)

        clus = mclust.kMedoids(metric="precomputed", heuristic=k, prefix=prefix).fit(distance_matrix)
        ans[clus.prefix] = clus.labels_

        if dump:
            name = "{:s}_{:s}_{:s}_{:s}.csv".format(self.metric_name,
                                                    self.labels.transforms_name,
                                                    self.labels.data_name,
                                                    prefix)
            output_fname = self.get_main_folder().joinpath(name)

            agg, years_agg, leaders_agg = self.__do_labels_aggregation(ans[clus.prefix])
            with open(output_fname, 'w') as f:
                agg.to_csv(f)
                f.write("\n")
            with open(output_fname, 'a') as f:
                years_agg.to_csv(f)
                f.write("\n")
                leaders_agg.to_csv(f)

        return ans


    def __do_kMeans(self, k, ans, dump=False):
        prefix = "k{}Means".format(k)
        clus = mclust.kMeans(metric="euclidean", heuristic=k, prefix=prefix).fit(self.data)
        ans[clus.prefix] = clus.labels_

        if dump:
            name = "{:s}_{:s}_{:s}_{:s}.csv".format(self.metric_name,
                                                    self.labels.transforms_name,
                                                    self.labels.data_name,
                                                    prefix)
            output_fname = self.get_main_folder().joinpath(name)

            agg, years_agg, leaders_agg = self.__do_labels_aggregation(ans[clus.prefix])
            with open(output_fname, 'w') as f:
                agg.to_csv(f)
                f.write("\n")
            with open(output_fname, 'a') as f:
                years_agg.to_csv(f)
                f.write("\n")
                leaders_agg.to_csv(f)

        return ans


    def __do_labels_aggregation(self, labels):
        agg = mclust.unique_entity_counts_by_cluster(labels,
                                                     unique_names=self.labels.intemporal_index,
                                                     cluster_totals=False)
        total_firm_years = agg.sum().rename('TOTAL_FIRMYEARS')
        unique_firms = agg.astype(bool).sum(axis=0).rename('UNIQUE_FIRMS')

        leaders_agg = pandas.concat((pandas.Series([x + '__({})'.format(num)
                                                    for num,x in sorted(zip(agg.loc[:,col_name], agg.index), reverse=True)
                                                    if num != 0])
                                     for col_name in agg.columns), axis=1)
        leaders_agg.columns = agg.columns


        years_agg = mclust.unique_entity_counts_by_cluster(labels, unique_names=self.labels.years_data)
        total_years = years_agg.sum().rename('TOTAL_YEARS')
        unique_years = years_agg.astype(bool).sum(axis=0).rename('UNIQUE_YEARS')

        years = np.array([[int(y) for y in years_agg.index]])
        mean_years = (years @ years_agg / total_years)
        mean_years.index = ['MEAN_YEARS']

        # agg = agg.append(total_firm_years).append(unique_firms)
        agg = pandas.concat([agg, total_firm_years.to_frame().T, unique_firms.to_frame().T])

        # years_agg = years_agg.append(unique_years).append(mean_years, )
        years_agg = pandas.concat([years_agg, unique_years.to_frame().T, mean_years])

        return agg, years_agg, leaders_agg
