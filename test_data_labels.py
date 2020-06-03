from a_data_labels import DataLabels
import analysis_merged_pca
import tempfile
import pytest
import numpy

@pytest.mark.parametrize(("data","mode"), [(1,0),
                                           pytest.param(1,2, marks=pytest.mark.comprehensive),
                                           pytest.param(0,0, marks=pytest.mark.comprehensive),
                                           pytest.param(0,2, marks=pytest.mark.comprehensive)])
@pytest.mark.parametrize("cos_trans", [pytest.param(True, marks=pytest.mark.comprehensive), False])
@pytest.mark.parametrize("program_mode", ["accumulate", "merge", "ma"])
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("log", [True, False])
@pytest.mark.parametrize("sum_to_one", [True, False])
@pytest.mark.parametrize("metric", ["correlation", "cosine", "bloom"])
def test_json_dump(program_mode, data, mode,
                   cos_trans, transpose, log, sum_to_one,
                   metric,
                   tmp_path):
    args = [program_mode,
            "--data", str(data),
            "--mode", str(mode),
            "--metric", metric,
            "--from_year", "1981",
            "--to_year", "1984"]
    if cos_trans: args += ["--cos_trans"]
    if transpose: args += ["--transpose"]
    if log: args += ["--log"]
    if sum_to_one: args += ["--sum_to_one"]

    if program_mode == "ma":
        args += ["-w", "2",
                 "-s", "2"]

    args = analysis_merged_pca.get_parser().parse_args(args)
    labels, _ = analysis_merged_pca.get_labels_and_data(args)

    fname = tmp_path / "datalabels.json"
    print(str(fname))
    labels.to_json_fname(fname)

    loaded_labels = DataLabels.from_json_fname(fname)

    assert labels.extra_desc == loaded_labels.extra_desc
    assert labels.data_name == loaded_labels.data_name
    assert labels.transforms_name == loaded_labels.transforms_name
    assert labels.mode_name == loaded_labels.mode_name

    if labels.p_sizes is not None:
        assert numpy.allclose(loaded_labels.p_sizes, labels.p_sizes)

    if labels.rgb_colors is not None:
        assert numpy.allclose(loaded_labels.rgb_colors, labels.rgb_colors)


    assert labels.unique_members == loaded_labels.unique_members

    if labels.years_data is not None:
        assert numpy.allclose(loaded_labels.years_data, labels.years_data)

    if labels.intemporal_index is not None:
        assert labels.intemporal_index.equals(loaded_labels.intemporal_index)
