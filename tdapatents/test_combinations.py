import pytest
import analysis_merged_pca


constant_options = ["--from_year", "1981",
                    "--to_year", "1984",
                    "--numbers", "10",
                    "--overlaps", "0.5",
                    "--kclusters", "6", "21",
                    "-o", "test_folder"]

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
def test_all(program_mode, data, mode,
             cos_trans, transpose, log, sum_to_one,
             metric):
    args = [program_mode,
            "--data", str(data),
            "--mode", str(mode),
            "--metric", metric]
    if cos_trans: args += ["--cos_trans"]
    if transpose: args += ["--transpose"]
    if log: args += ["--log"]
    if sum_to_one: args += ["--sum_to_one"]

    args += (constant_options)

    if program_mode == "ma":
        args += ["-w", "2",
                 "-s", "2"]

    analysis_merged_pca.main(args)
