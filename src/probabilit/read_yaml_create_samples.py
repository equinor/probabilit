from probabilit.modeling import (
    Distribution,
    Add,
    EmpiricalDistribution,
    CumulativeDistribution,
    DiscreteDistribution,
    NoOp,
)
import seaborn
import yaml
from yaml.loader import Loader
import pandas as pd
import scipy as sp
import operator
import functools

TYPE_MAPPING = {
    "empiricaldistribution": EmpiricalDistribution,
    "cumulativedistribution": CumulativeDistribution,
    "discretedistribution": DiscreteDistribution,
}

if __name__ == "__main__":
    file = "config.yml"

    # =================== LOAD ===================

    # Load data from file into a dictionary
    with open(file, "r") as file_handle:
        yaml_data = yaml.load(file_handle, Loader)

    # Load sections of the dictionary as variables
    metadata = yaml_data["metadata"]
    variables = yaml_data["variables"]
    correlations = yaml_data.get("correlations", [])
    plots = yaml_data.get("plot", [])
    derived = yaml_data.get("derived", [])

    # =================== CONVERT ===================

    # Convert dict of {name:data, ...} to {name:Distribution, ...}
    for varname in variables.keys():
        vardata = variables[varname]
        var_type = vardata.pop("type").lower()

        # See if variable matches one of the non-scipy distributions first
        if var_type in TYPE_MAPPING.keys():
            var_class = TYPE_MAPPING[var_type]
            variables[varname] = var_class(**vardata)
            continue

        # Not a non-scipy distribution, so try scipy next
        variables[varname] = Distribution(var_type, **vardata)

    # =================== CORRELATIONS ===================

    # Dummy expression to sample all parents
    expression = NoOp(*variables.values())

    # Add every correlation pair
    # NOTE if we define correlations between (a, b) and (c, d)
    # then the non-specified correlations (e.g. (a, c)) will be optimized
    # towards zero.
    for correlation in correlations:
        value, (varname1, varname2) = next(iter(correlation.items()))

        # Prepare input
        corr_mat = sp.linalg.circulant([1.0, value])
        var1, var2 = variables[varname1], variables[varname2]

        # Add correlation pair
        expression.correlate(var1, var2, corr_mat=corr_mat)

    # =================== SAMPLE ===================

    # Samle the expression, which populates `.samples_` on Distributions
    expression.sample(
        size=metadata["samples"],
        random_state=metadata["seed"],
        method=metadata["sampling_method"],
    )

    # Collect all samples into a dataframe
    df_samples = pd.DataFrame(
        {name: distr.samples_ for (name, distr) in variables.items()}
    )

    # =================== DERIVED ===================
    for derived_from, data in derived.items():
        for derived_to in data.keys():
            mapping = functools.reduce(operator.ior, data[derived_to])
            df_samples = df_samples.assign(
                **{derived_to: lambda df: df[derived_from].map(mapping)}
            )

    print(df_samples)
    print(df_samples.select_dtypes("number").corr())

    # =================== PLOTS ===================

    for vars_plot in plots:
        vars_plot = [vars_plot] if isinstance(vars_plot, str) else vars_plot
        df = pd.DataFrame(
            {var_plot: variables[var_plot].samples_ for var_plot in vars_plot}
        )
        seaborn.pairplot(df)
