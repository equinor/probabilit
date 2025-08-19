from probabilit.modeling import (
    Distribution,
    Add,
    EmpiricalDistribution,
    CumulativeDistribution,
)
import yaml
from yaml.loader import Loader
import pandas as pd
import scipy as sp

TYPE_MAPPING = {
    "empiricaldistribution": EmpiricalDistribution,
    "cumulativedistribution": CumulativeDistribution,
}

if __name__ == "__main__":
    file = "config.yml"

    # Load data from file into a dictionary
    with open(file, "r") as file_handle:
        yaml_data = yaml.load(file_handle, Loader)

    # Load sections of the dictionary as variables
    metadata = yaml_data["metadata"]
    variables = yaml_data["variables"]
    correlations = yaml_data.get("correlations", [])

    # Convert dict of {name:data, ...} to {name:Distribution, ...}
    for varname in variables.keys():
        vardata = variables[varname]
        var_type = vardata["type"].lower()

        # See if variable matches one of the non-scipy distributions first
        if var_type in TYPE_MAPPING.keys():
            var_class = TYPE_MAPPING[var_type]
            variables[varname] = var_class(**vardata["parameters"])
            continue

        # Not a non-scipy distribution, so try scipy next
        variables[varname] = Distribution(var_type, **vardata["parameters"])

    # Dummy expression to sample all parents
    expression = Add(*variables.values())

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

    # Samle the expression, which populates `.samples_` on Distributions
    expression.sample(size=metadata["samples"])

    # Collect all samples into a dataframe
    df_samples = pd.DataFrame(
        {name: distr.samples_ for (name, distr) in variables.items()}
    )

    print(df_samples)

    print(df_samples.corr())
