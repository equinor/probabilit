from probabilit.modeling import (
    Distribution,
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
import argparse

TYPE_MAPPING = {
    "empiricaldistribution": EmpiricalDistribution,
    "cumulativedistribution": CumulativeDistribution,
    "discretedistribution": DiscreteDistribution,
}


def design_matrix(config, verbose=0):
    """Given a dictionary config, returns a dataframe with samples."""

    # Load sections of the dictionary as variables
    metadata = config["metadata"]
    variables = config["variables"]
    correlations = config.get("correlations", [])
    plots = config.get("plot", [])
    derived = config.get("derived", [])

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
    # Derived variables AFTER correlations, since inducing correlations
    # on derived variables would break the connection.
    for derived_from, data in derived.items():
        for derived_to in data.keys():
            # Create mapping {from1: to1, from2: to2, ...} and apply it
            mapping = functools.reduce(operator.ior, data[derived_to])
            df_samples = df_samples.assign(
                **{derived_to: lambda df: df[derived_from].map(mapping)}
            )

    if verbose > 0:
        print("========== DESIGN MATRIX ==========")
        print(df_samples)
        print("========== CORRELATIONS ==========")
        print(df_samples.select_dtypes("number").corr())

    # =================== PLOTS ===================

    for vars_plot in plots:
        vars_plot = [vars_plot] if isinstance(vars_plot, str) else vars_plot
        df = pd.DataFrame(
            {var_plot: variables[var_plot].samples_ for var_plot in vars_plot}
        )
        seaborn.pairplot(df)

    return df_samples


def cmd_designmatr(args):
    """Execute the subcommand."""
    print(args)

    # Load data from file into a dictionary
    with open(args.config, "r") as file_handle:
        config = yaml.load(file_handle, Loader)
        print(f"Loaded: {args.config}")

    df_samples = design_matrix(config, verbose=args.verbose)
    df_samples.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


def one_by_one(df, defaults=None, verbose=0):
    """Transform a (n, p) dataframe to (n x p, p), keeping
    all but one variable (column) constant at a time."""
    if defaults is None:
        defaults = dict()

    for key in defaults.keys():
        if key not in df.columns:
            raise ValueError("Default {key=} not in {df.columns=}")

    # Average for numeric, mode for rest
    averages = df.select_dtypes("number").mean().to_dict()
    modes = df.select_dtypes("number").mode().T.to_dict()[0]
    constants = (averages | modes) | defaults

    assert set(constants.keys()) == set(df.columns)

    if verbose:
        print(f"Constants: {constants}")

    dfs = []
    for column in df.columns:
        # Set all columns except the current one to a constant
        avg_map = {k: v for (k, v) in constants.items() if k != column}
        dfs.append(df.assign(**avg_map))

    return pd.concat(dfs)


def cmd_onebyone(args):
    """Execute the subcommand."""
    print(args)

    df = pd.read_csv(args.designmatrix)
    print(f"Loaded: {args.designmatrix}")

    if args.config is not None:
        with open(args.config, "r") as file_handle:
            config = yaml.load(file_handle, Loader)
            print(f"Loaded: {args.config}")
            defaults = config["defaults"]  # Default map col -> const
    else:
        defaults = None

    df = one_by_one(df, defaults=defaults, verbose=args.verbose)
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    subparsers = parser.add_subparsers(help="")

    # Parse the 'designmatrix' subcommand
    p1 = subparsers.add_parser(
        "designmatrix", help="Creates a design matrix (random samples)."
    )
    p1.add_argument("config", help="A .yml config file.")
    p1.add_argument("--output", help="An output .csv file.", default="designmatrix.csv")
    p1.add_argument("--verbose", "-v", action="count", default=0)
    p1.set_defaults(func=cmd_designmatr)

    # Parse the 'onebyone' subcommand
    p2 = subparsers.add_parser(
        "onebyone", help="Transform a design matrix to a one-by-one matrix."
    )
    p2.add_argument("designmatrix", help="Input .csv file.", default="designmatrix.csv")
    p2.add_argument("--config", help="A .yml config file.", default=None)
    p2.add_argument(
        "--output",
        action="store",
        type=str,
        help="Output file name.",
        default="onebyone.csv",
    )
    p2.add_argument("--verbose", "-v", action="count", default=0)
    p2.set_defaults(func=cmd_onebyone)

    # Parse args and pass to function
    args = parser.parse_args()
    args.func(args)
