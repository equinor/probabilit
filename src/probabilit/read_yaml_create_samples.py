from probabilit.modeling import Distribution, Add
import yaml
from yaml.loader import Loader
import pandas as pd

file = "config.yml"


# Load data from file into a dictionary
with open(file, "r") as file_handle:
    yaml_data = yaml.load(file_handle, Loader)
    
# Load sections of the dictionary as variables
metadata = yaml_data["metadata"]
variables = yaml_data["variables"]

# Convert dict of {name:data, ...} to {name:Distribution, ...}
for variable_name in variables.keys():
    variable_data = variables[variable_name]
    variables[variable_name] = Distribution(variable_data["type"],
                            **variable_data["parameters"])
    
    
# Dummy expression to sample all parents
expression = Add(*variables.values())
expression.sample(size=metadata["samples"])

# Collect all samples into a dataframe
df_samples = pd.DataFrame({name:distr.samples_ for (name, distr) in variables.items()})

print(df_samples)






    
    
    