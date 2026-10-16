import joblib

# Importing the Path function from the pathlib module to handle file paths
from pathlib import Path

model_path = Path("models/ares_model.joblib")

# Checking if the model file exists at the specified path and raising an assertion error if it does not
assert model_path.exists(), f"Model not found: {model_path}"

# Loading the model artifact using joblib.load() and storing it in the variable 'bundle'
bundle = joblib.load(model_path)

# Printing the type of the loaded artifact to understand its structure
print("Loaded artifact type:", type(bundle))

# Checking if the loaded artifact is a dictionary, which indicates that it contains multiple components (like the model and other metadata). If it is a dictionary, we print the keys and extract the model. If not, we assume the loaded artifact is the model itself.
if isinstance(bundle, dict):
    # Print the keys of the dictionary
    print("Saved keys:", bundle.keys())

    model = bundle["model"]

else:
    model = bundle

print("Model type:", type(model))
print("Expected feature count:", model.n_features_in_)
