"""
Import the model and make example prediction

"""
from joblib import load
import pandas as pd


def restore_model():
    """
    return model object
    """
    # Restore model
    with open("models/final_model.joblib", "rb") as f:

        # we'll load the file here
        model = load(f)

    return model


if __name__ == "__main__":

    restored_model = restore_model()

    obs = pd.DataFrame(
        [[8.3252, 41., 6.98412698, 1.02380952, 322.,
          2.55555556, 37.88, -122.23]],
        columns=restored_model.feature_names_in_
    )

    print(
        restored_model.predict(obs)
    )
