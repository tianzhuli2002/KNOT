import h5py
import logging
import numpy as np

logger = logging.getLogger(__name__)

def build_observable_specs(observables) -> list[dict]:
    observable_specs = []
    for observable in observables:
        logger.debug(f"Processing observable: {observable}")
        observable_specs.append(dict(observable))
    return observable_specs

def parse_observable_spec_to_feature_keys(observable_specs: list[dict]) -> list[str]:
    feature_keys = []
    for ob_spec in observable_specs:
        logger.debug(f"Processing observable spec: {ob_spec}")
        features = ob_spec.get("INPUTS", None)
        if features is None:
            raise ValueError(f"Observable spec missing 'INPUTS' key: {ob_spec}")
        features = f"INPUTS:{features}"
        # Check variable(s)
        variables = ob_spec.get("variables", None)
        if variables is None:
            variables = ob_spec.get("variable", None)
            if variables is None:
                raise ValueError(f"Observable spec 'INPUTS' missing 'variables' or 'variable' key: {ob_spec}")
            variables = [variables]
        features = [f"{features}:{var}" for var in variables]
        # Finally check the number of particles
        num_particles = ob_spec.get("num_particles", None)
        if num_particles is None:
            logger.debug(f"Observable spec 'INPUTS' missing 'num_particles' key as global variable: {ob_spec}")
            feature_keys += features
        else:
            if num_particles <= 0:
                raise ValueError(f"Observable spec 'INPUTS' has non-positive 'num_particles': {ob_spec}")
            feature_keys += [f"{feat}:{i}" for feat in features for i in range(num_particles)]
    return feature_keys

def parse_feature_key_to_observable_spec(feature_key: str, observable_specs: list[dict]) -> dict:
    parts = feature_key.split(':')
    if len(parts) < 3:
        raise ValueError(f"Feature key '{feature_key}' does not have enough parts to parse into observable spec")
    for obs_spec in observable_specs:
        found = False
        if obs_spec.get("INPUTS", None) == parts[1]:
            found = True
            return obs_spec
    if not found:
        raise ValueError(f"Feature key '{feature_key}' does not match any observable spec in the provided list")

def get_transform_from_feature_key_and_spec(feature_key: str, obs_spec: dict) -> str:
    variables = obs_spec.get("variables", None)
    if variables is None:
        variables = obs_spec.get("variable", None)
        if variables is None:
            raise ValueError(f"Observable spec missing 'variables' or 'variable' key: {obs_spec}")
        variables = [variables]
    scalers = obs_spec.get("scalers", None)
    if scalers is None:
        logger.error(f"Observable spec missing 'scalers' key: {obs_spec}. If no scalers are needed, plesae build a vector of ['None', 'None', ...] with the same length as 'variables' and include it in the spec.")
        raise ValueError(f"Observable spec missing 'scalers' key: {obs_spec}")
    if len(variables) != len(scalers):
        logger.error(f"Observable spec has different number of 'variables' and 'scalers': {obs_spec}. If a variable does not require scaling, please set its scaler to 'None' in the spec.")
        raise ValueError(f"Observable spec has different number of 'variables' and 'scalers': {obs_spec}")

    # Get index of variable in feature key
    var_name = feature_key.split(':')[2]
    var_idx = None
    for i, var in enumerate(variables):
        if var == var_name:
            var_idx = i
            break
    if var_idx is None:
        raise ValueError(f"Variable name '{var_name}' from feature key '{feature_key}' not found in observable spec variables: {obs_spec}")
    return scalers[var_idx]

def get_feature_from_h5(file_path: str, feature_key: str, filter_indices: np.ndarray = None) -> np.ndarray:
    with h5py.File(file_path, 'r') as f:
        keys = feature_key.split(':')
        # Convert the keys that can be converted from strings to integers
        keys = [int(key) if key.isdigit() else key for key in keys]
        data = None
        for key in keys:
            logger.debug(f"Accessing key '{key}' in HDF5 file for feature '{feature_key}'")
            logger.debug(data)
            try:
                # Check if the key can be converted to an integer index (for particle indexing)
                if data is None:
                    data = f[key] if isinstance(key, str) else f[:,key]
                else:
                    data = data[key] if isinstance(key, str) else data[:,key]
            except (KeyError, ValueError, IndexError) as exc:
                raise KeyError(f"Key '{key}' not found in HDF5 file at path '{feature_key}'") from exc
        if filter_indices is not None:
            data = data[filter_indices]
        return data[:]

def get_sb_labels(file_path: str) -> np.ndarray:
    labels = get_feature_from_h5(file_path, "LABELS:GLOBAL")
    return labels.reshape(-1)
