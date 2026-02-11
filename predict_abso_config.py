"""
Configuration file for predict_abso batch processing.

This module defines batch configurations for telluric correction processing.
Each batch configuration is a dictionary with all necessary parameters.
"""

from typing import Dict, List, Any
import os
import yaml


# Default processing parameters
DEFAULT_PARAMS = {
    'molecules': ['H2O', 'CH4', 'CO2', 'O2'],
    'lowpass_filter_size': 101,
    'template_ratio_threshold_high': 3.0,
    'template_ratio_threshold_low': 0.3,
    'template_smooth_window': 501,
    'min_valid_ratio': 0.1,
    'low_flux_threshold': 0.2,
    'sky_rejection_threshold': 1.0,  # Reject pixels where sky/sp_corr > this value
    'dv_amp': 200,  # km/s, velocity search range
}


def _load_machine_config() -> Dict[str, Any]:
    """Load machine-specific config from batch_config.yaml."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'batch_config.yaml')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        machines = config.get('machines', {})
        for machine_name, machine_config in machines.items():
            detect_path = machine_config.get('detect_path', '')
            if detect_path and os.path.exists(detect_path):
                return machine_config
    
    return {}


def _load_telluric_config() -> Dict[str, Any]:
    """Load telluric config from telluric_config.yaml."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'telluric_config.yaml')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    return {}


def get_batch_config(batch_name: str, instrument: str, obj: str,
                     template_style: str = 'model') -> Dict[str, Any]:
    """
    Generate a batch configuration dictionary.

    Parameters
    ----------
    batch_name : str
        Name identifier for this batch (e.g., 'skypca_v5')
    instrument : str
        Instrument name ('NIRPS' or 'SPIROU')
    obj : str
        Object name (e.g., 'TOI4552')
    template_style : str
        Template type: 'model' or 'self'

    Returns
    -------
    config : dict
        Configuration dictionary with all processing parameters
    """

    # Validate inputs
    if instrument not in ['NIRPS', 'SPIROU']:
        raise ValueError(f"Unknown instrument: {instrument}. Must be 'NIRPS' or 'SPIROU'")

    if template_style not in ['model', 'self']:
        raise ValueError(f"Unknown template_style: {template_style}. Must be 'model' or 'self'")

    # Load machine-specific settings (doplot, n_cores)
    machine_config = _load_machine_config()
    doplot = machine_config.get('doplot', False)
    n_cores = machine_config.get('n_cores', 1)

    # Load telluric config for demo_order and processing options
    telluric_config = _load_telluric_config()
    demo_orders = telluric_config.get('demo_order', {'NIRPS': 55, 'SPIROU': 35})
    demo_order = demo_orders.get(instrument, 55)
    
    # Get randomize_files from processing section (default True)
    processing_config = telluric_config.get('processing', {})
    randomize_files = processing_config.get('randomize_files', True)

    # Build configuration
    config = {
        'batch_name': f"{batch_name}_{template_style}",
        'instrument': instrument,
        'object': obj,
        'template_style': template_style,
        'doplot': doplot,
        'n_cores': n_cores,
        'demo_order': demo_order,
        'randomize_files': randomize_files,
        **DEFAULT_PARAMS  # Include all default parameters
    }

    return config


def list_available_objects(instrument: str, project_path: str) -> List[str]:
    """
    List all available objects for a given instrument.

    Parameters
    ----------
    instrument : str
        Instrument name
    project_path : str
        Root project path

    Returns
    -------
    objects : list
        List of object names found in the scidata directory
    """
    import glob

    scidata_dir = os.path.join(project_path, f'scidata_{instrument}')
    if not os.path.exists(scidata_dir):
        return []

    # Find subdirectories (each is an object)
    objects = [d for d in os.listdir(scidata_dir)
               if os.path.isdir(os.path.join(scidata_dir, d))]

    return sorted(objects)


# Example batch configurations
EXAMPLE_BATCHES = {
    'toi4552_v5': {
        'instrument': 'NIRPS',
        'object': 'TOI4552',
        'batch_name': 'skypca_v5',
        'template_style': 'model',
    },
    'toi4552_self': {
        'instrument': 'NIRPS',
        'object': 'TOI4552',
        'batch_name': 'skypca_v5',
        'template_style': 'self',
    },
}


def get_example_batch(name: str) -> Dict[str, Any]:
    """
    Get a pre-defined example batch configuration.

    Parameters
    ----------
    name : str
        Name of the example batch

    Returns
    -------
    config : dict
        Full configuration dictionary
    """
    if name not in EXAMPLE_BATCHES:
        raise ValueError(f"Unknown example batch: {name}. Available: {list(EXAMPLE_BATCHES.keys())}")

    params = EXAMPLE_BATCHES[name]
    return get_batch_config(
        batch_name=params['batch_name'],
        instrument=params['instrument'],
        obj=params['object'],
        template_style=params['template_style']
    )


if __name__ == '__main__':
    # Example usage
    config = get_batch_config(
        batch_name='skypca_v5',
        instrument='NIRPS',
        obj='TOI4552',
        template_style='model'
    )

    print("Example configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
