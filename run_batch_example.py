"""
Example batch processing script for telluric correction.

This script demonstrates how to process multiple objects and configurations
using the refactored predict_abso pipeline.

Usage:
    python run_batch_example.py
"""

import os
from predict_abso_refactored import main
from predict_abso_config import get_batch_config, list_available_objects
import tellu_tools as tt
import json
from datetime import datetime


def save_batch_config(config: dict, output_dir: str):
    """
    Save batch configuration to JSON file for traceability.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    output_dir : str
        Directory to save configuration file
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = os.path.join(output_dir,
                               f"config_{config['batch_name']}_{timestamp}.json")

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to: {config_file}")
    return config_file


def example_1_single_object():
    """
    Example 1: Process a single object with default parameters.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Single object with default parameters")
    print("="*70 + "\n")

    main(
        batch_name='skypca_v5',
        instrument='NIRPS',
        obj='TOI4552',
        template_style='model'
    )


def example_2_multiple_template_styles():
    """
    Example 2: Process the same object with different template styles.

    This allows comparison between model templates and self-templates.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Multiple template styles for same object")
    print("="*70 + "\n")

    obj = 'TOI4552'
    instrument = 'NIRPS'
    batch_name = 'skypca_v5'

    for template_style in ['model', 'self']:
        print(f"\n>>> Processing with template_style='{template_style}'")

        try:
            main(
                batch_name=batch_name,
                instrument=instrument,
                obj=obj,
                template_style=template_style
            )
        except Exception as e:
            print(f"Error processing {obj} with {template_style}: {e}")
            continue


def example_3_multiple_objects():
    """
    Example 3: Process multiple objects in batch.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Multiple objects in batch")
    print("="*70 + "\n")

    instrument = 'NIRPS'
    batch_name = 'skypca_v5'
    template_style = 'model'

    # List of objects to process
    objects = ['TOI4552', 'TOI1234', 'HD189733']  # Replace with your objects

    # Get available objects from data directory
    project_path = tt.user_params()['project_path']
    available_objects = list_available_objects(instrument, project_path)

    print(f"Available objects: {available_objects}\n")

    # Process only objects that exist
    objects_to_process = [obj for obj in objects if obj in available_objects]

    if not objects_to_process:
        print("No valid objects to process!")
        return

    print(f"Processing {len(objects_to_process)} objects: {objects_to_process}\n")

    for obj in objects_to_process:
        print(f"\n{'*'*70}")
        print(f"*** Processing object: {obj}")
        print(f"{'*'*70}\n")

        try:
            main(
                batch_name=batch_name,
                instrument=instrument,
                obj=obj,
                template_style=template_style
            )
        except Exception as e:
            print(f"Error processing {obj}: {e}")
            continue


def example_4_custom_parameters():
    """
    Example 4: Process with custom parameters.

    Demonstrates how to modify default parameters for specific needs.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom parameters")
    print("="*70 + "\n")

    # Get standard configuration
    config = get_batch_config(
        batch_name='skypca_custom',
        instrument='NIRPS',
        obj='TOI4552',
        template_style='model'
    )

    # Customize parameters
    config['lowpass_filter_size'] = 151  # Increase smoothing
    config['sky_rejection_threshold'] = 0.8  # More conservative sky rejection
    config['dv_amp'] = 150  # Reduce velocity search range

    print("Custom configuration:")
    print("-" * 70)
    for key, value in config.items():
        if key in ['lowpass_filter_size', 'sky_rejection_threshold', 'dv_amp']:
            print(f"  {key}: {value} (MODIFIED)")
        else:
            print(f"  {key}: {value}")
    print("-" * 70 + "\n")

    # Save configuration for traceability
    project_path = tt.user_params()['project_path']
    output_dir = os.path.join(project_path, 'batch_configs')
    config_file = save_batch_config(config, output_dir)

    # Note: The current main() function doesn't accept custom config dict
    # This would require a small modification to main() to accept **config
    # For now, this demonstrates how to prepare custom configurations

    print("\nNote: To use custom parameters, modify main() to accept config dict")


def example_5_list_available_data():
    """
    Example 5: List all available objects for processing.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: List available data")
    print("="*70 + "\n")

    project_path = tt.user_params()['project_path']

    for instrument in ['NIRPS', 'SPIROU']:
        print(f"\n{instrument}:")
        print("-" * 40)

        objects = list_available_objects(instrument, project_path)

        if objects:
            for i, obj in enumerate(objects, 1):
                scidata_dir = os.path.join(project_path, f'scidata_{instrument}/{obj}')
                n_files = len([f for f in os.listdir(scidata_dir)
                              if f.endswith('.fits')])
                print(f"  {i:2d}. {obj:20s} ({n_files} files)")
        else:
            print("  No objects found")


def example_6_recovery_mode():
    """
    Example 6: Process only files that failed or were skipped.

    This is useful for resuming interrupted processing.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Recovery mode (process only missing files)")
    print("="*70 + "\n")

    instrument = 'NIRPS'
    obj = 'TOI4552'
    batch_name = 'skypca_v5'
    template_style = 'model'

    project_path = tt.user_params()['project_path']

    # Check which files have already been processed
    input_dir = os.path.join(project_path, f'scidata_{instrument}/{obj}')
    output_dir = os.path.join(project_path,
                              f'tellupatched_{instrument}/{obj}_{batch_name}_{template_style}/')

    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    input_files = [f for f in os.listdir(input_dir) if f.endswith('.fits')]

    if os.path.exists(output_dir):
        processed_files = [f.replace('tellupatched_t.fits', '.fits')
                          for f in os.listdir(output_dir)
                          if f.endswith('tellupatched_t.fits')]
    else:
        processed_files = []

    missing_files = set(input_files) - set(processed_files)

    print(f"Total input files: {len(input_files)}")
    print(f"Already processed: {len(processed_files)}")
    print(f"Missing/failed: {len(missing_files)}")

    if missing_files:
        print(f"\nMissing files:")
        for f in sorted(missing_files):
            print(f"  - {f}")

        # In recovery mode, you would process only these files
        # This requires modifying main() to accept a file list
        print("\nNote: Recovery mode requires main() modification to accept file list")
    else:
        print("\nAll files have been processed!")


def main_menu():
    """
    Interactive menu to select example.
    """
    examples = {
        '1': ('Single object with defaults', example_1_single_object),
        '2': ('Multiple template styles', example_2_multiple_template_styles),
        '3': ('Multiple objects', example_3_multiple_objects),
        '4': ('Custom parameters', example_4_custom_parameters),
        '5': ('List available data', example_5_list_available_data),
        '6': ('Recovery mode', example_6_recovery_mode),
    }

    print("\n" + "="*70)
    print("TELLURIC CORRECTION - BATCH PROCESSING EXAMPLES")
    print("="*70)
    print("\nSelect an example to run:\n")

    for key, (description, _) in examples.items():
        print(f"  {key}. {description}")

    print("\n  0. Run all examples (demonstration only)")
    print("  q. Quit")
    print("\n" + "="*70)

    choice = input("\nEnter your choice: ").strip()

    if choice == 'q':
        print("Exiting...")
        return

    if choice == '0':
        print("\nRunning all examples for demonstration...")
        # Run only non-processing examples
        example_5_list_available_data()
        example_4_custom_parameters()
        example_6_recovery_mode()
        print("\n" + "="*70)
        print("Demonstration complete!")
        print("To actually process data, run individual examples.")
        print("="*70 + "\n")
        return

    if choice in examples:
        _, example_func = examples[choice]
        example_func()
    else:
        print("Invalid choice!")


if __name__ == '__main__':
    # Run interactive menu
    main_menu()

    # Alternatively, uncomment one of the examples below to run directly:

    # example_1_single_object()
    # example_2_multiple_template_styles()
    # example_3_multiple_objects()
    # example_4_custom_parameters()
    # example_5_list_available_data()
    # example_6_recovery_mode()
