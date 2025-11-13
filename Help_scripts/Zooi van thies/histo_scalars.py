"""
SCALAR FEATURE VISUALIZER
This file creates plots comparing scalar features across different shapes.
It shows how features like sphericity, convexity, etc. differ between shapes.
It highlights specific example shapes (round, flat, elongated, irregular).
This helps understand what the scalar features actually measure.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
CSV_FILE_PATH = '../../Feature-matrix/all_features.csv'  # <-- IMPORTANT: Update with the path to your CSV file
SCALARS_TO_PLOT = [
    'Surface area', 'Sphericity', 'Rectangularity',
    'Diameter', 'Convexity', 'Eccentricity'
]
# Define shapes by Category, Object, and a descriptive Tag
SHAPES_TO_INCLUDE = [
    {'Category': 'AircraftBuoyant', 'Object': 'm1338_rm.obj', 'Tag': 'round'},
    {'Category': 'Cellphone', 'Object': 'D00192_rm.obj', 'Tag': 'flat'},
    {'Category': 'Bottle', 'Object': 'D00166_rm.obj', 'Tag': 'elongated'},
    {'Category': 'Quadruped', 'Object': 'D00226_rm.obj', 'Tag': 'irregular'}
]
# Define columns for category and object identifiers
CATEGORY_COLUMN = 'Category'
OBJECT_COLUMN = 'Object'
# The new column that will be created for plotting labels
SHAPE_LABEL_COLUMN = 'Shape Label'


# --- Script ---
try:
    # 1. Read the data from the CSV file
    df = pd.read_csv(CSV_FILE_PATH)

    # 2. Clean data by stripping whitespace from key columns
    df[CATEGORY_COLUMN] = df[CATEGORY_COLUMN].str.strip()
    df[OBJECT_COLUMN] = df[OBJECT_COLUMN].str.strip()

    # 3. Filter the DataFrame to include only the specified shapes
    conditions = [(df[CATEGORY_COLUMN] == shape['Category']) & (df[OBJECT_COLUMN] == shape['Object']) for shape in SHAPES_TO_INCLUDE]
    combined_condition = pd.concat(conditions, axis=1).any(axis=1)
    df_filtered = df[combined_condition].copy()

    # 4. Check if filtering resulted in an empty DataFrame
    if df_filtered.empty:
        print("Warning: The filtering resulted in an empty dataset.")
        print("Please check if the Category/Object pairs in 'SHAPES_TO_INCLUDE' exist in the CSV file.")
    else:
        # 5. Create a new column for clear labels on the plot, including the tag
        # Create a mapping from category to tag for easy lookup
        tag_map = {shape['Category']: shape['Tag'] for shape in SHAPES_TO_INCLUDE}
        # Apply the mapping to create the new label column
        df_filtered[SHAPE_LABEL_COLUMN] = (
            df_filtered[CATEGORY_COLUMN] + ' ' +
            df_filtered[OBJECT_COLUMN] + ' (' +
            df_filtered[CATEGORY_COLUMN].map(tag_map) + ')'
        )

        # 6. Calculate the average value for each scalar, grouped by the new shape label
        plot_data = df_filtered.groupby(SHAPE_LABEL_COLUMN)[SCALARS_TO_PLOT].mean()

        # 7. Transpose the DataFrame to swap axes (scalars on x-axis)
        plot_data_transposed = plot_data.T

        # 8. Plot the transposed data
        ax = plot_data_transposed.plot(
            kind='bar',
            figsize=(14, 8), # Increased height slightly for better legend spacing
            rot=0,  # Keeps the scalar labels horizontal
            width=0.8
        )

        # 9. Customize the plot for better readability
        plt.title('Comparison of Shape Scalars', fontsize=16)
        plt.xlabel('Scalar', fontsize=12)
        plt.ylabel('Mean Value', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Shape (Tag)')
        plt.tight_layout()

        # 10. Display the plot
        plt.show()

except FileNotFoundError:
    print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
except KeyError as e:
    print(f"Error: A required column was not found in the CSV: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")