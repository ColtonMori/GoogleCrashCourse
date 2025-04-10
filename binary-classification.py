import io
import keras
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import ml_edu.experiment
import ml_edu.results
import numpy as np
import pandas as pd
import plotly.express as px

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

rice_dataset_raw = pd.read_csv("Rice_Cammeo_Osmancik.csv")

rice_dataset = rice_dataset_raw[[
    'Area',
    'Perimeter',
    'Major_Axis_Length',
    'Minor_Axis_Length',
    'Eccentricity',
    'Convex_Area',
    'Extent',
    'Class'
]]

print(rice_dataset.describe())

print(
    f'The shortest grain is {rice_dataset.Major_Axis_Length.min():.1f}px long,'
    f' while the longest grain is {rice_dataset.Major_Axis_Length.max():.1f}px'
)
print(
    f'The smallest rice grain has an area of {rice_dataset.Area.min()}px, while'
    f' the largest rice grain has an area of {rice_dataset.Area.max()}px'
)
print(
    'The largest rice grain, with a perimeter of'
    f' {rice_dataset.Perimeter.max():.1f}px, is'
    f' ~{(rice_dataset.Perimeter.max() - rice_dataset.Perimeter.mean())/rice_dataset.Perimeter.std():.1f} standard'
    f' deviations ({rice_dataset.Perimeter.std():.1f}) from the mean'
    f' ({rice_dataset.Perimeter.mean():.1f}px).'
)
print(
    f'This is calculated as: ({rice_dataset.Perimeter.max():.1f} - '
    f' {rice_dataset.Perimeter.mean():.1f}) / {rice_dataset.Perimeter.std():.1f} = '
    f' {(rice_dataset.Perimeter.max() - rice_dataset.Perimeter.mean())/rice_dataset.Perimeter.std():.1f}'
)

for x_axis_data, y_axis_data in [
    ('Area', 'Eccentricity'),
    ('Convex_Area', 'Perimeter'),
    ('Major_Axis_Length', "Minor_Axis_Length"),
    ('Perimeter', 'Extent'),
    ('Eccentricity', 'Major_Axis_Length'),
]:
    fig = px.scatter(
        rice_dataset,
        x=x_axis_data,
        y=y_axis_data,
        color='Class',
        title=f'{x_axis_data} vs {y_axis_data}',
        labels={x_axis_data: x_axis_data, y_axis_data: y_axis_data},
    )
    fig.write_image(f"{x_axis_data}_vs_{y_axis_data}.png")

import plotly.io as pio

pio.write_html(
    px.scatter_3d(
        rice_dataset,
        x='Eccentricity',
        y='Area',
        z='Major_Axis_Length',
        color='Class',
    ),
    file="Eccentricity_Area_Major_Axis_Length.html",
    auto_open=False
)

feature_mean = rice_dataset.mean(numeric_only=True)
feature_std = rice_dataset.std(numeric_only=True)
numerical_features = rice_dataset.select_dtypes('number').columns
normalized_dataset = (
    rice_dataset[numerical_features] - feature_mean
) / feature_std

normalized_dataset['Class'] = rice_dataset['Class']

print(normalized_dataset.head())