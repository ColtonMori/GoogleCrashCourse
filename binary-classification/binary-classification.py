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

feature_mean = rice_dataset.mean(numeric_only=True)
feature_std = rice_dataset.std(numeric_only=True)
numerical_features = rice_dataset.select_dtypes('number').columns
normalized_dataset = (
    rice_dataset[numerical_features] - feature_mean
) / feature_std

normalized_dataset['Class'] = rice_dataset['Class']

print(normalized_dataset.head())

keras.utils.set_random_seed(42)

normalized_dataset['Class_Bool'] = (
    normalized_dataset['Class'] == 'Cammeo'
).astype(int)
normalized_dataset.sample(10)

number_samples = len(normalized_dataset)
index_80th = round(number_samples * 0.8)
index_90th = index_80th + round(number_samples * 0.1)

shuffled_dataset = normalized_dataset.sample(frac=1, random_state=100)
train_data = shuffled_dataset.iloc[0:index_80th]
validation_data = shuffled_dataset.iloc[index_80th:index_90th]
test_data = shuffled_dataset.iloc[index_90th:]

print(test_data.head())

label_columns = ['Class', 'Class_Bool']

train_featurse = train_data.drop(columns=label_columns)
train_labels = train_data['Class_Bool'].to_numpy()
validation_labels = validation_data.drop(columns=label_columns)
validation_labels = validation_data['Class_Bool'].to_numpy()
test_features = test_data.drop(columns=label_columns)
test_labels = test_data['Class_Bool'].to_numpy()

input_features = [
    'Eccentricity',
    'Major_Axis_Length',
    'Area',
]

def create_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
    model_inputs = [
        keras.Input(name=feature, shape=(1,))
        for feature in settings.input_features
    ]

    concatenated_inputs = keras.layers.Concatenate()(model_inputs)
    model_output = keras.layers.Dense(
        units = 1, name='dense_layer', activation=keras.activations.sigmoid
    )(concatenated_inputs)
    model = keras.Model(inputs=model_inputs, outputs=model_output)
    model.compile(
        optimizer=keras.optimizers.RMSprop(settings.learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )
    return model

def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    labels: np.ndarray,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
    features = {
        feature_name: np.array(dataset[feature_name])
        for feature_name in settings.input_features
    }

    history = model.fit(
        x=features,
        y=labels,
        batch_size=settings.batch_size,
        epochs=settings.number_epochs,
    )
    return ml_edu.experiment.Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history)
    )

print('Defined the create_model and train_model function')

settings = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.001,
    number_epochs=60,
    batch_size=100,
    classification_threshold=0.35,
    input_features=input_features,
)

metrics = [
    keras.metrics.BinaryAccuracy(
        name='accuracy', threshold=settings.classification_threshold
    ),
    keras.metrics.Precision(
        name='precision', thresholds=settings.classification_threshold
    ),
    keras.metrics.Recall(
        name='recall', thresholds=settings.classification_threshold
    ),
    keras.metrics.AUC(name='auc', curve='ROC', num_thresholds=200),
]

model = create_model(settings, metrics)

experiment = train_model(
    'baseline', model, train_featurse, train_labels, settings
)

ml_edu.results.plot_experiment_metrics(experiment, ['accuracy', 'precision', 'recall'])
plt.savefig("Accuracy_Precision_Recall.png")
ml_edu.results.plot_experiment_metrics(experiment, ['auc'])
plt.savefig("Auc.png")

def compare_train_test(experiment: ml_edu.experiment.Experiment, test_metrics: dict[str, float]):
    print('Comparing metrics between train and test:')
    for metric, test_value in test_metrics.items():
        print('------')
        print(f'Train {metric}: {experiment.get_final_metric_value(metric):.4f}')
        print(f'Test {metric}: {test_value:.4f}')

test_metrics = experiment.evaluate(test_features, test_labels)
compare_train_test(experiment, test_metrics)

all_input_features = [
    'Eccentricity',
    'Major_Axis_Length',
    'Minor_Axis_Length',
    'Area',
    'Convex_Area',
    'Perimeter',
    'Extent',
]

settings_all_features = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.001,
    number_epochs=60,
    batch_size=100,
    classification_threshold=0.5,
    input_features=all_input_features,
)

metrics_all_features = [
    keras.metrics.BinaryAccuracy(
        name='accuracy', threshold=settings_all_features.classification_threshold
    ),
    keras.metrics.Precision(
        name='precision', thresholds=settings_all_features.classification_threshold
    ),
    keras.metrics.Recall(
        name='recall', thresholds=settings_all_features.classification_threshold
    ),
    keras.metrics.AUC(name='auc', curve='ROC', num_thresholds=200),
]

model_all_features = create_model(settings_all_features, metrics_all_features)

experiment_all_features = train_model(
    'all_features',
    model_all_features,
    train_featurse,
    train_labels,
    settings_all_features,
)

ml_edu.results.plot_experiment_metrics(
    experiment_all_features, ['accuracy', 'precision', 'recall']
)
plt.savefig("Accuracy_Precision_Recall_all_features.png")
ml_edu.results.plot_experiment_metrics(experiment_all_features, ['auc'])
plt.savefig("Auc_all_features.png")

test_metrics_all_features = experiment_all_features.evaluate(
    test_features, test_labels
)
compare_train_test(experiment_all_features, test_metrics_all_features)

ml_edu.results.compare_experiment([experiment, experiment_all_features],
                                  ['accuracy', 'auc'],
                                  test_features, test_labels
)
plt.savefig("Compare_Experiment.png")