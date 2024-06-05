import parser
import nbformat as nbf


# parsing
args = parser.parse_arguments()
target_column = args.target_column
metric = args.metric
autoML = args.autoML
xai = args.xai
max_train_time = args.max_train_time
task = args.task
problem_type = args.problem_type
num_gpus = args.num_gpus
autogluon_preset = args.autogluon_preset
max_mem_size = args.max_mem_size


#assert dei parametri specificati
if num_gpus != 0:
    assert autoML == 'Autogluon'

if max_mem_size is not None:
    assert autoML == 'H2O'


nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {
    "name": "python3",
    "display_name": "Python 3",
    "language": "python"
}

markdown_cell = nbf.v4.new_markdown_cell("# Import delle librerie")
if autoML == 'Autogluon':
    autoML_import = f"""from autogluon.tabular import TabularDataset, TabularPredictor
"""
elif autoML == 'H2O':
    autoML_import = f"""import h2o
from h2o.automl import H2OAutoML
"""
else:
    autoML_import = f"""import autosklearn.classification
import pandas as pd
import pickle
import sklearn.metrics
"""

if xai == 'shap':
    xai_import = f"""import shap
import matplotlib.pyplot as plt
import pandas as pd
import math

def closestDivisors(n):
    a = round(math.sqrt(n))
    while n%a > 0: a -= 1
    return a,n//a
"""
else:
    xai_import = f"""import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def closestDivisors(n):
    a = round(math.sqrt(n))
    while n%a > 0: a -= 1
    return a,n//a"""
import_cell = autoML_import + xai_import


code_cell = nbf.v4.new_code_cell(import_cell)
nb.cells.append(markdown_cell)
nb.cells.append(code_cell)



if autoML == 'H2O':
    markdown_cell = nbf.v4.new_markdown_cell("# Avvia il server di H2O")
    code_cell = nbf.v4.new_code_cell(f"""h2o.init(
    ip = "localhost",
    port = 54321,
    start_h2o = True,
    max_mem_size="{max_mem_size}G",
    nthreads = -1)""")
    nb.cells.append(markdown_cell)
    nb.cells.append(code_cell)


markdown_cell = nbf.v4.new_markdown_cell("# Dataset")
if autoML == 'Autogluon':
    code_cell = nbf.v4.new_code_cell(f"""train_data = TabularDataset('Train.csv')
test_data = TabularDataset('Test.csv')

label = "{target_column}"
train_data[label].describe()""")
elif autoML == 'H2O':
    code_cell = nbf.v4.new_code_cell(f"""train = h2o.import_file(path='./Train.csv')
test = h2o.import_file(path='./Test.csv')

x = train.columns
y = "{target_column}"
x.remove(y)

train[y] = train[y].asfactor()
test[y] = test[y].asfactor()""")
else:
    code_cell = nbf.v4.new_code_cell(f"""train = pd.read_csv('Train.csv', header=0)
test = pd.read_csv('Test.csv', header=0)
train_y = train["{target_column}"]
train_y = train_y.to_numpy()
train = train.drop(columns=("{target_column}"))
train_X = train
train_X = train_X.to_numpy()

test_y = test["{target_column}"]
test_y = test_y.to_numpy()
test = test.drop(columns=("{target_column}"), axis=1)
test_X = test
test_X = test_X.to_numpy()""")
nb.cells.append(markdown_cell)
nb.cells.append(code_cell)




markdown_cell = nbf.v4.new_markdown_cell("# Preparazione dati")
code_cell = nbf.v4.new_code_cell("print('Hello, Jupyter!')")
nb.cells.append(markdown_cell)
nb.cells.append(code_cell)




markdown_cell = nbf.v4.new_markdown_cell("# Setup addestramento")
nb.cells.append(markdown_cell)

if autoML == 'Autogluon':
    code_cell = nbf.v4.new_code_cell(f"""save_path = 'agModels-predictClass'  # specifies folder to store trained models
predictor = TabularPredictor(label=label, problem_type='{problem_type}', path=save_path, eval_metric="{metric}") """)
elif autoML == 'H2O':
    code_cell = nbf.v4.new_code_cell(f"""aml = H2OAutoML(max_models=3, seed=1, max_runtime_secs_per_model={max_train_time})""")
else:
    code_cell = nbf.v4.new_code_cell(f"""automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task={max_train_time})
""")
nb.cells.append(code_cell)


markdown_cell = nbf.v4.new_markdown_cell("# Addestramento")
if autoML == 'Autogluon':
    code_cell = nbf.v4.new_code_cell(f"""predictor = predictor.fit(train_data, time_limit={max_train_time}, presets="{autogluon_preset}", ag_args_fit={{'num_gpus': {num_gpus}}})""")
elif autoML == 'H2O':
    code_cell = nbf.v4.new_code_cell(f"""aml.train(x=x, y=y, training_frame=train)""")
else:
    code_cell = nbf.v4.new_code_cell(f"""automl.fit(train_X, train_y, test_X, test_y)""")
nb.cells.append(markdown_cell)
nb.cells.append(code_cell)



markdown_cell = nbf.v4.new_markdown_cell("# Test modelli")
if autoML == 'Autogluon':
    code_cell = nbf.v4.new_code_cell(f"""predictor.evaluate(test_data, silent=True)""")
elif autoML == 'H2O':
    code_cell = nbf.v4.new_code_cell(f"""perf = aml.leader.model_performance(test)
perf""")
else:
    code_cell = nbf.v4.new_code_cell(f"""# Scegliendo di usare Autosklearn, il test viene fatto a fine fit""")
nb.cells.append(markdown_cell)
nb.cells.append(code_cell)


markdown_cell = nbf.v4.new_markdown_cell("# Salvataggio e caricamento del modello")
if autoML == 'Autogluon':
    code_cell = nbf.v4.new_code_cell(f"""""")
elif autoML == 'H2O':
    code_cell = nbf.v4.new_code_cell(f"""""")
else:
    code_cell = nbf.v4.new_code_cell(f"""# save model
with open('asklearn-gender-classifier.pkl', 'wb') as f:
    pickle.dump(automl, f)
    # load model
# with open('asklearn-gender-classifier.pkl', 'rb') as f:
#    automl = pickle.load(f)""")
nb.cells.append(markdown_cell)
nb.cells.append(code_cell)























markdown_cell = nbf.v4.new_markdown_cell(f"# Applica Explainability - {str(xai).upper()}")
if xai == 'shap' and autoML == 'Autogluon':
    code_cell = nbf.v4.new_code_cell(f"""def wrapped_model(x):
    column_names = [f'column_{{i}}' for i in range(train_data.shape[1]-1)]
    x = pd.DataFrame(x)
    x.columns = column_names
    preds = predictor.predict(x).to_numpy()
        
    return preds

test_data = TabularDataset('Test.csv')

to_be_explained = pd.DataFrame(test_data).drop(label, axis=1).to_numpy()[0]
explainer = shap.KernelExplainer(wrapped_model, pd.read_csv('./Train.csv').drop(label, axis=1).sample(n=100))
shap_values = explainer.shap_values(to_be_explained)
relevance = abs(shap_values.ravel())


norm_relevance = ((relevance - min(relevance)) / (max(relevance) - min(relevance)))

print(relevance)
plt.imshow(norm_relevance.reshape(closestDivisors(train_data.shape[1]-1)))
plt.colorbar()""")
elif xai == 'lime' and autoML == 'Autogluon':
    code_cell = nbf.v4.new_code_cell(f"""def wrapped_net(x):
    column_names = [f'column_{{i}}' for i in range(train_data.shape[1]-1)]
    x = pd.DataFrame(x)
    x.columns = column_names
    preds = predictor.predict_proba(x).to_numpy()
        
    return preds

background = pd.read_csv('./Train.csv').drop("{target_column}", axis=1).sample(n=100).to_numpy()
explainer = lime.lime_tabular.LimeTabularExplainer(
    background,
    feature_names=[str(i) for i in range(train_data.shape[1]-1)],
    verbose=True,
    mode='classification',
)

test = pd.read_csv('./Train.csv').drop("{target_column}", axis=1).to_numpy()[0]

exp = explainer.explain_instance(test, wrapped_net, num_features=train_data.shape[1]-1)
# exp.save_to_file('lime_explanationall.html')
# relevance = abs(np.asarray([float(i) for i in exp.domain_mapper.feature_values]))
relevance = abs(np.asarray([j for i, j in sorted(exp.local_exp[1], key=lambda i: i[0])]))


perc_relevance = 100 * relevance / sum(relevance)
# relevance = exp.local_exp[1]
norm_relevance = ((relevance - min(relevance)) / (max(relevance) - min(relevance)))


print(relevance)
plt.imshow(norm_relevance.reshape(closestDivisors(train_data.shape[1]-1)))
plt.colorbar()""")
elif xai == 'shap' and autoML == 'H2O':
    code_cell = nbf.v4.new_code_cell(f"""
def wrapped_model(x):
    column_names = [f'column_{{i}}' for i in range(train.shape[1]-1)]
    x = h2o.H2OFrame(x)
    x.col_names = column_names
    
    try:
        preds = aml.leader.predict(x).as_data_frame().to_numpy()[:, 0]
    except:
        preds = aml.predict(x).as_data_frame().to_numpy()[:, 0]
        
    return preds

to_be_explained = test.as_data_frame()[:1].drop("{target_column}", axis=1).to_numpy()
explainer = shap.KernelExplainer(wrapped_model, pd.read_csv('./Train.csv').drop("{target_column}", axis=1).sample(n=100))
shap_values = explainer.shap_values(to_be_explained)
relevance = abs(shap_values[0].ravel())


norm_relevance = ((relevance - min(relevance)) / (max(relevance) - min(relevance)))

print(relevance)
plt.imshow(norm_relevance.reshape(closestDivisors(train.shape[1]-1)))
plt.colorbar()""")
elif xai == 'lime' and autoML == 'H2O':
    code_cell = nbf.v4.new_code_cell(f"""def wrapped_net(x):
    column_names = [f'column_{{i}}' for i in range(train.shape[1]-1)]
    x = h2o.H2OFrame(x)
    x.col_names = column_names
    
    try:
        preds = aml.leader.predict(x).as_data_frame().to_numpy()[:, 1:]
    except:
        preds = aml.predict(x).as_data_frame().to_numpy()[:, 1:]
    return preds

background = pd.read_csv('./Train.csv').drop("{target_column}", axis=1).sample(n=100).to_numpy()
explainer = lime.lime_tabular.LimeTabularExplainer(
    background,
    feature_names=[str(i) for i in range(train.shape[1]-1)],
    verbose=True,
    mode='{task}',
)

test = pd.read_csv('./Test.csv').drop("{target_column}", axis=1).to_numpy()[0]

exp = explainer.explain_instance(test, wrapped_net, num_features=train.shape[1]-1)
# exp.save_to_file('lime_explanationall.html')
# relevance = abs(np.asarray([float(i) for i in exp.domain_mapper.feature_values]))
relevance = abs(np.asarray([j for i, j in sorted(exp.local_exp[1], key=lambda i: i[0])]))
# relevance = exp.local_exp[1]
norm_relevance = ((relevance - min(relevance)) / (max(relevance) - min(relevance)))


print(relevance)
plt.imshow(norm_relevance.reshape(closestDivisors(train.shape[1]-1)))
plt.colorbar()""")
elif xai == 'shap' and autoML == 'Autosklearn':
    code_cell = nbf.v4.new_code_cell(f"""def wrapped_model(x):
    column_names = [f'column_{{i}}' for i in range(train_X.shape[1])]
    x = pd.DataFrame(x)
    x.columns = column_names
    preds = automl.predict(x.to_numpy())
        
    return preds

test_data = pd.read_csv('Test.csv', header=0)

to_be_explained = test_data.drop("{target_column}", axis=1).to_numpy()[0]
explainer = shap.KernelExplainer(wrapped_model, pd.read_csv('./Train.csv', header=0).drop("{target_column}", axis=1).sample(n=100))
shap_values = explainer.shap_values(to_be_explained)
relevance = abs(shap_values.ravel())


norm_relevance = ((relevance - min(relevance)) / (max(relevance) - min(relevance)))

print(relevance)
plt.imshow(norm_relevance.reshape(train_X.shape[1]))
plt.colorbar()""")
elif xai == 'lime' and autoML == 'Autosklearn':
    code_cell = nbf.v4.new_code_cell(f"""def wrapped_net(x):
    column_names = [f'column_{{i}}' for i in range(train_X.shape[1])]
    x = pd.DataFrame(x)
    x.columns = column_names
    preds = automl.predict_proba(x.to_numpy())
        
    return preds

background = pd.read_csv('./Train.csv').drop("{target_column}", axis=1).sample(n=100).to_numpy()
explainer = lime.lime_tabular.LimeTabularExplainer(
    background,
    feature_names=[str(i) for i in range(train_X.shape[1])],
    verbose=True,
    mode='classification',
)

test = pd.read_csv('./Test.csv').drop("{target_column}", axis=1).to_numpy()[0]

exp = explainer.explain_instance(test, wrapped_net, num_features=train_X.shape[1])
# exp.save_to_file('lime_explanationall.html')
# relevance = abs(np.asarray([float(i) for i in exp.domain_mapper.feature_values]))
relevance = abs(np.asarray([j for i, j in sorted(exp.local_exp[1], key=lambda i: i[0])]))
# relevance = exp.local_exp[1]
norm_relevance = ((relevance - min(relevance)) / (max(relevance) - min(relevance)))


print(relevance)
plt.imshow(norm_relevance.reshape(closestDivisors(train_X.shape[1])))
plt.colorbar()""")
nb.cells.append(markdown_cell)
nb.cells.append(code_cell)



with open('notebook.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook creato con successo!")

