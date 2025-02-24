import kagglehub
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from google.colab import drive
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, precision_recall_curve, roc_curve, roc_auc_score, make_scorer, auc

# ignorar warnings
warnings.filterwarnings('ignore')

# LEITURA DA BASE DE DADOS
path = kagglehub.dataset_download("rajyellow46/wine-quality", path="winequalityN.csv")
wine_quality = pd.read_csv(path);

# TRATAMENTO DOS NOMES DAS COLUNAS
wine_quality.columns = wine_quality.columns.str.lower().str.replace(' ', '_').str.replace('(','').str.replace(')','')

# SETANDO O TARGET
target = 'opinion'

# RETIRANDO OS VALORES NULOS
wine_quality = wine_quality.dropna()

# SETANDO A VARIÁVEL 'opinion' EM 1 CASO A VARIÁVEL 'quality' SEJA MAIOR QUE 5, CASO CONTRÁRIO, SERÁ 0
wine_quality[target] = wine_quality['quality'].apply(lambda r: 0 if r <= 5 else 1)

# SETANDO A VARIÁVEL 'type' EM 0 CASO O TIPO DO VINHO NÃO SEJA BRANCO, E CASO CONTRÁRIO, SERÁ 1
wine_quality['type'] = wine_quality['type'].apply(lambda r: 1 if r == 'white' else 0)

# DESCONSIDERANDO A VARIÁVEL 'quality'
wine_quality = wine_quality.drop(columns=['quality'])

# PEGANDO APENAS OS VINHOS BRANCOS
wine_quality_type_white = wine_quality[wine_quality['type'] == 1]

# PEGANDO APENAS OS VINHOS TINTOS
wine_quality_red_wine = wine_quality[wine_quality['type'] == 0]

# SETANDO O X E y
X = wine_quality_type_white.drop(columns=[target])
y = wine_quality_type_white[[target]]

# SEPARANDO AS VARÍAVEIS NUMÉRICAS E CATEGÓRICAS
num = ['fixed_acidity', 'volatile_acidity', 'citric_acid','residual_sugar', 'chlorides',
       'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'ph', 'sulphates', 'alcohol']
cat = [col for col in X.columns if col not in num]

# MÉDIAS E DESVIOS PADRÕES
mean_std_white = wine_quality_type_white.describe().loc[['mean', 'std']]
mean_std_white

preprocessor = ColumnTransformer([
    ('category', OneHotEncoder(drop='first', handle_unknown='ignore'), cat),
    ('numeric', RobustScaler(), num)
])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=2)

splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

reglog = Pipeline([
    ('transformer', preprocessor),
    ('selector', SelectKBest(f_classif)),
    ('model', LogisticRegression(solver='saga'))
])

# configurar o espaço de busca
params_grid_reglog = {
    'model__penalty': ['l1', 'l2', 'elasticnet', None],
    'model__C': [0.001, 0.01, 0.1, 1, 10],
    'model__l1_ratio': [0.25, 0.5, 0.75],
    'model__class_weight': ['balanced', None],
    'selector__k': range(1, 11)
}

search_reglog = RandomizedSearchCV(
    estimator=reglog,
    param_distributions=params_grid_reglog,
    n_iter=70,
    scoring='f1',
    cv=splitter,
    refit=True,
    error_score=0,
    verbose=10
)

# realizar a busca
search_reglog.fit(x_train, y_train)

yhat_reglog_train = search_reglog.best_estimator_.predict(x_train)
print('Desempenho - Base de Treino')
print(classification_report(y_train, yhat_reglog_train))

yhat_reglog_test = search_reglog.best_estimator_.predict(x_test)
print('Desempenho - Base de Teste')
print(classification_report(y_test, yhat_reglog_test))

dt = Pipeline([
    ('transformer', preprocessor),
    ('selector', SelectKBest(f_classif)),
    ('model', DecisionTreeClassifier(random_state=2))
])

params_grid_dt = {
    'model__max_depth': range(2, 11),
    'model__criterion': ['gini', 'entropy'],
    'model__class_weight': ['balanced', None],
    'selector__k': range(1, 11)
}

search_dt = RandomizedSearchCV(
    estimator=dt,
    param_distributions=params_grid_dt,
    n_iter=70,
    scoring='f1',
    cv=splitter,
    refit=True,
    error_score=0,
    verbose=10
)


search_dt.fit(x_train, y_train)

yhat_dt_train = search_dt.best_estimator_.predict(x_train)
print('Desempenho - Base de Treino')
print(classification_report(y_train, yhat_dt_train))

yhat_dt_test = search_dt.best_estimator_.predict(x_test)
print('Desempenho - Base de Teste')
print(classification_report(y_test, yhat_dt_test))

svc = Pipeline([
    ('transformer', preprocessor),
    ('selector', SelectKBest(f_classif)),
    ('svc', SVC(random_state=2))
])

params_grid_svc = {
    'selector__k': range(1, 11)
}

search_svc = RandomizedSearchCV(
    estimator=svc,
    param_distributions=params_grid_svc,
    n_iter=50,
    scoring='f1',
    cv=splitter,
    refit=True,
    error_score=0,
    verbose=10
)

search_svc.fit(x_train, y_train)

yhat_svc_train = search_svc.best_estimator_.predict(x_train)
print('Desempenho - Base de Treino')
print(classification_report(y_train, yhat_svc_train))

yhat_svc_test = search_svc.best_estimator_.predict(x_test)
print('Desempenho - Base de Teste')
print(classification_report(y_test, yhat_svc_test))

"""Verificando a importância dos atributos"""

x_num  = X[num]
# criando os pipelines - apenas para aprendizado
dt_feat_imp = Pipeline([
    ('scaler', RobustScaler()),
    ('model', DecisionTreeClassifier(random_state=2))
])
reglog_feat_imp = Pipeline([
    ('scaler', RobustScaler()),
    ('model', LogisticRegression(solver='saga'))
])
svc_feat_imp = Pipeline([
    ('scaler', RobustScaler()),
    ('svc', SVC(random_state=2, kernel='linear'))
])

# treinar os modelos
dt_feat_imp.fit(x_train, y_train)
reglog_feat_imp.fit(x_train, y_train)
svc_feat_imp.fit(x_train, y_train)

# analisar as importâncias de cada atributo
# para regressão logistica
reglog_feat_imp['model'].coef_

# criar um dataframe de importância de atributos
imp = pd.DataFrame(x_train.columns, columns=['atributos'])
imp['importancia'] = reglog_feat_imp['model'].coef_[0]
imp.sort_values(by='importancia', inplace=True)

# construindo gráfico
plt.barh(y=imp['atributos'], width=imp['importancia'])
plt.show()

# criar um dataframe de importância de atributos
imp = pd.DataFrame(x_train.columns, columns=['atributos'])
imp['importancia'] = dt_feat_imp['model'].feature_importances_
imp.sort_values(by='importancia', inplace=True)

# construindo gráfico
plt.barh(y=imp['atributos'], width=imp['importancia'])
plt.show()

# criar um dataframe de importância de atributos
imp = pd.DataFrame(x_train.columns, columns=['atributos'])
imp['importancia'] = svc_feat_imp['svc'].coef_[0]
imp.sort_values(by='importancia', inplace=True)

# construindo gráfico
plt.barh(y=imp['atributos'], width=imp['importancia'])
plt.show()

# Função para plotar a curva ROC
def plot_roc_curve(y_true, y_pred, label):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

print("Conjunto de Teste")
plt.figure(figsize=(10, 8))
# Regressão Logística
y_pred_reglog = search_reglog.best_estimator_.predict_proba(x_test)[:, 1]
plot_roc_curve(y_test, y_pred_reglog, 'Regressão Logística')

# Árvore de Decisão
y_pred_dt = search_dt.best_estimator_.predict_proba(x_test)[:, 1]
plot_roc_curve(y_test, y_pred_dt, 'Árvore de Decisão')

# SVM
y_pred_svc = search_svc.best_estimator_.decision_function(x_test)
plot_roc_curve(y_test, y_pred_svc, 'SVM')

# Linha de referência para um modelo aleatório
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Aleatório')

# Configurações do gráfico
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# SETANDO O X E y
X_red_wine = wine_quality_red_wine.drop(columns=[target])
y_red_wine = wine_quality_red_wine[target]
yhat_red = search_svc.best_estimator_.predict(X_red_wine)

print('Desempenho - Base de Vinhos Tintos')
print(classification_report(y_red_wine, yhat_red))

print('Desempenho - Base de Vinhos Brancos')
print(classification_report(y_test, yhat_svc_test))
