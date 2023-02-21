import hydra
from hydra import utils
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import numpy as np
import pandas as pd

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

# metrics and Pipeline
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import *

# Train Test Split
from sklearn.model_selection import train_test_split

# Scaling
from sklearn.preprocessing import RobustScaler

import warnings
warnings.filterwarnings("ignore")


@hydra.main(version_base='1.1', config_path='configs', config_name="hyperparameters")
def foo(config: DictConfig) -> None:
    
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    print(OmegaConf.to_yaml(instantiate(config)))
    
    # Get current path
    current_path = utils.get_original_cwd() + "/"
    #print(current_path)
    
    data = pd.read_csv(current_path + config.dataset.data)
    data.info()
    
    cat_cols = ['sex', 'exng', 'caa', 'cp', 'fbs', 'restecg', 'slp', 'thall']
    con_cols = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
        
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
    
    X = data.drop([config.target_y],axis=1)
    y = data[[config.target_y]]
    
    scaler = RobustScaler()
    X[con_cols] = scaler.fit_transform(X[con_cols])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    
    print("Data Shape: {}".format(data.shape))
    print("Train Shape: {}".format(X_train.shape))
    print("Test Shape: {}".format(X_test.shape))
    
#     param_grid = dict(config.model.hyperparameters)
#     param_grid = {ele: (list(param_grid[ele])) for ele in param_grid}
    model = instantiate(config.model.randomforest)
    print(model)
#     defaults: List[Any] = field(default_factory=lambda: defaults)
    random_search = RandomizedSearchCV(model, 
                               scoring=config.RandomizedSearchCV.scoring,
                               cv=config.RandomizedSearchCV.cv,
                               n_jobs=config.RandomizedSearchCV.n_jobs,
                               refit=config.RandomizedSearchCV.refit,
                               return_train_score=config.RandomizedSearchCV.return_train_score)

    random_search.fit(X_train, y_train)

    labels_pred = random_search.predict(X_test)

    print(confusion_matrix(y_test, labels_pred))
    print(f1_score(y_test, labels_pred, average='macro'))
    print(roc_auc_score(y_test, labels_pred))
if __name__ == "__main__":
    foo()