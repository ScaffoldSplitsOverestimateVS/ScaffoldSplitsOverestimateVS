import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
# import xgboost as xgb
# from sklearn.svm import SVR

from data import prepare_data
from plot import calculate_resutls, cm_plot, create_logger


def define_models(seed: int) -> Dict[str, Tuple]:
    """Defines the hyperparameter grid for four different models and returns them as a dictionary.

    Returns:
        models (Dict[str, Tuple]): A dictionary containing four models and their respective hyperparameter grids.
    """
    # Define the hyperparameter grid for linear regression
    lr_param_grid: Dict[str, List] = {'fit_intercept': [True, False]}

    # Define the hyperparameter grid for Random Forest
    rf_param_grid: Dict[str, List] = {
        'n_estimators': [250, 500, 750, 1000],
        'max_features': [0.2, 0.3, 0.4, 0.6, 0.8, 0.9]
    }

    # Define the hyperparameter grid for SVM
    svm_param_grid: Dict[str, List] = {
        'C': [1, 5],
        'kernel': ['linear', 'rbf', 'poly']
    }

    # Define the hyperparameter grid for XGBoost
    xgb_param_grid: Dict[str, List] = {
        'n_estimators': [250, 500, 750, 1000],
        'max_depth': [5, 6, 7, 8, 9, 10],
        'colsample_bytree': [0.3, 0.4, 0.6, 0.8, 0.9]
    }

    # Define the models
    lr_model: LinearRegression = LinearRegression()
    rf_model: RandomForestRegressor = RandomForestRegressor(random_state=seed)
    # svm_model: SVR = SVR()
    # xgb_model: xgb.XGBRegressor = xgb.XGBRegressor()

    models: Dict[str, Tuple] = {
        'Linear_Regression': (lr_model, lr_param_grid),
        'Random_Forest': (rf_model, rf_param_grid),
        # 'SVM': (svm_model, svm_param_grid),
        # 'XGBoost': (xgb_model, xgb_param_grid)
    }
    return models



def grid_search(model: object, param_grid: Dict[str, List], train_dataset: object, val_dataset: object) -> Tuple[object, Dict[str, List], float]:
    """Performs a grid search to find the best hyperparameters for a given model.

    Args:
        model (object): The model to be trained.
        param_grid (Dict[str, List]): A dictionary containing the hyperparameter grid to be searched.
        train_dataset (object): The training dataset.
        val_dataset (object): The validation dataset.

    Returns:
        Tuple[object, Dict[str, List], float]: A tuple containing the best model, its hyperparameters, and the best RMSE.
    """
    best_rmse = float('inf')
    best_model = None
    best_params = None
    param_list = list(ParameterGrid(param_grid))
    for params in param_list:
        logger.info(f'Trying params: {params}')
        model.set_params(**params)
        tmp_time = time.time()
        model.fit(train_dataset.features, train_dataset.label)
        timing['train_time'] += time.time() - tmp_time
        logger.info(f'Train time {time.time() - tmp_time}')
        tmp_time = time.time()
        y_pred = model.predict(val_dataset.features)
        timing['val_time'] += time.time() - tmp_time
        logger.info(f'Val time {time.time() - tmp_time}')
        rmse = mean_squared_error(val_dataset.label, y_pred, squared=False)
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = params
    return best_model, best_params, best_rmse


def no_tune(model: object, train_dataset: object, val_dataset: object) -> Tuple[object, Dict[str, List], float]:
    best_rmse = float('inf')
    best_params = None
    features = np.concatenate([train_dataset.features, val_dataset.features], axis=0)
    label = np.concatenate([train_dataset.label, val_dataset.label], axis=0)
    tmp_time = time.time()
    model.fit(features, label)
    timing['train_time'] += time.time() - tmp_time
    logger.info(f'Train time {time.time() - tmp_time}')
    return model, best_params, best_rmse


def evaluate_outlier(args):
    from collections import Counter
    from data import prepare_outlier_data
    
    outlier_dir = os.path.join('..', 'results', f'{model_name}_{args.features}', 'outlier')
    os.makedirs(outlier_dir, exist_ok=True)
    all_params = []
    for i in range(1, 8):
        json_file_path = os.path.join('..', 'results', f'{model_name}_{args.features}', f'fold_{i}', 'best_params.json')
        with open(json_file_path, 'r') as f:
            all_params.append(json.load(f))

    best_params = Counter(tuple(sorted(d.items())) for d in all_params).most_common(1)[0][0] # count the most common parameters
    best_params = dict((k, v) for k, v in best_params) # convert the most common parameters to a dictionary
    
    logger.info(f'Trying best params: {best_params} on outlier set.')
    start_time = time.time()
    train_dataset, test_dataset = prepare_outlier_data(args.standarize, args.features)
    model.set_params(**best_params)
    model.fit(train_dataset.features, train_dataset.label)
    y_pred = model.predict(test_dataset.features)

    test_results = calculate_resutls(test_dataset.label, y_pred)
    cm_plot(os.path.join(outlier_dir, 'outlier_cm.png'), test_results, model_name, save=True)

    del test_results['y_test']
    del test_results['y_pred']
    del test_results['cm']
    test_results['time'] = time.time() - start_time
    with open(os.path.join(outlier_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)
    np.save(os.path.join(outlier_dir, 'test_pred.npy'), y_pred)
    np.save(os.path.join(outlier_dir, 'test_labels.npy'), test_dataset.label)

    with open(os.path.join(outlier_dir, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--test_fold', type=int, default=7, help='test fold')
    parser.add_argument('--standarize', type=bool, default=True, help='standarize')
    parser.add_argument('--features', type=str, default='fingerprints', help='features')
    parser.add_argument('--search', type=str, default='grid')
    parser.add_argument('--cell_line', type=str, default='TK-10')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split', type=str, default='UMAP')
    parser.add_argument('--cluster_id_path', type=str, default='../data/clustering_id_k7.csv')

    args = parser.parse_args()
    np.random.seed(args.seed)

    train_dataset, val_dataset, test_dataset = prepare_data(
        test_fold = args.test_fold, 
        standarize = args.standarize, 
        cell_line = args.cell_line, 
        features = args.features,
        split = args.split,
        cluster_id_path = args.cluster_id_path,
    )

    models = define_models(args.seed)
    for model_name, (model, param_grid) in models.items():
        save_dir = os.path.join(
            '..',
            'results',
            f'{model_name}_{args.features}',
            args.cell_line,
            f'split_{args.split}',
            f'fold_{args.test_fold}',
            f'seed_{args.seed}',
        )
        if not os.path.isfile(os.path.join(save_dir, 'test_results.json')):
            start_time = time.time()
            timing = {'train_time': 0, 'val_time': 0, 'test_time': 0}
            os.makedirs(save_dir, exist_ok=True)
    
            logger = create_logger(save_dir)
            logger.info(f'Running {model_name}...')
            logger.info(args)
            logger.info(model)

            if args.search == 'grid':
                best_model, best_params, best_rmse = grid_search(model, param_grid, train_dataset, val_dataset)
            elif args.search == 'no_tune':
                best_model, best_params, best_rmse = no_tune(model, train_dataset, val_dataset)

            tmp_time = time.time()
            y_pred = best_model.predict(test_dataset.features)
            timing['test_time'] += time.time() - tmp_time
            logger.info(f'Test time {time.time() - tmp_time}')

            np.save(os.path.join(save_dir, 'test_pred.npy'), y_pred)
            np.save(os.path.join(save_dir, 'test_labels.npy'), test_dataset.label)

            with open(os.path.join(save_dir, 'best_params.json'), 'w') as f:
                json.dump(best_params, f, indent=4)

            test_results = calculate_resutls(test_dataset.label, y_pred)
            cm_plot(os.path.join(save_dir,'cm.svg'), test_results, args.cell_line, model_name, save=True)

            del test_results['y_test']
            del test_results['y_pred']
            del test_results['cm']
            test_results['time'] = time.time() - start_time
            test_results['train_time'] = timing['train_time']
            test_results['val_time'] = timing['val_time']
            test_results['test_time'] = timing['test_time']
            logger.info(f'Total training time {test_results["time"]:2f}s.')
            logger.info(f'Triaining time: {timing["train_time"]} | Validation time: {timing["val_time"]} | Test time: {timing["test_time"]}')    

            with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
                json.dump(test_results, f, indent=4)

        else:
            print(f'Skipping {model_name}...')

        if not os.path.isfile(os.path.join('..', 'results', f'{model_name}_{args.features}', 'cross_validation_results.csv')):
            # Summarize results
            all_files_exist = True
            for fold_num in range(1, 8):
                if not os.path.isfile(os.path.join('..', 'results', f'{model_name}_{args.features}', f'fold_{fold_num}', 'test_pred.npy')):
                    all_files_exist = False
                    break
            if all_files_exist:
                with open(os.path.join('..', 'results', f'{model_name}_{args.features}', 'cross_validation_results.csv'), 'w') as f:
                    for fold_num in range(1, 8):
                        with open(os.path.join('..', 'results', f'{model_name}_{args.features}', f'fold_{fold_num}', 'test_results.json'), 'r') as json_file:
                            test_results = json.load(json_file)
                            if fold_num == 1:
                                f.write(','.join(test_results.keys()) + '\n')
                            f.write(','.join(str(val) for val in test_results.values()) + '\n') # We need to convert the values to strings before writing them to the file.

                evaluate_outlier(args)






