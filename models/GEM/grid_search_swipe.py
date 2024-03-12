import json
import os
import shutil


for test_fold in range(1, 8):
    grid_results = {}
    results_dir_path = os.path.join('..', '..', 'results', 'GEM', f'fold_{test_fold}')
    os.makedirs(results_dir_path, exist_ok=True)
    lowest_val_rmse = float('inf')
    best_epoch_path = None
    for grid in os.listdir(f'./output/GEM/fold_{test_fold}'):
        for epoch_files in os.listdir(f'./output/GEM/fold_{test_fold}/{grid}'):
            if epoch_files.startswith('best'):
                epoch_path = os.path.join(f'./output/GEM/fold_{test_fold}', grid, epoch_files)
                json_path = os.path.join(epoch_path, 'test_results.json')
                with open(json_path, 'r') as file:
                    test_results = json.load(file)
                    val_rmse = test_results['val_rmse']
                    if val_rmse < lowest_val_rmse:
                        lowest_val_rmse = val_rmse
                        best_epoch_path = epoch_path
                grid_results[epoch_path] = val_rmse

    print(f"Best epoch Path for fold {test_fold}: {best_epoch_path}")
    shutil.copytree(best_epoch_path, results_dir_path)

