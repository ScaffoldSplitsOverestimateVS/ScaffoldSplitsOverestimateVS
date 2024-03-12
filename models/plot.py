import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, gaussian_kde
from sklearn.metrics import (accuracy_score, auc, confusion_matrix,
                             matthews_corrcoef, mean_absolute_error,
                             mean_squared_error, r2_score, roc_curve)
from termcolor import colored


'''
def cm_plot(file_path, test_results, model_name, save=True):
    if np.isnan(test_results["hit_rate"]):
        test_results["hit_rate"] = 0            
    sns.set()
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

    ax.plot(test_results['y_test'], test_results['y_pred'], 'o', color='b', markersize=2,
            markeredgecolor='k', alpha=0.3)
    ax.plot([1.4, 12.5], [1.4, 12.5], 'k--', alpha=0.2)

    ax.set_xlabel('Measured pGI50')
    ax.set_ylabel('Predicted pGI50')
    ax.set_title(f'Cell line: TK-10 | Model: {model_name}', fontsize=10)

    text = (
        f'RMSE: {test_results["rmse"]:.3f}\n'
        f'MAE: {test_results["mae"]:.3f}\n'
        # f'$R^2$: {test_results["r2"]:.4f}\n'
        f'$R_p$: {test_results["pearson_rp"]:.3f}\n'
        f'$R_s$: {test_results["spearman_rs"]:.3f}\n'
        f'MCC: {test_results["mcc"]:.3f}\n'
        f'ROC AUC: {test_results["roc_auc"]:.3f}\n'
        # f'Accuracy: {test_results["accuracy"]:.4f}\n'
        f'Hit rate (%): {test_results["hit_rate"]:.1f}'
    )

    ax.annotate(text, xy=(1.4, 0.5), xytext=(-15, 15), fontsize=10,
                xycoords='axes fraction', textcoords='offset points',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round'),
                horizontalalignment='right', verticalalignment='bottom')

    bbox_props = {'facecolor': 'white', 'alpha': 1, 'pad': 0.5, 'edgecolor': 'black', 'boxstyle': 'round'}

    ax.text(3.2, 10.5, f'FP = {str(test_results["cm"].iloc[0, 1])}',
            fontsize=10, bbox=bbox_props)
    ax.text(3.2, 3.2, f'TN = {str(test_results["cm"].iloc[0, 0])}',
            fontsize=10, bbox=bbox_props)
    ax.text(9.5, 3.2, f'FN = {str(test_results["cm"].iloc[1, 0])}',
            fontsize=10, bbox=bbox_props)
    ax.text(9.5, 10.5, f'TP = {str(test_results["cm"].iloc[1, 1])}',
            fontsize=10, bbox=bbox_props)

    ax.margins(0)
    ax.axhspan(ymin=1.4, ymax=6, xmin=0.375, xmax=4, facecolor='red', alpha=0.05)
    ax.axhspan(ymin=6, ymax=12.5, xmin=0, xmax=0.375, facecolor='red', alpha=0.05)
    ax.set_ylim(3, 11)
    ax.set_xlim(3, 11)
    

    if save == True:
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1)
        print(f'Made a plots stored in {file_path}')
    # plt.show()
    plt.close()
'''

def cm_plot(file_path, test_results, model_name, save=True):
    if np.isnan(test_results["hit_rate"]):
        test_results["hit_rate"] = 0
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(5, 5), dpi=180)

    x = test_results['y_test']
    y = test_results['y_pred']
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    scatter = ax.scatter(x, y, c=z, cmap='magma', s=8, alpha=0.6, marker='.')
    
    ax.plot([1.4, 12.5], [1.4, 12.5], 'k--', alpha=0.2)

    ax.set_xlabel('Measured pGI50')
    ax.set_ylabel('Predicted pGI50')
    ax.set_title(f'Cell line: TK-10 | Model: {model_name}', fontsize=10)

    bbox_props = {'facecolor': 'white', 'alpha': 1, 'pad': 0.5, 'edgecolor': 'black', 'boxstyle': 'round'}

    ax.text(3.2, 10.5, f'FP = {str(test_results["cm"].iloc[0, 1])}',
            fontsize=10, bbox=bbox_props)
    ax.text(3.2, 3.2, f'TN = {str(test_results["cm"].iloc[0, 0])}',
            fontsize=10, bbox=bbox_props)
    ax.text(9.5, 3.2, f'FN = {str(test_results["cm"].iloc[1, 0])}',
            fontsize=10, bbox=bbox_props)
    ax.text(9.5, 10.5, f'TP = {str(test_results["cm"].iloc[1, 1])}',
            fontsize=10, bbox=bbox_props)

    ax.margins(0)
    ax.axhspan(ymin=1.4, ymax=6, xmin=0.375, xmax=4, facecolor='red', alpha=0.05)
    ax.axhspan(ymin=6, ymax=12.5, xmin=0, xmax=0.375, facecolor='red', alpha=0.05)
    ax.set_ylim(3, 11)
    ax.set_xlim(3, 11)

    # Create the table under the plot
    table_data = [
        ["RMSE", "$R_p$", "$R_s$", "MCC", "ROC AUC", "Hit Rate (%)"],
        [f'{test_results["rmse"]:.3f}', f'{test_results["pearson_rp"]:.3f}', f'{test_results["spearman_rs"]:.3f}', f'{test_results["mcc"]:.3f}', f'{test_results["roc_auc"]:.3f}', f'{test_results["hit_rate"]:.1f}']
    ]

    bbox = [0, -0.25, 1.2, 0.1]  # [left, bottom, width, height]
    table = ax.table(cellText=table_data, loc='bottom', cellLoc='center', bbox=bbox)
    table.auto_set_font_size(False)
    table.set_fontsize(7)

    # Add color bar
    cbar = plt.colorbar(scatter, shrink=0.5)
    cbar.set_label('Scatter Density', fontsize=10)

    if save:
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1)
        print(f'Made a plot stored in {file_path}')
    # plt.show()
    plt.close()


def calculate_resutls(y_test, y_pred):
    y_pred = np.array(y_pred).reshape(-1)
    y_test = np.array(y_test).reshape(-1)
    pearson_rp = pearsonr(y_test, y_pred)[0]
    spearman_rs = spearmanr(y_test, y_pred)[0]
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    obs = [0 if i < 6 else 1 for i in y_test]
    pred = [0 if i < 6 else 1 for i in y_pred] 
    mcc = matthews_corrcoef(obs, pred)
    accuracy = accuracy_score(obs, pred)
    cm = pd.DataFrame(confusion_matrix(obs, pred))
    try:
        hit_rate = (cm.iloc[1, 1] / (cm.iloc[1, 1] + cm.iloc[0, 1]) * 100)
    except:
        hit_rate = 0
    if hit_rate is np.nan:
        hit_rate = 0
    fpr, tpr, thresholds = roc_curve(obs, pred)
    roc_auc = auc(fpr, tpr)

    test_results = {
        'y_test': y_test,
        'y_pred': y_pred,
        'pearson_rp': np.float64(pearson_rp),
        'spearman_rs': np.float64(spearman_rs),
        'rmse': np.float64(rmse),
        'r2': np.float64(r2),
        'accuracy': np.float64(accuracy),
        'mcc': np.float64(mcc),
        'hit_rate': np.float64(hit_rate),
        'mae': np.float64(mae),
        'cm': cm,
        'roc_auc': np.float64(roc_auc)
    }

    print(f'RMSE: {test_results["rmse"]:.3f}|MAE: {test_results["mae"]:.3f}|$R_p$: {test_results["pearson_rp"]:.3f}|$R_s$: {test_results["spearman_rs"]:.3f}|MCC: {test_results["mcc"]:.3f}|ROC AUC: {test_results["roc_auc"]:.3f}|Hit rate (%): {test_results["hit_rate"]:.1f}')

    return test_results


# -----------------------------------------------------------------------------
# Log
# -----------------------------------------------------------------------------
def create_logger(output_dir):
    # log name
    time_str = time.strftime("%Y-%m-%d")
    log_name = f"{time_str}.log"

    # log dir
    log_dir = os.path.join(output_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = \
        colored('[%(asctime)s]', 'green') + \
        colored('(%(filename)s %(lineno)d): ', 'yellow') + \
        colored('%(levelname)-5s', 'magenta') + ' %(message)s'

    # create console handlers for master process
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger