import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from typing import List, Tuple


class NCI60Dataset(Dataset):
    """
    A class for creating a PyTorch dataset for LDMO model training.

    Args:
    data (pd.DataFrame): A pandas dataframe containing the data.
    smiles_col (str): The name of the column containing SMILES strings. Default is 'SMILES'.
    label_col (str): The name of the column containing the target variable. Default is 'NLOGGI50_N'.

    Attributes:
    data (pd.DataFrame): The input data.
    smiles (List[str]): A list of SMILES strings.
    label (List[float]): A list of target variable values.
    features (np.array): A dataframe containing the input features.

    """

    def __init__(
        self,
        data: pd.DataFrame,
        smiles_col: str = "SMILES",
        label_col: str = "NLOGGI50_N",
    ) -> None:
        self.data = data
        self.smiles: List[str] = data[smiles_col].values.tolist()
        self.label: List[float] = data[label_col].values.tolist()

        self.features: np.array = data.drop([smiles_col, label_col], axis=1).values

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        """
        Returns a single instance of the dataset.

        Args:
        idx (int): The index of the instance to return.

        Returns:
        A tuple containing a PyTorch tensor of the instance.

        """
        instance = self.features[idx].astype(np.float32)
        y = np.array(self.label[idx]).reshape(-1).astype(np.float32)

        return torch.tensor(instance), torch.tensor(y)


def standarize_data(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    smiles_col: str = "SMILES",
    label_col: str = "NLOGGI50_N",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Standardizes the input data using sklearn's StandardScaler.

    Args:
    train_data (pd.DataFrame): The training data.
    val_data (pd.DataFrame): The validation data.
    test_data (pd.DataFrame): The test data.
    smiles_col (str): The name of the column containing SMILES strings. Default is 'SMILES'.
    label_col (str): The name of the column containing the target variable. Default is 'NLOGGI50_N'.

    Returns:
    A tuple containing the standardized training, validation, and test data.

    """
    scaler = StandardScaler()
    standarized_train_data = scaler.fit_transform(train_data.drop([smiles_col, label_col], axis=1))
    if val_data is not None:
        standarized_val_data = scaler.transform(val_data.drop([smiles_col, label_col], axis=1))
    standarized_test_data = scaler.transform(test_data.drop([smiles_col, label_col], axis=1))

    train_data = pd.DataFrame(np.concatenate((train_data[[smiles_col, label_col]].values, pd.DataFrame(standarized_train_data)), axis=1)).rename(columns={0:smiles_col, 1:label_col})
    if val_data is not None:
        val_data = pd.DataFrame(np.concatenate((val_data[[smiles_col, label_col]].values, pd.DataFrame(standarized_val_data)), axis=1)).rename(columns={0:smiles_col, 1:label_col})
    test_data = pd.DataFrame(np.concatenate((test_data[[smiles_col, label_col]].values, pd.DataFrame(standarized_test_data)), axis=1)).rename(columns={0:smiles_col, 1:label_col})

    return train_data, val_data, test_data


def prepare_data(
    test_fold: int,
    standarize: bool,
    cell_line: str = "TK-10",
    features: str = "fingerprints",
    split: str = "UMAP",
    cluster_id_path: str = "../data/clustering_id_k7.csv",
) -> Tuple[NCI60Dataset, NCI60Dataset, NCI60Dataset]:
    """
    A function for preparing data for LDMO model training.

    Args:
    test_fold (int): The index of the test fold.
    standarize (bool): If standarize the data.
    features (str): The type of features to use. Default is 'fingerprints'.

    Returns:
    A tuple containing three NCI60Dataset objects for training, validation, and testing.

    """
    val_fold = 7 if test_fold == 1 else test_fold - 1
    if split == 'UMAP':
        group_col = 'Cluster_ID'
    elif split == 'scaffold':
        group_col = 'Scaffold_Cluster_ID'
    elif split == 'Butina':
        group_col = 'Butina_Cluster_ID'
    elif split == 'random':
        group_col = 'Random_Cluster_ID'
    else:
        raise ValueError(f"Split {split} not supported.")
    drop_cols = ["NSC", group_col]
    
    if features == "fingerprints":
        data_path = f"../data/60_cell_lines/{cell_line}.csv"
        data = pd.read_csv(data_path).dropna()
        cluster_id = pd.read_csv(cluster_id_path)[['NSC', group_col]]
        data = cluster_id.merge(data, how='inner', on='NSC')
        if len(data) == 0:
            raise
        
        train_data = data[(data[group_col]!=val_fold) & (data[group_col]!=test_fold)].drop(drop_cols, axis=1)
        val_data = data[data[group_col]==val_fold].drop(drop_cols, axis=1)
        test_data = data[data[group_col]==test_fold].drop(drop_cols, axis=1)

    elif features == "descriptors":
        data_path = "../data/All_tested_molecules.csv"
        data = pd.read_csv(data_path).dropna()
        train_data = data[(data[group_col]!=val_fold) & (data[group_col]!=test_fold)].drop(drop_cols, axis=1)
        val_data = data[data[group_col]==val_fold].drop(drop_cols, axis=1)
        test_data = data[data[group_col]==test_fold].drop(drop_cols, axis=1)

    else:
        raise ValueError(f"Molecule {features} not supported.")

    if standarize == True:
        train_data, val_data, test_data = standarize_data(train_data, val_data, test_data)

    train_dataset = NCI60Dataset(train_data)
    val_dataset = NCI60Dataset(val_data)
    test_dataset = NCI60Dataset(test_data)
    print(f'Total data: {data.shape[0]} | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}')
    
    return train_dataset, val_dataset, test_dataset


def prepare_outlier_data(
    standarize: bool,
    features: str = "fingerprints",
) -> Tuple[NCI60Dataset, NCI60Dataset, NCI60Dataset]:
    """
    Prepare outlier data for training and testing.

    Args:
    standarize (bool): Whether to standardize the data.
    features (str): Type of features to use. Can be "fingerprints" or "descriptors".

    Returns:
    Tuple[NCI60Dataset, NCI60Dataset]: A tuple containing the training dataset and outlier dataset.
    """
    drop_cols = ["NSC", "Cluster_ID"]

    if features == "fingerprints":
        data_path = "../data/TK-10_nooutliers.csv"
        outlier_data_path = "../data/outliers.csv"
        train_data = pd.read_csv(data_path).dropna().drop(drop_cols, axis=1)
        test_data = pd.read_csv(outlier_data_path).dropna().drop(drop_cols, axis=1)

    elif features == "descriptors":
        data_path = "../data/TK-10_nooutliers_rdNormalizedDescriptors.csv"
        outlier_data_path = "../data/outliers_rdNormalizedDescriptors.csv"
        train_data = pd.read_csv(data_path).dropna().drop(drop_cols, axis=1)
        test_data = pd.read_csv(outlier_data_path).dropna().drop(drop_cols, axis=1)

    else:
        raise ValueError(f"Molecule {features} not supported.")

    if standarize == True:
        train_data, _, test_data = standarize_data(train_data, None, test_data)

    train_dataset = NCI60Dataset(train_data)
    test_dataset = NCI60Dataset(test_data)
    print(f'Total data: {len(train_dataset)+len(test_dataset)} | Train: {len(train_dataset)} | Test: {len(test_dataset)}')
    
    return train_dataset, test_dataset

