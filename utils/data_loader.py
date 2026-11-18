import os

# Step 0: C-Drive ko bachane ke liye saari downloading strictly D-Drive mein set karo
DATA_DIR = r"D:\BTech_Capstone_ASAF\data"
os.makedirs(DATA_DIR, exist_ok=True)
os.environ["KAGGLEHUB_CACHE"] = DATA_DIR
os.environ["DGL_DOWNLOAD_DIR"] = DATA_DIR

import torch
import pandas as pd
import numpy as np
import dgl
import kagglehub

def load_elliptic_data():
    processed_file = os.path.join(DATA_DIR, "elliptic_processed.pt")
    
    # Agar pehle se process hokar D-Drive mein save hai, toh 2 second mein load kar lo
    if os.path.exists(processed_file):
        print("Step 1: D-Drive se Saved Elliptic Dataset load ho raha hai...")
        saved_data = torch.load(processed_file)
        g = dgl.graph((saved_data['edge_src'], saved_data['edge_dst']), num_nodes=saved_data['num_nodes'])
        g = dgl.add_self_loop(g)
        return (
            g,
            saved_data['features'],
            saved_data['labels'],
            saved_data['train_mask'],
            saved_data['val_mask'],
            saved_data['test_mask']
        )

    print("Step 1: Official Elliptic Bitcoin Dataset D-Drive mein download ho raha hai...")
    print("(Note: Yeh ~50 MB ka real dataset hai, pehli baar mein 1-2 minute lagenge)...")
    dataset_path = kagglehub.dataset_download("ellipticco/elliptic-data-set")

    # CSV files ka exact path dhoondho
    feat_path, class_path, edge_path = None, None, None
    for root, dirs, files in os.walk(dataset_path):
        if "elliptic_txs_features.csv" in files:
            feat_path = os.path.join(root, "elliptic_txs_features.csv")
            class_path = os.path.join(root, "elliptic_txs_classes.csv")
            edge_path = os.path.join(root, "elliptic_txs_edgelist.csv")
            break

    print("Step 2: CSV files padhi jaa rahi hain (2,03,769 Bitcoin Transactions)...")
    df_features = pd.read_csv(feat_path, header=None)
    df_classes = pd.read_csv(class_path)
    df_edges = pd.read_csv(edge_path)

    # Column 0 Transaction ID (txId) hai, Column 1 Time Step (1 se 49) hai
    tx_ids = df_features[0].values
    time_steps = torch.tensor(df_features[1].values, dtype=torch.long)

    # Base Paper Table 1 ke mutabiq 166 node features (Column 1 se 166 tak)
    features = torch.tensor(df_features.iloc[:, 1:].values, dtype=torch.float32)

    # Labels map karna: Elliptic mein '1' = Fraud (4,545), '2' = Normal (42,019), 'unknown' = (157,205)
    # Hum isko Base Paper ke hisaab se: 1 = Fraud, 0 = Normal, -1 = Unknown banayenge
    raw_classes = df_classes['class'].astype(str).values
    labels = torch.full((len(raw_classes),), -1, dtype=torch.long)
    labels[raw_classes == '2'] = 0  # Normal (Benign) nodes
    labels[raw_classes == '1'] = 1  # Fraud (Illicit) nodes

    # Transaction IDs ko 0 se 203768 node index mein convert karna
    tx_index = pd.Index(tx_ids)
    src_nodes = tx_index.get_indexer(df_edges['txId1'].values)
    dst_nodes = tx_index.get_indexer(df_edges['txId2'].values)

    valid_edges = (src_nodes >= 0) & (dst_nodes >= 0)
    edge_src = torch.tensor(src_nodes[valid_edges], dtype=torch.long)
    edge_dst = torch.tensor(dst_nodes[valid_edges], dtype=torch.long)

    # Base Paper Section 5.1 ka Exact Chronological Data Split:
    # Time steps 1-23: Train | Time steps 24-34: Validation | Time steps 35-49: Test
    labeled_mask = (labels != -1)
    train_mask = (time_steps <= 23) & labeled_mask
    val_mask = (time_steps >= 24) & (time_steps <= 34) & labeled_mask
    test_mask = (time_steps >= 35) & labeled_mask

    num_nodes = features.shape[0]
    num_edges = edge_src.shape[0]
    num_normals = (labels == 0).sum().item()
    num_frauds = (labels == 1).sum().item()
    num_unknowns = (labels == -1).sum().item()

    print("\n" + "=" * 55)
    print("✅ ELLIPTIC DATASET LOADED (MATCHED WITH BASE PAPER TABLE 1)")
    print("=" * 55)
    print(f"Total Nodes (Transactions) : {num_nodes}   (Paper Table 1: 203769)")
    print(f"Total Edges (Payment Flows): {num_edges}   (Paper Table 1: 234355)")
    print(f"Feature Dimensions         : {features.shape[1]}      (Paper Table 1: 166)")
    print(f"Normal Nodes (Benign)      : {num_normals}    (Paper Table 1: 42019)")
    print(f"Fraud Nodes (Illicit)      : {num_frauds}     (Paper Table 1: 4545)")
    print(f"Unknown Nodes              : {num_unknowns}   (Paper Table 1: 157205)")
    print("-" * 55)
    print(f"Train Set Labeled Nodes (Steps 1-23) : {train_mask.sum().item()}")
    print(f"Val Set Labeled Nodes   (Steps 24-34): {val_mask.sum().item()}")
    print(f"Test Set Labeled Nodes  (Steps 35-49): {test_mask.sum().item()}")
    print("=" * 55)

    # D-Drive mein processed file save kar lo taaki agli baar 1 second mein load ho
    torch.save({
        'num_nodes': num_nodes,
        'edge_src': edge_src,
        'edge_dst': edge_dst,
        'features': features,
        'labels': labels,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask
    }, processed_file)
    print(f"=> Processed dataset saved at: {processed_file}")

    g = dgl.graph((edge_src, edge_dst), num_nodes=num_nodes)
    g = dgl.add_self_loop(g)
    return g, features, labels, train_mask, val_mask, test_mask

if __name__ == "__main__":
    load_elliptic_data()