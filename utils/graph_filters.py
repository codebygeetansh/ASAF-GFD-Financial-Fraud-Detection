import os
import math
import torch
import torch.nn.functional as F
import dgl

DATA_DIR = r"D:\BTech_Capstone_ASAF\data"

def build_undirected_adj(edge_src, edge_dst, num_nodes):
    """
    Original Graph G1 ko undirected banakar Normalized Adjacency Matrix (D^-1/2 A D^-1/2)
    sparse format mein tayyar karta hai (Base Paper Eq. 3).
    """
    # Undirected graph: (u -> v) aur (v -> u) dono edges rakho
    row = torch.cat([edge_src, edge_dst])
    col = torch.cat([edge_dst, edge_src])
    
    # Degree D calculate karo
    deg = torch.bincount(row, minlength=num_nodes).float()
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    
    # Edge weights = D^(-1/2)_u * D^(-1/2)_v
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    indices = torch.stack([row, col], dim=0)
    adj_norm = torch.sparse_coo_tensor(
        indices, edge_weight, (num_nodes, num_nodes)
    ).coalesce()
    return adj_norm

def beta_wavelet_filter(adj_norm, X, p, q):
    """
    Base Paper Eq. (4) & (5): Beta Wavelet Operator W_{p,q}(X)
    L = I - A_norm  =>  L/2 = 0.5 * (I - A_norm)  aur  I - L/2 = 0.5 * (I + A_norm)
    Complexity: O(C * |E|) via iterative sparse propagation (Section 5.1)
    """
    # Beta constant: B(p+1, q+1) = (p! * q!) / (p + q + 1)!
    beta_const = (math.factorial(p) * math.factorial(q)) / math.factorial(p + q + 1)
    scale = 1.0 / (2.0 * beta_const)
    
    out = X
    # Apply (L / 2)^p
    for _ in range(p):
        out = 0.5 * (out - torch.sparse.mm(adj_norm, out))
        
    # Apply (I - L / 2)^q
    for _ in range(q):
        out = 0.5 * (out + torch.sparse.mm(adj_norm, out))
        
    return scale * out

def build_knn_graph_chunked(features, k=20, chunk_size=1024):
    """
    Base Paper Section 4.2 (Eq. 9, 10, 11):
    Cosine Similarity se har node ke top-K nearest neighbors nikaalta hai.
    Chunking use ki gayi hai taaki RAM kabhi full na ho.
    """
    knn_cache = os.path.join(DATA_DIR, f"elliptic_knn_k{k}.pt")
    if os.path.exists(knn_cache):
        print(f"=> D-Drive se saved kNN Graph (k={k}) load ho raha hai...")
        saved = torch.load(knn_cache)
        return saved['knn_src'], saved['knn_dst']

    print(f"Step: Feature Similarity se kNN Graph (k={k}) ban raha hai (30-45 seconds lagenge)...")
    num_nodes = features.shape[0]
    
    # Cosine similarity ke liye features ko L2-normalize karo (Eq. 9)
    X_norm = F.normalize(features, p=2, dim=1)
    
    src_list = []
    dst_list = []
    
    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)
        chunk = X_norm[start:end]  # [chunk_size, F]
        
        # Cosine similarity matrix for current chunk: [chunk_size, N]
        sim = torch.mm(chunk, X_norm.t())
        
        # Top-(k+1) neighbors nikaalo (khud node ko hata kar top-k bachein)
        _, topk_indices = torch.topk(sim, k=k + 1, dim=1, largest=True, sorted=True)
        topk_neighbors = topk_indices[:, 1:k + 1]  # [chunk_size, k]
        
        rows = torch.arange(start, end).unsqueeze(1).expand(-1, k).reshape(-1)
        cols = topk_neighbors.reshape(-1)
        
        src_list.append(rows)
        dst_list.append(cols)
        
        if (start // chunk_size) % 40 == 0:
            print(f"   Processed {end}/{num_nodes} nodes...")

    knn_src = torch.cat(src_list)
    knn_dst = torch.cat(dst_list)
    
    torch.save({'knn_src': knn_src, 'knn_dst': knn_dst}, knn_cache)
    print(f"=> kNN Graph saved to D-Drive: {knn_cache}")
    return knn_src, knn_dst

def prepare_dual_path_features(g, features, C=2, k=20):
    """
    Path 1 (Beta Wavelet Multi-Frequency Filters on G1) aur
    Path 2 (Low-Pass Consistency Filter on kNN Graph G2) compute karke save karta hai.
    """
    cache_file = os.path.join(DATA_DIR, f"dual_path_features_C{C}_k{k}.pt")
    if os.path.exists(cache_file):
        print(f"=> D-Drive se Dual-Path Filtered Features (C={C}, k={k}) load ho rahe hain...")
        data = torch.load(cache_file)
        return data['Z1_bands'], data['Z2_lowpass']

    num_nodes = features.shape[0]
    edge_src, edge_dst = g.edges()
    # Self-loops hata kar clean undirected adjacency banao
    mask = edge_src != edge_dst
    adj_g1 = build_undirected_adj(edge_src[mask], edge_dst[mask], num_nodes)

    print(f"\n--- Path 1: Running Beta Wavelet Filters (Order C={C}) on Original Graph G1 ---")
    Z1_bands = []
    for p in range(C + 1):
        q = C - p
        band_feat = beta_wavelet_filter(adj_g1, features, p, q)
        Z1_bands.append(band_feat)
        print(f"   Computed Beta Kernel W_({p},{q}) -> shape: {band_feat.shape}")

    print(f"\n--- Path 2: Constructing kNN Graph G2 (k={k}) & Running Low-Pass Filter ---")
    knn_src, knn_dst = build_knn_graph_chunked(features, k=k)
    adj_g2 = build_undirected_adj(knn_src, knn_dst, num_nodes)
    
    # Low-pass component W_(0, C) on kNN Graph G2 (Section 4.2)
    Z2_lowpass = beta_wavelet_filter(adj_g2, features, p=0, q=C)
    print(f"   Computed kNN Low-Pass Representation Z2 -> shape: {Z2_lowpass.shape}")

    torch.save({
        'Z1_bands': Z1_bands,
        'Z2_lowpass': Z2_lowpass
    }, cache_file)
    print(f"\n=> SUCCESS! Dual-Path Features saved at: {cache_file}")
    return Z1_bands, Z2_lowpass

if __name__ == "__main__":
    from data_loader import load_elliptic_data
    g, features, labels, train_mask, val_mask, test_mask = load_elliptic_data()
    Z1_bands, Z2_lowpass = prepare_dual_path_features(g, features, C=2, k=20)
    print("\n🎉 Path 1 aur Path 2 dono 100% taiyaar hain!")