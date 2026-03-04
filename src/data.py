from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
import torch


def load_data(path):
    """
    Loads PDE solution data from a .pt file and constructs PyG Data objects.
    """
    # Load the data from the specified path
    data = torch.load(path)

    # Extract node positions, solution values, and edge indices
    x, y, u = data["x"], data["y"], data["u"]
    edge_index = to_undirected(data["edge_index"])

    # Normalize solution values for airfoil dataset
    if "airfoil" in path:
        u = u / 100.0

    # Extract triangulation, parameters and number of graphs
    triang = data["triang"]
    params = data["mu"]
    _, num_graphs = u.shape

    # Create a list of PyG graph objects for each graph in the dataset
    graphs = []
    for g in range(num_graphs):

        # Node positions --  shape [num_nodes, 2]
        pos = torch.stack([x[:, g], y[:, g]], dim=1)

        # Edge attributes and weights -- shape [num_edges, 2] and [num_edges, 1]
        ei, ej = pos[edge_index[0]], pos[edge_index[1]]
        edge_attr = torch.abs(ej - ei)
        edge_weight = edge_attr.norm(p=2, dim=1, keepdim=True)

        # Node features (solution values) -- shape [num_nodes, 1]
        node_features = u[:, g].unsqueeze(-1)

        # Posterior node features combining positions and parameters
        post_feats = torch.cat([pos, params[g].repeat(pos.shape[0], 1)], dim=1)

        # Build PyG Data
        graphs.append(
            Data(
                x=node_features,
                edge_index=edge_index,
                edge_weight=edge_weight,
                edge_attr=edge_attr,
                pos=pos,
                posterior_node_feats=post_feats,
            )
        )

    return params, graphs, triang
