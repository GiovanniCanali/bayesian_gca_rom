from pina.model.block.message_passing import InteractionNetworkBlock
from torch_geometric.utils import to_dense_batch
from pina.model import FeedForward
import torch


class ReductionNetwork(torch.nn.Module):
    """
    The network that maps the PDE solution to a latent representation and back.
    The encoder and decoder are both based on Message Passing Neural Networks.
    """

    def __init__(
        self,
        num_nodes,
        node_feature_dim,
        edge_feature_dim,
        latent_dimension,
        n_message_layers_encoder=2,
        n_message_layers_decoder=2,
        n_update_layers_encoder=2,
        n_update_layers_decoder=2,
        hidden_dim_encoder=64,
        hidden_dim_decoder=64,
        n_layers_encoder=2,
        n_layers_decoder=2,
        activation=torch.nn.SiLU,
    ):
        """
        Initialization of the reduction network.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim

        # Message Passing Neural Network for the encoder
        self.encoder = torch.nn.ModuleList(
            [
                InteractionNetworkBlock(
                    node_feature_dim=node_feature_dim,
                    edge_feature_dim=edge_feature_dim,
                    hidden_dim=hidden_dim_encoder,
                    n_message_layers=n_message_layers_encoder,
                    n_update_layers=n_update_layers_encoder,
                    activation=activation,
                )
                for _ in range(n_layers_encoder)
            ]
        )

        # Message Passing Neural Network for the decoder
        self.decoder = torch.nn.ModuleList(
            [
                InteractionNetworkBlock(
                    node_feature_dim=node_feature_dim,
                    edge_feature_dim=edge_feature_dim,
                    hidden_dim=hidden_dim_decoder,
                    n_message_layers=n_message_layers_decoder,
                    n_update_layers=n_update_layers_decoder,
                    activation=activation,
                )
                for _ in range(n_layers_decoder)
            ]
        )

        # Layers to map from the graph representation to the latent space
        self.fc_encoder = torch.nn.Sequential(
            torch.nn.Linear(num_nodes * node_feature_dim, 256),
            activation(),
            torch.nn.Linear(256, latent_dimension),
        )

        # Layers to map from the latent space back to the graph representation
        self.fc_decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dimension, 256),
            activation(),
            torch.nn.Linear(256, num_nodes * node_feature_dim),
        )

    def encode(self, batch):
        """
        Encode the input graph into a latent representation.
        """
        # Store the current graph
        self.current_graph = batch.clone()
        x = batch.x

        # Pass through the encoder layers
        for layer in self.encoder:
            x = layer(
                x=x, edge_index=batch.edge_index, edge_attr=batch.edge_attr
            )

        # Reshape the node features to a graph-level representation
        # Final shape of x: [num_graphs, num_nodes * node_feature_dim]
        x = to_dense_batch(x, batch.batch)[0]
        x = x.reshape(x.size(0), -1)

        return self.fc_encoder(x)

    def decode(self, z, decoding_graph=None):
        """
        Decode the latent representation back to a graph representation.
        """
        # If a graph to decode is passed, use it
        batch = decoding_graph or self.current_graph

        # Map the latent representation back to the graph-level representation
        x = self.fc_decoder(z).reshape(
            z.shape[0] * self.num_nodes, self.node_feature_dim
        )

        # Pass through the decoder layers
        for layer in self.decoder:
            x = layer(
                x=x, edge_index=batch.edge_index, edge_attr=batch.edge_attr
            )

        return x


class MessagePassingNeuralNetwork(torch.nn.Module):
    """
    The full model that combines the reduction network with an interpolation
    network to map from the PDE parameters to the solution.
    """

    def __init__(
        self,
        parameter_dimension,
        pde_dimension,
        latent_dimension,
        num_nodes,
        edge_feature_dim,
        n_layers_encoder=2,
        n_layers_decoder=2,
        n_layers_interpolator=2,
        hidden_dim_encoder=64,
        hidden_dim_decoder=64,
        hidden_dim_interpolator=64,
        n_message_layers_encoder=2,
        n_message_layers_decoder=2,
        n_update_layers_encoder=2,
        n_update_layers_decoder=2,
        activation=torch.nn.SiLU,
    ):
        """
        Initialization of the full model.
        """
        super().__init__()
        self.activation = activation()

        # Interpolator
        self.interpolation_network = FeedForward(
            input_dimensions=parameter_dimension,
            output_dimensions=latent_dimension,
            n_layers=n_layers_interpolator,
            inner_size=hidden_dim_interpolator,
            func=activation,
        )

        # Reduction network
        self.reduction_network = ReductionNetwork(
            num_nodes=num_nodes,
            node_feature_dim=pde_dimension,
            edge_feature_dim=edge_feature_dim,
            latent_dimension=latent_dimension,
            n_message_layers_encoder=n_message_layers_encoder,
            n_message_layers_decoder=n_message_layers_decoder,
            n_update_layers_encoder=n_update_layers_encoder,
            n_update_layers_decoder=n_update_layers_decoder,
            n_layers_encoder=n_layers_encoder,
            n_layers_decoder=n_layers_decoder,
            hidden_dim_encoder=hidden_dim_encoder,
            hidden_dim_decoder=hidden_dim_decoder,
            activation=activation,
        )

    def forward(self, params, batch):
        """
        Forward pass of the model: map from the PDE parameters to the solution.
        """
        z = self.interpolation_network(x=params)
        output = self.reduction_network.decode(z=z, decoding_graph=batch)

        return to_dense_batch(output, batch.batch)[0]


class PosteriorNetwork(torch.nn.Module):
    """
    The newtwork that computes the  variational adaptive dropout coefficients.
    Used only in the bayesian version of the model.
    """

    def __init__(
        self,
        parameter_dimension,
        node_dimension,
        edge_dimension,
        hidden_dim,
        num_alphas_node,
        num_alphas_edge,
        num_alphas_param,
        n_layers,
        activation=torch.nn.SiLU,
    ):
        """
        Initialization of the posterior network.
        """
        super().__init__()

        # Blocks for nodes, edges and parameters
        self.blocks = torch.nn.ModuleDict(
            {
                "node_block": FeedForward(
                    input_dimensions=node_dimension,
                    output_dimensions=num_alphas_node,
                    n_layers=n_layers,
                    inner_size=hidden_dim,
                    func=activation,
                ),
                "edge_block": FeedForward(
                    input_dimensions=edge_dimension,
                    output_dimensions=num_alphas_edge,
                    n_layers=n_layers,
                    inner_size=hidden_dim,
                    func=activation,
                ),
                "param_block": FeedForward(
                    input_dimensions=parameter_dimension,
                    output_dimensions=num_alphas_param,
                    n_layers=n_layers,
                    inner_size=hidden_dim,
                    func=activation,
                ),
            }
        )

    def forward(self, params, batch):
        """
        Forward pass of the posterior network.
        """
        # Node embedding -- shape [num_nodes, num_alphas_node]
        emb_node = self.blocks["node_block"](batch.posterior_node_feats)

        # Edge embedding -- shape [num_edges, num_alphas_edge]
        emb_edge = self.blocks["edge_block"](batch.edge_attr)

        # Parameter embedding -- shape [num_params, num_alphas_param]
        emb_param = self.blocks["param_block"](params)

        return {
            "alpha_node": torch.nn.functional.softplus(emb_node),
            "alpha_edge": torch.nn.functional.softplus(emb_edge),
            "alpha_param": torch.nn.functional.softplus(emb_param),
        }
