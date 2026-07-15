import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len

        self.fc_full = nn.Linear(2*self.atom_fea_len + self.nbr_fea_len,
                                 2*self.atom_fea_len)

        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]

        total_nbr_fea = torch.cat(
                [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                 atom_nbr_fea,
                 nbr_fea], dim=2)

        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(-1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)

        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)

        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)

        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class ResidualBlockMLP(nn.Module):
    def __init__(self, h_fea_len, dropout_rate=0.1):
        super(ResidualBlockMLP, self).__init__()
        self.fc = nn.Linear(h_fea_len, h_fea_len)
        self.ln = nn.LayerNorm(h_fea_len)
        self.activation = nn.Softplus()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.ln(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + residual


class MagneticCrystalGraphConvNet(nn.Module):
    def __init__(self, 
                 orig_atom_fea_len,
                 orig_mag_fea_len,
                 cg_nbr_fea_len,
                 mag_nbr_fea_len,
                 atom_fea_len=64,
                 n_conv=3,
                 n_conv_mag=3,
                 h_fea_len=256,
                 n_h=2,
                 dropout_rate=0.1,
                 classification=False):
        super(MagneticCrystalGraphConvNet, self).__init__()
        self.classification = classification

        # structure graph
        self.cg_embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.cg_convs = nn.ModuleList([
            ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=cg_nbr_fea_len)
            for _ in range(n_conv)
        ])

        # magnetic subgraph
        self.mag_embedding = nn.Linear(orig_mag_fea_len, atom_fea_len)
        self.mag_convs = nn.ModuleList([
            ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=mag_nbr_fea_len)
            for _ in range(n_conv_mag)
        ])

        # coupling layers
        self.cross_linears = nn.ModuleList([
            nn.Linear(atom_fea_len, atom_fea_len, bias=False)
            for _ in range(n_conv_mag)
        ])

        # concatenation & MLP -- plain concatenation
        fusion_dim = 2 * atom_fea_len
        self.fusion_ln = nn.LayerNorm(fusion_dim)
        self.conv_to_fc = nn.Linear(fusion_dim, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        self.conv_to_fc_dropout = nn.Dropout(p=dropout_rate)
        self.res_fcs = nn.ModuleList([
            ResidualBlockMLP(h_fea_len=h_fea_len, dropout_rate=dropout_rate)
            for _ in range(n_h)
        ])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
    
    def forward(self, standard_data, magnetic_data, mag_to_struct_fea):
        # extract data
        cg_atom_fea, cg_nbr_fea, cg_nbr_idx, cg_batch_idx = standard_data
        mag_atom_fea, mag_nbr_fea, mag_nbr_idx, mag_batch_idx = magnetic_data

        # graph convolution with cross-coupling
        cg_fea = self.cg_embedding(cg_atom_fea)
        mag_fea = self.mag_embedding(mag_atom_fea)
        for cg_conv, mag_conv, cross_linear in zip(self.cg_convs, self.mag_convs, self.cross_linears):
            # 1. Update Crystal Graph
            cg_fea_updated = cg_conv(cg_fea, cg_nbr_fea, cg_nbr_idx)
            # 2. Extract features of magnetic atoms from the updated crystal graph
            mag_fea_struc_graph = cg_fea_updated[mag_to_struct_fea]
            # 3. Pass through layer-specific bias-free linear layer
            mag_fea_struc_graph = cross_linear(mag_fea_struc_graph)
            # 4. Residual addition into the magnetic graph
            mag_fea = mag_fea + mag_fea_struc_graph
            # 5. Update Magnetic Graph
            mag_fea_updated = mag_conv(mag_fea, mag_nbr_fea, mag_nbr_idx)
            # 6. Update the base features for the next loop iteration!
            cg_fea = cg_fea_updated
            mag_fea = mag_fea_updated
            
        # pooling
        cg_fea_pool = self.pooling(cg_fea, cg_batch_idx)
        mag_fea_pool = self.pooling(mag_fea, mag_batch_idx)

        # concatenation
        combined_fea = torch.cat([cg_fea_pool, mag_fea_pool], dim=1)

        # fusion
        combined_fea = self.fusion_ln(combined_fea)
        out_fea = self.conv_to_fc_softplus(self.conv_to_fc(combined_fea))
        out_fea = self.conv_to_fc_dropout(out_fea)

        # deep residual blocks as MLP
        for res_block in self.res_fcs:
            out_fea = res_block(out_fea)

        # output
        out = self.fc_out(out_fea)
        if self.classification:
            out = self.logsoftmax(out)

        return out


    def pooling(self, atom_fea, batch_idx):
        """Vectorized, native average pooling using PyTorch's index_add_."""
        num_graphs = batch_idx.max().item() + 1
        summed = torch.zeros(num_graphs, atom_fea.shape[1], device=atom_fea.device)
        summed.index_add_(0, batch_idx, atom_fea)
        
        counts = torch.zeros(num_graphs, device=atom_fea.device)
        ones = torch.ones_like(batch_idx, dtype=torch.float)
        counts.index_add_(0, batch_idx, ones)
        counts = torch.clamp(counts, min=1.0).unsqueeze(1)
        
        return summed / counts


