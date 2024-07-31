#   Continet architecture for Classification of ModelNet40 Dataset 

import torch
import torch.nn as nn
import torch.nn.functional as F


# T-net (Spatial Transformer Network)
class Tnet(nn.Module):
    ''' T-Net learns a Transformation matrix with a specified dimension '''
    def __init__(self, dim, num_points=2500):
        super(Tnet, self).__init__()

        # dimensions for transform matrix
        self.dim = dim 

        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim**2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.max_pool = nn.MaxPool1d(kernel_size=num_points)

        # Dropout
        self.dropout = nn.Dropout(p=0.12)

    def forward(self, x):
        bs = x.shape[0]

        # pass through shared MLP layers (conv1d)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))

        # max pool over num points
        x = self.max_pool(x).view(bs, -1)
        
        # pass through MLP
        x = self.bn4(F.relu(self.linear1(x)))
        x = self.bn5(F.relu(self.linear2(x)))
        x = self.dropout(x)
        x = self.linear3(x)

        # initialize identity matrix
        iden = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()

        x = x.view(-1, self.dim, self.dim) + iden

        return x
    
class ContiNetBackbone(nn.Module):
    def __init__(self, num_points=2500, num_global_feats=1024, local_feat=True):
        ''' Initializers:
                num_points - number of points in point cloud
                num_global_feats - number of Global Features for the main 
                                   Max Pooling layer
                local_feat - if True, forward() returns the concatenation 
                             of the local and global features
            '''
        super(ContiNetBackbone, self).__init__()

        self.num_points = num_points
        self.num_global_feats = num_global_feats
        self.local_feat = local_feat
        

        # Spatial Transformer Networks (T-nets)
        self.tnet1 = Tnet(dim=3, num_points=num_points)
        self.tnet2 = Tnet(dim=64, num_points=num_points)

        # Convolutions layers 1
        self.conv1 = nn.Conv1d(6, 24, kernel_size=1)
        self.conv2 = nn.Conv1d(24, 64, kernel_size=1)  #(27, 64)

        # Convolutions layers 2
        self.conv3 = nn.Conv1d(64, 180, kernel_size=1)
        self.conv4 = nn.Conv1d(204, 600, kernel_size=1)
        self.conv5 = nn.Conv1d(603, self.num_global_feats, kernel_size=1)

        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(24)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(180)
        self.bn4 = nn.BatchNorm1d(600)
        self.bn5 = nn.BatchNorm1d(self.num_global_feats)

        # Max Pooling
        self.max_pool = nn.MaxPool1d(kernel_size=num_points, return_indices=True)


    def forward(self, x):

        # get batch size
        bs = x.shape[0]

        # storing the input x for feature concatenation 
        feature_0 = x.clone()

        # Pass through first Tnet to get transform matrix
        tnet_out_1 = (self.tnet1(x))

        # perform first transformation across each point in the batch(matrix-muliplication)
        x = torch.bmm(x.transpose(2, 1), tnet_out_1).transpose(2, 1)

        # storing the transformed input (first T-net Transformation)
        feature_1 = x.clone()  # T-net feature (nx3)

        # feature concatenation (feature_0 and feature_1)
        x = torch.cat((x, feature_0), dim=1)    #(nx6)

        x = self.bn1(F.relu(self.conv1(x)))   #(6, 24)
        #x = torch.cat((x, feature_1), dim=1)
        feature_2 = x.clone()     # (nx24)
        x = self.bn2(F.relu(self.conv2(x)))

        # get feature transform (T-net-2)
        tnet__2 = (self.tnet2(x))

        # perform second transformation across each (64 dim) feature in the batch
        x = (torch.bmm(x.transpose(2, 1), tnet__2).transpose(2, 1))
        tnet_2_out = x.clone()

        x = self.bn3(F.relu(self.conv3(x)))
        x = torch.cat((x, feature_2), dim=1)
        x = self.bn4(F.relu(self.conv4(x)))
        x = torch.cat((x, feature_1), dim=1)
        x = self.bn5(F.relu(self.conv5(x)))

        # get global feature vectore and critical indexes
        global_features, critical_indexes = self.max_pool(x)
        global_features =  global_features.view(bs, -1)
        critical_indexes = critical_indexes.view(bs, -1)

        if self.local_feat:
            features = torch.cat((tnet_2_out, 
                                  global_features.unsqueeze(-1).repeat(1, 1, self.num_points)),
                                  dim=1)
            return features, critical_indexes, tnet__2
        
        else:
            return global_features, critical_indexes, tnet__2
        


class ContiNetClassification(nn.Module):
    """Classification Head"""
    def __init__(self, num_points=2500, num_global_feats=1024, k=16):
        super(ContiNetClassification, self).__init__()

        # get the backbone (only need global features for classification)
        self.backbone = ContiNetBackbone(num_points, num_global_feats, local_feat=False)

        # MLP for classification
        self.linear1 = nn.Linear(num_global_feats, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, k)

        # batchnorm for the first 2 linear layers
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.4)

    def forward(self, x):
        # get global features
        x, crit_idxs, tnet__2 = self.backbone(x)

        x = self.bn1(F.relu(self.linear1(x)))
        x = self.bn2(F.relu(self.linear2(x)))
        x = self.dropout1(x)
        x = self.bn3(F.relu(self.linear3(x)))
        x = self.dropout2(x)
        x = self.linear4(x)

        # return logits
        return x, crit_idxs, tnet__2  


class ContiNetSegmentation(nn.Module):
    def __init__(self, num_points=4096, num_global_feats=1024, m=14):
        super(ContiNetSegmentation, self).__init__()

        self.num_points = num_points
        self.num_global_feats = num_global_feats
        self.m = m

        # Spatial Transformer Networks (T-nets)
        self.tnet1 = Tnet(dim=3, num_points=num_points)
        self.tnet2 = Tnet(dim=64, num_points=num_points)

        # 1d Convolution layers
        self.conv1 = nn.Conv1d(3, 27, kernel_size=1)
        self.conv2 = nn.Conv1d(27, 64, kernel_size=1)
        self.conv3 = nn.Conv1d(64, 256, kernel_size=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=1)
        self.conv5 = nn.Conv1d(512, self.num_global_feats, kernel_size=1)

        self.conv6 = nn.Conv1d(self.num_global_feats+512, 1050, kernel_size=1)
        self.conv7 = nn.Conv1d(1306, 812, kernel_size=1)
        self.conv8 = nn.Conv1d(940, 512, kernel_size=1)
        self.conv9 = nn.Conv1d(603, 412, kernel_size=1)
        self.conv11 = nn.Conv1d(412, 256, kernel_size=1)
        self.conv12 = nn.Conv1d(256, m, kernel_size=1)

        self.conv10 = nn.Conv1d(3, 64, kernel_size=1)

        # batch norms for both Convolution & MLPs
        self.bn1 = nn.BatchNorm1d(27)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(self.num_global_feats)
        self.bn6 = nn.BatchNorm1d(1050)
        self.bn7 = nn.BatchNorm1d(812)
        self.bn8 = nn.BatchNorm1d(512)
        self.bn9 = nn.BatchNorm1d(412)
        self.bn10 = nn.BatchNorm1d(64)
        self.bn11 = nn.BatchNorm1d(256)

        # max pool to get the global features
        self.max_pool = nn.MaxPool1d(kernel_size=num_points, return_indices=True)

        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, x):

        # Get the batch size
        bs = x.shape[0]

        # pass through the first T-net to get transformed input
        A_input = self.tnet1(x)

        # perform first transformation across each point in the batch
        x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)

        feature_00 = x.clone()           # (nx3) trasformed input
        feature_01 = self.bn10(F.relu(self.conv10(feature_00)))   # (nx64) features
        x = self.bn1(F.relu(self.conv1(x)))
        feature_1 = x.clone()           # (nx27) features
        x = self.bn2(F.relu(self.conv2(x)))
        feature_2 = x.clone()           # (nx64) features

        # pass through the second T-net to get transformed features(nx64)
        A_feat = self.tnet2(x)

        # perform first transformation across each point in the batch
        x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1)
        feature_02 = x.clone()          # (nx64) trasformed feature(T-net)

        x = self.bn3(F.relu(self.conv3(x)))
        feature_3 = x.clone()           # (nx256) features

        x = self.bn4(F.relu(self.conv4(x)))
        feature_4 = x.clone()             # (nx512) features

        x = self.bn5(F.relu(self.conv5(x)))

        # get global feature vector and critical indexes
        global_features, critical_indexes = self.max_pool(x)
        global_features = global_features.view(bs, -1)
        critical_indexes = critical_indexes.view(bs, -1)

        # After Max Pooling feature Concatenation of (nx512) & (self.num_global_feats)
        x = torch.cat((feature_4, 
                       global_features.unsqueeze(-1).repeat(1, 1, self.num_points)), 
                       dim=1)              # here x become size of (nx(512+self.num_global_feats))

        x = self.bn6(F.relu(self.conv6(x)))   # x.shape= (nx1050)
        # feature concatenate
        x = torch.cat((x, feature_3), dim=1)  # feature concatenate (nx1306) 1306=1050+256
        x = self.bn7(F.relu(self.conv7(x)))   # (nx812)
        x = torch.cat((x, feature_2, feature_02), dim=1)        # (nx940)
        x = self.bn8(F.relu(self.conv8(x)))    # (nx512)
        x = torch.cat((x, feature_1, feature_01), dim=1)         # (nx603)
        x = self.dropout1(x)
        x = self.bn9(F.relu(self.conv9(x)))                 # (nx412)
        x = self.dropout2(x)
        x = self.bn11(F.relu(self.conv11(x)))
        # Logits scores
        x = self.conv12(x)

        x = x.transpose(2, 1)

        return x, critical_indexes, A_feat


# ============================================================================
# This code is used for testing the architecture for any shape and size error 
# Test 
def main():
    test_data = torch.rand(32, 3, 2500)

    ## test T-net
    tnet = Tnet(dim=3)
    transform = tnet(test_data)
    print(f'T-net output shape: {transform.shape}')

    ## test backbone
    pointfeat = ContiNetBackbone(local_feat=False)
    out, _, _ = pointfeat(test_data)
    print(f'Global Features shape: {out.shape}')

    pointfeat = ContiNetBackbone(local_feat=True)
    out, _, _ = pointfeat(test_data)
    print(f'Combined Features shape: {out.shape}')

    # test on single batch (should throw error if there is an issue)
    pointfeat = ContiNetBackbone(local_feat=True).eval()
    out, _, _ = pointfeat(test_data[0, :, :].unsqueeze(0))

    ## test classification head
    classifier = ContiNetClassification(k=5)
    out, _, _ = classifier(test_data)
    print(f'Class output shape: {out.shape}')

    classifier = ContiNetClassification(k=5).eval()
    out, _, _ = classifier(test_data[0, :, :].unsqueeze(0))

    ## test segmentation head
    seg = ContiNetSegmentation(m=3)
    out, _, _ = seg(test_data)
    print(f'Seg shape: {out.shape}')


if __name__ == '__main__':
    main()











