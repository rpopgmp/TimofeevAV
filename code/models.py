import torch
import torch.nn as nn


class _Bottleneck1D(nn.Module):
    def __init__(self, in_ch, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return self.relu(self.bn3(self.conv3(out)) + identity)


class ResNet50_1D_ECG(nn.Module):
    def __init__(self, n_tab_features, in_channels=12):
        super().__init__()
        self.in_planes = 64
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1 = self._stage(64, 3, 1)
        self.layer2 = self._stage(128, 4, 2)
        self.layer3 = self._stage(256, 6, 2)
        self.layer4 = self._stage(512, 3, 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.tab_mlp = nn.Sequential(
            nn.Linear(n_tab_features, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(2048 + 64, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, 1),
        )

    def _stage(self, planes, blocks, stride):
        out_planes = planes * 4
        downsample = None
        if stride != 1 or self.in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_planes, out_planes, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_planes),
            )
        layers = [_Bottleneck1D(self.in_planes, planes, stride, downsample)]
        self.in_planes = out_planes
        for _ in range(1, blocks):
            layers.append(_Bottleneck1D(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, tab):
        x = self.pool(self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))).squeeze(-1)
        return self.head(torch.cat([x, self.tab_mlp(tab)], dim=1))

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask.div(keep_prob)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        y = self.norm1(x)
        y = self.attn(y, y, y, need_weights=False)[0]
        x = x + self.drop_path1(y)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class ECGTransformer(nn.Module):
    def __init__(self, n_tab_features, in_channels=12, d_model=512,
                 nhead=8, num_layers=8, dim_feedforward=2048,
                 dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, d_model, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )

        n_tokens = 2800 // 8
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if num_layers > 1:
            dpr = torch.linspace(0, drop_path_rate, num_layers).tolist()
        else:
            dpr = [drop_path_rate]

        self.encoder = nn.Sequential(*[
            TransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                drop_path=dpr[i],
            )
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.tab_mlp = nn.Sequential(
            nn.Linear(n_tab_features, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(d_model + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x, tab):
        tab = torch.nan_to_num(tab, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.stem(x).transpose(1, 2)
        b, t, _ = x.shape
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed[:, :t + 1]
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        tab = self.tab_mlp(tab)
        return self.head(torch.cat([x, tab], dim=1))
