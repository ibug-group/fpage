import torch
from torch import nn
from .module import ChannelGate, ConvBlock


class FPAge(nn.Module):
    def __init__(self, n_blocks=4, age_classes=97, attention=True, face_classes=14):
        super(FPAge, self).__init__()

        self.age_vector = torch.arange(age_classes).float()
        self.age_classes = age_classes
        self.face_classes = face_classes
        self.attention = attention
        if self.attention:
            self.mid_channel = int(512 // face_classes)
            self.attened_channel = self.mid_channel * face_classes
            self.conv1x1_attened = nn.Conv2d(
                512, self.attened_channel, kernel_size=1, stride=1, padding=0
            )
            n_in_features = 512 + self.attened_channel
            self.se = ChannelGate(
                self.attened_channel,
                face_classes,
                reduction_ratio=16,
                pool_types=["avg"],
            )
        else:
            n_in_features = 512 * 2 + face_classes

        self.conv1x1 = nn.Conv2d(n_in_features, 256, kernel_size=1, stride=1, padding=0)
        n_features = [(256, 256)] * (n_blocks)

        convs = []
        for in_f, out_f in n_features:
            convs.append(ConvBlock(in_f, out_f))
            convs.append(nn.MaxPool2d(2, 2))
        self.age_net = nn.Sequential(*convs, nn.AdaptiveAvgPool2d(1))

        self.age_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, age_classes),
        )

    def get_attended_features(self, c2, high, logits):
        hm = logits.softmax(1)
        if self.attention:
            batch_size, _, h, w = high.shape
            tmp_out_new = self.conv1x1_attened(high)  # 506, 64, 64
            tmp_out_new = tmp_out_new.view(
                batch_size, -1, self.face_classes, h, w
            )  # 46, 11, 64, 64
            hm = hm.unsqueeze(1)  # 1, 11, 64, 64
            seg_feat_attended = tmp_out_new * hm
            seg_feat_attended = seg_feat_attended.view(batch_size, -1, h, w)
            if self.se is not None:
                seg_feat_attended = self.se(seg_feat_attended)
            feat = torch.cat([c2, seg_feat_attended], dim=1)
        else:
            feat = torch.cat([c2, high, hm], dim=1)
        return feat, logits

    def forward(self, c2, high, logits):
        self.age_vector = self.age_vector.to(c2.device)
        feat, logits = self.get_attended_features(c2, high, logits)
        feat = self.conv1x1(feat)
        age_features = self.age_net(feat)
        batch_size = age_features.shape[0]
        final_features = age_features.view(batch_size, -1)
        final_features = self.age_fc(final_features)
        pred = torch.squeeze(
            (final_features.softmax(1) * self.age_vector).sum(1, keepdim=True), dim=1
        )
        return pred
