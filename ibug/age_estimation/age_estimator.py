from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models._utils import IntermediateLayerGetter
import ibug.roi_tanh_warping.reference_impl as ref
from ibug.face_parsing.parser import FaceParser
from .fpage import FPAge

WEIGHT = {
    # 'resnet50-fcn-14-97': (Path(__file__).parent / 'weights/resnet50-fcn-14-97.torch', 0.5, 0.5, (513, 513)),
    "resnet50-fcn-14-97": (
        Path(__file__).parent / "weights/fpage-resnet50-fcn-14-97.torch",
        0.5,
        0.5,
        (513, 513),
    ),
}


def hflip_bbox(bbox, W):
    """Flip bounding boxes horizontally.
    Args:
        bbox (~numpy.ndarray): See the table below.
        W: width of the image
    .. csv-table::
        :header: name, shape, dtype, format
        :obj:`bbox`, ":math:`(R, 4)`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
    Returns:
        ~numpy.ndarray:
        Bounding boxes flipped according to the given flips.
    """
    bbox = bbox.copy()
    x_max = W - bbox[:, 1]
    x_min = W - bbox[:, 3]
    bbox[:, 1] = x_min
    bbox[:, 3] = x_max
    return bbox


class AgeEstimator(FaceParser):
    def __init__(
        self,
        device="cuda:0",
        ckpt=None,
        encoder="resnet50",
        decoder="fcn",
        face_classes=14,
        n_blocks=4,
        age_classes=97,
        flip_eval=True,
    ):
        self.device = device
        model_name = "-".join([encoder, decoder, str(face_classes), str(age_classes)])
        assert model_name in WEIGHT, "Availabe models {}".format(WEIGHT.keys())

        self.face_parser = FaceParser(
            device=device, encoder=encoder, decoder=decoder, num_classes=face_classes
        ).model
        self.low_level = getattr(self.face_parser.decoder, "low_level", False)
        if decoder == "fcn":
            self.face_parser.decoder = IntermediateLayerGetter(
                self.face_parser.decoder, {"2": "high", "4": "logits"}
            )
        else:
            raise NotImplementedError
        self.age_estimator = FPAge(
            n_blocks=n_blocks, age_classes=age_classes, face_classes=face_classes
        )
        pretrained_ckpt, mean, std, sz = WEIGHT[model_name]
        self.sz = sz
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        if ckpt is None:
            ckpt = pretrained_ckpt
        ckpt = torch.load(ckpt, "cpu")
        ckpt = ckpt.get("state_dict", ckpt)
        self.age_estimator.load_state_dict(ckpt, True)
        self.age_estimator.eval()
        self.age_estimator.to(device)
        self.flip_eval = flip_eval
        self.face_classes = face_classes

    def face_parser_forward(self, x, rois):
        input_shape = x.shape[-2:]
        features = self.face_parser.encoder(x, rois)
        out = self.face_parser.decoder(features["c4"])
        logits, high = out["logits"], out["high"]
        c2 = features["c2"]
        mask = F.interpolate(
            logits, size=input_shape, mode="bilinear", align_corners=False
        )

        return c2, high, logits, mask

    @torch.no_grad()
    def predict_img(self, img, bboxes, rgb=False):
        if isinstance(img, str):
            img = cv2.imread(img)
        elif isinstance(img, np.ndarray):
            if rgb:
                img = img[:, :, ::-1]
        else:
            raise TypeError
        h, w = img.shape[:2]

        num_faces = len(bboxes)

        imgs = [
            ref.roi_tanh_polar_warp(img, b, *self.sz, keep_aspect_ratio=True)
            for b in bboxes
        ]

        if self.flip_eval:
            img_flip, bboxes_flip = self.flip_image(img, bboxes)
            imgs += [
                ref.roi_tanh_polar_warp(img_flip, b, *self.sz, keep_aspect_ratio=True)
                for b in bboxes_flip
            ]
            bboxes = np.concatenate([bboxes, bboxes_flip], axis=0)
            num_faces *= 2

        bboxes_tensor = torch.tensor(bboxes).to(self.device).view(num_faces, -1)

        imgs = [self.transform(img) for img in imgs]

        img = torch.stack(imgs).to(self.device)
        low, high, logits, mask = self.face_parser_forward(img, bboxes_tensor)
        age_pred = self.age_estimator(low, high, logits)
        if self.flip_eval:
            age_pred, age_pred_flip = torch.chunk(age_pred, 2)
            age_pred += age_pred_flip
            age_pred /= 2

        if self.flip_eval:
            mask = torch.chunk(mask, 2)[0]
        mask = self.restore_warp(h, w, mask, bboxes_tensor)
        return age_pred, mask

    def flip_image(self, img, bboxes):
        width = img.shape[1]
        flip_bboxes = hflip_bbox(np.array(bboxes), width)
        flip_img = np.fliplr(img)
        return flip_img, flip_bboxes
