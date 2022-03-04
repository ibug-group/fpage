from pathlib import Path
import cv2
import numpy as np
from timm.models import create_model
from torch import nn
import torch
import torchvision.transforms as T
import ibug.roi_tanh_warping.reference_impl as ref

from .age_estimator import hflip_bbox
from PIL import Image

WEIGHT = {
    "resnet50-dex-101": (
        Path(__file__).parent / "weights/resnet50-dex-101.pth.tar",
        0.5,
        0.5,
        (224, 224),
    ),
    "resnet50-mv-101": (
        Path(__file__).parent / "weights/resnet50-mv-101.pth.tar",
        0.5,
        0.5,
        (224, 224),
    ),
    "resnet50-dldlv2-101": (
        Path(__file__).parent / "weights/resnet50-dldlv2-101.pth.tar",
        0.5,
        0.5,
        (224, 224),
    ),
    "resnet50-dldl-101": (
        Path(__file__).parent / "weights/resnet50-dldl-101.pth.tar",
        0.5,
        0.5,
        (224, 224),
    ),
    "resnet50-ord-101": (
        Path(__file__).parent / "weights/resnet50-ord-101.pth.tar",
        0.5,
        0.5,
        (224, 224),
    ),
}


class DEX(nn.Module):
    def __init__(
        self,
        encoder,
        loss="dex",
        ckpt=None,
        age_classes=101,
        warping="crop-resize",
        device="cuda:0",
        flip_eval=True,
    ):
        super(DEX, self).__init__()
        self.device = device
        self.age_classes = age_classes
        model_name = "-".join([encoder, loss, str(age_classes)])
        assert model_name in WEIGHT, "Availabe baseline models {}".format(WEIGHT.keys())
        assert warping in ["roi-tp", "crop-resize"]

        pretrained_ckpt, mean, std, sz = WEIGHT[model_name]
        self.model_name = model_name
        if loss.lower() == "ord":
            age_classes = (age_classes - 1) * 2
        self.model = create_model(
            encoder,
            checkpoint_path=ckpt if ckpt else pretrained_ckpt,
            num_classes=age_classes,
        )
        self.sz = sz
        self.warping = warping
        tsfms = [T.ToTensor(), T.Normalize(mean, std)]
        if self.warping == "crop-resize":
            tsfms = [T.Resize(self.sz)] + tsfms
        self.transform = T.Compose(tsfms)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.vector = torch.arange(age_classes).to(self.device)
        self.flip_eval = flip_eval

    @torch.no_grad()
    def predict_img(self, img, bboxes, rgb=False):
        # import ipdb; ipdb.set_trace()
        if isinstance(img, str):
            img = cv2.imread(img)
        elif isinstance(img, np.ndarray):
            if rgb:
                img = img[:, :, ::-1]
        else:
            raise TypeError
        h, w = img.shape[:2]

        num_faces = len(bboxes)

        if self.warping == "roi-tp":
            imgs = [
                ref.roi_tanh_polar_warp(img, b, *self.sz, keep_aspect_ratio=True)
                for b in bboxes
            ]
        elif self.warping == "crop-resize":
            imgs = [crop_face_with_margin(img, b)[0] for b in bboxes]
        if self.flip_eval:
            if self.warping == "roi-tp":
                img_flip, bboxes_flip = self.flip_image(img, bboxes)
                imgs += [
                    ref.roi_tanh_polar_warp(
                        img_flip, b, *self.sz, keep_aspect_ratio=True
                    )
                    for b in bboxes_flip
                ]
            elif self.warping == "crop-resize":
                imgs += [np.fliplr(img) for img in imgs]
            num_faces *= 2

        imgs = [Image.fromarray(i) for i in imgs]
        imgs = [self.transform(img) for img in imgs]

        img = torch.stack(imgs).to(self.device)

        pred_logits = self.model(img)
        if "ord" in self.model_name:
            B = pred_logits.shape[0]
            probas = pred_logits.view(B, -1, 2).softmax(dim=2)[:, :, 1]
            pred = probas > 0.5
            age_pred = torch.sum(pred, dim=1).float() - 1

        else:
            age_pred = torch.squeeze(
                (pred_logits.softmax(1) * self.vector).sum(1, keepdim=True), dim=1
            )

        if self.flip_eval:
            age_pred, age_pred_flip = torch.chunk(age_pred, 2)
            age_pred += age_pred_flip
            age_pred /= 2

        return age_pred

    def flip_image(self, img, bboxes):
        width = img.shape[1]
        flip_bboxes = hflip_bbox(np.array(bboxes), width)
        flip_img = np.fliplr(img)
        return flip_img, flip_bboxes


def crop_face_with_margin(img, box, landmark=None, crop_margin=[0.4, 0.4, 0.4, 0.4]):
    """
    img: H,W,3 array
    box: x1,y1,x2,y2 list
    landmark: N,2 array
    adapted from https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/extractSubImage.m
    """
    box = box[:4]
    if isinstance(crop_margin, (float, int)):

        crop_margin = [float(crop_margin)] * 4
    elif isinstance(crop_margin, (list, tuple)):
        assert (
            len(crop_margin) == 4
        ), f"crop_margin has to be a float value or a list of four margins, got{crop_margin}"
    else:
        raise ValueError

    h, w = img.shape[:2]

    is_color = len(img.shape) > 2

    # size of face
    orig_size = [0, 0]
    orig_size[0] = box[3] - box[1] + 1
    orig_size[1] = box[2] - box[0] + 1

    # add margin
    full_crop = [0, 0, 0, 0]
    full_crop[0] = round(box[0] - crop_margin[0] * orig_size[1])
    full_crop[1] = round(box[1] - crop_margin[1] * orig_size[0])
    full_crop[2] = round(box[2] + crop_margin[2] * orig_size[1])
    full_crop[3] = round(box[3] + crop_margin[3] * orig_size[0])

    # size of face with margin
    new_size = [0, 0]
    new_size[0] = full_crop[3] - full_crop[1] + 1
    new_size[1] = full_crop[2] - full_crop[0] + 1

    # ensure that the region cropped from the original image with margin doesn't go beyond the image size
    crop = [0, 0, 0, 0]
    crop[0] = max(full_crop[0], 0)
    crop[1] = max(full_crop[1], 0)
    crop[2] = min(full_crop[2], w - 1)
    crop[3] = min(full_crop[3], h - 1)

    # size of the actual region being cropped from the original image
    crop_size = [0, 0]
    crop_size[0] = crop[3] - crop[1] + 1
    crop_size[1] = crop[2] - crop[0] + 1

    if is_color:
        new_img = np.zeros(new_size + [3], dtype=np.uint8)
    else:
        new_img = np.zeros(new_size, dtype=np.uint8)
    # coordinates of region taken out of the original image in the new image
    new_location = [0, 0, 0, 0]
    new_location[0] = crop[0] - full_crop[0]
    new_location[1] = crop[1] - full_crop[1]
    new_location[2] = crop[0] - full_crop[0] + crop_size[1] - 1
    new_location[3] = crop[1] - full_crop[1] + crop_size[0] - 1

    # # coordinates of the face in the new image
    new_box = [0, 0, 0, 0]
    new_box[0] = new_location[0] + box[0] - crop[0]
    new_box[1] = new_location[1] + box[1] - crop[1]
    new_box[2] = new_location[2] + box[2] - crop[2]
    new_box[3] = new_location[3] + box[3] - crop[3]
    new_box = np.array(new_box, int)
    # do the crop
    new_img[
        new_location[1] : new_location[3], new_location[0] : new_location[2], ...
    ] = img[crop[1] : crop[3], crop[0] : crop[2], ...]

    # landmark has to be (N, 2)
    if landmark is not None:
        landmark = np.array(landmark).reshape(-1, 2).astype(int)
        diff = np.array(full_crop[:2]).reshape(1, 2)
        landmark = landmark - diff
        return new_img, new_box, landmark
    else:
        return new_img, new_box
