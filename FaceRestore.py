import os
import logging
import numpy as np
import cv2
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import normalize

from fr_basicsr.utils import img2tensor, tensor2img
from fr_facelib.utils.face_restoration_helper import FaceRestoreHelper
from fr_facelib.utils import load_file_from_url

import comfy.model_management as model_management
import comfy.utils
import folder_paths

current_paths = [os.path.join(folder_paths.models_dir, "facerestorer")]
folder_paths.folder_names_and_paths["facerestorer"] = (
    current_paths,
    folder_paths.supported_pt_extensions,
)


logger = logging.getLogger("FaceRestorer")
logger.setLevel(logging.INFO)


arch_model_path = {
    "RestoreFormer": "https://huggingface.co/tungdop2/FaceRestorer/resolve/main/RestoreFormer.ckpt",
    "RestoreFormer++": "https://huggingface.co/tungdop2/FaceRestorer/resolve/main/RestoreFormer++.ckpt",
    # "GFPGANv1.4": "https://huggingface.co/tungdop2/FaceRestorer/resolve/main/GFPGANv1.4.pth",
}


class FaceRestorerLoader:
    """Helper for restoration with FaceRestorer.

    It will detect and crop faces, and then resize the faces to 512x512.
    FaceRestorer is used to restored the resized faces.
    The background is upsampled with the bg_upsampler.
    Finally, the faces will be pasted back to the upsample background image.

    Args:
        model_path (str): The path to the GFPGAN model. It can be urls (will first download it automatically).
        upscale (float): The upscale of the final output. Default: 2.
        arch (str): The architecture. Option: RestoreFormer | RestoreFormer++. Default: RestoreFormer++.
        bg_upsampler (nn.Module): The upsampler for the background. Default: None.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"arch": (list(arch_model_path.keys()), {"default": "RestoreFormer++"})}}

    RETURN_TYPES = ("FACE_RESTORE_MODEL",)

    FUNCTION = "main"

    CATEGORY = "face_restorer"

    def main(self, arch="RestoreFormer++"):
        if arch == "RestoreFormer":
            from .archs.arch_RestoreFormer import VQVAEGANMultiHeadTransformer

            RF = VQVAEGANMultiHeadTransformer(head_size=8, ex_multi_scale_num=0)
        elif arch == "RestoreFormer++":
            from .archs.arch_RestoreFormer import VQVAEGANMultiHeadTransformer

            RF = VQVAEGANMultiHeadTransformer(head_size=4, ex_multi_scale_num=1)
        else:
            raise NotImplementedError(f"Not support arch: {arch}.")

        device = model_management.get_torch_device()

        model_path = load_file_from_url(
            arch_model_path[arch],
            model_dir="facerestorer",
            progress=True,
            file_name=None,
            save_dir=current_paths[0],
        )

        weights = comfy.utils.load_torch_file(
            model_path,
            safe_load=True,
            device=device,
        )
        strict = False
        new_weights = {}
        for k, v in weights.items():
            if k.startswith("vqvae."):
                k = k.replace("vqvae.", "")
            new_weights[k] = v
        RF.load_state_dict(new_weights, strict=strict)
        RF.eval().to(device)

        return (RF, )


class FaceRestorer:
    def __init__(self):
        self.device = model_management.get_torch_device()

        # initialize face helper
        self.face_helper = FaceRestoreHelper(
            1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self.device,
            model_rootpath="models/facexlib",
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "face_restore_model": ("FACE_RESTORE_MODEL",),
                "has_aligned": (
                    "BOOLEAN",
                    {"default": False, "label_off": "false", "label_on": "true"},
                ),
                "only_center_face": (
                    "BOOLEAN",
                    {"default": False, "label_off": "false", "label_on": "true"},
                ),
                "paste_back": (
                    "BOOLEAN",
                    {"default": True, "label_off": "false", "label_on": "true"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "enhance"

    CATEGORY = "face_restorer"

    @torch.no_grad()
    def enhance(
        self, input_image, face_restore_model, has_aligned, only_center_face, paste_back
    ):
        restored_imgs = []
        transform = T.ToPILImage()
        for item in input_image:

            self.face_helper.clean_all()
            img = transform(item.permute(2, 0, 1))
            img = np.array(img)

            if has_aligned:  # the inputs are already aligned
                img = cv2.resize(img, (512, 512))
                self.face_helper.cropped_faces = [img]
            else:
                self.face_helper.read_image(img)
                self.face_helper.get_face_landmarks_5(
                    only_center_face=only_center_face, eye_dist_threshold=5
                )
                # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
                # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
                # align and warp each face
                self.face_helper.align_warp_face()

            # face restoration
            for cropped_face in self.face_helper.cropped_faces:
                # prepare data
                cropped_face_t = img2tensor(
                    cropped_face / 255.0, bgr2rgb=True, float32=True
                )
                normalize(
                    cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True
                )
                cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

                try:
                    output = face_restore_model(cropped_face_t)[0]
                    restored_face = tensor2img(
                        output.squeeze(0), rgb2bgr=True, min_max=(-1, 1)
                    )
                except RuntimeError as error:
                    print(f"\tFailed inference for RestoreFormer: {error}.")
                    restored_face = cropped_face

                restored_face = restored_face.astype("uint8")
                self.face_helper.add_restored_face(restored_face)

            if not has_aligned and paste_back:
                # # upsample the background
                # if bg_upsampler is not None:
                #     # Now only support RealESRGAN for upsampling background
                #     bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
                # else:
                #     bg_img = None

                self.face_helper.get_inverse_affine(None)
                # paste each restored face to the input image
                restored_img = self.face_helper.paste_faces_to_input_image(
                    upsample_img=None
                )
                restored_img = TF.to_tensor(restored_img)
                restored_imgs.append(restored_img)
            else:
                for restored_face in self.face_helper.restored_faces:
                    restored_imgs.append(TF.to_tensor(restored_face))

        return (torch.stack(restored_imgs, dim=0).permute(0, 2, 3, 1),)
