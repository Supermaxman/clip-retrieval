"""mapper module transform images and text to embeddings"""

import torch
from clip_retrieval.load_clip import load_clip
from sentence_transformers import SentenceTransformer


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class ClipMapper:
    """transforms images and texts into clip embeddings"""

    def __init__(
        self,
        enable_image,
        enable_text,
        enable_metadata,
        enable_joint_text_image,
        use_mclip,
        clip_model,
        use_jit,
        mclip_model,
        warmup_batch_size=1,
        clip_cache_path=None,
    ):
        self.enable_image = enable_image
        self.enable_text = enable_text
        self.enable_metadata = enable_metadata
        self.enable_joint_text_image = enable_joint_text_image
        self.use_mclip = use_mclip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = load_clip(
            clip_model=clip_model,
            use_jit=use_jit,
            warmup_batch_size=warmup_batch_size,
            clip_cache_path=clip_cache_path,
        )
        self.model_img = model.encode_image
        self.model_txt = model.encode_text
        if use_mclip:
            print("\nLoading MCLIP model for text embedding\n")
            mclip = SentenceTransformer(mclip_model)
            self.model_txt = mclip.encode

    def __call__(self, item):
        with torch.no_grad():
            image_embs = None
            text_embs = None
            joint_embs = None
            image_filename = None
            text = None
            metadata = None
            if self.enable_image:
                image_features = self.model_img(item["image_tensor"].to(self.device))
                image_features_norm = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                image_embs = image_features_norm.cpu().to(torch.float16).numpy()
                image_filename = item["image_filename"]
            if self.enable_text:
                if self.use_mclip:
                    text_embs = normalized(self.model_txt(item["text"]))
                else:
                    text_features = self.model_txt(item["text_tokens"].to(self.device))
                    text_features_norm = text_features / text_features.norm(
                        dim=-1, keepdim=True
                    )
                    text_embs = text_features_norm.cpu().to(torch.float16).numpy()
                text = item["text"]
            if self.enable_metadata:
                metadata = item["metadata"]
            if self.enable_joint_text_image:
                # mean of image and text embeddings
                joint_features = (image_features + text_features) / 2.0
                joint_features_norm = joint_features / joint_features.norm(
                    dim=-1, keepdim=True
                )
                joint_embs = joint_features_norm.cpu().to(torch.float16).numpy()

            return {
                "image_embs": image_embs,
                "text_embs": text_embs,
                "joint_embs": joint_embs,
                "image_filename": image_filename,
                "text": text,
                "metadata": metadata,
            }
