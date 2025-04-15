import os
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy

import torch
import torchvision
from torch import nn

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"

# from loguru import logger


@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)





def create(image_list, data_list, save_folder, precompute_seg_path = None):
    assert image_list is not None, "image_list must be provided to generate features"
    embed_size=512
    seg_maps = []
    total_lengths = []
    timer = 0
    img_embeds = torch.zeros((len(image_list), 300, embed_size))
    if precompute_seg_path is not None:
        seg_maps = torch.zeros((len(image_list), 4, *image_list[0].shape[1:])) 
        pre_compute_seg_list = os.listdir(precompute_seg_path)
        pre_compute_seg_list.sort()
    else:
        seg_maps = torch.zeros((len(image_list), 4, *image_list[0].shape[1:])) 
    # mask_generator.predictor.model.to('cuda')
    for i, img in tqdm(enumerate(image_list), desc="Embedding images", leave=False):
        timer += 1
        if precompute_seg_path is not None:
            seg_path = os.path.join(precompute_seg_path,pre_compute_seg_list[i])
        else:
            seg_path = None
        try:
            img_embed, seg_map = _embed_clip_sam_tiles(img.unsqueeze(0), sam_encoder, seg_path)
        except:
            import ipdb; ipdb.set_trace()
            raise ValueError(timer)

        lengths = [len(v) for k, v in img_embed.items()]
        total_length = sum(lengths)
        total_lengths.append(total_length)
        
        if total_length > img_embeds.shape[1]:
            pad = total_length - img_embeds.shape[1]
            img_embeds = torch.cat([
                img_embeds,
                torch.zeros((len(image_list), pad, embed_size))
            ], dim=1)

        img_embed = torch.cat([v for k, v in img_embed.items()], dim=0)
        assert img_embed.shape[0] == total_length
        img_embeds[i, :total_length] = img_embed
        
        seg_map_tensor = []
        lengths_cumsum = lengths.copy()
        for j in range(1, len(lengths)):
            lengths_cumsum[j] += lengths_cumsum[j-1]
        for j, (k, v) in enumerate(seg_map.items()):
            if j == 0:
                seg_map_tensor.append(torch.from_numpy(v))
                continue
            # if v.max() != lengths[j] - 1:
            #     import ipdb; ipdb.set_trace()
            assert v.max() == lengths[j] - 1, f"{j}, {v.max()}, {lengths[j]-1}"
            v[v != -1] += lengths_cumsum[j-1]
            seg_map_tensor.append(torch.from_numpy(v))
        seg_map = torch.stack(seg_map_tensor, dim=0)
        seg_maps[i] = seg_map

    # mask_generator.predictor.model.to('cpu')
        
    for i in range(img_embeds.shape[0]):
        save_path = os.path.join(save_folder, data_list[i].split('.')[0])
        assert total_lengths[i] == int(seg_maps[i].max() + 1)
        curr = {
            'feature': img_embeds[i, :total_lengths[i]],
            'seg_maps': seg_maps[i]
        }
        sava_numpy(save_path, curr)

def sava_numpy(save_path, data):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, data['seg_maps'].numpy())
    np.save(save_path_f, data['feature'].numpy())

def _embed_clip_sam_tiles(image, sam_encoder, precomp_seg_path=None):
    aug_imgs = torch.cat([image])
    seg_images, seg_map = sam_encoder(aug_imgs, precomp_seg_path)

    clip_embeds = {}
    # if precomp_seg_path:
    #     mode = 'video'
    #     tiles = seg_images[mode]
    #     tiles = tiles.to("cuda")
    #     # import ipdb; ipdb.set_trace()
    #     with torch.no_grad():
    #         clip_embed = model.encode_image(tiles)
    #     clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
    #     clip_embeds[mode] = clip_embed.detach().cpu().half()
    # else:
    for mode in ['default', 's', 'm', 'l']:
        tiles = seg_images[mode]
        tiles = tiles.to("cuda")
        with torch.no_grad():
            clip_embed = model.encode_image(tiles)
        clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
        clip_embeds[mode] = clip_embed.detach().cpu().half()
    
    return clip_embeds, seg_map

def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def sam_encoder(image, precomp_seg_path=None):

    image = cv2.cvtColor(image[0].permute(1,2,0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)

    # if precomp_seg_path is not None:
    assert precomp_seg_path is not None, "precomp_seg_path must be provided to generate features"
    mask_video_np = np.load(precomp_seg_path)
    mask_all = []
    # import ipdb; ipdb.set_trace()
    for i in range(4):
        mask_ans = []
        # import ipdb; ipdb.set_trace()
        for j in range(1,int(mask_video_np[i].max())+1):
            positions = np.where(mask_video_np[i] == j)
            if len(positions[0]) == 0:
                continue

            # 计算边界框的坐标
            y_min, y_max = positions[0].min(), positions[0].max()
            x_min, x_max = positions[1].min(), positions[1].max()

            # 存储边界框信息
            mask_ans.append({
                'segmentation': mask_video_np[i] == j,
                'label': j,
                'bbox': [x_min, y_min, x_max-x_min, y_max-y_min]
            })
        mask_all.append(mask_ans)
    masks_default, masks_s, masks_m, masks_l = mask_all[0], mask_all[1], mask_all[2], mask_all[3]

    # else:
    #     # pre-compute masks
    #     masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    #     # pre-compute postprocess
    #     masks_default, masks_s, masks_m, masks_l = \
    #         masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    
    def mask2segmap(masks, image):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        filter_masks = [] # 删去bbox宽度大于0的mask
        for mask in masks:
            if mask['bbox'][2] != 0 and mask['bbox'][3] != 0:
                filter_masks.append(mask)
        masks = filter_masks
        for i in range(len(masks)):
            mask = masks[i]
            
            seg_img = get_seg_img(mask, image)
            ans = pad_img(seg_img)
            # if len(ans) == 0:
            #     import ipdb; ipdb.set_trace()
            # print(ans.shape)
            pad_seg_img = cv2.resize(ans, (224,224))
            seg_img_list.append(pad_seg_img)

            seg_map[masks[i]['segmentation']] = i
        seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

        return seg_imgs, seg_map

    seg_images, seg_maps = {}, {}
    # import ipdb; ipdb.set_trace()
    # if precomp_seg_path is not None:
    #     seg_images['video'], seg_maps['video'] = mask2segmap(mask_video, image)
    # else:
    seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image)
    if len(masks_s) != 0:
        seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image)
    if len(masks_m) != 0:
        seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image)
    if len(masks_l) != 0:
        seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image)
    
    # 0:default 1:s 2:m 3:l
    return seg_images, seg_maps

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_type', choices=['hypernerf','dynerf'],default='hypernerf')
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str, default="ckpts/sam_vit_h_4b8939.pth")
    parser.add_argument('--precompute_seg', type=str, default=None, help="path to the segmentation computed by other models (e.g. video tracker)")
    parser.add_argument('--output_name',type=str,default="language_features_video")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    # import time
    # timestamp = time.strftime("%Y%m%d_%H%M%S")
    # log_file_name = f"log_{timestamp}.log"

    # logger.add(os.path.join("log","log_file_name"), rotation="500 MB")  # 将日志写入文件，当文件大小达到500MB时进行轮转
    if args.dataset_type == 'hypernerf':
        resolution_list = os.listdir(os.path.join(dataset_path, 'rgb'))
        resolution_list = ['2x'] #! 节约时间，只提取2x的
        for resolution in resolution_list:
            
            img_folder = os.path.join(dataset_path, 'rgb',resolution)
            # logger.info(f"img_folder: {img_folder}")
            save_folder = os.path.join(dataset_path, 'language_features' if args.precompute_seg is None else args.output_name) 

            os.makedirs( os.path.join(dataset_path, resolution), exist_ok=True)
            os.makedirs(save_folder, exist_ok=True)
            # logger.info(f"save_folder: {save_folder}")
            data_list = os.listdir(img_folder)
            data_list.sort()

            model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
            # sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
            # mask_generator = SamAutomaticMaskGenerator(
            #     model=sam,
            #     points_per_side=32,
            #     pred_iou_thresh=0.7,
            #     box_nms_thresh=0.7,
            #     stability_score_thresh=0.85,
            #     crop_n_layers=1,
            #     crop_n_points_downscale_factor=1,
            #     min_mask_region_area=100,
            # )

            img_list = []
            WARNED = False
            for data_path in data_list:
                image_path = os.path.join(img_folder, data_path)
                image = cv2.imread(image_path)

                orig_w, orig_h = image.shape[1], image.shape[0]
                if args.resolution == -1:
                    if orig_h > 1080:
                        if not WARNED:
                            print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                                "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                            WARNED = True
                        global_down = orig_h / 1080
                    else:
                        global_down = 1
                else:
                    global_down = orig_w / args.resolution
                    
                scale = float(global_down)
                resolution = (int( orig_w  / scale), int(orig_h / scale))
                
                image = cv2.resize(image, resolution)
                image = torch.from_numpy(image)
                img_list.append(image)
            images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
            imgs = torch.cat(images)


            create(imgs, data_list, save_folder, args.precompute_seg)
    elif args.dataset_type == 'dynerf':
        cam_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('cam')]
        for cam_dir in tqdm(cam_dirs):
            img_folder = os.path.join(dataset_path, cam_dir, 'images')
            # logger.info(f"img_folder: {img_folder}")
            save_folder = os.path.join(dataset_path, cam_dir, 'language_features' if args.precompute_seg is None else args.output_name)
            if os.path.exists(save_folder):
                # logger.info(f"{save_folder} already exist, skip to another one")
                continue
            os.makedirs( os.path.join(dataset_path, cam_dir), exist_ok=True)
            os.makedirs(save_folder, exist_ok=True)
            # logger.info(f"save_folder: {save_folder}")
            data_list = os.listdir(img_folder)
            data_list.sort()

            model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
            # sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
            # mask_generator = SamAutomaticMaskGenerator(
            #     model=sam,
            #     points_per_side=32,
            #     pred_iou_thresh=0.7,
            #     box_nms_thresh=0.7,
            #     stability_score_thresh=0.85,
            #     crop_n_layers=1,
            #     crop_n_points_downscale_factor=1,
            #     min_mask_region_area=100,
            # )

            img_list = []
            WARNED = False
            for data_path in data_list:
                image_path = os.path.join(img_folder, data_path)
                image = cv2.imread(image_path)

                orig_w, orig_h = image.shape[1], image.shape[0]
                if args.resolution == -1:
                    if orig_h > 1080:
                        if not WARNED:
                            print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                                "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                            WARNED = True
                        global_down = orig_h / 1080
                    else:
                        global_down = 1
                else:
                    global_down = orig_w / args.resolution
                    
                scale = float(global_down)
                resolution = (int( orig_w  / scale), int(orig_h / scale))
                
                image = cv2.resize(image, resolution)
                image = torch.from_numpy(image)
                img_list.append(image)
            images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
            imgs = torch.cat(images)


            create(imgs, data_list, save_folder, os.path.join(args.precompute_seg,cam_dir,"concat_npy"))
    else:
        raise NotImplementedError
