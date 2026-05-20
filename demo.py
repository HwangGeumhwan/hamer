from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.utils.render_openpose import render_openpose

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def _vector_normalization_3d(joint):
    """HaMeR 3D 키포인트 기반 관절 벡터/각도 계산 (joint: (21, 3))

    Returns:
        v     : (20, 3) 정규화된 관절 방향 벡터
        angle : (15,)  인접 뼈대 간 각도 (degrees)
    Feature dim: vector 60 + angle 15 = 75-dim
    """
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19]]  # parent joints
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]  # child joints
    v = v2 - v1
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    angle = np.arccos(np.clip(np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18]],
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]]), -1.0, 1.0))
    return v, np.degrees(angle).astype(np.float32)

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--skeleton', dest='skeleton', action='store_true', default=False, help='If set, render skeleton instead of mesh')

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]

    # Iterate over all images in folder
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        all_kp3d = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                if args.skeleton:
                    # Project 3D keypoints using crop camera (same approach as _all view)
                    kp3d = out['pred_keypoints_3d'][n].detach().cpu().numpy()  # (21, 3)
                    cam_t_crop = out['pred_cam_t'][n].detach().cpu().numpy()   # (3,)
                    H_crop, W_crop = input_patch.shape[:2]  # 256, 192
                    f_crop = float(model_cfg.EXTRA.FOCAL_LENGTH)  # 5000
                    kp3d_cam = kp3d + cam_t_crop
                    x_pixel = f_crop * kp3d_cam[:, 0] / kp3d_cam[:, 2] + W_crop / 2
                    y_pixel = f_crop * kp3d_cam[:, 1] / kp3d_cam[:, 2] + H_crop / 2
                    kp_with_conf = np.concatenate([
                        np.stack([x_pixel, y_pixel], axis=1),
                        np.ones((kp3d.shape[0], 1))
                    ], axis=1)
                    skel_img = render_openpose((input_patch * 255).astype(np.uint8), kp_with_conf) / 255.
                    final_img = np.concatenate([input_patch, skel_img], axis=1)
                else:
                    regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            batch['img'][n],
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            )
                    if args.side_view:
                        side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                                out['pred_cam_t'][n].detach().cpu().numpy(),
                                                white_img,
                                                mesh_base_color=LIGHT_BLUE,
                                                scene_bg_color=(1, 1, 1),
                                                side_view=True)
                        final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                    else:
                        final_img = np.concatenate([input_patch, regression_img], axis=1)

                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

                # 3D 피처 저장 (76-dim: vector 60 + angle 15 + label 1)
                kp3d = out['pred_keypoints_3d'][n].detach().cpu().numpy()  # (21, 3)
                vector, angle = _vector_normalization_3d(kp3d)
                feature = np.concatenate([
                    vector.flatten(),                           # 60
                    angle.flatten(),                            # 15
                    np.array([-1], dtype=np.float32),          # label (미지정)
                ])                                              # → 76-dim
                hand_side = 'right' if is_right else 'left'
                np.savez(
                    os.path.join(args.out_folder, f'{img_fn}_{person_id}_{hand_side}_3d.npz'),
                    keypoints_3d=kp3d,              # (21, 3) 원본 3D 키포인트
                    vectors=vector,                 # (20, 3) 정규화 방향 벡터
                    angles=angle,                   # (15,)  인접 뼈대 각도 (degrees)
                    features=feature,               # (76,)  통합 피처
                    is_right=np.array([is_right]),
                )

                if args.skeleton:
                    kp3d = out['pred_keypoints_3d'][n].detach().cpu().numpy()  # (21, 3)
                    kp3d[:, 0] = (2 * is_right - 1) * kp3d[:, 0]
                    all_kp3d.append(kp3d)

                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

        # Render front view
        if args.full_frame and len(all_verts) > 0:
            if args.skeleton:
                img_h_full, img_w_full = img_cv2.shape[:2]
                f = float(scaled_focal_length)
                full_skel_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
                for kp3d, cam_t in zip(all_kp3d, all_cam_t):
                    kp3d_cam = kp3d + cam_t
                    kp2d_full = np.stack([
                        f * kp3d_cam[:, 0] / kp3d_cam[:, 2] + img_w_full / 2,
                        f * kp3d_cam[:, 1] / kp3d_cam[:, 2] + img_h_full / 2,
                    ], axis=1)
                    kp_with_conf = np.concatenate([kp2d_full, np.ones((kp2d_full.shape[0], 1))], axis=1)
                    full_skel_img = render_openpose((full_skel_img * 255).astype(np.uint8), kp_with_conf) / 255.
                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), 255 * full_skel_img[:, :, ::-1])
            else:
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

                # Overlay image
                input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
                input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])

if __name__ == '__main__':
    main()
