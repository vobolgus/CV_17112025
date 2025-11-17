"""
Detection & Segmentation with LARGE SAM Models
Using: sam_vit_h_4b8939.pth (SAM ViT-H) and sam2.1_hiera_large.pt (SAM2.1 Large)
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class LargeSAMDetection:
    def __init__(
        self,
        image_path,
        confidence_threshold=0.15,
        sam_checkpoint="models/sam_vit_h_4b8939.pth",
        sam2_checkpoint="models/sam2.1_hiera_large.pt",
        sam2_config="configs/sam2.1_hiera_l.yaml",
    ):
        self.image_path = image_path
        self.confidence_threshold = confidence_threshold

        # Load image
        self.original_image = cv2.imread(image_path)
        self.original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.original_image.shape[:2]

        print(f"Original image size: {self.width}x{self.height}")

        # Preprocess image
        self.preprocessed_image = self.preprocess_image(self.original_image_rgb)

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Model paths
        self.sam_checkpoint = Path(sam_checkpoint)
        self.sam2_checkpoint = Path(sam2_checkpoint)
        self.sam2_config = Path(sam2_config)

    def preprocess_image(self, image):
        """Apply image preprocessing to enhance features"""
        print("Applying image preprocessing...")
        pil_image = Image.fromarray(image)

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.2)

        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.3)

        # Enhance color
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(1.1)

        enhanced = np.array(pil_image)
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)

        return enhanced

    def draw_boxes(self, image, boxes, labels, scores, title):
        """Draw bounding boxes on image"""
        fig, ax = plt.subplots(1, figsize=(14, 10))
        ax.imshow(image)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(COCO_CLASSES)))

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            color = colors[label % len(colors)]
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2,
                                     edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
            ax.text(x1, y1-5, f'{class_name}: {score:.2f}',
                   bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')

        ax.set_title(f"{title}\nDetections: {len(boxes)}", fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        return fig

    def draw_masks(self, image, masks, labels, scores, title):
        """Draw segmentation masks on image"""
        fig, ax = plt.subplots(1, figsize=(14, 10))
        ax.imshow(image)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(COCO_CLASSES)))
        overlay = image.copy()

        for i, (mask, label, score) in enumerate(zip(masks, labels, scores)):
            color = colors[label % len(colors)]
            color_mask = (np.array(color[:3]) * 255).astype(np.uint8)

            if len(mask.shape) == 2:
                mask_binary = mask > 0.5
            else:
                mask_binary = mask

            overlay[mask_binary] = overlay[mask_binary] * 0.5 + color_mask * 0.5

            # Draw contours
            mask_uint8 = (mask_binary * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                contour = contour.squeeze()
                if len(contour.shape) == 2 and contour.shape[0] > 2:
                    ax.plot(contour[:, 0], contour[:, 1], color=color, linewidth=2)

            # Add label
            y_indices, x_indices = np.where(mask_binary)
            if len(y_indices) > 0:
                centroid_x = int(np.mean(x_indices))
                centroid_y = int(np.mean(y_indices))
                class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
                ax.text(centroid_x, centroid_y, f'{class_name}\n{score:.2f}',
                       bbox=dict(facecolor=color, alpha=0.7), fontsize=8,
                       color='white', ha='center')

        ax.imshow(overlay.astype(np.uint8), alpha=0.7)
        ax.set_title(f"{title}\nSegments: {len(masks)}", fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        return fig

    def run_yolo11_detection(self):
        """Run YOLO11 for bounding boxes"""
        print("\n" + "="*70)
        print("Running YOLO11x Detection for bounding boxes...")
        print("="*70)

        model = YOLO('yolo11x.pt')

        results = model(
            self.image_path,
            conf=self.confidence_threshold,
            iou=0.45,
            augment=True,
            imgsz=1280,
            verbose=False
        )[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy().astype(int)

        print(f"✓ Detected {len(boxes)} objects")

        fig = self.draw_boxes(self.original_image_rgb, boxes, labels, scores,
                             "YOLO11x Object Detection")
        fig.savefig('output/large_sam_1_yolo11_detection.jpg', dpi=150, bbox_inches='tight')
        plt.close()

        return boxes, labels, scores

    def run_sam_vit_h(self, boxes, labels):
        """Run SAM with ViT-H (huge) model"""
        print("\n" + "="*70)
        print("Running SAM with ViT-H (Huge) Model...")
        print("="*70)

        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            print("SAM not installed. Install with: pip install segment-anything")
            return

        if not self.sam_checkpoint.exists():
            print(f"SAM checkpoint missing at {self.sam_checkpoint}")
            return

        print(f"Loading SAM ViT-H from {self.sam_checkpoint}")
        print("Model size: ~2.4GB - This is the LARGEST SAM model!")

        try:
            sam = sam_model_registry["vit_h"](checkpoint=str(self.sam_checkpoint))
            sam.to(device=self.device)
            predictor = SamPredictor(sam)
        except Exception as exc:
            print(f"Failed to load SAM: {exc}")
            return

        predictor.set_image(self.original_image_rgb)

        masks_list = []
        labels_list = []
        scores_list = []

        print(f"Processing {len(boxes)} bounding boxes...")
        for i, (box, label) in enumerate(zip(boxes, labels)):
            if (i + 1) % 5 == 0:
                print(f"  Processing box {i+1}/{len(boxes)}...")

            input_box = np.array(box)
            masks, iou_scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            masks_list.append(masks[0])
            labels_list.append(label)
            scores_list.append(float(iou_scores[0]))

        print(f"✓ Segmented {len(masks_list)} objects with SAM ViT-H")

        if len(masks_list) > 0:
            fig = self.draw_masks(self.original_image_rgb, masks_list, labels_list,
                                 scores_list, "SAM ViT-H (Huge) Segmentation")
            fig.savefig('output/large_sam_2_sam_vit_h_segmentation.jpg', dpi=150, bbox_inches='tight')
            plt.close()

        return masks_list, labels_list, scores_list

    def run_sam2_large(self, boxes, labels):
        """Run SAM2 with Hiera-L (large) model"""
        print("\n" + "="*70)
        print("Running SAM2.1 with Hiera-Large Model...")
        print("="*70)

        try:
            import sam2
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            print("SAM2 not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
            return

        if not self.sam2_checkpoint.exists():
            print(f"SAM2 checkpoint missing at {self.sam2_checkpoint}")
            return

        if not self.sam2_config.exists():
            print(f"SAM2 config missing at {self.sam2_config}")
            return

        print(f"Loading SAM2.1 Hiera-Large from {self.sam2_checkpoint}")
        print("Model size: ~856MB - This is a LARGE SAM2 model!")

        # Find config relative to sam2 package
        sam2_pkg_path = Path(sam2.__file__).resolve().parent

        # Try to use the config from our configs directory
        # We need to copy it to the sam2 package location
        target_config_dir = sam2_pkg_path / "configs" / "sam2.1"
        target_config_dir.mkdir(parents=True, exist_ok=True)

        target_config_file = target_config_dir / "sam2.1_hiera_l.yaml"
        if not target_config_file.exists() and self.sam2_config.exists():
            import shutil
            shutil.copy(str(self.sam2_config), str(target_config_file))
            print(f"Copied config to {target_config_file}")

        model_cfg_name = "configs/sam2.1/sam2.1_hiera_l"

        try:
            sam2_model = build_sam2(model_cfg_name, str(self.sam2_checkpoint), device=self.device)
            predictor = SAM2ImagePredictor(sam2_model)
        except Exception as exc:
            print(f"Failed to load SAM2: {exc}")
            return

        predictor.set_image(self.original_image_rgb)

        masks_list = []
        labels_list = []
        scores_list = []

        print(f"Processing {len(boxes)} bounding boxes...")
        for i, (box, label) in enumerate(zip(boxes, labels)):
            if (i + 1) % 5 == 0:
                print(f"  Processing box {i+1}/{len(boxes)}...")

            input_box = np.array(box)
            masks, iou_scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            masks_list.append(masks[0])
            labels_list.append(label)
            scores_list.append(float(iou_scores[0]))

        print(f"✓ Segmented {len(masks_list)} objects with SAM2.1 Hiera-Large")

        if len(masks_list) > 0:
            fig = self.draw_masks(self.original_image_rgb, masks_list, labels_list,
                                 scores_list, "SAM2.1 Hiera-Large Segmentation")
            fig.savefig('output/large_sam_3_sam2_hiera_large_segmentation.jpg', dpi=150, bbox_inches='tight')
            plt.close()

        return masks_list, labels_list, scores_list

    def run_all(self):
        """Run complete pipeline with large SAM models"""
        print("\n" + "="*70)
        print("DETECTION & SEGMENTATION WITH LARGE SAM MODELS")
        print("="*70)
        print("\nModels:")
        print(f"  - Detection: YOLO11x")
        print(f"  - SAM: ViT-H (Huge) - 2.4GB")
        print(f"  - SAM2: Hiera-Large - 856MB")
        print("="*70)

        # 1. Get bounding boxes from YOLO
        boxes, labels, scores = self.run_yolo11_detection()

        if len(boxes) == 0:
            print("\nNo objects detected by YOLO. Cannot run SAM models.")
            return

        # 2. SAM ViT-H segmentation
        sam_masks, sam_labels, sam_scores = self.run_sam_vit_h(boxes, labels)

        # 3. SAM2 Hiera-L segmentation
        sam2_masks, sam2_labels, sam2_scores = self.run_sam2_large(boxes, labels)

        print("\n" + "="*70)
        print("COMPLETE!")
        print("="*70)
        print("\nGenerated outputs:")
        print("  1. large_sam_1_yolo11_detection.jpg")
        print("  2. large_sam_2_sam_vit_h_segmentation.jpg")
        print("  3. large_sam_3_sam2_hiera_large_segmentation.jpg")
        print("\nModel Comparison:")
        print(f"  SAM ViT-H:       {len(sam_masks) if sam_masks else 0} segments")
        print(f"  SAM2 Hiera-L:    {len(sam2_masks) if sam2_masks else 0} segments")
        print("="*70)


if __name__ == "__main__":
    print("="*70)
    print("LARGE SAM MODELS TEST")
    print("="*70)

    detector = LargeSAMDetection(
        image_path="input/image_to_detect_and_segment.jpg",
        confidence_threshold=0.15,
        sam_checkpoint="models/sam_vit_h_4b8939.pth",
        sam2_checkpoint="models/sam2.1_hiera_large.pt",
        sam2_config="configs/sam2.1_hiera_l.yaml",
    )

    detector.run_all()