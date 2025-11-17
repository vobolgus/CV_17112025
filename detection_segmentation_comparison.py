"""
Comprehensive Object Detection and Segmentation Comparison Script
Using: DETR, YOLO11, YOLO11-seg, MaskFormer, Mask2Former, SAM, SAM2
All using COCO 80 classes
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import (
    DetrImageProcessor, DetrForObjectDetection,
    MaskFormerImageProcessor, MaskFormerForInstanceSegmentation,
    Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation,
    AutoImageProcessor, AutoModelForUniversalSegmentation
)
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# COCO class names
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


class DetectionSegmentationComparison:
    def __init__(
        self,
        image_path,
        confidence_threshold=0.5,
        sam_checkpoint="models/sam_vit_b_01ec64.pth",
        sam2_checkpoint="models/sam2.1_hiera_base_plus.pt",
    ):
        self.image_path = image_path
        self.confidence_threshold = confidence_threshold
        self.original_image = cv2.imread(image_path)
        self.original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.original_image.shape[:2]

        print(f"Original image size: {self.width}x{self.height}")

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Local checkpoints (adjustable so we can point to downloaded files)
        self.sam_checkpoint = Path(sam_checkpoint)
        self.sam2_checkpoint = Path(sam2_checkpoint)

    def draw_boxes(self, image, boxes, labels, scores, title):
        """Draw bounding boxes on image"""
        fig, ax = plt.subplots(1, figsize=(12, 8))
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
                   bbox=dict(facecolor=color, alpha=0.5), fontsize=8, color='white')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        return fig

    def draw_masks(self, image, masks, labels, scores, title):
        """Draw segmentation masks on image"""
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(COCO_CLASSES)))

        # Create overlay
        overlay = image.copy()

        for i, (mask, label, score) in enumerate(zip(masks, labels, scores)):
            color = colors[label % len(colors)]
            color_mask = (np.array(color[:3]) * 255).astype(np.uint8)

            # Apply mask
            if len(mask.shape) == 2:
                mask_binary = mask > 0.5
            else:
                mask_binary = mask

            overlay[mask_binary] = overlay[mask_binary] * 0.5 + color_mask * 0.5

            # Find contours for boundary
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
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        return fig

    def run_detr(self):
        """1. Object Detection with DETR"""
        print("\n" + "="*50)
        print("Running DETR Object Detection...")
        print("="*50)

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(self.device)

        # Note: DETR requires resizing - it doesn't natively handle arbitrary resolutions
        inputs = processor(images=self.original_image_rgb, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process
        target_sizes = torch.tensor([[self.height, self.width]]).to(self.device)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes,
                                                          threshold=self.confidence_threshold)[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        print(f"Detected {len(boxes)} objects")

        fig = self.draw_boxes(self.original_image_rgb, boxes, labels, scores,
                             "DETR Object Detection")
        fig.savefig('output/1_detr_detection.jpg', dpi=150, bbox_inches='tight')
        plt.close()

        return boxes, labels, scores

    def run_yolo11_detection(self):
        """2. Object Detection with YOLO11"""
        print("\n" + "="*50)
        print("Running YOLO11 Object Detection...")
        print("="*50)

        model = YOLO('models/yolo11n.pt')  # Using nano model for speed

        # YOLO11 can handle various input sizes but prefers multiples of 32
        # We can pass imgsz parameter or let it handle automatically
        results = model(self.image_path, conf=self.confidence_threshold, verbose=False)[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy().astype(int)

        print(f"Detected {len(boxes)} objects")

        fig = self.draw_boxes(self.original_image_rgb, boxes, labels, scores,
                             "YOLO11 Object Detection")
        fig.savefig('output/2_yolo11_detection.jpg', dpi=150, bbox_inches='tight')
        plt.close()

        return boxes, labels, scores

    def run_yolo11_segmentation(self):
        """3. Object Segmentation with YOLO11-seg"""
        print("\n" + "="*50)
        print("Running YOLO11-seg Segmentation...")
        print("="*50)

        model = YOLO('models/yolo11n-seg.pt')  # Segmentation model

        results = model(self.image_path, conf=self.confidence_threshold, verbose=False)[0]

        if results.masks is None:
            print("No masks detected")
            return

        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy().astype(int)

        # Resize masks to original image size
        masks_resized = []
        for mask in masks:
            mask_resized = cv2.resize(mask, (self.width, self.height),
                                     interpolation=cv2.INTER_LINEAR)
            masks_resized.append(mask_resized)

        print(f"Segmented {len(masks_resized)} objects")

        fig = self.draw_masks(self.original_image_rgb, masks_resized, labels, scores,
                             "YOLO11-seg Segmentation")
        fig.savefig('output/3_yolo11_segmentation.jpg', dpi=150, bbox_inches='tight')
        plt.close()

    def run_maskformer(self):
        """4. Object Segmentation with MaskFormer"""
        print("\n" + "="*50)
        print("Running MaskFormer Segmentation...")
        print("="*50)

        processor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
        model = MaskFormerForInstanceSegmentation.from_pretrained(
            "facebook/maskformer-swin-base-coco").to(self.device)

        # MaskFormer also requires resizing - doesn't natively handle arbitrary resolutions
        inputs = processor(images=self.original_image_rgb, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process
        results = processor.post_process_instance_segmentation(
            outputs, target_sizes=[(self.height, self.width)])[0]

        # Filter by confidence
        masks_list = []
        labels_list = []
        scores_list = []

        # MaskFormer returns segments_info with scores
        for segment_info in results.get("segments_info", []):
            if segment_info.get("score", 0) > self.confidence_threshold:
                # Get the mask for this segment
                mask = (results["segmentation"] == segment_info["id"]).cpu().numpy()
                masks_list.append(mask)
                labels_list.append(segment_info["label_id"])
                scores_list.append(segment_info["score"])

        print(f"Segmented {len(masks_list)} objects")

        if len(masks_list) > 0:
            fig = self.draw_masks(self.original_image_rgb, masks_list, labels_list,
                                 scores_list, "MaskFormer Segmentation")
            fig.savefig('output/4_maskformer_segmentation.jpg', dpi=150, bbox_inches='tight')
            plt.close()

    def run_mask2former(self):
        """5. Object Segmentation with Mask2Former"""
        print("\n" + "="*50)
        print("Running Mask2Former Segmentation...")
        print("="*50)

        processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-instance")
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-base-coco-instance").to(self.device)

        # Mask2Former also requires resizing - doesn't natively handle arbitrary resolutions
        inputs = processor(images=self.original_image_rgb, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process
        results = processor.post_process_instance_segmentation(
            outputs, target_sizes=[(self.height, self.width)])[0]

        # Filter by confidence
        masks_list = []
        labels_list = []
        scores_list = []

        for segment_info in results["segments_info"]:
            if segment_info["score"] > self.confidence_threshold:
                mask = results["segmentation"] == segment_info["id"]
                masks_list.append(mask.cpu().numpy())
                labels_list.append(segment_info["label_id"])
                scores_list.append(segment_info["score"])

        print(f"Segmented {len(masks_list)} objects")

        if len(masks_list) > 0:
            fig = self.draw_masks(self.original_image_rgb, masks_list, labels_list,
                                 scores_list, "Mask2Former Segmentation")
            fig.savefig('output/5_mask2former_segmentation.jpg', dpi=150, bbox_inches='tight')
            plt.close()

    def run_sam(self, boxes):
        """6. Object Segmentation with SAM using YOLO11 bboxes"""
        print("\n" + "="*50)
        print("Running SAM Segmentation with YOLO11 bboxes...")
        print("="*50)

        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            print("SAM not installed. Install with: pip install segment-anything")
            return

        # Use the downloaded checkpoint if available; default to vit-b since sam_vit_b_01ec64.pth is present
        if not self.sam_checkpoint.exists():
            print(f"SAM checkpoint missing at {self.sam_checkpoint}.")
            print("Place the downloaded .pth in this path or pass sam_checkpoint= to the class.")
            return

        model_type = "vit_b" if "vit_b" in self.sam_checkpoint.name else "vit_h"

        try:
            sam = sam_model_registry[model_type](checkpoint=str(self.sam_checkpoint))
        except Exception as exc:  # pragma: no cover - defensive guard
            print(f"Failed to load SAM checkpoint {self.sam_checkpoint}: {exc}")
            return

        sam.to(device=self.device)
        predictor = SamPredictor(sam)

        # SAM can handle arbitrary input sizes natively
        predictor.set_image(self.original_image_rgb)

        masks_list = []
        labels_list = []
        scores_list = []

        for i, box in enumerate(boxes[:20]):  # Limit to first 20 for speed
            input_box = np.array(box)
            masks, iou_scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            masks_list.append(masks[0])
            labels_list.append(0)  # Generic label for SAM
            scores_list.append(float(iou_scores[0]))

        print(f"Segmented {len(masks_list)} objects")

        if len(masks_list) > 0:
            fig = self.draw_masks(self.original_image_rgb, masks_list, labels_list,
                                 scores_list, "SAM Segmentation (with YOLO11 boxes)")
            fig.savefig('output/6_sam_segmentation.jpg', dpi=150, bbox_inches='tight')
            plt.close()

    def run_sam2(self, boxes):
        """7. Object Segmentation with SAM2 using YOLO11 bboxes"""
        print("\n" + "="*50)
        print("Running SAM2 Segmentation with YOLO11 bboxes...")
        print("="*50)

        try:
            import sam2
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            print("SAM2 not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
            return

        if not self.sam2_checkpoint.exists():
            print(f"SAM2 checkpoint missing at {self.sam2_checkpoint}.")
            print("Place the downloaded .pt in this path or pass sam2_checkpoint= to the class.")
            return

        # Hydra needs the config name relative to the sam2 package search path (no extension).
        sam2_pkg_path = Path(sam2.__file__).resolve().parent
        config_name_candidates = [
            "configs/sam2.1/sam2.1_hiera_b+",
            "configs/sam2.1/sam2.1_hiera_base+",
        ]
        model_cfg_name = None
        for cfg_name in config_name_candidates:
            cfg_file = sam2_pkg_path / f"{cfg_name}.yaml"
            if cfg_file.exists():
                model_cfg_name = cfg_name
                break

        if model_cfg_name is None:
            print("SAM2 config YAML not found. Expected configs/sam2.1/sam2.1_hiera_b+.yaml inside the sam2 package.")
            print("Download the YAML from the SAM2 repo and place it under site-packages/sam2/configs/sam2.1/.")
            return

        try:
            sam2_model = build_sam2(model_cfg_name, str(self.sam2_checkpoint), device=self.device)
            predictor = SAM2ImagePredictor(sam2_model)
        except Exception as exc:  # pragma: no cover - defensive guard
            print(f"Failed to load SAM2: {exc}")
            return

        # SAM2 also handles arbitrary input sizes natively
        predictor.set_image(self.original_image_rgb)

        masks_list = []
        labels_list = []
        scores_list = []

        for i, box in enumerate(boxes[:20]):  # Limit to first 20 for speed
            input_box = np.array(box)
            masks, iou_scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            masks_list.append(masks[0])
            labels_list.append(0)  # Generic label for SAM2
            scores_list.append(float(iou_scores[0]))

        print(f"Segmented {len(masks_list)} objects")

        if len(masks_list) > 0:
            fig = self.draw_masks(self.original_image_rgb, masks_list, labels_list,
                                 scores_list, "SAM2 Segmentation (with YOLO11 boxes)")
            fig.savefig('output/7_sam2_segmentation.jpg', dpi=150, bbox_inches='tight')
            plt.close()

    def run_all(self):
        """Run all detection and segmentation methods"""
        print("\n" + "="*60)
        print("STARTING COMPREHENSIVE DETECTION & SEGMENTATION COMPARISON")
        print("="*60)

        # 1. DETR Detection
        detr_boxes, detr_labels, detr_scores = self.run_detr()

        # 2. YOLO11 Detection
        yolo_boxes, yolo_labels, yolo_scores = self.run_yolo11_detection()

        # 3. YOLO11 Segmentation
        self.run_yolo11_segmentation()

        # 4. MaskFormer
        self.run_maskformer()

        # 5. Mask2Former
        self.run_mask2former()

        # 6. SAM with YOLO11 boxes
        self.run_sam(yolo_boxes)

        # 7. SAM2 with YOLO11 boxes
        self.run_sam2(yolo_boxes)

        print("\n" + "="*60)
        print("COMPARISON COMPLETE!")
        print("="*60)
        print("\nGenerated output images:")
        print("  1. 1_detr_detection.jpg")
        print("  2. 2_yolo11_detection.jpg")
        print("  3. 3_yolo11_segmentation.jpg")
        print("  4. 4_maskformer_segmentation.jpg")
        print("  5. 5_mask2former_segmentation.jpg")
        print("  6. 6_sam_segmentation.jpg (if SAM is installed)")
        print("  7. 7_sam2_segmentation.jpg (if SAM2 is installed)")

        print("\n" + "="*60)
        print("MODEL RESOLUTION HANDLING CAPABILITIES:")
        print("="*60)
        print("✗ DETR: Requires resizing (typically to 800x1333)")
        print("✓ YOLO11: Flexible, prefers multiples of 32, can handle various sizes")
        print("✓ YOLO11-seg: Same as YOLO11, flexible with sizes")
        print("✗ MaskFormer: Requires resizing to specific input dimensions")
        print("✗ Mask2Former: Requires resizing to specific input dimensions")
        print("✓ SAM: Natively handles arbitrary resolutions!")
        print("✓ SAM2: Natively handles arbitrary resolutions!")
        print("="*60)


if __name__ == "__main__":
    # Run comparison
    # Using lower threshold for better detection on cartoon images
    comparison = DetectionSegmentationComparison(
        image_path="input/image_to_detect_and_segment.jpg",
        confidence_threshold=0.3
    )
    comparison.run_all()
