"""
IMPROVED Object Detection and Segmentation Comparison Script
Optimized for challenging images (cartoons, low contrast, etc.)
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import (
    DetrImageProcessor, DetrForObjectDetection,
    MaskFormerImageProcessor, MaskFormerForInstanceSegmentation,
    Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation,
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


class ImprovedDetectionSegmentation:
    def __init__(
        self,
        image_path,
        confidence_threshold=0.15,  # Lower for cartoon images
        iou_threshold=0.45,  # NMS threshold
        sam_checkpoint="models/sam_vit_h_4b8939.pth",  # Using ViT-H (Huge)
        sam2_checkpoint="models/sam2.1_hiera_large.pt",  # Using Hiera-Large
        use_larger_models=True,  # Use larger, more accurate models
        enhance_image=True,  # Apply preprocessing
    ):
        self.image_path = image_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.use_larger_models = use_larger_models
        self.enhance_image = enhance_image

        # Load original image
        self.original_image = cv2.imread(image_path)
        self.original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.original_image.shape[:2]

        print(f"Original image size: {self.width}x{self.height}")

        # Apply preprocessing if enabled
        if self.enhance_image:
            self.preprocessed_image = self.preprocess_image(self.original_image_rgb)
        else:
            self.preprocessed_image = self.original_image_rgb

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.sam_checkpoint = Path(sam_checkpoint)
        self.sam2_checkpoint = Path(sam2_checkpoint)

    def preprocess_image(self, image):
        """Apply image preprocessing to enhance features"""
        print("Applying image preprocessing...")

        # Convert to PIL for enhancement
        pil_image = Image.fromarray(image)

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.2)

        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.3)

        # Enhance color saturation (helps with cartoon detection)
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(1.1)

        # Convert back to numpy
        enhanced = np.array(pil_image)

        # Apply slight bilateral filtering to reduce noise while preserving edges
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)

        return enhanced

    def apply_tta(self, model, image_path, is_yolo=True):
        """Apply Test-Time Augmentation for better detection"""
        print("Applying Test-Time Augmentation (TTA)...")

        if not is_yolo:
            return None  # TTA mainly beneficial for YOLO

        # YOLO has built-in augmentation parameter
        return model(image_path, augment=True, verbose=False)[0]

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

        ax.set_title(f"{title}\nDetections: {len(boxes)}", fontsize=14, fontweight='bold')
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
        ax.set_title(f"{title}\nSegments: {len(masks)}", fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        return fig

    def run_detr(self):
        """1. IMPROVED DETR Detection"""
        print("\n" + "="*50)
        print("Running IMPROVED DETR Object Detection...")
        print("="*50)

        # Use larger DETR model for better accuracy
        model_name = "facebook/detr-resnet-101" if self.use_larger_models else "facebook/detr-resnet-50"
        print(f"Using model: {model_name}")

        processor = DetrImageProcessor.from_pretrained(model_name)
        model = DetrForObjectDetection.from_pretrained(model_name).to(self.device)

        inputs = processor(images=self.preprocessed_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[self.height, self.width]]).to(self.device)
        results = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.confidence_threshold  # Lower threshold
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        print(f"✓ Detected {len(boxes)} objects")

        fig = self.draw_boxes(self.original_image_rgb, boxes, labels, scores,
                             "IMPROVED DETR Detection")
        fig.savefig('output/improved_1_detr_detection.jpg', dpi=150, bbox_inches='tight')
        plt.close()

        return boxes, labels, scores

    def run_yolo11_detection(self):
        """2. IMPROVED YOLO11 Detection with TTA"""
        print("\n" + "="*50)
        print("Running IMPROVED YOLO11 Object Detection...")
        print("="*50)

        # Use larger YOLO model for better accuracy
        if self.use_larger_models:
            model_path = 'yolo11x.pt'  # Extra large model (best accuracy)
            print(f"Using model: YOLO11x (downloading if needed...)")
        else:
            model_path = 'yolo11l.pt'  # Large model (good balance)
            print(f"Using model: YOLO11l")

        model = YOLO(model_path)

        # Run with augmentation and optimized parameters
        results = model(
            self.image_path,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            augment=True,  # Test-Time Augmentation
            imgsz=1280,  # Larger input size for better detection
            verbose=False
        )[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy().astype(int)

        print(f"✓ Detected {len(boxes)} objects")

        fig = self.draw_boxes(self.original_image_rgb, boxes, labels, scores,
                             "IMPROVED YOLO11 Detection (with TTA)")
        fig.savefig('output/improved_2_yolo11_detection.jpg', dpi=150, bbox_inches='tight')
        plt.close()

        return boxes, labels, scores

    def run_yolo11_segmentation(self):
        """3. IMPROVED YOLO11-seg Segmentation"""
        print("\n" + "="*50)
        print("Running IMPROVED YOLO11-seg Segmentation...")
        print("="*50)

        if self.use_larger_models:
            model_path = 'yolo11x-seg.pt'
            print(f"Using model: YOLO11x-seg (downloading if needed...)")
        else:
            model_path = 'yolo11l-seg.pt'
            print(f"Using model: YOLO11l-seg")

        model = YOLO(model_path)

        results = model(
            self.image_path,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            augment=True,
            imgsz=1280,
            verbose=False
        )[0]

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

        print(f"✓ Segmented {len(masks_resized)} objects")

        fig = self.draw_masks(self.original_image_rgb, masks_resized, labels, scores,
                             "IMPROVED YOLO11-seg Segmentation")
        fig.savefig('output/improved_3_yolo11_segmentation.jpg', dpi=150, bbox_inches='tight')
        plt.close()

        return boxes, labels, scores

    def run_maskformer(self):
        """4. IMPROVED MaskFormer"""
        print("\n" + "="*50)
        print("Running IMPROVED MaskFormer Segmentation...")
        print("="*50)

        # Use larger model
        model_name = "facebook/maskformer-swin-large-coco" if self.use_larger_models else "facebook/maskformer-swin-base-coco"
        print(f"Using model: {model_name}")

        processor = MaskFormerImageProcessor.from_pretrained(model_name)
        model = MaskFormerForInstanceSegmentation.from_pretrained(model_name).to(self.device)

        inputs = processor(images=self.preprocessed_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs, target_sizes=[(self.height, self.width)])[0]

        masks_list = []
        labels_list = []
        scores_list = []

        for segment_info in results.get("segments_info", []):
            if segment_info.get("score", 0) > self.confidence_threshold:
                mask = (results["segmentation"] == segment_info["id"]).cpu().numpy()
                masks_list.append(mask)
                labels_list.append(segment_info["label_id"])
                scores_list.append(segment_info["score"])

        print(f"✓ Segmented {len(masks_list)} objects")

        if len(masks_list) > 0:
            fig = self.draw_masks(self.original_image_rgb, masks_list, labels_list,
                                 scores_list, "IMPROVED MaskFormer Segmentation")
            fig.savefig('output/improved_4_maskformer_segmentation.jpg', dpi=150, bbox_inches='tight')
            plt.close()

    def run_mask2former(self):
        """5. IMPROVED Mask2Former"""
        print("\n" + "="*50)
        print("Running IMPROVED Mask2Former Segmentation...")
        print("="*50)

        # Use larger model
        model_name = "facebook/mask2former-swin-large-coco-instance" if self.use_larger_models else "facebook/mask2former-swin-base-coco-instance"
        print(f"Using model: {model_name}")

        processor = Mask2FormerImageProcessor.from_pretrained(model_name)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(self.device)

        inputs = processor(images=self.preprocessed_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs, target_sizes=[(self.height, self.width)])[0]

        masks_list = []
        labels_list = []
        scores_list = []

        for segment_info in results["segments_info"]:
            if segment_info["score"] > self.confidence_threshold:
                mask = results["segmentation"] == segment_info["id"]
                masks_list.append(mask.cpu().numpy())
                labels_list.append(segment_info["label_id"])
                scores_list.append(segment_info["score"])

        print(f"✓ Segmented {len(masks_list)} objects")

        if len(masks_list) > 0:
            fig = self.draw_masks(self.original_image_rgb, masks_list, labels_list,
                                 scores_list, "IMPROVED Mask2Former Segmentation")
            fig.savefig('output/improved_5_mask2former_segmentation.jpg', dpi=150, bbox_inches='tight')
            plt.close()

    def run_sam(self, boxes):
        """6. IMPROVED SAM with better prompts"""
        print("\n" + "="*50)
        print("Running IMPROVED SAM Segmentation...")
        print("="*50)

        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            print("SAM not installed. Skipping...")
            return

        if not self.sam_checkpoint.exists():
            print(f"SAM checkpoint missing at {self.sam_checkpoint}. Skipping...")
            return

        # Auto-detect model type from filename
        if "vit_h" in self.sam_checkpoint.name or "4b8939" in self.sam_checkpoint.name:
            model_type = "vit_h"
            print("Using SAM ViT-H (Huge) model - 2.4GB")
        elif "vit_l" in self.sam_checkpoint.name:
            model_type = "vit_l"
            print("Using SAM ViT-L (Large) model")
        else:
            model_type = "vit_b"
            print("Using SAM ViT-B (Base) model")

        try:
            sam = sam_model_registry[model_type](checkpoint=str(self.sam_checkpoint))
        except Exception as exc:
            print(f"Failed to load SAM: {exc}")
            return

        sam.to(device=self.device)
        predictor = SamPredictor(sam)
        predictor.set_image(self.original_image_rgb)

        masks_list = []
        labels_list = []
        scores_list = []

        # Process all boxes (not just 20)
        for i, box in enumerate(boxes):
            input_box = np.array(box)
            masks, iou_scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            masks_list.append(masks[0])
            labels_list.append(0)
            scores_list.append(float(iou_scores[0]))

        print(f"✓ Segmented {len(masks_list)} objects")

        if len(masks_list) > 0:
            fig = self.draw_masks(self.original_image_rgb, masks_list, labels_list,
                                 scores_list, "IMPROVED SAM Segmentation")
            fig.savefig('output/improved_6_sam_segmentation.jpg', dpi=150, bbox_inches='tight')
            plt.close()

    def run_sam2(self, boxes):
        """7. IMPROVED SAM2"""
        print("\n" + "="*50)
        print("Running IMPROVED SAM2 Segmentation...")
        print("="*50)

        try:
            import sam2
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            print("SAM2 not installed. Skipping...")
            return

        if not self.sam2_checkpoint.exists():
            print(f"SAM2 checkpoint missing at {self.sam2_checkpoint}. Skipping...")
            return

        sam2_pkg_path = Path(sam2.__file__).resolve().parent

        # Determine config based on checkpoint filename
        if "large" in self.sam2_checkpoint.name.lower() or "hiera_l" in self.sam2_checkpoint.name.lower():
            config_name_candidates = [
                "configs/sam2.1/sam2.1_hiera_l",
                "configs/sam2.1/sam2.1_hiera_large",
            ]
            print("Using SAM2.1 Hiera-Large model - 856MB")
        else:
            config_name_candidates = [
                "configs/sam2.1/sam2.1_hiera_b+",
                "configs/sam2.1/sam2.1_hiera_base+",
            ]
            print("Using SAM2.1 Hiera-Base+ model")

        model_cfg_name = None
        for cfg_name in config_name_candidates:
            cfg_file = sam2_pkg_path / f"{cfg_name}.yaml"
            if cfg_file.exists():
                model_cfg_name = cfg_name
                break

        # Try to copy config from local configs/ directory if not in package
        if model_cfg_name is None:
            local_config_candidates = [
                Path("configs/sam2.1_hiera_l.yaml"),
                Path("configs/sam2.1_hiera_b+.yaml"),
            ]
            for local_config in local_config_candidates:
                if local_config.exists():
                    target_config_dir = sam2_pkg_path / "configs" / "sam2.1"
                    target_config_dir.mkdir(parents=True, exist_ok=True)

                    import shutil
                    target_file = target_config_dir / local_config.name
                    shutil.copy(str(local_config), str(target_file))
                    print(f"Copied config from {local_config} to {target_file}")

                    model_cfg_name = f"configs/sam2.1/{local_config.stem}"
                    break

        if model_cfg_name is None:
            print("SAM2 config YAML not found.")
            print("Place the YAML config in configs/ directory or in the sam2 package.")
            return

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

        # Process all boxes
        for i, box in enumerate(boxes):
            input_box = np.array(box)
            masks, iou_scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            masks_list.append(masks[0])
            labels_list.append(0)
            scores_list.append(float(iou_scores[0]))

        print(f"✓ Segmented {len(masks_list)} objects")

        if len(masks_list) > 0:
            fig = self.draw_masks(self.original_image_rgb, masks_list, labels_list,
                                 scores_list, "IMPROVED SAM2 Segmentation")
            fig.savefig('output/improved_7_sam2_segmentation.jpg', dpi=150, bbox_inches='tight')
            plt.close()

    def run_all(self):
        """Run all improved detection and segmentation methods"""
        print("\n" + "="*70)
        print("STARTING IMPROVED DETECTION & SEGMENTATION COMPARISON")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  - Confidence threshold: {self.confidence_threshold}")
        print(f"  - IOU threshold: {self.iou_threshold}")
        print(f"  - Using larger models: {self.use_larger_models}")
        print(f"  - Image enhancement: {self.enhance_image}")
        print(f"  - Test-Time Augmentation: Enabled for YOLO")
        print("="*70)

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
        if len(yolo_boxes) > 0:
            self.run_sam(yolo_boxes)

        # 7. SAM2 with YOLO11 boxes
        if len(yolo_boxes) > 0:
            self.run_sam2(yolo_boxes)

        print("\n" + "="*70)
        print("IMPROVED COMPARISON COMPLETE!")
        print("="*70)
        print("\nKey Improvements:")
        print("  ✓ Larger, more accurate models (YOLO11x, DETR-ResNet-101, etc.)")
        print("  ✓ Lower confidence threshold (0.15) for cartoon images")
        print("  ✓ Image preprocessing (contrast, sharpness, saturation enhancement)")
        print("  ✓ Test-Time Augmentation for YOLO models")
        print("  ✓ Larger input resolution (1280px)")
        print("  ✓ Optimized NMS threshold")
        print("\nGenerated improved output images in output/ directory")
        print("="*70)


if __name__ == "__main__":
    # Run IMPROVED comparison
    print("="*70)
    print("IMPROVED DETECTION & SEGMENTATION SCRIPT")
    print("Optimized for challenging images like cartoons and low-contrast photos")
    print("="*70)

    comparison = ImprovedDetectionSegmentation(
        image_path="input/image_to_detect_and_segment.jpg",
        confidence_threshold=0.15,  # Very low for cartoon images
        iou_threshold=0.45,  # Standard NMS threshold
        use_larger_models=True,  # Use best models (slower but more accurate)
        enhance_image=True,  # Apply preprocessing
    )
    comparison.run_all()