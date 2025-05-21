import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from scipy.spatial import distance

sys.path.append("/home/nt646jh/directory/folder/Sam_LoRA")
from segment_anything import SamAutomaticMaskGenerator, SamPredictor

from src.segment_anything import build_sam_vit_b
from src.lora import LoRA_sam

def filter_sinter_stones(masks, area_range, min_aspect_ratio, min_iou_with_largest):
    filtered_masks = []
    if not masks:
        return filtered_masks
    
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    for i, mask in enumerate(sorted_masks):
        area = mask['area']
        bbox = mask['bbox']
        width = bbox[2]
        height = bbox[3]
        aspect_ratio = min(width / height, height / width)
        
        is_redundant = False
        for larger_mask in sorted_masks[:i]:
            larger_seg = larger_mask['segmentation']
            current_seg = mask['segmentation']
            intersection = np.logical_and(larger_seg, current_seg).sum()
            union = np.logical_or(larger_seg, current_seg).sum()
            iou = intersection / union if union > 0 else 0
            if iou > min_iou_with_largest:
                is_redundant = True
                break
        
        if (area_range[0] <= area <= area_range[1] and 
            aspect_ratio >= min_aspect_ratio and 
            not is_redundant):
            filtered_masks.append(mask)
    
    return filtered_masks

def merge_overlapping_masks(masks, iou_threshold):
    if not masks:
        return masks
    
    merged_masks = []
    remaining_masks = masks.copy()
    
    while remaining_masks:
        current_mask = remaining_masks.pop(0)
        current_seg = current_mask['segmentation']
        current_bbox = current_mask['bbox']
        
        overlapping = []
        for i, other_mask in enumerate(remaining_masks):
            other_seg = other_mask['segmentation']
            intersection = np.logical_and(current_seg, other_seg).sum()
            union = np.logical_or(current_seg, other_seg).sum()
            iou = intersection / union if union > 0 else 0
            
            if iou > iou_threshold:
                overlapping.append(i)
        
        for idx in sorted(overlapping, reverse=True):
            other_mask = remaining_masks.pop(idx)
            current_seg = np.logical_or(current_seg, other_mask['segmentation'])
            other_bbox = other_mask['bbox']
            current_bbox = [
                min(current_bbox[0], other_bbox[0]),
                min(current_bbox[1], other_bbox[1]),
                max(current_bbox[0] + current_bbox[2], other_bbox[0] + other_bbox[2]) - min(current_bbox[0], other_bbox[0]),
                max(current_bbox[1] + current_bbox[3], other_bbox[1] + other_bbox[3]) - min(current_bbox[1], other_bbox[1])
            ]
        
        current_mask['segmentation'] = current_seg
        current_mask['bbox'] = current_bbox
        merged_masks.append(current_mask)
    
    return merged_masks

def exclude_text_regions(masks, image_shape, text_regions):
    height, width = image_shape[:2]
    filtered_masks = []
    
    for mask in masks:
        m = mask['segmentation']
        mask_height, mask_width = m.shape
        overlaps_text = False
        
        for x, y, w, h in text_regions:
            mask_x, mask_y = int(x * mask_width / width), int(y * mask_height / height)
            mask_w, mask_h = int(w * mask_width / width), int(h * mask_height / height)
            
            if np.any(m[max(0, mask_y):min(mask_height, mask_y + mask_h),
                        max(0, mask_x):min(mask_width, mask_x + mask_w)]):
                overlaps_text = True
                break
        
        if not overlaps_text:
            filtered_masks.append(mask)
    
    return filtered_masks

def color_segmentation(masks, base_image):
    segmented_image = base_image.copy()
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]]
    
    for i, mask in enumerate(masks):
        m = mask['segmentation']
        m_resized = cv2.resize(m.astype(np.uint8), (segmented_image.shape[1], segmented_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        color = colors[i % len(colors)]
        color_mask = np.zeros_like(segmented_image, dtype=np.float32)
        
        for c in range(3):
            color_mask[:, :, c] = m_resized * color[c]
        color_mask = np.uint8(color_mask * 0.3 * 255)
        segmented_image = cv2.addWeighted(segmented_image, 1, color_mask, 0.7, 0)
    
    return segmented_image



def get_rotated_min_bounding_rect(mask):
    binary_mask = mask['segmentation'].astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    min_rect = cv2.minAreaRect(largest_contour)
    width, height = min_rect[1]
    
    if width < height:
        width, height = height, width
    
    perimeter = 2 * width + 2 * height
    
    circumference_diameter = perimeter / np.pi
    
    box_points = cv2.boxPoints(min_rect)
    box_points = box_points.astype(np.int32)
    

    moments = cv2.moments(largest_contour)
    if moments['m00'] == 0: 
        return None
    

    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']
    

    mu20 = moments['mu20'] / moments['m00']  
    mu02 = moments['mu02'] / moments['m00']  
    mu11 = moments['mu11'] / moments['m00']  
    
    
    a = mu20 + mu02
    b = np.sqrt(4 * mu11**2 + (mu20 - mu02)**2)
    lambda1 = (a + b) / 2  
    lambda2 = (a - b) / 2 
    
    major_axis = 4 * np.sqrt(lambda1)
    minor_axis = 4 * np.sqrt(lambda2)
    
    if mu11 == 0 and mu20 == mu02:
        orientation = 0
    else:
        orientation = 0.5 * np.arctan2(2 * mu11, mu20 - mu02) * 180 / np.pi

    area = np.pi * (major_axis / 2) * (minor_axis / 2)
    
    return area, perimeter, circumference_diameter, major_axis, minor_axis, orientation, box_points

def draw_bounding_boxes(image, box_points_list, color=(0, 255, 0), thickness=1):
    result_image = image.copy()
    for box_points in box_points_list:
        cv2.polylines(result_image, [box_points], isClosed=True, color=color, thickness=thickness)
    return result_image

def measure_box_dimensions(masks):
    print("\nBounding Box Dimensions (width, height) in pixels:")
    for i, mask in enumerate(masks, 1):
        bbox = mask['bbox']
        width = bbox[2]  
        height = bbox[3]  

 
image = cv2.imread('/home/nt646jh/directory/folder/bc_nazarii_tymochko/img1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# sam_checkpoint = '/home/nt646jh/directory/folder/bc_nazarii_tymochko/SegmentAnything/sam_vit_h_4b8939.pth'
# model_type = "vit_h"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)

# sam_checkpoint = '/home/nt646jh/directory/folder/bc_nazarii_tymochko/Sam_LoRA/sam_vit_b_01ec64.pth'
sam_checkpoint = '/home/nt646jh/directory/folder/Sam_LoRA/sam_vit_b_01ec64.pth'
lora_weights = '/home/nt646jh/directory/folder/Sam_LoRA/lora_rank512.safetensors'  # ← заміни на твій файл
rank = 512  # ← той самий rank, що був під час тренування

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Побудова базової SAM-моделі
sam_base = build_sam_vit_b(checkpoint=sam_checkpoint)

# Завантаження LoRA і об'єднання з SAM
sam_lora = LoRA_sam(sam_base, rank=rank)
sam_lora.load_lora_parameters(lora_weights)

# Отримуємо модель із LoRA
sam = sam_lora.sam
sam.to(device)
sam.eval()  # важливо для inference


mask_generator1_ = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,
    pred_iou_thresh=0.5,
    stability_score_thresh=0.5,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=20,  
)

image_original = cv2.imread('/home/nt646jh/directory/folder/bc_nazarii_tymochko/img1.jpg')
image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)

####### Default filters:

# image = cv2.GaussianBlur(image_original, (5, 5), 0)
# clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
# image = np.stack([clahe.apply(image[:, :, i]) for i in range(3)], axis=2)
# image_resized = cv2.resize(image, (1024, 768))


####### Filter: Zhang et al. used image enhancement techniques to improve the quality of captured images, 
# making edges and particle boundaries more distinguishable for subsequent edge detection. 
# This likely involved adjusting contrast, brightness, or applying filters to reduce noise while preserving details.

image = cv2.cvtColor(image_original, cv2.COLOR_RGB2GRAY)  
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))  
image = clahe.apply(image)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) 
image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)  
image_resized = cv2.resize(image, (1024, 768))



masks_original = mask_generator1_.generate(image_resized)
print(f"Number of masks generated: {len(masks_original)}")

text_regions = [
    (0, 0, 310, 50),  # Top-left text
    (image_resized.shape[1] - 63 - 208, image_resized.shape[0] - 72 - (27 // 2), 208, 25)  # Bottom-right text (exact size and position)
]



filtered_masks = exclude_text_regions(masks_original, image_resized.shape, text_regions)
# sinter_stone_masks = filter_sinter_stones(filtered_masks, area_range=(10, 3500), min_aspect_ratio=0.7, min_iou_with_largest=0.5)
# sinter_stone_masks = merge_overlapping_masks(sinter_stone_masks, iou_threshold=0.5)

sinter_stone_masks = filter_sinter_stones(filtered_masks, area_range=(10, 5000), min_aspect_ratio=0.6, min_iou_with_largest=0.6)
sinter_stone_masks = merge_overlapping_masks(sinter_stone_masks, iou_threshold=0.2)

circumference_diameters = []
major_axes = []
minor_axes = []
areas = []
orientations = []
perimeters = []
box_points_list = []
for mask in sinter_stone_masks:
    result = get_rotated_min_bounding_rect(mask)
    if result is None:
        continue
    area, perimeter, circumference_diameter, major_axis, minor_axis, orientation, box_points = result
    circumference_diameters.append(circumference_diameter)
    major_axes.append(major_axis)
    areas.append(area)
    minor_axes.append(minor_axis)
    orientations.append(orientation)
    perimeters.append(perimeter)
    box_points_list.append(box_points)

# Print all metrics
print("\nPerimeters, Circumference Diameters, and Legendre Ellipse Properties of Rotated Minimum Bounding Boxes (in pixels):")
for i, (are, perim, diam, major, minor, orient) in enumerate(zip(areas, perimeters, circumference_diameters, major_axes, minor_axes, orientations), 1):
    print(f"Box {i}: Perimeter = {perim:.2f} pixels, Circumference Diameter = {diam:.2f} pixels, Legendre area = {are:.2f}")


segmented_image = color_segmentation(sinter_stone_masks, image_resized)
segmented_image_with_boxes = draw_bounding_boxes(segmented_image, box_points_list, color=(0, 255, 0), thickness=1)

plt.figure(figsize=(50, 25))
plt.imshow(segmented_image_with_boxes)
plt.axis('off')
plt.savefig('/home/nt646jh/directory/folder/bc_nazarii_tymochko/64_MBR_LoRA_2.png')
plt.close()



# Histogram for Area of Legendre Ellipse
counts, bin_edges = np.histogram(areas, bins=40)
plt.figure(figsize=(12, 6))
plt.hist(areas, bins=40, color='blue', alpha=0.8, edgecolor='black')
plt.title('Histogram of Legendre Ellipse Major Axis')
plt.xlabel('Major Axis (pixels)')
plt.ylabel('Frequency')
plt.xticks(bin_edges, rotation=90, ha='right')
plt.tight_layout()
plt.savefig('/home/nt646jh/directory/folder/bc_nazarii_tymochko/64_histogram_area_LoRA_2.png')
plt.close()


counts, bin_edges = np.histogram(circumference_diameters, bins=40)
plt.figure(figsize=(12, 6))
plt.hist(circumference_diameters, bins=40, color='purple', alpha=0.8, edgecolor='black')
plt.title('Histogram of MBR Circumference Diameter')
plt.xlabel('Circumference Diameter (pixels)')
plt.ylabel('Frequency')
plt.xticks(bin_edges, rotation=90, ha='right')
plt.tight_layout()
plt.savefig('/home/nt646jh/directory/folder/bc_nazarii_tymochko/64_histogram_circumference_diameter_LoRA_2.png')
plt.close()

# # Histogram for Major Axis of Legendre Ellipse
# counts, bin_edges = np.histogram(major_axes, bins=40)
# plt.figure(figsize=(12, 6))
# plt.hist(major_axes, bins=40, color='blue', alpha=0.8, edgecolor='black')
# plt.title('Histogram of Legendre Ellipse Major Axis')
# plt.xlabel('Major Axis (pixels)')
# plt.ylabel('Frequency')
# plt.xticks(bin_edges, rotation=90, ha='right')
# plt.tight_layout()
# plt.savefig('/home/nt646jh/directory/folder/bc_nazarii_tymochko/128_histogram_major_axis.png')
# plt.close()

# # Histogram for Minor Axis of Legendre Ellipse
# counts, bin_edges = np.histogram(minor_axes, bins=40)
# plt.figure(figsize=(12, 6))
# plt.hist(minor_axes, bins=40, color='green', alpha=0.8, edgecolor='black')
# plt.title('Histogram of Legendre Ellipse Minor Axis')
# plt.xlabel('Minor Axis (pixels)')
# plt.ylabel('Frequency')
# plt.xticks(bin_edges, rotation=90, ha='right')
# plt.tight_layout()
# plt.savefig('/home/nt646jh/directory/folder/bc_nazarii_tymochko/128_histogram_minor_axis.png')
# plt.close()

# # Histogram for Orientation of Legendre Ellipse
# counts, bin_edges = np.histogram(orientations, bins=40)
# plt.figure(figsize=(12, 6))
# plt.hist(orientations, bins=40, color='orange', alpha=0.8, edgecolor='black')
# plt.title('Histogram of Legendre Ellipse Orientation')
# plt.xlabel('Orientation (degrees)')
# plt.ylabel('Frequency')
# plt.xticks(bin_edges, rotation=90, ha='right')
# plt.tight_layout()
# plt.savefig('/home/nt646jh/directory/folder/bc_nazarii_tymochko/128_histogram_orientation.png')
# plt.close()

def get_feret_diameter_and_angle(mask):
    binary_mask = mask['segmentation'].astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    largest_contour = max(contours, key=cv2.contourArea)
    points = largest_contour[:, 0, :]
    if len(points) < 2:
        return None, None, None
    # Vectorized pairwise distance calculation
    dists = distance.cdist(points, points)
    max_idx = np.unravel_index(np.argmax(dists), dists.shape)
    pt1, pt2 = points[max_idx[0]], points[max_idx[1]]
    feret_diameter = dists[max_idx]
    angle = np.degrees(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))
    M = cv2.moments(largest_contour)
    if M['m00'] == 0:
        cx, cy = 0, 0
    else:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    return feret_diameter, angle, (cx, cy)

def draw_feret_ellipses(image, feret_params_list, color=(255, 0, 0), thickness=2):
    img = image.copy()
    for feret_diameter, angle, center in feret_params_list:
        if feret_diameter is not None and center is not None:
            axes = (int(feret_diameter // 2), int(feret_diameter // 4))  # Minor axis is arbitrary for visualization
            cv2.ellipse(img, center, axes, angle, 0, 360, color, thickness)
    return img

feret_diameters = []
feret_params_list = []
for mask in sinter_stone_masks:
    feret_diameter, feret_angle, feret_center = get_feret_diameter_and_angle(mask)
    feret_diameters.append(feret_diameter)
    feret_params_list.append((feret_diameter, feret_angle, feret_center))

segmented_image_with_feret = draw_feret_ellipses(segmented_image_with_boxes, feret_params_list, color=(255, 0, 0), thickness=2)

plt.figure(figsize=(50, 25))
plt.imshow(segmented_image_with_feret)
plt.axis('off')
plt.savefig('/home/nt646jh/directory/folder/bc_nazarii_tymochko/128_feret_ellipse.png')
plt.close()

# Histogram for Feret diameter
feret_diameters_clean = [d for d in feret_diameters if d is not None]
counts, bin_edges = np.histogram(feret_diameters_clean, bins=40)
plt.figure(figsize=(12, 6))
plt.hist(feret_diameters_clean, bins=40, color='red', alpha=0.8, edgecolor='black')
plt.title('Histogram of Feret Diameter (Contour)')
plt.xlabel('Feret Diameter (pixels)')
plt.ylabel('Frequency')
plt.xticks(bin_edges, rotation=90, ha='right')
plt.tight_layout()
plt.savefig('/home/nt646jh/directory/folder/bc_nazarii_tymochko/64_histogram_feret_diameter_contour.png')
plt.close()