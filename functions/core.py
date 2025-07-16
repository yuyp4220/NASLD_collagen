import os
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.ndimage import binary_fill_holes
from scipy.spatial import ConvexHull
from skimage import color, filters
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_closing, disk, skeletonize, remove_small_objects
)
from sklearn.cluster import KMeans, DBSCAN


def efficient_histogram_downsample(data, max_total=10000, return_indices=False):
    data = data.flatten()
    hist, bin_edges = np.histogram(data, bins=180, range=(0, 180))
    bin_indices = np.digitize(data, bin_edges[:-1], right=True)

    total = 0
    selected_indices = []

    for b in range(1, 181):
        idx = np.where(bin_indices == b)[0]
        if idx.size == 0:
            continue
        n_select = max(1, int((idx.size / len(data)) * max_total))
        if total + n_select > max_total:
            n_select = max_total - total
        if n_select <= 0:
            break
        chosen = np.random.choice(idx, size=n_select, replace=False)
        selected_indices.extend(chosen.tolist())
        total += n_select

    selected_indices = np.array(selected_indices)
    downsampled = data[selected_indices]

    if return_indices:
        return selected_indices
    else:
        return downsampled


def fast_lab_spread_from_hsv(h, s, v):
    rgb = cv2.cvtColor(np.stack([h, s, v], axis=1).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_HSV2RGB)
    lab = color.rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)
    if len(lab) == 0:
        return np.nan
    q75, q25 = np.percentile(lab, [75, 25], axis=0)
    return float(np.linalg.norm(q75 - q25))


def extract_global_hue_range_with_spread(tile_folder, max_sample_ratio=0.1, spread_thresh=10):
    hue_all = []
    s_all = []
    v_all = []

    for fname in tqdm(os.listdir(tile_folder)):
        if not fname.lower().endswith(".jpg"):
            continue
        path = os.path.join(tile_folder, fname)
        bgr = cv2.imread(path)
        if bgr is None:
            continue

        try:
            image, gray = load_image(path)
            foreground_mask, foreground_img = get_otsu_mask(image, gray)
        except Exception:
            foreground_img = bgr

        hsv = cv2.cvtColor(foreground_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        mask = s > 0
        hue_all.append(h[mask])
        s_all.append(s[mask])
        v_all.append(v[mask])

    hue_concat = np.concatenate(hue_all)
    s_all = np.concatenate(s_all)
    v_all = np.concatenate(v_all)

    max_total = int(len(hue_concat) * max_sample_ratio)
    indices = efficient_histogram_downsample(hue_concat, max_total=max_total, return_indices=True)

    hue_ds = hue_concat[indices]
    s_ds = s_all[indices]
    v_ds = v_all[indices]


    
    hist, bin_edges = np.histogram(hue_ds, bins=180, range=(0, 180))
    top3_indices = np.argsort(hist)[-3:]  
    top3_hues = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in top3_indices]

    
    red_count = sum((hue < 10 or hue > 160) for hue in top3_hues)

    
    k = 3 if red_count >= 2 else 5
    print(f"Top3 hue centers: {[f'{h:.1f}' for h in top3_hues]}, red count = {red_count}, using K={k}")

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    hue_labels = kmeans.fit_predict(hue_ds.reshape(-1, 1))
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_idx = np.argsort(cluster_centers)

    hue_ranges = []
    for new_id, old_id in enumerate(sorted_idx):
        cluster_hue = hue_ds[hue_labels == old_id]
        hmin, hmax = (0, 0) if cluster_hue.size == 0 else (int(cluster_hue.min()), int(cluster_hue.max()))
        mean_hue = 0 if cluster_hue.size == 0 else float(cluster_hue.mean())
        hue_ranges.append((hmin, hmax))
        print(f"Hue Cluster {new_id}: Mean = {mean_hue:.2f}, Range = [{hmin}, {hmax}]")

    if k == 3:
        selected_ids = [sorted_idx[-1]]
        hsv_selected = []
        for cid in selected_ids:
            mask = (hue_labels == cid)
            h_sel, s_sel, v_sel = hue_ds[mask], s_ds[mask], v_ds[mask]
            hsv_selected.append(np.stack([h_sel, s_sel, v_sel], axis=1))
        hsv_selected = np.concatenate(hsv_selected, axis=0)
        k_hsv = 5
        kmeans_hsv = KMeans(n_clusters=k_hsv, random_state=0, n_init=10)
        hsv_labels = kmeans_hsv.fit_predict(hsv_selected)
        center_scores = kmeans_hsv.cluster_centers_.sum(axis=1)
        sorted_hsv_idx = np.argsort(center_scores)
        label_remap = {old: new for new, old in enumerate(sorted_hsv_idx)}
        labels_sorted = np.vectorize(label_remap.get)(hsv_labels)
        target_label = k_hsv - 1
        group = hsv_selected[labels_sorted == target_label]
        hmin, hmax = int(group[:, 0].min()), int(group[:, 0].max())
        smin, smax = int(group[:, 1].min()), int(group[:, 1].max())
        vmin, vmax = int(group[:, 2].min()), int(group[:, 2].max())
        print(f"Selected HSV Cluster {target_label} (H+S+V strongest):")
        print(f"  H = [{hmin}, {hmax}], S = [{smin}, {smax}], V = [{vmin}, {vmax}]")
        selected_ranges = [(hmin, hmax), (smin, smax), (vmin, vmax)]
        return selected_ranges, True

    else:
        hmin1, hmax1 = hue_ranges[0]
        hmin2, hmax2 = hue_ranges[-1]
        mask_combined = ((hue_ds >= hmin1) & (hue_ds <= hmax1)) | ((hue_ds >= hmin2) & (hue_ds <= hmax2))
        h_sel = hue_ds[mask_combined]
        s_sel = s_ds[mask_combined]
        v_sel = v_ds[mask_combined]
        spread = fast_lab_spread_from_hsv(h_sel, s_sel, v_sel)
        print(f"Fast LAB spread = {spread:.2f}")
        if spread < spread_thresh:
            selected_ranges = [(hmin1, hmax1), (hmin2, hmax2)]
            print("Spread < threshold, accept original hue ranges.")
            return selected_ranges, False
        else:
            kmeans_ref = KMeans(n_clusters=5, random_state=42, n_init=10)
            labels_ref = kmeans_ref.fit_predict(h_sel.reshape(-1, 1))
            centers_ref = kmeans_ref.cluster_centers_.flatten()
            sorted_ref = np.argsort(centers_ref)
            hue_ranges_ref = []
            for new_id, old_id in enumerate(sorted_ref):
                cluster = h_sel[labels_ref == old_id]
                if cluster.size == 0:
                    continue
                hue_ranges_ref.append((int(cluster.min()), int(cluster.max())))
            final_ranges = [hue_ranges_ref[0], hue_ranges_ref[-1]]
            print("Refined hue ranges due to high spread:")
            for i, (hmin, hmax) in enumerate(final_ranges):
                print(f"  Refined Hue Cluster {i}: H = [{hmin}, {hmax}]")
            return final_ranges, False

    
    
def generate_collagen_mask_direct_robust(foreground_bgr, selected_ranges, slide_is_red, s_thresh=0):
    hsv = cv2.cvtColor(foreground_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if slide_is_red:
        # selected_ranges: [(hmin,hmax), (smin,smax), (vmin,vmax)]
        (hmin, hmax), (smin, smax), (vmin, vmax) = selected_ranges
        mask = (
            (h >= hmin) & (h <= hmax) &
            (s >= smin) & (s <= smax) &
            (v >= vmin) & (v <= vmax)
        )
    else:
        # selected_ranges: [(hmin1,hmax1), (hmin2,hmax2)]
        valid_saturation = s > s_thresh
        mask = np.zeros_like(h, dtype=bool)
        for (hmin, hmax) in selected_ranges:
            mask |= (h >= hmin) & (h <= hmax)
        mask &= valid_saturation

    return mask


def load_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image, gray

def get_otsu_mask(original_img,gray_img):
    threshold = filters.threshold_otsu(gray_img)
    mask = gray_img < threshold

    
    foreground = original_img.copy()
    if mask.shape != original_img.shape[:2]:
        raise ValueError("Mask and image shape mismatch at Otsu")
    foreground[~mask] = 255
    mask = (gray_img < threshold).astype(np.uint8)
    # plt.figure(figsize=(10, 4))
    # plt.imshow(mask, cmap='gray')
    # plt.title("Otsu Mask")
    # plt.axis('off')
    # plt.show()

    return mask, foreground

def mass_for_contour(foreground_mask):
    closed_tissue_mask = binary_closing(foreground_mask, footprint=disk(10))
    mass_mask = binary_fill_holes(closed_tissue_mask)
    # plt.title('Tissue Mask')
    # plt.axis('off')
    # plt.imshow(mass_mask,cmap='gray')
    return mass_mask

def red_pixel_ratio_bgr(bgr_image, s_thresh=50, v_thresh=50):
    """ """
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    
    
    red_mask = (((h <= 20) | (h >= 160)) &
                (s >= s_thresh) &
                (v >= v_thresh))

    red_ratio = np.sum(red_mask) / red_mask.size
    return red_ratio

def hue_kmeans_visualize(foreground, cluster):
    """ """
    
    tissue_dist = cv2.distanceTransform((tissue_contour_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)

    
    near_overlap = np.logical_and(edge_collagen_mask > 0, tissue_dist <= distance_threshold)

    ys, xs = np.where(near_overlap)
    points = np.stack([xs, ys], axis=1)
    return points

def plot_clusters_on_masks(collagen_mask, boundary_mask, points, eps=100, min_samples=10):
    """ """
    if points is None or len(points) == 0:
        return np.array([]), set()
    background = collagen_mask   

    # DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_

    
    # fig, ax = plt.subplots()
    # ax.imshow(background, cmap='gray', interpolation='none')

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    colors = plt.get_cmap('tab20', len(unique_labels))

    # for k in unique_labels:
    #     class_member_mask = (labels == k)
    #     xy = points[class_member_mask]
    #     if k == -1:
    
    #         ax.plot(xy[:, 0], xy[:, 1], 'k.', markersize=5)
    #     else:
    #         ax.plot(xy[:, 0], xy[:, 1], '.', markersize=5, color=colors(k))

    # ax.axis('off')
    # plt.show()
    return labels, unique_labels

def plot_convex_hulls_on_mask(collagen_mask, points, labels, unique_labels, linearity_threshold=1e-5):
    """ """
    
    # fig, ax = plt.subplots()
    # ax.imshow(collagen_mask, cmap='gray', interpolation='none')

    
    hull_list = []

    for k in unique_labels:
        if k == -1:
            continue
        
        cluster_points = points[labels == k]

        if len(cluster_points) >= 10:
            
            cov = np.cov(cluster_points.T)
            eigvals = np.linalg.eigvalsh(cov)
            if eigvals[0] < linearity_threshold:
                print(f"Cluster {k}: Skipped (near-linear, eigval={eigvals[0]:.2e})")
                continue

            
            hull = ConvexHull(cluster_points)
            hull_points = cluster_points[hull.vertices]

    
            # ax.plot(np.append(hull_points[:,0], hull_points[0,0]),
            #         np.append(hull_points[:,1], hull_points[0,1]),
            #         '-', label=f'Cluster {k}')
            
            hull_list.append(hull_points)

    
    # ax.set_title('Convex Hulls on Collagen Mask')
    # ax.axis('off')
    # plt.show()
    # plt.close(fig)

    
    hull_edge_mask = np.zeros(collagen_mask.shape, dtype=np.uint8)

    for hull_points in hull_list:
        if hull_points.shape[0] >= 10:  
            
            pts = np.round(hull_points).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(hull_edge_mask, [pts], isClosed=True, color=255, thickness=1)

    return hull_edge_mask

def get_closed_tissue_mask(contour_mask,hull_mask,image,foreground_mask):
    merge_contour_hull = np.logical_or(contour_mask > 0, hull_mask > 0).astype(np.uint8)
    vis, contour_mask = draw_precise_tissue_contour(merge_contour_hull,image)
    new_tissue_mask = np.logical_or(foreground_mask,contour_mask)
    return new_tissue_mask

def get_hole_mask(merged_mask):
    closed_tissue_mask = binary_closing(merged_mask, footprint=disk(10))
    tissue_filled_mask = binary_fill_holes(closed_tissue_mask)
    hole_mask = (tissue_filled_mask == 1) & (closed_tissue_mask == 0)
    # plt.title('Holes inside tissue mask')
    # plt.axis('off')
    # plt.imshow(hole_mask,cmap='gray')
    return tissue_filled_mask,hole_mask

def get_params(path,selected_ranges,slide_is_red):    
    image, gray = load_image(path)
    foreground_mask, foreground_image = get_otsu_mask(image,gray)
    mass_mask = mass_for_contour(foreground_mask)
    
    collagen_mask = generate_collagen_mask_direct_robust(foreground_image,selected_ranges,slide_is_red)
    edge_collagen_mask = get_edge_collagen(collagen_mask)
    vis,contour_mask = draw_precise_tissue_contour(mass_mask,image)
    points = find_overlapping_points(contour_mask,edge_collagen_mask)
    labels, unique_labels = plot_clusters_on_masks(edge_collagen_mask,contour_mask,points)
    hull_mask = plot_convex_hulls_on_mask(edge_collagen_mask,points,labels,unique_labels)
    closed_edge_mask = get_closed_tissue_mask(contour_mask,hull_mask,image,foreground_mask)
    tissue_filled_mask,hole_mask = get_hole_mask(closed_edge_mask)
    params = {
    'mass_mask': tissue_filled_mask,
    'collagen_mask': collagen_mask,
    'pixel_per_micron': 4.1667}
    
    return params

def extract_basic_collagen_features(mass_mask, collagen_mask, pixel_per_micron=4.1667):
    import numpy as np
    import pandas as pd

    pixel_area_to_micron2 = (1 / pixel_per_micron) ** 2

    tissue_area = np.sum(mass_mask > 0) * pixel_area_to_micron2
    collagen_area = np.sum(collagen_mask > 0) * pixel_area_to_micron2
    collagen_pct = (collagen_area / tissue_area * 100) if tissue_area > 0 else 0

    data = {
        "Tissue Area (μm²)": [round(tissue_area, 2)],
        "Collagen Area (μm²)": [round(collagen_area, 2)],
        "Collagen %": [round(collagen_pct, 2)]
    }

    return pd.DataFrame(data)


def aggregate_basic_collagen_features(json_folder_path):
    json_folder = Path(json_folder_path)

    total_tissue_area = 0.0
    total_collagen_area = 0.0

    for json_file in json_folder.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
            total_tissue_area += data.get("Tissue Area (μm²)", 0.0)
            total_collagen_area += data.get("Collagen Area (μm²)", 0.0)

    collagen_pct = (total_collagen_area / total_tissue_area * 100) if total_tissue_area > 0 else 0.0

    results = [
        {
            "Feature": "Total Tissue Area",
            "Aggregated Value": round(total_tissue_area, 2),
            "Unit": "μm²"
        },
        {
            "Feature": "Total Collagen Area",
            "Aggregated Value": round(total_collagen_area, 2),
            "Unit": "μm²"
        },
        {
            "Feature": "Collagen %",
            "Aggregated Value": round(collagen_pct, 2),
            "Unit": "%"
        }
    ]

    return pd.DataFrame(results)