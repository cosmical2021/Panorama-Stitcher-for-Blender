bl_info = {
    "name": "Spherical Panorama Heal Stitcher (v2.8 Fixed UI & Auto-Span)",
    "author": "AI Collaborator",
    "version": (2, 8, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar (N) > Pano Stitcher & View3D Header > Pano Stitcher",
    "description": "Auto-detects 0-360° x 0-180° panorama span, fixes distortion rings, levels WB/Exposure, OpenCV seams, and Frequency-Separated Heal blending",
    "category": "Material",
}

import bpy
import os
import math
import struct
import tempfile
import shutil
import numpy as np
from mathutils import Matrix, Euler, Vector
from bpy_extras.io_utils import ImportHelper
from bpy.props import (
    CollectionProperty,
    FloatProperty,
    IntProperty,
    BoolProperty,
    EnumProperty,
    StringProperty,
    PointerProperty,
)
from bpy.types import Operator, Panel, Menu, PropertyGroup

OPENCV_AVAILABLE = False
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    pass


# =====================================================================
# 1. PURE-PYTHON EXIF FOCAL LENGTH & ORIENTATION PARSER
# =====================================================================

def read_exif_focal_and_orientation(filepath, default_sensor_width=36.0, default_sensor_height=24.0):
    focal_mm = None
    focal_35mm = None
    orientation = 1

    try:
        with open(filepath, 'rb') as f:
            data = f.read(65536)
        if data[0:2] == b'\xff\xd8':
            idx = 2
            while idx < len(data) - 4:
                if data[idx] != 0xFF:
                    break
                marker = data[idx + 1]
                length = struct.unpack('>H', data[idx + 2:idx + 4])[0]
                if marker == 0xE1 and data[idx + 4:idx + 10] == b'Exif\x00\x00':
                    tiff = data[idx + 10:idx + 2 + length]
                    focal_mm, focal_35mm, orientation = _parse_tiff_exif(tiff)
                    break
                idx += 2 + length
    except Exception as e:
        print(f"[Pano Stitcher] EXIF read warning for {filepath}: {e}")

    if focal_35mm:
        return focal_35mm, 36.0, 24.0, orientation
    if focal_mm:
        return focal_mm, default_sensor_width, default_sensor_height, orientation
    return None, default_sensor_width, default_sensor_height, orientation


def _parse_tiff_exif(tiff):
    endian = '<' if tiff[0:2] == b'II' else '>'
    ifd0_offset = struct.unpack(endian + 'I', tiff[4:8])[0]

    focal_mm = None
    focal_35mm = None
    orientation = 1

    def read_ifd(offset):
        nonlocal focal_mm, focal_35mm, orientation
        if offset + 2 > len(tiff):
            return None
        num_entries = struct.unpack(endian + 'H', tiff[offset:offset + 2])[0]
        exif_ifd_ptr = None
        for i in range(num_entries):
            entry = offset + 2 + i * 12
            if entry + 12 > len(tiff):
                break
            tag, fmt, count, val_raw = struct.unpack(endian + 'HHII', tiff[entry:entry + 12])
            if tag == 0x8769:
                exif_ifd_ptr = val_raw
            elif tag == 0x0112 and fmt == 3:
                orientation = struct.unpack(endian + 'H', tiff[entry + 8:entry + 10])[0]
            elif tag == 0x920A and fmt == 5:
                if val_raw + 8 <= len(tiff):
                    num, den = struct.unpack(endian + 'II', tiff[val_raw:val_raw + 8])
                    if den != 0:
                        focal_mm = num / den
            elif tag == 0xA405 and fmt == 3:
                val_short = struct.unpack(endian + 'H', tiff[entry + 8:entry + 10])[0]
                if val_short > 0:
                    focal_35mm = float(val_short)
        return exif_ifd_ptr

    exif_ptr = read_ifd(ifd0_offset)
    if exif_ptr:
        read_ifd(exif_ptr)

    return focal_mm, focal_35mm, orientation


def compute_orientation_fov_scales(img_w, img_h, focal_mm, sensor_long_mm=36.0, sensor_short_mm=24.0, mode='AUTO'):
    if mode == 'PORTRAIT':
        is_portrait = True
    elif mode == 'LANDSCAPE':
        is_portrait = False
    else:
        is_portrait = img_h > img_w

    aspect = img_w / max(img_h, 1)
    if is_portrait:
        sensor_x = min(sensor_long_mm, sensor_short_mm)
        fov_scale_x = sensor_x / focal_mm
        fov_scale_y = fov_scale_x / aspect
    else:
        sensor_x = max(sensor_long_mm, sensor_short_mm)
        fov_scale_x = sensor_x / focal_mm
        fov_scale_y = fov_scale_x / aspect

    return is_portrait, fov_scale_x, fov_scale_y


# =====================================================================
# 2. WIDE-ANGLE ROI OPTICAL FLOW & SAFE OPENCV FEATURE MATCHING
# =====================================================================

def _create_safe_opencv_detector():
    if not OPENCV_AVAILABLE:
        return None, None

    for mod in (cv2, getattr(cv2, 'features2d', None)):
        if mod is not None and hasattr(mod, 'AKAZE_create'):
            try:
                return mod.AKAZE_create(), cv2.NORM_HAMMING
            except Exception:
                pass
        if mod is not None and hasattr(mod, 'AKAZE') and hasattr(mod.AKAZE, 'create'):
            try:
                return mod.AKAZE.create(), cv2.NORM_HAMMING
            except Exception:
                pass

    for mod in (cv2, getattr(cv2, 'features2d', None)):
        if mod is not None and hasattr(mod, 'ORB_create'):
            try:
                return mod.ORB_create(nfeatures=2500), cv2.NORM_HAMMING
            except Exception:
                pass
        if mod is not None and hasattr(mod, 'ORB') and hasattr(mod.ORB, 'create'):
            try:
                return mod.ORB.create(nfeatures=2500), cv2.NORM_HAMMING
            except Exception:
                pass

    for mod in (cv2, getattr(cv2, 'features2d', None)):
        if mod is not None and hasattr(mod, 'SIFT_create'):
            try:
                return mod.SIFT_create(nfeatures=2500), cv2.NORM_L2
            except Exception:
                pass
        if mod is not None and hasattr(mod, 'SIFT') and hasattr(mod.SIFT, 'create'):
            try:
                return mod.SIFT.create(nfeatures=2500), cv2.NORM_L2
            except Exception:
                pass

    for mod in (cv2, getattr(cv2, 'features2d', None)):
        if mod is not None and hasattr(mod, 'BRISK_create'):
            try:
                return mod.BRISK_create(), cv2.NORM_HAMMING
            except Exception:
                pass

    return None, None


def estimate_roi_pre_shift_numpy(img_path_a, img_path_b, max_shift_ratio=0.70):
    if OPENCV_AVAILABLE and hasattr(cv2, 'imread') and hasattr(cv2, 'phaseCorrelate'):
        try:
            im_a = cv2.imread(img_path_a, cv2.IMREAD_GRAYSCALE)
            im_b = cv2.imread(img_path_b, cv2.IMREAD_GRAYSCALE)
            if im_a is not None and im_b is not None:
                small_w, small_h = 256, 256
                ga = cv2.resize(im_a, (small_w, small_h)).astype(np.float32)
                gb = cv2.resize(im_b, (small_w, small_h)).astype(np.float32)
                win = cv2.createHanningWindow((small_w, small_h), cv2.CV_32F)
                (dx, dy), response = cv2.phaseCorrelate(ga, gb, win)
                du = dx / small_w
                dv = -dy / small_h
                if abs(du) < max_shift_ratio and abs(dv) < max_shift_ratio and response > 0.05:
                    return du, dv
        except Exception:
            pass
    return 0.0, 0.0


def track_pair_with_blender_clip_roi(context, img_path_a, img_path_b, search_size=220, roi_overlap_ratio=0.55):
    temp_dir = tempfile.mkdtemp(prefix="blender_pano_flow_")
    ext = os.path.splitext(img_path_a)[1] or ".jpg"
    f1_name = f"seq_0001{ext}"
    f2_name = f"seq_0002{ext}"
    frame1_path = os.path.join(temp_dir, f1_name)
    frame2_path = os.path.join(temp_dir, f2_name)

    du_pre, dv_pre = estimate_roi_pre_shift_numpy(img_path_a, img_path_b)

    if OPENCV_AVAILABLE and hasattr(cv2, 'warpAffine') and (abs(du_pre) > 0.04 or abs(dv_pre) > 0.04):
        im_a = cv2.imread(img_path_a)
        im_b = cv2.imread(img_path_b)
        if im_a is not None and im_b is not None:
            h, w = im_b.shape[:2]
            tx = -du_pre * w
            ty = dv_pre * h
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            im_b_shifted = cv2.warpAffine(im_b, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(frame1_path, im_a)
            cv2.imwrite(frame2_path, im_b_shifted)
        else:
            du_pre, dv_pre = 0.0, 0.0
            shutil.copy2(img_path_a, frame1_path)
            shutil.copy2(img_path_b, frame2_path)
    else:
        du_pre, dv_pre = 0.0, 0.0
        shutil.copy2(img_path_a, frame1_path)
        shutil.copy2(img_path_b, frame2_path)

    area = context.area
    orig_type = area.type
    area.type = 'CLIP_EDITOR'
    space = area.spaces.active

    orig_frame = context.scene.frame_current
    matches = []
    clip = None

    try:
        existing_clips = set(bpy.data.movieclips)
        with context.temp_override(area=area, space_data=space):
            bpy.ops.clip.open(
                directory=temp_dir,
                files=[{"name": f1_name}, {"name": f2_name}],
                use_sequence_detection=True
            )

        new_clips = set(bpy.data.movieclips) - existing_clips
        clip = new_clips.pop() if new_clips else space.clip
        space.clip = clip

        settings = clip.tracking.settings
        settings.default_motion_model = 'Affine'
        settings.default_pattern_size = 21
        settings.default_search_size = min(max(search_size, 61), 250)
        settings.default_correlation_min = 0.62
        settings.use_default_normalization = True

        context.scene.frame_set(1)
        with context.temp_override(area=area, space_data=space, edit_movieclip=clip):
            bpy.ops.clip.detect_features(
                placement='FRAME',
                margin=16,
                threshold=0.20,
                min_distance=32
            )

            if abs(du_pre) > 0.04 or abs(dv_pre) > 0.04:
                u_min = max(0.02, -du_pre - roi_overlap_ratio)
                u_max = min(0.98, 1.0 - du_pre + roi_overlap_ratio)
                for tr in list(clip.tracking.tracks):
                    m = tr.markers.find_frame(1)
                    if m and not (u_min <= m.co[0] <= u_max):
                        clip.tracking.tracks.remove(tr)

            bpy.ops.clip.select_all(action='SELECT')
            bpy.ops.clip.track_markers(backwards=False, sequence=False)

        for track in clip.tracking.tracks:
            m1 = track.markers.find_frame(1)
            m2 = track.markers.find_frame(2)
            if m1 and m2 and not m1.mute and not m2.mute:
                u1, v1 = m1.co[0] - 0.5, m1.co[1] - 0.5
                u2 = (m2.co[0] + du_pre) - 0.5
                v2 = (m2.co[1] + dv_pre) - 0.5
                if -0.5 <= u2 <= 0.5 and -0.5 <= v2 <= 0.5:
                    matches.append((u1, v1, u2, v2))

    except Exception as e:
        print(f"[Pano Stitcher] Optical flow error: {e}")
    finally:
        area.type = orig_type
        context.scene.frame_set(orig_frame)
        if clip and clip.name in bpy.data.movieclips:
            bpy.data.movieclips.remove(clip)
        shutil.rmtree(temp_dir, ignore_errors=True)

    return matches


def match_pair_opencv(img_path_a, img_path_b, roi_fraction=0.75):
    detector, norm_type = _create_safe_opencv_detector()
    if detector is None:
        return []

    try:
        im_a = cv2.imread(img_path_a, cv2.IMREAD_GRAYSCALE)
        im_b = cv2.imread(img_path_b, cv2.IMREAD_GRAYSCALE)
        if im_a is None or im_b is None:
            return []

        ha, wa = im_a.shape[:2]
        hb, wb = im_b.shape[:2]

        max_dim = 1200
        scale_a = min(1.0, max_dim / max(wa, ha))
        scale_b = min(1.0, max_dim / max(wb, hb))
        if scale_a < 1.0:
            im_a = cv2.resize(im_a, (int(wa * scale_a), int(ha * scale_a)))
        if scale_b < 1.0:
            im_b = cv2.resize(im_b, (int(wb * scale_b), int(hb * scale_b)))

        sha, swa = im_a.shape[:2]
        shb, swb = im_b.shape[:2]

        kp_a, desc_a = detector.detectAndCompute(im_a, None)
        kp_b, desc_b = detector.detectAndCompute(im_b, None)

        if desc_a is None or desc_b is None or len(kp_a) < 8 or len(kp_b) < 8:
            return []

        matcher = cv2.BFMatcher(norm_type)
        raw_matches = matcher.knnMatch(desc_a, desc_b, k=2)

        matches = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.76 * n.distance:
                    pt_a = kp_a[m.queryIdx].pt
                    pt_b = kp_b[m.trainIdx].pt
                    u1 = (pt_a[0] / swa) - 0.5
                    v1 = 0.5 - (pt_a[1] / sha)
                    u2 = (pt_b[0] / swb) - 0.5
                    v2 = 0.5 - (pt_b[1] / shb)
                    matches.append((u1, v1, u2, v2))

        return matches
    except Exception as e:
        print(f"[Pano Stitcher] OpenCV matcher fallback warning: {e}")
        return []


# =====================================================================
# 3. RADIAL DISTORTION, AUTO-SPAN DETECTOR & SPHERICAL ROTATION SOLVER
# =====================================================================

def undistort_uv_point(u, v, aspect, k1, k2):
    if k1 == 0.0 and k2 == 0.0:
        return u, v
    x = u * 2.0
    y = (v * 2.0) / max(aspect, 1e-4)
    xu, yu = x, y
    for _ in range(5):
        r2 = xu * xu + yu * yu
        radial = max(0.35, 1.0 + k1 * r2 + k2 * r2 * r2)
        xu = x / radial
        yu = y / radial
    return xu * 0.5, (yu * max(aspect, 1e-4)) * 0.5


def solve_relative_rotation_svd(matches, fov_ax, fov_ay, fov_bx, fov_by, k1=0.0, k2=0.0):
    if len(matches) < 4:
        return None, 0, []

    aspect_a = fov_ax / max(fov_ay, 1e-5)
    aspect_b = fov_bx / max(fov_by, 1e-5)

    rays_a = []
    rays_b = []
    for u1, v1, u2, v2 in matches:
        u1_u, v1_u = undistort_uv_point(u1, v1, aspect_a, k1, k2)
        u2_u, v2_u = undistort_uv_point(u2, v2, aspect_b, k1, k2)

        ra = np.array([u1_u * fov_ax, v1_u * fov_ay, -1.0], dtype=np.float64)
        rb = np.array([u2_u * fov_bx, v2_u * fov_by, -1.0], dtype=np.float64)
        ra /= np.linalg.norm(ra)
        rb /= np.linalg.norm(rb)
        rays_a.append(ra)
        rays_b.append(rb)

    rays_a = np.array(rays_a)
    rays_b = np.array(rays_b)

    def kabsch_rotation(P, Q):
        H = Q.T @ P
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1.0, 1.0, d])
        return Vt.T @ D @ U.T

    n_pts = len(matches)
    best_R = kabsch_rotation(rays_a, rays_b)
    best_inliers = np.ones(n_pts, dtype=bool)
    max_inliers = n_pts

    if n_pts >= 6:
        rng = np.random.default_rng(42)
        max_inliers = 0
        cos_thresh = math.cos(math.radians(1.5))

        for _ in range(min(200, n_pts * 12)):
            idx = rng.choice(n_pts, size=3, replace=False)
            R_cand = kabsch_rotation(rays_a[idx], rays_b[idx])
            rotated_b = (R_cand @ rays_b.T).T
            dots = np.sum(rays_a * rotated_b, axis=1)
            inliers = dots > cos_thresh
            count = int(np.sum(inliers))
            if count > max_inliers:
                max_inliers = count
                best_inliers = inliers

        if max_inliers >= 4:
            best_R = kabsch_rotation(rays_a[best_inliers], rays_b[best_inliers])
        else:
            return None, 0, []

    inlier_list = [matches[idx] for idx in range(n_pts) if best_inliers[idx]]
    return Matrix(best_R.tolist()), max_inliers, inlier_list


def estimate_radial_distortion_k1_k2(all_pair_inliers):
    if not all_pair_inliers:
        return 0.0, 0.0

    best_k1, best_k2 = 0.0, 0.0
    best_score = 1e9

    for k1_cand in np.linspace(-0.12, 0.08, 17):
        for k2_cand in np.linspace(-0.03, 0.03, 5):
            total_err = 0.0
            total_pts = 0
            for matches, fax, fay, fbx, fby in all_pair_inliers:
                if len(matches) < 8:
                    continue
                aspect_a = fax / max(fay, 1e-5)
                aspect_b = fbx / max(fby, 1e-5)
                ra_list, rb_list = [], []
                for u1, v1, u2, v2 in matches[:40]:
                    u1_u, v1_u = undistort_uv_point(u1, v1, aspect_a, k1_cand, k2_cand)
                    u2_u, v2_u = undistort_uv_point(u2, v2, aspect_b, k1_cand, k2_cand)
                    ra = np.array([u1_u * fax, v1_u * fay, -1.0])
                    rb = np.array([u2_u * fbx, v2_u * fby, -1.0])
                    ra /= np.linalg.norm(ra)
                    rb /= np.linalg.norm(rb)
                    ra_list.append(ra)
                    rb_list.append(rb)
                P = np.array(ra_list)
                Q = np.array(rb_list)
                H = Q.T @ P
                U, _, Vt = np.linalg.svd(H)
                d = np.linalg.det(Vt.T @ U.T)
                R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
                rot_Q = (R @ Q.T).T
                err = np.mean(1.0 - np.sum(P * rot_Q, axis=1))
                total_err += err * len(P)
                total_pts += len(P)

            if total_pts >= 12:
                reg_penalty = 1e-4 * (k1_cand ** 2 + 4.0 * (k2_cand ** 2))
                score = (total_err / total_pts) + reg_penalty
                if score < best_score:
                    best_score = score
                    best_k1 = float(k1_cand)
                    best_k2 = float(k2_cand)

    return best_k1, best_k2


def detect_panorama_angular_span(empties):
    if not empties:
        return 360.0, 180.0, True, (0.0, 1.0, 0.0, 1.0)

    inside_masks, _, _ = _build_spherical_projection_grid(empties, grid_w=360, grid_h=180, k1=0.0, k2=0.0)
    union_mask = np.any(inside_masks, axis=0)

    col_any = np.any(union_mask, axis=0)
    row_any = np.any(union_mask, axis=1)

    if not np.any(col_any):
        return 360.0, 180.0, True, (0.0, 1.0, 0.0, 1.0)

    valid_rows = np.where(row_any)[0]
    r_min, r_max = int(valid_rows[0]), int(valid_rows[-1])
    v_span_deg = float(min(180.0, max(10.0, (r_max - r_min + 1))))
    v_min = max(0.0, r_min / 180.0)
    v_max = min(1.0, (r_max + 1) / 180.0)

    empty_cols = np.where(~col_any)[0]
    if len(empty_cols) <= 5:
        h_span_deg = 360.0
        is_full_360 = True
        u_min, u_max = 0.0, 1.0
    else:
        doubled = np.concatenate([~col_any, ~col_any])
        max_gap = 0
        cur_gap = 0
        for is_empty in doubled:
            if is_empty:
                cur_gap += 1
                if cur_gap > max_gap:
                    max_gap = cur_gap
            else:
                cur_gap = 0
        max_gap = min(360, max_gap)
        h_span_deg = float(max(15.0, 360.0 - max_gap))
        is_full_360 = h_span_deg >= 340.0

        valid_cols = np.where(col_any)[0]
        u_min = max(0.0, float(valid_cols[0]) / 360.0)
        u_max = min(1.0, float(valid_cols[-1] + 1) / 360.0)

    return h_span_deg, v_span_deg, is_full_360, (u_min, u_max, v_min, v_max)


# =====================================================================
# 4. AUTOMATIC EXPOSURE & WHITE-BALANCE GAIN LEVELING
# =====================================================================

def _build_spherical_projection_grid(empties, grid_w=256, grid_h=128, k1=0.0, k2=0.0):
    n = len(empties)
    u = (np.arange(grid_w, dtype=np.float32) + 0.5) / grid_w
    v = (np.arange(grid_h, dtype=np.float32) + 0.5) / grid_h
    lon = (u - 0.5) * (2.0 * math.pi)
    lat = (v - 0.5) * math.pi

    cos_lat = np.cos(lat)[:, None]
    sin_lat = np.sin(lat)[:, None]
    cos_lon = np.cos(lon)[None, :]
    sin_lon = np.sin(lon)[None, :]

    dirs = np.stack([
        cos_lat * sin_lon,
        cos_lat * cos_lon,
        sin_lat * np.ones_like(cos_lon)
    ], axis=-1)

    inside_masks = np.zeros((n, grid_h, grid_w), dtype=bool)
    uv_coords = np.zeros((n, grid_h, grid_w, 2), dtype=np.float32)
    scores = np.full((n, grid_h, grid_w), -1.0, dtype=np.float32)

    for i, emp in enumerate(empties):
        R_world = emp.rotation_euler.to_matrix()
        R_inv = np.array(R_world.transposed(), dtype=np.float32)
        fax = float(emp.get("fov_scale_x", 1.0))
        fay = float(emp.get("fov_scale_y", 1.0))
        aspect = fax / max(fay, 1e-4)

        local_dirs = np.einsum('ij,hwj->hwi', R_inv, dirs)
        lz = local_dirs[..., 2]
        front = lz < -1e-4

        proj_x = np.where(front, local_dirs[..., 0] / (-lz * fax), 10.0)
        proj_y = np.where(front, local_dirs[..., 1] / (-lz * fay), 10.0)

        in_rect_frustum = front & (np.abs(proj_x) < 0.52) & (np.abs(proj_y) < 0.52)

        rx = np.clip(proj_x * 2.0, -1.5, 1.5)
        ry = np.clip((proj_y * 2.0) / aspect, -1.5, 1.5)
        r2 = rx * rx + ry * ry
        radial = np.maximum(0.4, 1.0 + k1 * r2 + k2 * r2 * r2)
        uv_x = proj_x * radial + 0.5
        uv_y = proj_y * radial + 0.5

        inside = in_rect_frustum & (uv_x >= 0.02) & (uv_x <= 0.98) & (uv_y >= 0.02) & (uv_y <= 0.98)
        dist_edge = np.maximum(np.abs(uv_x - 0.5), np.abs(uv_y - 0.5)) * 2.0

        inside_masks[i] = inside
        uv_coords[i, ..., 0] = np.clip(uv_x, 0.0, 1.0)
        uv_coords[i, ..., 1] = np.clip(uv_y, 0.0, 1.0)
        scores[i] = np.where(inside, 1.0 - dist_edge, -1.0)

    return inside_masks, uv_coords, scores


def _sample_blender_image_grid(emp, uv_grid):
    path = emp.get("image_path", "")
    if OPENCV_AVAILABLE and hasattr(cv2, 'imread'):
        im_bgr = cv2.imread(path)
        if im_bgr is not None:
            ih, iw = im_bgr.shape[:2]
            map_x = (uv_grid[..., 0] * (iw - 1)).astype(np.float32)
            map_y = ((1.0 - uv_grid[..., 1]) * (ih - 1)).astype(np.float32)
            warped_bgr = cv2.remap(im_bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            return (warped_bgr[..., ::-1].astype(np.float32)) / 255.0

    b_img = bpy.data.images.load(path, check_existing=True)
    iw, ih = b_img.size[0], b_img.size[1]
    px = np.array(b_img.pixels[:], dtype=np.float32).reshape((ih, iw, 4))
    ix = np.clip((uv_grid[..., 0] * (iw - 1)).astype(np.int32), 0, iw - 1)
    iy = np.clip((uv_grid[..., 1] * (ih - 1)).astype(np.int32), 0, ih - 1)
    return px[iy, ix, :3]


def solve_exposure_wb_gains(empties, strength=1.0, lock_white_balance=False, k1=0.0, k2=0.0):
    n = len(empties)
    if n < 2 or strength <= 0.0:
        return [(1.0, 1.0, 1.0) for _ in range(n)]

    inside_masks, uv_coords, _ = _build_spherical_projection_grid(empties, grid_w=180, grid_h=90, k1=k1, k2=k2)
    sampled_rgb = [_sample_blender_image_grid(emp, uv_coords[i]) for i, emp in enumerate(empties)]

    gains_rgb = np.ones((n, 3), dtype=np.float64)
    reg_weight = 0.15

    for ch in range(3):
        A = np.zeros((n, n), dtype=np.float64)
        b = np.zeros(n, dtype=np.float64)

        for i in range(n):
            A[i, i] += reg_weight
            b[i] += reg_weight * 1.0

            for j in range(i + 1, n):
                overlap = inside_masks[i] & inside_masks[j]
                count = int(np.sum(overlap))
                if count < 12:
                    continue

                val_i = sampled_rgb[i][overlap, ch]
                val_j = sampled_rgb[j][overlap, ch]

                valid = (val_i > 0.04) & (val_i < 0.94) & (val_j > 0.04) & (val_j < 0.94)
                if np.sum(valid) < 8:
                    continue

                mean_i = float(np.mean(val_i[valid]))
                mean_j = float(np.mean(val_j[valid]))
                w = float(np.sum(valid)) / 200.0

                A[i, i] += w * (mean_i ** 2)
                A[j, j] += w * (mean_j ** 2)
                A[i, j] -= w * mean_i * mean_j
                A[j, i] -= w * mean_i * mean_j

        try:
            g_ch = np.linalg.solve(A, b)
            g_ch = np.clip(g_ch, 0.45, 2.2)
            g_ch /= max(np.mean(g_ch), 1e-4)
            gains_rgb[:, ch] = g_ch
        except Exception as e:
            print(f"[Pano Stitcher] Exposure/WB solver warning on channel {ch}: {e}")

    if lock_white_balance:
        lum_gain = np.mean(gains_rgb, axis=1, keepdims=True)
        gains_rgb = np.repeat(lum_gain, 3, axis=1)

    gains_rgb = 1.0 + strength * (gains_rgb - 1.0)
    return [(float(g[0]), float(g[1]), float(g[2])) for g in gains_rgb]


def apply_gains_to_material(sphere, empties, gains):
    if not sphere or not sphere.data.materials:
        return
    nodes = sphere.data.materials[0].node_tree.nodes
    for i, emp in enumerate(empties):
        if i >= len(gains):
            break
        r_g, g_g, b_g = gains[i]
        emp["gain_r"] = r_g
        emp["gain_g"] = g_g
        emp["gain_b"] = b_g

        for prefix in ("GainHi_", "GainLow_"):
            g_node = nodes.get(f"{prefix}{i:02d}")
            if g_node:
                g_node.inputs[1].default_value = (r_g, g_g, b_g)


# =====================================================================
# 5. SHADER NODE GROUP (Ring-Free Projector + Normalized Blending)
# =====================================================================

def create_camera_projection_node_group():
    group_name = "Camera_Heal_Projector_v28"
    if group_name in bpy.data.node_groups:
        bpy.data.node_groups.remove(bpy.data.node_groups[group_name])

    ng = bpy.data.node_groups.new(group_name, 'ShaderNodeTree')

    sock_defs_in = [
        ("Vector", 'NodeSocketVector'),
        ("FOV_Scale_X", 'NodeSocketFloat'),
        ("FOV_Scale_Y", 'NodeSocketFloat'),
        ("Distortion_K1", 'NodeSocketFloat'),
        ("Distortion_K2", 'NodeSocketFloat'),
        ("Feather", 'NodeSocketFloat'),
    ]
    sock_defs_out = [
        ("UV", 'NodeSocketVector'),
        ("Soft_Mask", 'NodeSocketFloat'),
        ("Hard_Mask", 'NodeSocketFloat'),
    ]

    if hasattr(ng, 'interface'):
        for name, stype in sock_defs_in:
            ng.interface.new_socket(name, in_out='INPUT', socket_type=stype)
        for name, stype in sock_defs_out:
            ng.interface.new_socket(name, in_out='OUTPUT', socket_type=stype)
    else:
        for name, stype in sock_defs_in:
            ng.inputs.new(stype, name)
        for name, stype in sock_defs_out:
            ng.outputs.new(stype, name)

    nodes = ng.nodes
    links = ng.links

    n_in = nodes.new('NodeGroupInput')
    n_in.location = (-1200, 0)
    n_out = nodes.new('NodeGroupOutput')
    n_out.location = (1250, 0)

    sep = nodes.new('ShaderNodeSeparateXYZ')
    sep.location = (-950, 0)
    links.new(n_in.outputs["Vector"], sep.inputs[0])

    less_than = nodes.new('ShaderNodeMath')
    less_than.operation = 'LESS_THAN'
    less_than.inputs[1].default_value = -0.001
    less_than.location = (-450, -380)
    links.new(sep.outputs[2], less_than.inputs[0])

    neg_z = nodes.new('ShaderNodeMath')
    neg_z.operation = 'MULTIPLY'
    neg_z.inputs[1].default_value = -1.0
    neg_z.location = (-750, -50)
    links.new(sep.outputs[2], neg_z.inputs[0])

    div_x = nodes.new('ShaderNodeMath')
    div_x.operation = 'DIVIDE'
    div_x.location = (-550, 150)
    links.new(sep.outputs[0], div_x.inputs[0])
    links.new(neg_z.outputs[0], div_x.inputs[1])

    div_y = nodes.new('ShaderNodeMath')
    div_y.operation = 'DIVIDE'
    div_y.location = (-550, 0)
    links.new(sep.outputs[1], div_y.inputs[0])
    links.new(neg_z.outputs[0], div_y.inputs[1])

    scale_x = nodes.new('ShaderNodeMath')
    scale_x.operation = 'DIVIDE'
    scale_x.location = (-350, 150)
    links.new(div_x.outputs[0], scale_x.inputs[0])
    links.new(n_in.outputs["FOV_Scale_X"], scale_x.inputs[1])

    scale_y = nodes.new('ShaderNodeMath')
    scale_y.operation = 'DIVIDE'
    scale_y.location = (-350, 0)
    links.new(div_y.outputs[0], scale_y.inputs[0])
    links.new(n_in.outputs["FOV_Scale_Y"], scale_y.inputs[1])

    # Rectilinear Frustum Gate prevents negative K1/K2 from folding back into rings!
    rect_abs_x = nodes.new('ShaderNodeMath')
    rect_abs_x.operation = 'ABSOLUTE'
    rect_abs_x.location = (-150, -220)
    links.new(scale_x.outputs[0], rect_abs_x.inputs[0])

    rect_abs_y = nodes.new('ShaderNodeMath')
    rect_abs_y.operation = 'ABSOLUTE'
    rect_abs_y.location = (-150, -340)
    links.new(scale_y.outputs[0], rect_abs_y.inputs[0])

    rect_max = nodes.new('ShaderNodeMath')
    rect_max.operation = 'MAXIMUM'
    rect_max.location = (20, -280)
    links.new(rect_abs_x.outputs[0], rect_max.inputs[0])
    links.new(rect_abs_y.outputs[0], rect_max.inputs[1])

    frustum_gate = nodes.new('ShaderNodeMath')
    frustum_gate.operation = 'LESS_THAN'
    frustum_gate.inputs[1].default_value = 0.55
    frustum_gate.location = (200, -320)
    links.new(rect_max.outputs[0], frustum_gate.inputs[0])

    valid_ray = nodes.new('ShaderNodeMath')
    valid_ray.operation = 'MULTIPLY'
    valid_ray.location = (380, -350)
    links.new(less_than.outputs[0], valid_ray.inputs[0])
    links.new(frustum_gate.outputs[0], valid_ray.inputs[1])

    x2 = nodes.new('ShaderNodeMath')
    x2.operation = 'MULTIPLY'
    x2.location = (-100, 320)
    links.new(scale_x.outputs[0], x2.inputs[0])
    links.new(scale_x.outputs[0], x2.inputs[1])

    y2 = nodes.new('ShaderNodeMath')
    y2.operation = 'MULTIPLY'
    y2.location = (-100, 200)
    links.new(scale_y.outputs[0], y2.inputs[0])
    links.new(scale_y.outputs[0], y2.inputs[1])

    sum_xy2 = nodes.new('ShaderNodeMath')
    sum_xy2.operation = 'ADD'
    sum_xy2.location = (80, 260)
    links.new(x2.outputs[0], sum_xy2.inputs[0])
    links.new(y2.outputs[0], sum_xy2.inputs[1])

    r2_raw = nodes.new('ShaderNodeMath')
    r2_raw.operation = 'MULTIPLY'
    r2_raw.inputs[1].default_value = 4.0
    r2_raw.location = (220, 260)
    links.new(sum_xy2.outputs[0], r2_raw.inputs[0])

    r2 = nodes.new('ShaderNodeMath')
    r2.operation = 'MINIMUM'
    r2.inputs[1].default_value = 2.25
    r2.location = (360, 260)
    links.new(r2_raw.outputs[0], r2.inputs[0])

    r4 = nodes.new('ShaderNodeMath')
    r4.operation = 'MULTIPLY'
    r4.location = (500, 340)
    links.new(r2.outputs[0], r4.inputs[0])
    links.new(r2.outputs[0], r4.inputs[1])

    k1_term = nodes.new('ShaderNodeMath')
    k1_term.operation = 'MULTIPLY'
    k1_term.location = (500, 220)
    links.new(n_in.outputs["Distortion_K1"], k1_term.inputs[0])
    links.new(r2.outputs[0], k1_term.inputs[1])

    k2_term = nodes.new('ShaderNodeMath')
    k2_term.operation = 'MULTIPLY'
    k2_term.location = (650, 300)
    links.new(n_in.outputs["Distortion_K2"], k2_term.inputs[0])
    links.new(r4.outputs[0], k2_term.inputs[1])

    k_sum = nodes.new('ShaderNodeMath')
    k_sum.operation = 'ADD'
    k_sum.location = (790, 250)
    links.new(k1_term.outputs[0], k_sum.inputs[0])
    links.new(k2_term.outputs[0], k_sum.inputs[1])

    rad_factor = nodes.new('ShaderNodeMath')
    rad_factor.operation = 'ADD'
    rad_factor.inputs[1].default_value = 1.0
    rad_factor.location = (920, 250)
    links.new(k_sum.outputs[0], rad_factor.inputs[0])

    dist_x = nodes.new('ShaderNodeMath')
    dist_x.operation = 'MULTIPLY'
    dist_x.location = (400, 120)
    links.new(scale_x.outputs[0], dist_x.inputs[0])
    links.new(rad_factor.outputs[0], dist_x.inputs[1])

    dist_y = nodes.new('ShaderNodeMath')
    dist_y.operation = 'MULTIPLY'
    dist_y.location = (400, -20)
    links.new(scale_y.outputs[0], dist_y.inputs[0])
    links.new(rad_factor.outputs[0], dist_y.inputs[1])

    add_x = nodes.new('ShaderNodeMath')
    add_x.operation = 'ADD'
    add_x.inputs[1].default_value = 0.5
    add_x.location = (600, 120)
    links.new(dist_x.outputs[0], add_x.inputs[0])

    add_y = nodes.new('ShaderNodeMath')
    add_y.operation = 'ADD'
    add_y.inputs[1].default_value = 0.5
    add_y.location = (600, -20)
    links.new(dist_y.outputs[0], add_y.inputs[0])

    comb_uv = nodes.new('ShaderNodeCombineXYZ')
    comb_uv.location = (820, 80)
    links.new(add_x.outputs[0], comb_uv.inputs[0])
    links.new(add_y.outputs[0], comb_uv.inputs[1])
    links.new(comb_uv.outputs[0], n_out.inputs["UV"])

    abs_x = nodes.new('ShaderNodeMath')
    abs_x.operation = 'ABSOLUTE'
    abs_x.location = (550, -180)
    links.new(dist_x.outputs[0], abs_x.inputs[0])

    abs_y = nodes.new('ShaderNodeMath')
    abs_y.operation = 'ABSOLUTE'
    abs_y.location = (550, -300)
    links.new(dist_y.outputs[0], abs_y.inputs[0])

    max_dist = nodes.new('ShaderNodeMath')
    max_dist.operation = 'MAXIMUM'
    max_dist.location = (700, -230)
    links.new(abs_x.outputs[0], max_dist.inputs[0])
    links.new(abs_y.outputs[0], max_dist.inputs[1])

    edge_dist = nodes.new('ShaderNodeMath')
    edge_dist.operation = 'MULTIPLY'
    edge_dist.inputs[1].default_value = 2.0
    edge_dist.location = (840, -230)
    links.new(max_dist.outputs[0], edge_dist.inputs[0])

    sub_feather = nodes.new('ShaderNodeMath')
    sub_feather.operation = 'SUBTRACT'
    sub_feather.inputs[0].default_value = 0.98
    sub_feather.location = (680, -420)
    links.new(n_in.outputs["Feather"], sub_feather.inputs[1])

    map_soft = nodes.new('ShaderNodeMapRange')
    map_soft.interpolation_type = 'SMOOTHSTEP'
    map_soft.clamp = True
    map_soft.inputs[2].default_value = 0.98
    map_soft.inputs[3].default_value = 1.0
    map_soft.inputs[4].default_value = 0.0
    map_soft.location = (880, -400)
    links.new(edge_dist.outputs[0], map_soft.inputs[0])
    links.new(sub_feather.outputs[0], map_soft.inputs[1])

    final_soft = nodes.new('ShaderNodeMath')
    final_soft.operation = 'MULTIPLY'
    final_soft.location = (1080, -260)
    links.new(map_soft.outputs[0], final_soft.inputs[0])
    links.new(valid_ray.outputs[0], final_soft.inputs[1])
    links.new(final_soft.outputs[0], n_out.inputs["Soft_Mask"])

    map_hard = nodes.new('ShaderNodeMapRange')
    map_hard.interpolation_type = 'SMOOTHSTEP'
    map_hard.clamp = True
    map_hard.inputs[1].default_value = 0.75
    map_hard.inputs[2].default_value = 0.98
    map_hard.inputs[3].default_value = 1.0
    map_hard.inputs[4].default_value = 0.0
    map_hard.location = (880, -620)
    links.new(edge_dist.outputs[0], map_hard.inputs[0])

    final_hard = nodes.new('ShaderNodeMath')
    final_hard.operation = 'MULTIPLY'
    final_hard.location = (1080, -520)
    links.new(map_hard.outputs[0], final_hard.inputs[0])
    links.new(valid_ray.outputs[0], final_hard.inputs[1])
    links.new(final_hard.outputs[0], n_out.inputs["Hard_Mask"])

    return ng


# =====================================================================
# 6. ADD-ON PROPERTIES, OPERATORS & MENUS
# =====================================================================

class PANO_PG_settings_v28(PropertyGroup):
    use_exif: BoolProperty(
        name="Read EXIF Focal Length & Orientation",
        description="Automatically parse JPEG EXIF metadata for lens focal length and orientation",
        default=True
    )
    orientation_mode: EnumProperty(
        name="Orientation",
        items=[
            ('AUTO', "Auto-Detect (Dimensions & EXIF)", "Automatically detect portrait vs landscape per image"),
            ('LANDSCAPE', "Force Landscape", "Treat all images as horizontal landscape"),
            ('PORTRAIT', "Force Portrait", "Treat all images as vertical portrait"),
        ],
        default='AUTO'
    )
    pano_span_mode: EnumProperty(
        name="Panorama Span",
        description="Auto-detect whether your shots cover 90°, 180°, or a full 360° x 180° sphere",
        items=[
            ('AUTO', "Auto-Detect Span (0°-360° x 0°-180°)", "Automatically calculate angular coverage from lens FOV & feature tracks"),
            ('PARTIAL', "Partial Panorama (< 360°)", "Do not force 360° loop closure"),
            ('FULL_360', "Full 360° Horizon Loop", "Force 360° loop closure between last and first photo"),
        ],
        default='AUTO'
    )
    detected_span_info: StringProperty(
        name="Detected Span",
        default="Not aligned yet"
    )
    sensor_width: FloatProperty(
        name="Sensor Long Edge (mm)",
        default=36.0,
        min=4.0,
        max=100.0
    )
    sensor_height: FloatProperty(
        name="Sensor Short Edge (mm)",
        default=24.0,
        min=3.0,
        max=100.0
    )
    fallback_focal_mm: FloatProperty(
        name="Fallback Lens (mm)",
        default=28.0,
        min=6.0,
        max=200.0
    )
    distro_k1: FloatProperty(
        name="Lens Distortion K1",
        default=0.0,
        min=-0.25,
        max=0.25
    )
    distro_k2: FloatProperty(
        name="Lens Distortion K2",
        default=0.0,
        min=-0.15,
        max=0.15
    )
    auto_estimate_distortion: BoolProperty(
        name="Auto-Estimate Distortion (Safe Bounds)",
        default=False
    )
    auto_exposure_wb: BoolProperty(
        name="Auto Exposure & White-Balance Leveling",
        default=True
    )
    exposure_wb_strength: FloatProperty(
        name="Exposure/WB Leveling Strength",
        default=1.0,
        min=0.0,
        max=1.5
    )
    lock_white_balance: BoolProperty(
        name="Exposure Only (Lock Color Balance)",
        default=False
    )
    feather_softness: FloatProperty(
        name="Low-Freq Feather",
        default=0.45,
        min=0.05,
        max=0.95
    )
    detail_sharpness: FloatProperty(
        name="Heal Detail Retention",
        default=0.85,
        min=0.0,
        max=2.0
    )
    align_solver: EnumProperty(
        name="Tracker Engine",
        items=[
            ('HYBRID', "Hybrid (OpenCV + Blender ROI Flow)", "Uses OpenCV wide-angle matcher + Blender KLT optical flow"),
            ('BLENDER_ROI', "Blender Optical Flow + ROI Pre-Shift", "Uses Blender's native KLT tracker with ROI pre-shift"),
            ('OPENCV_FEATURES', "OpenCV Feature Matcher (AKAZE/ORB/SIFT)", "Uses OpenCV multi-scale descriptors"),
        ],
        default='HYBRID'
    )
    optical_flow_search_px: IntProperty(
        name="ROI Tracker Search Box (px)",
        default=210,
        min=64,
        max=250
    )
    roi_overlap_ratio: FloatProperty(
        name="Expected Image Overlap",
        default=0.45,
        min=0.15,
        max=0.85
    )
    crop_bake_to_detected_span: BoolProperty(
        name="Auto-Crop Bake to Detected Span",
        default=True
    )
    bake_res_x: IntProperty(
        name="Bake Width",
        default=4096,
        min=512,
        max=16384
    )


class PANO_OT_setup_rig_v28(Operator, ImportHelper):
    bl_idname = "pano.setup_rig"
    bl_label = "1. Load Images & Auto-Space Panorama Rig"
    bl_options = {'REGISTER', 'UNDO'}

    files: CollectionProperty(
        name="File Path",
        type=bpy.types.OperatorFileListElement,
    )

    def execute(self, context):
        settings = context.scene.pano_settings
        if not self.files:
            self.report({'WARNING'}, "No images selected!")
            return {'CANCELLED'}

        folder_dir = os.path.dirname(self.filepath)

        old_sphere = bpy.data.objects.get("Pano_Canvas_Sphere")
        if old_sphere:
            bpy.data.objects.remove(old_sphere, do_unlink=True)
        old_col = bpy.data.collections.get("Pano_Projectors")
        if old_col:
            for obj in list(old_col.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.collections.remove(old_col)

        bpy.ops.mesh.primitive_uv_sphere_add(segments=128, ring_count=64, radius=10.0, location=(0, 0, 0))
        sphere = context.active_object
        sphere.name = "Pano_Canvas_Sphere"
        bpy.ops.object.shade_smooth()

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.flip_normals()
        bpy.ops.object.mode_set(mode='OBJECT')

        proj_col = bpy.data.collections.new("Pano_Projectors")
        context.scene.collection.children.link(proj_col)

        mat = bpy.data.materials.new(name="Pano_Heal_Material")
        mat.use_nodes = True
        sphere.data.materials.append(mat)
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        proj_group = create_camera_projection_node_group()

        out_node = nodes.new('ShaderNodeOutputMaterial')
        out_node.location = (1800, 0)
        emit_node = nodes.new('ShaderNodeEmission')
        emit_node.location = (1600, 0)
        links.new(emit_node.outputs[0], out_node.inputs[0])

        sorted_files = sorted(self.files, key=lambda f: f.name)
        num_imgs = len(sorted_files)
        exif_hits = 0
        portrait_count = 0

        img_meta = []
        for file_elem in sorted_files:
            filepath = os.path.join(folder_dir, file_elem.name)
            img = bpy.data.images.load(filepath, check_existing=True)
            img_w, img_h = img.size[0] or 1920, img.size[1] or 1080

            focal_mm, sensor_w, sensor_h, exif_orient = (
                None, settings.sensor_width, settings.sensor_height, 1
            )
            if settings.use_exif:
                focal_mm, sensor_w, sensor_h, exif_orient = read_exif_focal_and_orientation(
                    filepath, settings.sensor_width, settings.sensor_height
                )
            if focal_mm:
                exif_hits += 1
            else:
                focal_mm = settings.fallback_focal_mm

            is_portrait, fov_scale_x, fov_scale_y = compute_orientation_fov_scales(
                img_w, img_h, focal_mm, sensor_w, sensor_h, mode=settings.orientation_mode
            )
            if is_portrait:
                portrait_count += 1
            hfov_rad = 2.0 * math.atan(fov_scale_x * 0.5)
            img_meta.append((file_elem, filepath, img, focal_mm, is_portrait, fov_scale_x, fov_scale_y, hfov_rad))

        avg_hfov = sum(m[7] for m in img_meta) / max(num_imgs, 1)
        natural_step = avg_hfov * (1.0 - settings.roi_overlap_ratio)
        estimated_total_span = natural_step * max(1, num_imgs - 1) + avg_hfov

        if settings.pano_span_mode == 'FULL_360' or estimated_total_span >= math.radians(330.0):
            step_yaw = (2.0 * math.pi) / num_imgs
            start_yaw = 0.0
            span_deg = 360.0
        else:
            step_yaw = natural_step
            start_yaw = -0.5 * step_yaw * (num_imgs - 1)
            span_deg = min(360.0, math.degrees(estimated_total_span))

        v_span_deg = math.degrees(2.0 * math.atan(img_meta[0][6] * 0.5))
        settings.detected_span_info = f"Est: {span_deg:.0f}° H x {v_span_deg:.0f}° V"

        # Use pure ShaderNodeVectorMath for Blender 3.6 + 4.x compatibility
        sum_low_color = None
        sum_low_weight = None
        sum_high_detail = None
        sum_high_weight = None

        for i, (file_elem, filepath, img, focal_mm, is_portrait, fov_scale_x, fov_scale_y, _) in enumerate(img_meta):
            yaw = -(start_yaw + i * step_yaw)
            pitch = math.radians(90.0)

            orient_tag = "P" if is_portrait else "L"
            empty = bpy.data.objects.new(f"Projector_{i:02d}_{orient_tag}_{file_elem.name}", None)
            empty.empty_display_type = 'SINGLE_ARROW'
            empty.empty_display_size = 2.0
            empty.rotation_euler = (pitch, 0.0, yaw)

            empty["image_path"] = filepath
            empty["focal_mm"] = float(focal_mm)
            empty["is_portrait"] = bool(is_portrait)
            empty["fov_scale_x"] = float(fov_scale_x)
            empty["fov_scale_y"] = float(fov_scale_y)
            empty["proj_index"] = i
            empty["gain_r"] = 1.0
            empty["gain_g"] = 1.0
            empty["gain_b"] = 1.0
            proj_col.objects.link(empty)

            y_offset = -i * 750

            tex_coord = nodes.new('ShaderNodeTexCoord')
            tex_coord.object = empty
            tex_coord.location = (-1200, y_offset)

            pg = nodes.new('ShaderNodeGroup')
            pg.name = f"ProjGroup_{i:02d}"
            pg.node_tree = proj_group
            pg.location = (-920, y_offset)
            pg.inputs["FOV_Scale_X"].default_value = fov_scale_x
            pg.inputs["FOV_Scale_Y"].default_value = fov_scale_y
            pg.inputs["Distortion_K1"].default_value = settings.distro_k1
            pg.inputs["Distortion_K2"].default_value = settings.distro_k2
            pg.inputs["Feather"].default_value = settings.feather_softness
            links.new(tex_coord.outputs["Object"], pg.inputs["Vector"])

            tex_hi = nodes.new('ShaderNodeTexImage')
            tex_hi.image = img
            tex_hi.extension = 'CLIP'
            tex_hi.location = (-580, y_offset + 160)
            links.new(pg.outputs["UV"], tex_hi.inputs["Vector"])

            # Blur UV via VectorMath for low-frequency base (compatible with Blender 3.x & 4.x)
            noise = nodes.new('ShaderNodeTexNoise')
            noise.inputs['Scale'].default_value = 500.0
            noise.location = (-920, y_offset - 260)

            noise_sub = nodes.new('ShaderNodeVectorMath')
            noise_sub.operation = 'SUBTRACT'
            noise_sub.inputs[1].default_value = (0.5, 0.5, 0.5)
            noise_sub.location = (-760, y_offset - 260)
            links.new(noise.outputs["Color"], noise_sub.inputs[0])

            noise_scale = nodes.new('ShaderNodeVectorMath')
            noise_scale.operation = 'SCALE'
            noise_scale.inputs["Scale"].default_value = 0.009
            noise_scale.location = (-600, y_offset - 260)
            links.new(noise_sub.outputs["Vector"], noise_scale.inputs[0])

            blur_uv = nodes.new('ShaderNodeVectorMath')
            blur_uv.operation = 'ADD'
            blur_uv.location = (-720, y_offset - 140)
            links.new(pg.outputs["UV"], blur_uv.inputs[0])
            links.new(noise_scale.outputs["Vector"], blur_uv.inputs[1])

            tex_low = nodes.new('ShaderNodeTexImage')
            tex_low.image = img
            tex_low.extension = 'CLIP'
            tex_low.location = (-580, y_offset - 160)
            links.new(blur_uv.outputs["Vector"], tex_low.inputs["Vector"])

            # Exposure/WB Gain Nodes using VectorMath MULTIPLY
            gain_hi = nodes.new('ShaderNodeVectorMath')
            gain_hi.name = f"GainHi_{i:02d}"
            gain_hi.operation = 'MULTIPLY'
            gain_hi.inputs[1].default_value = (1.0, 1.0, 1.0)
            gain_hi.location = (-300, y_offset + 160)
            links.new(tex_hi.outputs["Color"], gain_hi.inputs[0])

            gain_low = nodes.new('ShaderNodeVectorMath')
            gain_low.name = f"GainLow_{i:02d}"
            gain_low.operation = 'MULTIPLY'
            gain_low.inputs[1].default_value = (1.0, 1.0, 1.0)
            gain_low.location = (-300, y_offset - 160)
            links.new(tex_low.outputs["Color"], gain_low.inputs[0])

            # High-Frequency Detail = (GainHi - GainLow) * DetailSharpness
            sub_detail = nodes.new('ShaderNodeVectorMath')
            sub_detail.operation = 'SUBTRACT'
            sub_detail.location = (-100, y_offset + 160)
            links.new(gain_hi.outputs["Vector"], sub_detail.inputs[0])
            links.new(gain_low.outputs["Vector"], sub_detail.inputs[1])

            scale_detail = nodes.new('ShaderNodeVectorMath')
            scale_detail.operation = 'SCALE'
            scale_detail.inputs["Scale"].default_value = settings.detail_sharpness
            scale_detail.location = (80, y_offset + 160)
            links.new(sub_detail.outputs["Vector"], scale_detail.inputs[0])

            w_low_col = nodes.new('ShaderNodeVectorMath')
            w_low_col.operation = 'SCALE'
            w_low_col.location = (280, y_offset - 120)
            links.new(gain_low.outputs["Vector"], w_low_col.inputs[0])
            links.new(pg.outputs["Soft_Mask"], w_low_col.inputs["Scale"])

            w_high_col = nodes.new('ShaderNodeVectorMath')
            w_high_col.operation = 'SCALE'
            w_high_col.location = (280, y_offset + 160)
            links.new(scale_detail.outputs["Vector"], w_high_col.inputs[0])
            links.new(pg.outputs["Hard_Mask"], w_high_col.inputs["Scale"])

            if sum_low_color is None:
                sum_low_color = w_low_col.outputs["Vector"]
                sum_low_weight = pg.outputs["Soft_Mask"]
                sum_high_detail = w_high_col.outputs["Vector"]
                sum_high_weight = pg.outputs["Hard_Mask"]
            else:
                add_lc = nodes.new('ShaderNodeVectorMath')
                add_lc.operation = 'ADD'
                add_lc.location = (520, y_offset - 120)
                links.new(sum_low_color, add_lc.inputs[0])
                links.new(w_low_col.outputs["Vector"], add_lc.inputs[1])
                sum_low_color = add_lc.outputs["Vector"]

                add_lw = nodes.new('ShaderNodeMath')
                add_lw.operation = 'ADD'
                add_lw.location = (520, y_offset - 260)
                links.new(sum_low_weight, add_lw.inputs[0])
                links.new(pg.outputs["Soft_Mask"], add_lw.inputs[1])
                sum_low_weight = add_lw.outputs[0]

                add_hc = nodes.new('ShaderNodeVectorMath')
                add_hc.operation = 'ADD'
                add_hc.location = (520, y_offset + 160)
                links.new(sum_high_detail, add_hc.inputs[0])
                links.new(w_high_col.outputs["Vector"], add_hc.inputs[1])
                sum_high_detail = add_hc.outputs["Vector"]

                add_hw = nodes.new('ShaderNodeMath')
                add_hw.operation = 'ADD'
                add_hw.location = (520, y_offset + 20)
                links.new(sum_high_weight, add_hw.inputs[0])
                links.new(pg.outputs["Hard_Mask"], add_hw.inputs[1])
                sum_high_weight = add_hw.outputs[0]

        if sum_low_color and sum_high_detail:
            safe_lw = nodes.new('ShaderNodeMath')
            safe_lw.operation = 'MAXIMUM'
            safe_lw.inputs[1].default_value = 0.0001
            safe_lw.location = (900, -180)
            links.new(sum_low_weight, safe_lw.inputs[0])

            inv_lw = nodes.new('ShaderNodeMath')
            inv_lw.operation = 'DIVIDE'
            inv_lw.inputs[0].default_value = 1.0
            inv_lw.location = (1060, -180)
            links.new(safe_lw.outputs[0], inv_lw.inputs[1])

            norm_low = nodes.new('ShaderNodeVectorMath')
            norm_low.operation = 'SCALE'
            norm_low.location = (1220, -120)
            links.new(sum_low_color, norm_low.inputs[0])
            links.new(inv_lw.outputs[0], norm_low.inputs["Scale"])

            safe_hw = nodes.new('ShaderNodeMath')
            safe_hw.operation = 'MAXIMUM'
            safe_hw.inputs[1].default_value = 0.0001
            safe_hw.location = (900, 120)
            links.new(sum_high_weight, safe_hw.inputs[0])

            inv_hw = nodes.new('ShaderNodeMath')
            inv_hw.operation = 'DIVIDE'
            inv_hw.inputs[0].default_value = 1.0
            inv_hw.location = (1060, 120)
            links.new(safe_hw.outputs[0], inv_hw.inputs[1])

            norm_high = nodes.new('ShaderNodeVectorMath')
            norm_high.operation = 'SCALE'
            norm_high.location = (1220, 160)
            links.new(sum_high_detail, norm_high.inputs[0])
            links.new(inv_hw.outputs[0], norm_high.inputs["Scale"])

            recombine = nodes.new('ShaderNodeVectorMath')
            recombine.operation = 'ADD'
            recombine.location = (1420, 0)
            links.new(norm_low.outputs["Vector"], recombine.inputs[0])
            links.new(norm_high.outputs["Vector"], recombine.inputs[1])
            links.new(recombine.outputs["Vector"], emit_node.inputs[0])

        landscape_count = num_imgs - portrait_count
        self.report(
            {'INFO'},
            f"Loaded {num_imgs} images ({settings.detected_span_info} | {portrait_count} Portrait, {landscape_count} Landscape)."
        )
        return {'FINISHED'}


class PANO_OT_auto_align_optical_flow_v28(Operator):
    bl_idname = "pano.auto_align_optical_flow"
    bl_label = "2. Auto-Align & Auto-Detect Span (0-360°)"
    bl_description = "Aligns projectors, auto-detects exact angular span (0-360° x 0-180°), and levels Exposure/WB"
    bl_options = {'REGISTER', 'UNDO'}

    def _get_pair_matches(self, context, settings, path_a, path_b):
        matches = []
        if settings.align_solver in {'HYBRID', 'OPENCV_FEATURES'} and OPENCV_AVAILABLE:
            matches = match_pair_opencv(path_a, path_b, roi_fraction=settings.roi_overlap_ratio)
        if len(matches) < 8 and settings.align_solver in {'HYBRID', 'BLENDER_ROI'}:
            flow_matches = track_pair_with_blender_clip_roi(
                context,
                path_a,
                path_b,
                search_size=settings.optical_flow_search_px,
                roi_overlap_ratio=settings.roi_overlap_ratio
            )
            matches.extend(flow_matches)
        return matches

    def execute(self, context):
        proj_col = bpy.data.collections.get("Pano_Projectors")
        if not proj_col or len(proj_col.objects) < 2:
            self.report({'ERROR'}, "Need at least 2 projectors in 'Pano_Projectors' collection!")
            return {'CANCELLED'}

        settings = context.scene.pano_settings
        empties = sorted(proj_col.objects, key=lambda o: o.get("proj_index", 0))
        n = len(empties)

        raw_pair_matches = {}
        all_inlier_sets = []

        for i in range(n - 1):
            emp_a, emp_b = empties[i], empties[i + 1]
            m = self._get_pair_matches(context, settings, emp_a["image_path"], emp_b["image_path"])
            raw_pair_matches[(i, i + 1)] = m
            _, count, inliers = solve_relative_rotation_svd(
                m, emp_a["fov_scale_x"], emp_a["fov_scale_y"], emp_b["fov_scale_x"], emp_b["fov_scale_y"],
                settings.distro_k1, settings.distro_k2
            )
            if count >= 6:
                all_inlier_sets.append((inliers, emp_a["fov_scale_x"], emp_a["fov_scale_y"], emp_b["fov_scale_x"], emp_b["fov_scale_y"]))

        if settings.auto_estimate_distortion and all_inlier_sets:
            est_k1, est_k2 = estimate_radial_distortion_k1_k2(all_inlier_sets)
            settings.distro_k1 = est_k1
            settings.distro_k2 = est_k2

        base_rot = Euler((math.radians(90.0), 0.0, 0.0), 'XYZ').to_matrix()
        world_rots = [None] * n
        world_rots[0] = base_rot
        aligned_pairs = 0

        for i in range(n - 1):
            emp_a, emp_b = empties[i], empties[i + 1]
            m = raw_pair_matches.get((i, i + 1), [])
            R_rel, count, _ = solve_relative_rotation_svd(
                m, emp_a["fov_scale_x"], emp_a["fov_scale_y"], emp_b["fov_scale_x"], emp_b["fov_scale_y"],
                settings.distro_k1, settings.distro_k2
            )
            if R_rel is not None and count >= 4:
                world_rots[i + 1] = world_rots[i] @ R_rel
                aligned_pairs += 1

        for target_idx in range(1, n):
            if world_rots[target_idx] is not None:
                continue
            best_R_world = None
            best_count = 0
            emp_b = empties[target_idx]
            for ref_idx in range(n):
                if ref_idx == target_idx or world_rots[ref_idx] is None:
                    continue
                emp_a = empties[ref_idx]
                m = self._get_pair_matches(context, settings, emp_a["image_path"], emp_b["image_path"])
                R_rel, count, _ = solve_relative_rotation_svd(
                    m, emp_a["fov_scale_x"], emp_a["fov_scale_y"], emp_b["fov_scale_x"], emp_b["fov_scale_y"],
                    settings.distro_k1, settings.distro_k2
                )
                if R_rel is not None and count > best_count:
                    best_count = count
                    best_R_world = world_rots[ref_idx] @ R_rel
            if best_R_world is not None and best_count >= 4:
                world_rots[target_idx] = best_R_world
                aligned_pairs += 1
            else:
                hfov = 2.0 * math.atan(empties[target_idx]["fov_scale_x"] * 0.5)
                step_angle = -hfov * (1.0 - settings.roi_overlap_ratio)
                world_rots[target_idx] = world_rots[target_idx - 1] @ Euler((0.0, step_angle, 0.0), 'XYZ').to_matrix()

        for i, emp in enumerate(empties):
            emp.rotation_euler = world_rots[i].to_euler('XYZ')

        h_span_deg, v_span_deg, is_full_360, bbox_uv = detect_panorama_angular_span(empties)

        should_close_loop = (
            settings.pano_span_mode == 'FULL_360' or
            (settings.pano_span_mode == 'AUTO' and is_full_360)
        )

        if should_close_loop and n >= 3:
            matches_loop = self._get_pair_matches(
                context, settings, empties[-1]["image_path"], empties[0]["image_path"]
            )
            R_loop, count_loop, _ = solve_relative_rotation_svd(
                matches_loop,
                empties[-1]["fov_scale_x"], empties[-1]["fov_scale_y"],
                empties[0]["fov_scale_x"], empties[0]["fov_scale_y"],
                settings.distro_k1, settings.distro_k2
            )
            if R_loop is not None and count_loop >= 6:
                R_end_pred = world_rots[-1] @ R_loop
                R_err = R_end_pred.transposed() @ world_rots[0]
                q_err = R_err.to_quaternion()
                q_ident = Matrix.Identity(3).to_quaternion()

                for i in range(1, n):
                    frac = i / n
                    q_corr = q_ident.slerp(q_err, frac)
                    world_rots[i] = world_rots[i] @ q_corr.to_matrix()
                for i, emp in enumerate(empties):
                    emp.rotation_euler = world_rots[i].to_euler('XYZ')
                h_span_deg, v_span_deg, is_full_360, bbox_uv = detect_panorama_angular_span(empties)

        sphere = bpy.data.objects.get("Pano_Canvas_Sphere")
        if sphere:
            sphere["h_span_deg"] = float(h_span_deg)
            sphere["v_span_deg"] = float(v_span_deg)
            sphere["is_full_360"] = bool(is_full_360)
            sphere["crop_uv"] = [float(x) for x in bbox_uv]

        mode_str = "Full 360°" if is_full_360 else "Partial"
        settings.detected_span_info = f"{h_span_deg:.1f}° H x {v_span_deg:.1f}° V ({mode_str})"

        mat_nodes = sphere.data.materials[0].node_tree.nodes if (sphere and sphere.data.materials) else None
        for i, emp in enumerate(empties):
            if mat_nodes:
                pg = mat_nodes.get(f"ProjGroup_{i:02d}")
                if pg:
                    pg.inputs["Distortion_K1"].default_value = settings.distro_k1
                    pg.inputs["Distortion_K2"].default_value = settings.distro_k2
                    pg.inputs["Feather"].default_value = settings.feather_softness

        if settings.auto_exposure_wb and sphere:
            gains = solve_exposure_wb_gains(
                empties,
                strength=settings.exposure_wb_strength,
                lock_white_balance=settings.lock_white_balance,
                k1=settings.distro_k1,
                k2=settings.distro_k2
            )
            apply_gains_to_material(sphere, empties, gains)

        self.report(
            {'INFO'},
            f"Aligned {aligned_pairs} pairs! Auto-Detected Span: {settings.detected_span_info}"
        )
        return {'FINISHED'}


class PANO_OT_level_exposure_wb_v28(Operator):
    bl_idname = "pano.level_exposure_wb"
    bl_label = "Re-Level Exposure & White Balance"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sphere = bpy.data.objects.get("Pano_Canvas_Sphere")
        proj_col = bpy.data.collections.get("Pano_Projectors")
        if not sphere or not proj_col or len(proj_col.objects) < 2:
            self.report({'ERROR'}, "Setup the panorama sphere and projectors first!")
            return {'CANCELLED'}

        settings = context.scene.pano_settings
        empties = sorted(proj_col.objects, key=lambda o: o.get("proj_index", 0))
        gains = solve_exposure_wb_gains(
            empties,
            strength=settings.exposure_wb_strength if settings.auto_exposure_wb else 0.0,
            lock_white_balance=settings.lock_white_balance,
            k1=settings.distro_k1,
            k2=settings.distro_k2
        )
        apply_gains_to_material(sphere, empties, gains)
        self.report({'INFO'}, "Updated Exposure & White-Balance gains across all projectors!")
        return {'FINISHED'}


class PANO_OT_update_distortion_live_v28(Operator):
    bl_idname = "pano.update_distortion_live"
    bl_label = "Apply K1/K2 & Feather to Shader"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sphere = bpy.data.objects.get("Pano_Canvas_Sphere")
        if not sphere or not sphere.data.materials:
            return {'CANCELLED'}
        settings = context.scene.pano_settings
        for node in sphere.data.materials[0].node_tree.nodes:
            if node.type == 'GROUP' and node.node_tree and "Distortion_K1" in node.inputs:
                node.inputs["Distortion_K1"].default_value = settings.distro_k1
                node.inputs["Distortion_K2"].default_value = settings.distro_k2
                node.inputs["Feather"].default_value = settings.feather_softness
        return {'FINISHED'}


class PANO_OT_bake_equirectangular_v28(Operator):
    bl_idname = "pano.bake_equirectangular"
    bl_label = "3. Bake Panorama (Full or Auto-Cropped Span)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sphere = bpy.data.objects.get("Pano_Canvas_Sphere")
        if not sphere or not sphere.data.materials:
            self.report({'ERROR'}, "Pano_Canvas_Sphere not found!")
            return {'CANCELLED'}

        settings = context.scene.pano_settings
        mat = sphere.data.materials[0]
        nodes = mat.node_tree.nodes

        width = settings.bake_res_x
        height = width // 2
        if "Stitched_Panorama" in bpy.data.images:
            bpy.data.images.remove(bpy.data.images["Stitched_Panorama"])
        bake_img = bpy.data.images.new("Stitched_Panorama", width=width, height=height, float_buffer=True)

        bake_node = nodes.new('ShaderNodeTexImage')
        bake_node.image = bake_img
        bake_node.location = (1600, -350)

        for n in nodes:
            n.select = False
        bake_node.select = True
        nodes.active = bake_node

        orig_engine = context.scene.render.engine
        context.scene.render.engine = 'CYCLES'
        context.scene.cycles.bake_type = 'EMIT'
        context.scene.cycles.samples = 32

        bpy.ops.object.select_all(action='DESELECT')
        sphere.select_set(True)
        context.view_layer.objects.active = sphere

        bpy.ops.object.bake(type='EMIT')
        context.scene.render.engine = orig_engine

        proj_col = bpy.data.collections.get("Pano_Projectors")
        if settings.crop_bake_to_detected_span and proj_col and len(proj_col.objects) > 0:
            empties = sorted(proj_col.objects, key=lambda o: o.get("proj_index", 0))
            h_deg, v_deg, is_full_360, (u_min, u_max, v_min, v_max) = detect_panorama_angular_span(empties)
            if not is_full_360 or v_deg < 165.0:
                px = np.array(bake_img.pixels[:], dtype=np.float32).reshape((height, width, 4))
                x0 = max(0, int(u_min * width))
                x1 = min(width, int(u_max * width))
                y0 = max(0, int(v_min * height))
                y1 = min(height, int(v_max * height))
                if (x1 - x0) > 64 and (y1 - y0) > 64:
                    cropped = px[y0:y1, x0:x1, :]
                    ch, cw = cropped.shape[:2]
                    crop_name = "Stitched_Panorama_Cropped"
                    if crop_name in bpy.data.images:
                        bpy.data.images.remove(bpy.data.images[crop_name])
                    crop_img = bpy.data.images.new(crop_name, width=cw, height=ch, float_buffer=True)
                    crop_img.pixels = cropped.ravel()
                    crop_img.pack()
                    self.report(
                        {'INFO'},
                        f"Baked 360x180 + Auto-Cropped ({h_deg:.0f}°x{v_deg:.0f}° -> {cw}x{ch}px) to '{crop_name}'!"
                    )
                    return {'FINISHED'}

        self.report({'INFO'}, f"Full Panorama ({width}x{height}) baked to 'Stitched_Panorama'!")
        return {'FINISHED'}


def draw_pano_ui(layout, settings):
    box1 = layout.box()
    box1.label(text="1. EXIF, Span Mode & Rig:", icon='CAMERA_DATA')
    box1.prop(settings, "use_exif")
    box1.prop(settings, "orientation_mode")
    box1.prop(settings, "pano_span_mode")
    row = box1.row(align=True)
    row.prop(settings, "sensor_width", text="Sensor W")
    row.prop(settings, "sensor_height", text="Sensor H")
    box1.prop(settings, "fallback_focal_mm")
    box1.prop(settings, "roi_overlap_ratio")
    box1.prop(settings, "feather_softness")
    box1.prop(settings, "detail_sharpness")
    box1.operator("pano.setup_rig", icon='FILE_FOLDER')

    box2 = layout.box()
    box2.label(text="2. Auto-Align & Span Detection:", icon='TRACKER')
    box2.label(text=f"Span: {settings.detected_span_info}", icon='SPHERE')
    box2.prop(settings, "align_solver")
    box2.prop(settings, "optical_flow_search_px")
    box2.separator()
    box2.prop(settings, "auto_estimate_distortion")
    row_k = box2.row(align=True)
    row_k.prop(settings, "distro_k1", text="K1")
    row_k.prop(settings, "distro_k2", text="K2")
    box2.operator("pano.update_distortion_live", icon='FILE_REFRESH')
    box2.separator()
    box2.prop(settings, "auto_exposure_wb")
    box2.prop(settings, "exposure_wb_strength")
    box2.prop(settings, "lock_white_balance")
    box2.operator("pano.level_exposure_wb", icon='COLOR')
    box2.separator()
    box2.operator("pano.auto_align_optical_flow", icon='AUTO')

    box3 = layout.box()
    box3.label(text="3. Bake Panorama:", icon='RENDER_STILL')
    box3.prop(settings, "crop_bake_to_detected_span")
    box3.prop(settings, "bake_res_x")
    box3.operator("pano.bake_equirectangular", icon='IMAGE_DATA')


class VIEW3D_PT_pano_heal_stitcher_v28(Panel):
    bl_label = "Pano Heal Stitcher v2.8"
    bl_idname = "VIEW3D_PT_pano_heal_stitcher_v28"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Pano Stitcher'

    def draw(self, context):
        settings = getattr(context.scene, "pano_settings", None)
        if not settings:
            self.layout.label(text="Reloading settings...", icon='ERROR')
            return
        draw_pano_ui(self.layout, settings)


class VIEW3D_MT_pano_heal_menu_v28(Menu):
    bl_label = "Pano Stitcher"
    bl_idname = "VIEW3D_MT_pano_heal_menu_v28"

    def draw(self, context):
        layout = self.layout
        layout.operator("pano.setup_rig", icon='FILE_FOLDER')
        layout.operator("pano.auto_align_optical_flow", icon='AUTO')
        layout.operator("pano.level_exposure_wb", icon='COLOR')
        layout.operator("pano.update_distortion_live", icon='FILE_REFRESH')
        layout.separator()
        layout.operator("pano.bake_equirectangular", icon='RENDER_STILL')


def draw_header_menu(self, context):
    self.layout.menu("VIEW3D_MT_pano_heal_menu_v28", icon='SPHERE')


classes = (
    PANO_PG_settings_v28,
    PANO_OT_setup_rig_v28,
    PANO_OT_auto_align_optical_flow_v28,
    PANO_OT_level_exposure_wb_v28,
    PANO_OT_update_distortion_live_v28,
    PANO_OT_bake_equirectangular_v28,
    VIEW3D_PT_pano_heal_stitcher_v28,
    VIEW3D_MT_pano_heal_menu_v28,
)


def _cleanup_legacy_classes():
    """Removes any stale PANO_PT_panel or old operators left in memory from v2.0 - v2.7."""
    for legacy_name in ("PANO_PT_panel", "PANO_PG_settings", "PANO_OT_setup_rig"):
        legacy_cls = getattr(bpy.types, legacy_name, None)
        if legacy_cls is not None:
            try:
                bpy.utils.unregister_class(legacy_cls)
            except Exception:
                pass


def register():
    _cleanup_legacy_classes()
    for cls in classes:
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
        bpy.utils.register_class(cls)
    bpy.types.Scene.pano_settings = PointerProperty(type=PANO_PG_settings_v28)
    bpy.types.VIEW3D_MT_editor_menus.append(draw_header_menu)


def unregister():
    try:
        bpy.types.VIEW3D_MT_editor_menus.remove(draw_header_menu)
    except Exception:
        pass
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
    if hasattr(bpy.types.Scene, "pano_settings"):
        del bpy.types.Scene.pano_settings


if __name__ == "__main__":
    register()