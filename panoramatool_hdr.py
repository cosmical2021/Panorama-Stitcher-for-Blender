bl_info = {
    "name": "OpenCV Pano Stitch, Align, Mask, EXR & IBL World",
    "author": "Blender Addon Developer",
    "version": (1, 4, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar (N) > PanoStitch",
    "description": "Spherical bundle adjustment, 3D vertex masks, 32-bit EXR export, and automatic World IBL lighting",
    "category": "3D View",
}

import bpy
import bmesh
import os
import sys
import subprocess
import math
from mathutils import Euler

CV2_AVAILABLE = False
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# =====================================================================
# 1. DEPENDENCY INSTALLER OPERATOR
# =====================================================================
class PANO_OT_InstallOpenCV(bpy.types.Operator):
    """Install opencv-contrib-python and numpy into Blender's Python environment"""
    bl_idname = "pano.install_opencv"
    bl_label = "Install OpenCV & NumPy"
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        global CV2_AVAILABLE
        python_exe = sys.executable
        try:
            self.report({'INFO'}, "Installing opencv-contrib-python... Please wait.")
            subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
            subprocess.check_call([python_exe, "-m", "pip", "install", "opencv-contrib-python", "numpy"])
            import cv2
            import numpy as np
            CV2_AVAILABLE = True
            self.report({'INFO'}, "OpenCV installed successfully!")
        except Exception as e:
            self.report({'ERROR'}, f"Installation failed: {str(e)}")
            return {'CANCELLED'}
        return {'FINISHED'}


# =====================================================================
# 2. ROTATION MATRIX & VERTEX MASK RASTERIZATION HELPERS
# =====================================================================
def rmat_to_ypr_deg(R):
    """Extract Yaw (Y), Pitch (X), Roll (Z) in degrees from OpenCV 3x3 rotation matrix."""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def apply_ypr_delta_to_rmat(R_base, dyaw_deg, dpitch_deg, droll_deg):
    """Applies manual Yaw, Pitch, Roll offsets (in degrees) on top of the calibrated 3x3 R matrix."""
    rx = math.radians(dpitch_deg)
    ry = math.radians(dyaw_deg)
    rz = math.radians(droll_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)]
    ], dtype=np.float32)

    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)]
    ], dtype=np.float32)

    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    R_delta = Rz @ Ry @ Rx
    return (R_delta @ R_base).astype(np.float32)


def ensure_pano_mask_attribute(mesh):
    """Ensures a 'PanoMask' vertex color attribute exists and defaults to white (1.0 = keep)."""
    if "PanoMask" not in mesh.color_attributes:
        attr = mesh.color_attributes.new(
            name="PanoMask",
            type='BYTE_COLOR',
            domain='CORNER'
        )
        for i in range(len(attr.data)):
            attr.data[i].color = (1.0, 1.0, 1.0, 1.0)
    mesh.color_attributes.active_color = mesh.color_attributes["PanoMask"]
    return mesh.color_attributes["PanoMask"]


def extract_2d_vertex_mask_from_shell(shell_obj, img_w, img_h):
    """Rasterizes the user-painted 'PanoMask' vertex color layer on a 3D CamPatch shell into a 2D OpenCV mask."""
    if shell_obj is None or shell_obj.type != 'MESH':
        return np.full((img_h, img_w), 255, dtype=np.uint8)

    mesh = shell_obj.data
    if "PanoMask" not in mesh.color_attributes or not mesh.uv_layers.active:
        return np.full((img_h, img_w), 255, dtype=np.uint8)

    color_attr = mesh.color_attributes["PanoMask"]
    uv_layer = mesh.uv_layers.active.data

    raster_w = min(img_w, 1024)
    raster_h = min(img_h, int(1024 * (img_h / max(1.0, float(img_w)))))
    mask_canvas = np.full((raster_h, raster_w), 255, dtype=np.uint8)
    has_painted_exclusions = False

    for poly in mesh.polygons:
        pts = []
        luminances = []
        for loop_idx in poly.loop_indices:
            uv = uv_layer[loop_idx].uv
            px = int(round(uv.x * (raster_w - 1)))
            py = int(round((1.0 - uv.y) * (raster_h - 1)))
            pts.append([px, py])

            col = color_attr.data[loop_idx].color
            lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
            luminances.append(lum)

        if min(luminances) < 0.85:
            has_painted_exclusions = True
            avg_lum = sum(luminances) / len(luminances)
            val = 0 if avg_lum < 0.5 else int(avg_lum * 255)
            poly_pts = np.array([pts], dtype=np.int32)
            cv2.fillPoly(mask_canvas, poly_pts, val)

    if not has_painted_exclusions:
        return np.full((img_h, img_w), 255, dtype=np.uint8)

    _, binary_mask = cv2.threshold(mask_canvas, 180, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_mask = cv2.erode(binary_mask, kernel, iterations=1)

    if (raster_w, raster_h) != (img_w, img_h):
        binary_mask = cv2.resize(binary_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    return binary_mask


# =====================================================================
# 3. CAMERA SOLVING & BAKING PIPELINE
# =====================================================================
def solve_cameras_from_folder(folder_path, conf_thresh=0.8, work_megapix=0.6):
    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".exr", ".hdr"}
    image_files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in valid_exts
        and not f.startswith("stitched_panorama")
    ])

    if len(image_files) < 2:
        raise ValueError("Need at least 2 valid source images in the selected folder.")

    valid_paths = []
    work_images = []
    work_scales = []
    full_sizes = []

    for fpath in image_files:
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        h, w = img.shape[:2]
        valid_paths.append(fpath)
        full_sizes.append((w, h))

        w_scale = min(1.0, math.sqrt(work_megapix * 1e6 / (w * h)))
        work_img = cv2.resize(img, (0, 0), fx=w_scale, fy=w_scale, interpolation=cv2.INTER_LINEAR_EXACT)
        work_images.append(work_img)
        work_scales.append(w_scale)

    if len(valid_paths) < 2:
        raise ValueError("Could not decode at least 2 images from folder.")

    try:
        finder = cv2.SIFT_create()
    except Exception:
        finder = cv2.ORB_create(nfeatures=3000)

    features = [cv2.detail.computeImageFeatures2(finder, w_img) for w_img in work_images]
    matcher = cv2.detail_BestOf2NearestMatcher(False, 0.3)
    pairwise_matches = matcher.apply2(features)
    matcher.collectGarbage()

    indices = cv2.detail.leaveBiggestComponent(features, pairwise_matches, conf_thresh)
    indices = [int(i) for i in indices.flatten()] if hasattr(indices, "flatten") else [int(i) for i in indices]

    if len(indices) < 2:
        raise RuntimeError("Fewer than 2 cameras matched. Lower 'Camera Match Confidence' or verify overlap.")

    valid_paths = [valid_paths[i] for i in indices]
    work_scales = [work_scales[i] for i in indices]
    full_sizes = [full_sizes[i] for i in indices]

    estimator = cv2.detail_HomographyBasedEstimator()
    b_ok, cameras = estimator.apply(features, pairwise_matches, None)
    if not b_ok:
        raise RuntimeError("Homography camera estimation failed.")

    for cam in cameras:
        cam.R = cam.R.astype(np.float32)

    adjuster = cv2.detail_BundleAdjusterRay()
    adjuster.setConfThresh(conf_thresh)
    b_ok, cameras = adjuster.apply(features, pairwise_matches, cameras)
    if not b_ok:
        raise RuntimeError("Camera Bundle Adjustment failed.")

    rmats = [np.copy(cam.R) for cam in cameras]
    rmats = cv2.detail.waveCorrect(rmats, cv2.detail.WAVE_CORRECT_HORIZ)

    solved_data = []
    for idx, cam in enumerate(cameras):
        R_mat = rmats[idx].astype(np.float32)
        focal_full = float(cam.focal / work_scales[idx])
        w_full, h_full = full_sizes[idx]
        yaw_deg, pitch_deg, roll_deg = rmat_to_ypr_deg(R_mat)

        solved_data.append({
            "filepath": valid_paths[idx],
            "filename": os.path.basename(valid_paths[idx]),
            "width": w_full,
            "height": h_full,
            "focal_px": focal_full,
            "R_flat": R_mat.flatten().tolist(),
            "base_yaw": yaw_deg,
            "base_pitch": pitch_deg,
            "base_roll": roll_deg,
        })

    return solved_data


def bake_aligned_cameras_to_panorama(
    cam_items,
    seam_megapix=0.1,
    blend_strength=5.0,
    inpaint_radius=7,
    inpaint_method='TELEA',
    heal_seams=True,
    is_hdr=False
):
    active_entries = []
    for idx, item in enumerate(cam_items):
        if item.enabled and os.path.isfile(item.filepath):
            shell_name = f"CamPatch_{idx:02d}_{item.filename}"
            shell_obj = bpy.data.objects.get(shell_name)
            active_entries.append((item, shell_obj))

    if len(active_entries) < 2:
        raise ValueError("At least 2 enabled cameras are required to bake a stitched panorama.")

    full_img_list = []
    full_user_masks = []
    seam_images = []
    seam_user_masks = []
    seam_scales = []
    R_matrices = []
    focals_full = []
    ppx_list = []
    ppy_list = []
    exposure_gains = []

    for item, shell_obj in active_entries:
        img = cv2.imread(item.filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        h, w = img.shape[:2]
        full_img_list.append(img)

        user_mask_full = extract_2d_vertex_mask_from_shell(shell_obj, w, h)
        full_user_masks.append(user_mask_full)

        s_scale = min(1.0, math.sqrt(seam_megapix * 1e6 / (w * h)))
        seam_img = cv2.resize(img, (0, 0), fx=s_scale, fy=s_scale, interpolation=cv2.INTER_LINEAR_EXACT)
        seam_mask = cv2.resize(user_mask_full, (seam_img.shape[1], seam_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        seam_images.append(seam_img)
        seam_user_masks.append(seam_mask)
        seam_scales.append(s_scale)

        R_base = np.array(item.r_matrix_flat, dtype=np.float32).reshape((3, 3))
        R_final = apply_ypr_delta_to_rmat(
            R_base, dyaw_deg=item.manual_yaw, dpitch_deg=item.manual_pitch, droll_deg=item.manual_roll
        )
        R_matrices.append(R_final)

        eff_focal = max(50.0, item.focal_px * item.manual_scale)
        focals_full.append(eff_focal)
        ppx_list.append(w * 0.5 + item.shift_x_px)
        ppy_list.append(h * 0.5 + item.shift_y_px)
        exposure_gains.append(item.exposure_gain)

    warped_image_scale = float(np.median(focals_full))

    corners_seam = []
    masks_warped_seam = []
    images_warped_seam_f = []

    for idx, s_img in enumerate(seam_images):
        s_scale = seam_scales[idx]
        warper_seam = cv2.PyRotationWarper('spherical', warped_image_scale * s_scale)
        K_seam = np.array([
            [focals_full[idx] * s_scale, 0.0, ppx_list[idx] * s_scale],
            [0.0, focals_full[idx] * s_scale, ppy_list[idx] * s_scale],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        corner, img_wp = warper_seam.warp(s_img, K_seam, R_matrices[idx], cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        corners_seam.append(corner)
        images_warped_seam_f.append(img_wp.astype(np.float32))

        _, mask_wp = warper_seam.warp(seam_user_masks[idx], K_seam, R_matrices[idx], cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        masks_warped_seam.append(mask_wp)

    compensator = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS)
    compensator.feed(corners=corners_seam, images=[im.astype(np.float32) for im in images_warped_seam_f], masks=masks_warped_seam)

    try:
        seam_finder = cv2.detail_GraphCutSeamFinder('COST_COLOR')
        masks_warped_seam = seam_finder.find(images_warped_seam_f, corners_seam, masks_warped_seam)
    except Exception:
        seam_finder = cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_VORONOI_SEAM)
        masks_warped_seam = seam_finder.find(images_warped_seam_f, corners_seam, masks_warped_seam)

    warper_full = cv2.PyRotationWarper('spherical', warped_image_scale)
    corners_full = []
    sizes_full = []
    warped_imgs_full = []
    warped_masks_full = []

    for idx, full_img in enumerate(full_img_list):
        K_full = np.array([
            [focals_full[idx], 0.0, ppx_list[idx]],
            [0.0, focals_full[idx], ppy_list[idx]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        corner, img_wp = warper_full.warp(full_img, K_full, R_matrices[idx], cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        size = (img_wp.shape[1], img_wp.shape[0])

        _, user_mask_wp = warper_full.warp(full_user_masks[idx], K_full, R_matrices[idx], cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        compensator.apply(idx, corner, img_wp, user_mask_wp)

        if abs(exposure_gains[idx] - 1.0) > 1e-3:
            img_wp = np.clip(img_wp.astype(np.float32) * exposure_gains[idx], 0, 65535 if is_hdr else 255)

        dilated_seam = cv2.dilate(masks_warped_seam[idx], None)
        resized_seam = cv2.resize(dilated_seam, size, 0, 0, cv2.INTER_LINEAR_EXACT)
        final_mask = cv2.bitwise_and(resized_seam, user_mask_wp)

        corners_full.append(corner)
        sizes_full.append(size)
        warped_imgs_full.append(img_wp)
        warped_masks_full.append(final_mask)

    dst_roi = cv2.detail.resultRoi(corners=corners_full, sizes=sizes_full)
    roi_w, roi_h = dst_roi[2], dst_roi[3]

    blend_width = math.sqrt(roi_w * roi_h) * (blend_strength / 100.0)
    if blend_width < 1.0:
        blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
    else:
        num_bands = int(min(7, max(2, math.ceil(math.log(blend_width) / math.log(2.0)) - 1)))
        blender = cv2.detail_MultiBandBlender(0, num_bands)

    blender.prepare(dst_roi)
    for img_wp, mask_wp, corner in zip(warped_imgs_full, warped_masks_full, corners_full):
        img_s16 = img_wp[:, :, :3].astype(np.float32) if is_hdr else img_wp[:, :, :3].astype(np.int16)
        blender.feed(img_s16, mask_wp, corner)

    result_s16, result_mask = blender.blend(None, None)
    stitched = np.clip(result_s16, 0, 65535 if is_hdr else 255)

    coords = cv2.findNonZero(result_mask)
    if coords is not None:
        x, y, w_valid, h_valid = cv2.boundingRect(coords)
        stitched = stitched[y:y+h_valid, x:x+w_valid]
        result_mask = result_mask[y:y+h_valid, x:x+w_valid]
    else:
        h_valid, w_valid = stitched.shape[:2]

    true_hfov_deg = round(max(1.0, min(360.0, math.degrees(w_valid / warped_image_scale))), 2)
    true_vfov_deg = round(max(1.0, min(180.0, math.degrees(h_valid / warped_image_scale))), 2)

    if heal_seams and inpaint_radius > 0 and not is_hdr:
        missing_mask = cv2.bitwise_not(result_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        inpaint_mask = cv2.dilate(missing_mask, kernel, iterations=2)
        algo = cv2.INPAINT_TELEA if inpaint_method == 'TELEA' else cv2.INPAINT_NS
        stitched = cv2.inpaint(stitched.astype(np.uint8), inpaint_mask, inpaintRadius=float(inpaint_radius), flags=algo)

    return stitched, true_hfov_deg, true_vfov_deg, len(active_entries)


# =====================================================================
# 4. 3D SPHERE, PREVIEW SHELLS & WORLD IBL SETUP
# =====================================================================
def create_or_update_pano_sphere(context, name, radius, hfov_deg, vfov_deg, seg_u, seg_v, preserve_mask=False):
    mesh_name = f"{name}_Mesh"
    obj_name = name

    hfov_rad = math.radians(max(1.0, min(360.0, hfov_deg)))
    vfov_rad = math.radians(max(1.0, min(180.0, vfov_deg)))

    actual_seg_u = max(8, int(seg_u * (hfov_deg / 360.0))) if hfov_deg < 360.0 else seg_u
    actual_seg_v = max(8, int(seg_v * (vfov_deg / 180.0))) if vfov_deg < 180.0 else seg_v

    theta_start = -hfov_rad * 0.5
    phi_start = -vfov_rad * 0.5
    expected_verts = (actual_seg_u + 1) * (actual_seg_v + 1)

    if preserve_mask and obj_name in bpy.data.objects:
        obj = bpy.data.objects[obj_name]
        mesh = obj.data
        if len(mesh.vertices) == expected_verts:
            v_idx = 0
            for iv in range(actual_seg_v + 1):
                v_norm = iv / float(actual_seg_v)
                phi = phi_start + v_norm * vfov_rad
                for iu in range(actual_seg_u + 1):
                    u_norm = iu / float(actual_seg_u)
                    theta = theta_start + u_norm * hfov_rad
                    x = radius * math.cos(phi) * math.sin(theta)
                    y = radius * math.cos(phi) * math.cos(theta)
                    z = radius * math.sin(phi)
                    mesh.vertices[v_idx].co = (x, y, z)
                    v_idx += 1
            mesh.update()
            ensure_pano_mask_attribute(mesh)
            return obj

    if obj_name in bpy.data.objects:
        obj = bpy.data.objects[obj_name]
        mesh = obj.data
        mesh.clear_geometry()
    else:
        mesh = bpy.data.meshes.new(mesh_name)
        obj = bpy.data.objects.new(obj_name, mesh)
        context.collection.objects.link(obj)

    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    verts_grid = []
    for iv in range(actual_seg_v + 1):
        v_norm = iv / float(actual_seg_v)
        phi = phi_start + v_norm * vfov_rad
        row_verts = []
        for iu in range(actual_seg_u + 1):
            u_norm = iu / float(actual_seg_u)
            theta = theta_start + u_norm * hfov_rad
            x = radius * math.cos(phi) * math.sin(theta)
            y = radius * math.cos(phi) * math.cos(theta)
            z = radius * math.sin(phi)
            vert = bm.verts.new((x, y, z))
            row_verts.append((vert, u_norm, v_norm))
        verts_grid.append(row_verts)

    bm.verts.ensure_lookup_table()

    for iv in range(actual_seg_v):
        for iu in range(actual_seg_u):
            v00, u00, vn00 = verts_grid[iv][iu]
            v10, u10, vn10 = verts_grid[iv][iu + 1]
            v11, u11, vn11 = verts_grid[iv + 1][iu + 1]
            v01, u01, vn01 = verts_grid[iv + 1][iu]
            face = bm.faces.new((v00, v01, v11, v10))
            face.smooth = True
            uvs = [(u00, vn00), (u01, vn01), (u11, vn11), (u10, vn10)]
            for loop, uv_coord in zip(face.loops, uvs):
                loop[uv_layer].uv = uv_coord

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    if preserve_mask:
        ensure_pano_mask_attribute(mesh)

    return obj


def spawn_or_update_camera_preview_shells(context):
    props = context.scene.pano_stitch_props
    coll_name = "Pano_Manual_Align_Preview"

    if coll_name not in bpy.data.collections:
        preview_coll = bpy.data.collections.new(coll_name)
        context.scene.collection.children.link(preview_coll)
    else:
        preview_coll = bpy.data.collections[coll_name]

    base_radius = props.sphere_radius

    for idx, item in enumerate(props.camera_items):
        shell_name = f"CamPatch_{idx:02d}_{item.filename}"
        if not item.enabled:
            if shell_name in bpy.data.objects:
                bpy.data.objects[shell_name].hide_viewport = True
            continue

        eff_focal = max(50.0, item.focal_px * item.manual_scale)
        cam_hfov = math.degrees(2.0 * math.atan((item.width * 0.5) / eff_focal))
        cam_vfov = math.degrees(2.0 * math.atan((item.height * 0.5) / eff_focal))
        shell_radius = base_radius - (idx * 0.015)

        obj = create_or_update_pano_sphere(
            context, name=shell_name, radius=shell_radius,
            hfov_deg=cam_hfov, vfov_deg=cam_vfov, seg_u=96, seg_v=64, preserve_mask=True
        )

        for c in obj.users_collection:
            c.objects.unlink(obj)
        preview_coll.objects.link(obj)
        obj.hide_viewport = False

        total_yaw = math.radians(-(item.base_yaw + item.manual_yaw))
        total_pitch = math.radians(item.base_pitch + item.manual_pitch)
        total_roll = math.radians(item.base_roll + item.manual_roll)
        obj.rotation_euler = Euler((total_pitch, total_roll, total_yaw), 'XYZ')

        mat_name = f"{shell_name}_Mat"
        mat = bpy.data.materials.get(mat_name)
        if not mat:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            if hasattr(mat, "blend_method"):
                mat.blend_method = 'BLEND'

            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()

            tex_node = nodes.new(type='ShaderNodeTexImage')
            tex_node.name = "Cam_Tex"
            tex_node.location = (-500, 150)
            if os.path.isfile(item.filepath):
                tex_node.image = bpy.data.images.load(item.filepath, check_existing=True)

            vcol_node = nodes.new(type='ShaderNodeVertexColor')
            vcol_node.layer_name = "PanoMask"
            vcol_node.location = (-500, -150)

            math_mult = nodes.new(type='ShaderNodeMath')
            math_mult.operation = 'MULTIPLY'
            math_mult.inputs[1].default_value = 0.90
            math_mult.location = (-200, -150)

            emis = nodes.new(type='ShaderNodeEmission')
            emis.name = "Cam_Emis"
            emis.location = (-200, 150)
            emis.inputs['Strength'].default_value = item.exposure_gain

            transp = nodes.new(type='ShaderNodeBsdfTransparent')
            transp.location = (-200, 0)

            mix_shader = nodes.new(type='ShaderNodeMixShader')
            mix_shader.location = (100, 100)

            out = nodes.new(type='ShaderNodeOutputMaterial')
            out.location = (350, 100)

            links.new(tex_node.outputs['Color'], emis.inputs['Color'])
            links.new(vcol_node.outputs['Color'], math_mult.inputs[0])
            links.new(math_mult.outputs['Value'], mix_shader.inputs['Fac'])
            links.new(transp.outputs['BSDF'], mix_shader.inputs[1])
            links.new(emis.outputs['Emission'], mix_shader.inputs[2])
            links.new(mix_shader.outputs['Shader'], out.inputs['Surface'])

        if mat.use_nodes:
            emis = mat.node_tree.nodes.get("Cam_Emis")
            if emis:
                emis.inputs['Strength'].default_value = item.exposure_gain

        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)


def setup_pano_shader(obj, bl_image, props):
    mat_name = f"{obj.name}_Mat"
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_tex_coord = nodes.new(type='ShaderNodeTexCoord')
    node_tex_coord.location = (-800, 0)

    node_mapping = nodes.new(type='ShaderNodeMapping')
    node_mapping.name = "Pano_Wrap_Mapping"
    node_mapping.location = (-600, 0)
    node_mapping.inputs['Location'].default_value = (props.wrap_offset_u, props.wrap_offset_v, 0.0)
    node_mapping.inputs['Scale'].default_value = (props.wrap_scale_u, props.wrap_scale_v, 1.0)

    node_tex = nodes.new(type='ShaderNodeTexImage')
    node_tex.name = "Pano_Image_Node"
    node_tex.location = (-300, 0)
    node_tex.image = bl_image
    node_tex.extension = props.wrap_extension_mode
    node_tex.interpolation = 'Smart'

    node_emission = nodes.new(type='ShaderNodeEmission')
    node_emission.location = (50, 0)
    node_emission.inputs['Strength'].default_value = 1.0

    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    node_out.location = (300, 0)

    links.new(node_tex_coord.outputs['UV'], node_mapping.inputs['Vector'])
    links.new(node_mapping.outputs['Vector'], node_tex.inputs['Vector'])
    links.new(node_tex.outputs['Color'], node_emission.inputs['Color'])
    links.new(node_emission.outputs['Emission'], node_out.inputs['Surface'])

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def setup_world_ibl(context, bl_image, hfov_deg):
    """Automatically configures Blender World Environment IBL lighting using the baked panorama."""
    props = context.scene.pano_stitch_props
    if not props.setup_world_ibl or hfov_deg < 350.0:
        return

    scene = context.scene
    if not scene.world:
        scene.world = bpy.data.worlds.new("PanoWorld")
    world = scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    tex_coord.location = (-800, 0)

    mapping = nodes.new(type='ShaderNodeMapping')
    mapping.location = (-550, 0)
    mapping.inputs['Rotation'].default_value = (0.0, 0.0, math.radians(props.ibl_rotation))

    tex_env = nodes.new(type='ShaderNodeTexEnvironment')
    tex_env.image = bl_image
    tex_env.location = (-300, 0)

    bg = nodes.new(type='ShaderNodeBackground')
    bg.inputs['Strength'].default_value = props.ibl_strength
    bg.location = (0, 0)

    world_out = nodes.new(type='ShaderNodeOutputWorld')
    world_out.location = (250, 0)

    links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], tex_env.inputs['Vector'])
    links.new(tex_env.outputs['Color'], bg.inputs['Color'])
    links.new(bg.outputs['Background'], world_out.inputs['Surface'])


def update_live_projection(self, context):
    props = context.scene.pano_stitch_props
    obj = bpy.data.objects.get("Stitched_Pano_Sphere")
    if not obj:
        return

    create_or_update_pano_sphere(
        context, name="Stitched_Pano_Sphere", radius=props.sphere_radius,
        hfov_deg=props.hfov_deg, vfov_deg=props.vfov_deg, seg_u=props.segments_u, seg_v=props.segments_v
    )

    mat = bpy.data.materials.get("Stitched_Pano_Sphere_Mat")
    if mat and mat.use_nodes:
        nodes = mat.node_tree.nodes
        mapping = nodes.get("Pano_Wrap_Mapping")
        if mapping:
            mapping.inputs['Location'].default_value = (props.wrap_offset_u, props.wrap_offset_v, 0.0)
            mapping.inputs['Scale'].default_value = (props.wrap_scale_u, props.wrap_scale_v, 1.0)
        tex_node = nodes.get("Pano_Image_Node")
        if tex_node:
            tex_node.extension = props.wrap_extension_mode


def update_manual_camera_transform(self, context):
    props = context.scene.pano_stitch_props
    if props.show_3d_align_shells:
        spawn_or_update_camera_preview_shells(context)


# =====================================================================
# 5. OPERATORS
# =====================================================================
class PANO_OT_AnalyzeFolderCameras(bpy.types.Operator):
    bl_idname = "pano.analyze_cameras"
    bl_label = "1. Analyze & Auto-Align Folder"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not CV2_AVAILABLE:
            self.report({'ERROR'}, "OpenCV is not installed. Click 'Install OpenCV' first.")
            return {'CANCELLED'}

        props = context.scene.pano_stitch_props
        folder = bpy.path.abspath(props.input_folder)

        if not os.path.isdir(folder):
            self.report({'ERROR'}, "Please select a valid image directory.")
            return {'CANCELLED'}

        try:
            solved = solve_cameras_from_folder(folder, conf_thresh=props.confidence_thresh)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        props.camera_items.clear()
        for data in solved:
            item = props.camera_items.add()
            item.filepath = data["filepath"]
            item.filename = data["filename"]
            item.width = data["width"]
            item.height = data["height"]
            item.focal_px = data["focal_px"]
            item.r_matrix_flat = data["R_flat"]
            item.base_yaw = data["base_yaw"]
            item.base_pitch = data["base_pitch"]
            item.base_roll = data["base_roll"]
            item.manual_yaw = 0.0
            item.manual_pitch = 0.0
            item.manual_roll = 0.0
            item.manual_scale = 1.0
            item.exposure_gain = 1.0

        props.active_camera_index = 0
        if props.show_3d_align_shells:
            spawn_or_update_camera_preview_shells(context)

        self.report({'INFO'}, f"Matched & aligned {len(solved)} stitchable cameras!")
        return {'FINISHED'}


class PANO_OT_TogglePreviewShells(bpy.types.Operator):
    bl_idname = "pano.toggle_preview_shells"
    bl_label = "Toggle 3D Manual Align Shells"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.pano_stitch_props
        if len(props.camera_items) == 0:
            self.report({'WARNING'}, "Run '1. Analyze & Auto-Align Folder' first.")
            return {'CANCELLED'}

        props.show_3d_align_shells = not props.show_3d_align_shells
        coll_name = "Pano_Manual_Align_Preview"

        if props.show_3d_align_shells:
            spawn_or_update_camera_preview_shells(context)
            if coll_name in bpy.data.collections:
                bpy.data.collections[coll_name].hide_viewport = False
        else:
            if context.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            if coll_name in bpy.data.collections:
                bpy.data.collections[coll_name].hide_viewport = True
        return {'FINISHED'}


class PANO_OT_PaintCameraVertexMask(bpy.types.Operator):
    bl_idname = "pano.paint_camera_mask"
    bl_label = "Paint Exclusion Mask on Active Shell"
    bl_options = {'REGISTER', 'UNDO'}

    brush_mode: bpy.props.EnumProperty(
        items=[
            ('EXCLUDE', "Paint Exclude (Black)", "Paint black over objects/tripods to remove them"),
            ('RESTORE', "Paint Keep (White)", "Paint white to restore masked areas")
        ],
        default='EXCLUDE'
    )

    def execute(self, context):
        props = context.scene.pano_stitch_props
        idx = props.active_camera_index
        if not (0 <= idx < len(props.camera_items)):
            return {'CANCELLED'}

        if not props.show_3d_align_shells:
            props.show_3d_align_shells = True
            spawn_or_update_camera_preview_shells(context)
            coll_name = "Pano_Manual_Align_Preview"
            if coll_name in bpy.data.collections:
                bpy.data.collections[coll_name].hide_viewport = False

        item = props.camera_items[idx]
        shell_name = f"CamPatch_{idx:02d}_{item.filename}"
        shell_obj = bpy.data.objects.get(shell_name)
        if not shell_obj:
            spawn_or_update_camera_preview_shells(context)
            shell_obj = bpy.data.objects.get(shell_name)

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.select_all(action='DESELECT')
        shell_obj.hide_viewport = False
        shell_obj.select_set(True)
        context.view_layer.objects.active = shell_obj

        ensure_pano_mask_attribute(shell_obj.data)
        bpy.ops.object.mode_set(mode='VERTEX_PAINT')

        target_col = (0.0, 0.0, 0.0) if self.brush_mode == 'EXCLUDE' else (1.0, 1.0, 1.0)
        ts = context.tool_settings
        if ts and ts.vertex_paint and ts.vertex_paint.brush:
            ts.vertex_paint.brush.color = target_col
        if ts and hasattr(ts, "unified_paint_settings"):
            ts.unified_paint_settings.color = target_col
        return {'FINISHED'}


class PANO_OT_ClearCameraVertexMask(bpy.types.Operator):
    bl_idname = "pano.clear_camera_mask"
    bl_label = "Clear Mask on Selected Camera"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.pano_stitch_props
        idx = props.active_camera_index
        if not (0 <= idx < len(props.camera_items)):
            return {'CANCELLED'}

        item = props.camera_items[idx]
        shell_obj = bpy.data.objects.get(f"CamPatch_{idx:02d}_{item.filename}")
        if not shell_obj or shell_obj.type != 'MESH':
            return {'CANCELLED'}

        prev_mode = context.mode
        if prev_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        attr = ensure_pano_mask_attribute(shell_obj.data)
        for i in range(len(attr.data)):
            attr.data[i].color = (1.0, 1.0, 1.0, 1.0)
        shell_obj.data.update()

        if prev_mode == 'PAINT_VERTEX':
            bpy.ops.object.mode_set(mode='VERTEX_PAINT')
        return {'FINISHED'}


class PANO_OT_SyncFrom3DShells(bpy.types.Operator):
    bl_idname = "pano.sync_from_3d_shells"
    bl_label = "Sync Rotations from 3D Viewport"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.pano_stitch_props
        for idx, item in enumerate(props.camera_items):
            obj = bpy.data.objects.get(f"CamPatch_{idx:02d}_{item.filename}")
            if obj:
                rot = obj.rotation_euler
                item.manual_yaw = -math.degrees(rot.z) - item.base_yaw
                item.manual_pitch = math.degrees(rot.x) - item.base_pitch
                item.manual_roll = math.degrees(rot.y) - item.base_roll
        self.report({'INFO'}, "Synced manual 3D rotations.")
        return {'FINISHED'}


class PANO_OT_ResetCameraTweaks(bpy.types.Operator):
    bl_idname = "pano.reset_camera_tweaks"
    bl_label = "Reset Selected Camera Tweaks"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.pano_stitch_props
        idx = props.active_camera_index
        if 0 <= idx < len(props.camera_items):
            item = props.camera_items[idx]
            item.manual_yaw = 0.0
            item.manual_pitch = 0.0
            item.manual_roll = 0.0
            item.manual_scale = 1.0
            item.shift_x_px = 0.0
            item.shift_y_px = 0.0
            item.exposure_gain = 1.0
            update_manual_camera_transform(self, context)
        return {'FINISHED'}


class PANO_OT_BakeFinalPanorama(bpy.types.Operator):
    bl_idname = "pano.bake_final_panorama"
    bl_label = "2. Bake Final Seamless Panorama"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not CV2_AVAILABLE:
            self.report({'ERROR'}, "OpenCV is not installed.")
            return {'CANCELLED'}

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        props = context.scene.pano_stitch_props
        folder = bpy.path.abspath(props.input_folder)

        if len(props.camera_items) == 0:
            res = bpy.ops.pano.analyze_cameras()
            if 'FINISHED' not in res:
                return {'CANCELLED'}

        is_hdr = (props.output_format == 'EXR')

        try:
            stitched_bgr, true_hfov, true_vfov, used_cams = bake_aligned_cameras_to_panorama(
                cam_items=props.camera_items,
                blend_strength=props.multiband_strength,
                inpaint_radius=props.inpaint_radius,
                inpaint_method=props.inpaint_method,
                heal_seams=props.enable_inpaint_heal,
                is_hdr=is_hdr
            )
        except Exception as e:
            self.report({'ERROR'}, f"Bake failed: {str(e)}")
            return {'CANCELLED'}

        ext = "exr" if is_hdr else "png"
        out_path = os.path.join(folder, f"stitched_panorama_baked.{ext}")

        if is_hdr:
            cv2.imwrite(out_path, stitched_bgr.astype(np.float32))
        else:
            cv2.imwrite(out_path, stitched_bgr.astype(np.uint8))

        bl_img = bpy.data.images.load(out_path, check_existing=True)
        bl_img.reload()

        if props.use_measured_fov:
            props.hfov_deg = true_hfov
            props.vfov_deg = true_vfov

        sphere_obj = create_or_update_pano_sphere(
            context, name="Stitched_Pano_Sphere", radius=props.sphere_radius,
            hfov_deg=props.hfov_deg, vfov_deg=props.vfov_deg, seg_u=props.segments_u, seg_v=props.segments_v
        )

        setup_pano_shader(sphere_obj, bl_img, props)
        setup_world_ibl(context, bl_img, props.hfov_deg)

        coll_name = "Pano_Manual_Align_Preview"
        if coll_name in bpy.data.collections:
            bpy.data.collections[coll_name].hide_viewport = True
        props.show_3d_align_shells = False

        bpy.ops.object.select_all(action='DESELECT')
        sphere_obj.select_set(True)
        context.view_layer.objects.active = sphere_obj

        self.report({'INFO'}, f"Baked {used_cams} cameras to '{out_path}'! Span: {props.hfov_deg:.1f}° x {props.vfov_deg:.1f}°")
        return {'FINISHED'}


# =====================================================================
# 6. PROPERTY GROUPS & UI
# =====================================================================
class PanoCameraItem(bpy.types.PropertyGroup):
    filename: bpy.props.StringProperty(name="File")
    filepath: bpy.props.StringProperty(name="Path")
    enabled: bpy.props.BoolProperty(name="Include", default=True, update=update_manual_camera_transform)
    width: bpy.props.IntProperty(name="Width", default=1920)
    height: bpy.props.IntProperty(name="Height", default=1080)
    focal_px: bpy.props.FloatProperty(name="Solved Focal (px)", default=1000.0)
    r_matrix_flat: bpy.props.FloatVectorProperty(name="R Matrix", size=9, default=(1, 0, 0, 0, 1, 0, 0, 0, 1))
    base_yaw: bpy.props.FloatProperty(name="Base Yaw", default=0.0)
    base_pitch: bpy.props.FloatProperty(name="Base Pitch", default=0.0)
    base_roll: bpy.props.FloatProperty(name="Base Roll", default=0.0)
    manual_yaw: bpy.props.FloatProperty(name="Yaw Offset (°)", default=0.0, min=-180.0, max=180.0, update=update_manual_camera_transform)
    manual_pitch: bpy.props.FloatProperty(name="Pitch Offset (°)", default=0.0, min=-90.0, max=90.0, update=update_manual_camera_transform)
    manual_roll: bpy.props.FloatProperty(name="Roll Offset (°)", default=0.0, min=-180.0, max=180.0, update=update_manual_camera_transform)
    manual_scale: bpy.props.FloatProperty(name="FOV / Focal Scale", default=1.0, min=0.2, max=5.0, update=update_manual_camera_transform)
    shift_x_px: bpy.props.FloatProperty(name="Shift X", default=0.0, min=-2000.0, max=2000.0)
    shift_y_px: bpy.props.FloatProperty(name="Shift Y", default=0.0, min=-2000.0, max=2000.0)
    exposure_gain: bpy.props.FloatProperty(name="Exposure", default=1.0, min=0.1, max=5.0, update=update_manual_camera_transform)


class PANO_UL_CameraList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        row.label(text=item.filename, icon='CAMERA_DATA')
        sub = row.row(align=True)
        sub.alignment = 'RIGHT'
        sub.label(text=f"Y:{item.base_yaw + item.manual_yaw:.1f}° P:{item.base_pitch + item.manual_pitch:.1f}°")


class PanoStitchProperties(bpy.types.PropertyGroup):
    input_folder: bpy.props.StringProperty(name="Image Folder", default="", subtype='DIR_PATH')
    confidence_thresh: bpy.props.FloatProperty(name="Camera Match Confidence", default=0.8, min=0.1, max=2.0)
    camera_items: bpy.props.CollectionProperty(type=PanoCameraItem)
    active_camera_index: bpy.props.IntProperty(name="Active Camera", default=0)
    show_3d_align_shells: bpy.props.BoolProperty(name="Show 3D Preview Shells", default=False)

    output_format: bpy.props.EnumProperty(
        name="Output Format",
        items=[
            ('PNG', "PNG (8-bit)", "Standard 8-bit PNG image"),
            ('EXR', "OpenEXR (32-bit HDR Float)", "32-bit Float HDR OpenEXR panorama")
        ],
        default='PNG'
    )
    setup_world_ibl: bpy.props.BoolProperty(
        name="Setup World IBL for Lighting",
        description="Automatically configure Blender World environment lighting when panorama is near 360x180",
        default=True
    )
    ibl_strength: bpy.props.FloatProperty(name="IBL Strength", default=1.0, min=0.0, max=100.0)
    ibl_rotation: bpy.props.FloatProperty(name="IBL Rotation (°)", default=0.0, min=-360.0, max=360.0)

    multiband_strength: bpy.props.FloatProperty(name="Multi-Band Seam Blend (%)", default=5.0, min=0.0, max=50.0)
    enable_inpaint_heal: bpy.props.BoolProperty(name="Inpaint & Heal Seams/Masks", default=True)
    inpaint_method: bpy.props.EnumProperty(
        name="Heal Algorithm",
        items=[('TELEA', "Fast Marching (Telea)", ""), ('NS', "Navier-Stokes (Fluid)", "")],
        default='TELEA'
    )
    inpaint_radius: bpy.props.IntProperty(name="Heal Brush Radius", default=9, min=1, max=50)

    use_measured_fov: bpy.props.BoolProperty(name="Use Measured Camera FOV", default=True)
    hfov_deg: bpy.props.FloatProperty(name="Horizontal Span (°)", default=120.0, min=1.0, max=360.0, update=update_live_projection)
    vfov_deg: bpy.props.FloatProperty(name="Vertical Span (°)", default=65.0, min=1.0, max=180.0, update=update_live_projection)
    sphere_radius: bpy.props.FloatProperty(name="Sphere Radius", default=10.0, min=0.1, max=1000.0, update=update_live_projection)
    segments_u: bpy.props.IntProperty(name="Segments U", default=128, min=16, max=512, update=update_live_projection)
    segments_v: bpy.props.IntProperty(name="Segments V", default=64, min=8, max=256, update=update_live_projection)

    wrap_extension_mode: bpy.props.EnumProperty(
        name="Wrap Mode",
        items=[('EXTEND', "Extend", ""), ('MIRROR', "Mirror", ""), ('REPEAT', "Repeat", ""), ('CLIP', "Clip", "")],
        default='EXTEND', update=update_live_projection
    )
    wrap_offset_u: bpy.props.FloatProperty(name="Wrap Offset U", default=0.0, min=-2.0, max=2.0, update=update_live_projection)
    wrap_offset_v: bpy.props.FloatProperty(name="Wrap Offset V", default=0.0, min=-2.0, max=2.0, update=update_live_projection)
    wrap_scale_u: bpy.props.FloatProperty(name="Wrap Scale U", default=1.0, min=0.05, max=10.0, update=update_live_projection)
    wrap_scale_v: bpy.props.FloatProperty(name="Wrap Scale V", default=1.0, min=0.05, max=10.0, update=update_live_projection)


class PANO_PT_MainPanel(bpy.types.Panel):
    bl_label = "OpenCV Pano Align, Mask, EXR & IBL"
    bl_idname = "PANO_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'PanoStitch'

    def draw(self, context):
        layout = self.layout
        props = context.scene.pano_stitch_props

        if not CV2_AVAILABLE:
            box = layout.box()
            box.alert = True
            box.label(text="OpenCV Not Found!", icon='ERROR')
            box.operator("pano.install_opencv", icon='IMPORT')
            return

        box = layout.box()
        box.label(text="1. Folder & Camera Filtering", icon='FILE_FOLDER')
        box.prop(props, "input_folder", text="")
        box.prop(props, "confidence_thresh", slider=True)
        row = box.row(align=True)
        row.scale_y = 1.3
        row.operator("pano.analyze_cameras", icon='VIEWZOOM')

        box = layout.box()
        box.label(text="2. Align & Paint Exclusion Masks", icon='ORIENTATION_GIMBAL')
        if len(props.camera_items) > 0:
            row = box.row(align=True)
            row.operator(
                "pano.toggle_preview_shells",
                text="Hide 3D Align Shells" if props.show_3d_align_shells else "Show 3D Align Shells",
                icon='HIDE_OFF' if props.show_3d_align_shells else 'HIDE_ON'
            )
            if props.show_3d_align_shells:
                row.operator("pano.sync_from_3d_shells", text="Sync 3D", icon='FILE_REFRESH')

            box.template_list("PANO_UL_CameraList", "", props, "camera_items", props, "active_camera_index", rows=4)

            idx = props.active_camera_index
            if 0 <= idx < len(props.camera_items):
                cam = props.camera_items[idx]
                mask_box = box.box()
                mask_box.label(text=f"Object / Tripod Mask ({cam.filename}):", icon='VPAINT_HLT')
                m_row = mask_box.row(align=True)
                op_ex = m_row.operator("pano.paint_camera_mask", text="Paint Exclude", icon='BRUSH_DATA')
                op_ex.brush_mode = 'EXCLUDE'
                op_re = m_row.operator("pano.paint_camera_mask", text="Paint Keep", icon='ADD')
                op_re.brush_mode = 'RESTORE'
                m_row2 = mask_box.row(align=True)
                m_row2.operator("pano.clear_camera_mask", text="Clear Mask", icon='TRASH')

                col = box.column(align=True)
                col.label(text=f"Transform ({cam.focal_px * cam.manual_scale:.0f}px):")
                col.prop(cam, "manual_yaw", slider=True)
                col.prop(cam, "manual_pitch", slider=True)
                col.prop(cam, "manual_roll", slider=True)
                col.separator()
                col.prop(cam, "manual_scale", slider=True)
                col.prop(cam, "exposure_gain", slider=True)
                row_shift = col.row(align=True)
                row_shift.prop(cam, "shift_x_px", text="Shift X")
                row_shift.prop(cam, "shift_y_px", text="Shift Y")
                col.separator()
                col.operator("pano.reset_camera_tweaks", icon='LOOP_BACK')
        else:
            box.label(text="Click '1. Analyze' to load cameras.", icon='INFO')

        box = layout.box()
        box.label(text="3. Seam Healing, EXR & Bake", icon='BRUSH_DATA')
        box.prop(props, "output_format")
        box.prop(props, "setup_world_ibl")
        if props.setup_world_ibl:
            col_ibl = box.column(align=True)
            col_ibl.prop(props, "ibl_strength")
            col_ibl.prop(props, "ibl_rotation", slider=True)
        box.prop(props, "multiband_strength", slider=True)
        box.prop(props, "enable_inpaint_heal")
        if props.enable_inpaint_heal:
            box.prop(props, "inpaint_method", text="Mode")
            box.prop(props, "inpaint_radius", slider=True)

        row = box.row(align=True)
        row.scale_y = 1.6
        row.operator("pano.bake_final_panorama", icon='RENDER_STILL')

        box = layout.box()
        box.label(text="4. Baked Sphere Span (1-360° x 1-180°)", icon='MESH_UVSPHERE')
        box.prop(props, "use_measured_fov")
        col = box.column(align=True)
        col.prop(props, "hfov_deg", slider=True)
        col.prop(props, "vfov_deg", slider=True)
        col.separator()
        col.prop(props, "sphere_radius")
        row = col.row(align=True)
        row.prop(props, "segments_u", text="Seg U")
        row.prop(props, "segments_v", text="Seg V")

        box = layout.box()
        box.label(text="5. Image Wrap Control", icon='TEXTURE')
        box.prop(props, "wrap_extension_mode")
        col = box.column(align=True)
        col.prop(props, "wrap_offset_u", slider=True)
        col.prop(props, "wrap_offset_v", slider=True)
        col.separator()
        col.prop(props, "wrap_scale_u", slider=True)
        col.prop(props, "wrap_scale_v", slider=True)


classes = (
    PANO_OT_InstallOpenCV,
    PANO_OT_AnalyzeFolderCameras,
    PANO_OT_TogglePreviewShells,
    PANO_OT_PaintCameraVertexMask,
    PANO_OT_ClearCameraVertexMask,
    PANO_OT_SyncFrom3DShells,
    PANO_OT_ResetCameraTweaks,
    PANO_OT_BakeFinalPanorama,
    PanoCameraItem,
    PANO_UL_CameraList,
    PanoStitchProperties,
    PANO_PT_MainPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.pano_stitch_props = bpy.props.PointerProperty(type=PanoStitchProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.pano_stitch_props

if __name__ == "__main__":
    register()