bl_info = {
    "name": "OpenCV True-FOV Pano Stitch, Manual Align & Bake",
    "author": "Blender Addon Developer",
    "version": (1, 2, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar (N) > PanoStitch",
    "description": "Spherical camera bundle adjustment, manual per-camera alignment, and multi-band + inpaint baking",
    "category": "3D View",
}

import bpy
import bmesh
import os
import sys
import subprocess
import math
from mathutils import Euler, Matrix

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
# 2. ROTATION MATRIX <-> EULER HELPERS (FOR MANUAL CAMERA TWEAKING)
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


# =====================================================================
# 3. STEP 1: AUTO-ANALYZE & SOLVE CAMERA POSES FROM FOLDER
# =====================================================================
def solve_cameras_from_folder(folder_path, conf_thresh=0.8, work_megapix=0.6):
    """
    Finds all stitchable cameras in folder, runs Homography + Ray Bundle Adjustment
    + Wave Correction, and returns calibrated camera intrinsics/extrinsics for manual editing or baking.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
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
        img = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if img is None:
            continue
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
        raise RuntimeError(
            "Fewer than 2 cameras matched. Lower 'Camera Match Confidence' or verify image overlap."
        )

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


# =====================================================================
# 4. STEP 2: BAKE ALIGNED CAMERAS INTO SEAMLESS PANORAMA
# =====================================================================
def bake_aligned_cameras_to_panorama(
    cam_items,
    seam_megapix=0.1,
    blend_strength=5.0,
    inpaint_radius=7,
    inpaint_method='TELEA',
    heal_seams=True
):
    """
    Takes the camera collection (including any user manual Yaw/Pitch/Roll/Focal/Exposure tweaks),
    warps each enabled camera onto the sphere, runs Graph-Cut seam finding, Multi-Band Laplacian
    blending, and OpenCV Inpainting, and returns the final baked image + exact physical HFOV/VFOV.
    """
    active_items = [item for item in cam_items if item.enabled and os.path.isfile(item.filepath)]
    if len(active_items) < 2:
        raise ValueError("At least 2 enabled cameras are required to bake a stitched panorama.")

    full_img_list = []
    seam_images = []
    seam_scales = []
    R_matrices = []
    focals_full = []
    ppx_list = []
    ppy_list = []
    exposure_gains = []

    for item in active_items:
        img = cv2.imread(item.filepath, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Ensure strictly 3-channel BGR (fixes any 4-channel PNG mismatch)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        h, w = img.shape[:2]
        full_img_list.append(img)

        s_scale = min(1.0, math.sqrt(seam_megapix * 1e6 / (w * h)))
        seam_img = cv2.resize(img, (0, 0), fx=s_scale, fy=s_scale, interpolation=cv2.INTER_LINEAR_EXACT)
        seam_images.append(seam_img)
        seam_scales.append(s_scale)

        R_base = np.array(item.r_matrix_flat, dtype=np.float32).reshape((3, 3))
        R_final = apply_ypr_delta_to_rmat(
            R_base,
            dyaw_deg=item.manual_yaw,
            dpitch_deg=item.manual_pitch,
            droll_deg=item.manual_roll
        )
        R_matrices.append(R_final)

        eff_focal = max(50.0, item.focal_px * item.manual_scale)
        focals_full.append(eff_focal)
        ppx_list.append(w * 0.5 + item.shift_x_px)
        ppy_list.append(h * 0.5 + item.shift_y_px)
        exposure_gains.append(item.exposure_gain)

    warped_image_scale = float(np.median(focals_full))

    # 1. Warp at Seam Resolution for Exposure Compensation & Graph-Cut Seam Finding
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

        mask = np.full((s_img.shape[0], s_img.shape[1]), 255, np.uint8)
        _, mask_wp = warper_seam.warp(mask, K_seam, R_matrices[idx], cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        masks_warped_seam.append(mask_wp)

    compensator = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS)
    compensator.feed(
        corners=corners_seam,
        images=[im.astype(np.uint8) for im in images_warped_seam_f],
        masks=masks_warped_seam
    )

    try:
        seam_finder = cv2.detail_GraphCutSeamFinder('COST_COLOR')
        masks_warped_seam = seam_finder.find(images_warped_seam_f, corners_seam, masks_warped_seam)
    except Exception:
        seam_finder = cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_VORONOI_SEAM)
        masks_warped_seam = seam_finder.find(images_warped_seam_f, corners_seam, masks_warped_seam)

    # 2. Full-Resolution Spherical Warp & Multi-Band Blending
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

        mask = np.full((full_img.shape[0], full_img.shape[1]), 255, np.uint8)
        _, mask_wp = warper_full.warp(mask, K_full, R_matrices[idx], cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)

        compensator.apply(idx, corner, img_wp, mask_wp)

        # Apply manual per-camera exposure multiplier if tweaked
        if abs(exposure_gains[idx] - 1.0) > 1e-3:
            img_wp = np.clip(img_wp.astype(np.float32) * exposure_gains[idx], 0, 255).astype(np.uint8)

        dilated_seam = cv2.dilate(masks_warped_seam[idx], None)
        resized_seam = cv2.resize(dilated_seam, size, 0, 0, cv2.INTER_LINEAR_EXACT)
        final_mask = cv2.bitwise_and(resized_seam, mask_wp)

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
        # FIXED: Directly slice 3 BGR channels as int16 (no invalid cv2.COLOR_BGR2BGR)
        img_s16 = img_wp[:, :, :3].astype(np.int16)
        blender.feed(img_s16, mask_wp, corner)

    result_s16, result_mask = blender.blend(None, None)
    stitched = np.clip(result_s16, 0, 255).astype(np.uint8)

    coords = cv2.findNonZero(result_mask)
    if coords is not None:
        x, y, w_valid, h_valid = cv2.boundingRect(coords)
        stitched = stitched[y:y+h_valid, x:x+w_valid]
        result_mask = result_mask[y:y+h_valid, x:x+w_valid]
    else:
        h_valid, w_valid = stitched.shape[:2]

    # Exact physical FOV (1° to 360° horizontally, 1° to 180° vertically)
    true_hfov_deg = round(max(1.0, min(360.0, math.degrees(w_valid / warped_image_scale))), 2)
    true_vfov_deg = round(max(1.0, min(180.0, math.degrees(h_valid / warped_image_scale))), 2)

    # Inpainting / Healbrush across unprojected borders and seam micro-dropouts
    if heal_seams and inpaint_radius > 0:
        missing_mask = cv2.bitwise_not(result_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        inpaint_mask = cv2.dilate(missing_mask, kernel, iterations=2)

        algo = cv2.INPAINT_TELEA if inpaint_method == 'TELEA' else cv2.INPAINT_NS
        stitched = cv2.inpaint(stitched, inpaint_mask, inpaintRadius=float(inpaint_radius), flags=algo)

    return stitched, true_hfov_deg, true_vfov_deg, len(active_items)


# =====================================================================
# 5. 3D SPHERE & LIVE MANUAL ALIGNMENT SHELLS GENERATOR
# =====================================================================
def create_or_update_pano_sphere(context, name, radius, hfov_deg, vfov_deg, seg_u, seg_v):
    """Generates a spherical patch or full sphere spanning exact HFOV (1-360°) and VFOV (1-180°)."""
    mesh_name = f"{name}_Mesh"
    obj_name = name

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

    hfov_rad = math.radians(max(1.0, min(360.0, hfov_deg)))
    vfov_rad = math.radians(max(1.0, min(180.0, vfov_deg)))

    actual_seg_u = max(8, int(seg_u * (hfov_deg / 360.0))) if hfov_deg < 360.0 else seg_u
    actual_seg_v = max(8, int(seg_v * (vfov_deg / 180.0))) if vfov_deg < 180.0 else seg_v

    theta_start = -hfov_rad * 0.5
    phi_start = -vfov_rad * 0.5

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
    return obj


def spawn_or_update_camera_preview_shells(context):
    """
    Creates/updates individual 3D spherical sub-patches in a 'Pano_Manual_Align_Preview'
    collection so users can visually align individual cameras in the 3D Viewport before baking.
    """
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

        # Slight radius offset per shell to prevent Z-fighting while aligning overlapping images
        shell_radius = base_radius - (idx * 0.015)

        obj = create_or_update_pano_sphere(
            context,
            name=shell_name,
            radius=shell_radius,
            hfov_deg=cam_hfov,
            vfov_deg=cam_vfov,
            seg_u=64,
            seg_v=32
        )

        # Ensure object is inside the preview collection
        for c in obj.users_collection:
            c.objects.unlink(obj)
        preview_coll.objects.link(obj)
        obj.hide_viewport = False

        # Orient shell according to Base + Manual Yaw/Pitch/Roll
        total_yaw = math.radians(-(item.base_yaw + item.manual_yaw))
        total_pitch = math.radians(item.base_pitch + item.manual_pitch)
        total_roll = math.radians(item.base_roll + item.manual_roll)
        obj.rotation_euler = Euler((total_pitch, total_roll, total_yaw), 'XYZ')

        # Assign feathered semi-transparent emission shader for easy visual alignment
        mat_name = f"{shell_name}_Mat"
        mat = bpy.data.materials.get(mat_name)
        if not mat:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            mat.blend_method = 'BLEND' if hasattr(mat, "blend_method") else 'HASHED'

            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()

            tex_node = nodes.new(type='ShaderNodeTexImage')
            tex_node.name = "Cam_Tex"
            if os.path.isfile(item.filepath):
                tex_node.image = bpy.data.images.load(item.filepath, check_existing=True)

            emis = nodes.new(type='ShaderNodeEmission')
            emis.name = "Cam_Emis"
            emis.inputs['Strength'].default_value = item.exposure_gain

            transp = nodes.new(type='ShaderNodeBsdfTransparent')
            mix_shader = nodes.new(type='ShaderNodeMixShader')
            mix_shader.name = "Cam_Opacity"
            mix_shader.inputs['Fac'].default_value = 0.85

            out = nodes.new(type='ShaderNodeOutputMaterial')

            links.new(tex_node.outputs['Color'], emis.inputs['Color'])
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
    """Creates an Emission Shader with interactive Mapping & Wrap controls for the baked sphere."""
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


def update_live_projection(self, context):
    """Callback for live adjustment of Sphere FOV (1-360 x 1-180) and UV Wrap properties."""
    props = context.scene.pano_stitch_props
    obj = bpy.data.objects.get("Stitched_Pano_Sphere")
    if not obj:
        return

    create_or_update_pano_sphere(
        context,
        name="Stitched_Pano_Sphere",
        radius=props.sphere_radius,
        hfov_deg=props.hfov_deg,
        vfov_deg=props.vfov_deg,
        seg_u=props.segments_u,
        seg_v=props.segments_v
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
    """Live callback when user scrubs manual Yaw/Pitch/Roll/Scale on an individual camera."""
    props = context.scene.pano_stitch_props
    if props.show_3d_align_shells:
        spawn_or_update_camera_preview_shells(context)


# =====================================================================
# 6. OPERATORS: ANALYZE, PREVIEW SHELLS, SYNC FROM 3D, AND BAKE
# =====================================================================
class PANO_OT_AnalyzeFolderCameras(bpy.types.Operator):
    """Filter stitchable cameras in folder and compute initial bundle-adjusted orientations"""
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

        self.report({'INFO'}, f"Matched & aligned {len(solved)} stitchable cameras! Ready to tweak or bake.")
        return {'FINISHED'}


class PANO_OT_TogglePreviewShells(bpy.types.Operator):
    """Spawn or hide individual 3D spherical camera shells for interactive manual alignment"""
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
            self.report({'INFO'}, "3D Camera Shells visible. Adjust sliders or rotate shells in 3D view.")
        else:
            if coll_name in bpy.data.collections:
                bpy.data.collections[coll_name].hide_viewport = True
            self.report({'INFO'}, "3D Camera Shells hidden.")

        return {'FINISHED'}


class PANO_OT_SyncFrom3DShells(bpy.types.Operator):
    """Read rotations directly from the 3D CamPatch objects if you rotated them manually in the viewport"""
    bl_idname = "pano.sync_from_3d_shells"
    bl_label = "Sync Rotations from 3D Viewport"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.pano_stitch_props
        synced = 0
        for idx, item in enumerate(props.camera_items):
            shell_name = f"CamPatch_{idx:02d}_{item.filename}"
            obj = bpy.data.objects.get(shell_name)
            if obj:
                rot = obj.rotation_euler
                total_yaw_deg = -math.degrees(rot.z)
                total_pitch_deg = math.degrees(rot.x)
                total_roll_deg = math.degrees(rot.y)

                item.manual_yaw = total_yaw_deg - item.base_yaw
                item.manual_pitch = total_pitch_deg - item.base_pitch
                item.manual_roll = total_roll_deg - item.base_roll
                synced += 1

        self.report({'INFO'}, f"Synced manual 3D viewport rotations for {synced} cameras.")
        return {'FINISHED'}


class PANO_OT_ResetCameraTweaks(bpy.types.Operator):
    """Reset manual Yaw/Pitch/Roll/Scale offsets for the selected camera back to OpenCV auto-solve"""
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
    """Bake all enabled cameras (with manual adjustments) into a seamless inpainted panorama on the sphere"""
    bl_idname = "pano.bake_final_panorama"
    bl_label = "2. Bake Final Seamless Panorama"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not CV2_AVAILABLE:
            self.report({'ERROR'}, "OpenCV is not installed. Click 'Install OpenCV' first.")
            return {'CANCELLED'}

        props = context.scene.pano_stitch_props
        folder = bpy.path.abspath(props.input_folder)

        # If user hasn't analyzed yet, auto-analyze first
        if len(props.camera_items) == 0:
            res = bpy.ops.pano.analyze_cameras()
            if 'FINISHED' not in res:
                return {'CANCELLED'}

        try:
            stitched_bgr, true_hfov, true_vfov, used_cams = bake_aligned_cameras_to_panorama(
                cam_items=props.camera_items,
                blend_strength=props.multiband_strength,
                inpaint_radius=props.inpaint_radius,
                inpaint_method=props.inpaint_method,
                heal_seams=props.enable_inpaint_heal
            )
        except Exception as e:
            self.report({'ERROR'}, f"Bake failed: {str(e)}")
            return {'CANCELLED'}

        out_path = os.path.join(folder, "stitched_panorama_baked.png")
        cv2.imwrite(out_path, stitched_bgr)

        bl_img = bpy.data.images.load(out_path, check_existing=True)
        bl_img.reload()

        if props.use_measured_fov:
            props.hfov_deg = true_hfov
            props.vfov_deg = true_vfov

        sphere_obj = create_or_update_pano_sphere(
            context,
            name="Stitched_Pano_Sphere",
            radius=props.sphere_radius,
            hfov_deg=props.hfov_deg,
            vfov_deg=props.vfov_deg,
            seg_u=props.segments_u,
            seg_v=props.segments_v
        )

        setup_pano_shader(sphere_obj, bl_img, props)

        # Hide 3D preview shells so the baked sphere is cleanly visible
        coll_name = "Pano_Manual_Align_Preview"
        if coll_name in bpy.data.collections:
            bpy.data.collections[coll_name].hide_viewport = True
        props.show_3d_align_shells = False

        bpy.ops.object.select_all(action='DESELECT')
        sphere_obj.select_set(True)
        context.view_layer.objects.active = sphere_obj

        self.report(
            {'INFO'},
            f"Baked {used_cams} cameras to '{out_path}'! Exact Span: {props.hfov_deg:.1f}° x {props.vfov_deg:.1f}°"
        )
        return {'FINISHED'}


# =====================================================================
# 7. DATA STRUCTURES & UI LIST FOR PER-CAMERA MANUAL ALIGNMENT
# =====================================================================
class PanoCameraItem(bpy.types.PropertyGroup):
    filename: bpy.props.StringProperty(name="File")
    filepath: bpy.props.StringProperty(name="Path")
    enabled: bpy.props.BoolProperty(
        name="Include",
        description="Include this camera in the panorama bake",
        default=True,
        update=update_manual_camera_transform
    )
    width: bpy.props.IntProperty(name="Width", default=1920)
    height: bpy.props.IntProperty(name="Height", default=1080)
    focal_px: bpy.props.FloatProperty(name="Solved Focal (px)", default=1000.0)
    r_matrix_flat: bpy.props.FloatVectorProperty(name="R Matrix", size=9, default=(1, 0, 0, 0, 1, 0, 0, 0, 1))

    base_yaw: bpy.props.FloatProperty(name="Base Yaw", default=0.0)
    base_pitch: bpy.props.FloatProperty(name="Base Pitch", default=0.0)
    base_roll: bpy.props.FloatProperty(name="Base Roll", default=0.0)

    # Manual alignment offsets
    manual_yaw: bpy.props.FloatProperty(
        name="Yaw Offset (°)",
        description="Manually nudge horizontal angle of this image in degrees",
        default=0.0, min=-180.0, max=180.0,
        update=update_manual_camera_transform
    )
    manual_pitch: bpy.props.FloatProperty(
        name="Pitch Offset (°)",
        description="Manually nudge vertical angle of this image in degrees",
        default=0.0, min=-90.0, max=90.0,
        update=update_manual_camera_transform
    )
    manual_roll: bpy.props.FloatProperty(
        name="Roll Offset (°)",
        description="Manually nudge rotation/tilt of this image in degrees",
        default=0.0, min=-180.0, max=180.0,
        update=update_manual_camera_transform
    )
    manual_scale: bpy.props.FloatProperty(
        name="FOV / Focal Scale",
        description="Manually scale focal length to fix lens zoom mismatch",
        default=1.0, min=0.2, max=5.0,
        update=update_manual_camera_transform
    )
    shift_x_px: bpy.props.FloatProperty(
        name="Principal Point X Offset (px)",
        description="Fine horizontal optical center shift in pixels",
        default=0.0, min=-2000.0, max=2000.0
    )
    shift_y_px: bpy.props.FloatProperty(
        name="Principal Point Y Offset (px)",
        description="Fine vertical optical center shift in pixels",
        default=0.0, min=-2000.0, max=2000.0
    )
    exposure_gain: bpy.props.FloatProperty(
        name="Exposure Multiplier",
        description="Manually brighten or darken this image before multi-band blending",
        default=1.0, min=0.1, max=5.0,
        update=update_manual_camera_transform
    )


class PANO_UL_CameraList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        row.label(text=item.filename, icon='CAMERA_DATA')
        sub = row.row(align=True)
        sub.alignment = 'RIGHT'
        sub.label(text=f"Y:{item.base_yaw + item.manual_yaw:.1f}° P:{item.base_pitch + item.manual_pitch:.1f}°")


class PanoStitchProperties(bpy.types.PropertyGroup):
    input_folder: bpy.props.StringProperty(
        name="Image Folder",
        description="Folder containing camera images to automatically filter and stitch",
        default="",
        subtype='DIR_PATH'
    )
    confidence_thresh: bpy.props.FloatProperty(
        name="Camera Match Confidence",
        description="Confidence threshold for filtering stitchable cameras via Bundle Adjustment",
        default=0.8, min=0.1, max=2.0
    )
    camera_items: bpy.props.CollectionProperty(type=PanoCameraItem)
    active_camera_index: bpy.props.IntProperty(name="Active Camera", default=0)
    show_3d_align_shells: bpy.props.BoolProperty(name="Show 3D Preview Shells", default=False)

    multiband_strength: bpy.props.FloatProperty(
        name="Multi-Band Seam Blend (%)",
        description="Width of Laplacian pyramid frequency blending across camera seams",
        default=5.0, min=0.0, max=50.0
    )
    enable_inpaint_heal: bpy.props.BoolProperty(
        name="Inpaint & Heal Seams/Borders",
        description="Fill jagged warp borders and seam micro-gaps using OpenCV inpainting",
        default=True
    )
    inpaint_method: bpy.props.EnumProperty(
        name="Heal Algorithm",
        items=[
            ('TELEA', "Fast Marching (Telea)", "Fast radial healbrush-style inpainting"),
            ('NS', "Navier-Stokes (Fluid)", "Fluid dynamics inpainting that preserves continuous edges")
        ],
        default='TELEA'
    )
    inpaint_radius: bpy.props.IntProperty(
        name="Heal Brush Radius (px)",
        description="Pixel radius for OpenCV inpainting around seams and missing borders",
        default=7, min=1, max=50
    )
    use_measured_fov: bpy.props.BoolProperty(
        name="Use Measured Camera FOV",
        description="Set sphere arc to the exact physical FOV (1-360° x 1-180°) measured from OpenCV focal length",
        default=True
    )
    hfov_deg: bpy.props.FloatProperty(
        name="Horizontal Span (°)",
        description="Exact horizontal angular width of the panorama (1° to 360°)",
        default=120.0, min=1.0, max=360.0,
        update=update_live_projection
    )
    vfov_deg: bpy.props.FloatProperty(
        name="Vertical Span (°)",
        description="Exact vertical angular height of the panorama (1° to 180°)",
        default=65.0, min=1.0, max=180.0,
        update=update_live_projection
    )
    sphere_radius: bpy.props.FloatProperty(
        name="Sphere Radius",
        default=10.0, min=0.1, max=1000.0,
        update=update_live_projection
    )
    segments_u: bpy.props.IntProperty(
        name="Max Full-Sphere Segs U",
        default=128, min=16, max=512,
        update=update_live_projection
    )
    segments_v: bpy.props.IntProperty(
        name="Max Full-Sphere Segs V",
        default=64, min=8, max=256,
        update=update_live_projection
    )
    wrap_extension_mode: bpy.props.EnumProperty(
        name="Wrap Mode",
        items=[
            ('EXTEND', "Extend (Clamp Edges)", "Clamp edge pixels without repeating"),
            ('MIRROR', "Mirror (Seamless Edge Reflect)", "Mirror image at boundaries"),
            ('REPEAT', "Repeat (Tile)", "Tile texture across boundaries"),
            ('CLIP', "Clip (Cut Outside 0-1)", "Clip outside normalized UV space")
        ],
        default='EXTEND',
        update=update_live_projection
    )
    wrap_offset_u: bpy.props.FloatProperty(
        name="Wrap Offset U", default=0.0, min=-2.0, max=2.0, update=update_live_projection
    )
    wrap_offset_v: bpy.props.FloatProperty(
        name="Wrap Offset V", default=0.0, min=-2.0, max=2.0, update=update_live_projection
    )
    wrap_scale_u: bpy.props.FloatProperty(
        name="Wrap Scale U", default=1.0, min=0.05, max=10.0, update=update_live_projection
    )
    wrap_scale_v: bpy.props.FloatProperty(
        name="Wrap Scale V", default=1.0, min=0.05, max=10.0, update=update_live_projection
    )


# =====================================================================
# 8. SIDEBAR UI PANEL
# =====================================================================
class PANO_PT_MainPanel(bpy.types.Panel):
    bl_label = "OpenCV Pano Align & Bake"
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

        # 1. Folder & Auto-Analysis
        box = layout.box()
        box.label(text="1. Folder & Camera Filtering", icon='FILE_FOLDER')
        box.prop(props, "input_folder", text="")
        box.prop(props, "confidence_thresh", slider=True)
        row = box.row(align=True)
        row.scale_y = 1.3
        row.operator("pano.analyze_cameras", icon='VIEWZOOM')

        # 2. Manual Per-Camera Alignment
        box = layout.box()
        box.label(text="2. Manual Image Alignment", icon='ORIENTATION_GIMBAL')
        if len(props.camera_items) > 0:
            row = box.row(align=True)
            row.operator(
                "pano.toggle_preview_shells",
                text="Hide 3D Align Shells" if props.show_3d_align_shells else "Show 3D Align Shells",
                icon='HIDE_OFF' if props.show_3d_align_shells else 'HIDE_ON'
            )
            if props.show_3d_align_shells:
                row.operator("pano.sync_from_3d_shells", text="Sync 3D", icon='FILE_REFRESH')

            box.template_list(
                "PANO_UL_CameraList", "",
                props, "camera_items",
                props, "active_camera_index",
                rows=4
            )

            idx = props.active_camera_index
            if 0 <= idx < len(props.camera_items):
                cam = props.camera_items[idx]
                col = box.column(align=True)
                col.label(text=f"Adjust: {cam.filename} (Focal: {cam.focal_px * cam.manual_scale:.0f}px)")
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
            box.label(text="Click '1. Analyze' to load stitchable cameras.", icon='INFO')

        # 3. Seam Blending, Inpainting & Final Bake
        box = layout.box()
        box.label(text="3. Seam Healing & Final Bake", icon='BRUSH_DATA')
        box.prop(props, "multiband_strength", slider=True)
        box.prop(props, "enable_inpaint_heal")
        if props.enable_inpaint_heal:
            box.prop(props, "inpaint_method", text="Mode")
            box.prop(props, "inpaint_radius", slider=True)

        row = box.row(align=True)
        row.scale_y = 1.6
        row.operator("pano.bake_final_panorama", icon='RENDER_STILL')

        # 4. Exact Angular Span (1-360° x 1-180°)
        box = layout.box()
        box.label(text="4. Baked Sphere Span (1-360° x 1-180°)", icon='MESH_UVSPHERE')
        box.prop(props, "use_measured_fov")
        col = box.column(align=True)
        col.prop(props, "hfov_deg", slider=True)
        col.prop(props, "vfov_deg", slider=True)
        col.separator()
        col.prop(props, "sphere_radius")
        row = col.row(align=True)
        row.prop(props, "segments_u", text="Density U")
        row.prop(props, "segments_v", text="Density V")

        # 5. Image Wrap Control
        box = layout.box()
        box.label(text="5. Image Wrap Control", icon='TEXTURE')
        box.prop(props, "wrap_extension_mode")
        col = box.column(align=True)
        col.prop(props, "wrap_offset_u", slider=True)
        col.prop(props, "wrap_offset_v", slider=True)
        col.separator()
        col.prop(props, "wrap_scale_u", slider=True)
        col.prop(props, "wrap_scale_v", slider=True)


# =====================================================================
# REGISTRATION
# =====================================================================
classes = (
    PANO_OT_InstallOpenCV,
    PANO_OT_AnalyzeFolderCameras,
    PANO_OT_TogglePreviewShells,
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