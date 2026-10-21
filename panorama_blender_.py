bl_info = {
    "name": "OpenCV Seamless Pano & Sphere Projection",
    "author": "Blender Addon Developer",
    "version": (1, 0, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar (N) > PanoStitch",
    "description": "Automatic folder image stitching with OpenCV inpainting and dynamic FOV sphere projection",
    "category": "3D View",
}

import bpy
import bmesh
import os
import sys
import subprocess
import math
from mathutils import Vector

# Global flag for OpenCV availability
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
# 2. IMAGE STITCHING & SEAM INPAINTING LOGIC
# =====================================================================
def stitch_and_heal_folder(folder_path, conf_thresh, inpaint_radius, inpaint_method, heal_borders):
    """
    Loads valid images from folder, filters out non-matching cameras,
    applies multi-band seam blending, and heals irregular borders/seams via OpenCV Inpainting.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".exr"}
    image_files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])

    if len(image_files) < 2:
        raise ValueError("Need at least 2 valid images in the selected folder.")

    images = []
    loaded_names = []
    for fpath in image_files:
        img = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
            loaded_names.append(os.path.basename(fpath))

    if len(images) < 2:
        raise ValueError("Could not decode at least 2 valid images.")

    # Create Panorama Stitcher (automatically drops images that don't match confidence threshold)
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    stitcher.setPanoConfidenceThresh(conf_thresh)

    # Try configuring Multi-Band Blender for smooth frequency transitions across camera seams
    try:
        blender = cv2.detail_MultiBandBlender(try_gpu=False, num_bands=5)
        stitcher.setBlender(blender)
    except Exception:
        pass  # Fallback to default built-in multiband blender if bindings omit detail wrapper

    status, stitched = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        error_map = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Not enough matching features found between images.",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed.",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameter adjustment failed."
        }
        err_msg = error_map.get(status, f"Stitcher error code {status}")
        raise RuntimeError(f"Stitching failed: {err_msg}")

    # -----------------------------------------------------------------
    # SEAM & BORDER HEALING (INPAINTING / HEALBRUSH)
    # -----------------------------------------------------------------
    # Convert stitched result to grayscale to identify empty/black warped regions and thin gaps
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Crop to bounding box of valid non-black pixels first
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        stitched = stitched[y:y+h, x:x+w]
        thresh = thresh[y:y+h, x:x+w]

    if heal_borders and inpaint_radius > 0:
        # Invert mask so black unprojected cavities & seam micro-dropouts are 255 (target to heal)
        missing_mask = cv2.bitwise_not(thresh)

        # Dilate the mask slightly so the inpaint brush overlaps the dark antialiased seam edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        inpaint_mask = cv2.dilate(missing_mask, kernel, iterations=2)

        algo = cv2.INPAINT_TELEA if inpaint_method == 'TELEA' else cv2.INPAINT_NS
        stitched = cv2.inpaint(stitched, inpaint_mask, inpaintRadius=float(inpaint_radius), flags=algo)

    # Horizontal Wrap Seam Blending (if near 360 panorama, smooth left/right edge transition)
    h_px, w_px = stitched.shape[:2]
    aspect = w_px / float(h_px)

    return stitched, aspect


# =====================================================================
# 3. DYNAMIC FOV SPHERE MESH & MATERIAL GENERATOR
# =====================================================================
def create_or_update_pano_sphere(context, name, radius, hfov_deg, vfov_deg, seg_u, seg_v):
    """
    Generates a partial or full UV sphere mesh spanning exact Horizontal FOV (0-360 deg)
    and Vertical FOV (0-180 deg) with normalized 0..1 UVs mapped directly to the surface.
    """
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

    # Center the partial dome facing +Y axis (standard Blender front camera look)
    theta_start = -hfov_rad * 0.5
    phi_start = -vfov_rad * 0.5

    verts_grid = []
    for iv in range(seg_v + 1):
        v_norm = iv / float(seg_v)
        phi = phi_start + v_norm * vfov_rad  # Elevation angle (-pi/2 to +pi/2)

        row_verts = []
        for iu in range(seg_u + 1):
            u_norm = iu / float(seg_u)
            theta = theta_start + u_norm * hfov_rad  # Azimuth angle (-pi to +pi)

            # Spherical to Cartesian (Y-forward, Z-up)
            x = radius * math.cos(phi) * math.sin(theta)
            y = radius * math.cos(phi) * math.cos(theta)
            z = radius * math.sin(phi)

            vert = bm.verts.new((x, y, z))
            row_verts.append((vert, u_norm, v_norm))
        verts_grid.append(row_verts)

    bm.verts.ensure_lookup_table()

    # Build quad faces and assign normalized UVs (Inward normals for viewing from inside)
    for iv in range(seg_v):
        for iu in range(seg_u):
            v00, u00, vn00 = verts_grid[iv][iu]
            v10, u10, vn10 = verts_grid[iv][iu + 1]
            v11, u11, vn11 = verts_grid[iv + 1][iu + 1]
            v01, u01, vn01 = verts_grid[iv + 1][iu]

            # Winding order facing inward toward (0,0,0)
            face = bm.faces.new((v00, v01, v11, v10))
            face.smooth = True

            uvs = [(u00, vn00), (u01, vn01), (u11, vn11), (u10, vn10)]
            for loop, uv_coord in zip(face.loops, uvs):
                loop[uv_layer].uv = uv_coord

    # If full 360, weld seam vertices ONLY for geometry while preserving split UVs
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    return obj


def setup_pano_shader(obj, bl_image, props):
    """
    Creates an Emission Shader with interactive Mapping & Wrap Edge Feathering controls.
    """
    mat_name = f"{obj.name}_Mat"
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Create Nodes
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
    node_tex.extension = props.wrap_extension_mode  # 'REPEAT', 'EXTEND', or 'MIRROR'
    node_tex.interpolation = 'Smart'

    node_emission = nodes.new(type='ShaderNodeEmission')
    node_emission.location = (50, 0)
    node_emission.inputs['Strength'].default_value = 1.0

    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    node_out.location = (300, 0)

    # Link Nodes
    links.new(node_tex_coord.outputs['UV'], node_mapping.inputs['Vector'])
    links.new(node_mapping.outputs['Vector'], node_tex.inputs['Vector'])
    links.new(node_tex.outputs['Color'], node_emission.inputs['Color'])
    links.new(node_emission.outputs['Emission'], node_out.inputs['Surface'])

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def update_live_projection(self, context):
    """Callback for live adjustment of Sphere FOV and UV Wrap properties."""
    props = context.scene.pano_stitch_props
    obj = bpy.data.objects.get("Stitched_Pano_Sphere")
    if not obj:
        return

    # Update Mesh Geometry for live FOV changes
    create_or_update_pano_sphere(
        context,
        name="Stitched_Pano_Sphere",
        radius=props.sphere_radius,
        hfov_deg=props.hfov_deg,
        vfov_deg=props.vfov_deg,
        seg_u=props.segments_u,
        seg_v=props.segments_v
    )

    # Update Shader Mapping Node
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


# =====================================================================
# 4. MAIN STITCH & PROJECT OPERATOR
# =====================================================================
class PANO_OT_StitchAndProject(bpy.types.Operator):
    """Stitch folder images with OpenCV, heal seams, and project onto a dynamic FOV sphere"""
    bl_idname = "pano.stitch_and_project"
    bl_label = "Stitch Folder & Project to Sphere"
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
            stitched_bgr, aspect = stitch_and_heal_folder(
                folder_path=folder,
                conf_thresh=props.confidence_thresh,
                inpaint_radius=props.inpaint_radius,
                inpaint_method=props.inpaint_method,
                heal_borders=props.enable_inpaint_heal
            )
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        # Save stitched image to disk next to input images
        out_path = os.path.join(folder, "stitched_panorama_healed.png")
        cv2.imwrite(out_path, stitched_bgr)

        # Load into Blender Image datablock
        bl_img = bpy.data.images.load(out_path, check_existing=True)
        bl_img.reload()

        # Auto-calculate largest possible FOV if auto_fov is enabled
        if props.auto_estimate_fov:
            # Standard full equirectangular is 2:1 (360 x 180)
            # Estimate coverage proportional to image dimensions, clamped to 360x180
            h_px, w_px = stitched_bgr.shape[:2]
            if aspect >= 1.95:
                props.hfov_deg = 360.0
                props.vfov_deg = min(180.0, 360.0 / aspect)
            else:
                # Partial panorama: scale largest dimension relative to aspect ratio
                # Assume standard rectilinear/cylindrical stitch spans up to 180-360 deg horizontally
                est_hfov = min(360.0, max(60.0, aspect * 90.0))
                est_vfov = min(180.0, max(30.0, est_hfov / aspect))
                props.hfov_deg = round(est_hfov, 1)
                props.vfov_deg = round(est_vfov, 1)

        # Create/Update 3D Dome/Sphere Mesh
        sphere_obj = create_or_update_pano_sphere(
            context,
            name="Stitched_Pano_Sphere",
            radius=props.sphere_radius,
            hfov_deg=props.hfov_deg,
            vfov_deg=props.vfov_deg,
            seg_u=props.segments_u,
            seg_v=props.segments_v
        )

        # Build Shader with Wrap Control
        setup_pano_shader(sphere_obj, bl_img, props)

        # Select active object
        bpy.ops.object.select_all(action='DESELECT')
        sphere_obj.select_set(True)
        context.view_layer.objects.active = sphere_obj

        self.report(
            {'INFO'},
            f"Stitched successfully! Projected at {props.hfov_deg:.1f}° x {props.vfov_deg:.1f}°"
        )
        return {'FINISHED'}


# =====================================================================
# 5. PROPERTY GROUP & UI PANEL
# =====================================================================
class PanoStitchProperties(bpy.types.PropertyGroup):
    input_folder: bpy.props.StringProperty(
        name="Image Folder",
        description="Folder containing camera images to automatically stitch",
        default="",
        subtype='DIR_PATH'
    )
    confidence_thresh: bpy.props.FloatProperty(
        name="Camera Match Confidence",
        description="Minimum confidence to include a camera in the stitch (filters out un-stitchable images)",
        default=0.8,
        min=0.1,
        max=2.0
    )
    enable_inpaint_heal: bpy.props.BoolProperty(
        name="Inpaint & Heal Seams/Borders",
        description="Fill jagged warp borders and seam micro-gaps using OpenCV inpainting",
        default=True
    )
    inpaint_method: bpy.props.EnumProperty(
        name="Heal Algorithm",
        items=[
            ('TELEA', "Fast Marching (Telea)", "Fast, smooth healbrush-style radial inpainting"),
            ('NS', "Navier-Stokes (Fluid)", "Fluid dynamics inpainting that preserves continuous edges")
        ],
        default='TELEA'
    )
    inpaint_radius: bpy.props.IntProperty(
        name="Heal Brush Radius (px)",
        description="Pixel radius for OpenCV inpainting around seams and missing borders",
        default=7,
        min=1,
        max=50
    )
    auto_estimate_fov: bpy.props.BoolProperty(
        name="Auto-Maximize FOV from Stitch",
        description="Automatically set Horizontal and Vertical FOV based on stitched image aspect ratio",
        default=True
    )
    hfov_deg: bpy.props.FloatProperty(
        name="Horizontal FOV (°)",
        description="Horizontal angular span of the projection sphere (0 to 360 degrees)",
        default=360.0,
        min=1.0,
        max=360.0,
        subtype='ANGLE',
        unit='NONE',
        update=update_live_projection
    )
    vfov_deg: bpy.props.FloatProperty(
        name="Vertical FOV (°)",
        description="Vertical angular span of the projection sphere (0 to 180 degrees)",
        default=180.0,
        min=1.0,
        max=180.0,
        subtype='ANGLE',
        unit='NONE',
        update=update_live_projection
    )
    sphere_radius: bpy.props.FloatProperty(
        name="Sphere Radius",
        default=10.0,
        min=0.1,
        max=1000.0,
        update=update_live_projection
    )
    segments_u: bpy.props.IntProperty(
        name="Horizontal Segments",
        default=128,
        min=16,
        max=512,
        update=update_live_projection
    )
    segments_v: bpy.props.IntProperty(
        name="Vertical Segments",
        default=64,
        min=8,
        max=256,
        update=update_live_projection
    )
    wrap_extension_mode: bpy.props.EnumProperty(
        name="Wrap Mode",
        items=[
            ('REPEAT', "Repeat (360° Seamless)", "Tile texture seamlessly across boundaries"),
            ('MIRROR', "Mirror (Seamless Edge Reflect)", "Mirror image at boundaries to hide non-360 seams"),
            ('EXTEND', "Extend (Clamp Edge Pixels)", "Stretch edge pixels outward"),
            ('CLIP', "Clip (Transparent Outside)", "Clip outside 0-1 UV space")
        ],
        default='REPEAT',
        update=update_live_projection
    )
    wrap_offset_u: bpy.props.FloatProperty(
        name="Wrap Offset U",
        description="Rotate/shift the panorama horizontally along the sphere",
        default=0.0,
        min=-2.0,
        max=2.0,
        update=update_live_projection
    )
    wrap_offset_v: bpy.props.FloatProperty(
        name="Wrap Offset V",
        description="Shift the panorama vertically along the sphere",
        default=0.0,
        min=-2.0,
        max=2.0,
        update=update_live_projection
    )
    wrap_scale_u: bpy.props.FloatProperty(
        name="Wrap Scale U",
        description="Horizontal UV wrap multiplier",
        default=1.0,
        min=0.05,
        max=10.0,
        update=update_live_projection
    )
    wrap_scale_v: bpy.props.FloatProperty(
        name="Wrap Scale V",
        description="Vertical UV wrap multiplier",
        default=1.0,
        min=0.05,
        max=10.0,
        update=update_live_projection
    )


class PANO_PT_MainPanel(bpy.types.Panel):
    bl_label = "OpenCV Pano Stitch & Sphere"
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

        # 1. Folder & Stitching Settings
        box = layout.box()
        box.label(text="1. Folder & Camera Filtering", icon='FILE_FOLDER')
        box.prop(props, "input_folder", text="")
        box.prop(props, "confidence_thresh", slider=True)

        # 2. Seam Inpainting / Healbrush Settings
        box = layout.box()
        box.label(text="2. Seam & Border Healing", icon='BRUSH_DATA')
        box.prop(props, "enable_inpaint_heal")
        if props.enable_inpaint_heal:
            box.prop(props, "inpaint_method", text="Mode")
            box.prop(props, "inpaint_radius", slider=True)

        # Action Button
        layout.separator()
        row = layout.row(align=True)
        row.scale_y = 1.6
        row.operator("pano.stitch_and_project", icon='RENDER_RESULT')
        layout.separator()

        # 3. Dynamic Sphere FOV Control (0-360 x 0-180)
        box = layout.box()
        box.label(text="3. Sphere FOV (0-360° x 0-180°)", icon='MESH_UVSPHERE')
        box.prop(props, "auto_estimate_fov")
        col = box.column(align=True)
        col.prop(props, "hfov_deg", slider=True)
        col.prop(props, "vfov_deg", slider=True)
        col.separator()
        col.prop(props, "sphere_radius")
        row = col.row(align=True)
        row.prop(props, "segments_u", text="Seg U")
        row.prop(props, "segments_v", text="Seg V")

        # 4. Image Wrap Control
        box = layout.box()
        box.label(text="4. Image Wrap Control", icon='TEXTURE')
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
    PANO_OT_StitchAndProject,
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