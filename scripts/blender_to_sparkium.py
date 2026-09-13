#!/usr/bin/env python3
"""Export the current Blender scene to a self-contained Sparkium JSON scene.

Run with Blender, for example:
  blender -b input.blend --python scripts/blender_to_sparkium.py -- output_dir

The exporter preserves evaluated static meshes and collection instances, splits
multi-material meshes, copies packed/external textures, converts Cycles lights,
and writes all paths relative to scene.json. Animation and volumes are outside
the static Sparkium scene v1 format.
"""
import bpy
import json
import math
import os
import re
import shutil
import struct
import sys
from array import array
from collections import defaultdict
from pathlib import Path
from mathutils import Matrix, Vector


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "unnamed"


def find_socket(node, names):
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            return socket
    return None


def scalar(socket, fallback):
    if socket is None or socket.is_linked:
        return fallback
    try:
        return float(socket.default_value)
    except (TypeError, ValueError):
        return fallback


def color(socket, fallback):
    if socket is None or socket.is_linked:
        return list(fallback)
    value = socket.default_value
    return [float(value[0]), float(value[1]), float(value[2])]


def upstream_nodes(socket):
    if socket is None or not socket.is_linked:
        return []
    result, queue, seen = [], [link.from_node for link in socket.links], set()
    while queue:
        node = queue.pop(0)
        if node in seen:
            continue
        seen.add(node)
        result.append(node)
        for item in node.inputs:
            if item.is_linked:
                queue.extend(link.from_node for link in item.links)
    return result


def first_image(socket, require_normal_map=False):
    nodes = upstream_nodes(socket)
    if require_normal_map and not any(n.bl_idname == "ShaderNodeNormalMap" for n in nodes):
        return None
    for node in nodes:
        if node.bl_idname == "ShaderNodeTexImage" and node.image is not None:
            return node.image
    return None


def reachable_surface_nodes(material):
    if not material or not material.node_tree:
        return []
    output = next((n for n in material.node_tree.nodes
                   if n.bl_idname == "ShaderNodeOutputMaterial" and n.is_active_output), None)
    return upstream_nodes(output.inputs.get("Surface") if output else None)


def select_shader(nodes):
    priority = ("ShaderNodeBsdfPrincipled", "ShaderNodeBsdfDiffuse",
                "ShaderNodeBsdfAnisotropic", "ShaderNodeBsdfGlossy",
                "ShaderNodeBsdfGlass", "ShaderNodeEmission", "ShaderNodeGroup")
    for kind in priority:
        found = next((node for node in nodes if node.bl_idname == kind), None)
        if found:
            return found
    return None


def socket_value(socket, fallback=0.0):
    if socket is None:
        return fallback
    value = socket.default_value
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return [float(value[i]) for i in range(min(len(value), 4))]
    except (TypeError, ValueError):
        return fallback


def shader_graph_spec(material, shader, texture_dir, image_cache, warnings, closure_root=None):
    """Translate a Blender value-node DAG feeding one surface shader."""
    nodes, node_ids, unsupported = {}, {}, []

    def ref(socket, fallback=0.0):
        if socket is None:
            return fallback
        if not socket.is_linked:
            return socket_value(socket, fallback)
        link = socket.links[0]
        node_id = emit(link.from_node)
        output = link.from_socket.name.lower().replace(" ", "_")
        return {"node": node_id, "output": output}

    def input_named(node, names, fallback=0.0):
        return ref(find_socket(node, names), fallback)

    def emit(node):
        key = node.as_pointer()
        if key in node_ids:
            return node_ids[key]
        node_id = f"n{len(node_ids)}"
        node_ids[key] = node_id
        kind = node.bl_idname
        spec = None
        if kind == "ShaderNodeTexCoord":
            spec = {"type": "texture_coordinate"}
        elif kind == "ShaderNodeTexImage" and node.image:
            path = write_image(node.image, texture_dir, image_cache, warnings)
            if path:
                spec = {"type": "image_texture", "path": path,
                        "color_space": "linear" if node.image.colorspace_settings.is_data else "srgb",
                        "inputs": {}}
                vector_socket = find_socket(node, ("Vector",))
                # Blender image textures use the active UV map when Vector is
                # unconnected. Omitting it preserves the same runtime default.
                if vector_socket is not None and vector_socket.is_linked:
                    spec["inputs"]["vector"] = ref(vector_socket)
        elif kind == "ShaderNodeMapping":
            spec = {"type": "mapping", "inputs": {
                "vector": input_named(node, ("Vector",), [0, 0, 0]),
                "location": input_named(node, ("Location",), [0, 0, 0]),
                "rotation": input_named(node, ("Rotation",), [0, 0, 0]),
                "scale": input_named(node, ("Scale",), [1, 1, 1])}}
        elif kind == "ShaderNodeTexNoise":
            spec = {"type": "noise_texture", "inputs": {
                "vector": input_named(node, ("Vector",), [0, 0, 0]),
                "scale": input_named(node, ("Scale",), 5.0),
                "detail": input_named(node, ("Detail",), 2.0),
                "roughness": input_named(node, ("Roughness",), 0.5)}}
        elif kind == "ShaderNodeTexVoronoi":
            spec = {"type": "voronoi_texture", "inputs": {
                "vector": input_named(node, ("Vector",), [0, 0, 0]),
                "scale": input_named(node, ("Scale",), 5.0)}}
        elif kind == "ShaderNodeTexGradient":
            spec = {"type": "gradient_texture", "gradient_type": node.gradient_type,
                    "inputs": {"vector": input_named(node, ("Vector",), [0, 0, 0])}}
        elif kind == "ShaderNodeTexWave":
            spec = {"type": "wave_texture", "wave_type": node.wave_type,
                    "bands_direction": node.bands_direction, "rings_direction": node.rings_direction,
                    "wave_profile": node.wave_profile, "inputs": {
                        "vector": input_named(node, ("Vector",), [0, 0, 0]),
                        "scale": input_named(node, ("Scale",), 5.0),
                        "distortion": input_named(node, ("Distortion",), 0.0),
                        "phase": input_named(node, ("Phase Offset",), 0.0)}}
        elif kind == "ShaderNodeTexSky":
            spec = {"type": "sky_texture", "sun_direction": list(node.sun_direction),
                    "turbidity": float(node.turbidity),
                    "inputs": {"vector": input_named(node, ("Vector",), [0, 0, 1])}}
        elif kind == "ShaderNodeAttribute":
            spec = {"type": "vertex_attribute", "name": node.attribute_name}
        elif kind == "ShaderNodeObjectInfo":
            spec = {"type": "object_info"}
        elif kind == "ShaderNodeNewGeometry":
            spec = {"type": "geometry_info"}
        elif kind == "ShaderNodeLightPath":
            spec = {"type": "light_path"}
        elif kind == "ShaderNodeMix":
            # Blender exposes duplicate Factor/A/B sockets. `enabled` is the reliable
            # discriminator; inactive sockets may still carry links from old edits.
            def active_socket(name, default, factor=False):
                candidates = [s for s in node.inputs if s.name == name]
                if factor and node.data_type != "VECTOR":
                    candidates = [s for s in candidates if s.bl_idname in ("NodeSocketFloat", "NodeSocketFloatFactor")]
                else:
                    socket_type = {"RGBA": "NodeSocketColor", "VECTOR": "NodeSocketVector",
                                   "FLOAT": "NodeSocketFloat", "ROTATION": "NodeSocketRotation"}.get(node.data_type)
                    if socket_type:
                        candidates = [s for s in candidates if s.bl_idname == socket_type]
                chosen = next((s for s in candidates if s.enabled and s.is_linked), None)
                chosen = chosen or next((s for s in candidates if s.enabled), None)
                chosen = chosen or next((s for s in candidates if s.is_linked), None)
                chosen = chosen or (candidates[0] if candidates else None)
                return ref(chosen, default)
            spec = {"type": "mix", "blend_type": node.blend_type, "inputs": {
                "factor": active_socket("Factor", 0.5, True), "a": active_socket("A", [0, 0, 0, 1]),
                "b": active_socket("B", [1, 1, 1, 1])}}
        elif kind in ("ShaderNodeMixRGB",):
            spec = {"type": "mix", "blend_type": node.blend_type, "inputs": {
                "factor": input_named(node, ("Fac",), 0.5), "a": ref(node.inputs[1], [0, 0, 0, 1]),
                "b": ref(node.inputs[2], [1, 1, 1, 1])}}
        elif kind == "ShaderNodeMath":
            spec = {"type": "math", "operation": node.operation, "inputs": {
                "a": ref(node.inputs[0], 0.0), "b": ref(node.inputs[1], 0.0),
                "c": ref(node.inputs[2], 0.0) if len(node.inputs) > 2 else 0.0}}
        elif kind == "ShaderNodeValToRGB":
            spec = {"type": "color_ramp", "interpolation": node.color_ramp.interpolation,
                    "elements": [{"position": float(e.position), "color": list(e.color)}
                                 for e in node.color_ramp.elements],
                    "inputs": {"factor": input_named(node, ("Fac",), 0.0)}}
        elif kind == "ShaderNodeInvert":
            spec = {"type": "invert", "inputs": {"factor": input_named(node, ("Fac",), 1.0),
                                                     "color": input_named(node, ("Color",), [0, 0, 0, 1])}}
        elif kind == "ShaderNodeLayerWeight":
            spec = {"type": "layer_weight", "inputs": {"blend": input_named(node, ("Blend",), 0.5)}}
        elif kind == "ShaderNodeNormalMap":
            spec = {"type": "normal_map", "inputs": {
                "color": input_named(node, ("Color",), [0.5, 0.5, 1, 1]),
                "strength": input_named(node, ("Strength",), 1.0)}}
        elif kind == "ShaderNodeBump":
            bump_inputs = {"height": input_named(node, ("Height",), 0.0)}
            normal_input = find_socket(node, ("Normal",))
            if normal_input and normal_input.is_linked:
                bump_inputs["normal"] = ref(normal_input)
            spec = {"type": "bump", "inputs": bump_inputs}
        elif kind == "ShaderNodeRGB":
            spec = {"type": "rgb", "inputs": {"value": list(node.outputs[0].default_value)}}
        elif kind == "ShaderNodeValue":
            spec = {"type": "value", "inputs": {"value": float(node.outputs[0].default_value)}}
        elif kind in ("ShaderNodeCombineColor", "ShaderNodeCombineXYZ"):
            spec = {"type": "combine", "inputs": {
                "x": input_named(node, ("Red", "X"), 0.0),
                "y": input_named(node, ("Green", "Y"), 0.0),
                "z": input_named(node, ("Blue", "Z"), 0.0)}}
        elif kind in ("ShaderNodeSeparateColor", "ShaderNodeSeparateXYZ"):
            spec = {"type": "separate", "inputs": {"color": input_named(node, ("Color", "Image", "Vector"), [0, 0, 0, 1])}}
        elif kind == "NodeReroute":
            spec = {"type": "passthrough", "inputs": {"value": ref(node.inputs[0], 0.0)}}
        elif kind == "ShaderNodeBlackbody":
            temperature = socket_value(find_socket(node, ("Temperature",)), 6500.0)
            t = max(float(temperature), 1000.0) / 100.0
            if t <= 66:
                rgb = [255.0, 99.4708025861 * math.log(t) - 161.1195681661,
                       0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307]
            else:
                rgb = [329.698727446 * ((t - 60) ** -0.1332047592),
                       288.1221695283 * ((t - 60) ** -0.0755148492), 255.0]
            spec = {"type": "rgb", "inputs": {"value": [max(0, min(255, x)) / 255 for x in rgb] + [1]}}
        elif kind == "ShaderNodeTexBrick":
            spec = {"type": "brick_texture", "inputs": {
                "vector": input_named(node, ("Vector",), [0, 0, 0]),
                "color1": input_named(node, ("Color1",), [0.8, 0.2, 0.1, 1]),
                "color2": input_named(node, ("Color2",), [0.4, 0.05, 0.02, 1]),
                "mortar": input_named(node, ("Mortar",), [0.0, 0.0, 0.0, 1]),
                "scale": input_named(node, ("Scale",), 5.0),
                "mortar_size": input_named(node, ("Mortar Size",), 0.02)}}
        elif kind == "ShaderNodeRGBCurve":
            # Sample Blender's curve mapping so the runtime graph remains compact and
            # independent of Blender's bezier curve implementation.
            mapping = node.mapping
            try:
                mapping.initialize()
            except Exception:
                pass
            samples = []
            for channel in range(3):
                values = []
                for i in range(33):
                    x = i / 32.0
                    combined = mapping.evaluate(mapping.curves[3], x)
                    values.append(float(mapping.evaluate(mapping.curves[channel], combined)))
                samples.append(values)
            spec = {"type": "rgb_curves", "samples": samples, "inputs": {
                "factor": input_named(node, ("Fac",), 1.0),
                "color": input_named(node, ("Color",), [0, 0, 0, 1])}}
        elif kind == "ShaderNodeHueSaturation":
            spec = {"type": "hue_saturation", "inputs": {
                "hue": input_named(node, ("Hue",), 0.5),
                "saturation": input_named(node, ("Saturation",), 1.0),
                "value": input_named(node, ("Value",), 1.0),
                "factor": input_named(node, ("Fac",), 1.0),
                "color": input_named(node, ("Color",), [0, 0, 0, 1])}}
        elif kind == "ShaderNodeGamma":
            spec = {"type": "gamma", "inputs": {
                "color": input_named(node, ("Color",), [1, 1, 1, 1]),
                "gamma": input_named(node, ("Gamma",), 1.0)}}
        elif kind == "ShaderNodeBrightContrast":
            spec = {"type": "bright_contrast", "inputs": {
                "color": input_named(node, ("Color",), [1, 1, 1, 1]),
                "brightness": input_named(node, ("Brightness",), 0.0),
                "contrast": input_named(node, ("Contrast",), 0.0)}}
        elif kind == "ShaderNodeGroup":
            # Value groups in the benchmark are coordinate helpers. The current
            # group mirrors Y around one, which can be represented directly.
            vector_input = find_socket(node, ("Vector",))
            if vector_input is not None and any(s.name == "Vector" for s in node.outputs):
                spec = {"type": "invert_y", "inputs": {"vector": ref(vector_input, [0, 0, 0])}}
        if spec is None:
            unsupported.append(kind)
            spec = {"type": "value", "inputs": {"value": 0.0}}
        nodes[node_id] = spec
        return node_id

    defaults = {"base_color": [0.8, 0.8, 0.8, 1], "roughness": 0.5, "metallic": 0.0,
                "specular": 0.5, "anisotropic": 0.0, "anisotropic_rotation": 0.0,
                "sheen": 0.0, "clearcoat": 0.0, "clearcoat_roughness": 0.03,
                "ior": 1.45, "transmission": 0.0, "transmission_roughness": 0.0,
                "emission": [0, 0, 0, 1], "opacity": 1.0, "thin_walled": 0.0}
    defaults.update({"subsurface": 0.0, "subsurface_scale": 0.0})
    defaults["subsurface_radius"] = [1.0, 0.2, 0.1]
    defaults["subsurface_method"] = 0.0
    closure_serial = [0]

    def leaf_surface(leaf):
        if leaf.bl_idname == "ShaderNodeBsdfTransparent":
            return {"base_color": input_named(leaf, ("Color",), [1, 1, 1, 1]),
                    "opacity": 0.0}
        if leaf.bl_idname == "ShaderNodeGroup":
            return {"base_color": input_named(leaf, ("Color",), defaults["base_color"]),
                    "roughness": input_named(leaf, ("Roughness",), defaults["roughness"]),
                    "metallic": 0.0, "specular": 0.5}
        socket_names = {
            "base_color": ("Base Color", "Color"), "roughness": ("Roughness",),
            "metallic": ("Metallic",), "specular": ("Specular IOR Level", "Specular"),
            "anisotropic": ("Anisotropic IOR Level", "Anisotropic"),
            "anisotropic_rotation": ("Anisotropic Rotation",), "sheen": ("Sheen Weight", "Sheen"),
            "clearcoat": ("Coat Weight", "Clearcoat"),
            "clearcoat_roughness": ("Coat Roughness", "Clearcoat Roughness"), "ior": ("IOR",),
            "transmission": ("Transmission Weight", "Transmission"),
            "transmission_roughness": ("Transmission Roughness",),
            "subsurface": ("Subsurface Weight",), "subsurface_scale": ("Subsurface Scale",),
            "subsurface_radius": ("Subsurface Radius",),
            "normal": ("Normal",)}
        result = {}
        for name, names in socket_names.items():
            socket = find_socket(leaf, names)
            if socket is not None and (name != "normal" or socket.is_linked):
                result[name] = ref(socket, defaults.get(name, 0.0))
        if leaf.bl_idname in {"ShaderNodeBsdfPrincipled", "ShaderNodeSubsurfaceScattering"}:
            method = getattr(leaf, "subsurface_method", "BURLEY")
            result["subsurface_method"] = (2.0 if method == "RANDOM_WALK_SKIN"
                                             else 1.0 if method == "RANDOM_WALK"
                                             else 0.0)
        # A legacy Diffuse BSDF has no dielectric reflection lobe. Giving it
        # Principled's default specular value made old Blender assets look like
        # polished plastic, especially classroom desks and floors.
        if leaf.bl_idname == "ShaderNodeBsdfDiffuse":
            result["specular"] = 0.0
            result["metallic"] = 0.0
        transmission_socket = find_socket(leaf, ("Transmission Weight", "Transmission"))
        if transmission_socket and transmission_socket.is_linked:
            upstream = upstream_nodes(transmission_socket)
            opacity_images = [node for node in upstream if node.bl_idname == "ShaderNodeTexImage" and node.image and
                              "opacity" in (node.image.name + " " + node.image.filepath).lower()]
            if opacity_images:
                result["thin_walled"] = 1.0
        if leaf.bl_idname in ("ShaderNodeBsdfAnisotropic", "ShaderNodeBsdfGlossy"):
            result["metallic"] = 1.0
        if leaf.bl_idname == "ShaderNodeBsdfGlass":
            result["transmission"] = 1.0
        emission_names = (("Emission Color", "Emission", "Color") if leaf.bl_idname == "ShaderNodeEmission"
                          else ("Emission Color", "Emission"))
        emission_socket = find_socket(leaf, emission_names)
        strength_socket = find_socket(leaf, ("Emission Strength", "Strength"))
        if leaf.bl_idname == "ShaderNodeEmission":
            result["base_color"] = [0, 0, 0, 1]
        if emission_socket:
            node_id = f"closure_emission_{closure_serial[0]}"; closure_serial[0] += 1
            nodes[node_id] = {"type": "mix", "blend_type": "MULTIPLY", "inputs": {
                "factor": 1.0, "a": ref(emission_socket, [0, 0, 0, 1]),
                "b": ref(strength_socket, 1.0 if leaf.bl_idname == "ShaderNodeEmission" else 0.0)}}
            result["emission"] = {"node": node_id, "output": "result"}
        return result

    def linked_closure(socket):
        return socket.links[0].from_node if socket and socket.is_linked else None

    def closure_surface(node):
        if node is None:
            return {}
        if node.bl_idname not in ("ShaderNodeMixShader", "ShaderNodeAddShader"):
            return leaf_surface(node)
        if node.bl_idname == "ShaderNodeMixShader":
            a_node = linked_closure(node.inputs[1])
            b_node = linked_closure(node.inputs[2])
            # Old files sometimes leave one Mix Shader input disconnected. It
            # is not a glossy closure and must not inject Principled defaults.
            if a_node is None:
                return closure_surface(b_node)
            if b_node is None:
                return closure_surface(a_node)
            a = closure_surface(a_node)
            b = closure_surface(b_node)
            factor = ref(node.inputs[0], 0.5)
            glossy_types = {"ShaderNodeBsdfAnisotropic", "ShaderNodeBsdfGlossy"}
            if ((a_node.bl_idname == "ShaderNodeBsdfDiffuse" and b_node.bl_idname in glossy_types) or
                    (b_node.bl_idname == "ShaderNodeBsdfDiffuse" and a_node.bl_idname in glossy_types)):
                diffuse = a if a_node.bl_idname == "ShaderNodeBsdfDiffuse" else b
                glossy = b if a_node.bl_idname == "ShaderNodeBsdfDiffuse" else a
                glossy_factor = factor
                if a_node.bl_idname in glossy_types:
                    inverse_id = f"closure_mix_{closure_serial[0]}"; closure_serial[0] += 1
                    nodes[inverse_id] = {"type": "math", "operation": "SUBTRACT", "inputs": {
                        "a": 1.0, "b": factor}}
                    glossy_factor = {"node": inverse_id, "output": "value"}
                specular_id = f"closure_mix_{closure_serial[0]}"; closure_serial[0] += 1
                nodes[specular_id] = {"type": "math", "operation": "MULTIPLY", "inputs": {
                    "a": glossy_factor, "b": 12.5}}
                result = dict(diffuse)
                result["roughness"] = glossy.get("roughness", 0.5)
                result["specular"] = {"node": specular_id, "output": "value"}
                result["metallic"] = 0.0
                if "normal" in glossy:
                    result["normal"] = glossy["normal"]
                return result
        else:
            a_node = linked_closure(node.inputs[0])
            b_node = linked_closure(node.inputs[1])
            a = closure_surface(a_node)
            b = closure_surface(b_node)
            factor = 0.5
            # Adding an Emission closure does not dilute the other closure's
            # reflectance. Preserve the scattering branch and only add emitted
            # radiance; averaging every field made emissive props too dark.
            if ((a_node and a_node.bl_idname == "ShaderNodeEmission") or
                    (b_node and b_node.bl_idname == "ShaderNodeEmission")):
                scattering = b if a_node and a_node.bl_idname == "ShaderNodeEmission" else a
                result = dict(scattering)
                node_id = f"closure_mix_{closure_serial[0]}"; closure_serial[0] += 1
                nodes[node_id] = {"type": "mix", "blend_type": "ADD", "inputs": {
                    "factor": 1.0, "a": a.get("emission", defaults["emission"]),
                    "b": b.get("emission", defaults["emission"])} }
                result["emission"] = {"node": node_id, "output": "result"}
                return result
        result = {}
        for name in set(defaults) | set(a) | set(b):
            fallback = defaults.get(name, a.get(name, b.get(name)))
            node_id = f"closure_mix_{closure_serial[0]}"; closure_serial[0] += 1
            blend = "ADD" if node.bl_idname == "ShaderNodeAddShader" and name == "emission" else "MIX"
            nodes[node_id] = {"type": "mix", "blend_type": blend, "inputs": {
                "factor": 1.0 if blend == "ADD" else factor,
                "a": a.get(name, fallback), "b": b.get(name, fallback)}}
            result[name] = {"node": node_id, "output": "result"}
        return result

    surface = closure_surface(closure_root or shader)
    reachable_types = {node.bl_idname for node in (nodes and reachable_surface_nodes(material) or [])}
    if {"ShaderNodeBsdfTransparent", "ShaderNodeLightPath"} <= reachable_types and surface.get("transmission") is not None:
        surface["thin_walled"] = 2.0
    if unsupported:
        warnings.append(f"material '{material.name}' graph substituted unsupported nodes: {', '.join(sorted(set(unsupported)))}")
    emission_hint = [0, 0, 0]
    for candidate in reachable_surface_nodes(material):
        names = (("Emission Color", "Emission", "Color") if candidate.bl_idname == "ShaderNodeEmission"
                 else ("Emission Color", "Emission"))
        emission_socket = find_socket(candidate, names)
        if emission_socket:
            strength_socket = find_socket(candidate, ("Emission Strength", "Strength"))
            strength = scalar(strength_socket, 1.0 if candidate.bl_idname == "ShaderNodeEmission" else 0.0)
            hint = ([strength] * 3 if emission_socket.is_linked
                    else [component * strength for component in color(emission_socket, (1, 1, 1))])
            emission_hint = [max(a, b) for a, b in zip(emission_hint, hint)]
    return {"type": "shader_graph", "graph": {"nodes": nodes, "surface": surface},
            "emission_hint": emission_hint}


def write_image(image, texture_dir, image_cache, warnings):
    if image is None:
        return None
    key = image.as_pointer()
    if key in image_cache:
        return image_cache[key]
    ext_by_format = {"JPEG": ".jpg", "PNG": ".png", "TARGA": ".tga",
                     "BMP": ".bmp", "OPEN_EXR": ".exr", "HDR": ".hdr",
                     "TIFF": ".tif"}
    source_path = Path(bpy.path.abspath(image.filepath, library=image.library)) if image.filepath else None
    ext = source_path.suffix.lower() if source_path and source_path.suffix else ext_by_format.get(image.file_format, ".png")
    filename = safe_name(image.name) + ext
    destination = texture_dir / filename
    suffix = 2
    while destination.exists():
        filename = safe_name(image.name) + f"_{suffix}" + ext
        destination = texture_dir / filename
        suffix += 1
    try:
        if image.source == "TILED":
            warnings.append(f"image '{image.name}' is tiled; only its active tile was exported")
            image.save_render(str(destination))
        elif image.packed_file is not None:
            destination.write_bytes(bytes(image.packed_file.data))
        elif source_path and source_path.is_file():
            shutil.copy2(source_path, destination)
        else:
            image.save_render(str(destination))
    except Exception as exc:
        warnings.append(f"could not export image '{image.name}': {exc}")
        return None
    relative = "textures/" + filename
    image_cache[key] = relative
    return relative


def material_spec(material, texture_dir, image_cache, warnings):
    if material is None:
        return {"type": "principled", "base_color": [0.8, 0.8, 0.8], "roughness": 0.5}
    nodes = reachable_surface_nodes(material)
    shader = select_shader(nodes)
    if shader is None:
        warnings.append(f"material '{material.name}' has no supported surface shader; using gray Principled")
        return {"type": "principled", "base_color": [0.8, 0.8, 0.8], "roughness": 0.5}

    graph_inputs = [socket for socket in shader.inputs if socket.is_linked]
    graph_nodes = {node.bl_idname for socket in graph_inputs for node in upstream_nodes(socket)}
    if (nodes and nodes[0] is not shader) or graph_nodes - {"ShaderNodeTexImage", "ShaderNodeNormalMap", "ShaderNodeTexCoord"}:
        return shader_graph_spec(material, shader, texture_dir, image_cache, warnings, nodes[0] if nodes else shader)

    kind = shader.bl_idname
    base_socket = find_socket(shader, ("Base Color", "Color"))
    rough_socket = find_socket(shader, ("Roughness",))
    normal_socket = find_socket(shader, ("Normal",))
    metallic_socket = find_socket(shader, ("Metallic",))
    specular_socket = find_socket(shader, ("Specular IOR Level", "Specular"))
    transmission_socket = find_socket(shader, ("Transmission Weight", "Transmission"))
    ior_socket = find_socket(shader, ("IOR",))
    emission_socket = find_socket(shader, ("Emission Color", "Emission", "Color" if kind == "ShaderNodeEmission" else ""))
    emission_strength_socket = find_socket(shader, ("Emission Strength", "Strength"))

    spec = {
        "type": "principled",
        "base_color": color(base_socket, (0.8, 0.8, 0.8)),
        "roughness": scalar(rough_socket, 0.5),
        "metallic": scalar(metallic_socket, 1.0 if kind in ("ShaderNodeBsdfAnisotropic", "ShaderNodeBsdfGlossy") else 0.0),
        "specular": scalar(specular_socket, 0.5),
        "ior": scalar(ior_socket, 1.45),
        "transmission": scalar(transmission_socket, 1.0 if kind == "ShaderNodeBsdfGlass" else 0.0),
    }
    if kind == "ShaderNodeEmission":
        spec["base_color"] = [0.0, 0.0, 0.0]
        spec["emission_color"] = color(emission_socket, (1.0, 1.0, 1.0))
        spec["emission_strength"] = scalar(emission_strength_socket, 1.0)
    else:
        spec["emission_color"] = color(emission_socket, (1.0, 1.0, 1.0))
        spec["emission_strength"] = scalar(emission_strength_socket, 0.0)

    textures = {}
    for key, socket in (("base_color", base_socket), ("roughness", rough_socket),
                        ("metallic", metallic_socket), ("specular", specular_socket),
                        ("emission", emission_socket)):
        path = write_image(first_image(socket), texture_dir, image_cache, warnings)
        if path:
            textures[key] = path
    normal_path = write_image(first_image(normal_socket, require_normal_map=True), texture_dir, image_cache, warnings)
    if normal_path:
        textures["normal"] = normal_path
        textures["normal_reverse_y"] = False
    if textures:
        spec["textures"] = textures
    if nodes and nodes[0] is not shader:
        warnings.append(f"material '{material.name}' uses {nodes[0].bl_idname}; exported its primary {kind} branch")
    return spec


def convert_matrix(matrix, coordinate_matrix):
    converted = coordinate_matrix @ matrix @ coordinate_matrix.inverted()
    return [float(converted[row][col]) for col in range(4) for row in range(4)]


def converted_vector(vector):
    return Vector((vector.x, vector.z, -vector.y))


def write_binary_mesh(mesh, material_index, path, attribute_name=None):
    mesh.calc_loop_triangles()
    corner_normals = mesh.corner_normals
    records, faces, lookup = [], [], {}
    uv_layer = mesh.uv_layers.active.data if mesh.uv_layers.active else None
    color_layer = mesh.color_attributes.get(attribute_name) if attribute_name else None
    for triangle in mesh.loop_triangles:
        if triangle.material_index != material_index:
            continue
        face = []
        for loop_index in triangle.loops:
            loop = mesh.loops[loop_index]
            position = converted_vector(mesh.vertices[loop.vertex_index].co)
            normal = converted_vector(corner_normals[loop_index].vector).normalized()
            uv = uv_layer[loop_index].uv if uv_layer else (0.0, 0.0)
            color_index = loop.vertex_index if color_layer and color_layer.domain == "POINT" else loop_index
            vertex_color = color_layer.data[color_index].color if color_layer else (1.0, 1.0, 1.0, 1.0)
            record = (position.x, position.y, position.z, normal.x, normal.y, normal.z, float(uv[0]), float(uv[1]),
                      float(vertex_color[0]), float(vertex_color[1]), float(vertex_color[2]))
            index = lookup.get(record)
            if index is None:
                index = len(records) + 1
                lookup[record] = index
                records.append(record)
            face.append(index)
        # Coordinate conversion is a rotation, so winding remains unchanged.
        faces.append(face)
    if not faces:
        return 0
    indices = array("I", (index - 1 for face in faces for index in face))
    positions = array("f", (value for r in records for value in r[0:3]))
    normals = array("f", (value for r in records for value in r[3:6]))
    tex_coords = array("f", (value for r in records for value in r[6:8])) if uv_layer else None
    colors = array("f", (value for r in records for value in r[8:11])) if color_layer else None
    with path.open("wb") as stream:
        flags = 1 | (2 if tex_coords else 0) | (4 if colors else 0)
        stream.write(struct.pack("<8sIII", b"SPKMESH1", len(records), len(indices), flags))
        for values in (indices, positions, normals, tex_coords, colors):
            if values is None:
                continue
            if sys.byteorder != "little":
                values.byteswap()
            values.tofile(stream)
    return len(faces)


def write_binary_hair(obj, particle_system, path):
    """Write evaluated parent and child particle strands in world space."""
    settings = particle_system.settings
    strand_count = len(particle_system.particles) + len(particle_system.child_particles)
    render_max_step = 2 ** int(settings.render_step)
    sample_count = min(render_max_step + 1, 65)
    offsets = array("I", [0])
    points = array("f")
    radii = array("f")
    root_radius = float(settings.root_radius * settings.radius_scale)
    tip_radius = float(settings.tip_radius * settings.radius_scale)
    # Avoid exactly degenerate last rings in the mesh fallback used by current
    # render backends while retaining Blender's visually sharp taper.
    tip_radius = max(tip_radius, root_radius * 0.02)
    written = 0
    for particle_index in range(strand_count):
        strand = []
        for sample_index in range(sample_count):
            step = round(sample_index * render_max_step / (sample_count - 1))
            point = converted_vector(particle_system.co_hair(
                obj, particle_no=particle_index, step=step))
            strand.append(point)
        if len(strand) < 2 or (strand[-1] - strand[0]).length_squared < 1.0e-12:
            continue
        for step, point in enumerate(strand):
            t = step / (len(strand) - 1)
            radius = root_radius * (1.0 - t) + tip_radius * t
            points.extend((point.x, point.y, point.z))
            radii.append(radius)
        written += len(strand)
        offsets.append(written)
    if len(offsets) <= 1:
        return 0, 0
    for values in (offsets, points, radii):
        if sys.byteorder != "little":
            values.byteswap()
    with path.open("wb") as stream:
        stream.write(struct.pack("<8sII", b"SPKHAIR1", len(offsets) - 1, written))
        offsets.tofile(stream)
        points.tofile(stream)
        radii.tofile(stream)
    return len(offsets) - 1, written


def world_background(scene):
    fallback = [float(x) for x in scene.world.color] if scene.world else [0.0, 0.0, 0.0]
    if not scene.world or not scene.world.node_tree:
        return fallback
    output = next((n for n in scene.world.node_tree.nodes if n.bl_idname == "ShaderNodeOutputWorld" and n.is_active_output), None)
    nodes = upstream_nodes(output.inputs.get("Surface") if output else None)
    bg = next((n for n in nodes if n.bl_idname == "ShaderNodeBackground"), None)
    if not bg:
        return fallback
    c = color(bg.inputs.get("Color"), fallback)
    strength = scalar(bg.inputs.get("Strength"), 1.0)
    # Handle Blender's common Blackbody -> Background setup with a close RGB approximation.
    if bg.inputs.get("Color") and bg.inputs["Color"].is_linked:
        blackbody = next((n for n in upstream_nodes(bg.inputs["Color"]) if n.bl_idname == "ShaderNodeBlackbody"), None)
        if blackbody:
            t = scalar(blackbody.inputs.get("Temperature"), 6500.0) / 100.0
            if t <= 66:
                red = 255
                green = 99.4708025861 * math.log(max(t, 1e-6)) - 161.1195681661
                blue = 0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
            else:
                red = 329.698727446 * ((t - 60) ** -0.1332047592)
                green = 288.1221695283 * ((t - 60) ** -0.0755148492)
                blue = 255
            c = [max(0, min(255, x)) / 255.0 for x in (red, green, blue)]
    return [x * strength for x in c]


def blackbody_rgb(temperature):
    """Approximate Blender's linear RGB blackbody socket."""
    t = max(float(temperature), 1000.0) / 100.0
    if t <= 66.0:
        values = (255.0,
                  99.4708025861 * math.log(t) - 161.1195681661,
                  0.0 if t <= 19.0 else 138.5177312231 * math.log(t - 10.0) - 305.0447927307)
    else:
        values = (329.698727446 * ((t - 60.0) ** -0.1332047592),
                  288.1221695283 * ((t - 60.0) ** -0.0755148492), 255.0)
    srgb = [max(0.0, min(1.0, value / 255.0)) for value in values]
    return [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
            for value in srgb]


def light_node_emission(data):
    """Return constant node emission as linear RGB radiance, when available."""
    if not data.use_nodes or not data.node_tree:
        return None
    output = next((node for node in data.node_tree.nodes
                   if node.bl_idname == "ShaderNodeOutputLight" and node.is_active_output), None)
    reachable = upstream_nodes(output.inputs.get("Surface") if output else None)
    emission = next((node for node in reachable if node.bl_idname == "ShaderNodeEmission"), None)
    if emission is None:
        return None
    strength_socket = emission.inputs.get("Strength")
    if strength_socket is None or strength_socket.is_linked:
        return None
    strength = scalar(strength_socket, 0.0)
    color_socket = emission.inputs.get("Color")
    light_color = color(color_socket, data.color)
    if color_socket and color_socket.is_linked:
        blackbody = next((node for node in upstream_nodes(color_socket)
                          if node.bl_idname == "ShaderNodeBlackbody"), None)
        if blackbody:
            light_color = blackbody_rgb(scalar(blackbody.inputs.get("Temperature"), 6500.0))
        else:
            return None
    return [component * strength for component in light_color]


def light_node_falloff_distance(data):
    """Recognize Ray Length / distance -> clamped Mix Shader falloff rigs."""
    if not data.use_nodes or not data.node_tree:
        return 0.0
    output = next((node for node in data.node_tree.nodes
                   if node.bl_idname == "ShaderNodeOutputLight" and node.is_active_output), None)
    nodes = upstream_nodes(output.inputs.get("Surface") if output else None)
    mix = next((node for node in nodes if node.bl_idname == "ShaderNodeMixShader"), None)
    if not mix or not mix.inputs[0].is_linked or not mix.inputs[1].is_linked or mix.inputs[2].is_linked:
        return 0.0
    mapping = mix.inputs[0].links[0].from_node
    if mapping.bl_idname != "ShaderNodeMapRange" or not mapping.clamp or not mapping.inputs[0].is_linked:
        return 0.0
    math_node = mapping.inputs[0].links[0].from_node
    if math_node.bl_idname != "ShaderNodeMath" or math_node.operation != "DIVIDE":
        return 0.0
    if not math_node.inputs[0].is_linked or math_node.inputs[1].is_linked:
        return 0.0
    light_path = math_node.inputs[0].links[0].from_node
    if light_path.bl_idname != "ShaderNodeLightPath":
        return 0.0
    return max(scalar(math_node.inputs[1], 0.0), 0.0)


def export(output_dir):
    output_dir = Path(output_dir).resolve()
    mesh_dir, texture_dir = output_dir / "meshes", output_dir / "textures"
    if mesh_dir.exists():
        shutil.rmtree(mesh_dir)
    if texture_dir.exists():
        shutil.rmtree(texture_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    coordinate_matrix = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))
    warnings, image_cache = [], {}

    document = {
        "format": "sparkium-scene", "version": 1,
        "name": scene.name,
        "film": {"width": int(scene.render.resolution_x * scene.render.resolution_percentage / 100),
                 "height": int(scene.render.resolution_y * scene.render.resolution_percentage / 100),
                 "persistence": 1.0, "clamping": 100.0, "max_exposure": 16.0,
                 "view_transform": ("filmic" if scene.view_settings.view_transform == "Filmic"
                                    else "standard"),
                 "exposure": float(scene.view_settings.exposure),
                 "gamma": float(scene.view_settings.gamma),
                 "contrast": (1.2 if getattr(scene.view_settings, "look", "") == "High Contrast"
                              else 1.0)},
        "renderer": {"pipeline": "ray_tracing", "samples_per_dispatch": 8, "max_bounces": 16,
                     "alpha_shadow": True, "ambient_light": [0.02, 0.02, 0.02],
                     "background_color": world_background(scene)},
        "camera": {}, "materials": {}, "geometries": {}, "entities": []
    }

    camera_object = scene.camera
    if camera_object is None:
        raise RuntimeError("the Blender scene has no active camera")
    camera_matrix = camera_object.matrix_world
    eye = converted_vector(camera_matrix.translation)
    forward = converted_vector(camera_matrix.to_quaternion() @ Vector((0, 0, -1))).normalized()
    up = converted_vector(camera_matrix.to_quaternion() @ Vector((0, 1, 0))).normalized()
    focus_distance = camera_object.data.dof.focus_distance
    if camera_object.data.dof.focus_object:
        focus_distance = (camera_object.data.dof.focus_object.matrix_world.translation - camera_matrix.translation).length
    if focus_distance <= 0:
        focus_distance = 10.0
    aperture_radius = 0.0
    if camera_object.data.dof.use_dof and camera_object.data.dof.aperture_fstop > 0:
        aperture_radius = camera_object.data.lens / (2000.0 * camera_object.data.dof.aperture_fstop)
    render_width = document["film"]["width"] * scene.render.pixel_aspect_x
    render_height = document["film"]["height"] * scene.render.pixel_aspect_y
    render_aspect = render_width / render_height
    sensor_fit = camera_object.data.sensor_fit
    if sensor_fit == "AUTO":
        sensor_fit = "HORIZONTAL" if render_aspect >= 1.0 else "VERTICAL"
    if sensor_fit == "HORIZONTAL":
        vertical_fov = 2.0 * math.atan(math.tan(camera_object.data.angle_x * 0.5) / render_aspect)
    else:
        vertical_fov = camera_object.data.angle_y
    document["camera"] = {
        "eye": list(eye), "target": list(eye + forward), "up": list(up),
        "fov_degrees": math.degrees(vertical_fov),
        "aperture_radius": aperture_radius, "focus_distance": focus_distance,
        "aperture_blades": int(camera_object.data.dof.aperture_blades),
        "aperture_rotation": float(camera_object.data.dof.aperture_rotation),
        "aperture_ratio": float(camera_object.data.dof.aperture_ratio)
    }

    material_ids = {}
    def ensure_material(material):
        key = material.as_pointer() if material else 0
        if key not in material_ids:
            base = safe_name(material.name if material else "default")
            identifier = base
            suffix = 2
            while identifier in document["materials"]:
                identifier = f"{base}_{suffix}"; suffix += 1
            material_ids[key] = identifier
            document["materials"][identifier] = material_spec(material, texture_dir, image_cache, warnings)
        return material_ids[key]

    object_cache = {}
    used_mesh_names = set()
    instance_count = triangle_count = 0
    for instance in depsgraph.object_instances:
        obj = instance.object
        if obj.type not in {"MESH", "CURVE", "SURFACE", "FONT", "META"} or obj.hide_render or not instance.show_self:
            continue
        original = obj.original
        cache_key = original.as_pointer()
        if cache_key not in object_cache:
            mesh = obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
            groups = []
            if mesh:
                mesh.calc_loop_triangles()
                material_indices = sorted(set(t.material_index for t in mesh.loop_triangles))
                for material_index in material_indices:
                    material = mesh.materials[material_index] if material_index < len(mesh.materials) else None
                    attributes = []
                    if material and material.use_nodes:
                        attributes = [n.attribute_name for n in material.node_tree.nodes
                                      if n.bl_idname == "ShaderNodeAttribute" and n.attribute_name]
                    attribute_name = "Col" if "Col" in attributes else (attributes[0] if attributes else None)
                    base = safe_name(original.name) + f"_m{material_index}"
                    name = base; suffix = 2
                    while name in used_mesh_names:
                        name = f"{base}_{suffix}"; suffix += 1
                    used_mesh_names.add(name)
                    relative = f"meshes/{name}.spmesh"
                    count = write_binary_mesh(mesh, material_index, output_dir / relative, attribute_name)
                    if count:
                        document["geometries"][name] = {"type": "binary_mesh", "path": relative}
                        groups.append((name, ensure_material(material)))
                        triangle_count += count
                obj.to_mesh_clear()
            object_cache[cache_key] = groups
        transform = {"matrix": convert_matrix(instance.matrix_world, coordinate_matrix)}
        for geometry_id, material_id in object_cache[cache_key]:
            document["entities"].append({"type": "mesh", "geometry": geometry_id,
                                         "material": material_id, "transform": transform,
                                         "raster_light": False})
            instance_count += 1

    hair_strand_count = hair_point_count = 0
    for source_obj in scene.objects:
        if source_obj.type != "MESH" or source_obj.hide_render or not source_obj.particle_systems:
            continue
        obj = source_obj.evaluated_get(depsgraph)
        for particle_system_index, particle_system in enumerate(obj.particle_systems):
            if particle_system.settings.type != "HAIR":
                continue
            base = safe_name(source_obj.name + "_" + particle_system.name + "_hair")
            name = base
            suffix = 2
            while name in used_mesh_names:
                name = f"{base}_{suffix}"
                suffix += 1
            used_mesh_names.add(name)
            relative = f"meshes/{name}.spkhair"
            strands, points = write_binary_hair(obj, particle_system, output_dir / relative)
            if not strands:
                continue
            source_settings = source_obj.particle_systems[particle_system_index].settings
            material_index = max(int(source_settings.material) - 1, 0)
            material = (source_obj.material_slots[material_index].material
                        if material_index < len(source_obj.material_slots) else None)
            document["geometries"][name] = {"type": "hair", "path": relative, "radial_segments": 3}
            document["entities"].append({"type": "mesh", "geometry": name,
                                         "material": ensure_material(material),
                                         "transform": {"matrix": convert_matrix(Matrix.Identity(4), coordinate_matrix)},
                                         "raster_light": False})
            hair_strand_count += strands
            hair_point_count += points
            instance_count += 1

    # Unit area-light plane: local Blender -Z emission maps to local Sparkium -Y.
    document["geometries"]["__area_light_quad"] = {
        "type": "inline_mesh",
        "positions": [[-0.5, 0, 0.5], [0.5, 0, 0.5], [0.5, 0, -0.5], [-0.5, 0, -0.5]],
        "indices": [0, 2, 1, 0, 3, 2]
    }
    bounds = [converted_vector(obj.matrix_world @ Vector(corner)) for obj in scene.objects
              if obj.type in {"MESH", "CURVE"} and not obj.hide_render for corner in obj.bound_box]
    center = sum(bounds, Vector()) / len(bounds) if bounds else Vector()
    radius = max(((p - center).length for p in bounds), default=10.0)
    light_count = 0
    for obj in scene.objects:
        if obj.type != "LIGHT" or obj.hide_render or obj.data.energy <= 0:
            continue
        data, matrix = obj.data, obj.matrix_world
        color_value = [float(x) for x in data.color]
        node_emission = light_node_emission(data)
        falloff_distance = light_node_falloff_distance(data)
        if data.type == "POINT":
            document["entities"].append({"type": "point_light",
                                         "position": list(converted_vector(matrix.translation)),
                                         "color": color_value, "strength": float(data.energy),
                                         "radius": float(data.shadow_soft_size),
                                         "soft_falloff": bool(data.use_soft_falloff)})
        elif data.type == "SUN":
            direction = converted_vector(matrix.to_quaternion() @ Vector((0, 0, -1))).normalized()
            distance = max(radius * 100.0, 1000.0)
            position = center - direction * distance
            sun_strength = max(node_emission) if node_emission else float(data.energy)
            if node_emission:
                color_value = [value / max(sun_strength, 1.0e-8) for value in node_emission]
            strength = sun_strength * 4.0 * math.pi * distance * distance
            # A finite sphere at the proxy distance subtends the same angular
            # disk as a Blender Sun, preserving its angle-controlled penumbra.
            angular_radius = distance * math.tan(max(float(data.angle), 0.0) * 0.5)
            # The distant-point strength preserves irradiance, but its total
            # emitted power is meaningless for light selection: almost all of
            # that sphere misses the scene. Weight it by the power intercepted
            # by the scene's projected bounding sphere instead.
            sampling_weight = sun_strength * math.pi * radius * radius
            document["entities"].append({"type": "point_light", "position": list(position),
                                         "color": color_value, "strength": strength,
                                         "sampling_weight": sampling_weight,
                                         "radius": angular_radius,
                                         "soft_falloff": False})
            warnings.append(f"sun light '{obj.name}' exported as a distant point light")
        elif data.type == "AREA":
            sx = float(data.size)
            sy = float(data.size_y if data.shape in {"RECTANGLE", "ELLIPSE"} else data.size)
            area = max(sx * sy, 1e-8)
            material_id = safe_name("__light_" + obj.name)
            while material_id in document["materials"]:
                material_id += "_2"
            emission = (node_emission if node_emission else
                        [x * float(data.energy) / (math.pi * area) for x in color_value])
            document["materials"][material_id] = {"type": "light", "emission": emission,
                                                  "two_sided": False, "block_ray": False,
                                                  "camera_visible": False}
            if falloff_distance > 0.0:
                document["materials"][material_id]["falloff_distance"] = falloff_distance
            scaled = matrix @ Matrix.Diagonal((sx, sy, 1.0, 1.0))
            document["entities"].append({"type": "mesh", "geometry": "__area_light_quad",
                                         "material": material_id,
                                         "transform": {"matrix": convert_matrix(scaled, coordinate_matrix)},
                                         "raster_light": True})
            if data.shape in {"DISK", "ELLIPSE"}:
                warnings.append(f"area light '{obj.name}' shape {data.shape} exported as a rectangle")
        else:
            warnings.append(f"unsupported light '{obj.name}' ({data.type}) was skipped")
            continue
        light_count += 1

    has_random_walk = any(
        material.get("type") == "shader_graph" and
        ((isinstance(material.get("graph", {}).get("surface", {}).get("subsurface"), (int, float)) and
          material["graph"]["surface"]["subsurface"] > 0.0) or
         isinstance(material.get("graph", {}).get("surface", {}).get("subsurface"), dict))
        for material in document["materials"].values())
    if has_random_walk:
        document["renderer"]["max_bounces"] = 32

    (output_dir / "scene.json").write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    source = Path(bpy.data.filepath)
    report = {
        "source": str(source), "blender_version": bpy.app.version_string,
        "mesh_instances": instance_count, "triangles": triangle_count,
        "materials": len(document["materials"]), "textures": len(image_cache),
        "lights": light_count, "hair_strands": hair_strand_count,
        "hair_points": hair_point_count, "warnings": sorted(set(warnings))
    }
    (output_dir / "conversion-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "README.md").write_text(
        f"# {scene.name}\n\nConverted from Blender's official `{source.name}` benchmark scene with "
        f"[`scripts/blender_to_sparkium.py`](../../../scripts/blender_to_sparkium.py). "
        f"[Upstream benchmark](https://projects.blender.org/blender/blender-benchmarks/src/branch/main/cycles/{source.stem}). "
        f"The directory is self-contained; see `conversion-report.json` "
        f"for conversion statistics and approximations.\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(argv) != 1:
        raise SystemExit("usage: blender -b input.blend --python blender_to_sparkium.py -- OUTPUT_DIR")
    export(argv[0])
