import bpy
bpy.ops.wm.open_mainfile(filepath='data_generation/assets/base_scene_centered.blend')
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        print(f"Name: {obj.name}, Scale: {list(obj.scale)}, Loc: {list(obj.location)}, Dim: {list(obj.dimensions)}")
