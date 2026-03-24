import bpy
bpy.ops.wm.open_mainfile(filepath='data_generation/assets/base_scene_centered.blend')
e = bpy.data.objects['Empty']
print(f"Empty Loc: {list(e.location)}")
