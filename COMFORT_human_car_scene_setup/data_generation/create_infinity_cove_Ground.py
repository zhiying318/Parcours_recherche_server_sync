import bpy
import bmesh
import math
import sys


def create_infinity_cove(size=30.0, wall_height=20.0, curve_radius=5.0, segments=20, base_color=(0.6, 0.6, 0.6, 1.0)):
    """
    创建一个四面无缝的 infinity cove 背景
    size:         地面半径（从中心到墙根的距离）
    wall_height:  墙的总高度
    curve_radius: 地面到墙的过渡曲线半径，值越大过渡越平滑
    segments:     曲线段数，越大越平滑
    base_color:   背景颜色 RGBA
    """
    # 删除原有背景物体
    for name in ["Ground", "Backdrop", "Wall", "InfinityCove"]:
        if name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

    mesh = bpy.data.meshes.new("InfinityCove")
    obj = bpy.data.objects.new("InfinityCove", mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()

    floor_size = size - curve_radius  # 地面平坦区域的半径
    steps = 12  # 地面网格细分数

    # --------------------------------------------------
    # 1. 地面（水平平面）
    # --------------------------------------------------
    floor_verts = []
    for i in range(steps + 1):
        row = []
        for j in range(steps + 1):
            x = -floor_size + 2 * floor_size * i / steps
            y = -floor_size + 2 * floor_size * j / steps
            row.append(bm.verts.new((x, y, 0.0)))
        floor_verts.append(row)

    for i in range(steps):
        for j in range(steps):
            bm.faces.new([
                floor_verts[i][j],
                floor_verts[i+1][j],
                floor_verts[i+1][j+1],
                floor_verts[i][j+1],
            ])

    # --------------------------------------------------
    # 2. 四面弯曲墙
    # 每面墙由两部分组成：
    #   A) 从地面弯曲过渡到竖直的曲面（curve_radius 高度范围内）
    #   B) 曲面顶端到 wall_height 的竖直平面
    # --------------------------------------------------
    def add_curved_wall(axis, positive):
        """
        axis=0: 沿 X 轴方向的墙（+X 或 -X）
        axis=1: 沿 Y 轴方向的墙（+Y 或 -Y）
        positive: True = 正方向，False = 负方向
        """
        sign = 1.0 if positive else -1.0

        # perp_coords: 垂直于墙面方向的坐标列表（steps+1 个点）
        perp_coords = [-floor_size + 2 * floor_size * k / steps for k in range(steps + 1)]

        # 构建所有行的顶点：行 = along 方向上的切片
        # 行 0..segments: 曲面过渡段
        # 行 segments+1:  曲线顶端（与竖直段底部共用）
        # 行 segments+2:  墙顶

        all_rows = []  # all_rows[row_idx] = list of (steps+1) verts

        # --- 曲面段 ---
        for si in range(segments + 1):
            angle = (math.pi / 2.0) * si / segments  # 0 → π/2
            # 沿地面方向（水平）走了多少：从 floor_size 开始往外延伸
            along = floor_size + curve_radius * math.sin(angle)
            # 高度
            up = curve_radius * (1.0 - math.cos(angle))

            row = []
            for perp in perp_coords:
                if axis == 0:
                    v = bm.verts.new((sign * along, perp, up))
                else:
                    v = bm.verts.new((perp, sign * along, up))
                row.append(v)
            all_rows.append(row)

        # --- 竖直段顶部（wall_height） ---
        # all_rows[-1] 已经是曲线顶端（along = floor_size + curve_radius, up = curve_radius）
        # 只需再加一行在 wall_height 处
        wall_along = floor_size + curve_radius
        top_row = []
        for perp in perp_coords:
            if axis == 0:
                v = bm.verts.new((sign * wall_along, perp, wall_height))
            else:
                v = bm.verts.new((perp, sign * wall_along, wall_height))
            top_row.append(v)
        all_rows.append(top_row)

        # --- 建面 ---
        for i in range(len(all_rows) - 1):
            for j in range(steps):
                bm.faces.new([
                    all_rows[i][j],
                    all_rows[i][j+1],
                    all_rows[i+1][j+1],
                    all_rows[i+1][j],
                ])

    add_curved_wall(axis=0, positive=True)   # +X 墙
    add_curved_wall(axis=0, positive=False)  # -X 墙
    add_curved_wall(axis=1, positive=True)   # +Y 墙
    add_curved_wall(axis=1, positive=False)  # -Y 墙

    # --------------------------------------------------
    # 3. 四个角落的填充（避免角落出现缝隙）
    # --------------------------------------------------
    def add_corner(sx, sy):
        """填充一个 45° 角落区域"""
        corner_rows = []
        for si in range(segments + 1):
            angle = (math.pi / 2.0) * si / segments
            along = floor_size + curve_radius * math.sin(angle)
            up = curve_radius * (1.0 - math.cos(angle))
            row = []
            for sj in range(segments + 1):
                angle2 = (math.pi / 2.0) * sj / segments
                along2 = floor_size + curve_radius * math.sin(angle2)
                v = bm.verts.new((sx * along2, sy * along, up))
                row.append(v)
            corner_rows.append(row)

        # 顶部一行
        top_row = []
        wall_along = floor_size + curve_radius
        for sj in range(segments + 1):
            angle2 = (math.pi / 2.0) * sj / segments
            along2 = floor_size + curve_radius * math.sin(angle2)
            v = bm.verts.new((sx * along2, sy * wall_along, wall_height))
            top_row.append(v)
        corner_rows.append(top_row)

        for i in range(len(corner_rows) - 1):
            for j in range(segments):
                bm.faces.new([
                    corner_rows[i][j],
                    corner_rows[i][j+1],
                    corner_rows[i+1][j+1],
                    corner_rows[i+1][j],
                ])

    add_corner(sx=1, sy=1)
    add_corner(sx=1, sy=-1)
    add_corner(sx=-1, sy=1)
    add_corner(sx=-1, sy=-1)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    # --------------------------------------------------
    # 4. 材质
    # --------------------------------------------------
    mat = bpy.data.materials.new(name="CoveMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = base_color
        bsdf.inputs["Roughness"].default_value = 1.0
    obj.data.materials.append(mat)

    print(f"InfinityCove created: size={size}, wall_height={wall_height}, curve_radius={curve_radius}", file=sys.stderr)
    return obj