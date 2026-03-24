import bpy
import bmesh
import math
import sys


def create_infinity_cove(size=30.0, wall_height=20.0, curve_radius=5.0, segments=20, base_color=(0.6, 0.6, 0.6, 1.0)):
    for name in ["Ground", "Backdrop", "Wall", "InfinityCove"]:
        if name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

    mesh = bpy.data.meshes.new("InfinityCove")
    obj = bpy.data.objects.new("InfinityCove", mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()

    floor_size = size - curve_radius
    num_phi = 256       # 水平方向切片数（必须是4的倍数）
    num_r   = 32        # 径向方向切片数（地面平坦区）
    # segments 用于曲线过渡段

    # def box_radius(phi):
    #     """正方形边界在 phi 方向的距离"""
    #     cos_p = abs(math.cos(phi))
    #     sin_p = abs(math.sin(phi))
    #     eps = 1e-9
    #     tx = floor_size / (cos_p + eps)
    #     ty = floor_size / (sin_p + eps)
    #     return min(tx, ty)
    def box_radius(phi, p=2.0): # p=2 is close to a circle, p=6 is more square-like;
        cos_p = abs(math.cos(phi))
        sin_p = abs(math.sin(phi))
        eps = 1e-9
        return 1.0 / (((cos_p / (floor_size + eps)) ** p) + ((sin_p / (floor_size + eps)) ** p) + eps) ** (1.0 / p)

    # --------------------------------------------------
    # 整个cove用一套 (ri, phi_i) 参数统一描述：
    #
    # ri = 0              : 中心点 (0,0,0)
    # ri = 1..num_r       : 地面平坦区，r 从 0 线性增加到 box_radius(phi)
    # ri = num_r+1..
    #      num_r+segments : 曲线过渡区，从地面弯曲到竖直
    # ri = num_r+segments+1: 墙顶
    #
    # 所有顶点在同一网格内，无需拼接
    # --------------------------------------------------

    total_rings = num_r + segments + 2  # 包含中心圈和墙顶圈

    # 构建所有顶点：verts[ri][phi_i]
    # ri=0 是中心点（所有 phi 共用一个点，用 num_phi 个重合顶点表示以便建面）
    verts = []

    for ri in range(total_rings + 1):
        ring = []
        for pi in range(num_phi):
            phi = 2 * math.pi * pi / num_phi
            cos_p = math.cos(phi)
            sin_p = math.sin(phi)
            r_box = box_radius(phi)
            # r_box = floor_size # try to get a perfect circle to compare, ca va not very perfect.

            if ri == 0:
                # 中心
                x, y, z = 0.0, 0.0, 0.0

            elif ri <= num_r:
                # 地面平坦区：r 从 0 线性到 r_box
                t = ri / num_r
                r = t * r_box
                x, y, z = r * cos_p, r * sin_p, 0.0

            elif ri <= num_r + segments:
                # 曲线过渡区
                si = ri - num_r  # 1..segments
                angle = (math.pi / 2.0) * si / segments
                r = r_box + curve_radius * math.sin(angle)
                z = curve_radius * (1.0 - math.cos(angle))
                x, y = r * cos_p, r * sin_p

            else:
                # 墙顶
                r = r_box + curve_radius
                x, y, z = r * cos_p, r * sin_p, wall_height

            ring.append(bm.verts.new((x, y, z)))
        verts.append(ring)

    # --------------------------------------------------
    # 建面：每相邻两圈之间建四边形（phi 方向环绕）
    # --------------------------------------------------
    for ri in range(total_rings):
        for pi in range(num_phi):
            pi_next = (pi + 1) % num_phi
            v00 = verts[ri][pi]
            v01 = verts[ri][pi_next]
            v10 = verts[ri + 1][pi]
            v11 = verts[ri + 1][pi_next]
            try:
                bm.faces.new([v00, v01, v11, v10])
            except Exception:
                pass  # 跳过重复面（中心点可能重合）

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    # --------------------------------------------------
    # 材质
    # --------------------------------------------------
    mat = bpy.data.materials.new(name="CoveMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = base_color
        bsdf.inputs["Roughness"].default_value = 1.0
    obj.data.materials.append(mat)

    print(f"InfinityCove (unified radial): size={size}, wall_height={wall_height}, curve_radius={curve_radius}", file=sys.stderr)

    for poly in obj.data.polygons:
        poly.use_smooth = True

    return obj