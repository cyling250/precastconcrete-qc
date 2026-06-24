"""
构件组装：七种装配式混凝土构件的参数化点云生成。

设计原则（与旧版 generate_synthetic_dataset.py 一致）：
  - 混凝土：仅表面采样（模拟扫描仪只能捕获外表面）
  - 纵向钢筋：仅端面露出，不贯穿混凝土内部
  - U 形发夹筋（叠合梁）：倒 U 形立在混凝土上表面
  - 箍筋：沿构件长度均匀布置的闭环钢筋
  - 套筒：在纵筋位置替换钢筋，从端面向内延伸
"""

import numpy as np
from typing import Optional

from ._config import (
    LABEL_CONCRETE,
    LABEL_REBAR,
    LABEL_SLEEVE,
    COLUMN_WIDTH_RANGE,
    COLUMN_DEPTH_RANGE,
    COLUMN_HEIGHT_RANGE,
    L_SHAPED_COLUMN_FLANGE_RANGE,
    L_SHAPED_COLUMN_WEB_RANGE,
    BEAM_WIDTH_RANGE,
    BEAM_HEIGHT_RANGE,
    BEAM_LENGTH_RANGE,
    T_BEAM_FLANGE_WIDTH_RANGE,
    T_BEAM_FLANGE_HEIGHT_RANGE,
    T_BEAM_WEB_WIDTH_RANGE,
    T_BEAM_TOTAL_HEIGHT_RANGE,
    REBAR_DIAM_RANGE,
    CONCRETE_COVER,
    CONCRETE_ROUGHNESS,
    STIRRUP_DIAM_RANGE,
    STIRRUP_SPACING_RANGE,
    SLEEVE_INNER_DIAM_RANGE,
    SLEEVE_DEPTH_RANGE,
    SLEEVE_WALL_THICKNESS,
    SLEEVE_MIN_COVER,
    REBAR_POINTS_PER_LENGTH,
    STANDARD_REBAR_DIAMETERS,
    SLEEVE_SPEC_MAP,
)
from ._concrete import sample_concrete_prism
from ._rebar import (
    generate_stirrups_rect,
    generate_stirrups_polygon,
    generate_u_hairpins,
)
from ._sleeve import generate_sleeves_for_component, get_sleeve_hole_circles
from ._geometry import sample_cylinder_segment, sample_polygon_surface


# ─── 辅助函数 ─────────────────────────────────────────────


def _shrink_polygon(polygon: np.ndarray, offset: float) -> Optional[np.ndarray]:
    """多边形内缩（保护层）。"""
    from shapely.geometry import Polygon as ShapelyPolygon

    poly = ShapelyPolygon(polygon)
    if not poly.is_valid:
        return None
    shrunk = poly.buffer(-offset, resolution=8)
    if shrunk.is_empty or not shrunk.is_valid:
        return None
    ext = np.array(shrunk.exterior.coords)[:-1]
    return ext


def _get_rebar_positions_rect(
    half_w: float, half_d: float, cover: float, count: int, rng: np.random.Generator
) -> list:
    """矩形截面纵筋位置（角部优先 + 沿边均匀补足）。"""
    inner_w, inner_d = half_w - cover, half_d - cover
    positions = [
        (-inner_w, -inner_d),
        (inner_w, -inner_d),
        (-inner_w, inner_d),
        (inner_w, inner_d),
    ]
    if count > 4:
        n_extra = count - 4
        per_side = n_extra // 2
        remainder = n_extra % 2
        n_fb = per_side + remainder
        for i in range(n_fb):
            x = -inner_w + (2 * inner_w) * (i + 1) / (n_fb + 1)
            positions.append((x, -inner_d))
            if len(positions) >= count:
                break
        if len(positions) < count:
            for i in range(per_side):
                y = -inner_d + (2 * inner_d) * (i + 1) / (per_side + 1)
                positions.append((-inner_w, y))
    return positions[:count]


def _get_rebar_positions_polygon(
    polygon: np.ndarray, cover: float, count: int, rng: np.random.Generator
) -> list:
    """多边形截面纵筋位置（角部优先 + 长边中部补足）。"""
    inner = _shrink_polygon(polygon, cover)
    if inner is None or len(inner) < 3:
        inner = polygon

    # 跳过间距过小的相邻顶点（消除凹角处 buffer 产生的圆弧伪影）
    min_dist = cover * 0.5
    merged = [inner[0]]
    for i in range(1, len(inner)):
        if np.linalg.norm(inner[i] - inner[i - 1]) >= min_dist:
            merged.append(inner[i])
    inner = np.array(merged)

    positions = [(inner[i, 0], inner[i, 1]) for i in range(len(inner))]
    if count > len(positions):
        n_extra = count - len(positions)
        edges, edge_lens = [], []
        for i in range(len(inner)):
            j = (i + 1) % len(inner)
            edges.append((inner[i], inner[j]))
            edge_lens.append(np.linalg.norm(inner[j] - inner[i]))
        total_len = sum(edge_lens)
        if total_len == 0:
            return positions[:count]
        extra_per_edge = [max(1, int(n_extra * el / total_len)) for el in edge_lens]
        while sum(extra_per_edge) < n_extra:
            extra_per_edge[np.argmax(edge_lens)] += 1
        while sum(extra_per_edge) > n_extra:
            extra_per_edge[np.argmax(extra_per_edge)] -= 1
        for (e0, e1), n_e in zip(edges, extra_per_edge):
            if n_e <= 0:
                continue
            for k in range(1, n_e + 1):
                t = k / (n_e + 1)
                pt = e0 + t * (e1 - e0)
                positions.append((pt[0], pt[1]))
    return positions[:count]


def _generate_rebar_ends(
    positions_2d: list,
    start_x: float,
    end_x: float,
    rebar_diam: float,
    protrusion: float,
    density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """在纵筋位置生成两端露出的钢筋（不贯穿混凝土内部）。"""
    radius = rebar_diam / 2.0
    pts_list = []
    for py, pz in positions_2d:
        # 左端露出
        left_start = np.array([start_x, py, pz])
        left_end = np.array([start_x - protrusion, py, pz])
        pts_list.append(
            sample_cylinder_segment(left_start, left_end, radius, density, rng)
        )
        # 右端露出
        right_start = np.array([end_x, py, pz])
        right_end = np.array([end_x + protrusion, py, pz])
        pts_list.append(
            sample_cylinder_segment(right_start, right_end, radius, density, rng)
        )
    return np.vstack(pts_list) if pts_list else np.empty((0, 3))


def _generate_rebar_ends_z(
    positions_2d: list,
    z_bottom: float,
    z_top: float,
    rebar_diam: float,
    protrusion: float,
    density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """在纵筋位置生成两端露出的钢筋（柱：沿 Z 轴）。"""
    radius = rebar_diam / 2.0
    pts_list = []
    for px, py in positions_2d:
        # 底部露出
        bot_start = np.array([px, py, z_bottom])
        bot_end = np.array([px, py, z_bottom - protrusion])
        pts_list.append(
            sample_cylinder_segment(bot_start, bot_end, radius, density, rng)
        )
        # 顶部露出
        top_start = np.array([px, py, z_top])
        top_end = np.array([px, py, z_top + protrusion])
        pts_list.append(
            sample_cylinder_segment(top_start, top_end, radius, density, rng)
        )
    return np.vstack(pts_list) if pts_list else np.empty((0, 3))


def _remove_concrete_holes(
    con_pts: np.ndarray,
    positions_2d: list,
    inner_diam: float,
    axis: int,
    end_coords: tuple[float, float],
    tolerance: float = 10.0,
) -> np.ndarray:
    """
    从混凝土点云中移除套筒端口处的圆形孔区域。

    Parameters
    ----------
    con_pts : (N, 3) 混凝土点云
    positions_2d : list of (c1, c2) 套筒位置
    inner_diam : float 套筒内径 (mm)
    axis : int 轴向 (0=X, 2=Z)
    end_coords : (coord_min, coord_max) 两端面坐标
    tolerance : float 端面判定容差 (mm)

    Returns
    -------
    ndarray 挖孔后的混凝土点云
    """
    radius = inner_diam / 2.0
    coord_min, coord_max = end_coords
    mask = np.ones(len(con_pts), dtype=bool)

    for cx, cy in positions_2d:
        for end_coord in [coord_min, coord_max]:
            if axis == 0:
                # X 轴向（梁）：端面在 x=coord，截面在 YZ
                near_face = np.abs(con_pts[:, 0] - end_coord) < tolerance
                in_circle = (con_pts[:, 1] - cx) ** 2 + (
                    con_pts[:, 2] - cy
                ) ** 2 < radius**2
            else:
                # Z 轴向（柱）：端面在 z=coord，截面在 XY
                near_face = np.abs(con_pts[:, 2] - end_coord) < tolerance
                in_circle = (con_pts[:, 0] - cx) ** 2 + (
                    con_pts[:, 1] - cy
                ) ** 2 < radius**2
            mask &= ~(near_face & in_circle)
    return con_pts[mask]


def _add_sleeves_to_ends(
    positions_2d: list,
    end_x: float,
    sleeve_inner_diam: float,
    sleeve_depth: float,
    density: float,
    rng: np.random.Generator,
    direction: int = 1,
    axis: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    在端面纵筋位置生成套筒（从端面向内延伸）。
    direction: 1=正向, -1=负向
    axis: 轴向 (0=X, 2=Z)
    """
    sl_pts, sl_lbls = generate_sleeves_for_component(
        positions_2d,
        end_x,
        sleeve_inner_diam,
        sleeve_depth,
        density,
        rng,
        direction=direction,
        axis=axis,
    )
    return sl_pts, sl_lbls


# ─── 柱 构建器（竖向，Z 为高度轴） ─────────────────────


def build_rectangular_column(
    width: Optional[float] = None,
    depth: Optional[float] = None,
    height: Optional[float] = None,
    rebar_diam: Optional[float] = None,
    n_rebars: Optional[int] = None,
    stirrup_diam: Optional[float] = None,
    stirrup_spacing: Optional[float] = None,
    density: float = 0.01,
    rng: Optional[np.random.Generator] = None,
    variant: str = "rebar",
    sleeve_inner_diam: Optional[float] = None,
    sleeve_depth: Optional[float] = None,
) -> dict:
    """
    矩形柱：长方体 + 箍筋 + 纵筋(rebar) 或套筒(sleeve)。
    variant="rebar": 纵筋两端露出
    variant="sleeve": 套筒从端面向内延伸，端面混凝土挖孔
    Z 为高度轴，截面在 XY 平面。
    """
    if rng is None:
        rng = np.random.default_rng()
    w = width or rng.uniform(*COLUMN_WIDTH_RANGE)
    d = depth or rng.uniform(*COLUMN_DEPTH_RANGE)
    h = height or rng.uniform(*COLUMN_HEIGHT_RANGE)
    rd = rebar_diam or float(rng.choice(STANDARD_REBAR_DIAMETERS))
    nr = n_rebars or rng.integers(8, 16)
    sd = stirrup_diam or rng.uniform(*STIRRUP_DIAM_RANGE)
    ss = stirrup_spacing or rng.uniform(*STIRRUP_SPACING_RANGE)
    protrusion = 200.0

    # 保护层厚度：套筒 variant 需保证套筒外壁 ≥ SLEEVE_MIN_COVER
    if variant == "sleeve":
        # 套筒规格匹配钢筋直径 (JG/T 398-2019, JGJ 355-2015 §3.1.2)
        if sleeve_inner_diam is None:
            sid = SLEEVE_SPEC_MAP[rd][1]  # 从映射表获取套筒内径
        else:
            sid = sleeve_inner_diam
        sdep = sleeve_depth or rng.uniform(*SLEEVE_DEPTH_RANGE)
        sleeve_outer_radius = sid / 2 + SLEEVE_WALL_THICKNESS
        cc = sleeve_outer_radius + SLEEVE_MIN_COVER
    else:
        sid = 0.0  # rebar variant 不使用套筒
        sdep = 0.0
        cc = CONCRETE_COVER

    half_w, half_d = w / 2, d / 2
    half_h = h / 2

    # 混凝土：棱柱表面采样（Z 轴拉伸）
    rect_poly = np.array(
        [
            [-half_w, -half_d],
            [half_w, -half_d],
            [half_w, half_d],
            [-half_w, half_d],
        ]
    )
    con_pts, con_lbls = sample_concrete_prism(rect_poly, h, density, rng, roughness=CONCRETE_ROUGHNESS)

    # 纵筋位置（矩形截面）
    positions = _get_rebar_positions_rect(half_w, half_d, cc, nr, rng)

    polygon_rect = np.array(
        [
            [-half_w, -half_d],
            [half_w, -half_d],
            [half_w, half_d],
            [-half_w, half_d],
        ]
    )
    # 箍筋（沿 Z 轴，覆盖全柱高）
    st_pts, st_lbls = generate_stirrups_rect(
        polygon_rect, -half_h, half_h, sd, cc, ss, density, rng
    )

    # 纵筋或套筒（二选一）
    all_pts, all_lbls = [con_pts, st_pts], [con_lbls, st_lbls]
    if variant == "rebar":
        reb_pts = _generate_rebar_ends_z(positions, -half_h, half_h, rd, protrusion, density, rng)
        all_pts.append(reb_pts)
        all_lbls.append(np.full(len(reb_pts), LABEL_REBAR, dtype=np.int32))
    else:
        rebar_xy = [(p[0], p[1]) for p in positions]
        bot_sl, bsl = generate_sleeves_for_component(
            rebar_xy,
            -half_h,
            sid,
            sdep,
            density,
            rng,
            direction=-1,
        )
        top_sl, tsl = generate_sleeves_for_component(
            rebar_xy,
            half_h,
            sid,
            sdep,
            density,
            rng,
            direction=1,
        )
        all_pts.extend([bot_sl, top_sl])
        all_lbls.extend([bsl, tsl])
        # 混凝土端面挖孔
        con_pts = _remove_concrete_holes(con_pts, positions, sid, 2, (-half_h, half_h))
        all_pts[0] = con_pts
        all_lbls[0] = np.full(len(con_pts), LABEL_CONCRETE, dtype=np.int32)

    points = np.vstack(all_pts)
    labels = np.concatenate(all_lbls)

    return {
        "points": points,
        "labels": labels,
        "metadata": {
            "type": "rectangular_column",
            "width": w,
            "depth": d,
            "height": h,
            "rebar_diam": rd,
            "n_rebars": nr,
            "variant": variant,
            "n_points_total": len(points),
            "density": density,
        },
    }


def build_l_shaped_column(
    flange: Optional[float] = None,
    web: Optional[float] = None,
    height: Optional[float] = None,
    rebar_diam: Optional[float] = None,
    n_rebars: Optional[int] = None,
    stirrup_diam: Optional[float] = None,
    stirrup_spacing: Optional[float] = None,
    density: float = 0.01,
    rng: Optional[np.random.Generator] = None,
    variant: str = "rebar",
    sleeve_inner_diam: Optional[float] = None,
    sleeve_depth: Optional[float] = None,
) -> dict:
    """L 形柱：L 形截面棱柱 + 箍筋 + 纵筋(rebar) 或套筒(sleeve)。"""
    if rng is None:
        rng = np.random.default_rng()
    w = flange or rng.uniform(*L_SHAPED_COLUMN_FLANGE_RANGE)
    d = web or rng.uniform(*L_SHAPED_COLUMN_WEB_RANGE)
    wf = w * rng.uniform(0.35, 0.65)
    dw = d * rng.uniform(0.35, 0.65)
    h = height or rng.uniform(*COLUMN_HEIGHT_RANGE)
    rd = rebar_diam or float(rng.choice(STANDARD_REBAR_DIAMETERS))
    nr = n_rebars or rng.integers(8, 16)
    sd = stirrup_diam or rng.uniform(*STIRRUP_DIAM_RANGE)
    ss = stirrup_spacing or rng.uniform(*STIRRUP_SPACING_RANGE)
    protrusion = 200.0

    # 保护层厚度：套筒 variant 需保证套筒外壁 ≥ SLEEVE_MIN_COVER
    if variant == "sleeve":
        # 套筒规格匹配钢筋直径 (JG/T 398-2019, JGJ 355-2015 §3.1.2)
        if sleeve_inner_diam is None:
            sid = SLEEVE_SPEC_MAP[rd][1]  # 从映射表获取套筒内径
        else:
            sid = sleeve_inner_diam
        sdep = sleeve_depth or rng.uniform(*SLEEVE_DEPTH_RANGE)
        sleeve_outer_radius = sid / 2 + SLEEVE_WALL_THICKNESS
        cc = sleeve_outer_radius + SLEEVE_MIN_COVER
    else:
        sid = 0.0  # rebar variant 不使用套筒
        sdep = 0.0
        cc = CONCRETE_COVER

    # L 形截面（XY 平面）
    polygon = np.array(
        [
            [0.0, 0.0],
            [w, 0.0],
            [w, wf],
            [dw, wf],
            [dw, d],
            [0.0, d],
        ]
    )
    center = polygon.mean(axis=0)
    polygon -= center

    # 混凝土
    con_pts, con_lbls = sample_concrete_prism(polygon, h, density, rng, roughness=CONCRETE_ROUGHNESS)

    # 纵筋位置
    positions = _get_rebar_positions_polygon(polygon, cc, nr, rng)

    # 箍筋（L 形截面）
    half_h = h / 2
    st_pts, st_lbls = generate_stirrups_polygon(
        polygon, -half_h, half_h, sd, cc, ss, density, rng
    )

    # 纵筋或套筒（二选一）
    all_pts, all_lbls = [con_pts, st_pts], [con_lbls, st_lbls]
    if variant == "rebar":
        reb_pts = _generate_rebar_ends_z(
            positions, -half_h, half_h, rd, protrusion, density, rng
        )
        all_pts.append(reb_pts)
        all_lbls.append(np.full(len(reb_pts), LABEL_REBAR, dtype=np.int32))
    else:
        rebar_xy = [(p[0], p[1]) for p in positions]
        bot_sl, bsl = generate_sleeves_for_component(
            rebar_xy,
            -half_h,
            sid,
            sdep,
            density,
            rng,
            direction=-1,
        )
        top_sl, tsl = generate_sleeves_for_component(
            rebar_xy,
            half_h,
            sid,
            sdep,
            density,
            rng,
            direction=1,
        )
        all_pts.extend([bot_sl, top_sl])
        all_lbls.extend([bsl, tsl])
        # 混凝土端面挖孔
        con_pts = _remove_concrete_holes(con_pts, positions, sid, 2, (-half_h, half_h))
        all_pts[0] = con_pts
        all_lbls[0] = np.full(len(con_pts), LABEL_CONCRETE, dtype=np.int32)

    points = np.vstack(all_pts)
    labels = np.concatenate(all_lbls)

    return {
        "points": points,
        "labels": labels,
        "metadata": {
            "type": "l_shaped_column",
            "flange": w,
            "web": d,
            "height": h,
            "rebar_diam": rd,
            "n_rebars": nr,
            "variant": variant,
            "n_points_total": len(points),
            "density": density,
        },
    }


# ─── 梁 构建器（横向，X 为长度轴，YZ 为截面） ────────


def _sample_beam_cross_section(
    cross_2d: np.ndarray, length: float, density: float, rng: np.random.Generator,
    roughness: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    梁表面采样（X 为长轴，YZ 为截面）。
    对 T/L 等复杂截面使用多边形分解填充。
    """
    if not np.allclose(cross_2d[0], cross_2d[-1]):
        cross_2d = np.vstack([cross_2d, cross_2d[0:1]])
    n_edges = len(cross_2d) - 1
    edge_lengths = [
        np.linalg.norm(cross_2d[i + 1] - cross_2d[i]) for i in range(n_edges)
    ]
    perimeter = sum(edge_lengths)
    yv, zv = cross_2d[:, 0], cross_2d[:, 1]
    area = 0.5 * abs(
        sum(yv[i] * zv[(i + 1) % len(cross_2d)] for i in range(len(cross_2d)))
        - sum(zv[i] * yv[(i + 1) % len(cross_2d)] for i in range(len(cross_2d)))
    )
    total_surface = 2 * area + perimeter * length
    n_pts = max(200, int(total_surface * density))

    n_endface = int(max(1, n_pts * area / total_surface))
    n_sides = max(1, n_pts - 2 * n_endface)

    pts_list = []

    # 两端面：使用多边形面采样
    endface_density = n_endface / area if area > 0 else density
    for x_off in [0.0, length]:
        endface_pts = sample_polygon_surface(cross_2d[:-1], endface_density, rng)
        # 端面在 YZ 平面，X=x_off
        pts_list.append(
            np.column_stack([np.full(len(endface_pts), x_off), endface_pts])
        )

    # 侧面：沿边拉伸
    for i in range(n_edges):
        n_edge = (
            max(1, int(n_sides * edge_lengths[i] / perimeter))
            if perimeter > 0
            else max(1, n_sides // n_edges)
        )
        if i == n_edges - 1:
            n_edge = n_sides - sum(
                max(1, int(n_sides * el / perimeter)) for el in edge_lengths[:-1]
            )
        p1, p2 = cross_2d[i], cross_2d[i + 1]
        su = rng.uniform(0, 1, n_edge)
        sx = rng.uniform(0, length, n_edge)
        edge_2d = p1[None, :] + su[:, None] * (p2 - p1)[None, :]
        pts_list.append(np.column_stack([sx, edge_2d[:, 0], edge_2d[:, 1]]))

    pts = np.vstack(pts_list)
    # 表面粗糙度
    if roughness is not None and roughness > 0:
        from ._concrete import _apply_surface_roughness
        pts = _apply_surface_roughness(pts, roughness, rng)
    lbls = np.full(len(pts), LABEL_CONCRETE, dtype=np.int32)
    return pts, lbls


def build_rectangular_beam(
    width: Optional[float] = None,
    height: Optional[float] = None,
    length: Optional[float] = None,
    rebar_diam: Optional[float] = None,
    n_rebars: Optional[int] = None,
    stirrup_diam: Optional[float] = None,
    stirrup_spacing: Optional[float] = None,
    density: float = 0.01,
    rng: Optional[np.random.Generator] = None,
    variant: str = "rebar",
    sleeve_inner_diam: Optional[float] = None,
    sleeve_depth: Optional[float] = None,
) -> dict:
    """矩形梁：长方体 + 箍筋 + 纵筋(rebar) 或套筒(sleeve, axis=0)。"""
    if rng is None:
        rng = np.random.default_rng()
    b = width or rng.uniform(*BEAM_WIDTH_RANGE)
    h = height or rng.uniform(*BEAM_HEIGHT_RANGE)
    l = length or rng.uniform(*BEAM_LENGTH_RANGE)
    rd = rebar_diam or float(rng.choice(STANDARD_REBAR_DIAMETERS))
    nr = n_rebars or rng.integers(4, 12)
    sd = stirrup_diam or rng.uniform(*STIRRUP_DIAM_RANGE)
    ss = stirrup_spacing or rng.uniform(*STIRRUP_SPACING_RANGE)
    protrusion = 200.0

    # 保护层厚度：套筒 variant 需保证套筒外壁 ≥ SLEEVE_MIN_COVER
    if variant == "sleeve":
        # 套筒规格匹配钢筋直径 (JG/T 398-2019, JGJ 355-2015 §3.1.2)
        if sleeve_inner_diam is None:
            sid = SLEEVE_SPEC_MAP[rd][1]  # 从映射表获取套筒内径
        else:
            sid = sleeve_inner_diam
        sdep = sleeve_depth or rng.uniform(*SLEEVE_DEPTH_RANGE)
        sleeve_outer_radius = sid / 2 + SLEEVE_WALL_THICKNESS
        cc = sleeve_outer_radius + SLEEVE_MIN_COVER
    else:
        sid = 0.0  # rebar variant 不使用套筒
        sdep = 0.0
        cc = CONCRETE_COVER

    # 混凝土：梁表面采样
    cross = np.array(
        [
            [-b / 2, -h / 2],
            [b / 2, -h / 2],
            [b / 2, h / 2],
            [-b / 2, h / 2],
        ]
    )
    con_pts, con_lbls = _sample_beam_cross_section(cross, l, density, rng, roughness=CONCRETE_ROUGHNESS)

    # 纵筋位置
    positions = _get_rebar_positions_rect(b / 2, h / 2, cc, nr, rng)

    # 箍筋
    st_pts, st_lbls = generate_stirrups_rect(
        cross, 0, l, sd, cc, ss, density, rng, axis=0
    )

    # 纵筋或套筒（二选一）
    all_pts, all_lbls = [con_pts, st_pts], [con_lbls, st_lbls]
    if variant == "rebar":
        reb_pts = _generate_rebar_ends(positions, 0, l, rd, protrusion, density, rng)
        all_pts.append(reb_pts)
        all_lbls.append(np.full(len(reb_pts), LABEL_REBAR, dtype=np.int32))
    else:
        rebar_xy = [(py, pz) for (py, pz) in positions]
        bot_sl, bsl = _add_sleeves_to_ends(
            rebar_xy, 0, sid, sdep, density, rng, -1, axis=0
        )
        top_sl, tsl = _add_sleeves_to_ends(
            rebar_xy, l, sid, sdep, density, rng, 1, axis=0
        )
        all_pts.extend([bot_sl, top_sl])
        all_lbls.extend([bsl, tsl])
        # 混凝土端面挖孔
        con_pts = _remove_concrete_holes(con_pts, positions, sid, 0, (0, l))
        all_pts[0] = con_pts
        all_lbls[0] = np.full(len(con_pts), LABEL_CONCRETE, dtype=np.int32)

    points = np.vstack(all_pts)
    labels = np.concatenate(all_lbls)

    return {
        "points": points,
        "labels": labels,
        "metadata": {
            "type": "rectangular_beam",
            "width": b,
            "height": h,
            "length": l,
            "rebar_diam": rd,
            "n_rebars": nr,
            "variant": variant,
            "n_points_total": len(points),
            "density": density,
        },
    }


def build_t_beam(
    flange_width: Optional[float] = None,
    flange_height: Optional[float] = None,
    web_width: Optional[float] = None,
    total_height: Optional[float] = None,
    length: Optional[float] = None,
    rebar_diam: Optional[float] = None,
    n_rebars: Optional[int] = None,
    stirrup_diam: Optional[float] = None,
    stirrup_spacing: Optional[float] = None,
    density: float = 0.01,
    rng: Optional[np.random.Generator] = None,
    variant: str = "rebar",
    sleeve_inner_diam: Optional[float] = None,
    sleeve_depth: Optional[float] = None,
) -> dict:
    """T 形梁：T 形截面 + 箍筋 + 纵筋(rebar) 或套筒(sleeve, axis=0)。"""
    if rng is None:
        rng = np.random.default_rng()
    B = flange_width or rng.uniform(*T_BEAM_FLANGE_WIDTH_RANGE)
    hf = flange_height or rng.uniform(*T_BEAM_FLANGE_HEIGHT_RANGE)
    bw = web_width or rng.uniform(*T_BEAM_WEB_WIDTH_RANGE)
    h = total_height or rng.uniform(*T_BEAM_TOTAL_HEIGHT_RANGE)
    l = length or rng.uniform(*BEAM_LENGTH_RANGE)
    rd = rebar_diam or float(rng.choice(STANDARD_REBAR_DIAMETERS))
    nr = n_rebars or rng.integers(6, 14)
    sd = stirrup_diam or rng.uniform(*STIRRUP_DIAM_RANGE)
    ss = stirrup_spacing or rng.uniform(*STIRRUP_SPACING_RANGE)
    protrusion = 200.0

    # 保护层厚度：套筒 variant 需保证套筒外壁 ≥ SLEEVE_MIN_COVER
    if variant == "sleeve":
        # 套筒规格匹配钢筋直径 (JG/T 398-2019, JGJ 355-2015 §3.1.2)
        if sleeve_inner_diam is None:
            sid = SLEEVE_SPEC_MAP[rd][1]  # 从映射表获取套筒内径
        else:
            sid = sleeve_inner_diam
        sdep = sleeve_depth or rng.uniform(*SLEEVE_DEPTH_RANGE)
        sleeve_outer_radius = sid / 2 + SLEEVE_WALL_THICKNESS
        cc = sleeve_outer_radius + SLEEVE_MIN_COVER
    else:
        sid = 0.0  # rebar variant 不使用套筒
        sdep = 0.0
        cc = CONCRETE_COVER

    # T 形截面居中（翼缘在上，腹板在下）
    B2, bw2 = B / 2, bw / 2
    half_z = h / 2
    cross = np.array(
        [
            [-B2, half_z],
            [B2, half_z],
            [B2, half_z - hf],
            [bw2, half_z - hf],
            [bw2, -half_z],
            [-bw2, -half_z],
            [-bw2, half_z - hf],
            [-B2, half_z - hf],
        ]
    )

    con_pts, con_lbls = _sample_beam_cross_section(cross, l, density, rng, roughness=CONCRETE_ROUGHNESS)

    # 纵筋位置（T 形截面：底部和翼缘顶部）
    positions = [
        (-bw2 + cc, -half_z + cc),  # 腹板左下
        (bw2 - cc, -half_z + cc),  # 腹板右下
        (-B2 + cc, half_z - cc),  # 翼缘左上
        (B2 - cc, half_z - cc),  # 翼缘右上
    ]
    if nr > 4:
        for i in range(1, nr - 3):
            f = i / (nr - 3)
            y_pos = -B2 + cc + (B - 2 * cc) * f
            positions.append((y_pos, half_z - cc))

    # 箍筋（T 形截面）
    st_pts, st_lbls = generate_stirrups_polygon(
        cross, 0, l, sd, cc, ss, density, rng, axis=0
    )

    # 纵筋或套筒（二选一）
    all_pts, all_lbls = [con_pts, st_pts], [con_lbls, st_lbls]
    if variant == "rebar":
        reb_pts = _generate_rebar_ends(positions, 0, l, rd, protrusion, density, rng)
        all_pts.append(reb_pts)
        all_lbls.append(np.full(len(reb_pts), LABEL_REBAR, dtype=np.int32))
    else:
        rebar_xy = [(py, pz) for (py, pz) in positions[:nr]]
        bot_sl, bsl = _add_sleeves_to_ends(
            rebar_xy, 0, sid, sdep, density, rng, -1, axis=0
        )
        top_sl, tsl = _add_sleeves_to_ends(
            rebar_xy, l, sid, sdep, density, rng, 1, axis=0
        )
        all_pts.extend([bot_sl, top_sl])
        all_lbls.extend([bsl, tsl])
        # 混凝土端面挖孔
        con_pts = _remove_concrete_holes(con_pts, positions, sid, 0, (0, l))
        all_pts[0] = con_pts
        all_lbls[0] = np.full(len(con_pts), LABEL_CONCRETE, dtype=np.int32)

    points = np.vstack(all_pts)
    labels = np.concatenate(all_lbls)

    return {
        "points": points,
        "labels": labels,
        "metadata": {
            "type": "t_beam",
            "flange_width": B,
            "flange_height": hf,
            "web_width": bw,
            "total_height": h,
            "length": l,
            "rebar_diam": rd,
            "n_rebars": nr,
            "variant": variant,
            "n_points_total": len(points),
            "density": density,
        },
    }


# ─── 叠合梁 构建器（基础梁 + U 形发夹筋） ────────────


def build_rectangular_composite_beam(
    width: Optional[float] = None,
    height: Optional[float] = None,
    length: Optional[float] = None,
    rebar_diam: Optional[float] = None,
    n_rebars: Optional[int] = None,
    stirrup_diam: Optional[float] = None,
    stirrup_spacing: Optional[float] = None,
    hairpin_diam: Optional[float] = None,
    hairpin_spacing: Optional[float] = None,
    density: float = 0.01,
    rng: Optional[np.random.Generator] = None,
    variant: str = "rebar",
    sleeve_inner_diam: Optional[float] = None,
    sleeve_depth: Optional[float] = None,
) -> dict:
    """矩形叠合梁：矩形梁 + U 形发夹筋（上表面）+ 箍筋 + 纵筋/套筒。"""
    result = build_rectangular_beam(
        width=width,
        height=height,
        length=length,
        rebar_diam=rebar_diam,
        n_rebars=n_rebars,
        stirrup_diam=stirrup_diam,
        stirrup_spacing=stirrup_spacing,
        density=density,
        rng=rng,
        variant=variant,
        sleeve_inner_diam=sleeve_inner_diam,
        sleeve_depth=sleeve_depth,
    )
    b = width or result["metadata"]["width"]
    h = height or result["metadata"]["height"]

    hp_diam = hairpin_diam or 10.0
    hp_spacing = hairpin_spacing or 200.0

    hp_pts, hp_lbls = generate_u_hairpins(
        beam_length=result["metadata"]["length"],
        beam_top_y=b / 2,
        beam_bottom_y=-b / 2,
        concrete_top_z=h / 2,
        diam=hp_diam,
        spacing_x=hp_spacing,
        density=density,
        rng=rng or np.random.default_rng(),
    )
    if len(hp_pts) > 0:
        result["points"] = np.vstack([result["points"], hp_pts])
        result["labels"] = np.concatenate([result["labels"], hp_lbls])
    result["metadata"]["type"] = "rectangular_composite_beam"
    result["metadata"]["hairpin_diam"] = hp_diam
    result["metadata"]["hairpin_spacing"] = hp_spacing
    return result


def build_t_composite_beam(
    flange_width: Optional[float] = None,
    flange_height: Optional[float] = None,
    web_width: Optional[float] = None,
    total_height: Optional[float] = None,
    length: Optional[float] = None,
    rebar_diam: Optional[float] = None,
    n_rebars: Optional[int] = None,
    stirrup_diam: Optional[float] = None,
    stirrup_spacing: Optional[float] = None,
    hairpin_diam: Optional[float] = None,
    hairpin_spacing: Optional[float] = None,
    density: float = 0.01,
    rng: Optional[np.random.Generator] = None,
    variant: str = "rebar",
    sleeve_inner_diam: Optional[float] = None,
    sleeve_depth: Optional[float] = None,
) -> dict:
    """T 形叠合梁：T 形梁 + U 形发夹筋（翼缘上表面）+ 纵筋/套筒。"""
    result = build_t_beam(
        flange_width=flange_width,
        flange_height=flange_height,
        web_width=web_width,
        total_height=total_height,
        length=length,
        rebar_diam=rebar_diam,
        n_rebars=n_rebars,
        stirrup_diam=stirrup_diam,
        stirrup_spacing=stirrup_spacing,
        density=density,
        rng=rng,
        variant=variant,
        sleeve_inner_diam=sleeve_inner_diam,
        sleeve_depth=sleeve_depth,
    )
    meta = result["metadata"]
    B = meta["flange_width"]
    hp_diam = hairpin_diam or 10.0
    hp_spacing = hairpin_spacing or 200.0

    hp_pts, hp_lbls = generate_u_hairpins(
        beam_length=meta["length"],
        beam_top_y=meta["web_width"] / 2,
        beam_bottom_y=-meta["web_width"] / 2,
        concrete_top_z=meta["total_height"] / 2,
        diam=hp_diam,
        spacing_x=hp_spacing,
        density=density,
        rng=rng or np.random.default_rng(),
    )
    if len(hp_pts) > 0:
        result["points"] = np.vstack([result["points"], hp_pts])
        result["labels"] = np.concatenate([result["labels"], hp_lbls])
    result["metadata"]["type"] = "t_composite_beam"
    result["metadata"]["hairpin_diam"] = hp_diam
    result["metadata"]["hairpin_spacing"] = hp_spacing
    return result


def build_l_composite_beam(
    flange: Optional[float] = None,
    web: Optional[float] = None,
    thickness: Optional[float] = None,
    height: Optional[float] = None,
    length: Optional[float] = None,
    rebar_diam: Optional[float] = None,
    n_rebars: Optional[int] = None,
    stirrup_diam: Optional[float] = None,
    stirrup_spacing: Optional[float] = None,
    hairpin_diam: Optional[float] = None,
    hairpin_spacing: Optional[float] = None,
    density: float = 0.01,
    rng: Optional[np.random.Generator] = None,
    variant: str = "rebar",
    sleeve_inner_diam: Optional[float] = None,
    sleeve_depth: Optional[float] = None,
) -> dict:
    """
    L 形叠合梁：L 形截面梁 + U 形发夹筋（上表面）+ 箍筋 + 纵筋/套筒。
    截面定义（YZ 平面，X 为长度）：
      (0,0)→(w,0)→(w,wf)→(dw,wf)→(dw,d)→(0,d)
    """
    if rng is None:
        rng = np.random.default_rng()
    w = flange or rng.uniform(500.0, 900.0)
    d = height or rng.uniform(400.0, 800.0)
    l = length or rng.uniform(*BEAM_LENGTH_RANGE)
    wf = thickness or rng.uniform(150.0, 300.0)
    dw = web or rng.uniform(200.0, 400.0)
    rd = rebar_diam or float(rng.choice(STANDARD_REBAR_DIAMETERS))
    nr = n_rebars or rng.integers(8, 16)
    sd = stirrup_diam or rng.uniform(*STIRRUP_DIAM_RANGE)
    ss = stirrup_spacing or rng.uniform(*STIRRUP_SPACING_RANGE)
    protrusion = 200.0
    hp_diam = hairpin_diam or 10.0
    hp_spacing = hairpin_spacing or 200.0

    # 保护层厚度：套筒 variant 需保证套筒外壁 ≥ SLEEVE_MIN_COVER
    if variant == "sleeve":
        # 套筒规格匹配钢筋直径 (JG/T 398-2019, JGJ 355-2015 §3.1.2)
        if sleeve_inner_diam is None:
            sid = SLEEVE_SPEC_MAP[rd][1]  # 从映射表获取套筒内径
        else:
            sid = sleeve_inner_diam
        sdep = sleeve_depth or rng.uniform(*SLEEVE_DEPTH_RANGE)
        sleeve_outer_radius = sid / 2 + SLEEVE_WALL_THICKNESS
        cc = sleeve_outer_radius + SLEEVE_MIN_COVER
    else:
        sid = 0.0  # rebar variant 不使用套筒
        sdep = 0.0
        cc = CONCRETE_COVER

    # L 形截面
    polygon = np.array(
        [
            [0.0, 0.0],
            [w, 0.0],
            [w, wf],
            [dw, wf],
            [dw, d],
            [0.0, d],
        ]
    )
    center = polygon.mean(axis=0)
    polygon -= center

    # 混凝土
    con_pts, con_lbls = _sample_beam_cross_section(polygon, l, density, rng, roughness=CONCRETE_ROUGHNESS)

    # 纵筋位置
    positions = _get_rebar_positions_polygon(polygon, cc, nr, rng)

    # 箍筋（L 形截面）
    st_pts, st_lbls = generate_stirrups_polygon(
        polygon, 0, l, sd, cc, ss, density, rng, axis=0
    )

    # U 形发夹筋
    zmax = polygon[:, 1].max()
    top_mask = np.abs(polygon[:, 1] - zmax) < 1e-6
    top_y_vals = polygon[top_mask, 0]
    y_top_max, y_top_min = top_y_vals.max(), top_y_vals.min()
    hp_pts, hp_lbls = generate_u_hairpins(
        beam_length=l,
        beam_top_y=y_top_max,
        beam_bottom_y=y_top_min,
        concrete_top_z=zmax,
        diam=hp_diam,
        spacing_x=hp_spacing,
        density=density,
        rng=rng,
    )

    # 纵筋或套筒（二选一）
    all_pts = [con_pts, st_pts, hp_pts]
    all_lbls = [con_lbls, st_lbls, hp_lbls]
    if variant == "rebar":
        reb_pts = _generate_rebar_ends(positions, 0, l, rd, protrusion, density, rng)
        all_pts.append(reb_pts)
        all_lbls.append(np.full(len(reb_pts), LABEL_REBAR, dtype=np.int32))
    else:
        rebar_xy = [(py, pz) for (py, pz) in positions]
        bot_sl, bsl = _add_sleeves_to_ends(
            rebar_xy, 0, sid, sdep, density, rng, -1, axis=0
        )
        top_sl, tsl = _add_sleeves_to_ends(
            rebar_xy, l, sid, sdep, density, rng, 1, axis=0
        )
        all_pts.extend([bot_sl, top_sl])
        all_lbls.extend([bsl, tsl])
        # 混凝土端面挖孔
        con_pts = _remove_concrete_holes(con_pts, positions, sid, 0, (0, l))
        all_pts[0] = con_pts
        all_lbls[0] = np.full(len(con_pts), LABEL_CONCRETE, dtype=np.int32)

    points = np.vstack(all_pts)
    labels = np.concatenate(all_lbls)

    return {
        "points": points,
        "labels": labels,
        "metadata": {
            "type": "l_composite_beam",
            "flange": w,
            "web": dw,
            "thickness": wf,
            "height": d,
            "length": l,
            "rebar_diam": rd,
            "n_rebars": nr,
            "variant": variant,
            "n_points_total": len(points),
            "density": density,
        },
    }


# ─── 统一入口 ──────────────────────────────────────────────

BUILDERS = {
    "rectangular_column": build_rectangular_column,
    "l_shaped_column": build_l_shaped_column,
    "rectangular_beam": build_rectangular_beam,
    "t_beam": build_t_beam,
    "rectangular_composite_beam": build_rectangular_composite_beam,
    "t_composite_beam": build_t_composite_beam,
    "l_composite_beam": build_l_composite_beam,
}


def _filter_internal_rebar(
    points: np.ndarray, labels: np.ndarray, margin: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    移除混凝土内部的钢筋点，只保留露出表面部分。
    实际工程中，埋在混凝土内部的钢筋不可见。
    """
    con_mask = labels == LABEL_CONCRETE
    rebar_mask = labels == LABEL_REBAR
    if con_mask.sum() == 0 or rebar_mask.sum() == 0:
        return points, labels

    c_min = points[con_mask].min(axis=0) - margin
    c_max = points[con_mask].max(axis=0) + margin

    inside_concrete = rebar_mask & (
        (points[:, 0] >= c_min[0]) & (points[:, 0] <= c_max[0]) &
        (points[:, 1] >= c_min[1]) & (points[:, 1] <= c_max[1]) &
        (points[:, 2] >= c_min[2]) & (points[:, 2] <= c_max[2])
    )
    n_removed = inside_concrete.sum()
    if n_removed > 0:
        keep = ~inside_concrete
        points, labels = points[keep], labels[keep]

    return points, labels


def make_point_cloud(
    comp_type: str,
    density: float = 0.01,
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> dict:
    """统一入口。"""
    if comp_type not in BUILDERS:
        raise ValueError(f"不支持的构件类型: {comp_type}")
    builder = BUILDERS[comp_type]
    result = builder(density=density, rng=rng, **kwargs)
    # 过滤混凝土内部的钢筋点（实际工程中不可见）
    result["points"], result["labels"] = _filter_internal_rebar(
        result["points"], result["labels"]
    )
    result["metadata"]["n_filtered_rebar"] = (
        result["metadata"].get("n_points_total", len(result["points"]))
        - len(result["points"])
    )
    return result
