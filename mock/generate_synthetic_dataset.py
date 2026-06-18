# -*- coding: utf-8 -*-
"""
@File    :   generate_synthetic_dataset.py
@Time    :   2026/06/18
@Desc    :   合成数据集生成器 - 生成预制混凝土构件 Mock 点云数据

构件类型及数量（共 55 个 × 3 密度 = 165 个文件）：

  ┌─ 柱（竖向，Z 为高度轴）────────────────────┐
  │  rectangular_column      矩形柱       10  │
  │  l_shaped_column         L 形柱       10  │
  └───────────────────────────────────────────┘

  ┌─ 梁（横向，X 为长度轴）────────────────────┐
  │  rectangular_beam        矩形梁       10  │
  │  t_beam                  T 形梁       10  │
  │  ── 叠合梁（上表面有 U 形发夹钢筋）──────────│
  │  rectangular_composite   矩形叠合梁    5  │
  │  t_composite             T 形叠合梁    5  │
  │  l_composite             L 形叠合梁    5  │
  └───────────────────────────────────────────┘

核心几何规则：
  - 混凝土：仅表面面片采样（模拟扫描仪只能捕捉外表面）
  - 纵向钢筋：仅外露端头（混凝土内部钢筋扫描不可见）
  - U 形发夹（叠合梁）：倒 U 形，两平行直腿 + 顶部弯钩连接，
    开口朝下立在混凝土上表面，间距 80~150mm，高出 100~150mm
  - 按 GB 50010-2010：钢筋最小净距 ≥ 30mm

用法：
  python generate_synthetic_dataset.py              # 生成全部
  python generate_synthetic_dataset.py --type rectangular_beam --count 1
"""

import numpy as np
import open3d as o3d
import os
import json
import argparse
from datetime import datetime
from typing import List, Tuple, Dict, Optional

# ============================================================
# 全局参数
# ============================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "res", "point")
DENSITY_MAP = {"high": 0.02, "medium": 0.01, "low": 0.005}
CONCRETE_COLOR = np.array([0.7, 0.7, 0.7])
REBAR_COLOR = np.array([0.6, 0.2, 0.2])
MIN_REBAR_SPACING = 30.0  # GB 50010-2010 最小净距


# ============================================================
# 基础几何采样工具
# ============================================================

def sample_cylinder_surface(start, end, radius, n_pts):
    """在圆柱体侧面上均匀采样（不含顶/底面）。"""
    direction = end - start
    length_val = np.linalg.norm(direction)
    if length_val < 1e-8:
        return np.tile(start, (n_pts, 1))
    dir_norm = direction / length_val
    if abs(dir_norm[0]) < 0.9:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = np.array([0.0, 1.0, 0.0])
    u -= np.dot(u, dir_norm) * dir_norm
    u /= np.linalg.norm(u)
    v = np.cross(dir_norm, u)
    theta = np.random.uniform(0, 2 * np.pi, n_pts)
    t = np.random.uniform(0, length_val, n_pts)
    pts = (start[None, :] + t[:, None] * dir_norm[None, :]
           + radius * (np.cos(theta)[:, None] * u[None, :]
                       + np.sin(theta)[:, None] * v[None, :]))
    return pts


def sample_rectangle_face(origin, u_vec, v_vec, n_pts):
    """在矩形面上均匀采样。"""
    su = np.random.uniform(0, 1, n_pts)
    sv = np.random.uniform(0, 1, n_pts)
    return origin[None, :] + su[:, None] * u_vec[None, :] + sv[:, None] * v_vec[None, :]


def _sample_polygon_area(vertices, n_pts):
    """在 2D 多边形内部均匀采样。"""
    if n_pts == 0:
        return np.empty((0, 2))
    triangles = []
    verts = vertices.tolist()
    remaining = list(range(len(verts)))
    max_iter = len(remaining) * 10
    iter_count = 0
    while len(remaining) > 3 and iter_count < max_iter:
        iter_count += 1
        found_ear = False
        for idx in range(len(remaining)):
            prev_idx = remaining[(idx - 1) % len(remaining)]
            curr_idx = remaining[idx]
            next_idx = remaining[(idx + 1) % len(remaining)]
            p_prev = np.array(verts[prev_idx])
            p_curr = np.array(verts[curr_idx])
            p_next = np.array(verts[next_idx])
            v1, v2 = p_prev - p_curr, p_next - p_curr
            if v1[0] * v2[1] - v1[1] * v2[0] >= 0:
                continue
            inside = False
            for j in remaining:
                if j in (prev_idx, curr_idx, next_idx):
                    continue
                if _point_in_triangle(np.array(verts[j]), p_curr, p_next, p_prev):
                    inside = True
                    break
            if not inside:
                triangles.append((prev_idx, curr_idx, next_idx))
                remaining.pop(idx)
                found_ear = True
                break
        if not found_ear:
            break
    if len(remaining) == 3:
        triangles.append((remaining[0], remaining[1], remaining[2]))
    elif len(remaining) > 3:
        for i in range(1, len(remaining) - 1):
            triangles.append((remaining[0], remaining[i], remaining[i + 1]))
    tri_areas, tri_verts = [], []
    for i1, i2, i3 in triangles:
        a, b, c = np.array(verts[i1]), np.array(verts[i2]), np.array(verts[i3])
        area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
        tri_areas.append(area)
        tri_verts.append((a, b, c))
    total_area = sum(tri_areas)
    if total_area == 0:
        return np.tile(vertices[0], (n_pts, 1))
    probs = np.array(tri_areas) / total_area
    tri_counts = np.random.multinomial(n_pts, probs)
    pts_list = []
    for cnt, (a, b, c) in zip(tri_counts, tri_verts):
        if cnt == 0:
            continue
        r1 = np.random.uniform(0, 1, cnt)
        r2 = np.random.uniform(0, 1, cnt)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        pts = a[None,:] + r1[:,None]*(b-a)[None,:] + r2[:,None]*(c-a)[None,:]
        pts_list.append(pts)
    return np.vstack(pts_list)


def _point_in_triangle(p, a, b, c):
    v0, v1, v2 = c - a, b - a, p - a
    d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
    d20, d21 = np.dot(v2, v0), np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return False
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return v >= 0 and w >= 0 and v + w <= 1


def sample_polygon_prism_surface(polygon_2d, height_val, n_pts, axis="z"):
    """多边形棱柱表面采样（axis 为拉伸轴）。"""
    if not np.allclose(polygon_2d[0], polygon_2d[-1]):
        polygon_2d = np.vstack([polygon_2d, polygon_2d[0:1]])
    n_edges = len(polygon_2d) - 1
    edge_lengths = [np.linalg.norm(polygon_2d[i+1]-polygon_2d[i]) for i in range(n_edges)]
    perimeter = sum(edge_lengths)
    x, y = polygon_2d[:, 0], polygon_2d[:, 1]
    area = 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))
    total_surface = 2 * area + perimeter * height_val
    n_top = int(max(1, n_pts * area / total_surface))
    n_sides = n_pts - 2 * n_top
    pts_list = []
    for offset, n_face in [(height_val, n_top), (0.0, n_top)]:
        pts = _sample_polygon_area(polygon_2d[:-1], n_face)
        if axis == "z":
            pts_list.append(np.hstack([pts, np.full((len(pts), 1), offset)]))
        else:
            pts_list.append(np.hstack([np.full((len(pts), 1), offset), pts]))
    for i in range(n_edges):
        n_edge = max(1, int(n_sides*edge_lengths[i]/perimeter)) if perimeter>0 else max(1, n_sides//n_edges)
        if i == n_edges-1:
            n_edge = n_sides - sum(max(1,int(n_sides*el/perimeter)) for el in edge_lengths[:-1]) if perimeter>0 else n_sides
        p1, p2 = polygon_2d[i], polygon_2d[i+1]
        su = np.random.uniform(0, 1, n_edge)
        sv = np.random.uniform(0, height_val, n_edge)
        edge_2d = p1[None,:] + su[:,None]*(p2-p1)[None,:]
        if axis == "z":
            pts_list.append(np.column_stack([edge_2d[:,0], edge_2d[:,1], sv]))
        else:
            pts_list.append(np.column_stack([sv, edge_2d[:,0], edge_2d[:,1]]))
    return np.vstack(pts_list)


def make_point_cloud(concrete_pts, rebar_pts):
    """合并混凝土与钢筋点云，赋予不同颜色。"""
    all_pts = np.vstack([concrete_pts, rebar_pts]) if len(rebar_pts) > 0 else concrete_pts
    colors = np.vstack([
        np.tile(CONCRETE_COLOR, (len(concrete_pts), 1)),
        np.tile(REBAR_COLOR, (len(rebar_pts), 1)),
    ]) if len(rebar_pts) > 0 else np.tile(CONCRETE_COLOR, (len(concrete_pts), 1))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# ============================================================
# 钢筋布置辅助
# ============================================================

def _enforce_spacing(positions):
    """移除间距过小的钢筋。"""
    kept = []
    for p in positions:
        ok = True
        for q in kept:
            if np.hypot(p[0]-q[0], p[1]-q[1]) < MIN_REBAR_SPACING:
                ok = False
                break
        if ok:
            kept.append(p)
    return kept


def _get_rebar_positions_rect(half_w, half_d, cover, count):
    """矩形截面的纵向钢筋位置（角部优先 + 沿边均匀补足）。"""
    inner_w, inner_d = half_w - cover, half_d - cover
    positions = [
        (-inner_w, -inner_d), (inner_w, -inner_d),
        (-inner_w, inner_d), (inner_w, inner_d),
    ]
    if count > 4:
        n_extra = count - 4
        per_side = n_extra // 2
        remainder = n_extra % 2
        n_fb = per_side + remainder
        for i in range(n_fb):
            x = -inner_w + (2*inner_w)*(i+1)/(n_fb+1)
            positions.append((x, -inner_d))
            if len(positions) >= count:
                break
        if len(positions) < count:
            for i in range(per_side):
                y = -inner_d + (2*inner_d)*(i+1)/(per_side+1)
                positions.append((-inner_w, y))
                if len(positions) >= count:
                    break
    return _enforce_spacing(positions)[:count]


def _shrink_polygon(polygon, offset):
    """多边形内缩（用于保护层）。"""
    if offset <= 0:
        return polygon.copy()
    inner = polygon.copy()
    for i in range(len(inner)):
        prev, curr, nxt = inner[(i-1)%len(inner)], inner[i], inner[(i+1)%len(inner)]
        d1, d2 = curr-prev, nxt-curr
        d1_len, d2_len = np.linalg.norm(d1), np.linalg.norm(d2)
        if d1_len < 1e-8 or d2_len < 1e-8:
            continue
        n1 = np.array([-d1[1], d1[0]]) / d1_len
        n2 = np.array([-d2[1], d2[0]]) / d2_len
        cross = d1[0]*d2[1] - d1[1]*d2[0]
        if cross > 0:
            bisector = n1 + n2
            bl = np.linalg.norm(bisector)
            if bl > 1e-8:
                inner[i] = curr + (bisector/bl)*offset
        else:
            inner[i] = curr + n1*offset
    return inner


def _get_rebar_positions_polygon(polygon, cover, count):
    """多边形截面纵向钢筋位置（角部优先 + 长边中部补足）。"""
    inner = _shrink_polygon(polygon, cover)
    if inner is None or len(inner) < 3:
        inner = polygon
    positions = [(inner[i, 0], inner[i, 1]) for i in range(len(inner))]
    if count > len(positions):
        n_extra = count - len(positions)
        edges, edge_lens = [], []
        for i in range(len(inner)):
            j = (i+1)%len(inner)
            edges.append((inner[i], inner[j]))
            edge_lens.append(np.linalg.norm(inner[j]-inner[i]))
        total_len = sum(edge_lens)
        if total_len == 0:
            return positions[:count]
        extra_per_edge = [max(1, int(n_extra*el/total_len)) for el in edge_lens]
        while sum(extra_per_edge) < n_extra:
            extra_per_edge[np.argmax(edge_lens)] += 1
        while sum(extra_per_edge) > n_extra:
            extra_per_edge[np.argmax(extra_per_edge)] -= 1
        for (e0, e1), n_e in zip(edges, extra_per_edge):
            if n_e <= 0:
                continue
            for k in range(1, n_e+1):
                t = k/(n_e+1)
                pt = e0 + t*(e1-e0)
                positions.append((pt[0], pt[1]))
    return _enforce_spacing(positions)[:count]


# ============================================================
# U 形发夹钢筋（叠合梁特征）
# ============================================================

def _generate_u_hairpins(beam_length, beam_top_y, beam_bottom_y,
                         concrete_top_z, diam, spacing_x, density):
    """为梁上表面生成倒 U 形发夹钢筋。

    U 形发夹几何：
      顶部弯钩  ===（水平圆柱，连接两直腿顶部）
      左直腿    |     右直腿    |（垂直圆柱，立在混凝土表面）
    混凝土表面 ================

    参数（符合装配式规范）：
      - 腿间距（开口宽度）：80~150mm
      - 高出表面高度：100~150mm
      - 沿梁长方向（X）均匀排列
    """
    pts_per_mm2 = DENSITY_MAP[density]
    radius = diam / 2.0
    y_range = abs(beam_top_y - beam_bottom_y)
    # 开口宽度 = 85% 表面宽度，边缘各留 ≥15mm 保护层
    gap = max(40.0, min(y_range * 0.85, y_range - 30.0))
    hairpin_height = min(150.0, max(100.0, gap * 0.5))
    mid_y = (beam_top_y + beam_bottom_y) / 2.0
    y_left = mid_y - gap / 2.0
    y_right = mid_y + gap / 2.0
    n_hairpins = max(1, int(beam_length / spacing_x))
    # 不等间距：梁端 0~0.25L 密 (×0.6)，跨中 0.25L~0.75L 疏 (×1.4)
    end_spacing = spacing_x * 0.6
    mid_spacing = spacing_x * 1.4
    transition = beam_length * 0.25
    pts_list, meta_list = [], []
    x_pos = transition * 0.2  # 第一个发夹离端头一点距离
    while x_pos < beam_length - end_spacing * 0.3:
        if x_pos < transition or x_pos > beam_length - transition:
            sp = end_spacing
        else:
            sp = mid_spacing
        z_base = concrete_top_z
        z_top = concrete_top_z + hairpin_height
        p_lb = np.array([x_pos, y_left, z_base])
        p_lt = np.array([x_pos, y_left, z_top])
        p_rb = np.array([x_pos, y_right, z_base])
        p_rt = np.array([x_pos, y_right, z_top])
        leg_len = hairpin_height
        leg_area = 2 * np.pi * radius * leg_len
        leg_n = max(12, int(leg_area * pts_per_mm2))
        pts_list.append(sample_cylinder_surface(p_lb, p_lt, radius, leg_n))
        pts_list.append(sample_cylinder_surface(p_rb, p_rt, radius, leg_n))
        bar_area = 2 * np.pi * radius * gap
        bar_n = max(8, int(bar_area * pts_per_mm2))
        pts_list.append(sample_cylinder_surface(p_lt, p_rt, radius, bar_n))
        meta_list.append({
            "x_mm": round(x_pos, 1), "diameter_mm": diam,
            "height_mm": hairpin_height, "gap_mm": gap,
            "spacing_mm": round(sp, 1),
        })
        x_pos += sp
    pts = np.vstack(pts_list) if pts_list else np.empty((0, 3))
    return pts, meta_list


# ============================================================
# 柱 构建器（竖向，Z 为高度轴）
# ============================================================

def build_rectangular_column(w, d, h, rebar_diam, rebar_count, density,
                              concrete_cover=50.0, protrusion=300.0):
    """矩形柱：长方体 + 纵向钢筋（两端伸出）。"""
    half_w, half_d = w/2, d/2
    pts_per_mm2 = DENSITY_MAP[density]
    face_areas = np.array([w*d, w*d, w*h, w*h, d*h, d*h])
    face_ratios = face_areas / face_areas.sum()
    target_n = max(100, int(2*(w*d+w*h+d*h)*pts_per_mm2))
    face_n = np.maximum(1, (face_ratios*target_n).astype(int))
    face_n[np.argmax(face_areas)] += target_n - face_n.sum()
    pts_list = [
        sample_rectangle_face(np.array([-half_w, -half_d, 0]), np.array([w,0,0]), np.array([0,d,0]), face_n[0]),
        sample_rectangle_face(np.array([-half_w, -half_d, h]), np.array([w,0,0]), np.array([0,d,0]), face_n[1]),
        sample_rectangle_face(np.array([-half_w, -half_d, 0]), np.array([w,0,0]), np.array([0,0,h]), face_n[2]),
        sample_rectangle_face(np.array([-half_w, half_d, 0]), np.array([w,0,0]), np.array([0,0,h]), face_n[3]),
        sample_rectangle_face(np.array([-half_w, -half_d, 0]), np.array([0,d,0]), np.array([0,0,h]), face_n[4]),
        sample_rectangle_face(np.array([half_w, -half_d, 0]), np.array([0,d,0]), np.array([0,0,h]), face_n[5]),
    ]
    concrete_pts = np.vstack(pts_list)
    positions = _get_rebar_positions_rect(half_w, half_d, concrete_cover, rebar_count)
    radius = rebar_diam / 2.0
    cyl_area = 2 * np.pi * radius * protrusion
    cyl_n = max(20, int(cyl_area * pts_per_mm2))
    rebar_pts_list, rebar_meta = [], []
    for px, py in positions:
        rebar_pts_list.append(sample_cylinder_surface(
            np.array([px,py,h]), np.array([px,py,h+protrusion]), radius, cyl_n))
        rebar_pts_list.append(sample_cylinder_surface(
            np.array([px,py,0]), np.array([px,py,-protrusion]), radius, cyl_n))
        rebar_meta.append({"position_2d": [px, py], "diameter_mm": rebar_diam})
    rebar_pts = np.vstack(rebar_pts_list) if rebar_pts_list else np.empty((0, 3))
    return concrete_pts, rebar_pts, {
        "type": "rectangular_column",
        "concrete": {"width_mm": w, "depth_mm": d, "height_mm": h},
        "rebar": rebar_meta,
    }


def build_l_shaped_column(w, d, h, wf, dw, rebar_diam, rebar_count, density,
                           concrete_cover=50.0, protrusion=300.0):
    """L 形柱。截面: (0,0)→(w,0)→(w,wf)→(dw,wf)→(dw,d)→(0,d)。"""
    pts_per_mm2 = DENSITY_MAP[density]
    polygon = np.array([[0.,0.],[w,0.],[w,wf],[dw,wf],[dw,d],[0.,d]])
    center = polygon.mean(axis=0)
    polygon -= center
    perimeter = sum(np.linalg.norm(polygon[(i+1)%len(polygon)]-polygon[i]) for i in range(len(polygon)))
    xv, yv = polygon[:,0], polygon[:,1]
    area = 0.5*abs(sum(xv[i]*yv[(i+1)%len(polygon)] for i in range(len(polygon)))
                    - sum(yv[i]*xv[(i+1)%len(polygon)] for i in range(len(polygon))))
    total_surface = 2*area + perimeter*h
    n_pts = max(100, int(total_surface*pts_per_mm2))
    concrete_pts = sample_polygon_prism_surface(polygon, h, n_pts, axis="z")
    positions = _get_rebar_positions_polygon(polygon, concrete_cover, rebar_count)
    radius = rebar_diam/2.0
    cyl_n = max(20, int(2*np.pi*radius*protrusion*pts_per_mm2))
    rebar_pts_list, rebar_meta = [], []
    for px, py in positions:
        rebar_pts_list.append(sample_cylinder_surface(
            np.array([px,py,h]), np.array([px,py,h+protrusion]), radius, cyl_n))
        rebar_pts_list.append(sample_cylinder_surface(
            np.array([px,py,0]), np.array([px,py,-protrusion]), radius, cyl_n))
        rebar_meta.append({"position_2d": [round(px,2), round(py,2)], "diameter_mm": rebar_diam})
    rebar_pts = np.vstack(rebar_pts_list) if rebar_pts_list else np.empty((0, 3))
    return concrete_pts, rebar_pts, {
        "type": "l_shaped_column",
        "concrete": {"width_mm": w, "depth_mm": d, "height_mm": h, "flange_w_mm": wf, "web_t_mm": dw},
        "rebar": rebar_meta,
    }


# ============================================================
# 梁 构建器（横向，X 为长度轴，YZ 为截面）
# ============================================================

def _sample_rect_decomposed(vertices, n_pts):
    """将正交多边形分解为矩形网格并填充采样。

    网格由顶点 Y/Z 唯一值确定，用射线法判断各格是否在形内。
    矩形梁→1格（等于填充矩形），T/L梁→多格（翼缘+腹板各自填充）。
    """
    if n_pts == 0:
        return np.empty((0, 2))
    yv, zv = vertices[:, 0], vertices[:, 1]
    yu = np.sort(np.unique(np.around(yv, decimals=6)))
    zu = np.sort(np.unique(np.around(zv, decimals=6)))
    nv = len(vertices)
    rects = []
    ny, nz = len(yu), len(zu)
    edges = [(vertices[i], vertices[(i + 1) % nv]) for i in range(nv)]
    for yi in range(ny - 1):
        for zi in range(nz - 1):
            y0, y1 = yu[yi], yu[yi + 1]
            z0, z1 = zu[zi], zu[zi + 1]
            yc, zc = (y0 + y1) / 2, (z0 + z1) / 2
            crossings = 0
            for (p1, p2) in edges:
                zl, zh = (p1[1], p2[1]) if p1[1] < p2[1] else (p2[1], p1[1])
                if zc > zl and zc <= zh:
                    x_intersect = p1[0] + (zc - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                    if yc < x_intersect:
                        crossings += 1
            if crossings % 2 == 1:
                rects.append((y0, y1, z0, z1, (y1 - y0) * (z1 - z0)))
    if not rects:
        return np.empty((0, 2))
    total_area = sum(r[4] for r in rects)
    if total_area <= 0:
        return np.empty((0, 2))
    remaining = n_pts
    pts_list = []
    for idx, (y0, y1, z0, z1, ra) in enumerate(rects):
        n_r = max(1, int(remaining * ra / total_area)) if idx < len(rects) - 1 and remaining > 1 else max(1, remaining)
        n_r = min(n_r, remaining)
        remaining -= n_r
        uy = np.random.uniform(y0, y1, n_r)
        uz = np.random.uniform(z0, z1, n_r)
        pts_list.append(np.column_stack([uy, uz]))
    return np.vstack(pts_list)


def _sample_beam_surface(cross_2d, length, n_pts):
    """对梁（X 为长轴）表面采样：两端面 + 侧面。

    Args:
        cross_2d: 截面多边形 [(y,z), ...]
        length: 梁长度 (X 方向)
        n_pts: 总点数
    """
    if not np.allclose(cross_2d[0], cross_2d[-1]):
        cross_2d = np.vstack([cross_2d, cross_2d[0:1]])
    n_edges = len(cross_2d) - 1
    edge_lengths = [np.linalg.norm(cross_2d[i+1]-cross_2d[i]) for i in range(n_edges)]
    perimeter = sum(edge_lengths)
    yv, zv = cross_2d[:,0], cross_2d[:,1]
    area = 0.5*abs(sum(yv[i]*zv[(i+1)%len(cross_2d)] for i in range(len(cross_2d)))
                    - sum(zv[i]*yv[(i+1)%len(cross_2d)] for i in range(len(cross_2d))))
    total = 2*area + perimeter*length
    n_endface = int(max(1, n_pts*area/total))
    n_sides = max(1, n_pts - 2*n_endface)
    pts_list = []
    # 端面：分解为矩形区域分别填充，避免 T/L 截面三角剖分成梯形
    for x_off in [0.0, length]:
        endface_pts = _sample_rect_decomposed(cross_2d[:-1], n_endface)
        pts_list.append(np.column_stack([np.full(len(endface_pts), x_off), endface_pts]))
    for i in range(n_edges):
        n_edge = max(1, int(n_sides*edge_lengths[i]/perimeter)) if perimeter>0 else max(1, n_sides//n_edges)
        if i == n_edges-1:
            n_edge = n_sides - sum(max(1,int(n_sides*el/perimeter)) for el in edge_lengths[:-1])
        p1, p2 = cross_2d[i], cross_2d[i+1]
        su = np.random.uniform(0, 1, n_edge)
        sx = np.random.uniform(0, length, n_edge)
        edge_2d = p1[None,:] + su[:,None]*(p2-p1)[None,:]
        pts_list.append(np.column_stack([sx, edge_2d[:,0], edge_2d[:,1]]))
    return np.vstack(pts_list)


def build_rectangular_beam(b, h, length, rebar_diam, rebar_count, density,
                            concrete_cover=50.0, protrusion=300.0):
    """矩形梁：长方体，钢筋从两端伸出。

    坐标系：X=长度, Y=宽度, Z=高度。
    截面 Z 居中于原点 Z ∈ [-h/2, h/2]，梁顶面 Z = h/2。
    """
    pts_per_mm2 = DENSITY_MAP[density]
    cross = np.array([[-b/2, -h/2], [b/2, -h/2], [b/2, h/2], [-b/2, h/2]])
    total_surface = 2*(b*h) + 2*(b*length) + 2*(h*length)
    n_pts = max(200, int(total_surface*pts_per_mm2))
    concrete_pts = _sample_beam_surface(cross, length, n_pts)
    # 钢筋位置在居中截面上（_get_rebar_positions_rect 输出对称于原点）
    positions = _get_rebar_positions_rect(b/2, h/2, concrete_cover, rebar_count)
    radius = rebar_diam/2.0
    cyl_n = max(20, int(2*np.pi*radius*protrusion*pts_per_mm2))
    rebar_pts_list, rebar_meta = [], []
    for py, pz in positions:
        rebar_pts_list.append(sample_cylinder_surface(
            np.array([0.,py,pz]), np.array([-protrusion,py,pz]), radius, cyl_n))
        rebar_pts_list.append(sample_cylinder_surface(
            np.array([length,py,pz]), np.array([length+protrusion,py,pz]), radius, cyl_n))
        rebar_meta.append({"position_2d": [round(py,2), round(pz,2)], "diameter_mm": rebar_diam})
    rebar_pts = np.vstack(rebar_pts_list) if rebar_pts_list else np.empty((0, 3))
    return concrete_pts, rebar_pts, {
        "type": "rectangular_beam",
        "concrete": {"width_mm": b, "height_mm": h, "length_mm": length},
        "rebar": rebar_meta,
    }


def build_rectangular_composite_beam(b, h, length, rebar_diam, rebar_count,
                                      hairpin_diam, hairpin_spacing, density,
                                      concrete_cover=50.0, protrusion=300.0):
    """矩形叠合梁：矩形梁 + 上表面 U 形发夹钢筋。"""
    concrete_pts, rebar_pts, meta = build_rectangular_beam(
        b, h, length, rebar_diam, rebar_count, density, concrete_cover, protrusion)
    hairpin_pts, hp_meta_list = _generate_u_hairpins(
        beam_length=length, beam_top_y=b/2, beam_bottom_y=-b/2,
        concrete_top_z=h/2, diam=hairpin_diam, spacing_x=hairpin_spacing, density=density)
    if len(hairpin_pts) > 0:
        rebar_pts = np.vstack([rebar_pts, hairpin_pts]) if len(rebar_pts)>0 else hairpin_pts
    meta["type"] = "rectangular_composite_beam"
    meta["hairpin"] = {"diameter_mm": hairpin_diam, "spacing_mm": hairpin_spacing,
                       "count": len(hp_meta_list), "positions": hp_meta_list}
    return concrete_pts, rebar_pts, meta


def build_t_beam(B, hf, bw, h, length, rebar_diam, rebar_count, density,
                 concrete_cover=50.0, protrusion=300.0):
    """T 形梁：翼缘在上，腹板在下。YZ 截面（居中），X 为长度。

    居中后 Z ∈ [-h/2, h/2]，翼缘顶 Z = h/2。
    """
    pts_per_mm2 = DENSITY_MAP[density]
    B2, bw2 = B/2, bw/2
    half_z = h/2
    cross = np.array([
        [-B2, half_z], [B2, half_z], [B2, half_z-hf], [bw2, half_z-hf],
        [bw2, -half_z], [-bw2, -half_z], [-bw2, half_z-hf], [-B2, half_z-hf],
    ])
    perimeter = sum(np.linalg.norm(cross[(i+1)%len(cross)]-cross[i]) for i in range(len(cross)))
    yv, zv = cross[:,0], cross[:,1]
    area = 0.5*abs(sum(yv[i]*zv[(i+1)%len(cross)] for i in range(len(cross)))
                    - sum(zv[i]*yv[(i+1)%len(cross)] for i in range(len(cross))))
    total = 2*area + perimeter*length
    n_pts = max(200, int(total*pts_per_mm2))
    concrete_pts = _sample_beam_surface(cross, length, n_pts)
    cover = concrete_cover
    positions = [
        (-bw2+cover, -half_z+cover),    # 腹板左下
        (bw2-cover, -half_z+cover),     # 腹板右下
        (-B2+cover, half_z-cover),      # 翼缘左上
        (B2-cover, half_z-cover),       # 翼缘右上
    ]
    if rebar_count > 4:
        n_extra = rebar_count - 4
        for i in range(1, n_extra+1):
            f = i/(n_extra+1)
            y_pos = -B2 + cover + (B-2*cover)*f
            positions.append((y_pos, half_z-cover))
    positions = _enforce_spacing(positions)[:rebar_count]
    radius = rebar_diam/2.0
    cyl_n = max(20, int(2*np.pi*radius*protrusion*pts_per_mm2))
    rebar_pts_list, rebar_meta = [], []
    for py, pz in positions:
        rebar_pts_list.append(sample_cylinder_surface(
            np.array([0.,py,pz]), np.array([-protrusion,py,pz]), radius, cyl_n))
        rebar_pts_list.append(sample_cylinder_surface(
            np.array([length,py,pz]), np.array([length+protrusion,py,pz]), radius, cyl_n))
        rebar_meta.append({"position_2d": [round(py,2), round(pz,2)], "diameter_mm": rebar_diam})
    rebar_pts = np.vstack(rebar_pts_list) if rebar_pts_list else np.empty((0, 3))
    return concrete_pts, rebar_pts, {
        "type": "t_beam",
        "concrete": {"flange_w_mm": B, "flange_h_mm": hf, "web_w_mm": bw,
                     "height_mm": h, "length_mm": length},
        "rebar": rebar_meta,
    }


def build_t_composite_beam(B, hf, bw, h, length, rebar_diam, rebar_count,
                            hairpin_diam, hairpin_spacing, density,
                            concrete_cover=50.0, protrusion=300.0):
    """T 形叠合梁：T 形梁 + 上表面 U 形发夹。"""
    concrete_pts, rebar_pts, meta = build_t_beam(
        B, hf, bw, h, length, rebar_diam, rebar_count, density, concrete_cover, protrusion)
    hairpin_pts, hp_meta_list = _generate_u_hairpins(
        beam_length=length, beam_top_y=bw/2, beam_bottom_y=-bw/2,
        concrete_top_z=h/2, diam=hairpin_diam, spacing_x=hairpin_spacing, density=density)
    if len(hairpin_pts) > 0:
        rebar_pts = np.vstack([rebar_pts, hairpin_pts]) if len(rebar_pts)>0 else hairpin_pts
    meta["type"] = "t_composite_beam"
    meta["hairpin"] = {"diameter_mm": hairpin_diam, "spacing_mm": hairpin_spacing,
                       "count": len(hp_meta_list), "positions": hp_meta_list}
    return concrete_pts, rebar_pts, meta


def build_l_composite_beam(w, d, length, wf, dw, rebar_diam, rebar_count,
                            hairpin_diam, hairpin_spacing, density,
                            concrete_cover=50.0, protrusion=300.0):
    """L 形叠合梁：L 形截面梁 + 上表面 U 形发夹。截面在 YZ 平面。"""
    pts_per_mm2 = DENSITY_MAP[density]
    polygon = np.array([[0.,0.],[w,0.],[w,wf],[dw,wf],[dw,d],[0.,d]])
    center = polygon.mean(axis=0)
    polygon -= center
    perimeter = sum(np.linalg.norm(polygon[(i+1)%len(polygon)]-polygon[i]) for i in range(len(polygon)))
    yv, zv = polygon[:,0], polygon[:,1]
    area = 0.5*abs(sum(yv[i]*zv[(i+1)%len(polygon)] for i in range(len(polygon)))
                    - sum(zv[i]*yv[(i+1)%len(polygon)] for i in range(len(polygon))))
    total = 2*area + perimeter*length
    n_pts = max(200, int(total*pts_per_mm2))
    concrete_pts = _sample_beam_surface(polygon, length, n_pts)
    positions = _get_rebar_positions_polygon(polygon, concrete_cover, rebar_count)
    radius = rebar_diam/2.0
    cyl_n = max(20, int(2*np.pi*radius*protrusion*pts_per_mm2))
    rebar_pts_list, rebar_meta = [], []
    for py, pz in positions:
        rebar_pts_list.append(sample_cylinder_surface(
            np.array([0.,py,pz]), np.array([-protrusion,py,pz]), radius, cyl_n))
        rebar_pts_list.append(sample_cylinder_surface(
            np.array([length,py,pz]), np.array([length+protrusion,py,pz]), radius, cyl_n))
        rebar_meta.append({"position_2d": [round(py,2), round(pz,2)], "diameter_mm": rebar_diam})
    rebar_pts = np.vstack(rebar_pts_list) if rebar_pts_list else np.empty((0, 3))
    # 高位顶面（max Z）：U 形发夹仅在此生成，排除低位挑耳台阶
    zmax = polygon[:,1].max()
    top_mask = np.abs(polygon[:,1] - zmax) < 1e-6
    top_y_vals = polygon[top_mask, 0]
    y_top_min, y_top_max = top_y_vals.min(), top_y_vals.max()
    hairpin_pts, hp_meta_list = _generate_u_hairpins(
        beam_length=length, beam_top_y=y_top_max, beam_bottom_y=y_top_min,
        concrete_top_z=zmax, diam=hairpin_diam, spacing_x=hairpin_spacing, density=density)
    if len(hairpin_pts) > 0:
        rebar_pts = np.vstack([rebar_pts, hairpin_pts]) if len(rebar_pts)>0 else hairpin_pts
    return concrete_pts, rebar_pts, {
        "type": "l_composite_beam",
        "concrete": {"width_mm": w, "depth_mm": d, "length_mm": length,
                     "flange_w_mm": wf, "web_t_mm": dw},
        "rebar": rebar_meta,
        "hairpin": {"diameter_mm": hairpin_diam, "spacing_mm": hairpin_spacing,
                    "count": len(hp_meta_list), "positions": hp_meta_list},
    }


# ============================================================
# 数据集生成器
# ============================================================

class DatasetGenerator:
    """合成数据集批量生成器。"""

    def __init__(self, output_dir=None):
        self.output_dir = output_dir or OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_rectangular_columns(self, count=10):
        rng = np.random.default_rng(42)
        files = []
        for i in range(count):
            w = float(rng.integers(400, 800))
            d = float(rng.integers(400, 800))
            h = float(rng.integers(2000, 6000))
            rd = float(rng.integers(16, 32))
            rc = int(rng.integers(4, 12))
            for dens in ["high", "medium", "low"]:
                c, r, m = build_rectangular_column(w, d, h, rd, rc, dens)
                files.append(self._save("rectangular_column", i+1, dens, c, r, m))
        return files

    def generate_l_shaped_columns(self, count=10):
        rng = np.random.default_rng(43)
        files = []
        for i in range(count):
            w = float(rng.integers(500, 900))
            d = float(rng.integers(500, 900))
            h = float(rng.integers(2000, 6000))
            wf = float(rng.integers(200, min(400, int(d)-100)))
            dw = float(rng.integers(200, min(400, int(w)-100)))
            rd = float(rng.integers(16, 32))
            rc = int(rng.integers(4, 10))
            for dens in ["high", "medium", "low"]:
                c, r, m = build_l_shaped_column(w, d, h, wf, dw, rd, rc, dens)
                files.append(self._save("l_shaped_column", i+1, dens, c, r, m))
        return files

    def generate_rectangular_beams(self, count=10):
        rng = np.random.default_rng(44)
        files = []
        for i in range(count):
            b = float(rng.integers(300, 600))
            h = float(rng.integers(400, 900))
            length = float(rng.integers(3000, 8000))
            rd = float(rng.integers(16, 32))
            rc = int(rng.integers(4, 12))
            for dens in ["high", "medium", "low"]:
                c, r, m = build_rectangular_beam(b, h, length, rd, rc, dens)
                files.append(self._save("rectangular_beam", i+1, dens, c, r, m))
        return files

    def generate_t_beams(self, count=10):
        rng = np.random.default_rng(45)
        files = []
        for i in range(count):
            B = float(rng.integers(600, 1200))
            bw = float(rng.integers(200, min(400, int(B)-100)))
            h = float(rng.integers(400, 900))
            hf = float(rng.integers(80, min(200, int(h)-150)))
            length = float(rng.integers(3000, 8000))
            rd = float(rng.integers(16, 32))
            rc = int(rng.integers(4, 10))
            for dens in ["high", "medium", "low"]:
                c, r, m = build_t_beam(B, hf, bw, h, length, rd, rc, dens)
                files.append(self._save("t_beam", i+1, dens, c, r, m))
        return files

    def generate_rectangular_composite(self, count=5):
        rng = np.random.default_rng(50)
        files = []
        for i in range(count):
            b = float(rng.integers(300, 600))
            h = float(rng.integers(400, 900))
            length = float(rng.integers(3000, 8000))
            rd = float(rng.integers(16, 32))
            rc = int(rng.integers(4, 12))
            hd = float(rng.integers(8, 12))
            hs = float(rng.integers(200, 500))
            for dens in ["high", "medium", "low"]:
                c, r, m = build_rectangular_composite_beam(b, h, length, rd, rc, hd, hs, dens)
                files.append(self._save("rectangular_composite", i+1, dens, c, r, m))
        return files

    def generate_t_composite(self, count=5):
        rng = np.random.default_rng(51)
        files = []
        for i in range(count):
            B = float(rng.integers(600, 1200))
            bw = float(rng.integers(200, min(400, int(B)-100)))
            h = float(rng.integers(400, 900))
            hf = float(rng.integers(80, min(200, int(h)-150)))
            length = float(rng.integers(3000, 8000))
            rd = float(rng.integers(16, 32))
            rc = int(rng.integers(4, 10))
            hd = float(rng.integers(8, 12))
            hs = float(rng.integers(200, 500))
            for dens in ["high", "medium", "low"]:
                c, r, m = build_t_composite_beam(B, hf, bw, h, length, rd, rc, hd, hs, dens)
                files.append(self._save("t_composite", i+1, dens, c, r, m))
        return files

    def generate_l_composite(self, count=5):
        rng = np.random.default_rng(52)
        files = []
        for i in range(count):
            w = float(rng.integers(500, 900))
            d = float(rng.integers(500, 900))
            length = float(rng.integers(3000, 8000))
            wf = float(rng.integers(200, min(400, int(d)-100)))
            dw = float(rng.integers(200, min(400, int(w)-100)))
            rd = float(rng.integers(16, 32))
            rc = int(rng.integers(4, 10))
            hd = float(rng.integers(8, 12))
            hs = float(rng.integers(200, 500))
            for dens in ["high", "medium", "low"]:
                c, r, m = build_l_composite_beam(w, d, length, wf, dw, rd, rc, hd, hs, dens)
                files.append(self._save("l_composite", i+1, dens, c, r, m))
        return files

    def generate_all(self):
        result = {}
        specs = [
            ("矩形柱", self.generate_rectangular_columns, 10),
            ("L形柱", self.generate_l_shaped_columns, 10),
            ("矩形梁", self.generate_rectangular_beams, 10),
            ("T形梁", self.generate_t_beams, 10),
            ("矩形叠合梁", self.generate_rectangular_composite, 5),
            ("T形叠合梁", self.generate_t_composite, 5),
            ("L形叠合梁", self.generate_l_composite, 5),
        ]
        print("=" * 60)
        print("  合成数据集生成器")
        print("=" * 60)
        for label, method, cnt in specs:
            print(f"\n 生成 {label} x {cnt} x 3 密度 ...", end=" ")
            files = method(cnt)
            result[label] = files
            print(f"完成 ({len(files)} 个文件)")
        print(f"\n{'='*60}")
        total = sum(len(v) for v in result.values())
        print(f"  总计 {total} 个点云文件")
        print(f"  输出: {self.output_dir}")
        print(f"{'='*60}")
        return result

    def _save(self, comp_type, idx, density, concrete_pts, rebar_pts, metadata, suffix=""):
        base = f"Mock_{comp_type}_{idx:03d}_{density}{suffix}"
        ply_path = os.path.join(self.output_dir, f"{base}.ply")
        json_path = os.path.join(self.output_dir, f"{base}.json")
        pcd = make_point_cloud(concrete_pts, rebar_pts)
        o3d.io.write_point_cloud(ply_path, pcd)
        meta_full = {
            "mock": True, "base_name": base, "density": density,
            "timestamp": datetime.now().isoformat(),
            "num_points": {"total": len(concrete_pts)+len(rebar_pts),
                           "concrete": len(concrete_pts), "rebar": len(rebar_pts)},
            **metadata,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta_full, f, indent=2, ensure_ascii=False)
        return ply_path


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="合成数据集生成器")
    parser.add_argument("--type", type=str, default="all",
                        choices=["all", "rectangular_column", "l_shaped_column",
                                 "rectangular_beam", "t_beam",
                                 "rectangular_composite", "t_composite", "l_composite"],
                        help="构件类型")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    gen = DatasetGenerator(output_dir=args.output)
    type_map = {
        "rectangular_column": gen.generate_rectangular_columns,
        "l_shaped_column": gen.generate_l_shaped_columns,
        "rectangular_beam": gen.generate_rectangular_beams,
        "t_beam": gen.generate_t_beams,
        "rectangular_composite": gen.generate_rectangular_composite,
        "t_composite": gen.generate_t_composite,
        "l_composite": gen.generate_l_composite,
    }

    if args.type == "all":
        gen.generate_all()
    else:
        cnt = args.count or 1
        files = type_map[args.type](cnt)
        print(f"生成完成: {len(files)} 个文件")


if __name__ == "__main__":
    main()
