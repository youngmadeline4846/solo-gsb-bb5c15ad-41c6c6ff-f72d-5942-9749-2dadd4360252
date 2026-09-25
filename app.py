"""纸张对折打孔展开预览 —— Flask 后端。

几何模型
========
- 原纸坐标系：原点 (0,0) 在左下角，x 向右、y 向上，单位 mm。
- 每次对折沿当前折叠面的水平或垂直中线，当前面对应尺寸减半。
- 折叠后逐条跟踪当前面四条边的身份：
  * "cut"  —— 开口边（对应原纸边界）
  * "fold" —— 折痕边
  重要不变量：历史上所有折痕最终都落在当前折叠面的边界上。因此打孔点只要
  与当前面四条边界的距离都大于打孔半径（带容差），展开后所有孔就不会压在
  原纸边界或任何折痕上，结果唯一、不含糊。
- 展开：按折叠顺序逆推（物理 mm 坐标）。每展开一层，静止层恒等映射，
  翻折层关于该层中线镜像；最终得到原纸上的全部孔位并按坐标去重。
- 全部折痕展开后构成 (2^nv - 1) x (2^nh - 1) 的等分网格。
- 反查打孔：给定原纸上的目标孔心，按折叠顺序正向逐层折回最终折叠面
  （fold_point，与 unfold_points 互逆），得到归一化孔心；再用同一套展开
  算法列出这一次打孔必然生成的全部孔位。目标孔心压纸张边界/折痕、或
  反查孔心压折叠面边界时，返回具体原因且前端不允许确认。
"""

from __future__ import annotations

import math
import os

from flask import Flask, jsonify, render_template, request, Response

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 配置（可用环境变量覆盖）
# ---------------------------------------------------------------------------
MAX_FOLDS = int(os.environ.get("MAX_FOLDS", "8"))
MAX_HOLES = int(os.environ.get("MAX_HOLES", "24"))
MIN_DIM = 20.0          # mm
MAX_DIM = 2000.0        # mm
MIN_RADIUS = 0.5        # mm
MAX_RADIUS = 50.0       # mm
EDGE_TOL = 0.05         # mm，边界判定容差（浮点误差 + 必须严格在内侧）

ALLOWED_FOLDS = ("vL", "vR", "hT", "hB")
EDGE_NAMES = {
    "fold": {"x0": "左侧折痕边", "x1": "右侧折痕边",
             "y0": "下侧折痕边", "y1": "上侧折痕边"},
    "cut": {"x0": "纸张左边界", "x1": "纸张右边界",
            "y0": "纸张下边界", "y1": "纸张上边界"},
}


# ---------------------------------------------------------------------------
# 折叠模型
# ---------------------------------------------------------------------------
def folded_state(width, height, folds):
    """按折叠顺序推进，返回 (当前面宽, 当前面高, 四条边身份, 垂直/水平次数)。

    边身份跟踪：每折一次，折线成为新的折痕边；被翻折一侧的旧边与静止一侧
    的旧边重合，两者身份合并（折痕优先——压折痕是更需要提示的情形）。
    例如 vR（右半折向左）：中线成为右侧折痕边，原左右两边界在左缘重合。
    """
    w, h = float(width), float(height)
    # 初始四条边都是原纸开口边界
    edges = {"x0": "cut", "x1": "cut", "y0": "cut", "y1": "cut"}
    nv = nh = 0

    def merge(a, b):
        return "fold" if "fold" in (a, b) else "cut"

    for f in folds:
        if f in ("vL", "vR"):
            w /= 2.0
            nv += 1
            if f == "vR":
                # 垂直对折，右半折向左：中线 → 右缘折痕；原右边镜像到左缘
                edges = {"x0": merge(edges["x0"], edges["x1"]), "x1": "fold",
                         "y0": edges["y0"], "y1": edges["y1"]}
            else:
                # 垂直对折，左半折向右：中线 → 左缘折痕；原左边镜像到右缘
                edges = {"x0": "fold", "x1": merge(edges["x0"], edges["x1"]),
                         "y0": edges["y0"], "y1": edges["y1"]}
        else:
            h /= 2.0
            nh += 1
            if f == "hT":
                # 水平对折，上半折向下：中线 → 上缘折痕；原上边镜像到下缘
                edges = {"x0": edges["x0"], "x1": edges["x1"],
                         "y0": merge(edges["y0"], edges["y1"]), "y1": "fold"}
            else:
                # 水平对折，下半折向上：中线 → 下缘折痕；原下边镜像到上缘
                edges = {"x0": edges["x0"], "x1": edges["x1"],
                         "y0": "fold", "y1": merge(edges["y0"], edges["y1"])}
    return w, h, edges, nv, nh


def unfold_points(points_mm, folds, width, height):
    """逆推孔位。输入最终折叠面局部 mm 坐标（原点为该面左下角），
    返回原纸 mm 坐标列表。

    逆推必须使用物理坐标：每展开一层，面宽/面高翻倍，中线位置随之变化，
    归一化坐标无法表达这一点。
    """
    cur = [(float(p["x"]), float(p["y"])) for p in points_mm]
    cw = width / (2.0 ** sum(1 for f in folds if f in ("vL", "vR")))
    ch = height / (2.0 ** sum(1 for f in folds if f in ("hT", "hB")))
    for f in reversed(folds):
        nxt = []
        if f in ("vL", "vR"):
            pw = 2.0 * cw          # 展开前一面的宽度
            line = pw / 2.0
            if f == "vR":          # 右侧折向左：静止左半恒等，右半镜像
                for x, y in cur:
                    nxt.append((x, y))
                    nxt.append((2.0 * line - x, y))
            else:                  # vL 左侧折向右：静止右半，左半镜像
                for x, y in cur:
                    nxt.append((line + x, y))
                    nxt.append((line - x, y))
            cw = pw
        else:
            ph = 2.0 * ch
            line = ph / 2.0
            if f == "hT":          # 上侧折向下：静止下半恒等，上半镜像
                for x, y in cur:
                    nxt.append((x, y))
                    nxt.append((x, 2.0 * line - y))
            else:                  # hB 下侧折向上：静止上半，下半镜像
                for x, y in cur:
                    nxt.append((x, line + y))
                    nxt.append((x, line - y))
            ch = ph
        cur = nxt
    return cur


def fold_point(point_mm, folds, width, height):
    """正向折回：把原纸上的点按折叠顺序映射到最终折叠面（物理 mm）。

    与 unfold_points 互逆：每折一层，静止半面的点恒等，翻折半面的点关于
    该层中线镜像，对应尺寸减半。unfold_points(fold_point(p)) 必包含 p。
    """
    x, y = float(point_mm[0]), float(point_mm[1])
    cw, ch = float(width), float(height)
    for f in folds:
        if f in ("vL", "vR"):
            line = cw / 2.0
            if f == "vR":          # 右半折向左：静止左半，右半镜像
                if x > line:
                    x = cw - x
            else:                  # vL 左半折向右：静止右半，左半镜像
                x = x - line if x >= line else line - x
            cw = line
        else:
            line = ch / 2.0
            if f == "hT":          # 上半折向下：静止下半，上半镜像
                if y > line:
                    y = ch - y
            else:                  # hB 下半折向上：静止上半，下半镜像
                y = y - line if y >= line else line - y
            ch = line
    return x, y


def dedupe_points(points, tol=1e-5):
    """按物理 mm 坐标去重（重合位置只保留一个）。"""
    out = []
    for x, y in points:
        if any(abs(x - q[0]) <= tol and abs(y - q[1]) <= tol for q in out):
            continue
        out.append((x, y))
    return [{"x": round(x, 6), "y": round(y, 6)} for x, y in out]


def check_point(px, py, w, h, radius, edges,
                label="孔位", hint="请调整孔位。"):
    """校验单个打孔点，返回中文错误信息；合法返回 None。"""
    r = radius + EDGE_TOL
    checks = (("x0", px), ("x1", w - px), ("y0", py), ("y1", h - py))
    for key, dist in checks:
        if dist < r:
            kind = edges[key]
            return (f"{label} ({px:.2f}, {py:.2f}) 距{EDGE_NAMES[kind][key]}仅 "
                    f"{max(dist, 0.0):.2f}mm，小于打孔半径 {radius:g}mm"
                    f"（{'折痕' if kind == 'fold' else '纸张边界'}，"
                    f"孔必须完整落在纸面内侧），{hint}")
    return None


def check_target_point(tx, ty, width, height, radius, nv, nh):
    """校验反查目标孔心（原纸 mm 坐标）：不能压纸张边界或任何折痕。

    返回中文原因；合法返回 None。
    """
    r = radius + EDGE_TOL
    for key, dist in (("x0", tx), ("x1", width - tx),
                      ("y0", ty), ("y1", height - ty)):
        if dist < r:
            return (f"目标孔心 ({tx:.2f}, {ty:.2f}) 距{EDGE_NAMES['cut'][key]}仅 "
                    f"{max(dist, 0.0):.2f}mm，小于打孔半径 {radius:g}mm"
                    "（纸张边界，孔必须完整落在纸面内侧），请改选其他目标点。")
    for k in range(1, 2 ** nv):
        pos = width * k / 2 ** nv
        if abs(tx - pos) < r:
            return (f"目标孔心 ({tx:.2f}, {ty:.2f}) 距垂直折痕 x={pos:.2f}mm 仅 "
                    f"{abs(tx - pos):.2f}mm，小于打孔半径 {radius:g}mm"
                    "（折痕，孔不能压在折痕上），请改选其他目标点。")
    for k in range(1, 2 ** nh):
        pos = height * k / 2 ** nh
        if abs(ty - pos) < r:
            return (f"目标孔心 ({tx:.2f}, {ty:.2f}) 距水平折痕 y={pos:.2f}mm 仅 "
                    f"{abs(ty - pos):.2f}mm，小于打孔半径 {radius:g}mm"
                    "（折痕，孔不能压在折痕上），请改选其他目标点。")
    return None


def crease_grid(width, height, nv, nh):
    """计算展开后全部折痕线段（mm，原纸坐标，y 自下而上）。"""
    lines = []
    for k in range(1, 2 ** nv):
        lines.append({"orient": "v", "pos": round(width * k / 2 ** nv, 6)})
    for k in range(1, 2 ** nh):
        lines.append({"orient": "h", "pos": round(height * k / 2 ** nh, 6)})
    return lines


# ---------------------------------------------------------------------------
# SVG 生成
# ---------------------------------------------------------------------------
def _svg_header(w_mm, h_mm):
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w_mm:.3f}mm" height="{h_mm:.3f}mm" '
        f'viewBox="0 0 {w_mm:.3f} {h_mm:.3f}">\n'
        f'<rect x="0" y="0" width="{w_mm:.3f}" height="{h_mm:.3f}" '
        'fill="#ffffff" stroke="#1f2937" stroke-width="0.35"/>\n'
    )


def _circle(cx, cy, r, extra=""):
    return (f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{r:.3f}" '
            f'fill="#dc2626" stroke="#7f1d1d" stroke-width="0.12"{extra}/>\n')


def build_unfolded_svg(width, height, holes, creases, radius, nv, nh):
    """展开图 SVG。y 由数学坐标（自下而上）换算为 SVG 坐标（自上而下）。"""
    w, h, r = float(width), float(height), float(radius)
    parts = [_svg_header(w, h)]
    parts.append(
        f'<desc>纸张 {w:g}x{h:g}mm；垂直对折 {nv} 次、水平对折 {nh} 次；'
        f'共 {len(holes)} 个孔</desc>\n')
    for c in creases:
        if c["orient"] == "v":
            parts.append(
                f'<line x1="{c["pos"]:.3f}" y1="0" x2="{c["pos"]:.3f}" y2="{h:.3f}" '
                'stroke="#9ca3af" stroke-width="0.2" stroke-dasharray="2 1.5"/>\n')
        else:
            yy = h - c["pos"]
            parts.append(
                f'<line x1="0" y1="{yy:.3f}" x2="{w:.3f}" y2="{yy:.3f}" '
                'stroke="#9ca3af" stroke-width="0.2" stroke-dasharray="2 1.5"/>\n')
    for hp in holes:
        parts.append(_circle(hp["x"], h - hp["y"], r))
    parts.append("</svg>\n")
    return "".join(parts)


def build_folded_svg(w, h, edges, points, radius, error_idx):
    """折叠面 SVG，使用 SVG 显示坐标（y 自上而下）。

    纸面在 viewBox 中内缩 pad；客户端靠返回的 pad / vbWidth / vbHeight
    把点击位置换算回折叠面 mm 坐标。
    """
    pad = max(radius * 2.2, 3.0)
    vb_w, vb_h = w + 2 * pad, h + 2 * pad

    def line(x1, y1, x2, y2, fold):
        color, dash, sw = (("#2563eb", "6 4", 0.9) if fold
                           else ("#1f2937", "none", 1.2))
        da = f' stroke-dasharray="{dash}"' if fold else ""
        return (f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
                f'stroke="{color}" stroke-width="{sw}"{da} stroke-linecap="round"/>\n')

    y_top, y_bot = pad, pad + h
    parts = [
        f'<svg id="folded-svg" viewBox="0 0 {vb_w:.3f} {vb_h:.3f}" '
        'preserveAspectRatio="xMidYMid meet" role="img" '
        'aria-label="折叠面，点击打孔">\n',
        # 点击层（覆盖整个 viewBox）
        f'<rect x="0" y="0" width="{vb_w:.3f}" height="{vb_h:.3f}" fill="transparent"/>\n',
        f'<rect x="{pad:.3f}" y="{pad:.3f}" width="{w:.3f}" height="{h:.3f}" '
        'fill="#fdf6e3" stroke="none"/>\n',
        line(pad, y_bot, pad + w, y_bot, edges["y0"] == "fold"),   # 下边
        line(pad, y_top, pad + w, y_top, edges["y1"] == "fold"),   # 上边
        line(pad, y_top, pad, y_bot, edges["x0"] == "fold"),       # 左边
        line(pad + w, y_top, pad + w, y_bot, edges["x1"] == "fold"),  # 右边
        # 安全区域：孔心进入此绿虚线框内才不会碰边/折痕
        f'<rect x="{pad + radius:.3f}" y="{pad + radius:.3f}" '
        f'width="{max(w - 2 * radius, 0):.3f}" height="{max(h - 2 * radius, 0):.3f}" '
        'fill="none" stroke="#16a34a" stroke-width="0.3" '
        'stroke-dasharray="1.5 1.5" opacity="0.55"/>\n',
    ]
    for i, p in enumerate(points):
        extra = ' data-hole="1"'
        if i == error_idx:
            extra += ' fill="#f59e0b"'
        # 数学坐标 y（自下而上）-> 显示坐标（自上而下）
        parts.append(_circle(pad + p["x"], pad + h - p["y"], radius, extra))
    parts.append("</svg>\n")
    return "".join(parts), pad, vb_w, vb_h


# ---------------------------------------------------------------------------
# 请求处理
# ---------------------------------------------------------------------------
def _bad(msg, code=400):
    return jsonify({"ok": False, "error": msg}), code


def _num(obj, key, lo, hi, name):
    val = obj.get(key)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{name}必须是数字")
    val = float(val)
    if not math.isfinite(val) or val < lo or val > hi:
        raise ValueError(f"{name}须在 {lo:g} ~ {hi:g} 之间")
    return val


def _validate_common(payload):
    """校验纸张/半径/折法，返回 (width, height, radius, folds)。"""
    if not isinstance(payload, dict):
        raise ValueError("请求体必须是 JSON 对象")

    width = _num(payload, "width", MIN_DIM, MAX_DIM, "纸张宽度")
    height = _num(payload, "height", MIN_DIM, MAX_DIM, "纸张高度")
    radius = _num(payload, "radius", MIN_RADIUS, MAX_RADIUS, "打孔半径")

    folds = payload.get("folds", [])
    if not isinstance(folds, list) or len(folds) > MAX_FOLDS:
        raise ValueError(f"折叠次数须为 0 ~ {MAX_FOLDS} 次")
    for f in folds:
        if f not in ALLOWED_FOLDS:
            raise ValueError("存在非法折叠方向，可选：vL / vR / hT / hB")
    return width, height, radius, folds


def _check_puncher_fits(w, h, radius):
    if w < 2 * radius + 2 * EDGE_TOL or h < 2 * radius + 2 * EDGE_TOL:
        raise ValueError(
            f"折叠后纸面为 {w:g} × {h:g}mm，放不下直径 {2 * radius:g}mm 的打孔器，"
            "请减少折叠次数或减小打孔半径。")


def compute(payload):
    """核心计算。成功返回结果 dict，失败抛出 ValueError。"""
    width, height, radius, folds = _validate_common(payload)

    points = payload.get("points", [])
    if not isinstance(points, list) or len(points) > MAX_HOLES:
        raise ValueError(f"打孔点数量须为 0 ~ {MAX_HOLES} 个")
    norm_pts = []
    for p in points:
        if not isinstance(p, dict):
            raise ValueError("打孔点格式错误")
        norm_pts.append({"x": _num(p, "x", 0.0, 1.0, "打孔点 x"),
                         "y": _num(p, "y", 0.0, 1.0, "打孔点 y")})

    w, h, edges, nv, nh = folded_state(width, height, folds)
    _check_puncher_fits(w, h, radius)

    # 客户端坐标是归一化的；换算为折叠面 mm 坐标
    mm_pts = [{"x": p["x"] * w, "y": p["y"] * h} for p in norm_pts]

    error_msg, error_idx = None, -1
    holes, creases, unfolded_svg = [], [], None
    for i, p in enumerate(mm_pts):
        err = check_point(p["x"], p["y"], w, h, radius, edges)
        if err:
            error_msg, error_idx = err, i
            break

    if error_msg is None:
        raw = unfold_points(mm_pts, folds, width, height)
        holes = dedupe_points(raw)
        creases = crease_grid(width, height, nv, nh)
        unfolded_svg = build_unfolded_svg(
            width, height, holes, creases, radius, nv, nh)

    folded_svg, pad, vb_w, vb_h = build_folded_svg(
        w, h, edges, mm_pts, radius, error_idx)

    return {
        "ok": True,
        "folded": {
            "width": round(w, 6), "height": round(h, 6),
            "layers": 2 ** len(folds),
            "edges": edges,
            "pad": round(pad, 6),
            "vbWidth": round(vb_w, 6), "vbHeight": round(vb_h, 6),
            "svg": folded_svg,
        },
        "creases": creases,
        "holeCount": len(holes),
        "expectedCount": len(mm_pts) * (2 ** len(folds)),
        "unfoldedSvg": unfolded_svg,
        "error": error_msg,
        "errorIndex": error_idx,
    }


def reverse_lookup(payload):
    """反查打孔：由原纸上的目标孔心折回最终折叠面，列出全部连带孔位。

    几何不合法（目标点或反查孔心压边界/折痕）时在结果中给出具体原因，
    由前端禁止确认；请求格式错误抛出 ValueError。
    """
    width, height, radius, folds = _validate_common(payload)

    target = payload.get("target")
    if not isinstance(target, dict):
        raise ValueError("缺少目标点 target（原纸归一化坐标 {x, y}）")
    tnx = _num(target, "x", 0.0, 1.0, "目标点 x")
    tny = _num(target, "y", 0.0, 1.0, "目标点 y")

    w, h, edges, nv, nh = folded_state(width, height, folds)
    _check_puncher_fits(w, h, radius)

    tx, ty = tnx * width, tny * height
    result = {
        "ok": True,
        "target": {"x": round(tx, 6), "y": round(ty, 6)},
        "foldedPoint": None,        # 最终折叠面归一化孔心
        "holes": [],                # 本次打孔必然生成的全部孔位（原纸 mm）
        "holeCount": 0,
        "expectedCount": 2 ** len(folds),
        "error": None,
    }

    # 1) 目标孔心自身：不能压纸张边界或任何折痕
    err = check_target_point(tx, ty, width, height, radius, nv, nh)
    if err is None:
        # 2) 按当前折叠顺序及折向逐层折回最终折叠面
        fx, fy = fold_point((tx, ty), folds, width, height)
        result["foldedPoint"] = {"x": round(fx / w, 6), "y": round(fy / h, 6)}
        # 3) 反查孔心也不能压折叠面边界（等价于某个连带孔压边界/折痕）
        err = check_point(fx, fy, w, h, radius, edges,
                          label="反查孔心", hint="请改选其他目标点。")
        if err is None:
            # 4) 用现有展开算法列出这一次打孔必然生成的全部孔位
            holes = dedupe_points(
                unfold_points([{"x": fx, "y": fy}], folds, width, height))
            result["holes"] = holes
            result["holeCount"] = len(holes)
    if err is not None:
        result["error"] = err
    return result


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        max_folds=MAX_FOLDS, max_holes=MAX_HOLES,
        min_dim=MIN_DIM, max_dim=MAX_DIM,
        min_radius=MIN_RADIUS, max_radius=MAX_RADIUS)


@app.route("/api/unfold", methods=["POST"])
def api_unfold():
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return _bad("请求不是合法 JSON")
    try:
        return jsonify(compute(payload))
    except ValueError as exc:
        return _bad(str(exc))


@app.route("/api/reverse", methods=["POST"])
def api_reverse():
    """反查打孔：入参同 /api/unfold，但 points 换成单个 target（原纸归一化）。"""
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return _bad("请求不是合法 JSON")
    try:
        return jsonify(reverse_lookup(payload))
    except ValueError as exc:
        return _bad(str(exc))


@app.route("/api/download", methods=["POST"])
def api_download():
    """只在计算合法时返回可下载的展开 SVG 文件。"""
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return _bad("请求不是合法 JSON")
    try:
        result = compute(payload)
    except ValueError as exc:
        return _bad(str(exc))
    if result["error"] or not result["unfoldedSvg"]:
        return _bad(result["error"] or "存在不合法孔位，无法生成展开图")
    return Response(
        result["unfoldedSvg"],
        mimetype="image/svg+xml",
        headers={"Content-Disposition":
                 'attachment; filename="unfolded.svg"'})


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
