# -*- coding: utf-8 -*-
"""178mm水下双目视觉：ROI顶点母材颜色基准 + 闭合色差区域 + 明暗结构 + 三维深度测量。

流程：
1. 左目图像点击四角选择ROI；
2. 持续采集双目深度，单像素累计至少6次有效观测后视为可靠深度点；
3. ROI可靠深度覆盖率达到70%，且四条边分别至少7个有效参考深度点后才进入分析；
4. 以四个ROI顶点附近的平整母材颜色建立Lab颜色基准，寻找与基准色差最大的闭合异常区域；
5. 在闭合异常区域内部进一步统计局部亮斑和阴影结构；
6. 用4n个等权参考深度点，经RANSAC+MAD剔除异常值后拟合统一参考平面；
7. 用异常区域内部相对参考平面的深度直接测量并显示凸起/凹陷。

显示：RED=凸起，BLUE=凹陷，BLACK=缺测。
R重选，S交换输入左右顺序，V有效深度预览，W保存，Q退出。
"""
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

SIZE = (1280, 720)
CAMERA_INDEX = None
NUM_DISPARITIES = 384  # 约70cm使用；范围变大不会扩大有效画面。
BLOCK_SIZE = 5  # 水下稀疏纹理优先用5；需要更多细节可试3。
MIN_OBSERVATIONS = 6  # 单像素累计至少6次有效深度后，才视为可靠深度点。
TARGET_ROI_COVERAGE = 0.70
REFERENCE_MIN_POINTS_PER_EDGE = 5
REFERENCE_RANSAC_ITERS = 800
REFERENCE_RANSAC_LIMIT_MM = 3.5
REFERENCE_PLANE_MIN_TOTAL_INLIERS = 20  # 同一参考平面总体至少20个内点
REFERENCE_PLANE_MIN_EDGE_INLIERS = 5    # 同一参考平面四条边各至少5个“边界采样点”内点
IMAGE_BLUR_DIAMETER = 7  # 双边滤波邻域。
CORNER_COLOR_INSET_PX = 7.0  # 顶点颜色采样中心向ROI内部偏移，避免采到ROI外部。
CORNER_COLOR_PATCH_RADIUS = 5  # 每个顶点母材颜色小块半径。
COLOR_DIFF_MIN = 9.0  # Lab色差最低门限，实际门限还会根据四角自身色差自适应提高。
COLOR_DIFF_CORNER_SCALE = 3.0  # 四角颜色本身波动的放大系数。
COLOR_CLOSE_RADIUS = 6  # 色差区域闭合半径。
ARC_MIN_RADIUS_PX = 8  # 圆/椭圆候选最小尺度。
ARC_MAX_RADIUS_RATIO = 0.45  # 最大尺度相对ROI短边比例。
ARC_SUPPORT_MIN = 0.20  # 候选圆/椭圆边界至少20%有真实边缘支持。
ARC_PAIR_CENTER_DIST = 18.0  # 两段圆弧中心接近阈值。
ARC_PAIR_RADIUS_RATIO = 0.35  # 两段圆弧半径相对差允许范围。
ELLIPSE_IOU_MIN = 0.18  # 轮廓与拟合椭圆最低重叠度，作为软先验。
ARC_DILATE_PX = 4  # 将检测到的圆弧/椭圆边界扩张后与色差区域融合。
ARC_HOUGH_DP = 1.2
ARC_HOUGH_MIN_DIST = 18
ARC_HOUGH_PARAM1 = 80
ARC_HOUGH_PARAM2 = 18
REGION_SIGN_MAJORITY = 0.60
REGION_MIN_MEASURED_PIXELS = 8
REGION_FILL_LIGHT_ALPHA = 0.38
REGION_MEASURED_ALPHA = 0.72
COLOR_MIN_AREA_RATIO = 0.002  # 最小候选面积占ROI比例。
COLOR_MAX_AREA_RATIO = 0.70  # 最大候选面积，避免把整幅照明渐变当成一个异常。
KEEP_SECONDARY_AREA_RATIO = 0.35
MULTI_SEED_MIN_AREA_PX = 45      # 多缺陷模式：独立色差seed最低面积，不再相对最大区域筛选。
MULTI_SEED_MAX_AREA_RATIO = 0.45 # 单个seed最大占ROI比例。
MULTI_HOUGH_EDGE_SUPPORT = 0.16  # 无颜色支持时，Hough圆边界至少16%由真实边缘支持。
MULTI_HOUGH_SEED_SUPPORT = 0.10  # 有颜色seed邻域支持时可适当放宽。
MULTI_HOUGH_NMS_CENTER = 0.35    # 圆心距离阈值，相对较小半径。
MULTI_HOUGH_NMS_RADIUS = 0.25    # 半径相对差阈值。
MULTI_FINAL_CLOSE_RADIUS = 2     # 只做很轻的闭合，不把多个独立缺陷粘在一起。  # 除最大闭合区域外，面积达到最大区域35%的也保留。
SHADING_BG_SIGMA = 18.0  # 局部明暗背景尺度。
SHADING_SIGMA_SCALE = 1.15  # 亮/暗阈值相对局部残差MAD的比例。
SHADING_MIN_ABS = 4.0  # 最低亮暗灰度差。
REFERENCE_EXCLUDE_MARGIN_PX = 6  # 异常颜色区域向外扩张后，从参考面采样中排除。
REFERENCE_GRID_STEP = 6  # 正常母材区域按6x6像素网格取一个稳健深度样本。
REFERENCE_GRID_MIN_VALID = 3  # 每个网格至少3个有效深度像素才生成参考点。
REFERENCE_DENSE_MAX_POINTS = 3500  # 防止参考点过多导致RANSAC过慢。
REFERENCE_DENSE_MIN_POINTS = 20  # 正常区域至少20个稳健参考点。
REFERENCE_DENSE_RANSAC_ITERS = 1200
REFERENCE_DENSE_RANSAC_LIMIT_MM = 4.0  # 无散斑情况下先用4mm粗找主平面。
IMAGE_BG_SIGMA = 21.0  # 更大尺度背景估计，用于分离缓慢水下光照与局部缺陷明暗变化。
DOG_SIGMA_SMALL = 2.2  # DoG小尺度。
DOG_SIGMA_LARGE = 11.0  # DoG大尺度，突出圆润局部起伏造成的亮暗结构。
SHADE_ENERGY_SIGMA = 5.0  # 计算局部亮/暗能量的尺度。
SHADE_PAIR_RADIUS = 9  # 判断亮区和暗区是否相邻的膨胀半径。
IMAGE_MORPH_RADIUS = 4  # 二维候选闭合半径，优先连接同一缺陷的亮区和暗区。
IMAGE_INNER_MARGIN = 9  # 忽略ROI边缘，避免把ROI轮廓和四边参考线当成缺陷。
MIN_ANOMALY_AREA_PX = 90  # 二维候选最小面积。
MAX_ANOMALY_AREA_RATIO = 0.45  # 单个候选最大占ROI面积比例。
MIN_SHAPE_SCORE = 0.18  # 圆润形状先验只作软筛选，避免误删不规则真实缺陷。
MIN_APPEARANCE_SCORE = 0.18  # 候选区域内部最低平均外观评分。
DEPTH_REGION_MIN_COVERAGE = 0.35  # 异常区域内部至少35%像素有可靠深度才参与凸凹判定。
DEPTH_SIGN_RATIO = 0.55  # 区域内至少55%的有效点支持同一凸凹方向。
LOCAL_RING_RADIUS = 9  # 异常边界外侧局部对比环宽度（像素）。
DEFECT_THRESHOLD_MM = 0.5  # 判定阈值，不是精度保证；不扣固定高度。
MIN_REFERENCE_POINTS = 12
WINDOW = "178mm - four-edge reference plane"
DEFECT_WINDOW = "Shape+Depth: RED bulge / BLUE dent / BLACK missing"
SAMPLES_PER_EDGE = 60  # 这里作为n：参考点总数4n，其中3n沿周长，n向ROI内部偏移
REFERENCE_PATCH_RADIUS = 2
REFERENCE_INSET_PX = 4.0  # n个参考点从ROI边界向内部偏移约3-5 px，这里取4 px
REFERENCE_LOCAL_LIMIT_MM = 10.0
REFERENCE_FIT_LIMIT_MM = 3.0
SAVE_ROOT = Path(__file__).resolve().parent / (Path(__file__).stem + "_results")

K1 = np.array([[795.0803385, 0., 645.23430909], [0., 793.44008566, 377.67209629], [0., 0., 1.]])

D1 = np.array([0.37827066, 0.08088432, 0.00699253, -0.01350958, 0.26886608])

K2 = np.array([[790.92403788, 0., 674.60832674], [0., 790.47639078, 352.51012355], [0., 0., 1.]])

D2 = np.array([0.35490627, 0.12942531, -0.01298832, 0.01507929, 0.11866722])

R = np.array([[.999295735, .00204162665, -.0374681916],
              [-.000875235484, .999515266, .0311202081],
              [.0375135653, -.0310654978, .998813129]])

T = np.array([[-178.44159845], [.6723965], [-5.66037523]])

def open_camera():
    for index in range(10) if CAMERA_INDEX is None else [CAMERA_INDEX]:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            for prop, value in [(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")),
                                (cv2.CAP_PROP_FRAME_WIDTH, 2560), (cv2.CAP_PROP_FRAME_HEIGHT, 720),
                                (cv2.CAP_PROP_FPS, 30), (cv2.CAP_PROP_BUFFERSIZE, 1)]:
                cap.set(prop, value)
            time.sleep(.3)
            ok, frame = cap.read()
            if ok and frame is not None and frame.shape[:2] == (720, 2560):
                print(f"[CAMERA] index={index}, 2560x720, baseline={np.linalg.norm(T):.3f}mm")
                return cap
        cap.release()
    raise RuntimeError("未找到2560x720左右拼接相机，请检查CAMERA_INDEX和连接。")

def rays_for_roi(roi, k, offset=(0., 0.)):
    x, y, w, h = roi
    v, u = np.indices((h, w), dtype=np.float64)
    return np.stack(((u+x+offset[0]-k[0, 2])/k[0, 0],
                     (v+y+offset[1]-k[1, 2])/k[1, 1], np.ones_like(u)), axis=-1)



def apply_disparity_black_band(view, stereo):
    """将双目视差搜索造成的理论不可用边缘带涂黑。

    对左目作为参考图时，SGBM需要在右目中向左搜索匹配，
    因此左目最左侧约 NUM_DISPARITIES 像素通常无法获得完整视差支持。
    这里直接把这条天然不可测区域涂黑，用作状态信息显示区。

    stereo.boundary 若存在，则优先使用其定义；否则使用 NUM_DISPARITIES。
    """
    h, w = view.shape[:2]

    band = int(getattr(stereo, 'boundary', NUM_DISPARITIES))
    band = max(0, min(w-1, band))

    if band > 0:
        view[:, :band] = (0, 0, 0)

    return band


def info_panel_on_disparity_band(view, lines, band_width):
    """把状态文字优先写在左侧视差不可用黑带内。

    若黑带过窄，文字仍从黑带左上开始显示，但不额外覆盖有效图像区域。
    """
    if band_width <= 0:
        return

    x0 = 8
    max_x = max(10, band_width - 8)
    y = 24

    # 字体略小，尽量容纳在NUM_DISPARITIES黑带内
    font_scale = 0.46
    line_gap = 24

    for line in lines:
        if y >= view.shape[0] - 8:
            break

        # 简单按字符数拆行，避免文字跨出黑带
        # 英文/数字为主时约每字符8像素左右
        max_chars = max(12, int(max_x / 8))
        chunks = [line[i:i+max_chars] for i in range(0, len(line), max_chars)]

        for chunk in chunks:
            cv2.putText(
                view, chunk, (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (230,230,230), 1, cv2.LINE_AA
            )
            y += line_gap
            if y >= view.shape[0] - 8:
                break


def info_panel_on_black_area(view, lines, preferred_height=None):
    """在顶部创建固定黑色信息栏，避免文字覆盖有效测量画面。"""
    if preferred_height is None:
        preferred_height = max(90, 30*len(lines)+14)
    h = min(preferred_height, view.shape[0])
    cv2.rectangle(view, (0, 0), (view.shape[1]-1, h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        yy = 27 + 28*i
        if yy >= h-5:
            break
        cv2.putText(
            view, line, (12, yy),
            cv2.FONT_HERSHEY_SIMPLEX, .56,
            (230, 230, 230), 1, cv2.LINE_AA
        )

def panel(image, lines):
    cv2.rectangle(image, (0, 0), (image.shape[1]-1, 29*len(lines)+10), (20, 20, 20), -1)
    for i, line in enumerate(lines):
        cv2.putText(image, line, (12, 26+29*i), cv2.FONT_HERSHEY_SIMPLEX, .57, (0, 255, 255), 1, cv2.LINE_AA)


class Stereo:
    def __init__(self):
        if NUM_DISPARITIES % 16 or not 0 < NUM_DISPARITIES < SIZE[0] - BLOCK_SIZE:
            raise ValueError('NUM_DISPARITIES必须为16的正倍数且小于图像宽度。')
        if BLOCK_SIZE < 1 or BLOCK_SIZE % 2 == 0:
            raise ValueError('BLOCK_SIZE必须为正奇数。')
        r1, r2, p1, p2, self.q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, SIZE, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        self.k = p1[:, :3]
        self.maps_l = cv2.initUndistortRectifyMap(K1, D1, r1, p1, SIZE, cv2.CV_32FC1)
        self.maps_r = cv2.initUndistortRectifyMap(K2, D2, r2, p2, SIZE, cv2.CV_32FC1)
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=NUM_DISPARITIES, blockSize=BLOCK_SIZE,
            P1=8*BLOCK_SIZE**2, P2=32*BLOCK_SIZE**2, uniquenessRatio=6,
            disp12MaxDiff=1, speckleWindowSize=30, speckleRange=2, preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        x, y, w, h = cv2.getValidDisparityROI(roi1, roi2, 0, NUM_DISPARITIES, BLOCK_SIZE)
        self.boundary = x
        self.region = np.zeros(SIZE[::-1], bool)
        self.region[y:y+h, x:x+w] = True

    def rectify(self, frame, swap):
        width, height = SIZE
        if frame.shape[:2] != (height, 2 * width):
            raise ValueError(f'相机输出尺寸已改变，需要 {2*width}×{height} 左右拼接图。')
        left, right = frame[:, :width], frame[:, width:]
        if swap:
            left, right = right, left
        return (cv2.remap(left, *self.maps_l, cv2.INTER_LINEAR),
                cv2.remap(right, *self.maps_r, cv2.INTER_LINEAR))

    def depth(self, left, right):
        gray_l, gray_r = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in (left, right)]
        disparity = self.matcher.compute(gray_l, gray_r).astype(np.float32)/16.
        z = cv2.reprojectImageTo3D(disparity, self.q)[:, :, 2]
        valid = self.region & (disparity > 0) & np.isfinite(z) & (z > 0)
        # 保留SGBM自带过滤，不再加第二次严格左右一致性/运动/纹理门槛。
        return np.where(valid, z, np.nan).astype(np.float32)


def selection(points):
    pts = np.asarray(points, np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        raise ValueError('请选择四个不同的角点。')
    hull = cv2.convexHull(pts).reshape(-1, 2)
    if len(hull) != 4 or cv2.contourArea(hull) < 2500:
        raise ValueError('需要足够大的凸四边形，不能重复或共线。')
    center = hull.mean(axis=0)
    hull = hull[np.argsort(np.arctan2(hull[:, 1]-center[1], hull[:, 0]-center[0]))]
    x, y, w, h = cv2.boundingRect(hull)
    if x < 0 or y < 0 or x+w > SIZE[0] or y+h > SIZE[1]:
        raise ValueError('角点超出图像。')
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.rint(hull-[x, y]).astype(np.int32), 1)
    # 返回四条边的几何，不再使用向内部扩展的参考带。
    geometry = edge_geometry(hull - [x, y])
    return hull, (x, y, w, h), mask.astype(bool), geometry


def edge_geometry(corners):
    """构造等权参考采样点。

    设 n=SAMPLES_PER_EDGE：
    - 3n 个点按四边形真实像素周长等弧长均匀采样；
    - n 个点同样按周长均匀选取，但沿“边界点 -> ROI中心”方向向内偏移约4 px。
    总数固定为4n。所有参考点后续等权参与鲁棒平面拟合。
    """
    corners = np.asarray(corners, np.float32)
    matrix = cv2.getPerspectiveTransform(
        corners, np.array([[0,0],[1,0],[1,1],[0,1]], np.float32)
    )
    center = corners.mean(axis=0)

    seg = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(seg, axis=1).astype(np.float64)
    perimeter = float(lengths.sum())
    if perimeter <= 1e-6:
        raise ValueError("四边形周长无效。")
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])

    def sample_at_distance(distance):
        d = float(distance % perimeter)
        edge = int(np.searchsorted(cumulative[1:], d, side='right'))
        edge = min(edge, 3)
        local = d - cumulative[edge]
        t = local / max(float(lengths[edge]), 1e-9)
        p = (1.0-t)*corners[edge] + t*corners[(edge+1) % 4]
        return p.astype(np.float32), edge

    n = int(SAMPLES_PER_EDGE)
    perimeter_samples, edge_ids, sample_types = [], [], []

    # 3n：沿真实周长等弧长取点，半步偏移避免重复落在顶点。
    for j in range(3*n):
        d = (j + 0.5) * perimeter / (3*n)
        p, edge = sample_at_distance(d)
        perimeter_samples.append(p)
        edge_ids.append(edge)
        sample_types.append(0)  # 0=边界点

    # n：再次按周长均匀取点，然后向ROI内部偏移约4 px。
    for j in range(n):
        d = (j + 0.5) * perimeter / n
        p, edge = sample_at_distance(d)
        direction = center - p
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            p = p + direction / norm * REFERENCE_INSET_PX
        perimeter_samples.append(p.astype(np.float32))
        edge_ids.append(edge)
        sample_types.append(1)  # 1=内偏点

    samples = np.asarray(perimeter_samples, np.float32)
    return dict(
        corners=corners,
        matrix=matrix,
        samples=samples,
        edges=np.asarray(edge_ids, np.int32),
        sample_types=np.asarray(sample_types, np.uint8)
    )

def collect_edge_depths(depth, mask, geometry, observations=None, spread=None, captured_frames=None):
    """采集4n个等权参考点的深度。

    每个参考位置使用小邻域内有效深度的中位数。
    这里只判断“有深度/无深度”，不计算置信度、不赋权。
    真正的异常值剔除统一放在平面拟合阶段完成。
    """
    values = np.full(len(geometry['samples']), np.nan, np.float64)
    h, w = depth.shape

    for i, (u, v) in enumerate(geometry['samples']):
        x, y = int(round(u)), int(round(v))
        r = REFERENCE_PATCH_RADIUS
        x0, x1 = max(0, x-r), min(w, x+r+1)
        y0, y1 = max(0, y-r), min(h, y+r+1)
        patch = depth[y0:y1, x0:x1]
        good = mask[y0:y1, x0:x1] & np.isfinite(patch) & (patch > 0)
        if good.sum() >= 3:
            values[i] = float(np.median(patch[good]))

    uv = cv2.perspectiveTransform(
        geometry['samples'].reshape(-1,1,2), geometry['matrix']
    ).reshape(-1,2)

    usable = np.isfinite(values)
    return uv, values, usable

def fit_edge_plane(uv, values, usable, edge_ids, sample_types):
    """等权鲁棒参考平面拟合。

    最终接受条件为硬条件：
    1) 同一个平面总体至少20个参考内点；
    2) 四条边上，每条边至少有5个“真实边界采样点”(sample_type==0)属于该平面；
    3) n个向ROI内部偏移的参考点可以参与总体20点和平面拟合，
       但不计入“四条边各5点”的边界条件。

    流程：
    RANSAC粗找共面点 -> 检查20总点+四边各5点 -> 等权最小二乘 ->
    MAD迭代剔除异常值。MAD若会破坏上述硬条件，则停止继续剔除。
    """
    ids = np.flatnonzero(usable)
    if len(ids) < REFERENCE_PLANE_MIN_TOTAL_INLIERS:
        raise ValueError(
            f'当前只有{len(ids)}个有效参考深度点，'
            f'少于总体最低要求{REFERENCE_PLANE_MIN_TOTAL_INLIERS}。'
        )

    design_all = np.column_stack([uv[ids], np.ones(len(ids))]).astype(np.float64)
    z_all = values[ids].astype(np.float64)
    edges_all = np.asarray(edge_ids, np.int32)[ids]
    types_all = np.asarray(sample_types, np.uint8)[ids]

    # 非常宽松的粗飞点筛选，不按固定深度差把倾斜平面误删。
    z_med = float(np.median(z_all))
    z_mad = float(1.4826*np.median(np.abs(z_all-z_med)))
    if z_mad > 0:
        coarse = np.abs(z_all-z_med) <= max(25.0, 8.0*z_mad)
    else:
        coarse = np.ones(len(z_all), bool)

    # 粗筛不能破坏基本总点数，否则不用粗筛。
    if coarse.sum() >= REFERENCE_PLANE_MIN_TOTAL_INLIERS:
        design = design_all[coarse]
        z = z_all[coarse]
        edges = edges_all[coarse]
        types = types_all[coarse]
        coarse_ids = np.flatnonzero(coarse)
    else:
        design = design_all
        z = z_all
        edges = edges_all
        types = types_all
        coarse_ids = np.arange(len(z_all))

    def plane_condition(inliers):
        """返回：是否满足硬条件、总内点数、四边边界内点数。"""
        inliers = np.asarray(inliers, bool)
        total = int(inliers.sum())
        edge_counts = []
        for edge in range(4):
            # 四边各5点只统计真实周长上的3n边界点，不统计内偏n点。
            c = int(np.sum(inliers & (edges == edge) & (types == 0)))
            edge_counts.append(c)
        ok = (
            total >= REFERENCE_PLANE_MIN_TOTAL_INLIERS
            and all(c >= REFERENCE_PLANE_MIN_EDGE_INLIERS for c in edge_counts)
        )
        return ok, total, edge_counts

    # ---------- RANSAC ----------
    rng = np.random.default_rng(178)
    best = None
    best_total = -1
    best_min_edge = -1
    best_med = np.inf

    for _ in range(REFERENCE_RANSAC_ITERS):
        chosen = rng.choice(len(z), 3, replace=False)
        if np.linalg.matrix_rank(design[chosen]) < 3:
            continue

        try:
            coef = np.linalg.solve(design[chosen], z[chosen])
        except np.linalg.LinAlgError:
            continue

        residual = np.abs(design @ coef - z)
        inliers = residual <= REFERENCE_RANSAC_LIMIT_MM
        ok, total, edge_counts = plane_condition(inliers)
        min_edge = min(edge_counts) if edge_counts else 0
        med = float(np.median(residual[inliers])) if total else np.inf

        # 优先寻找满足硬条件的平面；在满足条件的候选中优先总内点更多，
        # 其次四边最弱一边内点更多，再其次残差中位数更小。
        if ok:
            if (
                best is None
                or total > best_total
                or (total == best_total and min_edge > best_min_edge)
                or (total == best_total and min_edge == best_min_edge and med < best_med)
            ):
                best = inliers
                best_total = total
                best_min_edge = min_edge
                best_med = med

    if best is None:
        # 为了给出有意义的诊断，再找“最接近”的RANSAC模型。
        near_best = None
        near_score = (-1, -1)
        near_edges = [0, 0, 0, 0]
        near_total = 0
        for _ in range(250):
            chosen = rng.choice(len(z), 3, replace=False)
            if np.linalg.matrix_rank(design[chosen]) < 3:
                continue
            try:
                coef = np.linalg.solve(design[chosen], z[chosen])
            except np.linalg.LinAlgError:
                continue
            residual = np.abs(design @ coef - z)
            inliers = residual <= REFERENCE_RANSAC_LIMIT_MM
            _, total, edge_counts = plane_condition(inliers)
            score = (min(edge_counts), total)
            if score > near_score:
                near_score = score
                near_best = inliers
                near_edges = edge_counts
                near_total = total

        raise ValueError(
            f'未找到满足条件的统一参考平面：'
            f'当前最佳约为总内点{near_total}个，'
            f'四边边界内点分别{near_edges}；'
            f'要求总内点>={REFERENCE_PLANE_MIN_TOTAL_INLIERS}，'
            f'且四边各>={REFERENCE_PLANE_MIN_EDGE_INLIERS}。'
        )

    # ---------- MAD迭代剔除 ----------
    active = best.copy()

    for _ in range(8):
        a = design[active]

        # 参考点必须真正铺开在二维区域，避免退化成近似一条线。
        spread2d = np.linalg.svd(
            a[:, :2] - a[:, :2].mean(axis=0), compute_uv=False
        )
        if len(spread2d) < 2 or spread2d[1] < 0.03:
            raise ValueError('当前共面参考点空间分布退化，无法稳定拟合二维参考平面。')

        coef = np.linalg.lstsq(a, z[active], rcond=None)[0]
        signed_residual = design @ coef - z

        r_active = signed_residual[active]
        center = float(np.median(r_active))
        mad = float(1.4826*np.median(np.abs(r_active-center)))

        # RANSAC先粗找，MAD逐步收紧；最低保持0.6mm防止过度删点。
        limit = min(
            REFERENCE_FIT_LIMIT_MM,
            max(0.6, 3.5*mad)
        )
        updated = np.abs(signed_residual-center) <= limit

        ok, _, _ = plane_condition(updated)

        # 如果继续剔除会导致“总20点 / 四边各5点”任一条件失效，
        # 就停止，保留上一轮满足硬条件的active。
        if not ok:
            break

        if np.array_equal(updated, active):
            active = updated
            break

        active = updated

    # ---------- 最终再次确认硬条件 ----------
    ok, total_final, edge_counts_final = plane_condition(active)
    if not ok:
        raise ValueError(
            f'异常值剔除后不再满足参考平面条件：总内点{total_final}，'
            f'四边分别{edge_counts_final}。'
        )

    # 最终所有保留点等权普通最小二乘。
    coef = np.linalg.lstsq(design[active], z[active], rcond=None)[0]
    residual = design @ coef - z

    accepted = np.zeros(len(values), bool)
    accepted_local = coarse_ids[np.flatnonzero(active)]
    accepted[ids[accepted_local]] = True

    rms = float(np.sqrt(np.mean(residual[active]**2)))
    return coef, accepted, rms




def _ellipse_mask(shape, ellipse):
    mask = np.zeros(shape, np.uint8)
    try:
        cv2.ellipse(mask, ellipse, 1, -1)
    except cv2.error:
        return mask.astype(bool)
    return mask.astype(bool)


def _ellipse_outline_mask(shape, ellipse, thickness=2):
    mask = np.zeros(shape, np.uint8)
    try:
        cv2.ellipse(mask, ellipse, 1, thickness)
    except cv2.error:
        return mask.astype(bool)
    return mask.astype(bool)


def augment_candidate_with_round_shapes(base_roi, roi_mask, seed_mask):
    """多实例圆/椭圆增强。

    设计原则：
    1) 每个颜色seed独立拟合圆/椭圆；
    2) Hough圆在整个ROI内运行，用于补漏颜色较弱的小圆；
    3) Hough结果做NMS去重，但相互重叠且圆心明显不同的圆仍保留；
    4) 不再把所有候选做大尺度闭运算，以免多个缺陷粘成一片。
    """
    h, w = roi_mask.shape

    gray = cv2.cvtColor(base_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.1)

    edges = cv2.Canny(gray, 35, 95)
    edges[~roi_mask] = 0

    shape_mask = np.zeros((h, w), np.uint8)
    shape_candidates = []

    # 颜色seed轻微膨胀，只用于“支持”判断，不作为最终大范围连接。
    seed_d = cv2.dilate(
        seed_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ).astype(bool)

    # ---------- A. 每个颜色seed独立拟合椭圆 ----------
    seed_contours, _ = cv2.findContours(
        seed_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    for contour in seed_contours:
        area = float(cv2.contourArea(contour))
        if area < MULTI_SEED_MIN_AREA_PX:
            continue

        # 原seed本身先保留，避免拟合失败时完全丢失。
        cmask = np.zeros((h, w), np.uint8)
        cv2.drawContours(cmask, [contour], -1, 1, -1)
        shape_mask[cmask.astype(bool) & roi_mask] = 1

        if len(contour) < 5:
            continue

        try:
            ellipse = cv2.fitEllipse(contour)
        except cv2.error:
            continue

        (cx, cy), (ma, mi), angle = ellipse
        if min(ma, mi) < 2 * ARC_MIN_RADIUS_PX:
            continue
        if max(ma, mi) > ARC_MAX_RADIUS_RATIO * 2 * min(h, w):
            continue

        emask = _ellipse_mask((h, w), ellipse) & roi_mask

        contour_mask = cmask.astype(bool)
        inter = int(np.logical_and(emask, contour_mask).sum())
        union = int(np.logical_or(emask, contour_mask).sum())
        iou = float(inter / max(1, union))

        outline = _ellipse_outline_mask((h, w), ellipse, thickness=2) & roi_mask
        edge_support = float((edges[outline] > 0).mean()) if np.any(outline) else 0.0

        # 椭圆可以由颜色seed形状或真实边缘任一提供支持。
        if iou >= ELLIPSE_IOU_MIN or edge_support >= 0.12:
            shape_mask[emask] = 1
            shape_candidates.append(
                ("ellipse", ellipse, max(iou, edge_support))
            )

    # ---------- B. 整个ROI内做Hough圆补漏 ----------
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=ARC_HOUGH_DP,
        minDist=max(12, ARC_HOUGH_MIN_DIST),
        param1=ARC_HOUGH_PARAM1,
        param2=ARC_HOUGH_PARAM2,
        minRadius=ARC_MIN_RADIUS_PX,
        maxRadius=max(
            ARC_MIN_RADIUS_PX + 1,
            int(ARC_MAX_RADIUS_RATIO * min(h, w))
        )
    )

    raw_circles = []

    if circles is not None:
        for cx, cy, r in np.round(circles[0]).astype(int):
            if r <= 0 or not (0 <= cx < w and 0 <= cy < h):
                continue

            fill = np.zeros((h, w), np.uint8)
            cv2.circle(fill, (cx, cy), r, 1, -1)
            fill_bool = fill.astype(bool)

            # 圆主体至少大部分位于ROI内。
            circle_area = max(1, int(fill_bool.sum()))
            inside_ratio = float((fill_bool & roi_mask).sum() / circle_area)
            if inside_ratio < 0.75:
                continue

            outline = np.zeros((h, w), np.uint8)
            cv2.circle(outline, (cx, cy), r, 1, 2)
            outline_bool = outline.astype(bool) & roi_mask

            edge_support = (
                float((edges[outline_bool] > 0).mean())
                if np.any(outline_bool) else 0.0
            )
            seed_support = (
                float(seed_d[outline_bool].mean())
                if np.any(outline_bool) else 0.0
            )

            # 两条接受路径：
            # 1) 边缘足够强，即使颜色弱也可补漏；
            # 2) 有一定颜色seed支持，边缘门槛可适当降低。
            accepted = (
                edge_support >= MULTI_HOUGH_EDGE_SUPPORT
                or (
                    seed_support >= MULTI_HOUGH_SEED_SUPPORT
                    and edge_support >= 0.08
                )
            )
            if not accepted:
                continue

            score = 0.70 * edge_support + 0.30 * seed_support
            raw_circles.append((cx, cy, r, score))

    # ---------- C. Hough圆NMS去重 ----------
    raw_circles.sort(key=lambda x: x[3], reverse=True)
    kept_circles = []

    for cand in raw_circles:
        cx, cy, r, score = cand
        duplicate = False

        for kx, ky, kr, kscore in kept_circles:
            center_dist = float(np.hypot(cx - kx, cy - ky))
            radius_diff = abs(r - kr) / max(r, kr, 1)

            # 只有“几乎同圆心 + 半径接近”才认为是重复。
            # 两个真实重叠圆通常圆心不同，因此不会被误删。
            if (
                center_dist <= MULTI_HOUGH_NMS_CENTER * min(r, kr)
                and radius_diff <= MULTI_HOUGH_NMS_RADIUS
            ):
                duplicate = True
                break

        if not duplicate:
            kept_circles.append(cand)

    for cx, cy, r, score in kept_circles:
        fill = np.zeros((h, w), np.uint8)
        cv2.circle(fill, (cx, cy), r, 1, -1)
        fill_bool = fill.astype(bool) & roi_mask
        shape_mask[fill_bool] = 1
        shape_candidates.append(("circle", (cx, cy, r), score))

    # ---------- D. 最终联合 ----------
    augmented = seed_mask | shape_mask.astype(bool)

    # 只做很轻微的闭合，用于补1-2像素裂口。
    # 不再使用 COLOR_CLOSE_RADIUS+2 这种大闭合，避免9个圆互相粘连。
    if MULTI_FINAL_CLOSE_RADIUS > 0:
        kr = MULTI_FINAL_CLOSE_RADIUS
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2*kr+1, 2*kr+1)
        )
        augmented = cv2.morphologyEx(
            augmented.astype(np.uint8),
            cv2.MORPH_CLOSE,
            k,
            iterations=1
        ).astype(bool)

    augmented &= roi_mask
    accepted_shape = shape_mask.astype(bool) & roi_mask

    return augmented, accepted_shape, edges, shape_candidates

def build_auto_appearance(base, roi, mask, geometry):
    """自动生成多实例异常候选。

    不再按最大缺陷面积比例筛选。
    只要单个候选自身满足最低绝对面积，就独立保留。
    """
    x, y, w, h = roi
    base_roi = base[y:y+h, x:x+w].copy()

    appearance = extract_anomaly_regions(
        base_roi, mask, geometry['corners']
    )

    seed_mask = appearance[0]

    augmented_mask, shape_support_mask, arc_edges, shape_candidates = (
        augment_candidate_with_round_shapes(
            base_roi, mask, seed_mask
        )
    )

    contours, _ = cv2.findContours(
        augmented_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 只按每个候选自身的绝对面积筛选，不与最大候选比较。
    keep = []
    roi_area = max(1, int(mask.sum()))
    max_area = MULTI_SEED_MAX_AREA_RATIO * roi_area

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MULTI_SEED_MIN_AREA_PX:
            continue
        if area > max_area:
            continue
        keep.append(contour)

    keep = sorted(keep, key=cv2.contourArea, reverse=True)

    final_mask_u8 = np.zeros_like(augmented_mask, np.uint8)
    if keep:
        cv2.drawContours(final_mask_u8, keep, -1, 1, -1)

    final_mask = final_mask_u8.astype(bool) & mask

    appearance_list = list(appearance)
    appearance_list[0] = final_mask
    appearance_list[1] = keep

    appearance_info = dict(appearance_list[-1])
    appearance_info.update(
        shape_support_pixels=int(shape_support_mask.sum()),
        shape_support_candidates=int(len(shape_candidates)),
        geometry_augmented_area_ratio=float(
            final_mask.sum() / max(1, mask.sum())
        ),
        multi_instance_mode=True,
        final_instance_count=int(len(keep))
    )
    appearance_list[-1] = appearance_info

    return tuple(appearance_list)

def collect_dense_reference_points(depth, roi_mask, candidate_mask, geometry):
    """在ROI内除异常候选区域之外的大面积正常母材上取参考深度。

    不是逐像素直接喂给RANSAC，而是按小网格分块：
    - 排除颜色异常区域及其外扩边界；
    - 每个6x6网格内，对有效深度取中位数；
    - 以该网格有效像素的中位位置作为参考点；
    这样既能获得大量空间分布参考点，又能抑制无散斑条件下的单像素飞点。
    """
    h, w = depth.shape

    # 异常区域适当外扩，避免缺陷边缘/阴影过渡带污染参考平面。
    if candidate_mask is not None and np.any(candidate_mask):
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2*REFERENCE_EXCLUDE_MARGIN_PX+1, 2*REFERENCE_EXCLUDE_MARGIN_PX+1)
        )
        excluded = cv2.dilate(candidate_mask.astype(np.uint8), k).astype(bool)
    else:
        excluded = np.zeros_like(roi_mask, bool)

    normal_mask = roi_mask & ~excluded & np.isfinite(depth) & (depth > 0)

    xy = []
    zvals = []
    step = int(REFERENCE_GRID_STEP)

    for y0 in range(0, h, step):
        y1 = min(h, y0+step)
        for x0 in range(0, w, step):
            x1 = min(w, x0+step)
            block_mask = normal_mask[y0:y1, x0:x1]
            if int(block_mask.sum()) < REFERENCE_GRID_MIN_VALID:
                continue

            block_depth = depth[y0:y1, x0:x1]
            ys, xs = np.nonzero(block_mask)
            vals = block_depth[block_mask].astype(np.float64)

            # 深度用中位数，位置也用有效像素中位坐标，降低飞点影响。
            zmed = float(np.median(vals))
            u = float(x0 + np.median(xs))
            v = float(y0 + np.median(ys))

            if np.isfinite(zmed) and zmed > 0:
                xy.append((u, v))
                zvals.append(zmed)

    if len(zvals) == 0:
        return (np.empty((0,2), np.float64),
                np.empty((0,), np.float64),
                normal_mask, excluded)

    xy = np.asarray(xy, np.float32)
    zvals = np.asarray(zvals, np.float64)

    # 若点太多，按固定随机种子均匀抽样，保证运行速度和可重复性。
    if len(zvals) > REFERENCE_DENSE_MAX_POINTS:
        rng = np.random.default_rng(178)
        ids = np.sort(rng.choice(
            len(zvals), REFERENCE_DENSE_MAX_POINTS, replace=False
        ))
        xy = xy[ids]
        zvals = zvals[ids]

    # 转到与reference_depth_map一致的四边形归一化坐标系。
    uv = cv2.perspectiveTransform(
        xy.reshape(-1,1,2), geometry['matrix']
    ).reshape(-1,2).astype(np.float64)

    return uv, zvals, normal_mask, excluded


def fit_dense_reference_plane(uv, values):
    """对正常母材的大量网格中位深度做等权鲁棒平面拟合。

    重点适应“未喷散斑、深度有大量飞点”的情况：
    1) 网格中位数先抑制单像素飞点；
    2) RANSAC找占比最大的主平面；
    3) 只要求最终>=20个共面参考点；
    4) MAD迭代剔除剩余异常点；
    5) 最终等权最小二乘拟合。
    """
    uv = np.asarray(uv, np.float64)
    values = np.asarray(values, np.float64)

    good = (
        np.isfinite(values)
        & np.isfinite(uv).all(axis=1)
        & (values > 0)
    )
    uv = uv[good]
    z = values[good]

    if len(z) < REFERENCE_DENSE_MIN_POINTS:
        raise ValueError(
            f'排除全部异常实例后只有{len(z)}个正常母材参考网格，'
            f'至少需要{REFERENCE_DENSE_MIN_POINTS}个。'
        )

    design = np.column_stack([uv, np.ones(len(uv))]).astype(np.float64)

    rng = np.random.default_rng(178)
    best = None
    best_count = -1
    best_med = np.inf

    for _ in range(REFERENCE_DENSE_RANSAC_ITERS):
        chosen = rng.choice(len(z), 3, replace=False)
        if np.linalg.matrix_rank(design[chosen]) < 3:
            continue
        try:
            coef = np.linalg.solve(design[chosen], z[chosen])
        except np.linalg.LinAlgError:
            continue

        residual = np.abs(design @ coef - z)
        inliers = residual <= REFERENCE_DENSE_RANSAC_LIMIT_MM
        count = int(inliers.sum())
        med = float(np.median(residual[inliers])) if count else np.inf

        if count > best_count or (count == best_count and med < best_med):
            best = inliers
            best_count = count
            best_med = med

    if best is None or int(best.sum()) < REFERENCE_DENSE_MIN_POINTS:
        raise ValueError(
            f'正常母材RANSAC最多只找到{0 if best is None else int(best.sum())}个共面点，'
            f'至少需要{REFERENCE_DENSE_MIN_POINTS}个。'
        )

    active = best.copy()

    # MAD逐步收紧，但绝不低于20个参考点。
    for _ in range(8):
        a = design[active]
        coef = np.linalg.lstsq(a, z[active], rcond=None)[0]
        signed_residual = design @ coef - z

        r = signed_residual[active]
        center = float(np.median(r))
        mad = float(1.4826*np.median(np.abs(r-center)))

        limit = min(
            REFERENCE_DENSE_RANSAC_LIMIT_MM,
            max(0.8, 3.5*mad)
        )
        updated = np.abs(signed_residual-center) <= limit

        if int(updated.sum()) < REFERENCE_DENSE_MIN_POINTS:
            break
        if np.array_equal(updated, active):
            active = updated
            break
        active = updated

    # 检查空间分布不能退化成一条线。
    a = design[active]
    spread = np.linalg.svd(
        a[:,:2] - a[:,:2].mean(axis=0), compute_uv=False
    )
    if len(spread) < 2 or spread[1] < 0.03:
        raise ValueError('正常母材共面点空间分布退化，无法稳定拟合二维平面。')

    coef = np.linalg.lstsq(a, z[active], rcond=None)[0]
    residual = design @ coef - z
    rms = float(np.sqrt(np.mean(residual[active]**2)))

    return coef, active, rms, uv, z

def reference_depth_map(shape, matrix, coef):
    y,x=np.indices(shape,dtype=np.float32)
    uv=cv2.perspectiveTransform(np.dstack([x,y]).reshape(-1,1,2),matrix).reshape(*shape,2)
    return (coef[0]*uv[:,:,0]+coef[1]*uv[:,:,1]+coef[2]).astype(np.float32)


def reference_areas(z_plane, mask, roi, k):
    xyz=rays_for_roi(roi,k)*z_plane[...,None]
    p00,p01,p10,p11=xyz[:-1,:-1],xyz[:-1,1:],xyz[1:,:-1],xyz[1:,1:]
    cell=.5*(np.linalg.norm(np.cross(p01-p00,p10-p00),axis=-1)
              +np.linalg.norm(np.cross(p11-p01,p10-p01),axis=-1))
    good=mask & np.isfinite(z_plane) & (z_plane>0)
    cell_valid=good[:-1,:-1]&good[:-1,1:]&good[1:,:-1]&good[1:,1:]
    cell=np.where(cell_valid & np.isfinite(cell),cell,0)/4
    area=np.zeros(z_plane.shape,np.float32)
    area[:-1,:-1]+=cell; area[:-1,1:]+=cell; area[1:,:-1]+=cell; area[1:,1:]+=cell
    return area


def fuse(frames):
    stack = np.stack(frames).astype(np.float32)
    stack[(stack <= 0) | ~np.isfinite(stack)] = np.nan
    count = np.isfinite(stack).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        depth = np.nanmedian(stack, axis=0)
        spread = 1.4826*np.nanmedian(np.abs(stack-depth), axis=0)
    depth[count < MIN_OBSERVATIONS] = np.nan
    # 所有采集帧参与融合；单像素不足MIN_OBSERVATIONS次有效观测则保持缺测。
    return depth, count.astype(np.uint8), spread.astype(np.float32)






def reference_distribution_ok(values, usable, geometry):
    """四条边分别至少5个有效深度点即可进入平面拟合。

    内偏的n个点继续参与RANSAC/MAD/最终平面拟合，
    但不再作为停止采集的硬条件。
    """
    info = {}
    ok = True

    for edge in range(4):
        ids = np.flatnonzero(geometry['edges'] == edge)
        count = int(usable[ids].sum())
        info[f'edge_{edge}_count'] = count
        info[f'edge_{edge}_total'] = int(len(ids))
        if count < REFERENCE_MIN_POINTS_PER_EDGE:
            ok = False

    if 'sample_types' in geometry:
        inset_ids = np.flatnonzero(geometry['sample_types'] == 1)
        info['inset_count'] = int(usable[inset_ids].sum())
        info['inset_total'] = int(len(inset_ids))
    else:
        info['inset_count'] = 0
        info['inset_total'] = 0

    return ok, info

def current_fused_state(frames):
    """Return fused depth/count/spread for current accumulated frames."""
    return fuse(frames)


def reference_valid_ratio(depth, mask, geometry):
    """参考点总有效率仅用于显示，硬条件为四条边各>=7个有效深度点。"""
    _, values, usable = collect_edge_depths(depth, mask, geometry)
    ratio = float(usable.sum()/max(1, len(values)))
    distributed_ok, distribution = reference_distribution_ok(values, usable, geometry)
    return ratio, values, usable, distributed_ok, distribution

def extract_anomaly_regions(base_roi, roi_mask, corners):
    """四角Lab母材颜色基准 + 多实例色差seed提取。

    与原稳定版相同：
    - 仍用ROI四个顶点附近颜色建立正常母材Lab基准；
    - 仍使用自适应Lab色差阈值与轻度形态学去噪。

    与原版不同：
    - 不再只保留最大闭合色差区域；
    - 所有满足独立绝对面积条件的色差区域都保留，
      以适应1~9个甚至更多分散圆/椭圆缺陷。
    """
    h, w = roi_mask.shape
    corners = np.asarray(corners, np.float32)
    center = corners.mean(axis=0)

    smooth = cv2.bilateralFilter(
        base_roi, IMAGE_BLUR_DIAMETER, 35, 35
    )
    lab = cv2.cvtColor(
        smooth, cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    # ROI内部略微收缩，避免ROI边框本身成为异常。
    inner_u8 = roi_mask.astype(np.uint8)
    erode_r = max(2, IMAGE_INNER_MARGIN // 2)
    kernel_inner = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2*erode_r+1, 2*erode_r+1)
    )
    inner = cv2.erode(
        inner_u8, kernel_inner
    ).astype(bool)

    # ---------- A. 四角母材颜色 ----------
    corner_colors = []
    corner_sample_points = []
    r = CORNER_COLOR_PATCH_RADIUS

    for p in corners:
        direction = center - p
        norm = float(np.linalg.norm(direction))
        q = p.copy()

        if norm > 1e-6:
            q = q + direction / norm * CORNER_COLOR_INSET_PX

        x, y = int(round(q[0])), int(round(q[1]))
        x0, x1 = max(0, x-r), min(w, x+r+1)
        y0, y1 = max(0, y-r), min(h, y+r+1)

        patch = lab[y0:y1, x0:x1]
        patch_mask = roi_mask[y0:y1, x0:x1]
        pixels = patch[patch_mask]

        if len(pixels) >= 5:
            c = np.median(pixels, axis=0)
            corner_colors.append(c)
            corner_sample_points.append((x, y))

    if len(corner_colors) < 3:
        pixels = lab[roi_mask]
        reference_lab = np.median(pixels, axis=0)
        corner_spread = 0.0
    else:
        corner_colors = np.asarray(
            corner_colors, np.float32
        )
        reference_lab = np.median(
            corner_colors, axis=0
        )
        corner_dist = np.linalg.norm(
            corner_colors - reference_lab[None, :],
            axis=1
        )
        corner_spread = float(
            1.4826 * np.median(
                np.abs(
                    corner_dist - np.median(corner_dist)
                )
            )
        )

    # ---------- B. Lab色差 ----------
    color_diff = np.linalg.norm(
        lab - reference_lab[None, None, :],
        axis=2
    ).astype(np.float32)
    color_diff[~inner] = 0.0

    valid_vals = color_diff[inner]

    if valid_vals.size:
        p70 = float(np.percentile(valid_vals, 70))
        p85 = float(np.percentile(valid_vals, 85))

        adaptive_thr = max(
            COLOR_DIFF_MIN,
            p70,
            float(
                np.median(valid_vals)
                + COLOR_DIFF_CORNER_SCALE
                * max(1.0, corner_spread)
            )
        )

        adaptive_thr = min(
            adaptive_thr,
            max(COLOR_DIFF_MIN, p85)
        )
    else:
        adaptive_thr = COLOR_DIFF_MIN

    binary = (
        (color_diff >= adaptive_thr) & inner
    ).astype(np.uint8)

    # ---------- C. 比原版更保守的形态学 ----------
    # 原版close两次对单缺陷稳定，但多圆时容易相邻粘连。
    # 多实例版改为一次close + 一次轻open。
    kr = max(2, COLOR_CLOSE_RADIUS - 2)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2*kr+1, 2*kr+1)
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1
    )
    binary[~inner] = 0

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    roi_area = max(1, int(roi_mask.sum()))
    min_area = max(
        float(MULTI_SEED_MIN_AREA_PX),
        0.0008 * roi_area
    )
    max_area = (
        MULTI_SEED_MAX_AREA_RATIO * roi_area
    )

    valid_contours = []

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area < min_area or area > max_area:
            continue
        if cv2.arcLength(contour, True) < 18:
            continue

        valid_contours.append(
            (area, contour)
        )

    valid_contours.sort(
        key=lambda item: item[0],
        reverse=True
    )

    # 关键变化：全部独立合格seed都保留，不再只取最大一个。
    kept = [
        contour for area, contour
        in valid_contours
    ]

    candidate_mask_u8 = np.zeros_like(
        binary, np.uint8
    )

    if kept:
        cv2.drawContours(
            candidate_mask_u8,
            kept,
            -1,
            1,
            thickness=-1
        )

    candidate_mask = (
        candidate_mask_u8.astype(bool)
        & roi_mask
    )

    # ---------- D. 明暗结构只作描述 ----------
    gray = cv2.cvtColor(
        smooth, cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    background = cv2.GaussianBlur(
        gray, (0, 0), SHADING_BG_SIGMA
    )
    shading = gray - background

    shade_vals = shading[inner]

    if shade_vals.size:
        shade_center = float(
            np.median(shade_vals)
        )
        shade_mad = float(
            1.4826
            * np.median(
                np.abs(
                    shade_vals - shade_center
                )
            )
        )
    else:
        shade_center, shade_mad = 0.0, 0.0

    shade_thr = max(
        SHADING_MIN_ABS,
        SHADING_SIGMA_SCALE * shade_mad
    )

    bright_mask = (
        candidate_mask
        & (shading >= shade_center + shade_thr)
    )
    dark_mask = (
        candidate_mask
        & (shading <= shade_center - shade_thr)
    )

    # ---------- E. 外观得分，仅保存/观察 ----------
    if valid_vals.size:
        lo = float(
            np.percentile(valid_vals, 50)
        )
        hi = float(
            np.percentile(valid_vals, 97)
        )
        color_score = np.clip(
            (color_diff-lo)
            / max(hi-lo, 1e-6),
            0,
            1
        )
    else:
        color_score = np.zeros_like(
            color_diff, np.float32
        )

    abs_shade = np.abs(shading)
    sv = abs_shade[inner]

    if sv.size:
        slo, shi = np.percentile(
            sv, [50, 97]
        )
        shade_score = np.clip(
            (abs_shade-slo)
            / max(float(shi-slo), 1e-6),
            0,
            1
        )
    else:
        shade_score = np.zeros_like(
            color_diff, np.float32
        )

    appearance_score = (
        0.75*color_score
        + 0.25*shade_score
    ).astype(np.float32)

    appearance_score[~roi_mask] = 0

    appearance_gray = np.uint8(
        np.clip(
            color_diff
            / max(adaptive_thr, 1e-6)
            * 80,
            0,
            255
        )
    )

    appearance_info = dict(
        reference_lab=[
            float(v) for v in reference_lab
        ],
        corner_color_spread=float(
            corner_spread
        ),
        color_difference_threshold=float(
            adaptive_thr
        ),
        close_radius_px=int(kr),
        candidate_regions=len(kept),
        bright_pixels=int(
            bright_mask.sum()
        ),
        dark_pixels=int(
            dark_mask.sum()
        ),
        multi_seed_mode=True
    )

    return (
        candidate_mask,
        kept,
        appearance_score,
        appearance_gray,
        color_diff.astype(np.float32),
        bright_mask,
        dark_mask,
        shading.astype(np.float32),
        appearance_info
    )

def classify_anomaly_regions(height, valid, candidate_contours, roi_mask, area_mm2,
                             color_difference=None, bright_mask=None, dark_mask=None):
    """每个闭合异常区域只允许一种类别：整区凸起或整区凹陷。"""
    bulge_measured = np.zeros(height.shape, bool)
    dent_measured = np.zeros(height.shape, bool)
    bulge_inferred = np.zeros(height.shape, bool)
    dent_inferred = np.zeros(height.shape, bool)
    candidate_union = np.zeros(height.shape, bool)
    region_stats = []

    for idx, contour in enumerate(candidate_contours, 1):
        region_u8 = np.zeros(height.shape, np.uint8)
        cv2.drawContours(region_u8, [contour], -1, 1, thickness=-1)
        region = region_u8.astype(bool) & roi_mask
        if not np.any(region):
            continue

        candidate_union |= region
        inside_valid = region & valid & np.isfinite(height)
        measured_count = int(inside_valid.sum())
        coverage = float(measured_count / max(1, region.sum()))

        mean_color_difference = (
            float(np.mean(color_difference[region]))
            if isinstance(color_difference, np.ndarray) and np.any(region) else None
        )
        bright_ratio = (
            float(np.mean(bright_mask[region]))
            if isinstance(bright_mask, np.ndarray) and np.any(region) else None
        )
        dark_ratio = (
            float(np.mean(dark_mask[region]))
            if isinstance(dark_mask, np.ndarray) and np.any(region) else None
        )

        if measured_count < REGION_MIN_MEASURED_PIXELS:
            region_stats.append(dict(
                id=idx, area_px=int(region.sum()),
                depth_coverage_ratio=coverage,
                classification='insufficient_depth',
                measured_pixels=measured_count,
                positive_ratio=None, negative_ratio=None,
                median_height_mm=None,
                mean_color_difference=mean_color_difference,
                bright_ratio=bright_ratio, dark_ratio=dark_ratio,
                area_cm2=float(area_mm2[region].sum()/100.0) if area_mm2 is not None else 0.0
            ))
            continue

        vals = height[inside_valid].astype(np.float64)
        positive_ratio = float(np.mean(vals > 0))
        negative_ratio = float(np.mean(vals < 0))
        median_h = float(np.median(vals))

        if positive_ratio >= REGION_SIGN_MAJORITY:
            label = 'bulge'
        elif negative_ratio >= REGION_SIGN_MAJORITY:
            label = 'dent'
        else:
            label = 'bulge' if median_h >= 0 else 'dent'

        if label == 'bulge':
            measured = inside_valid & (height >= 0)
            if measured.sum() < max(1, int(0.25 * measured_count)):
                measured = inside_valid
            bulge_measured |= measured
            bulge_inferred |= region & ~valid
        else:
            measured = inside_valid & (height <= 0)
            if measured.sum() < max(1, int(0.25 * measured_count)):
                measured = inside_valid
            dent_measured |= measured
            dent_inferred |= region & ~valid

        region_stats.append(dict(
            id=idx, area_px=int(region.sum()),
            depth_coverage_ratio=coverage,
            classification=label,
            measured_pixels=measured_count,
            median_height_mm=median_h,
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            mean_color_difference=mean_color_difference,
            bright_ratio=bright_ratio, dark_ratio=dark_ratio,
            area_cm2=float(area_mm2[region].sum()/100.0) if area_mm2 is not None else 0.0
        ))

    return (candidate_union, bulge_measured, dent_measured,
            bulge_inferred, dent_inferred, region_stats)

def component_stats(mask, area):
    number, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    # 不对稀疏mask做开闭运算，也不把相隔缺测点的区域补成一个缺陷。
    sums = np.bincount(labels.ravel(), weights=area.ravel(), minlength=number)/100.
    return dict(area_cm2=float(sums[1:].sum()), regions=number-1,
                largest_cm2=float(sums[1:].max()) if number > 1 else 0.)


def analyze(frames, base, roi, mask, geometry, k):
    depth, count, spread = fuse(frames)
    valid = np.isfinite(depth) & mask
    x, y, w, h = roi
    base_roi = base[y:y+h, x:x+w].copy()

    # 1) 根据四角平整母材颜色提取多个独立异常seed，并进行圆/椭圆多实例补全。
    appearance = build_auto_appearance(base, roi, mask, geometry)

    (candidate_mask, candidate_contours, appearance_score, appearance_gray,
     color_difference, bright_mask, dark_mask, shading_map, appearance_info) = appearance

    output = dict(
        depth=depth, observations=count, temporal_spread_mm=spread, valid=valid,
        height=np.full(depth.shape, np.nan, np.float32),
        candidate_mask=candidate_mask,
        bulge=np.zeros(depth.shape, bool), dent=np.zeros(depth.shape, bool),
        appearance_score=appearance_score, appearance_gray=appearance_gray,
        color_difference=color_difference, bright_mask=bright_mask, dark_mask=dark_mask,
        shading_map=shading_map, appearance_info=appearance_info,
        reference_samples=np.empty((0,2), np.float32),
        reference_inliers=np.empty((0,), bool),
        reference_depths=np.empty((0,), np.float64),
        plane_ok=False, reason='', region_stats=[],
        stats=dict(
            valid_pixel_ratio=float(valid.sum()/max(1, mask.sum())),
            captured_frames=len(frames),
            image_candidate_regions=len(candidate_contours)
        )
    )

    # 2) 将所有检测到的异常实例及其邻域从参考面中彻底排除。
    #    剩余ROI正常母材大面积采样，6x6网格取中位深度。
    uv_ref, z_ref, normal_reference_mask, excluded_reference_mask = collect_dense_reference_points(
        depth, mask, candidate_mask, geometry
    )
    output['normal_reference_mask'] = normal_reference_mask
    output['excluded_reference_mask'] = excluded_reference_mask

    if len(z_ref) < REFERENCE_DENSE_MIN_POINTS:
        raise RuntimeError(
            f"REFERENCE_FIT_RETRY: 排除全部异常实例后只有{len(z_ref)}个正常母材参考点。"
        )

    try:
        coef, active, rms, uv_used, z_used = fit_dense_reference_plane(uv_ref, z_ref)
    except ValueError as exc:
        raise RuntimeError(f"REFERENCE_FIT_RETRY: {exc}")

    # 为可视化/保存，参考采样点存归一化坐标对应值。
    output['reference_depths'] = z_used
    output['reference_inliers'] = active
    output['reference_samples_uv'] = uv_used

    # 3) 生成整个ROI参考平面，并计算形变量。
    z_plane = reference_depth_map(depth.shape, geometry['matrix'], coef)
    valid &= np.isfinite(z_plane) & (z_plane > 0)
    height = z_plane - depth
    height[~valid] = np.nan

    area = reference_areas(z_plane, mask, roi, k)
    measured_usable = valid & (area > 0)

    # 4) 亮/暗只作为外观信息，不参与“凸/凹方向”判断。
    #    凸凹完全由拟合平面与候选区域深度差决定。
    (candidate_union, bulge_measured, dent_measured,
     bulge_inferred, dent_inferred, region_stats) = classify_anomaly_regions(
        height, valid, candidate_contours, mask, area,
        color_difference=color_difference,
        bright_mask=bright_mask,
        dark_mask=dark_mask
    )

    total = float(area[mask].sum())
    measured = float(area[measured_usable].sum())
    candidate_depth_valid = candidate_union & valid

    output.update(
        height=height,
        candidate_mask=candidate_union,
        bulge=bulge_measured,
        dent=dent_measured,
        bulge_measured=bulge_measured,
        dent_measured=dent_measured,
        bulge_inferred=bulge_inferred,
        dent_inferred=dent_inferred,
        area_mm2=area,
        valid=valid,
        reference_plane_coefficients=coef,
        reference_matrix=geometry['matrix'],
        reference_depth_map=z_plane,
        plane_ok=True,
        region_stats=region_stats
    )

    output['stats'].update(
        bulge=dict(
            regions=sum(1 for r in region_stats if r.get('classification') == 'bulge'),
            area_cm2=float(area[(bulge_measured | bulge_inferred)].sum()/100.0)
            if np.any(bulge_measured | bulge_inferred) else 0.0
        ),
        dent=dict(
            regions=sum(1 for r in region_stats if r.get('classification') == 'dent'),
            area_cm2=float(area[(dent_measured | dent_inferred)].sum()/100.0)
            if np.any(dent_measured | dent_inferred) else 0.0
        ),
        plane_rms_mm=rms,
        dense_reference_points_total=int(len(z_used)),
        dense_reference_points_used=int(active.sum()),
        dense_reference_inlier_ratio=float(active.sum()/max(1, len(active))),
        excluded_candidate_ratio=float(excluded_reference_mask.sum()/max(1, mask.sum())),
        measured_area_cm2=measured/100.0,
        area_coverage_ratio=measured/total if total > 0 else 0.0,
        threshold_mm=DEFECT_THRESHOLD_MM,
        candidate_area_cm2=float(area[candidate_union].sum()/100.0) if np.any(candidate_union) else 0.0,
        candidate_depth_coverage_ratio=float(
            candidate_depth_valid.sum()/max(1, candidate_union.sum())
        ) if np.any(candidate_union) else 0.0,
        median_temporal_spread_mm=float(np.nanmedian(spread[valid])) if np.any(valid) else None
    )
    return output

def defect_map(result, color_override=None):
    image = np.zeros((*result['depth'].shape, 3), np.uint8)
    if result['plane_ok']:
        bulge_m = result.get('bulge_measured', result.get('bulge'))
        dent_m = result.get('dent_measured', result.get('dent'))
        bulge_i = result.get('bulge_inferred', np.zeros(result['depth'].shape, bool))
        dent_i = result.get('dent_inferred', np.zeros(result['depth'].shape, bool))

        if color_override == 'blue':
            light_color = (255, 170, 120)
            deep_color = (255, 0, 0)
            image[bulge_i | dent_i] = light_color
            image[bulge_m | dent_m] = deep_color
        elif color_override == 'red':
            light_color = (120, 150, 255)
            deep_color = (0, 0, 255)
            image[bulge_i | dent_i] = light_color
            image[bulge_m | dent_m] = deep_color
        else:
            image[bulge_i] = (120, 150, 255)
            image[dent_i] = (255, 170, 120)
            image[bulge_m] = (0, 0, 255)
            image[dent_m] = (255, 0, 0)
    return image

def render(base, result, roi, stereo, click=None, color_override=None):
    view = base.copy()
    x, y, w, h = roi
    local = view[y:y+h, x:x+w]
    if result['plane_ok']:
        bulge_i = result.get('bulge_inferred', np.zeros(result['depth'].shape, bool))
        dent_i = result.get('dent_inferred', np.zeros(result['depth'].shape, bool))
        bulge_m = result.get('bulge_measured', result.get('bulge'))
        dent_m = result.get('dent_measured', result.get('dent'))

        if color_override == 'blue':
            light_regions = [(bulge_i | dent_i, (255,150,110))]
            measured_regions = [(bulge_m | dent_m, (255,0,0))]
        elif color_override == 'red':
            light_regions = [(bulge_i | dent_i, (110,135,255))]
            measured_regions = [(bulge_m | dent_m, (0,0,255))]
        else:
            light_regions = [(bulge_i, (110,135,255)), (dent_i, (255,150,110))]
            measured_regions = [(bulge_m, (0,0,255)), (dent_m, (255,0,0))]

        for region, color in light_regions:
            if np.any(region):
                local[region] = ((1.0-REGION_FILL_LIGHT_ALPHA)*local[region]
                                 + REGION_FILL_LIGHT_ALPHA*np.array(color)).astype(np.uint8)

        for region, color in measured_regions:
            if np.any(region):
                local[region] = ((1.0-REGION_MEASURED_ALPHA)*local[region]
                                 + REGION_MEASURED_ALPHA*np.array(color)).astype(np.uint8)

        # 候选边界用白色细线描出，便于观察“图像先验”与深度分类是否一致。
        n, labels = cv2.connectedComponents(result['candidate_mask'].astype(np.uint8))
        for lab in range(1, n):
            contours, _ = cv2.findContours((labels==lab).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(local, contours, -1, (255,255,255), 1)

        stats = result['stats']
        lines = [
            f"Detected bulges {stats['bulge']['regions']} | Detected dents {stats['dent']['regions']}",
            f"Plane RMS {stats['plane_rms_mm']:.2f} mm | Threshold {DEFECT_THRESHOLD_MM:.2f} mm | ROI depth {stats['valid_pixel_ratio']:.0%}",
            f"Candidate depth coverage {stats['candidate_depth_coverage_ratio']:.0%}"
        ]
        if stats['plane_rms_mm'] > DEFECT_THRESHOLD_MM:
            lines.append('Reference noise exceeds defect threshold: depth classification may be unreliable')
    else:
        good = result['valid']
        local[good] = (.65*local[good]+.35*np.array([180,180,180])).astype(np.uint8)
        lines = ['REFERENCE FAILED: not enough reliable depth on the four edge lines',
                 'R: reselect edges on sound surface. Gray=depth only.']

    for i, (u, v) in enumerate(result['reference_samples']):
        color = (0,255,0) if result['reference_inliers'][i] else (0,165,255)
        if np.isfinite(result['reference_depths'][i]):
            cv2.circle(local, (int(round(u)), int(round(v))), 2, color, -1)

    lines.append('DARK red/blue = measured depth | LIGHT red/blue = region-consistent fill')
    lines.append('R reselect | W save | Q quit')
    if click is not None:
        u, v = click
        a, b = u-x, v-y
        if 0 <= a < w and 0 <= b < h and result['valid'][b, a]:
            z, count = result['depth'][b, a], result['observations'][b, a]

            if result['plane_ok']:
                raw_h = float(result['height'][b, a])

                if color_override == 'red':
                    display_h = abs(raw_h)
                elif color_override == 'blue':
                    display_h = -abs(raw_h)
                else:
                    display_h = raw_h

                hs = f"{display_h:+.2f} mm"
            else:
                hs = 'unavailable'

            in_candidate = bool(result['candidate_mask'][b, a])
            lines.append(f"Z {z:.2f} mm | Height {hs} | Samples {count}/{result['stats']['captured_frames']} | Candidate {in_candidate}")
        else:
            lines.append('This point has no reliable measured depth; it is not zero depth')
        cv2.drawMarker(view, click, (0,255,255), cv2.MARKER_CROSS, 18, 2)
    band_width = apply_disparity_black_band(view, stereo)
    info_panel_on_disparity_band(view, lines, band_width)
    return view


def save(result, base, view, roi, quad, stereo):
    folder = SAVE_ROOT/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True, exist_ok=False)
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(folder/'measurement.npz', **arrays, roi=roi, quad=quad,
                        K_rect=stereo.k, Q=stereo.q, K_L=K1, D_L=D1, K_R=K2, D_R=D2, R=R, T=T)
    summary = dict(result['stats'], plane_ok=result['plane_ok'], reason=result['reason'],
                   region_stats=result.get('region_stats', []),
                   appearance_info=result.get('appearance_info', {}),
                   num_disparities=NUM_DISPARITIES, block_size=BLOCK_SIZE,
                   min_observations=MIN_OBSERVATIONS, target_roi_coverage=TARGET_ROI_COVERAGE, reference_min_points_per_edge=REFERENCE_MIN_POINTS_PER_EDGE, reference_plane_min_total_inliers=REFERENCE_PLANE_MIN_TOTAL_INLIERS, reference_plane_min_edge_inliers=REFERENCE_PLANE_MIN_EDGE_INLIERS,
                   baseline_mm=float(np.linalg.norm(T)), plane_tolerance_mm=REFERENCE_FIT_LIMIT_MM,
                   samples_per_edge=SAMPLES_PER_EDGE, reference_sample_definition='largest color-anomaly region excluded; remaining ROI sampled by 6x6 grid median depth', reference_patch_radius=REFERENCE_PATCH_RADIUS,
                   defect_threshold_mm=DEFECT_THRESHOLD_MM,
                   depth_region_min_coverage=DEPTH_REGION_MIN_COVERAGE,
                   depth_sign_ratio=DEPTH_SIGN_RATIO,
                   height_definition='reference Z minus measured Z',
                   fusion_definition='Each closed appearance region is assigned one unique class by the majority sign of reliable stereo depth; missing-depth pixels are filled with a lighter color of the same class')
    (folder/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    # 保存图像层中间结果，便于观察“顶点母材色差 -> 闭合区域 -> 亮暗结构”。
    score = result.get('appearance_score')
    score_img = np.uint8(np.clip(score*255, 0, 255)) if isinstance(score, np.ndarray) else np.zeros(result['depth'].shape, np.uint8)

    color_diff = result.get('color_difference')
    if isinstance(color_diff, np.ndarray):
        finite = np.isfinite(color_diff)
        cd_img = np.zeros(color_diff.shape, np.uint8)
        if finite.any():
            hi = max(1.0, float(np.percentile(color_diff[finite], 98)))
            cd_img = np.uint8(np.clip(color_diff/hi*255, 0, 255))
    else:
        cd_img = np.zeros(result['depth'].shape, np.uint8)

    bright_img = (np.asarray(result.get('bright_mask'), np.uint8)*255
                  if isinstance(result.get('bright_mask'), np.ndarray)
                  else np.zeros(result['depth'].shape, np.uint8))
    dark_img = (np.asarray(result.get('dark_mask'), np.uint8)*255
                if isinstance(result.get('dark_mask'), np.ndarray)
                else np.zeros(result['depth'].shape, np.uint8))

    images = [('result.png', view), ('rectified_left.png', base), ('defect_map.png', defect_map(result)),
              ('appearance_score.png', score_img), ('color_difference.png', cd_img),
              ('bright_mask.png', bright_img), ('dark_mask.png', dark_img)]
    for name, image in images:
        ok, data = cv2.imencode('.png', image)
        if not ok:
            raise IOError('图像编码失败')
        data.tofile(str(folder/name))
    print('[SAVED]', folder)
    return folder


def main(stereo=None, cap=None, left_window_only=False, stop_event=None):
    if stereo is None:
        stereo = Stereo()
    if cap is None:
        cap = open_camera()
    state = dict(mode='LIVE', points=[], quad=None, roi=None, mask=None, band=None,
                 frames=[], attempted_frames=0, observation_counts=None, result=None, base=None,
                 click=None, notice='', color_override=None)
    swap, preview = False, False

    def reset():
        state.update(mode='LIVE', points=[], quad=None, frames=[], attempted_frames=0,
                     observation_counts=None, result=None, click=None, notice='',
                     color_override=None)
        try:
            cv2.destroyWindow(DEFECT_WINDOW)
        except cv2.error:
            pass

    def mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state['mode'] == 'RESULT':
            state['click'] = (x, y)
        elif state['mode'] == 'LIVE':
            state['points'].append((x, y))
            if len(state['points']) == 4:
                try:
                    quad, roi, mask, band = selection(state['points'])
                    state.update(quad=quad, roi=roi, mask=mask, band=band, mode='CAPTURE',
                                 frames=[], attempted_frames=0,
                                 observation_counts=np.zeros(mask.shape, np.uint16))
                except ValueError as exc:
                    print('[SELECT]', exc)
                    reset(); state['notice'] = 'Invalid selection. Click four different corners again.'

    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, mouse)
        print('选择四角后持续采集深度；无固定帧数上限。ROI可靠深度>=70%，且四条边分别至少5个有效深度点后进入分析。')
        print('四条边上的采样点作为基准，边界应位于完好母材。R重选 S换序 V预览 W保存 Q退出。')
        print('ROI深度达到70%后自动提取外观异常区域并拟合参考平面。NUM_DISPARITIES对应的左侧视差不可用区域涂黑并用于显示状态信息。')
        failures = 0
        while stop_event is None or not stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if state['mode'] == 'CAPTURE':
                    reset()
                blank = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
                panel(blank, ['Camera read failed; capture cancelled. Check connection.'])
                cv2.imshow(WINDOW, blank)
                if failures >= 30:
                    raise RuntimeError('相机连续读取失败。')
                if cv2.waitKey(30) & 0xff in (27, ord('q'), ord('Q')):
                    break
                continue
            failures = 0
            left, right = stereo.rectify(frame, swap)
            view = left.copy()
            if state['mode'] == 'LIVE':
                if preview:
                    good = np.isfinite(stereo.depth(left, right))
                    view[good] = (.65*view[good]+.35*np.array([230, 216, 173])).astype(np.uint8)
                cv2.line(view, (stereo.boundary, 100), (stereo.boundary, SIZE[1]-1), (0, 165, 255), 1)
                band_width = apply_disparity_black_band(view, stereo)
                info_panel_on_disparity_band(view, [
                    f"Click four corners {len(state['points'])}/4",
                    "R reset | S swap",
                    "V preview | Q quit",
                    state['notice'] or 'Keep camera and plate still'
                ], band_width)
                for p in state['points']:
                    cv2.circle(view, p, 5, (0, 255, 255), -1)
            elif state['mode'] == 'CAPTURE':
                x, y, w, h = state['roi']
                z = stereo.depth(left, right)[y:y+h, x:x+w].copy()
                z[~state['mask']] = np.nan
                state['frames'].append(z)
                state['base'] = left.copy()

                # 所有帧持续累积融合。
                # 只有同时满足：
                #   1) ROI内可靠有效深度覆盖率 >= 70%
                #   2) 4n参考采样点有效率 >= 70%
                # 才允许停止采集并进入参考平面拟合与凸凹可视化。
                fused_depth, fused_count, fused_spread = current_fused_state(state['frames'])

                roi_reliable = np.isfinite(fused_depth) & state['mask']
                roi_ratio = float(roi_reliable.sum()/max(1, state['mask'].sum()))

                ref_ratio, ref_values, ref_usable, ref_dist_ok, ref_dist = reference_valid_ratio(
                    fused_depth, state['mask'], state['band']
                )

                local = view[y:y+h, x:x+w]
                for i, (u, v) in enumerate(state['band']['samples']):
                    color = (0,255,0) if ref_usable[i] else (0,165,255)
                    cv2.circle(local, (int(round(u)), int(round(v))), 2, color, -1)

                current_ratio = (
                    float(np.isfinite(z)[state['mask']].mean())
                    if state['mask'].any() else 0.0
                )

                band_width = apply_disparity_black_band(view, stereo)
                info_panel_on_disparity_band(view, [
                    f"Frames {len(state['frames'])}",
                    f"Current depth {current_ratio:.0%}",
                    f"Reliable ROI {roi_ratio:.0%}/70%",
                    state['notice'] or "Keep camera and plate still"
                ], band_width)

                if roi_ratio >= TARGET_ROI_COVERAGE:
                    band_width = apply_disparity_black_band(view, stereo)
                    info_panel_on_disparity_band(view, [
                        'ROI depth >=70%',
                        'Detecting anomaly...',
                        'Fitting plane...'
                    ], band_width)
                    cv2.imshow(WINDOW, view)
                    cv2.waitKey(1)

                    try:
                        state['result'] = analyze(
                            state['frames'], state['base'],
                            state['roi'], state['mask'],
                            state['band'], stereo.k
                        )
                    except RuntimeError as exc:
                        print('[CONTINUE CAPTURE]', exc)
                        state['notice'] = str(exc).replace('REFERENCE_FIT_RETRY: ', '')
                    else:
                        state['mode'] = 'RESULT'
                        state['frames'] = []
                        view = render(state['base'], state['result'], state['roi'], stereo, color_override=state['color_override'])
                        print('[RESULT]', json.dumps(state['result']['stats'], ensure_ascii=False), flush=True)
            else:
                view = render(state['base'], state['result'], state['roi'], stereo, state['click'], state['color_override'])
            if state['mode']=='RESULT' and not left_window_only:
                cv2.namedWindow(DEFECT_WINDOW,cv2.WINDOW_NORMAL)
                cv2.imshow(DEFECT_WINDOW,defect_map(state['result'], state['color_override']))
            if state['quad'] is not None:
                cv2.polylines(view, [state['quad'].astype(np.int32)], True, (0, 255, 255), 1)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xff
            if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 15 and state['mode'] == 'RESULT':
                # 隐藏三态循环：
                # 正常(None) -> 全红(red) -> 全蓝(blue) -> 正常(None)
                if state['color_override'] is None:
                    state['color_override'] = 'red'
                elif state['color_override'] == 'red':
                    state['color_override'] = 'blue'
                else:
                    state['color_override'] = None
            elif key in (ord('r'), ord('R')):
                reset()
            elif key in (ord('s'), ord('S')):
                swap = not swap; reset()
            elif key in (ord('v'), ord('V')):
                preview = not preview


            elif key in (ord('w'), ord('W')) and state['mode'] == 'RESULT':
                try:
                    save(state['result'], state['base'], view, state['roi'], state['quad'], stereo)
                except (OSError, ValueError, cv2.error) as exc:
                    print('[SAVE FAILED]', exc)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
