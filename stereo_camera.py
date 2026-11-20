"""导入 MATLAB 双目标定，识别左右拼接视频流并运行左目选点测量窗口。"""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
NO_CAMERA = "未检测到相机，请重新连接后再试"
NOT_STEREO = "相机可能不是双目相机，请重新连接双目相机后再试"
FIELDS = {"K1": "intrinsicMatrix1", "D1": "distortionCoefficients1",
          "K2": "intrinsicMatrix2", "D2": "distortionCoefficients2",
          "R": "rotationOfCamera2", "T": "translationOfCamera2", "imageSize": "imageSize"}


def numeric_parameters(fields):
    """数值 MAT 遵循 stereoParametersToOpenCV 的坐标和系数顺序。"""
    result = {}
    for key, alt in FIELDS.items():
        if key not in fields and alt not in fields:
            raise ValueError(f"缺少双目标定字段 {key}。")
        a = np.asarray(fields.get(key, fields.get(alt)), dtype=np.float64)
        if not np.isfinite(a).all():
            raise ValueError(f"{key} 含无效数值。")
        result[key] = a
    for key in ("K1", "K2"):
        k = result[key]
        if (k.shape != (3, 3) or min(k[0, 0], k[1, 1]) <= 0
                or not np.allclose(k[2], [0, 0, 1]) or not np.allclose([k[0,1], k[1,0]], 0)):
            raise ValueError(f"{key} 不是有效的 OpenCV 针孔相机内参矩阵。")
    for key in ("D1", "D2"):
        a = result[key].ravel()
        if a.size not in (4, 5, 8, 12, 14):
            raise ValueError(f"{key} 的畸变系数数量不正确。")
        result[key] = a
    rot, t = result["R"], result["T"].ravel()
    if (rot.shape != (3, 3) or not np.allclose(rot @ rot.T, np.eye(3), atol=1e-4)
            or not np.isclose(np.dot(rot[0], np.cross(rot[1], rot[2])), 1, atol=1e-4)):
        raise ValueError("双目旋转矩阵无效。")
    if t.size != 3 or np.linalg.norm(t) <= 0:
        raise ValueError("双目基线无效。")
    size = result.pop("imageSize").ravel()
    if size.size != 2 or not np.equal(size, np.floor(size)).all() or min(size) < 32:
        raise ValueError("imageSize 应为单目图像的 [高, 宽]。")
    units = str(np.asarray(fields.get("worldUnits", "")).squeeze()).lower().strip()
    scales = {"mm": 1, "millimeters": 1, "millimeter": 1, "cm": 10, "centimeters": 10,
              "centimeter": 10, "meters": 1000, "meter": 1000, "m": 1000,
              "inches": 25.4, "inch": 25.4, "in": 25.4}
    if units and units not in scales:
        raise ValueError(f"无法识别平移单位 {units}。请将 worldUnits 指定为 mm、cm、m 或 inches。")
    result["needs_units"] = not units
    result["T"] = t.reshape(3, 1) * scales.get(units, 1)
    result["size"] = (int(size[1]), int(size[0]))
    return result


def read_calibration(path):
    path = Path(path).resolve()
    if path.suffix.lower() != ".mat" or not path.is_file():
        raise ValueError("请选择 MATLAB 双目标定工具保存的 .mat 文件。")
    try:
        from scipy.io import loadmat
        fields = loadmat(str(path), squeeze_me=True)
    except (ImportError, NotImplementedError):  # 可直接用 MATLAB 读取 v7.3 文件或原生对象
        fields = {}
    if any(key in fields for key in ("K1", "intrinsicMatrix1")):
        return numeric_parameters(fields)
    if fields and not any(getattr(a, "dtype", None) is not None and
                          (a.dtype.names or a.dtype.kind == "O") for a in fields.values()):
        raise ValueError("文件中没有双目标定对象或双目数值参数。")
    matlab = shutil.which("matlab")
    if not matlab:
        raise ValueError("此 MAT 需要 MATLAB 读取。将 MATLAB 的 bin 目录加入 PATH，或在 MATLAB 中运行 "
                         "export_stereo_mat(stereoParams, 'stereo_numeric.mat') 后导入数值文件。")
    env = os.environ.copy()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "stereo.json"
        prefs = Path(tmp) / "preferences"
        prefs.mkdir()
        env.setdefault("MATLAB_PREFDIR", str(prefs))
        env.update(STEREO_INPUT_MAT=str(path), STEREO_OUTPUT_MAT=str(out))
        args = [matlab] + (["-wait"] if os.name == "nt" else [])
        args += ["-noFigureWindows", "-sd", str(ROOT), "-batch",
                 "export_stereo_mat(getenv('STEREO_INPUT_MAT'),getenv('STEREO_OUTPUT_MAT'))"]
        proc = subprocess.run(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=180, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if proc.returncode or not out.is_file():
            log = proc.stdout.decode(errors="replace").strip()
            raise ValueError("MATLAB 未能导出双目标定结果。请确认文件已完成双目标定，"
                             "且 MATLAB 和 Computer Vision Toolbox 可用。\n" + log[-800:])
        return numeric_parameters(json.loads(out.read_text(encoding="utf-8-sig")))


def detect_camera(cal_size, stop_event=None):
    import cv2
    width, height = cal_size
    seen_frame = False
    wide_sizes = set()
    backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF) if sys.platform == "win32" else (cv2.CAP_ANY,)
    modes = list(dict.fromkeys(((width*2, height), (2560, 720), (1280, 480),
                                (1280, 360), (3840, 1080))))
    for backend in backends:
        for index in range(10):
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("已取消相机连接。")
            cap = cv2.VideoCapture(index, backend)
            accepted = False
            try:
                if not cap.isOpened():
                    continue
                # 先读设备默认帧，以免一次不支持的分辨率协商使驱动停止输出。
                attempts = [None] + modes
                fallback = None
                for mode in attempts:
                    if mode:
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, mode[0])
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, mode[1])
                    for _ in range(3):
                        ok, frame = cap.read()
                        if ok and frame is not None and frame.size:
                            break
                        time.sleep(0.03)
                    else:
                        continue
                    h, w = frame.shape[:2]
                    seen_frame = True
                    if w % 2 or w/h < 2.35:
                        continue
                    wide_sizes.add((w, h))
                    if (w//2, h) == cal_size:
                        accepted = True
                        return cap, cal_size
                    if abs((w/2/h)/(width/height)-1) < 0.002:
                        fallback = (w, h)
                if fallback:
                    w, h = fallback
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.shape[:2] == (h, w):
                        accepted = True
                        return cap, (w//2, h)
            finally:
                if not accepted:
                    cap.release()
    if wide_sizes:
        sizes = "、".join(f"{w}×{h}" for w, h in sorted(wide_sizes))
        raise RuntimeError(f"检测到宽幅相机输出 {sizes}，与标定的单目比例 {width}×{height} 不匹配。"
                           "请选择对应视频模式的标定文件。")
    raise RuntimeError(NOT_STEREO if seen_frame else NO_CAMERA)


def run_camera(params, stop_event, connection):
    """在独立进程运行 OpenCV 窗口，Tk 主线程持续处理界面事件。"""
    cap = None
    try:
        import cv2
        spec = importlib.util.spec_from_file_location("stereo_engine", ROOT / "202610041842.py")
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)
        cap, size = detect_camera(params["size"], stop_event)
        sx, sy = size[0]/params["size"][0], size[1]/params["size"][1]
        resize = np.array([[sx,0,(sx-1)/2], [0,sy,(sy-1)/2], [0,0,1]])
        engine.K1, engine.K2 = resize @ params["K1"], resize @ params["K2"]
        engine.D1, engine.D2 = params["D1"], params["D2"]
        engine.R, engine.T, engine.SIZE = params["R"], params["T"], size
        engine.NUM_DISPARITIES = max(16, min((size[0]-engine.BLOCK_SIZE-1)//16*16,
                                            int(round(384*size[0]/1280/16))*16))
        engine.WINDOW = f"{np.linalg.norm(engine.T):.1f}mm - four-edge reference plane"
        stereo = engine.Stereo()
        if not stereo.region.any():
            raise ValueError("此标定和图像尺寸无法产生有效的水平视差区域。")
        if stereo.q[3,2] <= 0 or abs(engine.T[0,0]) <= abs(engine.T[1,0]):
            raise ValueError("当前测量需要从左到右标定的水平双目相机，请核对双目标定顺序。")
        connection.send(("ready", f"左目 {size[0]}×{size[1]}，点击四角开始测量。"))
        engine.main(stereo=stereo, cap=cap, left_window_only=True, stop_event=stop_event)
        connection.send(("done", "测量窗口已关闭。"))
    except Exception as exc:
        connection.send(("error", str(exc)))
    finally:
        if cap is not None:
            cap.release()
        connection.close()
