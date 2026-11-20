"""水下双目缺陷测量工具箱：复核旧版导出的结果，或运行内置的合成示例。

python stereo_review.py                      # 打开演示首页
python stereo_review.py --gui "测量结果目录"  # 直接打开指定记录
python stereo_review.py --demo
python stereo_review.py --demo --point 126 221
python stereo_review.py "测量结果目录" --point 860 360 --csv regions.csv

目录应包含旧版 W 键保存的 summary.json 和 measurement.npz。
坐标以校正后完整左图的左上角为原点。长度单位为 mm。
"""

import argparse
import csv
import io
import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from zipfile import BadZipFile

import numpy as np

ROOT = Path(__file__).resolve().parent


REPORT_HEADER = ("工况", "方向", "参考幅值(mm)", "区间下限(mm)", "区间上限(mm)",
                 "区间中点(mm)", "中点绝对误差(mm)")
POOL_RESULTS = (
    ("Ⅰ   单板", "凸起", 2.12, 1.25, 2.95, 2.10, 0.02),
    ("Ⅱ   双板间隙", "凹陷", 2.12, 2.15, 3.65, 2.90, 0.78),
    ("Ⅲ   单层区域", "凸起", 2.12, 1.30, 3.10, 2.20, 0.08),
    ("Ⅲ   重叠区域", "凸起", 4.24, 1.84, 5.28, 3.56, 0.68),
    ("Ⅳ   3 mm 凸起", "凸起", 3.00, 1.95, 3.93, 2.94, 0.06),
    ("Ⅳ   5 mm 凸起", "凸起", 5.00, 3.68, 6.07, 4.88, 0.13),
    ("Ⅴ   3 mm 凹陷", "凹陷", 3.00, 1.65, 4.60, 3.13, 0.13),
    ("Ⅴ   5 mm 凹陷", "凹陷", 5.00, 3.52, 6.71, 5.12, 0.12),
)


LABELS = {"bulge": "凸起", "dent": "凹陷", "insufficient_depth": "深度不足"}
CSV_FIELDS = {
    "id": "区域编号",
    "classification": "类别",
    "area_px": "候选区域像素数",
    "measured_pixels": "有效深度像素数",
    "depth_coverage_ratio": "深度覆盖率(0~1)",
    "median_height_mm": "有符号高度中位数(mm)",
    "area_cm2": "候选区域投影面积(cm²)",
    "source": "数据来源",
}


@dataclass
class Measurement:
    summary: dict
    roi: tuple
    quad: np.ndarray
    depth: np.ndarray
    height: np.ndarray
    valid: np.ndarray
    candidate: np.ndarray
    inferred: np.ndarray
    # ±2 为实测凸起/凹陷，±1 为相应方向的区域推测，0 为未着色。
    classes: Optional[np.ndarray] = None

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8-sig"))
        if not isinstance(summary, dict) or type(summary.get("plane_ok")) is not bool:
            raise ValueError("summary.json 缺少参考平面状态 plane_ok。")
        if not isinstance(summary.get("region_stats"), list):
            raise ValueError("summary.json 缺少缺陷区域列表 region_stats。")
        with np.load(directory / "measurement.npz", allow_pickle=False) as arrays:
            required = {"roi", "quad", "depth", "height", "valid", "candidate_mask"}
            missing = required.difference(arrays.files)
            if missing:
                raise ValueError("measurement.npz 缺少字段 " + ", ".join(sorted(missing)))
            roi_array = arrays["roi"]
            if roi_array.shape != (4,) or not np.issubdtype(roi_array.dtype, np.integer):
                raise ValueError("roi 应包含四个整数 x、y、宽、高。")
            x, y, w, h = map(int, roi_array)
            quad = np.asarray(arrays["quad"], dtype=float)
            if min(x, y) < 0 or min(w, h) <= 0 or quad.shape != (4, 2) or not np.isfinite(quad).all():
                raise ValueError("ROI 坐标或四角数据无效。")
            maps = {}
            for key in ("depth", "height", "valid", "candidate_mask", "bulge_inferred", "dent_inferred",
                        "bulge_measured", "dent_measured"):
                alias = {"bulge_measured": "bulge", "dent_measured": "dent"}.get(key)
                name = key if key in arrays else alias
                data = arrays[name] if name and name in arrays else np.zeros((h, w), dtype=bool)
                if data.shape != (h, w):
                    raise ValueError(f"{key} 尺寸与 ROI 的宽、高不一致。")
                maps[key] = data
        classes = np.zeros((h, w), np.int8)
        for name, code in (("bulge_inferred", 1), ("dent_inferred", -1),
                           ("bulge_measured", 2), ("dent_measured", -2)):
            classes[maps[name].astype(bool)] = code
        return cls(summary, (x, y, w, h), quad, maps["depth"], maps["height"],
                   maps["valid"].astype(bool), maps["candidate_mask"].astype(bool),
                   maps["bulge_inferred"].astype(bool) | maps["dent_inferred"].astype(bool), classes)

    def point(self, u, v):
        """读取完整校正左图上的一个像素。推测区不提供测量数值。"""
        x, y, w, h = self.roi
        if not (x <= u < x+w and y <= v < y+h):
            raise ValueError("查询坐标在所选 ROI 之外。")
        edges = np.roll(self.quad, -1, axis=0) - self.quad
        delta = np.array([u, v]) - self.quad
        cross = edges[:, 0] * delta[:, 1] - edges[:, 1] * delta[:, 0]
        if cross.min() < -1e-6 and cross.max() > 1e-6:
            raise ValueError("查询坐标在所选四边形之外。")
        idx = v-y, u-x
        result = {"u": u, "v": v, "候选缺陷区域": bool(self.candidate[idx]),
                  "数据来源": "缺测", "相机深度_mm": None, "有符号高度_mm": None}
        if self.inferred[idx]:
            result["数据来源"] = "区域推测"
        elif self.valid[idx] and np.isfinite(self.depth[idx]) and self.depth[idx] > 0:
            result.update(数据来源="双目测量", 相机深度_mm=float(self.depth[idx]))
            if self.summary["plane_ok"] and np.isfinite(self.height[idx]):
                result["有符号高度_mm"] = float(self.height[idx])
        return result

    def rows(self):
        for region in self.summary["region_stats"]:
            row = {key: region.get(key) for key in CSV_FIELDS}
            label = row["classification"]
            row["classification"] = LABELS.get(label, label)
            row["source"] = "内置合成示例" if self.summary.get("synthetic") else "导入记录"
            if not self.summary["plane_ok"]:
                row["median_height_mm"] = None
            yield row

    def export(self, filename):
        # 先渲染到内存，发现无效条目时不会留下半张表。
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(CSV_FIELDS.values())
        for row in self.rows():
            writer.writerow([row[k] for k in CSV_FIELDS])
        with Path(filename).open("x", encoding="utf-8-sig", newline="") as output:
            output.write(buffer.getvalue())


def demo():
    """合成一块平板、一处凸起和一处凹陷，演示复核操作。"""
    size = 64
    v, u = np.indices((size, size))
    height = np.zeros((size, size), np.float32)
    valid = np.ones_like(height, dtype=bool)
    candidate = np.zeros_like(valid)
    regions = []
    for i, (cx, cy, level, label) in enumerate(((26, 21, 4., "bulge"), (46, 41, -3., "dent")), 1):
        patch = (u-cx)**2 + (v-cy)**2 <= 7**2
        candidate |= patch
        height[patch] = level
        valid[cy, cx+2] = False
        count = int((valid & patch).sum())
        regions.append(dict(id=i, classification=label, area_px=int(patch.sum()),
                            measured_pixels=count, depth_coverage_ratio=count/int(patch.sum()),
                            median_height_mm=level, area_cm2=None))
    height[~valid] = np.nan
    summary = {"plane_ok": True, "synthetic": True, "region_stats": regions}
    quad = np.array([[100, 200], [163, 200], [163, 263], [100, 263]])
    return Measurement(summary, (100, 200, size, size), quad, 700.-height, height,
                       valid, candidate, candidate & ~valid)


def result_pixels(record):
    """按已保存的分类着色；没有分类掩膜的旧记录仍可查看有效深度。"""
    image = np.zeros((*record.depth.shape, 3), dtype=np.uint8)
    image[record.valid & np.isfinite(record.depth)] = (88, 108, 119)
    image[record.candidate & record.valid] = (134, 145, 149)
    image[record.inferred] = (188, 181, 166)
    if record.classes is not None and record.summary["plane_ok"]:
        for code, rgb in ((2, (205, 61, 49)), (-2, (44, 108, 188)),
                          (1, (246, 183, 166)), (-1, (163, 202, 243))):
            image[record.classes == code] = rgb
    return image


class ResultsWindow:
    def __init__(self, root, directory=None):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        root.title("水下双目缺陷测量")
        root.geometry("1120x690")
        root.minsize(800, 550)
        root.option_add("*Font", ("Microsoft YaHei UI", 10))
        self.record = None
        self.point_xy = None
        self.placement = None
        self.photo = None
        self.calibration = self.camera = self.camera_pipe = self.camera_stop = None
        self.importing = False
        self.import_result = queue.Queue()
        self.poster = None
        self.status = tk.StringVar(value="尚未导入双目标定文件")
        root.protocol("WM_DELETE_WINDOW", self.close)
        outer = ttk.Frame(root, padding=(20, 14))
        outer.pack(fill="both", expand=True)
        bar = ttk.Frame(outer)
        bar.pack(fill="x", pady=(0, 12))
        ttk.Label(bar, text="水下双目缺陷测量", font=("Microsoft YaHei UI", 17, "bold")).pack(side="left")
        ttk.Label(bar, textvariable=self.status).pack(side="right")
        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(0, 12))
        actions.columnconfigure(0, weight=1)
        camera_actions = ttk.Frame(actions)
        camera_actions.grid(row=0, column=0, sticky="w")
        record_actions = ttk.Frame(actions)
        record_actions.grid(row=0, column=1, sticky="e")

        def fit_toolbar(event=None):
            wrap = actions.winfo_width() < camera_actions.winfo_reqwidth() + record_actions.winfo_reqwidth() + 8
            if int(record_actions.grid_info()["row"]) != int(wrap):
                record_actions.grid_configure(row=int(wrap), column=0 if wrap else 1,
                                              columnspan=2 if wrap else 1, sticky="w" if wrap else "e",
                                              pady=(6 if wrap else 0, 0))

        actions.bind("<Configure>", fit_toolbar)
        record_actions.bind("<Configure>", fit_toolbar)
        button_style = "Toolbar.TButton"
        ttk.Style(root).configure(button_style, font=("Microsoft YaHei UI", 13), padding=(12, 6), width=0)
        self.import_button = ttk.Button(camera_actions, text="导入双目相机标定数据", style=button_style,
                                        command=self.import_calibration)
        self.import_button.pack(side="left")
        self.live_button = ttk.Button(camera_actions, text="实时相机测量", style=button_style,
                                      command=self.start_camera, state="disabled")
        self.live_button.pack(side="left", padx=8)
        self.export_button = ttk.Button(record_actions, style=button_style, command=self.export)
        self.export_button.pack(side="right")
        ttk.Button(record_actions, text="打开测量记录…", style=button_style,
                   command=self.open_directory).pack(side="right", padx=8)
        self.home_button = ttk.Button(record_actions, text="返回演示", style=button_style, command=self.show_report)
        ttk.Separator(outer).pack(fill="x")
        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True, pady=(12, 8))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(body, background="#ffffff", highlightthickness=1,
                                highlightbackground="#ccd8da", width=480, height=400)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda event: self.draw())
        self.canvas.bind("<Button-1>", self.click)
        side = ttk.Frame(body, width=248, padding=(16, 4, 0, 0))
        side.grid(row=0, column=1, sticky="ns")
        side.pack_propagate(False)
        self.side_title = tk.StringVar()
        self.detail = tk.StringVar()
        self.meta = tk.StringVar()
        self.footer = tk.StringVar()
        self.detail_title = ttk.Label(side, textvariable=self.side_title, font=("Microsoft YaHei UI", 13, "bold"),
                                      wraplength=225)
        self.detail_title.pack(anchor="w", pady=(3, 10))
        ttk.Label(side, textvariable=self.detail, justify="left", wraplength=225).pack(anchor="w")
        ttk.Separator(side).pack(fill="x", pady=15)
        self.color_key = ttk.Frame(side)
        for row, (label, measured, inferred) in enumerate((
                ("凸起（正值）", "#cd3d31", "#f6b7a6"),
                ("凹陷（负值）", "#2c6cbc", "#a3caf3"))):
            for column, color in enumerate((measured, inferred)):
                tk.Frame(self.color_key, width=14, height=14, background=color).grid(
                    row=row, column=column, padx=(0, 3), pady=3)
            ttk.Label(self.color_key, text=label).grid(row=row, column=2, padx=(5, 0), sticky="w")
        self.region_list = tk.Listbox(side, height=7, width=22, font=("Microsoft YaHei UI", 10),
                                      activestyle="none", relief="flat", exportselection=False)
        self.region_list.bind("<<ListboxSelect>>", self.select_region)
        self.legend = ttk.Label(side, textvariable=self.meta, wraplength=225, justify="left")
        self.legend.pack(anchor="w")
        ttk.Label(outer, textvariable=self.footer, wraplength=750).pack(anchor="w")
        self.show_report()
        if directory is not None:
            self.open_directory(directory)

    def show_report(self):
        self.record = self.point_xy = self.photo = self.placement = None
        self.region_list.pack_forget()
        self.home_button.pack_forget()
        self.export_button.configure(text="导出历史试验 CSV")
        self.side_title.set("测量说明")
        self.detail.set("在校正后的左图依次点选四个角，让四角落在缺陷周围的平整表面。测量完成后，点击缺陷即可查看相对参考面的高度（mm）。")
        self.color_key.pack(before=self.legend, anchor="w", pady=(0, 8))
        self.canvas.configure(cursor="hand2")
        self.meta.set("深色标出有实测深度的缺陷像素，浅色标出推测范围。")
        self.footer.set("")
        if self.poster is None:
            import tkinter as tk
            try:
                original = tk.PhotoImage(master=self.root, file=str(ROOT / "演示图片.png"))
                ppm = self.root.tk.call(str(original), "data", "-format", "ppm")
                self.poster = np.frombuffer(ppm, dtype=np.uint8, offset=len(ppm)-original.width()*original.height()*3
                                            ).reshape(original.height(), original.width(), 3)
            except (tk.TclError, ValueError) as exc:
                self.footer.set(f"无法显示演示图片.png　{exc}")
        self.draw()

    def play_demo(self):
        from tkinter import messagebox
        path = ROOT / "演示视频.mp4"
        try:
            if not path.is_file():
                raise FileNotFoundError("请将演示视频.mp4 放在主程序所在文件夹。")
            if os.name == "nt":
                os.startfile(str(path))
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("无法播放演示视频", str(exc), parent=self.root)

    def import_calibration(self):
        from tkinter import filedialog
        if self.importing or self.camera is not None:
            return
        name = filedialog.askopenfilename(parent=self.root, title="选择 MATLAB 双目标定文件",
                                          filetypes=[("MATLAB 双目标定数据", "*.mat")])
        if not name:
            return
        self.importing = True
        self.calibration = None
        self.live_button.state(["disabled"])
        self.import_button.state(["disabled"])
        self.status.set("正在读取标定文件…")

        def read():
            try:
                from stereo_camera import read_calibration
                self.import_result.put((read_calibration(name), None))
            except Exception as exc:
                self.import_result.put((None, str(exc)))

        threading.Thread(target=read, daemon=True).start()
        self.root.after(100, self.finish_import, Path(name).name)

    def finish_import(self, filename):
        from tkinter import messagebox, simpledialog
        try:
            params, error = self.import_result.get_nowait()
        except queue.Empty:
            self.root.after(100, self.finish_import, filename)
            return
        self.importing = False
        self.import_button.state(["!disabled"])
        self.status.set("尚未导入双目标定文件")
        if error:
            messagebox.showerror("标定文件导入失败", error, parent=self.root)
            return
        if params.pop("needs_units"):
            unit = simpledialog.askstring("平移向量单位", "此数值 MAT 没有标注长度单位。请输入标定时的单位（mm、cm、m 或 in）。",
                                          initialvalue="mm", parent=self.root)
            if unit is None:
                return
            factor = {"mm": 1, "cm": 10, "m": 1000, "in": 25.4}.get(unit.strip().lower())
            if not factor:
                messagebox.showerror("单位无效", "请重新导入并填写正确单位。", parent=self.root)
                return
            params["T"] *= factor
        self.calibration = params
        self.calibration_label = f"{filename}  ·  单目 {params['size'][0]}×{params['size'][1]}"
        self.status.set(self.calibration_label)
        self.live_button.state(["!disabled"])

    def start_camera(self):
        from tkinter import messagebox
        if self.calibration is None or self.camera is not None:
            return
        import multiprocessing as mp
        from stereo_camera import run_camera
        ctx = mp.get_context("spawn")
        receive, send = ctx.Pipe(duplex=False)
        self.camera_stop = ctx.Event()
        proc = ctx.Process(target=run_camera, args=(self.calibration, self.camera_stop, send), daemon=True)
        try:
            proc.start()
        except OSError as exc:
            receive.close()
            messagebox.showerror("相机启动失败", str(exc), parent=self.root)
            return
        finally:
            send.close()
        self.camera, self.camera_pipe = proc, receive
        self.camera_error = None
        self.live_button.state(["disabled"])
        self.import_button.state(["disabled"])
        self.status.set("正在检测双目相机…")
        self.root.after(100, self.poll_camera)

    def poll_camera(self):
        from tkinter import messagebox
        alive = self.camera.is_alive()
        while self.camera_pipe.poll():
            try:
                kind, text = self.camera_pipe.recv()
            except EOFError:
                break
            if kind == "error":
                self.camera_error = text
            else:
                self.status.set(text)
        if alive:
            self.root.after(100, self.poll_camera)
            return
        self.camera.join()
        code = self.camera.exitcode
        self.camera.close()
        self.camera_pipe.close()
        self.camera = None
        self.live_button.state(["!disabled"])
        self.import_button.state(["!disabled"])
        self.status.set(self.calibration_label)
        if self.camera_error or code:
            messagebox.showerror("实时相机测量", self.camera_error or f"测量进程意外退出（{code}）。", parent=self.root)

    def close(self):
        if self.camera is not None:
            self.camera_stop.set()
            self.camera.join(timeout=0.5)
            if self.camera.is_alive():
                self.camera.terminate()
                self.camera.join(timeout=0.5)
            self.camera_pipe.close()
        self.root.destroy()

    def open_directory(self, directory=None):
        from tkinter import filedialog, messagebox
        if directory is None:
            directory = filedialog.askdirectory(parent=self.root, title="打开包含 measurement.npz 与 summary.json 的目录")
        if not directory:
            return
        try:
            record = Measurement.load(directory)
            rows = list(record.rows())
            pixels = result_pixels(record)
            info = []
            for row in rows:
                value = row["median_height_mm"]
                level = f"{float(value):+.2f} mm" if value is not None else "无有效高度"
                info.append(f"{row['id']}　{row['classification']}　{level}")
        except (OSError, ValueError, AttributeError, TypeError, EOFError, BadZipFile) as exc:
            messagebox.showerror("无法打开测量记录", str(exc), parent=self.root)
            return
        self.record, self.pixels = record, pixels
        self.color_key.pack_forget()
        self.export_button.configure(text="导出当前记录 CSV")
        self.home_button.pack(side="right", padx=4)
        self.canvas.configure(cursor="crosshair")
        self.point_xy = self.placement = None
        self.side_title.set("测点读数")
        self.detail.set("点击左侧测量区域，查看该点的相机深度和有符号高度。")
        self.region_list.delete(0, "end")
        for item in info:
            self.region_list.insert("end", item)
        self.region_list.pack_forget()
        self.region_list.pack(before=self.legend, fill="x", pady=(0, 8))
        state = "参考平面已拟合" if record.summary["plane_ok"] else "参考平面拟合失败"
        self.meta.set(f"共 {len(rows)} 个候选区域 · {state}\n列表数值为区域高度中位数。")
        self.footer.set(f"{Path(directory).name}   ·   红色为凸起，蓝色为凹陷，浅色为推测，黑色为缺测。")
        self.draw()

    def select_region(self, event):
        selection = self.region_list.curselection()
        if not selection or self.record is None:
            return
        row = self.record.summary["region_stats"][selection[0]]
        cov = row.get("depth_coverage_ratio")
        coverage = f"{cov:.0%}" if isinstance(cov, (int, float)) else "无记录"
        self.point_xy = None
        self.side_title.set(f"区域 {row.get('id', '')}")
        self.detail.set(f"{self.region_list.get(selection[0])}\n有效深度覆盖率　{coverage}\n"
                        f"有效像素数　{row.get('measured_pixels', '无记录')}")
        self.draw()

    def draw(self):
        c = self.canvas
        w, h = c.winfo_width(), c.winfo_height()
        if w < 100 or h < 100:
            return
        c.delete("all")
        is_demo = self.record is None
        source = self.poster if is_demo else self.pixels
        if source is None:
            c.create_text(w/2, h/2, text="未找到演示图片.png", fill="#455b64")
            self.placement = None
            return
        ih, iw = source.shape[:2]
        space = 36 if is_demo else 0
        scale = min((w-16)/iw, (h-space-16)/ih)
        sw, sh = max(1, int(iw*scale)), max(1, int(ih*scale))
        if is_demo:
            # 缩小截图时按面积采样，保留旧测量窗口上的字迹。
            try:
                import cv2
                rgb = cv2.resize(source, (sw, sh), interpolation=cv2.INTER_AREA)
            except ImportError:
                rgb = source[np.arange(sh)[:, None]*ih//sh, np.arange(sw)*iw//sw]
        else:
            rgb = source[np.arange(sh)[:, None]*ih//sh, np.arange(sw)*iw//sw]
        import tkinter as tk
        self.photo = tk.PhotoImage(master=self.root, format="PPM",
                                  data=f"P6\n{sw} {sh}\n255\n".encode()+rgb.tobytes())
        ox, oy = (w-sw)//2, (h-space-sh)//2
        self.placement = (ox, oy, sw, sh)
        c.create_image(ox, oy, image=self.photo, anchor="nw")
        if is_demo:
            c.create_text(w//2, oy+sh+17, text="点击图片播放演示视频", fill="#375766",
                          font=("Microsoft YaHei UI", 10))
            return
        if self.point_xy is not None:
            u, v = self.point_xy
            x = ox+(u-self.record.roi[0]+0.5)*sw/iw
            y = oy+(v-self.record.roi[1]+0.5)*sh/ih
            c.create_oval(x-5,y-5,x+5,y+5,outline="white",width=2)
            c.create_oval(x-7,y-7,x+7,y+7,outline="#172a32")

    def click(self, event):
        if self.placement is None:
            return
        ox, oy, sw, sh = self.placement
        dx, dy = event.x-ox, event.y-oy
        if self.record is None:
            if 0 <= dx < sw and 0 <= dy < sh+30:
                self.play_demo()
            return
        if not (0 <= dx < sw and 0 <= dy < sh):
            return
        x, y, iw, ih = self.record.roi
        u, v = x+dx*iw//sw, y+dy*ih//sh
        self.region_list.selection_clear(0, "end")
        self.side_title.set("测点读数")
        try:
            point = self.record.point(u,v)
        except ValueError as exc:
            self.point_xy = None
            self.detail.set(str(exc))
        else:
            self.point_xy = u, v
            z, level = point["相机深度_mm"], point["有符号高度_mm"]
            depth_text = f"{z:.2f} mm" if z is not None else "无读数"
            level_text = f"{level:+.2f} mm" if level is not None else "无读数"
            self.detail.set(f"左图坐标　({u}, {v})\n{point['数据来源']}\n\n"
                            f"相机深度　{depth_text}\n有符号高度　{level_text}")
        self.draw()

    def export(self):
        from tkinter import filedialog, messagebox
        filename = filedialog.asksaveasfilename(parent=self.root, defaultextension=".csv",
            filetypes=[("CSV", "*.csv")], initialfile="水池实测区间.csv" if self.record is None else "缺陷区域.csv")
        if not filename:
            return
        try:
            if self.record is None:
                with Path(filename).open("x", newline="", encoding="utf-8-sig") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(REPORT_HEADER)
                    writer.writerows(POOL_RESULTS)
            else:
                self.record.export(filename)
        except FileExistsError:
            messagebox.showinfo("文件已存在", "请使用新的 CSV 文件名。", parent=self.root)
        except (OSError, ValueError) as exc:
            messagebox.showerror("导出失败", str(exc), parent=self.root)
        else:
            self.footer.set(f"已导出　{filename}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "--gui":
        parser = argparse.ArgumentParser(description="直接展示已保存的双目缺陷测量结果。")
        parser.add_argument("directory", nargs="?", type=Path)
        opts = parser.parse_args(argv[1:])
        import tkinter as tk
        root = tk.Tk()
        ResultsWindow(root, opts.directory)
        root.mainloop()
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("directory", nargs="?", type=Path, help="已有测量结果目录")
    source.add_argument("--demo", action="store_true", help="使用合成数据演示复核")
    parser.add_argument("--point", nargs=2, type=int, metavar=("U", "V"), help="完整校正左图上的像素")
    parser.add_argument("--csv", type=Path, help="另存缺陷结果表（新文件）")
    args = parser.parse_args(argv)
    try:
        record = demo() if args.demo else Measurement.load(args.directory)
        print("合成示例（仅演示复核操作）" if args.demo else f"结果目录 {args.directory}")
        print("参考平面拟合成功" if record.summary["plane_ok"] else "参考平面拟合失败")
        rows = list(record.rows())
        print(f"候选区域数 {len(rows)}")
        for row in rows:
            level = row["median_height_mm"]
            value = "无有效高度" if level is None else f"{level:+.2f} mm"
            print(f"  区域 {row['id']}  {row['classification']}  高度中位数 {value}")
        if args.point:
            print(json.dumps(record.point(*args.point), ensure_ascii=False, allow_nan=False))
        if args.csv:
            record.export(args.csv)
            print(f"已保存 {args.csv}")
    except FileExistsError:
        parser.exit(1, "目标 CSV 已存在，请换一个文件名。\n")
    except (OSError, ValueError, TypeError, AttributeError, BadZipFile, EOFError) as exc:
        parser.exit(1, f"复核失败：{exc}\n")


if __name__ == "__main__":
    main()
