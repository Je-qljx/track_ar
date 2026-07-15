import sys
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog
import threading
import queue
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration.coords import TrackGeometry, WorldCoord, ImageCoord
from calibration.frame_tracker import FrameTracker
from pipeline.main_pipeline import TrackARPipeline

CALIB_NAMES = ["起跑线×道1", "起跑线×道8", "终点线×道1", "终点线×道8"]
CALIB_TARGET_NAMES = [
    "起跑线侧·内道侧 (S-in)",
    "终点线侧·内道侧 (F-in)",
    "终点线侧·外道侧 (F-out)",
    "起跑线侧·外道侧 (S-out)",
]
CALIB_COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (0, 255, 255)]
_use_target_names = False


class TrackARApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("TrackAR - 田径赛道 AR 叠加")
        self.window.geometry("1280x780")
        self.window.minsize(1024, 640)

        self.video_path: str | None = None
        self.first_frame: np.ndarray | None = None
        self.calib_pixels: list[tuple[int, int]] | None = None
        self.pipeline: TrackARPipeline | None = None
        self.processing = False
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.cap_fps: float = 30.0
        self.total_frames: int = 0

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.window, padding=12)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(main, text="控制面板", width=280)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)

        # Video
        ttk.Label(left, text="模式", font=("", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(8, 2))
        self.demo_btn = ttk.Button(left, text="演示模式（合成8名运动员）", command=self._launch_demo)
        self.demo_btn.grid(row=1, column=0, sticky="ew", pady=2)

        ttk.Separator(left, orient="horizontal").grid(row=2, column=0, sticky="ew", pady=6)

        ttk.Label(left, text="视频处理:", font=("", 10, "bold")).grid(row=3, column=0, sticky="w")
        self.video_label = ttk.Label(left, text="未选择文件", foreground="gray")
        self.video_label.grid(row=4, column=0, sticky="ew", padx=4)
        ttk.Button(left, text="选择视频", command=self._browse_video).grid(row=5, column=0, sticky="ew", pady=(4, 8))

        ttk.Separator(left, orient="horizontal").grid(row=6, column=0, sticky="ew", pady=4)

        # Track type
        ttk.Label(left, text="赛道类型", font=("", 10, "bold")).grid(row=7, column=0, sticky="w")
        track_var = tk.StringVar(value="100m")
        self.track_var = track_var
        track_frame = ttk.Frame(left)
        track_frame.grid(row=8, column=0, sticky="ew", pady=2)
        ttk.Radiobutton(track_frame, text="100m", variable=track_var, value="100m").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(track_frame, text="400m（椭圆）", variable=track_var, value="400m").pack(side=tk.LEFT)

        ttk.Separator(left, orient="horizontal").grid(row=9, column=0, sticky="ew", pady=4)

        # Camera
        ttk.Label(left, text="相机参数", font=("", 10, "bold")).grid(row=10, column=0, sticky="w")
        fx_frame = ttk.Frame(left)
        fx_frame.grid(row=11, column=0, sticky="ew", pady=2)
        ttk.Label(fx_frame, text="焦距（全画幅mm）:").pack(side=tk.LEFT)
        self.fx_mm_var = tk.IntVar(value=200)
        fx_slider = ttk.Scale(fx_frame, from_=24, to=800, orient="horizontal",
                               variable=self.fx_mm_var, command=self._on_fx_change)
        fx_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        self.fx_label = ttk.Label(fx_frame, text="200mm", width=8)
        self.fx_label.pack(side=tk.LEFT, padx=(4, 0))

        ttk.Separator(left, orient="horizontal").grid(row=12, column=0, sticky="ew", pady=4)

        # Detection
        ttk.Label(left, text="检测", font=("", 10, "bold")).grid(row=13, column=0, sticky="w")
        self.yolo_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="使用 YOLO（需安装 ultralytics）", variable=self.yolo_var).grid(row=14, column=0, sticky="w", pady=2)

        ttk.Separator(left, orient="horizontal").grid(row=15, column=0, sticky="ew", pady=4)

        # Calibration target mode
        self.target_mode_var = tk.BooleanVar(value=False)
        self.target_frame = ttk.LabelFrame(left, text="标定物参数", labelanchor="n")
        self.target_check = ttk.Checkbutton(left, text="使用标定物替代起终点",
                                             variable=self.target_mode_var,
                                             command=self._on_target_mode_toggle)
        self.target_check.grid(row=16, column=0, sticky="w", pady=(0, 2))
        self.target_frame.grid(row=17, column=0, sticky="ew", padx=4)
        self.target_frame.columnconfigure(1, weight=1)
        # Target position
        ttk.Label(self.target_frame, text="位置距离 (m):").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.target_dm_var = tk.DoubleVar(value=50.0)
        ttk.Entry(self.target_frame, textvariable=self.target_dm_var, width=8).grid(row=0, column=1, sticky="w", pady=2)
        ttk.Label(self.target_frame, text="车道 #:").grid(row=1, column=0, sticky="w", padx=(0, 4))
        self.target_lane_var = tk.IntVar(value=5)
        ttk.Spinbox(self.target_frame, from_=1, to=8, textvariable=self.target_lane_var, width=4).grid(row=1, column=1, sticky="w", pady=2)
        # Target dimensions
        ttk.Label(self.target_frame, text="目标宽×高 (m):").grid(row=2, column=0, sticky="w", padx=(0, 4))
        dim_frame = ttk.Frame(self.target_frame)
        dim_frame.grid(row=2, column=1, sticky="ew", pady=2)
        self.target_w_var = tk.DoubleVar(value=0.420)
        self.target_h_var = tk.DoubleVar(value=0.297)
        ttk.Entry(dim_frame, textvariable=self.target_w_var, width=6).pack(side=tk.LEFT)
        ttk.Label(dim_frame, text="×").pack(side=tk.LEFT, padx=2)
        ttk.Entry(dim_frame, textvariable=self.target_h_var, width=6).pack(side=tk.LEFT)
        ttk.Label(self.target_frame, text="(默认 A3: 0.420×0.297)", foreground="gray", font=("", 8)).grid(row=3, column=0, columnspan=2, sticky="w")
        self.target_frame.grid_remove()  # hidden by default

        # Buttons
        self.calibrate_btn = ttk.Button(left, text="标定（点击4个点）", command=self._calibrate, state="disabled")
        self.calibrate_btn.grid(row=18, column=0, sticky="ew", pady=2)
        self.start_btn = ttk.Button(left, text="开始处理", command=self._start_processing, state="disabled")
        self.start_btn.grid(row=19, column=0, sticky="ew", pady=2)
        self.open_btn = ttk.Button(left, text="打开输出文件夹", command=self._open_output, state="disabled")
        self.open_btn.grid(row=20, column=0, sticky="ew", pady=2)

        ttk.Separator(left, orient="horizontal").grid(row=21, column=0, sticky="ew", pady=4)

        # Calibration info
        self.calib_label = ttk.Label(left, text="标定: 未完成", foreground="gray")
        self.calib_label.grid(row=22, column=0, sticky="w", pady=2)
        self.output_label = ttk.Label(left, text="", foreground="gray")
        self.output_label.grid(row=23, column=0, sticky="w")

        # Right: preview + progress
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        preview_frame = ttk.LabelFrame(right, text="预览")
        preview_frame.grid(row=0, column=0, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self.preview_label = ttk.Label(preview_frame, background="black")
        self.preview_label.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        # Dummy image
        self._show_placeholder()

        # Progress
        bottom = ttk.Frame(right)
        bottom.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        bottom.columnconfigure(1, weight=1)
        ttk.Label(bottom, text="进度:").grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(bottom, mode="determinate")
        self.progress.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.status_var = tk.StringVar(value="就绪")
        self.status_label = ttk.Label(bottom, textvariable=self.status_var, foreground="gray")
        self.status_label.grid(row=0, column=2, padx=(8, 0))

        # Log
        log_frame = ttk.LabelFrame(self.window, text="日志", padding=4)
        log_frame.pack(fill=tk.X, padx=12, pady=(0, 8))
        self.log_text = tk.Text(log_frame, height=4, state="disabled", wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.pack(fill=tk.X)
        self.log("TrackAR GUI 已启动。请选择视频或点击演示模式。")

    def _show_placeholder(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "选择视频或点击演示模式", (80, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
        cv2.putText(img, "开始使用", (250, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
        self._show_image(img)

    def _show_image(self, img: np.ndarray):
        from PIL import Image, ImageTk
        # Fit to preview area while keeping aspect ratio
        h, w = img.shape[:2]
        max_w, max_h = 640, 480
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self._tk_img = ImageTk.PhotoImage(pil_img)
        self.preview_label.config(image=self._tk_img)

    def _on_fx_change(self, val):
        mm = int(float(val))
        self.fx_label.config(text=f"{mm}mm")

    def _on_target_mode_toggle(self):
        global _use_target_names
        if self.target_mode_var.get():
            self.target_frame.grid()
            self.calibrate_btn.config(text="标定（点击目标4个角）")
            _use_target_names = True
            w_t = self.target_w_var.get()
            h_t = self.target_h_var.get()
            dm = self.target_dm_var.get()
            lane = self.target_lane_var.get()
            self.log(f"[标定] 标定物模式: {w_t:.3f}×{h_t:.3f}m 置于 dm={dm}m 车道{lane}")
            self.log(f"  请按跑道方向点击纸的4个角：")
            self.log(f"  ①起跑线侧·内道侧→②终点线侧·内道侧→③终点线侧·外道侧→④起跑线侧·外道侧")
            self.log(f"  （纸的长边沿跑道方向，短边横向；不受摄像头视角影响）")
        else:
            self.target_frame.grid_remove()
            self.calibrate_btn.config(text="标定（点击4个点）")
            _use_target_names = False

    def _launch_demo(self):
        if getattr(self, '_demo_running', False):
            self.log("演示已在运行中。")
            return
        self.demo_btn.config(state="disabled")
        self.log("启动演示模式...")
        thread = threading.Thread(target=self._run_demo, daemon=True)
        thread.start()

    def _run_demo(self):
        self._demo_running = True
        import numpy as np
        import cv2
        from calibration.coords import TrackGeometry, WorldCoord, ImageCoord
        from pipeline.main_pipeline import TrackARPipeline
        from ui.control_panel import ControlPanel
        from tests.synthetic_scene import SyntheticScene

        track_type = self.track_var.get()
        is_400m = track_type == "400m"
        K = np.array([[700, 0, 960], [0, 700, 540], [0, 0, 1]], dtype=np.float64)
        geom = TrackGeometry(track_type=track_type)
        pipeline = TrackARPipeline(camera_matrix=K, geometry=geom)
        if is_400m:
            speeds = [9.3, 9.5, 9.7, 9.9, 9.1, 8.8, 9.4, 10.1]
        else:
            speeds = [9.5, 9.8, 10.2, 9.0, 8.5, 9.3, 8.8, 10.5]
        # Compute calibration from a synthetic camera pose above the track
        if is_400m:
            rvec = np.array([[0.6], [0.0], [0.0]], dtype=np.float64)
            tvec = np.array([[-10], [5], [90]], dtype=np.float64)
        else:
            rvec = np.array([[-1.9865], [-0.7462], [-0.4312]], dtype=np.float64)
            tvec = np.array([[-40.827], [-14.574], [36.127]], dtype=np.float64)
        world_pts = geom.calibration_world_points()
        w_arr = np.array([w.as_array for w in world_pts], dtype=np.float64)
        proj, _ = cv2.projectPoints(w_arr, rvec, tvec, K, np.zeros((4, 1)))
        image_pts = [ImageCoord(float(p[0, 0]), float(p[0, 1])) for p in proj]
        pipeline.calibrate_from_points(world_pts, image_pts)
        scene = SyntheticScene(pipeline.projector, geom, speeds=speeds)
        for lane in range(1, 9):
            pipeline.set_athlete_name(lane, f"选手{lane}")
        self.window.after(0, self.log, f"演示模式：8名运动员，{track_type}")
        control = ControlPanel(pipeline)
        control.create_trackbars()
        pipeline.running = True
        frame_idx = 0

        def on_reset():
            nonlocal frame_idx
            frame_idx = 0

        control.on_reset = on_reset
        while pipeline.running:
            if control.state.paused:
                key = cv2.waitKey(100) & 0xFF
                if key > 0:
                    control.handle_key(key)
                continue
            t = frame_idx / 60.0
            athletes = scene.update(t)
            canvas = scene.render(athletes)
            detections = scene.get_detections(athletes)
            output = pipeline.process_frame(canvas, timestamp=t, external_detections=detections)
            if control.state.overlay_enabled:
                control.draw_controls(output)
            cv2.imshow("TrackAR Demo", output)
            key = cv2.waitKey(1) & 0xFF
            if key > 0:
                if not control.handle_key(key):
                    pipeline.running = False
            frame_idx += 1
            if frame_idx > 3600:
                frame_idx = 0
        cv2.destroyAllWindows()
        self._demo_running = False
        self.window.after(0, lambda: self.demo_btn.config(state="normal"))
        self.window.after(0, lambda: self.log("演示模式结束。"))

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="选择比赛视频",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.webm"), ("所有文件", "*.*")]
        )
        if not path:
            return
        self.video_path = path
        self.video_label.config(text=Path(path).name, foreground="black")
        self.calib_pixels = None
        self.calib_label.config(text="标定: 未完成", foreground="gray")
        self.start_btn.config(state="disabled")
        self.calibrate_btn.config(state="normal")
        self.open_btn.config(state="disabled")
        self.output_label.config(text="")

        # Read first frame
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self.first_frame = frame
                self._show_image(frame)
                self.cap_fps = cap.get(cv2.CAP_PROP_FPS)
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.log(f"已加载: {Path(path).name} ({self.video_w}x"
                         f"{self.video_h}, {self.cap_fps:.1f}fps, "
                         f"{self.total_frames} 帧)")
            cap.release()

    def _calibrate(self):
        if not self.video_path:
            return
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.log("[ERROR] Could not open video for calibration")
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = [0]
        frame = [None]
        points = []
        calib_frames = []
        calib_frame_idxs = []
        current = [0]

        def _seek(n):
            n = max(0, min(n, total - 1))
            idx[0] = n
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ret, f = cap.read()
            if ret:
                frame[0] = f
            return ret

        _seek(0)
        if frame[0] is None:
            cap.release()
            return
        h, w = frame[0].shape[:2]
        disp_w, disp_h = min(w, 1280), min(h, 720)
        scale_x = w / disp_w
        scale_y = h / disp_h

        def _names():
            return CALIB_TARGET_NAMES if _use_target_names else CALIB_NAMES

        TARGET_DESC = "标定物4角顺序: ①起跑线侧·内道侧→②终点线侧·内道侧→③终点线侧·外道侧→④起跑线侧·外道侧"
        STANDARD_DESC = "4点顺序: ①起跑线×道1→②起跑线×道8→③终点线×道1→④终点线×道8"

        def _make_display():
            if frame[0] is None:
                return
            d = cv2.resize(frame[0], (disp_w, disp_h))
            names = _names()
            for i, (px, py) in enumerate(points):
                dx, dy = int(px / scale_x), int(py / scale_y)
                cv2.circle(d, (dx, dy), 8, CALIB_COLORS[i], -1)
                cv2.putText(d, f"{i+1}", (dx + 10, dy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, CALIB_COLORS[i], 2)
                cv2.putText(d, names[i], (dx + 10, dy + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CALIB_COLORS[i], 1)
            y_off = 60
            desc = TARGET_DESC if _use_target_names else STANDARD_DESC
            cv2.putText(d, desc, (30, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            y_off += 35
            for j in range(current[0], 4):
                cv2.putText(d, f"  {chr(9311+j)} {names[j]}", (30, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_off += 30
            if points:
                cv2.putText(d, "点跨帧保留, 可在多帧上分别点击", (30, y_off + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                y_off += 22
            cv2.putText(d, f"帧 {idx[0]+1}/{total}" + (f"  (已点: {', '.join(f'#{i+1}' for i in range(len(points)))})" if points else ""),
                        (30, y_off + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            cv2.putText(d, "←/→ = 单帧  [/] = 10帧  SPACE=确认  r=重做  q=取消",
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            return d

        def mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and current[0] < 4:
                points.append((int(x * scale_x), int(y * scale_y)))
                calib_frames.append(frame[0].copy())
                calib_frame_idxs.append(idx[0])
                current[0] += 1
                cv2.imshow("Calibration", _make_display())

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse)
        cv2.imshow("Calibration", _make_display())

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' ') and current[0] == 4:
                break
            if key == ord('r'):
                points.clear()
                calib_frames.clear()
                calib_frame_idxs.clear()
                current[0] = 0
                cv2.imshow("Calibration", _make_display())
            if key == ord('q'):
                cv2.destroyWindow("Calibration")
                cap.release()
                return
            if key == 81:  # left arrow
                _seek(idx[0] - 1)
                cv2.imshow("Calibration", _make_display())
            if key == 83:  # right arrow
                _seek(idx[0] + 1)
                cv2.imshow("Calibration", _make_display())
            if key == ord('['):
                _seek(idx[0] - 10)
                cv2.imshow("Calibration", _make_display())
            if key == ord(']'):
                _seek(idx[0] + 10)
                cv2.imshow("Calibration", _make_display())
        cv2.destroyWindow("Calibration")
        cap.release()

        self.calib_pixels = points
        self.calib_frames = calib_frames
        self.calib_frame_idxs = calib_frame_idxs
        def _names():
            return CALIB_TARGET_NAMES if _use_target_names else CALIB_NAMES

        self.calibrate_btn.config(text="标定完成  ✓", state="normal")
        names_at_confirm = _names()
        self.calib_label.config(text="标定: 4点已设置", foreground="green")
        self.start_btn.config(state="normal")
        self.log("[标定] 4点确认完毕:")
        for i, pt in enumerate(points):
            self.log(f"  点{i+1} ({names_at_confirm[i]}): ({pt[0]}, {pt[1]}) 于帧 {calib_frame_idxs[i]+1}")

        # Show on preview
        preview = frame[0].copy()
        for i, (px, py) in enumerate(points):
            cv2.circle(preview, (px, py), 8, CALIB_COLORS[i], -1)
            cv2.putText(preview, f"{i+1}", (px + 10, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, CALIB_COLORS[i], 2)
            cv2.putText(preview, names_at_confirm[i], (px + 10, py + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CALIB_COLORS[i], 1)
        self._show_image(preview)

    def _log_status(self, msg: str):
        self.status_var.set(msg)

    def log(self, msg: str):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def _start_processing(self):
        if self.processing or not self.video_path or not self.calib_pixels:
            return
        self.processing = True
        self.start_btn.config(state="disabled")
        self.calibrate_btn.config(state="disabled")
        self.stop_event.clear()
        self.frame_queue = queue.Queue(maxsize=2)
        self.progress["value"] = 0
        self._log_status("初始化中...")

        fx_mm = self.fx_mm_var.get()
        w = getattr(self, 'video_w', 1920)
        h = getattr(self, 'video_h', 1080)
        fx_px = fx_mm * w / 36.0
        K = np.array([[fx_px, 0, w / 2],
                      [0, fx_px, h / 2],
                      [0, 0, 1]], dtype=np.float64)
        self.log(f"相机: {fx_mm}mm 全画幅等效 -> fx={fx_px:.0f}px (视频 {w}x{h})")

        track_type = self.track_var.get()
        geom = TrackGeometry(track_type=track_type)

        # Rectify calibration points to reference frame (click 1's frame)
        need_rectify = (hasattr(self, 'calib_frames') and len(self.calib_frames) == 4
                        and hasattr(self, 'calib_frame_idxs')
                        and len(set(self.calib_frame_idxs)) > 1)
        if need_rectify:
            self.log("[标定] 跨帧标定点配准中...")
            # Use full resolution + more features for cross-frame matching
            # (downscale to 640px loses fine details for 100m-apart views)
            ref_gray = cv2.cvtColor(self.calib_frames[0], cv2.COLOR_BGR2GRAY)
            ft = FrameTracker(max_width=max(ref_gray.shape[1], 1920))
            ft.orb = cv2.ORB.create(nfeatures=8000, scaleFactor=1.5, nlevels=12)
            ft.set_reference(ref_gray)
            ref_kp_count = len(ft._first_kp)
            self.log(f"  Reference frame (click 1): frame #{self.calib_frame_idxs[0]}, {ref_kp_count} ORB features (full res)")
            rectified_pixels = [self.calib_pixels[0]]
            rectify_ok = True
            for i in range(1, 4):
                gray = cv2.cvtColor(self.calib_frames[i], cv2.COLOR_BGR2GRAY)
                ft.update(gray)
                u, v = ft.current_to_calib(
                    float(self.calib_pixels[i][0]), float(self.calib_pixels[i][1]))
                rectified_pixels.append((int(round(u)), int(round(v))))
                info = ft.last_match_info
                self.log(f"  点{i+1} ({CALIB_NAMES[i]}):")
                self.log(f"    来自帧: #{self.calib_frame_idxs[i]}")
                self.log(f"    点击位置: ({self.calib_pixels[i][0]}, {self.calib_pixels[i][1]})")
                self.log(f"    配准后:   ({int(round(u))}, {int(round(v))})")
                self.log(f"    method={info.get('method','?')}"
                    f"  first_match={info.get('first_matches','?')}"
                    f"  first_inlier={info.get('first_inliers','?')}"
                    f"  pairwise_match={info.get('pairwise_matches','?')}"
                    f"  pairwise_inlier={info.get('pairwise_inliers','?')}")
                if info.get('method') == 'failed':
                    rectify_ok = False
                    self.log(f"    [WARN] 跨帧配准失败！")
            if not rectify_ok:
                self.log("[WARN] ============================================================")
                self.log("[WARN] 跨帧配准失败，标定将不准确！")
                self.log("[WARN] 原因：ORB 特征无法在两张差异很大的画面间匹配")
                self.log("[WARN] 建议：在同一帧内完成所有4个点击，或换更广角镜头")
                self.log("[WARN] ============================================================")
            self.log("[标定] 配准完成")
        else:
            rectified_pixels = self.calib_pixels
            if hasattr(self, 'calib_frame_idxs') and len(self.calib_frame_idxs) == 4:
                n_frames = len(set(self.calib_frame_idxs))
                self.log(f"[标定] 全部点来自 {n_frames} 帧，无需配准")

        if self.target_mode_var.get():
            # Calibration target: compute 4 corner world coords
            dm = self.target_dm_var.get()
            lane = self.target_lane_var.get()
            w_t = self.target_w_var.get()
            h_t = self.target_h_var.get()
            cy = geom.lane_center_y(lane)
            world_pts = [
                WorldCoord(dm - w_t/2, cy - h_t/2, 0.0),  # BL
                WorldCoord(dm + w_t/2, cy - h_t/2, 0.0),  # BR
                WorldCoord(dm + w_t/2, cy + h_t/2, 0.0),  # TR
                WorldCoord(dm - w_t/2, cy + h_t/2, 0.0),  # TL
            ]
            self.log(f"[标定] 目标位于 dm={dm}m, 车道{lane} (y={cy:.2f}m), {w_t}×{h_t}m")
        else:
            world_pts = geom.calibration_world_points()
        image_pts = [ImageCoord(float(u), float(v)) for u, v in rectified_pixels]

        calib_names = CALIB_TARGET_NAMES if _use_target_names else CALIB_NAMES
        for name, wp in zip(calib_names, world_pts):
            self.log(f"  {name}: world=({wp.x:.2f}, {wp.y:.2f}, {wp.z:.1f})")

        pipeline = TrackARPipeline(camera_matrix=K, geometry=geom)
        pipeline.calibrate_from_points(world_pts, image_pts)

        cal_err = pipeline.calibrator.get_projection_error(world_pts, image_pts)
        self.log(f"标定误差: {cal_err:.3f}px")
        # Per-point errors
        calib_names = CALIB_TARGET_NAMES if _use_target_names else CALIB_NAMES
        per_pt = pipeline.calibrator.get_per_point_errors(world_pts, image_pts)
        for name, err in zip(calib_names, per_pt):
            flag = " ***" if err > 15 else ""
            self.log(f"  {name}: 误差={err:.1f}px{flag}")
        pipeline.calibrator.print_calibration_debug(world_pts, image_pts)

        if self.yolo_var.get():
            try:
                from detection.detector import YOLODetector
                pipeline.set_detector(YOLODetector())
                self.log("使用 YOLO 检测器")
            except Exception as e:
                self.log(f"YOLO 不可用: {e}，使用虚拟检测器")
        else:
            self.log("使用虚拟检测器")

        for lane in range(1, 9):
            pipeline.set_athlete_name(lane, f"选手{lane}")

        self.pipeline = pipeline
        out_path = str(Path(self.video_path).with_name(
            Path(self.video_path).stem + "_overlay" + Path(self.video_path).suffix))

        self.output_label.config(text=f"Output: {Path(out_path).name}")
        thread = threading.Thread(target=self._process_video, args=(out_path,), daemon=True)
        thread.start()
        self.window.after(100, self._poll_preview)

    def _process_video(self, out_path: str):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.log("错误：无法打开视频")
            self.processing = False
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, (w, h))
        frame_idx = 0
        last_pct = -1
        self.pipeline.running = True
        t0 = time.time()

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            timestamp = frame_idx / fps
            output = self.pipeline.process_frame(frame, timestamp, frame_dt=1.0/fps)
            writer.write(output)

            # Throttled preview (every 15th frame)
            if frame_idx % 15 == 0:
                try:
                    self.frame_queue.put_nowait(output)
                except queue.Full:
                    pass

            pct = int(frame_idx / total * 100)
            if pct != last_pct:
                last_pct = pct
                self.window.after(0, self._update_progress, pct, frame_idx, total)

        elapsed = time.time() - t0
        cap.release()
        writer.release()
        fps_actual = frame_idx / max(elapsed, 0.001)
        self.window.after(0, self._finish_processing, out_path, frame_idx, fps_actual)

    def _update_progress(self, pct: int, frame: int, total: int):
        self.progress["value"] = pct
        self._log_status(f"处理中: {frame}/{total} ({pct}%)")

    def _poll_preview(self):
        if not self.processing:
            return
        try:
            frame = self.frame_queue.get_nowait()
            self._show_image(frame)
        except queue.Empty:
            pass
        self.window.after(100, self._poll_preview)

    def _finish_processing(self, out_path: str, frames: int, fps_actual: float):
        self.processing = False
        self.progress["value"] = 100
        self._log_status("完成")
        self.open_btn.config(state="normal")
        self.start_btn.config(state="normal")
        self.calibrate_btn.config(state="normal")
        self._output_path = out_path
        self.log(f"完成：{frames} 帧已处理 @ {fps_actual:.1f}fps")
        self.log(f"输出文件: {out_path}")

        # Show last frame
        self.window.after(200, self._show_last_frame, out_path)

    def _show_last_frame(self, path: str):
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
            ret, frame = cap.read()
            if ret:
                cv2.putText(frame, "处理完成", (frame.shape[1] // 2 - 120, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                self._show_image(frame)
            cap.release()

    def _open_output(self):
        import subprocess
        path = getattr(self, '_output_path', None)
        if path:
            subprocess.Popen(f'explorer /select,"{path}"')

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = TrackARApp()
    app.run()
