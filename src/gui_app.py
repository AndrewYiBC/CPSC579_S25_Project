import tempfile
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk, ImageOps

from segmentation_engine import SegmentationEngine
from camera_estimation_engine import CameraEstimationEngine
from reconstruction_engine import ReconstructionEngine 

class ReconstructionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("3D Reconstruction Application")
        self.geometry("1300x950")

        # ---------- engine & state ----------
        self.engine = SegmentationEngine()
        self.image_dir: Path | None = None
        self.image_paths: list[Path] = []
        self.thumbnails: list[ImageTk.PhotoImage] = []

        # segmentation display state
        self.current_display_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.segmented_display_image: Image.Image | None = None
        self.selected_mask_image: Image.Image | None = None

        # display transforms
        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0
        self._last_draw_prompts: bool = True

        # prompt state
        self.prompt_mode = tk.StringVar(value="point")
        self.prompt_points: list[tuple[int, int, int]] = []
        self.prompt_rectangles: list[tuple[int, int, int, int]] = []
        self.is_drawing_rect = False
        self.rect_start = (0, 0)
        self.temp_rect_id = None

        # auto‑seg vars
        self.points_per_side_var = tk.StringVar(value="16")
        self.points_per_batch_var = tk.StringVar(value="32")
        self.pred_iou_thresh_var = tk.StringVar(value="0.8")
        self.stability_score_thresh_var = tk.StringVar(value="0.95")
        self.stability_score_offset_var = tk.StringVar(value="1.0")
        self.crop_n_layers_var = tk.StringVar(value="1")
        self.box_nms_thresh_var = tk.StringVar(value="0.3")
        self.crop_n_points_downscale_factor_var = tk.StringVar(value="2")
        self.min_mask_region_area_var = tk.StringVar(value="25.0")
        self.use_m2m_var = tk.BooleanVar(value=True)

        self.use_overlay_var = tk.BooleanVar(value=True)
        self.overlay_borders_var = tk.BooleanVar(value=True)

        # load toolbar icons
        def _load_icon(fname: str) -> ImageTk.PhotoImage:
            img = Image.open(Path("icon") / fname).resize((32, 32), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)

        self.icon_auto_segment = _load_icon("auto.png")
        self.icon_prompt_tool = _load_icon("label.png")
        self.icon_selection_tool = _load_icon("select.png")

        # build UI
        self._build_menu()
        self._build_layout()
        self.status_label = tk.Label(self, text="Ready", bd=1, relief=tk.SUNKEN, anchor="w")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # start in folder view
        self._show_folder_view()

    # -------------------------------------------------------------------
    #                        helper & reset logic
    # -------------------------------------------------------------------
    def _reset_for_new_folder(self):
        """Resets the UI and state so the user is always back at the folder view."""
        self._show_folder_view()
        self.engine = SegmentationEngine()
        self.canvas.delete("all")
        self.prompt_points.clear()
        self.prompt_rectangles.clear()
        self.segmented_display_image = None
        self.selected_mask_image = None
        self.current_display_image = None
        self.image_paths.clear()
        self.thumbnails.clear()

        # disable menus until COLMAP is rerun
        self.file_menu.entryconfig("Save Selected Object", state=tk.DISABLED)
        self.menu.entryconfig("Actions", state=tk.DISABLED)
        self.menu.entryconfig("Help", state=tk.DISABLED)

        # remove previous thumbnail frame if it exists
        if hasattr(self, "thumb_frame") and self.thumb_frame.winfo_exists():
            self.thumb_frame.destroy()
        # restore listbox if it was destroyed previously
        if not hasattr(self, "folder_listbox") or not self.folder_listbox.winfo_exists():
            self.folder_listbox = tk.Listbox(self.folder_frame)
            self.folder_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    # -------------------------------------------------------------------
    #                             menu
    # -------------------------------------------------------------------
    def _build_menu(self):
        self.menu = tk.Menu(self)

        # File menu
        self.file_menu = tk.Menu(self.menu, tearoff=0)
        self.file_menu.add_command(label="Select Folder", command=self.load_folder)
        self.file_menu.add_command(label="Save Selected Object", command=self.save_object)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.quit)
        self.menu.add_cascade(label="File", menu=self.file_menu)

        # Actions menu
        self.actions_menu = tk.Menu(self.menu, tearoff=0)
        self.actions_menu.add_command(label="Segment Image (Auto)", command=self.segment_image_auto)
        self.actions_menu.add_command(label="Segment With Prompts", command=self.segment_image_prompt)
        self.menu.add_cascade(label="Actions", menu=self.actions_menu)

        # Help menu
        self.help_menu = tk.Menu(self.menu, tearoff=0)
        self.help_menu.add_command(label="Help", command=self.show_help)
        self.menu.add_cascade(label="Help", menu=self.help_menu)

        self.config(menu=self.menu)

        # disable until folder + CAM estimation
        self.file_menu.entryconfig("Save Selected Object", state=tk.DISABLED)
        self.menu.entryconfig("Actions", state=tk.DISABLED)
        self.menu.entryconfig("Help", state=tk.DISABLED)

    # -------------------------------------------------------------------
    #                             layout
    # -------------------------------------------------------------------
    def _build_layout(self):
        main = tk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True)

        # -------- folder selection --------
        self.folder_frame = tk.Frame(main)
        self.folder_listbox = tk.Listbox(self.folder_frame)
        self.folder_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        bf = tk.Frame(self.folder_frame)
        bf.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        self.start_button = ttk.Button(bf, text="Start Processing", state=tk.DISABLED,
                                       command=self.start_processing)
        self.start_button.pack()

        # ------- workspace thumbnails -------
        self.workspace_frame = tk.Frame(main)

        # ------- segmentation tools panel -------
        self.tools_frame = tk.Frame(main, bg="#d0d0d0", width=64)
        self.tools_frame.pack_propagate(False)
        for name, icon in (("auto", self.icon_auto_segment),
                           ("prompt", self.icon_prompt_tool),
                           ("selection", self.icon_selection_tool)):
            frame = tk.Frame(self.tools_frame, width=48, height=48, bg="#d0d0d0")
            frame.pack_propagate(False)
            frame.pack(pady=5, padx=5)
            ttk.Button(frame, image=icon,
                       command=lambda n=name: self.switch_tool_frame(n)).pack(expand=True, fill=tk.BOTH)

        # ------- segmentation canvas -------
        self.canvas = tk.Canvas(main, bg="black", bd=2, relief=tk.SUNKEN)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<Button-1>", self.on_left_button_down)
        self.canvas.bind("<B1-Motion>", self.on_left_button_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_button_up)
        self.canvas.bind("<Button-3>", self.on_right_button_down)

        # ------- settings panel -------
        self.settings_frame = tk.Frame(main, bg="#f0f0f0", width=300, bd=2, relief=tk.GROOVE)
        self.settings_frame.pack_propagate(False)
        self.auto_frame = self._build_auto_settings(self.settings_frame)
        self.prompt_frame = self._build_prompt_settings(self.settings_frame)
        self.selection_frame = self._build_selection_settings(self.settings_frame)
        for f in (self.auto_frame, self.prompt_frame, self.selection_frame):
            f.place(x=0, y=0, relwidth=1, relheight=1)
            f.lower()

    # -------------------------------------------------------------------
    #                          settings panels
    # -------------------------------------------------------------------
    def _build_auto_settings(self, parent):
        f = tk.Frame(parent, bg="#f0f0f0")
        tk.Label(f, text="Auto Segment Settings", bg="#f0f0f0", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        grid = tk.Frame(f, bg="#f0f0f0")
        grid.pack(anchor="w", padx=10)

        def add(lbl, var, row):
            tk.Label(grid, text=lbl, bg="#f0f0f0").grid(row=row, column=0, sticky="w")
            tk.Entry(grid, textvariable=var, width=8).grid(row=row, column=1, padx=5, pady=1, sticky="w")

        add("points_per_side", self.points_per_side_var, 0)
        add("points_per_batch", self.points_per_batch_var, 1)
        add("pred_iou_thresh", self.pred_iou_thresh_var, 2)
        add("stability_score", self.stability_score_thresh_var, 3)
        add("stability_offset", self.stability_score_offset_var, 4)
        add("crop_n_layers", self.crop_n_layers_var, 5)
        add("box_nms_thresh", self.box_nms_thresh_var, 6)
        add("crop_n_points_ds", self.crop_n_points_downscale_factor_var, 7)
        add("min_mask_area", self.min_mask_region_area_var, 8)
        ttk.Checkbutton(grid, text="use_m2m", variable=self.use_m2m_var).grid(row=9, column=0, columnspan=2, sticky="w")

        tk.Label(f, text="Overlay Options", bg="#f0f0f0", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        over = tk.Frame(f, bg="#f0f0f0")
        over.pack(anchor="w", padx=10)
        ttk.Checkbutton(over, text="Enable Overlay", variable=self.use_overlay_var).pack(anchor="w")
        ttk.Checkbutton(over, text="Draw Borders", variable=self.overlay_borders_var).pack(anchor="w")
        return f

    def _build_prompt_settings(self, parent):
        f = tk.Frame(parent, bg="#f0f0f0")
        tk.Label(f, text="Prompt Segment Tools", bg="#f0f0f0", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        mode = tk.Frame(f, bg="#f0f0f0")
        mode.pack(anchor="w", padx="10")
        tk.Label(mode, text="Mode:", bg="#f0f0f0").pack(anchor="w")
        ttk.Radiobutton(mode, text="Point", variable=self.prompt_mode, value="point").pack(anchor="w", padx=20)
        ttk.Radiobutton(mode, text="Rectangle", variable=self.prompt_mode, value="rectangle").pack(anchor="w", padx=20)
        ttk.Button(f, text="Clear Prompts", command=self.clear_prompts).pack(anchor="w", padx=10, pady=10)
        return f

    def _build_selection_settings(self, parent):
        f = tk.Frame(parent, bg="#f0f0f0")
        tk.Label(f, text="Selection Tools", bg="#f0f0f0", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 5))
        self.listbox = tk.Listbox(f)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.listbox.bind("<<ListboxSelect>>", self.on_object_select)

        buttons = tk.Frame(f, bg="#f0f0f0")
        buttons.pack(anchor="w", padx=10, pady=5)
        ttk.Button(buttons, text="Clear Selection", command=self.clear_selection).pack(side=tk.LEFT)
        ttk.Button(
            buttons,
            text="Object Reconstruction",
            command=self.reconstruct_object,
        ).pack(side=tk.LEFT, padx=8)
        return f

    # -------------------------------------------------------------------
    #                          view switching
    # -------------------------------------------------------------------
    def _show_folder_view(self):
        for w in (self.folder_frame, self.workspace_frame,
                  self.tools_frame, self.canvas, self.settings_frame):
            w.pack_forget()
        self.folder_frame.pack(fill=tk.BOTH, expand=True)

    def _show_segmentation_view(self):
        self.folder_frame.pack_forget()
        self.workspace_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        self.tools_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.settings_frame.pack(side=tk.RIGHT, fill=tk.Y)

    # -------------------------------------------------------------------
    #                         folder & COLMAP
    # -------------------------------------------------------------------
    def _reset_for_new_folder(self):
        self._show_folder_view()
        self.engine = SegmentationEngine()
        self.canvas.delete("all")
        self.prompt_points.clear()
        self.prompt_rectangles.clear()
        self.segmented_display_image = None
        self.selected_mask_image   = None
        self.current_display_image = None

        self.file_menu.entryconfig("Save Selected Object", state=tk.DISABLED)
        self.menu.entryconfig("Actions",  state=tk.DISABLED)
        self.menu.entryconfig("Help",     state=tk.DISABLED)


    def load_folder(self):
        self._reset_for_new_folder()

        folder = filedialog.askdirectory()
        if not folder:
            return
        self.image_dir = Path(folder)
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        self.image_paths = sorted(p for e in exts for p in self.image_dir.glob(e))
        if not self.image_paths:
            messagebox.showerror("Error", "No image files found in selected folder")
            return
        self.start_button.config(state=tk.NORMAL)

        # ---- build thumbnail strip that wraps ----
        if hasattr(self, "thumb_frame") and self.thumb_frame.winfo_exists():
            self.thumb_frame.destroy()
        self.thumb_frame = tk.Frame(self.folder_frame)
        self.thumb_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.thumb_labels = []
        THUMB = 150
        PAD = 5

        for path in self.image_paths:
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            thumb = ImageOps.contain(img, (THUMB, THUMB), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(thumb)
            lbl = tk.Label(self.thumb_frame, image=tk_img)
            lbl.image = tk_img
            self.thumb_labels.append(lbl)

        def _layout_thumbs(event=None):
            cols = max(1, self.thumb_frame.winfo_width() // (THUMB + PAD * 2))
            for i, lbl in enumerate(self.thumb_labels):
                r, c = divmod(i, cols)
                lbl.grid(row=r, column=c, padx=PAD, pady=PAD, sticky="nsew")
        _layout_thumbs()
        self.thumb_frame.bind("<Configure>", _layout_thumbs)

        self.folder_listbox.destroy()

    def start_processing(self):
        if not self.image_dir:
            return
        self.status_label.config(text="Estimating camera positions...")
        self.update_idletasks()
        try:
            CameraEstimationEngine.run(self.image_dir, self.image_dir / "colmap_sparse")
        except Exception as e:
            messagebox.showerror("Error", f"Camera estimation failed:\n{e}")
            self.status_label.config(text="Ready")
            return
        self.status_label.config(text="Camera estimation completed")

        self.file_menu.entryconfig("Save Selected Object", state=tk.NORMAL)
        self.menu.entryconfig("Actions", state=tk.NORMAL)
        self.menu.entryconfig("Help", state=tk.NORMAL)

        self._populate_workspace()
        self._show_segmentation_view()

    def _populate_workspace(self):
        for w in self.workspace_frame.winfo_children():
            w.destroy()
        self.thumbnails.clear()
        for path in self.image_paths[:4]:
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            thumb = ImageOps.contain(img, (150, 150), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(thumb)
            lbl = tk.Label(self.workspace_frame, image=tk_img)
            lbl.image = tk_img
            lbl.pack(side=tk.LEFT, padx=5)
            lbl.bind("<Button-1>", lambda e, p=path: self.load_workspace_image(p))
            self.thumbnails.append(tk_img)

    def load_workspace_image(self, path: Path):
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        self.engine.load_image(img)
        self.prompt_points.clear()
        self.prompt_rectangles.clear()
        self.segmented_display_image = None
        self.selected_mask_image = None
        self.listbox.delete(0, tk.END)
        self._show_image(img)

    # -------------------------------------------------------------------
    #                         file I/O & segmentation
    # -------------------------------------------------------------------
    def save_object(self):
        if self.selected_mask_image is None:
            messagebox.showwarning("Warning", "Select an object first")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png")])
        if path:
            self.selected_mask_image.save(path)
            messagebox.showinfo("Saved", f"Object saved to {path}")

    def _collect_auto_params(self):
        try:
            return dict(
                points_per_side=int(self.points_per_side_var.get()),
                points_per_batch=int(self.points_per_batch_var.get()),
                pred_iou_thresh=float(self.pred_iou_thresh_var.get()),
                stability_score_thresh=float(self.stability_score_thresh_var.get()),
                stability_score_offset=float(self.stability_score_offset_var.get()),
                crop_n_layers=int(self.crop_n_layers_var.get()),
                box_nms_thresh=float(self.box_nms_thresh_var.get()),
                crop_n_points_downscale_factor=int(self.crop_n_points_downscale_factor_var.get()),
                min_mask_region_area=float(self.min_mask_region_area_var.get()),
                use_m2m=bool(self.use_m2m_var.get()),
            )
        except ValueError:
            return None

    def segment_image_auto(self):
        params = self._collect_auto_params()
        if params is None:
            messagebox.showerror("Error", "Invalid auto‑segmentation parameters")
            return
        try:
            self.status_label.config(text="Segmenting...")
            self.update_idletasks()
            anns = self.engine.segment_auto(mask_generator_params=params)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="Ready")
            return
        self.listbox.delete(0, tk.END)
        for i, m in enumerate(anns):
            self.listbox.insert(tk.END, f"Object {i+1} (IoU {m.get('predicted_iou', 0):.2f})")
        if self.use_overlay_var.get():
            overlay = self.engine.create_mask_overlay(
                self.engine.original_image.size[::-1], anns,
                borders=self.overlay_borders_var.get()
            )
            img = Image.alpha_composite(
                self.engine.original_image.convert("RGBA"), overlay
            )
            self.segmented_display_image = img
        else:
            img = self.engine.original_image
            self.segmented_display_image = img
        self._show_image(img)
        self.status_label.config(text="Ready")

    def segment_image_prompt(self):
        if not (self.prompt_points or self.prompt_rectangles):
            messagebox.showinfo("Info", "Provide prompt points or rectangles first")
            return
        try:
            self.status_label.config(text="Segmenting...")
            self.update_idletasks()
            mask = self.engine.segment_prompt(
                points=self.prompt_points,
                rectangles=self.prompt_rectangles
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="Ready")
            return
        self.listbox.delete(0, tk.END)
        self.listbox.insert(tk.END, f"Prompt mask (area {mask.sum()})")
        if self.use_overlay_var.get():
            overlay = self.engine.create_mask_overlay(
                self.engine.original_image.size[::-1],
                self.engine.segmented_masks, borders=True
            )
            img = Image.alpha_composite(
                self.engine.original_image.convert("RGBA"), overlay
            )
            self.segmented_display_image = img
        else:
            img = self.engine.original_image
            self.segmented_display_image = img
        self._show_image(img)
        self.status_label.config(text="Ready")

    # -------------------------------------------------------------------
    #                        object selection
    # -------------------------------------------------------------------
    def on_object_select(self, _event):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if getattr(self, "_current_selection_idx", None) == idx:
            self._current_selection_idx = None
            self.selected_mask_image = None
            img = self.segmented_display_image or self.engine.original_image
            if img:
                self._show_image(img, draw_prompts=True)
            self.listbox.selection_clear(0, tk.END)
            return
        self._current_selection_idx = idx
        mask, _ = self.engine.object_info[idx]
        self.selected_mask_image = self.engine.extract_object(
            self.engine.original_image, mask
        )
        self._show_image(self.selected_mask_image, draw_prompts=False)

    def clear_selection(self):
        self.listbox.delete(0, tk.END)
        self.engine.object_info.clear()
        self.selected_mask_image = None
        self.segmented_display_image = None
        self.prompt_points.clear()
        self.prompt_rectangles.clear()
        if self.engine.original_image:
            self._show_image(self.engine.original_image)

    # -------------------------------------------------------------------
    #                     object‑level 3‑D reconstruction
    # -------------------------------------------------------------------
    def reconstruct_object(self):
        if self.selected_mask_image is None:
            messagebox.showwarning("Warning", "Select an object first")
            return

        tmp_dir = Path(tempfile.mkdtemp())
        input_png = tmp_dir / "object.png"
        self.selected_mask_image.save(input_png)

        out_dir = (self.image_dir or Path(".")).joinpath("reconstruction")
        self.status_label.config(text="Reconstructing object...")
        self.update_idletasks()
        try:
            mesh_paths = ReconstructionEngine.run(
                [input_png],
                output_dir=out_dir,
                bake_texture=False,
                render=False,
            )
        except Exception as exc:
            messagebox.showerror("Error", f"Reconstruction failed:\n{exc}")
            self.status_label.config(text="Ready")
            return

        self.status_label.config(text="Ready")
        messagebox.showinfo(
            "Done",
            f"Reconstruction complete!\nMesh saved to:\n{mesh_paths[0].parent}",
        )

    # -------------------------------------------------------------------
    #                     canvas rendering & prompts
    # -------------------------------------------------------------------
    def _show_image(self, img: Image.Image, *, draw_prompts: bool = True):
        self._last_draw_prompts = draw_prompts
        self.current_display_image = img
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            self.after(100, lambda: self._show_image(img))
            return
        iw, ih = img.size
        r = min(cw / iw, ch / ih)
        self.display_scale = r
        self.display_offset_x = (cw - iw * r) / 2
        self.display_offset_y = (ch - ih * r) / 2
        resized = img.resize((int(iw * r), int(ih * r)), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.display_offset_x, self.display_offset_y,
            anchor="nw", image=self.tk_image
        )
        if draw_prompts:
            self._redraw_prompts()

    def _redraw_prompts(self):
        for x, y, lab in self.prompt_points:
            cx, cy = self.img_to_canvas(x, y)
            color = "green" if lab == 1 else "red"
            r = 3
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    outline=color, fill=color)
        for x0, y0, x1, y1 in self.prompt_rectangles:
            cx0, cy0 = self.img_to_canvas(x0, y0)
            cx1, cy1 = self.img_to_canvas(x1, y1)
            self.canvas.create_rectangle(cx0, cy0, cx1, cy1,
                                         outline="blue", width=2)

    def on_canvas_resize(self, _):
        if self.current_display_image is not None:
            self._show_image(self.current_display_image,
                             draw_prompts=self._last_draw_prompts)

    # --------------------- prompt mouse handlers --------------------
    def on_left_button_down(self, event):
        if self.engine.original_image is None:
            return
        ix, iy = self.canvas_to_img(event.x, event.y)
        if self.prompt_mode.get() == "point":
            self.prompt_points.append((ix, iy, 1))
            self._redraw_prompts()
        else:
            self.is_drawing_rect = True
            self.rect_start = (ix, iy)

    def on_left_button_drag(self, event):
        if not self.is_drawing_rect or self.prompt_mode.get() != "rectangle":
            return
        if self.temp_rect_id is not None:
            self.canvas.delete(self.temp_rect_id)
        x0, y0 = self.rect_start
        x1, y1 = self.canvas_to_img(event.x, event.y)
        cx0, cy0 = self.img_to_canvas(x0, y0)
        cx1, cy1 = self.img_to_canvas(x1, y1)
        self.temp_rect_id = self.canvas.create_rectangle(
            cx0, cy0, cx1, cy1, outline="blue", width=2
        )

    def on_left_button_up(self, event):
        if self.is_drawing_rect and self.prompt_mode.get() == "rectangle":
            self.is_drawing_rect = False
            if self.temp_rect_id is not None:
                self.canvas.delete(self.temp_rect_id)
                self.temp_rect_id = None
            x0, y0 = self.rect_start
            x1, y1 = self.canvas_to_img(event.x, event.y)
            if abs(x1 - x0) > 2 and abs(y1 - y0) > 2:
                self.prompt_rectangles.append((x0, y0, x1, y1))
            self._redraw_prompts()

    def on_right_button_down(self, event):
        if self.engine.original_image is None or self.prompt_mode.get() != "point":
            return
        ix, iy = self.canvas_to_img(event.x, event.y)
        self.prompt_points.append((ix, iy, 0))
        self._redraw_prompts()

    # ---------------------- coordinate helpers ----------------------
    def canvas_to_img(self, cx, cy):
        ix = int((cx - self.display_offset_x) / self.display_scale)
        iy = int((cy - self.display_offset_y) / self.display_scale)
        return ix, iy

    def img_to_canvas(self, ix, iy):
        cx = ix * self.display_scale + self.display_offset_x
        cy = iy * self.display_scale + self.display_offset_y
        return cx, cy

    # -------------------------------------------------------------------
    #                                misc
    # -------------------------------------------------------------------
    def clear_prompts(self):
        self.prompt_points.clear()
        self.prompt_rectangles.clear()
        self._show_image(self.current_display_image or self.engine.original_image)

    def switch_tool_frame(self, name: str):
        mapping = {"auto": self.auto_frame,
                   "prompt": self.prompt_frame,
                   "selection": self.selection_frame}
        mapping[name].tkraise()

    def show_help(self):
        help_text = (
            "Auto Segment Settings:\n\n"
            "- points_per_side: Controls the grid resolution.\n"
            "- points_per_batch: Number of points processed per batch.\n"
            "- pred_iou_thresh: Minimum IoU for mask acceptance.\n"
            "- stability_score_thresh: Threshold for mask stability.\n"
            "- stability_score_offset: Stability offset.\n"
            "- crop_n_layers: Number of crop layers.\n"
            "- box_nms_thresh: NMS threshold.\n"
            "- crop_n_points_downscale: Factor to reduce points.\n"
            "- min_mask_area: Minimum region area.\n"
            "- use_m2m: Toggle mask-to-mask refinement.\n"
            "\nPrompt Segment Tools:\n\n"
            "- Point mode: Left-click = Positive (Green), Right-click = Negative (Red)\n"
            "- Rectangle mode: Left-click + drag draws a bounding box.\n"
            "- Clear Prompts resets all points/boxes.\n"
            "\nSelection Tools:\n\n"
            "- Shows the list of detected or prompt-based masks.\n"
            "- Selecting an item extracts that object.\n"
            "- 'Clear Selection' reverts to original or overlay image.\n"
        )
        messagebox.showinfo("Help", help_text)


# ------------------------------ main ------------------------------
if __name__ == "__main__":
    ReconstructionApp().mainloop()
