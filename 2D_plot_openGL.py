# gpu_lines_instanced_clip_fixed.py
# Millions of lines y = a*x + b with one-time GPU upload,
# instanced GL_LINES, shader-side clip planes (no GS), transparency,
# zoom-aware on-GPU LOD, and robust early Y-culling (clamped to [xmin,xmax]).

import sys
import ctypes as C
import numpy as np
import glfw
from OpenGL.GL import *

# --------------------------- utils ---------------------------

def ortho(l, r, b, t, n=-1.0, f=1.0):
    rl, tb, fn = (r-l), (t-b), (f-n)
    return np.array([
        [2.0/rl, 0.0,    0.0,    -(r+l)/rl],
        [0.0,    2.0/tb, 0.0,    -(t+b)/tb],
        [0.0,    0.0,   -2.0/fn, -(f+n)/fn],
        [0.0,    0.0,    0.0,     1.0     ],
    ], dtype=np.float32)

def compile_shader(src, stype):
    sid = glCreateShader(stype)
    glShaderSource(sid, src)
    glCompileShader(sid)
    if not glGetShaderiv(sid, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(sid).decode(errors="ignore"))
    return sid

def link_program(vs_src, fs_src):
    vs = compile_shader(vs_src, GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)
    pid = glCreateProgram()
    glAttachShader(pid, vs); glAttachShader(pid, fs)
    glLinkProgram(pid)
    glDeleteShader(vs); glDeleteShader(fs)
    if not glGetProgramiv(pid, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(pid).decode(errors="ignore"))
    return pid

# --------------------------- shaders ---------------------------

VS_SRC = r"""
#version 330 core
layout(location=0) in float a_t;     // 0 or 1 (two base vertices)
layout(location=1) in vec2  a_ab;    // (a,b) per instance (raw FP16/FP32)
layout(location=2) in vec4  a_col;   // per-instance color (RGBA8 norm or FP32)

uniform mat4  u_mvp;
uniform vec2  u_xrange;              // [xmin, xmax] for line domain
uniform vec4  u_window;              // (l, r, b, t) world window
uniform int   u_use_color;           // 0: black, 1: a_col
uniform float u_alpha;               // global alpha multiplier

// Zoom-aware LOD controls
uniform int   u_enable_subsample;    // 0/1
uniform float u_keep_prob;           // in (0,1]

out vec4 v_col;

void main() {
    // Endpoint along x within the domain
    float x = mix(u_xrange.x, u_xrange.y, a_t);
    float y = a_ab.x * x + a_ab.y;
    vec2  w = vec2(x, y);

    // Project
    gl_Position = u_mvp * vec4(w, 0.0, 1.0);

    // Clip planes in world (>=0 is inside)
    gl_ClipDistance[0] =  w.x - u_window.x; // left
    gl_ClipDistance[1] =  u_window.y - w.x; // right
    gl_ClipDistance[2] =  w.y - u_window.z; // bottom
    gl_ClipDistance[3] =  u_window.w - w.y; // top

    // ---- Robust early Y-culling -----------------------------------------
    // Clamp the test segment to the INTERSECTION of the window's x-span and [xmin,xmax].
    float l = u_window.x, r = u_window.y;
    float xmin = u_xrange.x, xmax = u_xrange.y;
    float xA = max(l, xmin);
    float xB = min(r, xmax);
    bool noOverlapX = (xA > xB);
    float yA = a_ab.x * xA + a_ab.y;
    float yB = a_ab.x * xB + a_ab.y;
    float bottom = u_window.z, top = u_window.w;
    bool outsideY = (yA > top && yB > top) || (yA < bottom && yB < bottom);

    // ---- On-GPU probabilistic LOD (deterministic per instance) ----------
    uint id = uint(gl_InstanceID);
    id ^= id >> 17; id *= 0xed5ad4bbu; id ^= id >> 11;
    id *= 0xac4c1b51u; id ^= id >> 15; id *= 0x31848babu;
    float rnd = float(id & 0x00FFFFFFu) * (1.0/16777215.0);
    bool drop = (u_enable_subsample == 1) && (rnd > u_keep_prob);

    if (noOverlapX || outsideY || drop) {
        // Force clip (discard) if the segment cannot contribute
        gl_ClipDistance[0] = -1.0;
        gl_ClipDistance[1] = -1.0;
        gl_ClipDistance[2] = -1.0;
        gl_ClipDistance[3] = -1.0;
    }

    v_col = (u_use_color == 1) ? a_col : vec4(0.0, 0.0, 0.0, 1.0);
    v_col.a *= u_alpha;
}
"""

FS_SRC = r"""
#version 330 core
in vec4 v_col;
out vec4 FragColor;
void main(){ FragColor = v_col; }
"""

# --------------------------- renderer ---------------------------

class GPULinePlot:
    """
    - One-time upload of (a,b) and optional colors.
    - Instanced GL_LINES (2 verts/instance). No geometry shader.
    - Shader clip planes (gl_ClipDistance[0..3]).
    - Transparency (straight alpha).
    - Zoom-aware LOD + robust early Y-cull (clamped to [xmin,xmax]).
    """
    def __init__(self, width=1280, height=800, title="GPU Lines — instanced, clip, LOD (fixed)"):
        self.width  = int(width)
        self.height = int(height)
        self.title  = title

        # Attribute packing
        self.use_fp16_ab      = True   # (a,b) as GL_HALF_FLOAT (FP16)
        self.use_packed_color = True   # colors as RGBA8 normalized

        # GL handles
        self.window   = None
        self.vao      = None
        self.vbo_base = None
        self.vbo_ab   = None
        self.vbo_col  = None
        self.prog     = None

        # Uniforms
        self.u_mvp = self.u_xrange = self.u_window = None
        self.u_use_color = self.u_alpha = None
        self.u_enable_sub = self.u_keep_prob = None

        # Data
        self.N          = 0
        self._xrange    = (-3.0, 3.0)   # default domain
        self._has_color = False

        # Pending upload
        self._pending_ab     = None
        self._pending_colors = None
        self._pending_xr     = self._xrange

        # Camera
        self.cx = 0.0; self.cy = 0.0; self.zoom = 1.0
        self.zoom_min = 0.02; self.zoom_max = 250.0

        # Redraw-on-demand
        self._drag = False
        self._last = (0.0, 0.0)
        self._dirty = True

        # Visual controls
        self.global_alpha = 0.25

        # LOD controls
        self.enable_subsample = True
        self.max_lines_per_px = 300      # target LPP horizontally
        self.lod_disable_zoom = 3.0      # when zoom exceeds this, LOD auto-off
        self.lod_disable_h    = 0.5      # when world height < this, LOD auto-off

    # ---------- API ----------
    def set_global_alpha(self, alpha: float):
        self.global_alpha = float(np.clip(alpha, 0.0, 1.0))
        self._dirty = True

    def set_lines_ab(self, ab: np.ndarray, x_range=(-3.0, 3.0), colors: np.ndarray | None = None):
        ab = np.ascontiguousarray(ab, np.float32)
        if ab.ndim != 2 or ab.shape[1] != 2:
            raise ValueError("ab must be (N,2) float32")
        self.N = ab.shape[0]
        self._xrange = (float(x_range[0]), float(x_range[1]))

        if self.vao is None:
            self._pending_ab = ab
            self._pending_xr = self._xrange
            self._pending_colors = (np.ascontiguousarray(colors, np.float32)
                                    if colors is not None else None)
            return

        self._upload_ab_and_colors(ab, colors)
        self._dirty = True

    def add_lines_ab(self, ab_new: np.ndarray, colors: np.ndarray | None = None, default_rgba=(0.0,0.0,0.0,1.0)):
        """Append lines to the current set. Works pre- or post-run()."""
        ab_new = np.ascontiguousarray(ab_new, np.float32)
        if ab_new.ndim != 2 or ab_new.shape[1] != 2:
            raise ValueError("ab_new must be (M,2) float32")
        M = int(ab_new.shape[0])

        cols_new = None
        if colors is not None:
            cols_new = np.ascontiguousarray(colors, np.float32)
            if cols_new.ndim != 2 or cols_new.shape[0] != M or cols_new.shape[1] not in (3,4):
                raise ValueError("colors must be (M,3) or (M,4) float32 in [0,1]")
            if cols_new.shape[1] == 3:
                cols_new = np.concatenate([cols_new, np.ones((M,1), np.float32)], axis=1)

        # If GL not initialized yet -> just accumulate pending
        if self.vao is None:
            if self._pending_ab is None:
                self._pending_ab = ab_new
                self._pending_colors = cols_new
            else:
                self._pending_ab = np.vstack([self._pending_ab, ab_new])
                if (self._pending_colors is None) and (cols_new is not None):
                    # backfill previous with default color
                    prevN = self._pending_ab.shape[0] - M
                    back = np.broadcast_to(np.array(default_rgba, np.float32), (prevN,4)).copy()
                    self._pending_colors = back
                if self._pending_colors is not None:
                    if cols_new is None:
                        cols_new = np.broadcast_to(np.array(default_rgba, np.float32), (M,4)).copy()
                    self._pending_colors = np.vstack([self._pending_colors, cols_new])
            self.N = int(self._pending_ab.shape[0])
            return

        # GL initialized: ensure CPU mirrors exist
        if self._cpu_ab is None:
            self._cpu_ab = np.zeros((0,2), np.float32)
        if self._has_color and self._cpu_cols is None:
            self._cpu_cols = np.zeros((0,4), np.float32)

        # If color mode transitions from no-color -> color (or vice versa), normalize
        if (self._cpu_cols is None) and (cols_new is not None):
            # Backfill existing with default color
            if self.N > 0:
                back = np.broadcast_to(np.array(default_rgba, np.float32), (self.N,4)).copy()
            else:
                back = np.zeros((0,4), np.float32)
            self._cpu_cols = back

        # Append to CPU mirrors
        self._cpu_ab = np.vstack([self._cpu_ab, ab_new]) if (self._cpu_ab is not None and self._cpu_ab.size) else ab_new.copy()
        if self._cpu_cols is not None or cols_new is not None:
            if cols_new is None:
                cols_new = np.broadcast_to(np.array(default_rgba, np.float32), (M,4)).copy()
            self._cpu_cols = np.vstack([self._cpu_cols, cols_new]) if (self._cpu_cols is not None and self._cpu_cols.size) else cols_new.copy()

        # Upload entire set (simple & robust; for very frequent appends consider capacity growth strategy)
        self._upload_ab_and_colors(self._cpu_ab, self._cpu_cols)
        self._dirty = True
        self._wake()

    # Backward-compatible alias
    append_lines_ab = add_lines_ab

    def run(self):
        self._init_window()
        self._init_gl()
        self._init_shaders()
        self._init_buffers()

        if self._pending_ab is not None:
            self._upload_ab_and_colors(self._pending_ab, self._pending_colors)
            self._xrange = self._pending_xr
            self._pending_ab = None
            self._pending_colors = None

        glfw.swap_interval(1)  # vsync
        while not glfw.window_should_close(self.window):
            if self._dirty:
                self._draw()
                glfw.swap_buffers(self.window)
                self._dirty = False
            glfw.wait_events()
        glfw.terminate()

    # ---------- init ----------
    def _init_window(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)

        glfw.window_hint(glfw.RED_BITS,   16)
        glfw.window_hint(glfw.GREEN_BITS, 16)
        glfw.window_hint(glfw.BLUE_BITS,  16)
        glfw.window_hint(glfw.ALPHA_BITS, 16)
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)   
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)

        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(self.window)

        # Callbacks
        glfw.set_window_size_callback(self.window, self._on_resize)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_key_callback(self.window, self._on_key)

    def _init_gl(self):
        glViewport(0, 0, self.width, self.height)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        #glEnable(GL_LINE_SMOOTH) # <<<< blend
        #glHint(GL_LINE_SMOOTH_HINT, GL_NICEST) # <<<< blend
        try: 
            #glEnable(GL_MULTISAMPLE)
            glDisable(GL_MULTISAMPLE)
        except Exception: pass

        # Enable 4 clip planes for gl_ClipDistance[0..3]
        glEnable(GL_CLIP_DISTANCE0)
        glEnable(GL_CLIP_DISTANCE1)
        glEnable(GL_CLIP_DISTANCE2)
        glEnable(GL_CLIP_DISTANCE3)

    def _init_shaders(self):
        self.prog = link_program(VS_SRC, FS_SRC)
        self.u_mvp        = glGetUniformLocation(self.prog, "u_mvp")
        self.u_xrange     = glGetUniformLocation(self.prog, "u_xrange")
        self.u_window     = glGetUniformLocation(self.prog, "u_window")
        self.u_use_color  = glGetUniformLocation(self.prog, "u_use_color")
        self.u_alpha      = glGetUniformLocation(self.prog, "u_alpha")
        self.u_enable_sub = glGetUniformLocation(self.prog, "u_enable_subsample")
        self.u_keep_prob  = glGetUniformLocation(self.prog, "u_keep_prob")

    def _init_buffers(self):
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Base: a_t = [0,1] (per-vertex)
        self.vbo_base = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_base)
        t = np.array([0.0, 1.0], dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, t.nbytes, t, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))

        # Per-instance (a,b)
        self.vbo_ab = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_ab)
        glBufferData(GL_ARRAY_BUFFER, 16, None, GL_STATIC_DRAW)  # placeholder
        glEnableVertexAttribArray(1)
        if self.use_fp16_ab:
            glVertexAttribPointer(1, 2, GL_HALF_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        else:
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        glVertexAttribDivisor(1, 1)

        # Optional per-instance color
        self.vbo_col = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col)
        glBufferData(GL_ARRAY_BUFFER, 16, None, GL_STATIC_DRAW)  # placeholder
        glEnableVertexAttribArray(2)
        if self.use_packed_color:
            glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, C.c_void_p(0))  # normalized
        else:
            glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        glVertexAttribDivisor(2, 1)

        glBindVertexArray(0)

    # ---------- one-time upload ----------
    def _upload_ab_and_colors(self, ab_f32: np.ndarray, cols_f32: np.ndarray | None):
        glBindVertexArray(self.vao)

        # (a,b) -> FP16 optional
        ab_u = ab_f32.astype(np.float16) if self.use_fp16_ab else ab_f32
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_ab)
        glBufferData(GL_ARRAY_BUFFER, ab_u.nbytes, ab_u, GL_STATIC_DRAW)

        # colors -> RGBA8 normalized optional
        self._has_color = cols_f32 is not None
        if self._has_color:
            if cols_f32.shape != (ab_f32.shape[0], 4):
                raise ValueError("colors must be (N,4)")
            if self.use_packed_color:
                cols_u8 = np.clip(cols_f32 * 255.0, 0, 255).astype(np.uint8, copy=False)
                glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col)
                glBufferData(GL_ARRAY_BUFFER, cols_u8.nbytes, cols_u8, GL_STATIC_DRAW)
            else:
                glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col)
                glBufferData(GL_ARRAY_BUFFER, cols_f32.nbytes, cols_f32, GL_STATIC_DRAW)

        glBindVertexArray(0)

    # ---------- camera ----------
    def _world_window(self):
        aspect = max(self.width,1)/max(self.height,1)
        half_h = 1.0/self.zoom
        half_w = half_h*aspect
        l = self.cx - half_w; r = self.cx + half_w
        b = self.cy - half_h; t = self.cy + half_h
        return l, r, b, t

    def _mvp(self):
        l, r, b, t = self._world_window()
        return ortho(l, r, b, t)

    def screen_to_world(self, sx, sy):
        l, r, b, t = self._world_window()
        x = l + (sx/self.width) * (r-l)
        y = b + ((self.height - sy)/self.height) * (t-b)
        return x, y

    def _apply_zoom_at_cursor(self, factor, mx, my):
        wx0, wy0 = self.screen_to_world(mx, my)
        self.zoom = float(np.clip(self.zoom * factor, self.zoom_min, self.zoom_max))
        wx1, wy1 = self.screen_to_world(mx, my)
        self.cx += (wx0 - wx1); self.cy += (wy0 - wy1)

    # ---------- draw ----------
    def _draw(self):
        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT)
        if self.N <= 0:
            return

        # For peak throughput, keep MSAA off in the heavy pass; blending stays on for alpha.
        try: glDisable(GL_MULTISAMPLE)
        except Exception: pass
        glLineWidth(1.0)

        # Zoom-aware keep_prob:
        # Baseline limit of horizontal "lines per pixel".
        base_keep = float(min(1.0, (self.max_lines_per_px * self.width) / max(1, self.N)))
        # Auto-disable LOD when zoomed-in enough (or world window is small)
        l, r, b, t = self._world_window()
        world_h = t - b
        lod_active = self.enable_subsample and (self.zoom < self.lod_disable_zoom) and (world_h > self.lod_disable_h)
        keep_prob = base_keep if lod_active else 1.0

        mvp = self._mvp()
        xr0, xr1 = self._xrange

        glUseProgram(self.prog)
        glUniformMatrix4fv(self.u_mvp, 1, GL_TRUE, mvp)
        glUniform2f(self.u_xrange, xr0, xr1)
        glUniform4f(self.u_window, l, r, b, t)
        glUniform1i(self.u_use_color, 1 if self._has_color else 0)
        glUniform1f(self.u_alpha, float(self.global_alpha))
        glUniform1i(self.u_enable_sub, 1 if lod_active else 0)
        glUniform1f(self.u_keep_prob, keep_prob)

        glBindVertexArray(self.vao)
        glDrawArraysInstanced(GL_LINES, 0, 2, self.N)
        glBindVertexArray(0)
        glUseProgram(0)
        
        self._draw_overlay()

    def _draw_overlay(self):
        """Draw axes, ticks, and the blue shaded region (Core-profile safe)."""
        # ------------------------------------------------------------------
        # 1. Initialize simple color shader (only once)
        # ------------------------------------------------------------------
        if not hasattr(self, "_overlay_prog"):
            vs = compile_shader("""
                #version 330 core
                layout(location=0) in vec2 pos;
                uniform mat4 u_mvp;
                void main() {
                    gl_Position = u_mvp * vec4(pos, 0.0, 1.0);
                }
            """, GL_VERTEX_SHADER)
            fs = compile_shader("""
                #version 330 core
                out vec4 FragColor;
                uniform vec4 u_color;
                void main() { FragColor = u_color; }
            """, GL_FRAGMENT_SHADER)
            self._overlay_prog = glCreateProgram()
            glAttachShader(self._overlay_prog, vs)
            glAttachShader(self._overlay_prog, fs)
            glLinkProgram(self._overlay_prog)
            glDeleteShader(vs)
            glDeleteShader(fs)

        prog = self._overlay_prog
        glUseProgram(prog)
        glUniformMatrix4fv(glGetUniformLocation(prog, "u_mvp"), 1, GL_TRUE, self._mvp())

        # ------------------------------------------------------------------
        # 2. Blue shaded region (optional)
        # ------------------------------------------------------------------
        x_left, x_right = -1.0, -0.6
        y_bottom, y_top = -1.0, 1.0
        quad = np.array([
            [x_left,  y_bottom],
            [x_right, y_bottom],
            [x_right, y_top],
            [x_left,  y_bottom],
            [x_right, y_top],
            [x_left,  y_top],
        ], dtype=np.float32)

        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        glUniform4f(glGetUniformLocation(prog, "u_color"), 0.6, 0.7, 1.0, 0.15)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glDeleteBuffers(1, [vbo])
        glDeleteVertexArrays(1, [vao])

        # ------------------------------------------------------------------
        # 3. Axes lines (X and Y)
        # ------------------------------------------------------------------
        l, r, b, t = self._world_window()
        axes = np.array([
            [l, 0.0], [r, 0.0],   # X-axis
            [0.0, b], [0.0, t],   # Y-axis
        ], dtype=np.float32)

        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, axes.nbytes, axes, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
        glUniform4f(glGetUniformLocation(prog, "u_color"), 0.0, 0.0, 0.0, 0.8)
        glLineWidth(1.0)
        glDrawArrays(GL_LINES, 0, 4)
        glDeleteBuffers(1, [vbo])
        glDeleteVertexArrays(1, [vao])

        # ------------------------------------------------------------------
        # 4. Tick marks
        # ------------------------------------------------------------------
        tick_spacing = 0.5
        tick_len = 0.02 * (t - b)  # relative to world height

        ticks = []
        # X ticks
        x_ticks = np.arange(np.ceil(l / tick_spacing) * tick_spacing, r, tick_spacing)
        for x in x_ticks:
            ticks.append([x, -tick_len])
            ticks.append([x, +tick_len])
        # Y ticks
        y_ticks = np.arange(np.ceil(b / tick_spacing) * tick_spacing, t, tick_spacing)
        for y in y_ticks:
            ticks.append([-tick_len, y])
            ticks.append([+tick_len, y])
        if ticks:
            tick_v = np.array(ticks, dtype=np.float32)
            vao = glGenVertexArrays(1)
            vbo = glGenBuffers(1)
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, tick_v.nbytes, tick_v, GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, C.c_void_p(0))
            glUniform4f(glGetUniformLocation(prog, "u_color"), 0.0, 0.0, 0.0, 0.7)
            glLineWidth(1.0)
            glDrawArrays(GL_LINES, 0, len(tick_v))
            glDeleteBuffers(1, [vbo])
            glDeleteVertexArrays(1, [vao])

        glUseProgram(0)


    # ---------- callbacks ----------
    def _on_resize(self, win, w, h):
        self.width = max(1, int(w)); self.height = max(1, int(h))
        self._dirty = True

    def _on_scroll(self, win, dx, dy):
        factor = 1.1 if dy > 0 else 1.0/1.1
        mx, my = glfw.get_cursor_pos(self.window)
        self._apply_zoom_at_cursor(factor, mx, my)
        self._dirty = True

    def _on_mouse_button(self, win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self._drag = True
                self._last = glfw.get_cursor_pos(self.window)
            elif action == glfw.RELEASE:
                self._drag = False

    def _on_cursor(self, win, x, y):
        if not self._drag: return
        lx, ly = self._last
        wx0, wy0 = self.screen_to_world(lx, ly)
        wx1, wy1 = self.screen_to_world(x, y)
        self.cx -= (wx1 - wx0); self.cy -= (wy1 - wy0)
        self._last = (x, y)
        self._dirty = True

    def _on_key(self, win, key, sc, action, mods):
        if action != glfw.PRESS: return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(self.window, True)
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):   # '+'
            self._apply_zoom_at_cursor(1.1, self.width*0.5, self.height*0.5); self._dirty = True
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            self._apply_zoom_at_cursor(1/1.1, self.width*0.5, self.height*0.5); self._dirty = True
        elif key == glfw.KEY_R:
            self.cx = self.cy = 0.0; self.zoom = 1.0; self._dirty = True
        # Live LOD controls
        elif key == glfw.KEY_LEFT_BRACKET:   # '['
            self.max_lines_per_px = max(1, int(self.max_lines_per_px*0.8)); self._dirty = True
            print("max_lines_per_px =", self.max_lines_per_px)
        elif key == glfw.KEY_RIGHT_BRACKET:  # ']'
            self.max_lines_per_px = int(self.max_lines_per_px*1.25); self._dirty = True
            print("max_lines_per_px =", self.max_lines_per_px)
        #elif key == glfw.KEY_S:
        #    self.enable_subsample = not self.enable_subsample; self._dirty = True
        #    print("enable_subsample =", self.enable_subsample)
        elif key == glfw.KEY_S:
            self.save_current_view()

    def save_current_view(self, filename="view.png"):
        """Save the actual GPU framebuffer with axis-scaled labels."""
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image

        # --- 1. Read back the OpenGL framebuffer ---
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        img = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
        #img = np.flipud(img)  # OpenGL’s origin is bottom-left

        # --- 2. Plot the captured image with matplotlib axes ---
        l, r, b, t = self._world_window()
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        ax.imshow(img, extent=(l, r, b, t), origin="lower")
        ax.set_xlabel("μ_H (eV)")
        ax.set_ylabel("ΔE (eV)")
        ax.set_title("GPU LinePlot Snapshot — Axes Scaled")
        ax.grid(True, alpha=0.3)

        plt.savefig(filename, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved axis-calibrated view with data as {filename}")


# --------------------------- demo ---------------------------


def objective_min_distance_to_electrochemicalhull(
    reference_potentials: dict,
    H_range: tuple = (-1.0, 0.5),
    steps: int = 100,
    unique_labels: list = None,
):
    """
    Objective function for GA: minimal distance of each structure to the convex hull
    across a range of applied electrochemical potentials (U).

    The electron chemical potential is varied via the CHE formalism:
        mu_e(U) = - e * U + pH- and p_H2-dependent terms.

    Parameters
    ----------
    reference_potentials : dict
        Dictionary of fixed chemical potentials, e.g. {'Cu': -3.5, 'O': -4.2, 'H2O': -14.25}.
        These are constants and serve as the baseline for non-variable species.
    H_range : tuple
        (H_min, H_max) range of applied potential (in eV).
    steps : int
        Number of discrete U values to sample between H_min and H_max.

    Returns
    -------
    compute : callable
        Function that, when called with a list of structures, returns
        min_distances: np.ndarray of shape (N_structs,)
            Minimum energy distance to convex hull for each structure across U_range.
    """
    unique_labels = {lbl for lbl in reference_potentials.keys()}.union({'O','H'}) - {'H2O'}
    unique_labels_dict = { u:i for i, u in enumerate(unique_labels) }
    M = len(unique_labels)

    def compute(dataset):
        """
        Compute min distance to convex hull for each structure across sampled U values.

        Structures are expected to provide:
            - structure.AtomPositionManager.E : total energy (eV)
            - structure.AtomPositionManager.latticeVectors : (3,3) array for cell vectors
        """

        # 1) Unique labels: hard coded for application
        #unique_labels = ['H','O','Cu']

        # 2) Build composition matrix X and energy array y
        N = len(dataset)

        # Fill composition counts and energies
        y = dataset.get_all_energies()

        species, species_order = dataset.get_all_compositions(return_species=True)
        mapping = dataset.get_species_mapping(order="stored")
        idx = np.fromiter((mapping.get(lbl, -1) for lbl in unique_labels), dtype=int, count=len(unique_labels))
        valid = (idx >= 0)
        X = np.zeros((species.shape[0], len(unique_labels)), dtype=species.dtype)
        if np.any(valid):
            X[:, valid] = species[:, idx[valid]]

        # 3) CHE adjustment Adjust for mu_O = mu_H2O - 2mu_H
        X[:,unique_labels_dict['H']] -= 2*X[:,unique_labels_dict['O']]

        # Reference chemical potentials for fixed species
        base_mu = np.array([reference_potentials.get(lbl, 0.0) for lbl in unique_labels])
        base_mu[ unique_labels_dict['O'] ] = reference_potentials.get('H2O', 0.0)

        # Formation energy reference
        fE_ref = y - X.dot(base_mu)
        nH = X[:, unique_labels_dict['H']]

        # Sample H potentials
        H_values = np.linspace(H_range[0], H_range[1], steps)

        # Vectorized formation energies
        fE_array = fE_ref[:, None] - nH[:, None]*H_values[None, :]
        fE_hull = fE_array.min(axis=0)
        min_distances = (fE_array - fE_hull).min(axis=1)

        return fE_array.astype(np.float32), min_distances

    return compute



if __name__ == "__main__":
    app = GPULinePlot(1280, 800, "GPU Lines — upload-once, clip planes, LOD (fixed)")
    #app.set_global_alpha(0.1)

    def plot(path, add=True, colors=[1, 0, 0]):
        from sage_lib.partition.Partition import Partition

        # --- 1. Define CHE energy functions ---
        func1 = objective_min_distance_to_electrochemicalhull(
            reference_potentials={
                "Cu":  -14.916443703626898 / 4,
                "H2O": -14.25,
                "H":   -6.81835453297334 / 2,
            },
            H_range=(-1.0, 0.5),
            steps=100,
        )

        func = objective_min_distance_to_electrochemicalhull(
            reference_potentials={
                "Cu":  -14.916443703626898 / 4,
                "H2O": -14.25,
                "H":   -6.81835453297334 / 2,
            },
            H_range=(0.0, 1.0),
            steps=2,
        )

        # --- 2. Load dataset ---
        p = Partition(storage="hybrid", local_root=path)
        ab, min_distances = func(p)
        ab1, min_distances1 = func1(p)
        ab = ab[:, [0, -1]]

        L = p.containers[0].AtomPositionManager.latticeVectors
        ab = ab[:, ::-1]
        ab[:, 0] -= ab[:, 1]
        ab /= float(np.linalg.norm(np.cross(L[:, 0], L[:, 1])))

        N = p.size

        # --- 3. Base colors ---
        cols = np.zeros((N, 4), np.float32)
        cols[:, :3] = colors  # RGB base

        # --- 4. Highlight points on the convex hull (min_dist == 0) ---
        # Allow small tolerance for floating-point precision
        tol = .001
        on_hull = np.abs(min_distances1 - np.min(min_distances1)) < tol

        # Alpha: more opaque for hull structures
        cols[:, 3] = np.where(on_hull, 0.9, 0.004)

        # Optional: brighten RGB for hull members
        cols[on_hull, :3] = np.clip(np.array(colors) * 1.3, 0, 1)
        #cols[:,  3] = np.e**( -(min_distances1-np.min(min_distances1)) / (0.0258/298 * 300) ) 
        #cols[:,  3] /= np.linalg.norm(cols[:,  3])
        #cols[:,  3] /= np.max(cols[:,  3])

        print(f"{np.count_nonzero(on_hull)} structures are on the convex hull (min_distance ≈ 0).")

        # --- 5. Send to GPU plot ---
        if add:
            app.add_lines_ab(ab, colors=cols)
        else:
            app.set_lines_ab(ab, x_range=(-1.0, 0.5), colors=cols)

    # 1st PLOT add=False
    path = '/Users/dimitry/Documents/Data/EZGA/9-superhero/database/data_base/end_8_4_1'
    plot(path,add=False,colors=[1,0,0])

    # extra PLOTs add=True
    path = '/Users/dimitry/Documents/Data/EZGA/9-superhero/database/data_base/end_4_4_1'
    plot(path,add=True,colors=[0,1,0])
    path = '/Users/dimitry/Documents/Data/EZGA/9-superhero/database/data_base/end_2_2_1'
    plot(path,add=True,colors=[0,0,1])

    app.run()


    '''
    # Generic plotting example
    # python -m pip install glfw PyOpenGL numpy
    N = 1_000_000
    rng = np.random.default_rng(123)
    a = rng.uniform(-0.8, 0.8, size=N).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, size=N).astype(np.float32)
    ab = np.column_stack([a, b]).astype(np.float32)

    # Optional colors (RGBA); alpha < 1.0 to see transparency
    cols = np.empty((N, 4), dtype=np.float32)
    cols[:, :3] = rng.uniform(0.15, 0.85, size=(N, 3))
    cols[:,  3] = 0.25

    app = GPULinePlot(1280, 800, "GPU Lines — upload-once, clip planes, LOD (fixed)")
    app.set_lines_ab(ab, x_range=(-3.0, 3.0), colors=cols)
    # app.set_global_alpha(0.35)
    app.run()
    '''
