"""
Generates the architecture diagrams in docs/diagrams/ as .drawio.svg files:
SVG that GitHub shows as an image, with the draw.io diagram embedded, so
draw.io (or the VS Code draw.io extension) opens them for editing.

This only made the first versions. Once a diagram has been edited in
draw.io, edit it there; rerunning this overwrites it.

usage: python3 docs/diagrams/make_diagrams.py
"""

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent

# draw.io's default palette: fill, stroke
COLORS = {
    "blue": ("#dae8fc", "#6c8ebf"),
    "green": ("#d5e8d4", "#82b366"),
    "yellow": ("#fff2cc", "#d6b656"),
    "orange": ("#ffe6cc", "#d79b00"),
    "red": ("#f8cecc", "#b85450"),
    "purple": ("#e1d5e7", "#9673a6"),
    "gray": ("#f5f5f5", "#666666"),
    "white": ("#ffffff", "#000000"),
    "none": ("none", "#666666"),
}
FONT = "Helvetica"


def char_w(size: float) -> float:
    return size * 0.56


class Diagram:
    def __init__(self, name: str, w: int, h: int):
        self.name, self.w, self.h = name, w, h
        self.cells = []  # (kind, dict)
        self.ids = {}

    # ---- shapes -------------------------------------------------------------
    def box(self, id, x, y, w, h, lines, color="blue", size=12, rounded=True,
            dashed=False, container=False, title_bold=True, align="center"):
        """lines: first line is the title (bold), the rest smaller"""
        if isinstance(lines, str):
            lines = [lines]
        c = dict(id=id, x=x, y=y, w=w, h=h, lines=lines, color=color, size=size,
                 rounded=rounded, dashed=dashed, container=container,
                 title_bold=title_bold, align=align)
        self.cells.append(("box", c))
        self.ids[id] = c
        return c

    def group(self, id, x, y, w, h, title, color="gray", subtitle=None):
        lines = [title] + ([subtitle] if subtitle else [])
        return self.box(id, x, y, w, h, lines, color=color, dashed=True,
                        container=True, rounded=False, align="left")

    def text(self, id, x, y, w, h, lines, size=11, bold=False, align="center", bg=True):
        if isinstance(lines, str):
            lines = [lines]
        c = dict(id=id, x=x, y=y, w=w, h=h, lines=lines, size=size, bold=bold,
                 align=align, bg=bg)
        self.cells.append(("text", c))
        self.ids[id] = c
        return c

    def edge(self, src, dst, exit=(1, 0.5), entry=(0, 0.5), points=(), both=False,
             dashed=False, color="#000000", width=1.5, label=None, label_at=None,
             label_w=None):
        """exit / entry: (fx, fy) on the source / target box; points: waypoints.
        label_at: (x, y) centre of the label (default: middle of the first
        segment)."""
        s, d = self.ids[src], self.ids[dst]
        p0 = (s["x"] + s["w"] * exit[0], s["y"] + s["h"] * exit[1])
        p1 = (d["x"] + d["w"] * entry[0], d["y"] + d["h"] * entry[1])
        pts = [p0, *points, p1]
        c = dict(id=f"e{len(self.cells)}", src=src, dst=dst, exit=exit, entry=entry,
                 points=list(points), pts=pts, both=both, dashed=dashed, color=color,
                 width=width)
        self.cells.append(("edge", c))
        if label:
            if label_at is None:
                (ax, ay), (bx, by) = pts[0], pts[1]
                label_at = ((ax + bx) / 2, (ay + by) / 2)
            lines = label if isinstance(label, list) else [label]
            lw = label_w or max(len(l) for l in lines) * char_w(11) + 10
            lh = 15 * len(lines)
            self.text(f"{c['id']}l", label_at[0] - lw / 2, label_at[1] - lh / 2, lw, lh,
                      lines, size=11)

    # ---- SVG ----------------------------------------------------------------
    def _svg_lines(self, lines, cx, x, w, y, h, size, title_bold, align, sub_size=None):
        sub_size = sub_size or size - 1
        sizes = [size] + [sub_size] * (len(lines) - 1)
        lh = [s * 1.3 for s in sizes]
        out = []
        if align == "left":
            ty = y + 6
            anchor, tx = "start", x + 8
        else:
            ty = y + (h - sum(lh)) / 2
            anchor, tx = "middle", cx
        for i, (line, s) in enumerate(zip(lines, sizes)):
            ty += lh[i]
            weight = "bold" if (i == 0 and title_bold) else "normal"
            out.append(f'<text x="{tx:.1f}" y="{ty - s * 0.3:.1f}" font-family="{FONT}" '
                       f'font-size="{s}" font-weight="{weight}" text-anchor="{anchor}" '
                       f'fill="#000000">{escape(line)}</text>')
        return out

    def svg_body(self):
        out = [f'<rect x="0" y="0" width="{self.w}" height="{self.h}" fill="#ffffff"/>']
        out.append('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" '
                   'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                   '<path d="M0,0 L10,5 L0,10 z" fill="context-stroke" '
                   'stroke="none"/></marker>'
                   '<marker id="arrk" viewBox="0 0 10 10" refX="9" refY="5" '
                   'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                   '<path d="M0,0 L10,5 L0,10 z" fill="#000000"/></marker></defs>')
        texts = []
        for kind, c in self.cells:
            if kind == "box":
                fill, stroke = COLORS[c["color"]]
                rx = 6 if c["rounded"] else 0
                dash = ' stroke-dasharray="6 4"' if c["dashed"] else ""
                out.append(f'<rect x="{c["x"]}" y="{c["y"]}" width="{c["w"]}" '
                           f'height="{c["h"]}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
                           f'stroke-width="1.2"{dash}/>')
                cx = c["x"] + c["w"] / 2
                out += self._svg_lines(c["lines"], cx, c["x"], c["w"], c["y"], c["h"],
                                       c["size"], c["title_bold"], c["align"])
            elif kind == "edge":
                d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in c["pts"])
                dash = ' stroke-dasharray="5 4"' if c["dashed"] else ""
                start = ' marker-start="url(#arrk)"' if c["both"] else ""
                out.append(f'<path d="{d}" fill="none" stroke="{c["color"]}" '
                           f'stroke-width="{c["width"]}"{dash}{start} '
                           f'marker-end="url(#arrk)"/>')
            else:  # text: drawn last, on top of edges
                texts.append(c)
        for c in texts:
            if c["bg"]:
                out.append(f'<rect x="{c["x"]:.1f}" y="{c["y"]:.1f}" width="{c["w"]:.1f}" '
                           f'height="{c["h"]:.1f}" fill="#ffffff"/>')
            out += self._svg_lines(c["lines"], c["x"] + c["w"] / 2, c["x"], c["w"], c["y"],
                                   c["h"], c["size"], c["bold"], c["align"],
                                   sub_size=c["size"])
        return out

    # ---- draw.io model ------------------------------------------------------
    def _value(self, lines, title_bold, size):
        parts = []
        for i, l in enumerate(lines):
            t = escape(l)
            if i == 0 and title_bold:
                t = f"<b>{t}</b>"
            elif i > 0:
                t = f'<font style="font-size: {size - 1}px;">{t}</font>'
            parts.append(t)
        return "<br>".join(parts)

    def mxgraph(self):
        cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
        for kind, c in self.cells:
            if kind == "box":
                fill, stroke = COLORS[c["color"]]
                style = (f"rounded={1 if c['rounded'] else 0};whiteSpace=wrap;html=1;"
                         f"fillColor={fill};strokeColor={stroke};fontFamily={FONT};"
                         f"fontSize={c['size']};arcSize=6;absoluteArcSize=1;")
                if c["dashed"]:
                    style += "dashed=1;dashPattern=6 4;"
                if c["align"] == "left":
                    style += "align=left;verticalAlign=top;spacingLeft=6;spacingTop=0;"
                val = self._value(c["lines"], c["title_bold"], c["size"])
                cells.append(f'<mxCell id="{c["id"]}" value="{escape(val)}" style="{style}" '
                             f'vertex="1" parent="1"><mxGeometry x="{c["x"]}" y="{c["y"]}" '
                             f'width="{c["w"]}" height="{c["h"]}" as="geometry"/></mxCell>')
            elif kind == "text":
                style = (f"text;html=1;align={c['align']};verticalAlign=middle;"
                         f"fontFamily={FONT};fontSize={c['size']};"
                         + ("labelBackgroundColor=#ffffff;" if c["bg"] else ""))
                if c["bold"]:
                    style += "fontStyle=1;"
                val = "<br>".join(escape(l) for l in c["lines"])
                cells.append(f'<mxCell id="{c["id"]}" value="{escape(val)}" style="{style}" '
                             f'vertex="1" parent="1"><mxGeometry x="{c["x"]:.1f}" '
                             f'y="{c["y"]:.1f}" width="{c["w"]:.1f}" height="{c["h"]:.1f}" '
                             f'as="geometry"/></mxCell>')
            else:
                ex, ey = c["exit"]
                nx, ny = c["entry"]
                style = (f"edgeStyle=none;html=1;rounded=0;endArrow=block;endFill=1;"
                         f"strokeColor={c['color']};strokeWidth={c['width']};"
                         f"exitX={ex};exitY={ey};exitDx=0;exitDy=0;"
                         f"entryX={nx};entryY={ny};entryDx=0;entryDy=0;")
                if c["both"]:
                    style += "startArrow=block;startFill=1;"
                if c["dashed"]:
                    style += "dashed=1;"
                pts = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in c["points"])
                arr = f'<Array as="points">{pts}</Array>' if pts else ""
                cells.append(f'<mxCell id="{c["id"]}" style="{style}" edge="1" parent="1" '
                             f'source="{c["src"]}" target="{c["dst"]}">'
                             f'<mxGeometry relative="1" as="geometry">{arr}</mxGeometry>'
                             f'</mxCell>')
        model = (f'<mxGraphModel dx="0" dy="0" grid="1" gridSize="10" guides="1" '
                 f'tooltips="1" connect="1" arrows="1" fold="1" page="0" pageScale="1" '
                 f'pageWidth="{self.w}" pageHeight="{self.h}" background="#ffffff" '
                 f'math="0" shadow="0"><root>{"".join(cells)}</root></mxGraphModel>')
        return (f'<mxfile host="drawio" type="device"><diagram id="{self.name}" '
                f'name="{self.name}">{model}</diagram></mxfile>')

    def write(self):
        body = "\n".join(self.svg_body())
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
               f'width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" '
               f'content="{escape(self.mxgraph())}">\n{body}\n</svg>\n')
        path = OUT / f"{self.name}.drawio.svg"
        path.write_text(svg)
        print(f"wrote {path}")


# =============================================================================
# 1. System overview
# =============================================================================
def system():
    d = Diagram("system", 1300, 700)
    d.box("host", 20, 250, 160, 150, ["Host PC", "host/chat.py: prompts", "host/image_tool.py:",
                                      "model image", "Vivado / xsdb"], color="gray")
    d.group("fpga", 210, 20, 900, 560, "Kintex-7 XC7K325T: top_soc", color="none")

    # CPU subsystem
    d.group("cpu", 230, 60, 360, 250, "CPU subsystem, 150 MHz", color="purple",
            subtitle="block design (scripts/gen_cpu.tcl)")
    d.box("mbv", 250, 110, 180, 100, ["MicroBlaze V", "RV32IMF, 5-stage pipeline",
                                      "8 KB I-cache", "32 KB D-cache (write-back)"],
          color="purple")
    d.box("lmb", 450, 110, 120, 100, ["Local memory", "256 KB block RAM", "firmware, stack,",
                                      "hot buffers (LMB)"], color="purple")
    d.box("mdm", 250, 235, 170, 55, ["Debug module", "JTAG: load / debug"], color="purple")
    d.box("clk", 460, 235, 110, 55, ["Clocking wizard", "225 -> 150 MHz"], color="purple")

    d.box("sc", 230, 350, 360, 60, ["SmartConnect", "AXI interconnect, 150 / 225 MHz crossing"],
          color="yellow")

    # llm_core
    d.group("core", 630, 60, 460, 350, "llm_core, 225 MHz", color="green",
            subtitle="src/sources/llm_core.sv")
    d.box("uart", 650, 115, 150, 55, ["UART commands", "+ console in / out"], color="green")
    d.box("regs", 650, 185, 150, 55, ["CPU registers", "AXI4-Lite"], color="green")
    d.box("sdc", 650, 255, 150, 55, ["microSD controller", "SPI mode, boot load"],
          color="green")
    d.box("axim", 650, 330, 150, 55, ["AXI4 master", "256-bit: weights, KV DMA"],
          color="green")
    d.box("vpu", 830, 330, 240, 55, ["fp32 vector unit (vpu)", "quantize, SwiGLU, softmax"],
          color="yellow")
    d.box("qmv", 830, 115, 240, 55, ["int8 matmul engine (qmv)",
                                     "Q8_0, 32 MACs / cycle, + argmax"], color="green")
    d.box("attn", 830, 185, 240, 55, ["fp32 attention unit (attn)",
                                      "8 lanes: scores, weighted sum"], color="orange")
    d.box("rdr", 830, 255, 115, 55, ["DDR reader", "weights, K, V"], color="green")
    d.box("yram", 955, 255, 115, 55, ["Y / V RAMs", "results, vectors"], color="green")

    d.box("mig", 230, 450, 860, 55, ["DDR3 controller (MIG)",
                                     "1800 MT/s, 256-bit AXI, 225 MHz user clock"],
          color="blue")

    d.box("ddr", 400, 620, 250, 55, ["DDR3, 1 GB", "model image, heap, KV cache"],
          color="blue")
    d.box("sd", 1150, 255, 130, 55, ["microSD card", "model image"], color="gray")
    d.box("flash", 20, 610, 170, 65, ["QSPI flash, 32 MB", "bitstream with the", "firmware built in"],
          color="gray")

    # edges
    d.edge("host", "uart", exit=(0.5, 0), entry=(0.9, 0), points=[(100, 10), (785, 10)],
           both=True, label="USB-UART, 3 Mbaud", label_at=(400, 10))
    d.edge("host", "mdm", exit=(1, 12.5 / 150), entry=(0, 0.5),
           label="USB-JTAG", label_at=(215, 248), label_w=60)
    d.edge("mbv", "lmb", exit=(1, 0.5), entry=(0, 0.5), both=True, label="LMB",
           label_at=(440, 145), label_w=30)
    d.edge("mbv", "sc", exit=(1, 0.9), entry=(210 / 360, 0), points=[(440, 200)],
           both=True)
    d.edge("sc", "regs", exit=(1, 0.3), entry=(0, 0.5), points=[(610, 368), (610, 212.5)],
           label=["AXI4-Lite", "registers"], label_at=(610, 290), label_w=62)
    d.edge("axim", "sc", exit=(0, 0.5), entry=(1, 0.7), points=[(620, 357.5), (620, 392)],
           label=None)
    d.text("axil", 520, 412, 120, 16, "S_AXI_EXT", size=10, bg=False)
    d.edge("sc", "mig", exit=(0.5, 1), entry=(0.2, 0), points=[], both=True,
           label="AXI4, 256-bit", label_at=(410, 430))
    d.edge("mig", "ddr", exit=(0.25, 1), entry=(0.5, 0), both=True,
           points=[(445, 560), (525, 560)], label="DDR3 pins", label_at=(525, 595))
    d.edge("sdc", "sd", exit=(1, 0.9), entry=(0, 0.5), points=[(815, 304.5), (815, 420),
                                                               (1125, 420), (1125, 282.5)],
           label="SPI", label_at=(1125, 350), label_w=30)
    # the CPU reads results straight from the Y RAM over its local memory bus
    d.edge("yram", "lmb", exit=(1, 0.5), entry=(0.5, 0), points=[(1100, 282.5), (1100, 45),
                                                                 (510, 45)],
           label="LMB: results and vectors at local-memory speed", label_at=(930, 45))
    d.edge("flash", "fpga", exit=(1, 0.5), entry=(90 / 900, 1), points=[(300, 642.5)],
           label=["configuration", "at power-on"], label_at=(250, 600), label_w=80)
    d.write()


# =============================================================================
# 2. llm_core internals
# =============================================================================
def core():
    d = Diagram("llm_core", 1240, 720)
    d.group("core", 20, 20, 1210, 690, "llm_core (225 MHz)", color="none",
            subtitle="src/sources/llm_core.sv")

    d.box("rx", 40, 140, 130, 50, ["UART RX", "3 Mbaud"], color="gray")
    d.box("fsm", 220, 125, 200, 80, ["Command state machine", "STATUS, WRITE, READ, MATMUL,",
                                     "SD_LOAD, SD_STORE"], color="green")
    d.box("sdc", 220, 250, 200, 60, ["sd_ctrl", "microSD (SPI), boot load"], color="green")
    d.box("ddr", 480, 50, 170, 45, ["SmartConnect", "-> DDR3"], color="blue")
    d.box("mux", 480, 140, 170, 170, ["AXI4 master", "SD controller, command", "logic or DMA",
                                      "(V RAM -> DDR, KV cache)"], color="yellow")
    d.box("rdr", 720, 175, 170, 100, ["wstream_reader", "2 KB bursts, 512-beat FIFO",
                                      "qmv format: splits off", "the scale beats;",
                                      "raw mode for K / V"], color="green")
    d.box("qmv", 960, 110, 220, 100, ["qmv: int8 matmul", "x RAM (x.q, x.s)",
                                      "32 int8 MACs / cycle", "16 rows in flight"],
          color="green")
    d.box("attn", 960, 260, 220, 100, ["attn: fp32 attention", "A RAM (q or a)",
                                       "8 lanes: 8 multiply-adds / cycle", "same sums, same order"],
          color="orange")
    d.box("cin", 220, 330, 200, 60, ["Console input FIFO", "prompt bytes for the CPU"],
          color="gray")
    d.box("regs", 220, 430, 200, 90, ["CPU registers (AXI4-Lite)", "start the jobs: MM_*, AT_*,",
                                      "VPU_*, DMA_*; ARGMAX,", "CONSOLE out / in"],
          color="green")
    d.box("yram", 960, 440, 220, 50, ["Y RAM, 32K words", "results of CPU jobs; argmax"],
          color="green")
    d.box("vpu", 690, 520, 200, 70, ["vpu: fp32 vector unit", "QUANT, SILU, SOFTMAX, COPY",
                                     "divide, exp, add, multiply"], color="yellow")
    d.box("vram", 690, 615, 200, 40, ["V RAM, 8K words", "vectors, scratch"], color="green")
    d.box("lmbp", 960, 615, 220, 40, ["CPU local memory bus", "(LMB): Y and V RAMs"],
          color="purple")
    d.box("yfifo", 960, 540, 220, 50, ["Result FIFO", "results of UART matmuls"],
          color="gray")
    d.box("cons", 220, 590, 200, 50, ["Console FIFO", "CPU text, between commands"],
          color="gray")
    d.box("tx", 40, 590, 130, 50, ["UART TX"], color="gray")

    d.edge("rx", "fsm", entry=(0, 0.5))
    d.edge("fsm", "mux", exit=(1, 0.5), entry=(0, 25 / 170))
    d.edge("sdc", "mux", exit=(1, 0.5), entry=(0, 140 / 170))
    d.edge("fsm", "sdc", exit=(0.5, 1), entry=(0.5, 0), label="SD commands",
           label_at=(370, 228), label_w=82)
    d.edge("mux", "ddr", exit=(0.5, 0), entry=(0.5, 1), both=True)
    d.text("axil", 575, 105, 100, 16, "AXI4, 256-bit", size=10, bg=False, align="left")
    d.edge("mux", "rdr", exit=(1, 0.5), entry=(0, 0.5), label="R beats",
           label_at=(685, 210), label_w=48)
    d.edge("rdr", "qmv", exit=(1, 0.2), entry=(0, 0.5), points=[(925, 195), (925, 160)],
           label=["weights", "+ scales"], label_at=(925, 128), label_w=55)
    d.edge("rdr", "attn", exit=(1, 0.8), entry=(0, 0.5), points=[(925, 255), (925, 310)],
           label="K / V", label_at=(925, 285), label_w=36)
    d.edge("fsm", "qmv", exit=(0.5, 0), entry=(0.5, 0), points=[(320, 35), (1070, 35)],
           label="x, over the UART", label_at=(800, 35))
    d.edge("qmv", "yram", exit=(1, 0.8), entry=(1, 0.4), points=[(1200, 190), (1200, 460)])
    d.edge("qmv", "yfifo", exit=(1, 0.6), entry=(1, 0.5), points=[(1215, 170), (1215, 565)])
    d.text("yl", 1160, 505, 80, 16, "y", size=11)
    d.edge("attn", "yram", exit=(0.5, 1), entry=(0.5, 0), label="scores / sums",
           label_at=(1070, 400), label_w=82)
    d.edge("yram", "regs", exit=(0, 0.6), entry=(1, 40 / 90), label="Y window",
           label_at=(560, 470), label_w=60)
    # the vector unit
    d.edge("yram", "vpu", exit=(0.15, 1), entry=(1, 0.3), points=[(993, 541)], both=True,
           label=["results,", "scratch"], label_at=(930, 520), label_w=56)
    d.edge("vpu", "vram", exit=(0.5, 1), entry=(0.5, 0), both=True)
    d.edge("vpu", "qmv", exit=(0.2, 0), entry=(0, 0.8), points=[(730, 400), (945, 400), (945, 190)],
           label="x (QUANT)", label_at=(830, 400), label_w=66)
    d.edge("vpu", "attn", exit=(0.85, 0), entry=(0.15, 1), points=[(860, 380), (993, 380)],
           label=["a (SOFTMAX),", "q (COPY)"], label_at=(800, 360), label_w=84)
    d.edge("vram", "lmbp", exit=(1, 0.5), entry=(0, 0.5), both=True)
    d.edge("vram", "mux", exit=(0, 0.5), entry=(0.5, 1), points=[(565, 635)],
           label="DMA", label_at=(565, 560), label_w=34)
    # console
    d.edge("regs", "cons", exit=(0.5, 1), entry=(0.5, 0), label="CONSOLE",
           label_at=(370, 555), label_w=60)
    d.edge("cons", "tx", exit=(0, 0.5), entry=(1, 0.5))
    d.edge("yfifo", "tx", exit=(0.5, 1), entry=(0.5, 1), points=[(1070, 690), (105, 690)],
           label="matmul results", label_at=(600, 690))
    d.edge("fsm", "tx", exit=(0, 0.8), entry=(1, 0.2), points=[(195, 189), (195, 600)],
           label=["replies,", "READ data"], label_at=(195, 560), label_w=62)
    d.edge("rx", "cin", exit=(0.5, 1), entry=(0, 0.5), points=[(105, 360)],
           label=["CONSOLE_IN", "command"], label_at=(105, 280), label_w=72)
    d.edge("cin", "regs", exit=(0.25, 1), entry=(0.25, 0), label="CONSOLE_IN",
           label_at=(270, 410), label_w=72)
    d.write()


# =============================================================================
# 3. One token, who does what
# =============================================================================
def token():
    d = Diagram("token_flow", 1200, 620)
    d.text("t", 20, 15, 1160, 24, "One token: forward() for each of the n_layers layers, then the classifier",
           size=13, bold=True, bg=False)
    # legend
    d.box("lc", 20, 50, 160, 30, "CPU (MicroBlaze V)", color="purple", title_bold=False)
    d.box("le", 190, 50, 160, 30, "int8 matmul engine", color="green", title_bold=False)
    d.box("la", 360, 50, 160, 30, "fp32 attention unit", color="orange", title_bold=False)
    d.box("lv", 530, 50, 160, 30, "fp32 vector unit", color="yellow", title_bold=False)
    d.box("ld", 700, 50, 160, 30, "data in DDR3", color="blue", title_bold=False)

    y1, y2, y3 = 130, 300, 470
    h = 70
    # row 1: attention block
    d.box("emb", 20, y1, 130, h, ["Embedding", "dequantize the", "token's row"],
          color="purple")
    d.box("n1", 180, y1, 120, h, ["RMSNorm (CPU)", "+ quantize", "(vector unit)"], color="purple")
    d.box("qkv", 330, y1, 120, h, ["q, k, v", "matmuls"], color="green")
    d.box("rope", 480, y1, 110, h, ["RoPE", "rotate q, k"], color="purple")
    d.box("kv", 620, y1, 130, h, ["Store k, v", "DMA into the", "unit's layouts"],
          color="purple")
    d.box("sc", 780, y1, 130, h, ["Scores", "q . k_t, t <= pos"], color="orange")
    d.box("sm", 940, y1, 110, h, ["Softmax", "scale, exp, sum"], color="yellow")
    d.box("ws", 1080, y1, 100, h, ["Weighted", "sum of v_t"], color="orange")
    # row 2: rest of the layer
    d.box("wo", 1070, y2, 120, h, ["quantize (VPU)", "+ wo matmul"], color="green")
    d.box("r1", 910, y2, 120, h, ["Residual", "x += ..."], color="purple")
    d.box("n2", 760, y2, 120, h, ["RMSNorm (CPU)", "+ quantize (VPU)"], color="purple")
    d.box("w13", 610, y2, 120, h, ["w1, w3", "matmuls"], color="green")
    d.box("glu", 460, y2, 120, h, ["SwiGLU", "+ quantize"], color="yellow")
    d.box("w2", 310, y2, 120, h, ["w2", "matmul"], color="green")
    d.box("r2", 160, y2, 120, h, ["Residual", "x += ..."], color="purple")
    # row 3: classifier
    d.box("nf", 160, y3, 140, h, ["Final RMSNorm", "+ quantize (VPU)"], color="purple")
    d.box("cls", 330, y3, 170, h, ["Classifier matmul", "32000 logits"], color="green")
    d.box("am", 530, y3, 140, h, ["Argmax", "in hardware, as the", "logits come out"],
          color="green")
    d.box("tok", 700, y3, 150, h, ["Tokenizer", "decode, print to", "host/chat.py"],
          color="purple")
    d.box("w", 900, y3 - 10, 280, 90, ["Weights: qmv DDR format", "streamed by the engine,",
                                       "17 MB per token (stories15M)", "KV cache: read by the unit"],
          color="blue")

    for a, b in [("emb", "n1"), ("n1", "qkv"), ("qkv", "rope"), ("rope", "kv"), ("kv", "sc"),
                 ("sc", "sm"), ("sm", "ws")]:
        d.edge(a, b)
    d.edge("ws", "wo", exit=(0.5, 1), entry=(0.5, 0))
    for a, b in [("wo", "r1"), ("r1", "n2"), ("n2", "w13"), ("w13", "glu"), ("glu", "w2"),
                 ("w2", "r2")]:
        d.edge(a, b, exit=(0, 0.5), entry=(1, 0.5))
    d.edge("r2", "n1", exit=(0, 0.5), entry=(0.5, 1), points=[(140, y2 + 35), (140, 250),
                                                               (240, 250)],
           dashed=True, label="next layer", label_at=(90, 250), label_w=70)
    d.edge("r2", "nf", exit=(0.5, 1), entry=(0.5, 0), points=[(220, 420), (230, 420)],
           label="after the last layer", label_at=(230, 420), label_w=120)
    for a, b in [("nf", "cls"), ("cls", "am"), ("am", "tok")]:
        d.edge(a, b)
    d.text("note", 20, 565, 1160, 40,
           ["The prompt comes from host/chat.py over the UART; the CPU encodes it and feeds its "
            "tokens through forward() before generating. The CPU orchestrates: it starts the jobs and "
            "does RMSNorm, RoPE and the residual adds.",
            "The float operations are runq.c's, in runq.c's order (no FMA): the board's logits are "
            "bit-identical to the x86 build of the same firmware."], size=11, bg=False)
    d.write()


# =============================================================================
# 4. Memory map and the model image
# =============================================================================
def memory():
    d = Diagram("memory", 1160, 600)
    d.text("t1", 20, 15, 500, 24, "CPU address map", size=13, bold=True, bg=False)
    d.text("t2", 600, 15, 540, 24, "DDR3 contents (offsets; CPU address = 0x8000_0000 + offset)",
           size=13, bold=True, bg=False)
    rows = [
        ("0x0000_0000", "Local memory, 256 KB", "firmware, stack, hot buffers (LMB)", "purple", 60),
        ("0x0010_0000", "llm_core's result RAM (Y), 128 KB",
         "on the local memory bus too: fast reads", "green", 40),
        ("0x0012_0000", "llm_core's V RAM, 32 KB",
         "the vector unit's; read / write", "green", 40),
        ("0x4000_0000", "llm_core registers, 256 KB", "uncached (AXI4-Lite)", "green", 60),
        ("0x8000_0000", "DDR3, cached (512 MB)", "model image, heap", "blue", 110),
        ("0xA000_0000", "DDR3, uncached (512 MB)", "KV cache (written by DMA)", "blue", 110),
    ]
    y = 50
    for i, (addr, name, sub, col, h) in enumerate(rows):
        d.text(f"a{i}", 20, y - 8, 100, 16, addr, size=11, bg=False, align="left")
        d.box(f"m{i}", 120, y, 380, h, [name, sub], color=col)
        y += h + 20

    drows = [
        ("0x0000_0000", "Model image (host/build_image.py)", None, "blue", 250),
        ("0x1000_0000", "Heap (malloc): run state, logits, tokenizer", None, "gray", 60),
        ("0x2000_0000", "KV cache: K per layer and head [block][i][t % 128],", "V per layer and head [t][i]", "orange", 70),
    ]
    y = 50
    for i, (addr, name, sub, col, h) in enumerate(drows):
        d.text(f"b{i}", 600, y - 8, 100, 16, addr, size=11, bg=False, align="left")
        d.box(f"d{i}", 700, y, 440, h, [name] + ([sub] if sub else []), color=col,
              container=(i == 0), align="left" if i == 0 else "center")
        y += h + 30
    parts = [
        ("Header (4 KB): config, section offsets, matrix table", "white"),
        ("RMSNorm weights, float32", "white"),
        ("Token embedding, Q8_0 (int8 + scales)", "white"),
        ("tokenizer.bin", "white"),
        ("Matrices in qmv DDR format: per layer wq wk wv wo w1 w2 w3, then the classifier", "green"),
    ]
    py = 80
    for i, (p, col) in enumerate(parts):
        hh = 36 if i < 4 else 52
        d.box(f"p{i}", 715, py, 410, hh, [p], color=col, size=11, title_bold=False)
        py += hh + 6
    d.text("sd", 700, 505, 440, 54,
           ["The image is stored on the microSD card and copied to DDR3 offset 0",
            "at power-on; host/image_tool.py uploads it over the UART.",
            "The QSPI flash holds the bitstream, with the firmware in the local memory."],
           size=11, bg=False)
    m3, m4 = d.ids["m4"], d.ids["m5"]
    y3 = m3["y"] + m3["h"] * 0.3
    y4 = m4["y"] + m4["h"] * 0.3
    d.edge("m4", "d0", exit=(1, 0.3), entry=(0, 0.2), points=[(560, y3), (560, 100)],
           dashed=True)
    d.edge("m5", "d2", exit=(1, 0.3), entry=(0, 0.5), points=[(580, y4), (580, 455)],
           dashed=True)
    d.write()


# =============================================================================
# 5. Clocks, resets and the clock-domain crossings
# =============================================================================
def clocks():
    d = Diagram("clocks", 1240, 640)
    d.text("t", 20, 12, 1200, 24, "Clocks, resets and where data crosses between clock domains",
           size=13, bold=True, bg=False)
    d.box("osc", 20, 85, 170, 60, ["Board oscillator", "200 MHz, differential"], color="gray")
    d.box("btn", 20, 290, 170, 60, ["CPU RESET button", "resets the whole design"],
          color="gray")
    d.box("mig", 230, 80, 280, 110, ["DDR3 controller (MIG)", "PLL: 900 MHz memory clock",
                                     "(1800 MT/s), 4:1 -> ui_clk 225 MHz;",
                                     "reset held until calibrated"], color="blue")
    d.box("ddr", 240, 235, 200, 55, ["DDR3 SDRAM", "32 bits wide, 7.2 GB/s peak"],
          color="blue")

    d.group("g225", 560, 50, 660, 210, "225 MHz domain (ui_clk)", color="green",
            subtitle="timing closes with +0.008 ns of slack")
    d.box("core", 580, 100, 280, 120, ["llm_core", "matmul engine, attention unit,",
                                       "vector unit, weight reader, DMA,",
                                       "registers, UART, microSD"], color="green")
    d.box("io", 900, 100, 300, 70, ["Slow I/O, divided from ui_clk", "UART 3 Mbaud (75 clocks per bit)",
                                    "microSD SPI 400 kHz, then 22.5 MHz"], color="gray")

    d.box("sc", 580, 290, 280, 70, ["SmartConnect (crossing)", "clock converters: CPU -> DDR3,",
                                    "CPU -> registers (AXI4-Lite)"], color="red")
    d.box("yv", 900, 290, 300, 70, ["Y and V RAMs (crossing)", "dual-port block RAM: port A 225 MHz",
                                    "(units), port B 150 MHz (CPU's LMB)"], color="red")

    d.group("g150", 560, 390, 660, 150, "150 MHz domain (CPU)", color="purple",
            subtitle="block design, scripts/gen_cpu.tcl")
    d.box("cw", 580, 440, 280, 85, ["Clocking wizard (MMCM)", "ui_clk 225 -> 150 MHz;",
                                    "proc_sys_reset"], color="purple")
    d.box("mbv", 900, 440, 300, 85, ["MicroBlaze V", "pipeline, FPU, caches,",
                                     "256 KB local memory, debug module"], color="purple")

    d.edge("osc", "mig", exit=(1, 0.5), entry=(0, 105 / 110 * 0 + 0.3))
    d.edge("btn", "mig", exit=(1, 0.3), entry=(0, 0.8), points=[(210, 308), (210, 168)],
           label="sys_rst", label_at=(210, 240), label_w=48)
    d.edge("mig", "ddr", exit=(0.2, 1), entry=(0.3, 0), both=True)
    d.edge("mig", "core", exit=(1, 50 / 110), entry=(0, 30 / 120),
           label=["ui_clk,", "reset"], label_at=(538, 130), label_w=44)
    d.edge("mig", "cw", exit=(1, 0.85), entry=(0, 0.5), points=[(530, 173.5), (530, 482.5)],
           label="ui_clk", label_at=(530, 420), label_w=44)
    d.edge("sc", "mig", exit=(0, 0.3), entry=(250 / 280, 1), points=[(480, 311)],
           label=["AXI4 256-bit", "225 MHz"], label_at=(480, 250), label_w=80)
    d.edge("core", "sc", exit=(0.5, 1), entry=(0.5, 0), both=True,
           label="DDR3 traffic, registers", label_at=(720, 255), label_w=140)
    d.edge("core", "yv", exit=(1, 0.85), entry=(100 / 300, 0), points=[(1000, 202)],
           label="results", label_at=(1000, 240), label_w=52)
    d.edge("core", "io", exit=(1, 0.2), entry=(0, 24 / 70))
    d.edge("cw", "mbv", exit=(1, 0.5), entry=(0, 0.5), label="150 MHz",
           label_at=(880, 470), label_w=52)
    d.edge("sc", "mbv", exit=(0.85, 1), entry=(30 / 300, 0), points=[(818, 410), (930, 410)],
           both=True)
    d.edge("yv", "mbv", exit=(0.5, 1), entry=(0.5, 0), both=True, label="LMB",
           label_at=(1050, 400), label_w=34)
    d.text("note", 20, 555, 1200, 75,
           ["Everything that touches DDR3 runs on the memory controller's own 225 MHz clock, so the "
            "256-bit data path (weights, K / V, DMA) never crosses a clock domain.",
            "The CPU runs at 150 MHz because the MicroBlaze V's pipeline does not close at 225 MHz. "
            "Only two places cross: the SmartConnect (handshaked clock converters) and the",
            "dual-port Y / V RAMs. The RAMs need no handshake: the CPU reads a job's results only after "
            "it has seen the job's busy flag drop through the (slower) register path.",
            "The reset button resets the DDR3 controller; its synchronized reset then resets llm_core "
            "and, through the clocking wizard and proc_sys_reset, the CPU."],
           size=11, bg=False, align="left")
    d.write()


# =============================================================================
# 6. Where a token's time goes, and the DDR3 floor
# =============================================================================
def budget():
    d = Diagram("budget", 1240, 600)
    d.text("t", 20, 12, 1200, 24,
           "One token: 6.1 ms (165 tokens/s), stories15M, 168-token story, measured on the board "
           "by the firmware's cycle counter", size=13, bold=True, bg=False)
    x0, scale = 160, 170.0  # px per ms

    def bar(prefix, y, segs, h=56):
        x = x0
        for i, (ms, lines, col) in enumerate(segs):
            w = ms * scale
            if w >= 45:
                d.box(f"{prefix}{i}", x, y, w, h, lines, color=col, size=11 if w >= 90 else 10,
                      rounded=False)
            else:
                d.box(f"{prefix}{i}", x, y, w, h, [""], color=col, size=11, rounded=False)
                d.text(f"{prefix}{i}t", x - 25 + w / 2, y + h + 4, 70, 30, lines, size=10,
                       bg=False)
            x += w
        return x

    d.text("l1", 20, 75, 135, 40, ["Now (measured)"], size=11, bold=True, bg=False, align="left")
    bar("a", 70, [
        (3.1, ["Matmuls 3.1", "engine, streaming 17.1 MB", "of weights from DDR3"], "green"),
        (0.84, ["Attention 0.84", "scores, softmax,", "weighted sums"], "orange"),
        (0.36, ["KV", "store 0.36"], "orange"),
        (0.6, ["RMSNorm", "0.6 (CPU)"], "purple"),
        (0.4, ["RoPE", "0.4 (CPU)"], "purple"),
        (0.3, ["Quant.", "0.3"], "yellow"),
        (0.1, ["SwiGLU", "0.1"], "yellow"),
        (0.3, ["Other", "0.3"], "gray"),
    ])
    # classifier share of the matmuls
    cx1 = x0 + 1.22 * scale
    cx2 = x0 + 3.1 * scale
    d.box("cls", cx1, 160, cx2 - cx1, 26, ["classifier ~1.9 ms (10.4 of the 17.1 MB)"],
          color="white", size=10, title_bold=False, rounded=False)
    d.box("lay", x0, 160, cx1 - x0, 26, ["6 layers ~1.2 ms"], color="white", size=10,
          title_bold=False, rounded=False)

    d.text("l2", 20, 235, 135, 40, ["DDR3 floor"], size=11, bold=True, bg=False, align="left")
    d.box("floor", x0, 230, 2.7 * scale, 56, ["~2.4-2.7 ms", "17.1 MB at 6.4-7.2 GB/s"],
          color="blue", size=11, rounded=False)
    d.text("fl", x0 + 2.7 * scale + 10, 230, 560, 56,
           ["if nothing but the weight streaming were left: ~370-420 tokens/s,",
            "the ceiling for this board with bit-exact int8 weights"], size=11, bg=False,
           align="left")

    d.text("l3", 20, 315, 135, 40, ["Estimates of", "what is left"], size=11, bold=True,
           bg=False, align="left")
    d.box("e1", x0, 310, 5.0 * scale, 40, ["RMSNorm and RoPE on the vector unit: ~5.0 ms, ~200 tokens/s"],
          color="white", size=11, title_bold=False, rounded=False, dashed=True)
    d.box("e2", x0, 360, 4.2 * scale, 40, ["+ a hardware job list per layer: ~4.0-4.3 ms, ~230-250 tokens/s"],
          color="white", size=11, title_bold=False, rounded=False, dashed=True)

    # scale
    for ms in range(0, 7):
        x = x0 + ms * scale
        d.text(f"s{ms}", x - 20, 412, 40, 16, f"{ms} ms", size=10, bg=False)

    d.text("data", 20, 450, 1200, 140,
           ["Data moved per token:",
            "- weights: 17.1 MB (15.2 M int8 + one float32 scale per 32), every token; the matmuls "
            "run at 5.5 GB/s effective, 77 % of the DDR3 peak",
            "- KV cache reads: ~1.7 MB on average over the story (3.5 MB at position 255); writes: "
            "13.8 KB",
            "- about 300 hardware jobs (43 matmuls, 144 attention / softmax / copy, 72 KV DMA, 31 "
            "quantize / SwiGLU), each started and polled by the CPU over AXI4-Lite",
            "Every step waits for the one before it (the forward pass is a dependency chain), so the "
            "parts add up. Even with the CPU's work and all control at zero, the weights alone "
            "take ~2.5 ms."],
           size=11, bg=False, align="left")
    d.write()


# =============================================================================
# 7. Boot and a prompt
# =============================================================================
def boot():
    d = Diagram("boot", 1240, 560)
    d.text("t", 20, 12, 1200, 24, "Power-on to a story: who does what, in order (left to right)",
           size=13, bold=True, bg=False)
    lanes = [("Host PC", "gray", 60), ("FPGA, DDR3", "blue", 160), ("llm_core", "green", 260),
             ("CPU firmware", "purple", 360)]
    for i, (name, col, y) in enumerate(lanes):
        d.box(f"lane{i}", 20, y, 1200, 85, [""], color="none", rounded=False, dashed=True)
        d.box(f"ln{i}", 30, y + 12, 110, 60, [name], color=col)

    y0, y1, y2, y3 = 72, 172, 272, 372
    h = 62
    d.box("cfg", 170, y1, 180, h, ["Configure from QSPI", "x4 SPI, 33 MHz, compressed;",
                                   "firmware is in the bitstream"], color="blue", size=11)
    d.box("cal", 380, y1, 170, h, ["DDR3 calibration", "then the 225 MHz resets", "release"],
          color="blue", size=11)
    d.box("sd", 590, y2, 250, h, ["SD_LOAD, by itself", "card init, model image -> DDR3",
                                  "offset 0, checksum: ~7 s"], color="green", size=11)
    d.box("poll", 590, y3, 250, h, ["Starts from local memory", "prints a banner, polls STATUS:",
                                    "calibrated, SD_OK (or SD_ERR)"], color="purple", size=11)
    d.box("open", 870, y3, 160, h, ["Open the model", "check the header, tokenizer,",
                                    "RMSNorm -> local memory"], color="purple", size=10)
    d.box("gen", 1050, y3, 160, h, ["generate()", "forward() per token:", "~300 jobs each"],
          color="purple", size=11)
    d.box("chat", 870, y0, 160, h, ["host/chat.py", "waits for READY,", "sends a prompt line"],
          color="gray", size=11)
    d.box("out", 1050, y0, 160, h, ["Story text", "as it is generated;", "LLM DONE, READY"],
          color="gray", size=11)
    d.box("jobs", 1050, y2, 160, h, ["Hardware jobs", "matmul, attention,", "vector unit, DMA"],
          color="green", size=11)

    d.edge("cfg", "cal")
    d.edge("cal", "sd", exit=(1, 0.5), entry=(0, 0.3), points=[(570, 203), (570, 290.6)])
    d.edge("cfg", "poll", exit=(0.5, 1), entry=(0, 0.5), points=[(260, 403)],
           label="CPU starts at once", label_at=(390, 403), label_w=110)
    d.edge("sd", "poll", exit=(0.5, 1), entry=(0.5, 0), label="STATUS",
           label_at=(715, 353), label_w=46)
    d.edge("poll", "open")
    d.edge("open", "chat", exit=(0.5, 0), entry=(0.5, 1), dashed=True,
           label="READY (console)", label_at=(950, 250), label_w=100)
    d.edge("chat", "gen", exit=(1, 0.75), entry=(0.2, 0), points=[(1040, 118.5), (1040, 345),
                                                                  (1082, 345)],
           label=["prompt:", "CONSOLE_IN"], label_at=(1040, 170), label_w=72)
    d.edge("gen", "jobs", exit=(0.7, 0), entry=(0.7, 1), both=True)
    d.edge("gen", "out", exit=(0.9, 0), entry=(0.9, 1), points=[], dashed=True)
    d.text("txt", 1180, 230, 40, 16, "text", size=10)
    d.text("note", 20, 460, 1200, 90,
           ["The FPGA boots by itself from the QSPI flash (mode jumper on QSPI); "
            "make TOP=top_soc program_fw loads the same design over JTAG instead.",
            "The model image gets onto the microSD card once: host/image_tool.py upload --store "
            "(UART -> DDR3, then SD_STORE writes it to the card, header last).",
            "The CPU polls rather than waits for an interrupt; after READY it blocks on the console "
            "until a line arrives. /steps N and /trace PROMPT are the other commands."],
           size=11, bg=False, align="left")
    d.write()


# =============================================================================
# 8. Verification: the chain of bit-exact goldens
# =============================================================================
def verify():
    d = Diagram("verification", 1240, 560)
    d.text("t", 20, 12, 1200, 24,
           "Verification: every link is checked bit for bit, from llama2.c to the board",
           size=13, bold=True, bg=False)
    d.text("h1", 20, 45, 600, 20, "Whole model (every step's token and logits)", size=12,
           bold=True, bg=False, align="left")
    d.box("runq", 20, 80, 230, 90, ["llama2.c runq.c", "ref/runq_dump (x86, libm):",
                                    "logits of every step, the 43", "matmuls at one position"],
          color="gray", size=11)
    d.box("glibm", 330, 80, 250, 90, ["fw/model on x86, libm", "fw/build/golden_libm:",
                                      "runq.c's forward(), same float", "ops in the same order"],
          color="purple", size=11)
    d.box("gold", 660, 80, 250, 90, ["fw/model on x86, mathf.c", "fw/build/golden: own expf /",
                                     "sincosf, same bits on x86", "and on the board's FPU"],
          color="purple", size=11)
    d.box("board", 990, 80, 230, 90, ["The board", "fw/llm + the hardware units;",
                                      "host/chat.py --trace:", "T pos token logits-hash"],
          color="green", size=11)
    d.edge("runq", "glibm", label="identical", label_at=(290, 110), label_w=60)
    d.edge("glibm", "gold", label=["same code,", "own math"], label_at=(620, 125), label_w=66)
    d.edge("gold", "board", label="identical", label_at=(950, 110), label_w=60)
    d.text("n1", 20, 180, 1200, 40,
           ["Compiled without FMA contraction (-ffp-contract=off). The board's 256-step trace "
            "matches fw/build/golden line for line; any changed bit in any logit changes the hash."],
           size=11, bg=False, align="left")

    d.text("h2", 20, 235, 600, 20, "Each unit (Verilator, sim/run_*.sh)", size=12, bold=True,
           bg=False, align="left")
    d.box("ref", 20, 270, 230, 160, ["References (Python)", "ref/q80.py: Q8_0 matmul,",
                                     "quantize (bit-exact)", "sim/fp_ref.py: m_expf port",
                                     "sim/gen_*.py: vectors,", "numpy float32"],
          color="gray", size=11)
    tbs = [
        ("tb_fp32", ["tb_fp32", "add, mul, convert, divide, exp:", "100k vectors, rounding ties"]),
        ("tb_qmv", ["tb_qmv", "real runq.c matmuls incl. the", "32000-row classifier, stalls"]),
        ("tb_attn", ["tb_attn", "scores and weighted sums:", "head sizes 8-64, 1 and 2 K blocks"]),
        ("tb_vpu", ["tb_vpu", "QUANT, SILU, SOFTMAX, COPY", "against the firmware's math"]),
        ("tb_core", ["tb_llm_core", "all commands and CPU jobs; AXI", "memory and SD card models"]),
        ("tb_top", ["tb_top_qmv, tb_sd_ctrl", "through the UART pins;", "SDHC and SDSC cards"]),
    ]
    for i, (id_, lines) in enumerate(tbs):
        x = 330 + (i % 3) * 300
        y = 270 + (i // 3) * 90
        d.box(id_, x, y, 270, 70, lines, color="green", size=11)
    d.edge("ref", "tb_fp32", exit=(1, 35 / 160), entry=(0, 0.5), label="vectors",
           label_at=(290, 290), label_w=50)
    d.edge("ref", "tb_vpu", exit=(1, 125 / 160), entry=(0, 0.5))
    d.text("n2", 20, 450, 1200, 90,
           ["Every result is compared bit for bit (no tolerances). The float units round to "
            "nearest even like the CPU and flush subnormals, a range these values never reach.",
            "On the board, fw/build/bringup.elf checks the CPU, FPU, DDR3 and one engine matmul "
            "against the CPU; the whole-model trace above checks everything else."],
           size=11, bg=False, align="left")
    d.write()


if __name__ == "__main__":
    system()
    core()
    token()
    memory()
    clocks()
    budget()
    boot()
    verify()
