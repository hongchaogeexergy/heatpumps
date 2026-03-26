import os
from typing import Dict, Tuple
from graphviz import Digraph

# ---- colors (light & dark themes) ----
THEMES = {
    "light": dict(
        bg="#FFFFFF",
        node="#111111",
        A="#1f77b4",
        B="#2ca02c",
        C="#d62728",
        D="#ff7f0e",
        E="#9467bd",
        e="#8c564b",
        edge="#222222",
    ),
    "dark": dict(
        bg="#0f1115",
        node="#EAEAEA",
        A="#5dade2",
        B="#58d68d",
        C="#ec7063",
        D="#f39c12",
        E="#c39bd3",
        e="#b97750",
        edge="#DDDDDD",
    ),
}

COMP_SHAPES = {
    "Compressor": "doublecircle",
    "Pump": "circle",
    "Condenser": "box",
    "HeatExchanger": "box",
    "SimpleHeatExchanger": "box",
    "Valve": "triangle",
    "CycleCloser": "point",
    "Merge": "diamond",
    "Splitter": "diamond",
    "DropletSeparator": "hexagon",
    "Source": "oval",
    "Sink": "oval",
    "PowerSource": "oval",
    "PowerBus": "box3d",
    "Motor": "circle",
}

def _comp_shape(c) -> str:
    return COMP_SHAPES.get(c.__class__.__name__, "box")

def _edge_style(label: str) -> Tuple[str, str, str]:
    if not label:
        return ("edge", "solid", "normal")
    lead = label[0]
    if lead in "ABCD":
        return (lead, "solid", "normal")
    if lead == "E":
        return ("E", "bold", "vee")
    if lead == "e":
        return ("e", "dashed", "vee")
    return ("edge", "solid", "normal")

def _cluster_of(label: str) -> str:
    if not label:
        return "misc"
    return {
        "A": "A (Upper Refrigerant)",
        "B": "B (Heat Source)",
        "C": "C (Heat Sink)",
        "D": "D (Lower Refrigerant)",
        "E": "E (Electrical)",
        "e": "e (Mechanical)",
    }.get(label[0], "misc")

def build_graph_from_hp(hp, theme: str = "light") -> Digraph:
    if theme not in THEMES:
        theme = "light"
    palette = THEMES[theme]

    dot = Digraph("HeatPumpTopology", format="svg")
    dot.attr(bgcolor=palette["bg"], rankdir="LR", splines="polyline", overlap="false", nodesep="0.6")

    order = [
        "B (Heat Source)",
        "D (Lower Refrigerant)",
        "A (Upper Refrigerant)",
        "C (Heat Sink)",
        "E (Electrical)",
        "e (Mechanical)",
        "misc",
    ]

    clusters: Dict[str, Digraph] = {}
    for name in order:
        sg = Digraph(name=f"cluster_{name}")
        sg.attr(label=name, color="#888888", style="rounded")
        clusters[name] = sg

    # Collect nodes
    used_nodes = set()
    for lbl, conn in hp.conns.items():
        src = getattr(conn.source, "label", str(conn.source))
        dst = getattr(conn.target, "label", str(conn.target))
        used_nodes.add(src)
        used_nodes.add(dst)

    # Map nodes to clusters
    node2cluster = {n: "A (Upper Refrigerant)" for n in used_nodes}
    for lbl, conn in hp.conns.items():
        cluster = _cluster_of(lbl)
        s = getattr(conn.source, "label", str(conn.source))
        t = getattr(conn.target, "label", str(conn.target))
        for n in (s, t):
            if node2cluster.get(n, "misc") in ("A (Upper Refrigerant)", "misc"):
                node2cluster[n] = cluster

    # Add nodes
    for name in used_nodes:
        comp = next((c for c in hp.comps.values() if c.label == name), None)
        shape = _comp_shape(comp) if comp else "box"
        clusters[node2cluster[name]].node(
            name, label=name, shape=shape,
            color=palette["node"], fontcolor=palette["node"]
        )

    # Add edges
    for lbl, conn in hp.conns.items():
        src = getattr(conn.source, "label", str(conn.source))
        dst = getattr(conn.target, "label", str(conn.target))
        color_key, style, arrow = _edge_style(lbl)
        color = palette.get(color_key, palette["edge"])
        dot.edge(src, dst, label=lbl, color=color,
                 fontcolor=color, style=style, arrowhead=arrow)

    for key in order:
        dot.subgraph(clusters[key])

    return dot

def render_to_file(hp, path: str, theme: str = "light") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dot = build_graph_from_hp(hp, theme=theme)
    svg_bytes = dot.pipe(format="svg")
    with open(path, "wb") as f:
        f.write(svg_bytes)
    return path
