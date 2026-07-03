# ============================================================
# sfnn_architecture_diagram.py
# SFNN architecture diagram — West Bengal measles
# Updated to include all feature groups in final V1 model
# ============================================================

import pydot
import os

OUT_DIR = "/.../.../.../output/figures/"
os.makedirs(OUT_DIR, exist_ok=True)

# ── GRAPH SETUP ───────────────────────────────────────────────
graph = pydot.Dot(graph_type='digraph', rankdir='LR')
graph.set_ranksep('3.0')
graph.set_nodesep('0.3')

# ── INPUT LAYER ───────────────────────────────────────────────
def create_layer_input(name, label):
    subgraph = pydot.Cluster(
        name,
        label=label,
        style='filled',
        color='lightgrey',
        fillcolor='#F8F9FA',
        fontsize='18',
        fontname='Helvetica-Bold'
    )
    node_names = [
        "Incidence\nLags",
        "Large District\nIncidence\nLags",
        "Large District\nDistances",
        "Near District\nIncidence\nLags",
        "Near District\nDistances",
        "Susceptible\nLags",
        "Population",
        "Births",
        "Vaccination\nLags\n(MCV1)"
    ]
    nodes = []
    for n in node_names:
        node = pydot.Node(
            n,
            shape='circle',
            width='1.5',
            height='1.5',
            fontsize='13',
            fontname='Helvetica-Bold',
            style='filled',
            fillcolor='#D0E8FF',
            color='#4A90D9'
        )
        subgraph.add_node(node)
        nodes.append(node)
    graph.add_subgraph(subgraph)
    return nodes

# ── HIDDEN LAYERS ─────────────────────────────────────────────
def create_layer(name, num_nodes, label):
    subgraph = pydot.Cluster(
        name,
        label=label,
        style='filled',
        color='lightgrey',
        fillcolor='#F8F9FA',
        fontsize='18',
        fontname='Helvetica-Bold'
    )
    nodes = []
    for i in range(num_nodes):
        node = pydot.Node(
            f'{name}_{i}',
            label='',
            shape='circle',
            width='0.9',
            height='0.9',
            style='filled',
            fillcolor='#D4EDDA',
            color='#28A745'
        )
        subgraph.add_node(node)
        nodes.append(node)
    graph.add_subgraph(subgraph)
    return nodes

# ── OUTPUT LAYER ──────────────────────────────────────────────
def create_layer_output(name, label):
    subgraph = pydot.Cluster(
        name,
        label=label,
        style='filled',
        color='lightgrey',
        fillcolor='#F8F9FA',
        fontsize='18',
        fontname='Helvetica-Bold'
    )
    node = pydot.Node(
        "Incidence\nForecast",
        shape='circle',
        width='1.5',
        height='1.5',
        fontsize='13',
        fontname='Helvetica-Bold',
        style='filled',
        fillcolor='#FFE0B2',
        color='#E65100'
    )
    subgraph.add_node(node)
    graph.add_subgraph(subgraph)
    return [node]

# ── EDGES ─────────────────────────────────────────────────────
def add_edges(from_nodes, to_nodes):
    for f in from_nodes:
        for t in to_nodes:
            graph.add_edge(pydot.Edge(
                f, t,
                color='#AAAAAA',
                penwidth='0.5',
                arrowsize='0.5'
            ))

# ── BUILD GRAPH ───────────────────────────────────────────────
# Input: 9 feature groups
input_nodes  = create_layer_input('input',  'Input Layer\n(1,071 features)')

# Hidden layers — use k=1 best architecture (hidden_dim=64, layers=1)
# Show representative hidden layer with 6 nodes for clarity
hidden1_nodes = create_layer('hidden1', 6, 'Hidden Layer\n(64 units, ReLU)')

# Output
output_nodes  = create_layer_output('output', 'Output Layer')

# Connect
add_edges(input_nodes,  hidden1_nodes)
add_edges(hidden1_nodes, output_nodes)

# ── TITLE ANNOTATION ──────────────────────────────────────────
# Add subtitle node outside clusters
graph.set_graph_defaults(
    label='Spatial Feedforward Neural Network (SFNN)\n'
          'k-step ahead measles incidence forecasting',
    labelloc='t',
    labeljust='c',
    fontsize='22',
    fontname='Helvetica-Bold'
)

# ── SAVE ──────────────────────────────────────────────────────
graph.write_png(OUT_DIR + 'sfnn_architecture.png')
graph.write_pdf(OUT_DIR + 'sfnn_architecture.pdf')   # write_pdf not write_png

print("Saved:")
print(f"  {OUT_DIR}sfnn_architecture.png")
print(f"  {OUT_DIR}sfnn_architecture.pdf")

# ── ALSO PRODUCE A MULTI-LAYER VERSION (k=34 best: layers=1 too) ──
# Show the general architecture with variable hidden layers
graph2 = pydot.Dot(graph_type='digraph', rankdir='LR')
graph2.set_ranksep('2.5')
graph2.set_nodesep('0.3')
graph2.set_graph_defaults(
    label='SFNN Architecture — General Form\n'
          '(1-3 hidden layers depending on forecast horizon k)',
    labelloc='t',
    fontsize='20',
    fontname='Helvetica-Bold'
)

def add_subgraph(g, name, label, nodes_list, fill, border):
    sg = pydot.Cluster(name, label=label,
                       style='filled',
                       fillcolor='#F8F9FA',
                       color='lightgrey',
                       fontsize='15',
                       fontname='Helvetica-Bold')
    nodes = []
    for nm in nodes_list:
        nd = pydot.Node(nm, label=nm if nm else '',
                        shape='circle',
                        width='1.0', height='1.0',
                        fontsize='11',
                        fontname='Helvetica',
                        style='filled',
                        fillcolor=fill,
                        color=border)
        sg.add_node(nd)
        nodes.append(nd)
    g.add_subgraph(sg)
    return nodes

inp = add_subgraph(graph2, 'inp',
    'Input Layer\n(1,071 features)',
    ["Incidence\nLags", "Spatial\nLags",
     "Distances", "Susceptible\nLags",
     "Population", "Births",
     "MCV1\nLags"],
    '#D0E8FF', '#4A90D9')

h1 = add_subgraph(graph2, 'h1',
    'Hidden Layer 1\n(64-240 units)',
    [f'h1_{i}' for i in range(5)],
    '#D4EDDA', '#28A745')

h2 = add_subgraph(graph2, 'h2',
    'Hidden Layer 2\n(64-240 units, optional)',
    [f'h2_{i}' for i in range(5)],
    '#D4EDDA', '#28A745')

h3 = add_subgraph(graph2, 'h3',
    'Hidden Layer 3\n(64-240 units, optional)',
    [f'h3_{i}' for i in range(5)],
    '#D4EDDA', '#28A745')

out = add_subgraph(graph2, 'out',
    'Output Layer',
    ['Incidence\nForecast'],
    '#FFE0B2', '#E65100')

def add_edges2(g, fn, tn):
    for f in fn:
        for t in tn:
            g.add_edge(pydot.Edge(f, t,
                color='#AAAAAA', penwidth='0.4',
                arrowsize='0.4'))

add_edges2(graph2, inp, h1)
add_edges2(graph2, h1,  h2)
add_edges2(graph2, h2,  h3)
add_edges2(graph2, h3,  out)

graph2.write_png(OUT_DIR + 'sfnn_architecture_general.png')
graph2.write_pdf(OUT_DIR + 'sfnn_architecture_general.pdf')

print(f"  {OUT_DIR}sfnn_architecture_general.png")
print(f"  {OUT_DIR}sfnn_architecture_general.pdf")
