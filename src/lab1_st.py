import streamlit as st

import os
import time
from collections import deque
from math import atan2, cos, sin

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from scipy.integrate import solve_ivp

Image.MAX_IMAGE_PIXELS = None  # Вимикає перевірку на "decompression bomb"

from graph_draw import draw_node, draw_graph, draw_nodes


# MARK: st config
st.set_page_config(
    page_title="Lab 1 TNPS",
    page_icon= "🧪",
    layout="wide" # wide, centered
)

st.markdown("""
    <style>
    [data-testid=stSidebarUserContent] [data-testid=stExpanderDetails]:nth-of-type(1) [data-testid=stVerticalBlock]{
        gap: 0.15rem;
    }
    [data-testid=stExpanderDetails]:nth-of-type(1) [data-testid=stVerticalBlock]{
        gap: 0.5rem;
    }
    [data-testid=stExpanderDetails]:nth-of-type(1){
        padding: 0.75rem;
    }
    [data-testid=stMainBlockContainer]:nth-of-type(1){
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    
    .stAppDeployButton {
        display: none !important;
    }


    [data-testid=stHeader] {
        background: transparent !important;
        # opacity: 0.5;
        pointer-events: none;
    }
            
    [data-testid=stToolbar] {
        pointer-events: none;
    }

    [data-testid=stToolbar] > div {
        pointer-events: none !important;
    }

    [data-testid=stToolbar] > div * {
        pointer-events: auto;
    }


    </style>
    """,
    unsafe_allow_html=True
)



time_stats = {}







# MARK: Sidebar
with st.sidebar:
    # st.header("Параметри інтегрування")
    integration_time = st.slider(
        "Час інтегрування (t, сек)", 
        min_value=100, max_value=15000, value=2500, step=100
    )

    lam_min_value = 0.0
    lam_max_value = 0.1
    lam_format_str = "%.5f"
    lam_step = 0.0001


    st.write("Значення λ для блоків:")

    lam_b1_h = round(st.number_input("b1_h (1.1)", min_value=lam_min_value, max_value=lam_max_value, value=0.0005, format=lam_format_str, step=lam_step), 6)
    lam_b1_s = round(st.number_input("b1_s (1.2)", min_value=lam_min_value, max_value=lam_max_value, value=0.0005, format=lam_format_str, step=lam_step), 6)
    lam_b2   = round(st.number_input("b2 (2)",     min_value=lam_min_value, max_value=lam_max_value, value=0.0004, format=lam_format_str, step=lam_step), 6)
    lam_b3   = round(st.number_input("b3 (3)",     min_value=lam_min_value, max_value=lam_max_value, value=0.0003, format=lam_format_str, step=lam_step), 6)
    lam_b4   = round(st.number_input("b4 (4)",     min_value=lam_min_value, max_value=lam_max_value, value=0.00025, format=lam_format_str, step=lam_step), 6)
    lam_b5_h = round(st.number_input("b5_h (5.1)", min_value=lam_min_value, max_value=lam_max_value, value=0.0005, format=lam_format_str, step=lam_step), 6)
    lam_b5_s = round(st.number_input("b5_s (5.2)", min_value=lam_min_value, max_value=lam_max_value, value=0.0001, format=lam_format_str, step=lam_step), 6)


start_time = time.time()


# MARK: LogicBlock
class LogicBlock:
    def __init__(self, block_id, type, state=1, lam=0.0):
        self.id = block_id
        self.type = type
        self.state = state
        self.inputs = []
        self.outputs = []
        self.lam = lam  # інтенсивність відмов (λ)

    def connect_to(self, other_block):
        self.outputs.append(other_block.id)
        other_block.inputs.append(self.id)

    def set_state(self, new_state):
        self.state = new_state

    def is_working(self):
        return self.state == 1




# MARK: Blocks
blocks = {
    0:    LogicBlock(0, "Start"),

    1.1:  LogicBlock(1.1, "H", lam=lam_b1_h),
    1.2:  LogicBlock(1.2, "S", lam=lam_b1_s),

    2:    LogicBlock(2, "H", lam=lam_b2),
    3:    LogicBlock(3, "H", lam=lam_b3),
    4:    LogicBlock(4, "H", lam=lam_b4),

    5.1:  LogicBlock(5.1, "H", lam=lam_b5_h),
    5.2:  LogicBlock(5.2, "S", lam=lam_b5_s),

    6:    LogicBlock(6, "End"),
}

# залежності блоків, які не можуть бути одночасно справними/поламаними
mutual_exclusions = [
    (1.1, 1.2),
    (5.1, 5.2)
]

blocks[0].connect_to(blocks[1.1])
blocks[0].connect_to(blocks[2])
blocks[0].connect_to(blocks[5.1])

blocks[1.1].connect_to(blocks[1.2])
blocks[5.1].connect_to(blocks[5.2])

blocks[1.2].connect_to(blocks[3])
blocks[1.2].connect_to(blocks[4])

blocks[2].connect_to(blocks[3])
blocks[2].connect_to(blocks[4])

blocks[3].connect_to(blocks[6])
blocks[4].connect_to(blocks[6])
blocks[5.2].connect_to(blocks[6])




# blocks = {
#     0:    LogicBlock(0, "Start"),

#     1:  LogicBlock(1, "H", lam=lam_b1_h),
#     2:  LogicBlock(2, "H", lam=lam_b1_s),
#     3:  LogicBlock(3, "H", lam=lam_b2),

#     4:  LogicBlock(4, "End"),
# }

# # залежності блоків, які не можуть бути одночасно справними/поламаними
# mutual_exclusions = []

# blocks[0].connect_to(blocks[1])


# blocks[1].connect_to(blocks[2])
# blocks[1].connect_to(blocks[3])

# blocks[2].connect_to(blocks[4])
# blocks[3].connect_to(blocks[4])













# MARK: can_reach()
def can_reach(start_id, end_id, broken_ids):
    # Set block states
    for b in blocks.values():
        b.set_state(0 if b.id in broken_ids else 1)

    visited = set()
    stack = [start_id]

    step = 0
    while stack:
        current = stack.pop()
        # print(f"Step {step}: Current block {current}, state: {'working' if blocks[current].is_working() else 'broken'}, stack: {stack}")
        step += 1
        if current == end_id:
            # print(f"Reached end block {end_id}")
            return True
        visited.add(current)

        # If block is working — go further
        if blocks[current].is_working():
            # print(f"Block {current} is working, checking outputs: {blocks[current].outputs}")
            for nxt in blocks[current].outputs:
                if nxt not in visited:
                    # print(f"Adding block {nxt} to stack")
                    stack.append(nxt)
                else:
                    # print(f"Block {nxt} already visited")
                    pass

    # print(f"Cannot reach end block {end_id}")
    return False


# print("Input and output connections for each block:")
# for block in blocks.values():
#     print(f"Block {block.id} ({block.type}): inputs={block.inputs}, outputs={block.outputs}")

# print()
# print("-" * 40)
# print()

# # --- Examples ---
# broken_sets = [
#     {1.1, 2, 5.1},      # 1 and 5 are broken
#     {1.2, 2, 5.1},   # 3, 4 and 5 are broken

# ]

# for broken in broken_sets:
#     result = can_reach(0, 6, broken)
#     print(f"Breakdowns {broken}: {'Can reach' if result else 'Cannot reach'}")
    
#     print()
#     print("-" * 40)
#     print()





## MARK: GraphNode
class GraphNode:
    def __init__(self, row, idx, num, node_parent, block_states=None):
        self.row = row
        self.idx = idx
        self.num = num  # порядковий номер валідної ноди
        self.node_parent = node_parent
        # Якщо block_states не передано — всі блоки справні
        block_ids = [b.id for b in blocks.values() if b.type not in ["Start", "End"]]
        self.block_states = block_states if block_states is not None else {bid: 1 for bid in block_ids}
        # Зберігаємо типи блоків для зручності
        self.block_types = {bid: blocks[bid].type for bid in block_ids}
        self.block_lams = {bid: blocks[bid].lam for bid in block_ids}  # <--- додано

        self.inputs = []
        self.outputs = []

        self.duplicate_of = []

        self.locked_blocks = []

        self.is_dead = False

    def mark_duplicate_of(self, other_node):
        self.duplicate_of.append(other_node.idx)

    def connect_to(self, other_block):
        self.outputs.append(other_block.idx)
        other_block.inputs.append(self.idx)

    # def print_states(self):
    #     print(f"Node (idx={self.idx}, row={self.row}, parent={self.node_parent}):")
    #     for bid in sorted(self.block_states):
    #         block_type = self.block_types[bid]
    #         is_integer = isinstance(bid, int) or (isinstance(bid, float) and bid.is_integer())
    #         state = self.block_states[bid]

    #         # Формуємо рядок
    #         if block_type and  not is_integer:
    #             print(f"{int(bid) if bid == int(bid) else int(bid)}.{block_type} - {state} {"x" if bid in self.locked_blocks else ""}")
    #         else:
    #             print(f"{int(bid)}   - {state} ")

    def print_states_lines(self):
        lines = []
        lines.append(f"Node (idx={self.idx}, num={self.num}, row={self.row}, parent={self.node_parent}):")
        for bid in sorted(self.block_states):
            block_type = self.block_types[bid]
            is_integer = isinstance(bid, int) or (isinstance(bid, float) and bid.is_integer())
            state = self.block_states[bid]

            # Формуємо рядок
            if block_type and  not is_integer:
                lines.append(f"{int(bid) if bid == int(bid) else int(bid)}.{block_type} - {state} {"x" if bid in self.locked_blocks else ""}")
            else:
                lines.append(f"{int(bid)}   - {state} ")
        return lines





# MARK: generate_graph()
def generate_graph():
    valid_node_num = 1
    block_ids = [b.id for b in blocks.values() if b.type not in ["Start", "End"]]
    total_blocks = len(block_ids)

    output_lines = []  # для streamlit

    # Стартова нода: всі блоки справні
    idx = 1
    row = 1
    start_states = {bid: 1 for bid in block_ids}
    start_node = GraphNode(row, idx, valid_node_num, None, start_states)
    valid_node_num += 1

    for line in start_node.print_states_lines():
        output_lines.append(line)
    output_lines.append("-" * 30)

    queue = deque()
    queue.append(start_node)
    idx += 1

    # Всі унікальні стани, щоб не повторювати
    seen = {}
    seen[tuple(sorted(start_states.items()))] = 1

    # Список всіх валідних граф-нод
    valid_nodes = [start_node]
    node_by_idx = {start_node.idx: start_node}

    all_nodes = [start_node]  # <-- new list for all nodes

    while queue:
        current_node = queue.popleft()
        row = current_node.row
        current_states = current_node.block_states
        parent_idx = current_node.idx
        next_row = row + 1

        # Знаходимо всі блоки, які ще не поламані
        working_blocks = [bid for bid, state in current_states.items() if state == 1]
        # Якщо всі блоки вже поламані — далі не йдемо
        if not working_blocks:
            continue


        filtered_blocks = []
        for bid in working_blocks:
            skip = False
            for a, b in mutual_exclusions:
                if bid == a and current_states.get(b, 1) == 0:
                    skip = True
                if bid == b and current_states.get(a, 1) == 0:
                    skip = True
            if not skip:
                filtered_blocks.append(bid)


        # Для кожного блоку, який ще не поламаний, створюємо новий стан з додатковою поломкою
        for broken_bid in filtered_blocks:
            new_states = current_states.copy()
            new_states[broken_bid] = 0


            locked_blocks = []
            for a, b in mutual_exclusions:
                if (new_states.get(a, 1) == 0) and (new_states.get(b, 1) == 1):
                    locked_blocks.append(b)
                if (new_states.get(b, 1) == 0) and (new_states.get(a, 1) == 1):
                    locked_blocks.append(a)


            state_tuple = tuple(sorted(new_states.items()))
            duplicate_idx = seen.get(state_tuple)
            node = GraphNode(next_row, idx, valid_node_num, parent_idx, new_states)
            node.locked_blocks = locked_blocks

            for line in node.print_states_lines():
                output_lines.append(line)


            broken_ids = [bid for bid, state in new_states.items() if state == 0]
            can_reach_result = can_reach(0, max(blocks.keys()), broken_ids)
            if not can_reach_result:
                node.is_dead = True

            output_lines.append(f"Endpoint check: {"✅" if can_reach_result else "❌"}")

            if duplicate_idx is not None:
                # Connect parent to the original node (duplicate_idx) instead of the duplicate
                if parent_idx in node_by_idx and duplicate_idx in node_by_idx:
                    node_by_idx[parent_idx].connect_to(node_by_idx[duplicate_idx])
                    node.mark_duplicate_of(node_by_idx[duplicate_idx])

                output_lines.append(f"Node {idx}: is Duplicate of {duplicate_idx}")
                output_lines.append("-" * 30)

                all_nodes.append(node) # <-- add every node created

                idx += 1
                continue

            
            if not can_reach_result:
                output_lines.append("-" * 30)
                # node.is_dead = True

                # Додаємо зв'язок між parent та дочірньою нодою
                if parent_idx in node_by_idx:
                    node_by_idx[parent_idx].connect_to(node)

                valid_nodes.append(node)
                valid_node_num += 1
                node_by_idx[idx] = node

                seen[state_tuple] = idx

                all_nodes.append(node)  # <-- add every node created

                idx += 1

                continue

            all_nodes.append(node)  # <-- add every node created

            # Додаємо зв'язок між parent та дочірньою нодою
            if parent_idx in node_by_idx:
                node_by_idx[parent_idx].connect_to(node)

            valid_nodes.append(node)
            valid_node_num += 1
            node_by_idx[idx] = node

            output_lines.append("-" * 30)
            queue.append(node)  # Додається лише якщо не дублікат і є шлях до кінця
            seen[state_tuple] = idx
            idx += 1

    return valid_nodes, all_nodes, output_lines


valid_nodes, all_nodes, graph_output_lines = generate_graph()

st.write(f"{len(all_nodes)} nodes generated / {len(valid_nodes)} valid.")

# Вивід через streamlit
with st.expander("Graph Generation Output", expanded=False):
    st.code('\n'.join(graph_output_lines), language="None")










# MARK: print_valid_nodes_connections
def get_valid_nodes_connections_text(valid_nodes):
    lines = []
    for node in valid_nodes:
        lines.append(f"Node #{node.idx} num={node.num} (row={node.row}, parent={node.node_parent}):")
        lines.append("  Block states:")
        for bid in sorted(node.block_states):
            block_type = node.block_types[bid]
            state = node.block_states[bid]
            lock = "x" if bid in node.locked_blocks else ""
            if block_type and not (isinstance(bid, int) or (isinstance(bid, float) and bid.is_integer())):
                lines.append(f"    {int(bid) if bid == int(bid) else int(bid)}.{block_type} - {state} {lock}")
            else:
                lines.append(f"    {int(bid)}   - {state} {lock}")
        lines.append(f"  Inputs: {node.inputs}")
        for inp_num in node.inputs:
            inp_node = next((n for n in valid_nodes if n.idx == inp_num), None)
            if inp_node:
                diff = [(bid, node.block_states[bid], inp_node.block_states[bid]) 
                        for bid in node.block_states if node.block_states[bid] != inp_node.block_states[bid]]
                for bid, st1, st2 in diff:
                    lam = node.block_lams.get(bid, None)
                    lines.append(f"    Input from node {inp_num}: Block {bid} λ={lam} (state: {st2}→{st1})")
        lines.append(f"  Outputs: {node.outputs}")
        for out_num in node.outputs:
            out_node = next((n for n in valid_nodes if n.idx == out_num), None)
            if out_node:
                diff = [(bid, node.block_states[bid], out_node.block_states[bid]) 
                        for bid in node.block_states if node.block_states[bid] != out_node.block_states[bid]]
                for bid, st1, st2 in diff:
                    lam = node.block_lams.get(bid, None)
                    lines.append(f"    Output to node {out_num}: Block {bid} λ={lam} (state: {st1}→{st2})")
        lines.append("-" * 40)
    return "\n".join(lines)

# Для Streamlit:
with st.expander("Valid Nodes and Connections", expanded=False):
    st.code(get_valid_nodes_connections_text(valid_nodes), language="None")








# MARK: build_kolmogorov_equations_latex
def build_kolmogorov_equations_latex(nodes):
    eqs_latex = []
    for node in nodes:
        # Позначення для ймовірності перебування у стані node.idx
        P = f"P_{{{node.idx}}}(t)"
        dPdt = f"\\frac{{d{P}}}{{dt}}"
        # Вхідні переходи (з яких можна потрапити у node)
        in_terms = []
        for inp_num in node.inputs:
            inp_node = next((n for n in nodes if n.idx == inp_num), None)
            if inp_node:
                # Знаходимо блок, який змінився
                diff = [(bid, node.block_states[bid], inp_node.block_states[bid]) 
                        for bid in node.block_states if node.block_states[bid] != inp_node.block_states[bid]]
                for bid, st1, st2 in diff:
                    lam = node.block_lams.get(bid, None)
                    in_terms.append(f"{lam} {P.replace(str(node.idx), str(inp_num))}")
        # Вихідні переходи (з яких можна піти з node)
        out_terms = []
        for out_num in node.outputs:
            out_node = next((n for n in nodes if n.idx == out_num), None)
            if out_node:
                diff = [(bid, node.block_states[bid], out_node.block_states[bid]) 
                        for bid in node.block_states if node.block_states[bid] != out_node.block_states[bid]]
                for bid, st1, st2 in diff:
                    lam = node.block_lams.get(bid, None)
                    out_terms.append(f"{lam} {P}")
        # Формуємо рівняння
        rhs = ""
        if in_terms:
            rhs += " + ".join(in_terms)
        if out_terms:
            if rhs:
                rhs += " - "
                rhs += " - ".join(out_terms)
            else:
                rhs += "- "
                rhs += " - ".join([f"{term}" for term in out_terms])
        if not rhs:
            rhs = "0"
        eqs_latex.append(f"{dPdt} = {rhs}")

    return eqs_latex

eqs_latex = build_kolmogorov_equations_latex(valid_nodes)


with st.expander("Equations", expanded=False):
    for idx, eq_latex in enumerate(eqs_latex, 1):
        st.latex(f"{idx}.\\quad {eq_latex}", width="content")





# MARK: kolmogorov_rhs
def kolmogorov_rhs(t, P, valid_nodes):

    dPdt = np.zeros_like(P) # dPdt — масив похідних ймовірностей для кожного стану (ноди)

    for i, node in enumerate(valid_nodes):

        in_terms = []
        out_terms = []

        # Перебираємо всі вхідні ноди (звідки можна потрапити у поточну)
        for inp_num in node.inputs:
            # Знаходимо індекс вхідної ноди у списку valid_nodes
            inp_idx = next((j for j, n in enumerate(valid_nodes) if n.idx == inp_num), None)
            if inp_idx is not None:
                # Визначаємо, які блоки змінили стан при переході з inp_node у node
                diff = [(bid, node.block_states[bid], valid_nodes[inp_idx].block_states[bid]) 
                        for bid in node.block_states 
                        if node.block_states[bid] != valid_nodes[inp_idx].block_states[bid]]
                # print(diff)
                
                # Для кожного такого блоку додаємо доданок у in_terms
                for bid, st1, st2 in diff:
                    lam = node.block_lams.get(bid, None)
                    # if lam:
                    if lam is not None:
                        in_terms.append(lam * P[inp_idx])
                    else:
                        raise ValueError(f"Lambda not found for block {bid} in node {node.idx}")

        # Перебираємо всі вихідні ноди (куди можна перейти з поточної)
        for out_num in node.outputs:
            # Знаходимо індекс вихідної ноди у списку valid_nodes
            out_idx = next((j for j, n in enumerate(valid_nodes) if n.idx == out_num), None)
            if out_idx is not None:
                # Визначаємо, які блоки змінили стан при переході з node у out_node
                diff = [(bid, node.block_states[bid], valid_nodes[out_idx].block_states[bid]) 
                        for bid in node.block_states 
                        if node.block_states[bid] != valid_nodes[out_idx].block_states[bid]]
                
                # Для кожного такого блоку додаємо доданок у out_terms
                for bid, st1, st2 in diff:
                    lam = node.block_lams.get(bid, None)
                    # if lam:
                    if lam is not None:
                        out_terms.append(lam * P[i])
                    else:
                        raise ValueError(f"Lambda not found for block {bid} in node {node.idx}")

        dPdt[i] = sum(in_terms) - sum(out_terms)

    return dPdt

# MARK: solve_kolmogorov
def solve_kolmogorov(valid_nodes, t_span, P0=None, t_eval=None):
    n = len(valid_nodes)
    if P0 is None:
        P0 = np.zeros(n)
        P0[0] = 1.0  # Початковий стан: вся ймовірність у першій ноді
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1500)
        # t_eval = np.linspace(t_span[0], t_span[1], t_span[1] + 1)
    sol = solve_ivp(
        fun=lambda t, P: kolmogorov_rhs(t, P, valid_nodes),
        t_span=t_span,
        y0=P0,
        t_eval=t_eval,
        # method='RK45'
        # rtol=1e-7, atol=1e-9
        rtol=1e-9, # Відносна похибка інтегратора
        atol=1e-12 # Абсолютна похибка інтегратора
    )
    return sol







P0 = None

ode_start_time = time.time()

sol = solve_kolmogorov(valid_nodes, t_span=(0, integration_time), P0=P0)

time_stats['solve_kolmogorov'] = time.time() - ode_start_time







# MARK: Графік ймовірностей станів
state_probs_chart_start_time = time.time()
STATE_LABELS = [str(node.idx) for node in valid_nodes]

fig = go.Figure()
for k, label in enumerate(STATE_LABELS):
    fig.add_trace(go.Scatter(
        x=sol.t, y=sol.y[k],
        mode='lines',
        name=f'P{label}(t)',
        line=dict(width=1)
    ))

fig.update_layout(
    xaxis_title='t',
    yaxis_title='Ймовірність',
    # title='Графік ймовірностей станів',
    legend=dict(font=dict(size=12), orientation="h"),
    height=625,
    dragmode='pan',
    # dragmode=False,
    margin=dict(l=0, r=0, t=30, b=0)
)


# with st.container(border=True):
with st.expander(f"Графік ймовірностей станів", 
                 expanded=True):
    st.plotly_chart(fig, use_container_width=True, 
                        config={
                            # "scrollZoom": True, 
                            "displayModeBar": True
                            })





# MARK: Графік працездатності системи
# Маска працездатних станів
alive_mask = np.array([not node.is_dead for node in valid_nodes])
# st.write(sol.y)

# Сума ймовірностей працездатних станів для кожної точки часу
alive_probs_sum = np.sum(sol.y[alive_mask, :], axis=0)  # shape: (len(sol.t),)
alive_probs_sum = np.round(alive_probs_sum, 6)  # Додаємо округлення
# st.write(alive_probs_sum)
# st.text(len(alive_probs_sum))

fig_alive = go.Figure()
fig_alive.add_trace(go.Scatter(
    x=sol.t,
    y=alive_probs_sum,
    mode='lines',
    name='Сума ймовірностей працездатних станів',
    line=dict(width=3, color='#2ecc40')
))
fig_alive.update_layout(
    xaxis_title='t',
    yaxis_title='Сума ймовірностей працездатних станів',
    title='Графік працездатності системи',
    height=400,
    margin=dict(l=0, r=0, t=30, b=0),
    dragmode=False,
)

# Середній час до відмови (MTTF)
mttf = np.trapezoid(alive_probs_sum, sol.t)

with st.expander(f"Середнє значення тривалості роботи до відмови `{mttf:.6f}`", 
                 expanded=True):
    st.plotly_chart(fig_alive, use_container_width=True)





# MARK: Probability Table and Bar Chart at Specific Time
# with st.container(border=True):
with st.expander(f"Розподіл ймовірностей по станах", 
                 expanded=True):
    t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns([1, 1, 1, 1, 1])

    with t_col1:
        query_time = st.number_input(
            "Час t для перегляду ймовірностей:",
            # min_value=float(sol.t[0]),
            # max_value=float(sol.t[-1]),
            # value=float(sol.t[0]),
            # step=1.0,
            # format="%.2f"

            min_value=int(sol.t[0]),
            max_value=int(sol.t[-1]),
            value=int(sol.t[0]),
            step=10,
            format="%d"
        )

    # Знаходимо індекс часу, який найближчий до введеного користувачем query_time
    idx = np.abs(sol.t - query_time).argmin()
    # Вибираємо ймовірності для всіх станів у цей момент часу
    probs = sol.y[:, idx]

    with t_col2:
        st.markdown(f"t = {sol.t[idx]:.2f}")


    with t_col3:
        st.markdown(f"Сума ймовірностей: {np.sum(probs):.12f}")


    alive_sum = np.sum([p for i, p in enumerate(probs) if not valid_nodes[i].is_dead]) # Ймовірність безвідмовної роботи
    dead_sum = np.sum([p for i, p in enumerate(probs) if valid_nodes[i].is_dead])

    alive_count = sum(1 for node in valid_nodes if not node.is_dead)
    dead_count = sum(1 for node in valid_nodes if node.is_dead)
    st.write(f"Працездатних станів: {alive_count} / Відмовних станів: {dead_count}")

    with t_col4:
        st.write(f"Ймовірність безвідмовної роботи: {alive_sum:.12f} / `{alive_sum:.2%}`")
    with t_col5:
        st.markdown(f"Ймовірність відмови: {dead_sum:.12f} / `{dead_sum:.2%}`")

    p_col_1, p_col_2 = st.columns([1, 3])

    with p_col_1:
        
        prob_table = []
        dead_mask = []

        for i, p in enumerate(probs):
            prob_table.append({
                "idx": f"P{valid_nodes[i].idx}(t)",
                "num": f"{valid_nodes[i].num}",
                "Ймовірність": f"{p:.12f}",
                "%" : f"{p:.2%}"
            })
            dead_mask.append(valid_nodes[i].is_dead)

        df = pd.DataFrame(prob_table)

        def highlight_row(row):
            if dead_mask[row.name]:
                return ['background-color: #651d1d; color: white' for _ in row.index]
            else:
                return ['background-color: #334f65; color: white' for _ in row.index]

        st.dataframe(df.style.apply(highlight_row, axis=1), hide_index=True, height=475)
        


    with p_col_2:
        # Підготовка даних для барчарту
        bar_colors = ['#83c9ff' if not valid_nodes[i].is_dead else '#ff4b4b' for i in range(len(probs))]
        # bar_names = [f"P{valid_nodes[i].idx}" for i in range(len(probs))]
        bar_names = [f"P{valid_nodes[i].num}" for i in range(len(probs))]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=bar_names,
            y=probs,
            marker_color=bar_colors
        ))
        fig_bar.update_layout(
            title="Розподіл ймовірностей по станах (сині — безвідмовні, червоні — відмовні)",
            xaxis_title="Стан",
            yaxis_title="Ймовірність",
            height=475,
            margin=dict(l=0, r=0, t=25, b=0),
            dragmode=False,
            # dragmode='pan',
            # xaxis=dict(tickangle=-90, tickfont=dict(size=10))
        )

        st.plotly_chart(fig_bar, 
                        use_container_width=True,
                        config={"scrollZoom": False, 
                                "displayModeBar": True})


time_stats['state_probs_chart'] = time.time() - state_probs_chart_start_time





# MARK: draw graph
with st.expander(f"Draw Graph", expanded=True):
    if st.button("Generate and Draw Graph"):
        with st.spinner("Wait for it...", show_time=True):

            os.makedirs("images/", exist_ok=True)

            # Засікання часу для малювання графа
            t0_graph = time.time()
            img_graph = draw_graph(valid_nodes)
            time_stats['draw_graph'] = time.time() - t0_graph
            # with st.expander("Graph of Valid Nodes", expanded=True):
            #     st.image(img_graph, caption="Graph of Valid Nodes", width="stretch")
            # Засікання часу для малювання всіх нодів
            t0_all = time.time()
            img_all = draw_nodes(all_nodes)
            time_stats['draw_all_nodes'] = time.time() - t0_all
            # with st.expander("All Generated Nodes", expanded=True):
            #     st.image(img_all, caption="All Generated Nodes", width="stretch")

            t0_save_images = time.time()
            img_graph.save("images/graph.png")
            img_all.save("images/all_nodes.png")
            time_stats['save_images'] = time.time() - t0_save_images

            st.success("Images saved to disk.")

            # img_graph.show()
            # img_all.show()

            # # Зберегти у 2 рази менший розмір
            # img_graph_2x = img_graph.resize((img_graph.width // 2, img_graph.height // 2), Image.LANCZOS)
            # img_graph_2x.save("graph_2x.png")
            # img_all_2x = img_all.resize((img_all.width // 2, img_all.height // 2), Image.LANCZOS)
            # img_all_2x.save("all_nodes_2x.png")











# MARK: Time stats
with st.sidebar:
    st.divider()
    time_stats['total'] = time.time() - start_time
    # display time stats
    for key, val in time_stats.items():
        st.write(f"**{key}**: {val:.2f} sec")




