import streamlit as st

import os
import time
from collections import deque
from math import atan2, cos, sin

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from PIL import Image, ImageDraw, ImageFont
from scipy.integrate import solve_ivp

Image.MAX_IMAGE_PIXELS = None  # Disables "decompression bomb" check

from graph_draw import draw_node, draw_graph, draw_nodes
from models import LogicBlock, can_reach, GraphNode

from typing import List


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
start_time = time.time()


# MARK: Sidebar
with st.sidebar:

    s_m_col1, s_m_col2 = st.columns(2)

    scheme = s_m_col1.radio(
        "Select block scheme:",
        options=["Simple Example", 
                 "Variant 5",
                #  "Variant 4"
                 ],
        index=0
    )

    enable_repair = s_m_col2.toggle("Enable repair", value=False)


    with st.container(border=True):
        # st.subheader("Integration Settings")

        integration_time = st.slider(
            "Integration time (t, sec)", 
            min_value=100, max_value=15000, value=2500, step=100
        )

        n_points = st.number_input(
            "Number of integration points (t_eval)", 
            min_value=100, max_value=20000, value=1500, step=100
        )

        pr_col1, pr_col2 = st.columns(2)

        rtol = pr_col1.number_input(
            "Integrator relative tolerance (rtol)", 
            min_value=1e-12, max_value=1e-3, value=1e-9, format="%.1e", disabled=True
        )

        atol = pr_col2.number_input(
            "Integrator absolute tolerance (atol)", 
            min_value=1e-15, max_value=1e-6, value=1e-12, format="%.1e", disabled=True
        )

    with st.expander("Block values", expanded=True):
        input_col1, input_col2 = st.columns(2)

    lam_min_value = 0.0
    lam_max_value = 0.1
    lam_format_str = "%.5f"
    lam_step = 0.0001


    with input_col1:
        # st.write("λ values for blocks:")

        lam_b1_h = round(st.number_input("λ b1_h (1.1)", min_value=lam_min_value, max_value=lam_max_value, value=0.0005, format=lam_format_str, step=lam_step), 6)
        lam_b1_s = round(st.number_input("λ b1_s (1.2)", min_value=lam_min_value, max_value=lam_max_value, value=0.0005, format=lam_format_str, step=lam_step), 6)
        lam_b2   = round(st.number_input("λ b2 (2)",     min_value=lam_min_value, max_value=lam_max_value, value=0.0004, format=lam_format_str, step=lam_step), 6)
        lam_b3   = round(st.number_input("λ b3 (3)",     min_value=lam_min_value, max_value=lam_max_value, value=0.0003, format=lam_format_str, step=lam_step), 6)
        lam_b4   = round(st.number_input("λ b4 (4)",     min_value=lam_min_value, max_value=lam_max_value, value=0.00025, format=lam_format_str, step=lam_step), 6)
        lam_b5_h = round(st.number_input("λ b5_h (5.1)", min_value=lam_min_value, max_value=lam_max_value, value=0.0005, format=lam_format_str, step=lam_step), 6)
        lam_b5_s = round(st.number_input("λ b5_s (5.2)", min_value=lam_min_value, max_value=lam_max_value, value=0.0001, format=lam_format_str, step=lam_step), 6)

    with input_col2:
        # st.write("μ values for blocks:")

        mu_min_value = 0.0
        mu_max_value = 0.1
        mu_format_str = "%.5f"
        mu_step = 0.0001

        mu_b1_h = round(st.number_input("μ b1_h (1.1)", min_value=mu_min_value, max_value=mu_max_value, value=0.005, format=mu_format_str, step=mu_step, disabled=not enable_repair), 6)
        mu_b1_s = round(st.number_input("μ b1_s (1.2)", min_value=mu_min_value, max_value=mu_max_value, value=0.005, format=mu_format_str, step=mu_step, disabled=not enable_repair), 6)
        mu_b2   = round(st.number_input("μ b2 (2)",     min_value=mu_min_value, max_value=mu_max_value, value=0.004, format=mu_format_str, step=mu_step, disabled=not enable_repair), 6)
        mu_b3   = round(st.number_input("μ b3 (3)",     min_value=mu_min_value, max_value=mu_max_value, value=0.003, format=mu_format_str, step=mu_step, disabled=not enable_repair), 6)
        mu_b4   = round(st.number_input("μ b4 (4)",     min_value=mu_min_value, max_value=mu_max_value, value=0.0025, format=mu_format_str, step=mu_step, disabled=not enable_repair), 6)
        mu_b5_h = round(st.number_input("μ b5_h (5.1)", min_value=mu_min_value, max_value=mu_max_value, value=0.005, format=mu_format_str, step=mu_step, disabled=not enable_repair), 6)
        mu_b5_s = round(st.number_input("μ b5_s (5.2)", min_value=mu_min_value, max_value=mu_max_value, value=0.001, format=mu_format_str, step=mu_step, disabled=not enable_repair), 6)




# MARK: Blocks

if scheme == "Simple Example":
    blocks = {
        0:    LogicBlock(0, "Start"),

        1:  LogicBlock(1, "S", lam=lam_b1_h, mu=mu_b1_h),
        2:  LogicBlock(2, "H", lam=lam_b1_s, mu=mu_b1_s),
        3:  LogicBlock(3, "H", lam=lam_b2, mu=mu_b2),

        4:  LogicBlock(4, "End"),
    }

    # dependencies of blocks that cannot be simultaneously working/faulty
    mutual_exclusions = []

    blocks[0].connect_to(blocks[1])


    blocks[1].connect_to(blocks[2])
    blocks[1].connect_to(blocks[3])

    blocks[2].connect_to(blocks[4])
    blocks[3].connect_to(blocks[4])

elif scheme == "Variant 5":
    blocks = {
        0:    LogicBlock(0, "Start"),

        1.1:  LogicBlock(1.1, "H", lam=lam_b1_h, mu=mu_b1_h),
        1.2:  LogicBlock(1.2, "S", lam=lam_b1_s, mu=mu_b1_s),

        2:    LogicBlock(2, "H", lam=lam_b2, mu=mu_b2),
        3:    LogicBlock(3, "H", lam=lam_b3, mu=mu_b3),
        4:    LogicBlock(4, "H", lam=lam_b4, mu=mu_b4),

        5.1:  LogicBlock(5.1, "H", lam=lam_b5_h, mu=mu_b5_h),
        5.2:  LogicBlock(5.2, "S", lam=lam_b5_s, mu=mu_b5_s),

        6:    LogicBlock(6, "End"),
    }

    # dependencies of blocks that cannot be simultaneously working/faulty
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

elif scheme == "Variant 4":
    blocks = {
        0:      LogicBlock(0, "Start"),

        1.1:    LogicBlock(1.1, "H", lam=lam_b1_h, mu=mu_b1_h),
        1.2:    LogicBlock(1.2, "S", lam=lam_b1_s, mu=mu_b1_s),
        2:      LogicBlock(2, "H", lam=lam_b2, mu=mu_b2),
        3.1:    LogicBlock(3.1, "H", lam=lam_b3, mu=mu_b3),
        3.2:    LogicBlock(3.2, "S", lam=lam_b4, mu=mu_b4),
        4:      LogicBlock(4, "H", lam=lam_b5_h, mu=mu_b5_h),
        5:      LogicBlock(5, "H", lam=lam_b5_s, mu=mu_b5_s),

        6:      LogicBlock(6, "End"),
    }
    mutual_exclusions = [
        (1.1, 1.2),
        (3.1, 3.2)
    ]

    # Зв'язки згідно з малюнком
    blocks[0].connect_to(blocks[1.1])
    blocks[0].connect_to(blocks[3.1])

    blocks[1.1].connect_to(blocks[1.2])
    blocks[3.1].connect_to(blocks[3.2])

    blocks[1.2].connect_to(blocks[2])

    blocks[2].connect_to(blocks[4])
    blocks[2].connect_to(blocks[5])
    blocks[3.2].connect_to(blocks[4])
    blocks[3.2].connect_to(blocks[5])

    blocks[4].connect_to(blocks[6])
    blocks[5].connect_to(blocks[6])


# MARK: generate_graph()
def generate_graph(enable_repair=False):
    valid_node_num = 1
    block_ids = [b.id for b in blocks.values() if b.type not in ["Start", "End"]]
    block_types = {bid: blocks[bid].type for bid in block_ids}
    block_lams = {bid: blocks[bid].lam for bid in block_ids}
    block_mus = {bid: blocks[bid].mu for bid in block_ids}
    # total_blocks = len(block_ids)

    block_repair_types = {bid: blocks[bid].type for bid in block_ids}

    output_lines = []  # for streamlit

    # Start node: all blocks are working
    idx = 1
    row = 1
    start_states = {bid: 1 for bid in block_ids}
    start_node = GraphNode(row, idx, valid_node_num, None, start_states, block_ids, block_types, block_lams, block_mus)
    valid_node_num += 1

    for line in start_node.print_states_lines():
        output_lines.append(line)
    output_lines.append("-" * 30)

    queue = deque()
    queue.append(start_node)
    idx += 1

    # All unique states to avoid duplicates
    # seen = {}
    # seen[tuple(sorted(start_states.items()))] = 1
    seen = {tuple(sorted(start_states.items())): 1}

    # Init lists
    valid_nodes = [start_node]
    node_by_idx = {start_node.idx: start_node}
    all_nodes = [start_node]

    while queue:
        current_node = queue.popleft()
        row = current_node.row
        current_states = current_node.block_states
        parent_idx = current_node.idx
        next_row = row + 1



        # --- 0. Перевірка: чи працює система в ПОТОЧНОМУ стані? ---
        # Нам це треба знати, щоб вирішити, чи можна ламати далі.
        current_broken_ids = [bid for bid, state in current_states.items() if state == 0 or state < 0]
        is_current_system_alive = can_reach(blocks, 0, max(blocks.keys()), current_broken_ids)

        # ==========================================
        # НОВА ЛОГІКА: ПЕРЕВІРКА НА "ВІЧНУ СМЕРТЬ"
        # ==========================================
        if not is_current_system_alive:
            # 1. Знаходимо блоки, які вже НЕМОЖЛИВО полагодити (досягли ліміту версій)
            irreparable_broken_ids = []
            for bid in current_broken_ids:
                val = current_states[bid]
                rtype = block_types.get(bid, 'S')
                # Якщо S-тип і стан -2 (або менше), ремонт неможливий
                if rtype == 'S' and abs(val) >= 2:
                    irreparable_broken_ids.append(bid)
            
            # 2. Перевіряємо: якщо ми полагодимо ВСЕ, крім вічно зламаних, чи запрацює система?
            # Тобто, чи є irreparable_broken_ids критичним набором?
            can_ever_recover = can_reach(blocks, 0, max(blocks.keys()), irreparable_broken_ids)

            # 3. Якщо надії немає — зупиняємо цю гілку.
            if not can_ever_recover:
                output_lines.append(f"Node {idx}: System is PERMANENTLY DEAD (Unrecoverable). Stopping branch.")
                output_lines.append("-" * 30)
                # Важливо: ми не додаємо нічого в queue і переходимо до наступної ноди в черзі
                current_node.is_permanently_dead = True
                continue 
        # ==========================================
        

        # --- 1. Збираємо всі можливі переходи (Transitions) ---
        transitions = [] # Список кортежів: (block_id, new_state_value, action_type)

        if is_current_system_alive:  # <--- ДОДАНО ЦЮ УМОВУ
        # Failure transitions
        # Перевірка: якщо блок у парі виключення вже зламаний, другий не можна ламати.
            # working_blocks = [bid for bid, state in current_states.items() if state == 1]
            # Шукаємо блоки, які зараз працюють (val > 0)
            working_blocks = [bid for bid, val in current_states.items() if val > 0]

        # filtered_blocks = []

            # До створення нового вузла.
            # Якщо заблоковано кимось — не додаємо
            for bid in working_blocks:
                # Перевіряємо, чи є хоч одна умова, яка блокує цей bid
                is_blocked = any(
                    (bid == a and current_states.get(b, 1) <= 0) or 
                    (bid == b and current_states.get(a, 1) <= 0)
                    for a, b in mutual_exclusions
                )
                if not is_blocked:
                    # filtered_blocks.append(bid)

                    current_val = current_states[bid]
                    rtype = block_types.get(bid, 'S')

                    if rtype == 'H':
                        new_val = 0 # Стандартний злам
                    else: # rtype == 'S'
                        new_val = -1 * current_val # 1 -> -1, 2 -> -2 (зберігаємо історію)

                    # transitions.append((bid, 0, "fail"))
                    transitions.append((bid, new_val, "fail"))

        # Б. ЛОГІКА РЕМОНТУ (0 -> 1)  <-- НОВИЙ ФУНКЦІОНАЛ
        if enable_repair:
            broken_blocks = [bid for bid, state in current_states.items() if state == 0 or state < 0]
            for bid in broken_blocks:
                current_val = current_states[bid]
                rtype = block_types.get(bid, 'S')
                new_val = None

                if rtype == 'H':
                    new_val = 1 # Завжди відновлюється в 1
                elif rtype == 'S':
                    # Логіка S: відновлюється лише 1 раз (тобто перехід 1 -> 2)
                    # Якщо current_val == -1 (зламалась 1-ша версія), то ремонтуємо в 2
                    # Якщо current_val == -2 (зламалась 2-га версія), ремонту більше немає
                    version_broken = abs(current_val)
                    if version_broken < 2: # Ліміт ремонтів (тут 1 ремонт, отже макс версія 2)
                        new_val = version_broken + 1

                if new_val is not None:
                    # transitions.append((bid, 1, "repair"))
                    transitions.append((bid, new_val, "repair"))


        # For each block that is not yet broken, create a new state with an additional failure
        # for broken_bid in filtered_blocks:
        for target_bid, target_val, action_type in transitions:
            new_states = current_states.copy()
            # new_states[broken_bid] = 0
            new_states[target_bid] = target_val

            # Визначаємо "заблоковані" блоки — ті, які не можна ламати через взаємні виключення.
            # Це потрібно для коректного відображення стану вузла.
            # "вішає ярлик" на блоки, які залишилися цілими.
            locked_blocks = []
            for a, b in mutual_exclusions:
                if (new_states.get(a, 1) <= 0) and (new_states.get(b, 1) > 0):
                    locked_blocks.append(b)
                if (new_states.get(b, 1) <= 0) and (new_states.get(a, 1) > 0):
                    locked_blocks.append(a)


            state_tuple = tuple(sorted(new_states.items()))
            duplicate_idx = seen.get(state_tuple)
            
            node = GraphNode(next_row, idx, valid_node_num, parent_idx, new_states, block_ids, block_types, block_lams, block_mus)
            node.locked_blocks = locked_blocks

            # node.action_type = action_type # Можна зберегти тип дії для розмальовки стрілочок (червона/зелена)
            # node.changed_block = target_bid # Який блок змінив стан

            for line in node.print_states_lines():
                output_lines.append(line)
            output_lines.append(f"Action: {action_type.upper()} block {target_bid}")

            # broken_ids = [bid for bid, state in new_states.items() if state == 0] 
            broken_ids = [bid for bid, state in new_states.items() if state == 0 or state < 0]
            # broken_ids - список зламаних блоків у новому стані
            can_reach_result = can_reach(blocks, 0, max(blocks.keys()), broken_ids)
            if not can_reach_result:
                node.is_dead = True

            output_lines.append(f"Endpoint check: {'✅' if can_reach_result else '❌'}")


            # --- LOGIC BRANCHING ---

            if duplicate_idx is not None:
                # --- CASE 1: DUPLICATE ---
                if parent_idx in node_by_idx and duplicate_idx in node_by_idx:
                    if action_type == "repair":
                        node_by_idx[parent_idx].connect_repair_to(node_by_idx[duplicate_idx])
                    else:
                        node_by_idx[parent_idx].connect_to(node_by_idx[duplicate_idx])
                    node.mark_duplicate_of(node_by_idx[duplicate_idx])
                
                output_lines.append(f"Node {idx}: is Duplicate of {duplicate_idx}")
                
            else:
                # --- CASE 2: NEW UNIQUE NODE (Dead or Alive) ---
                valid_nodes.append(node)
                # Only increment valid_node_num for unique nodes (based on original logic logic)
                valid_node_num += 1 
                node_by_idx[idx] = node
                seen[state_tuple] = idx
                
                # Connect parent to this new node
                if parent_idx in node_by_idx:
                    if action_type == "repair":
                        node_by_idx[parent_idx].connect_repair_to(node)
                    else:
                        node_by_idx[parent_idx].connect_to(node)
                
                # If system is still working, continue exploring (add to queue)
                # if can_reach_result:
                #     queue.append(node)

                # Додаємо в чергу, навіть якщо стан "мертвий"?
                # Зазвичай так, бо з мертвого стану можна "полагодитись" назад у живий.
                # Але якщо ви хочете зупиняти симуляцію при відмові системи, то лишіть `if can_reach_result`.
                # Для повного графа станів краще додавати завжди:
                queue.append(node)

            # --- FINALIZING ITERATION ---
            all_nodes.append(node)   # Add to all_nodes list regardless of type
            output_lines.append("-" * 30)
            idx += 1

    return valid_nodes, all_nodes, output_lines


valid_nodes, all_nodes, output_lines = generate_graph(enable_repair=enable_repair)



# Робочі вузли
working_nodes = [node for node in valid_nodes if not node.is_dead and not node.is_permanently_dead]
# Поламані вузли (але не вічна смерть)
failed_nodes = [node for node in valid_nodes if node.is_dead and not node.is_permanently_dead]
# Вузли у "вічній смерті"
perma_dead_nodes = [node for node in valid_nodes if node.is_permanently_dead]

st.markdown(f"##### `Generated (All): {len(all_nodes)}` `Valid: {len(valid_nodes)}` `Working: {len(working_nodes)}` `Failed: {len(failed_nodes)}` `Permanent Fail: {len(perma_dead_nodes)}`")

# Output via streamlit
tab_gen_conn, tab_eq, tab_charts, tab_graph = st.tabs([
    "Logs", 
    "Equations", 
    "Charts & Reliability", 
    "Graph Visualization" 
    ], 
    default="Charts & Reliability"
)





# MARK: get_valid_nodes_connections_text
def get_valid_nodes_connections_text(nodes: List[GraphNode]):
    lines = []
    for node in nodes:
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
            inp_node = next((n for n in nodes if n.idx == inp_num), None)
            if inp_node:
                diff = [(bid, node.block_states[bid], inp_node.block_states[bid]) 
                        for bid in node.block_states if node.block_states[bid] != inp_node.block_states[bid]]
                for bid, st1, st2 in diff:
                    lam = node.block_lams.get(bid, None)
                    lines.append(f"    Input from node {inp_num}: Block {bid} λ={lam} (state: {st2}→{st1})")

        lines.append(f"  Repair Inputs: {node.repair_inputs}")
        for inp_num in node.repair_inputs:
            inp_node = next((n for n in nodes if n.idx == inp_num), None)
            if inp_node:
                diff = [(bid, node.block_states[bid], inp_node.block_states[bid]) 
                        for bid in node.block_states if node.block_states[bid] != inp_node.block_states[bid]]
                for bid, st1, st2 in diff:
                    mu = node.block_mus.get(bid, None)
                    lines.append(f"    Repair from node {inp_num}: Block {bid} μ={mu} (state: {st2}→{st1})")
        
        lines.append(f"  Outputs: {node.outputs}")
        for out_num in node.outputs:
            out_node = next((n for n in nodes if n.idx == out_num), None)
            if out_node:
                diff = [(bid, node.block_states[bid], out_node.block_states[bid]) 
                        for bid in node.block_states if node.block_states[bid] != out_node.block_states[bid]]
                for bid, st1, st2 in diff:
                    lam = node.block_lams.get(bid, None)
                    lines.append(f"    Output to node {out_num}: Block {bid} λ={lam} (state: {st1}→{st2})")

        lines.append(f"  Repair Outputs: {node.repair_outputs}")
        for out_num in node.repair_outputs:
            out_node = next((n for n in nodes if n.idx == out_num), None)
            if out_node:
                diff = [(bid, node.block_states[bid], out_node.block_states[bid]) 
                        for bid in node.block_states if node.block_states[bid] != out_node.block_states[bid]]
                for bid, st1, st2 in diff:
                    mu = node.block_mus.get(bid, None)
                    lines.append(f"    Repair to node {out_num}: Block {bid} μ={mu} (state: {st1}→{st2})")
                    
        lines.append("-" * 40)
    return "\n".join(lines)

# For Streamlit:
with tab_gen_conn:
    col_gen, col_conn = st.columns(2)
    with col_gen:
        st.subheader("Generation Log")
        st.code('\n'.join(output_lines), language="None")
    with col_conn:
        st.subheader("Connections")
        st.code(get_valid_nodes_connections_text(valid_nodes), language="None")





# MARK: build_kolmogorov_equations_latex
def build_kolmogorov_equations_latex(nodes: List[GraphNode]):
    eqs_latex = []
    for node in nodes:
        # Notation for probability of being in state node.idx
        P = f"P_{{{node.num}}}(t)"
        dPdt = f"\\frac{{d{P}}}{{dt}}"
        # Incoming transitions (from which you can reach node)
        in_terms = []
        for inp_num in node.inputs:
            inp_node = next((n for n in nodes if n.idx == inp_num), None)
            if inp_node:
                # Find block that changed
                diff = [(bid, node.block_states[bid], inp_node.block_states[bid]) 
                        for bid in node.block_states if node.block_states[bid] != inp_node.block_states[bid]]
                for bid, st1, st2 in diff:
                    lam = node.block_lams.get(bid, None)
                    in_terms.append(f"{lam} P_{{{inp_node.num}}}(t)")
        # Outgoing transitions (from which you can leave node)
        out_terms = []
        for out_num in node.outputs:
            out_node = next((n for n in nodes if n.idx == out_num), None)
            if out_node:
                diff = [(bid, node.block_states[bid], out_node.block_states[bid]) 
                        for bid in node.block_states if node.block_states[bid] != out_node.block_states[bid]]
                for bid, st1, st2 in diff:
                    lam = node.block_lams.get(bid, None)
                    out_terms.append(f"{lam} {P}")
        # Repair transitions
        for inp_num in getattr(node, "repair_inputs", []):
            inp_node = next((n for n in nodes if n.idx == inp_num), None)
            if inp_node:
                diff = [(bid, node.block_states[bid], inp_node.block_states[bid]) 
                        for bid in node.block_states if node.block_states[bid] != inp_node.block_states[bid]]
                for bid, st1, st2 in diff:
                    mu = node.block_mus.get(bid, None)
                    in_terms.append(f"{mu} P_{{{inp_node.num}}}(t)")
        for out_num in getattr(node, "repair_outputs", []):
            out_node = next((n for n in nodes if n.idx == out_num), None)
            if out_node:
                diff = [(bid, node.block_states[bid], out_node.block_states[bid]) 
                        for bid in node.block_states if node.block_states[bid] != out_node.block_states[bid]]
                for bid, st1, st2 in diff:
                    mu = node.block_mus.get(bid, None)
                    out_terms.append(f"{mu} {P}")
        
        # Form equation
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

with tab_eq:
    for idx, eq_latex in enumerate(eqs_latex, 1):
        st.latex(f"{idx}.\\quad {eq_latex}", width="content")





# MARK: kolmogorov_rhs
def kolmogorov_rhs(t, P, nodes: List[GraphNode]):

    dPdt = np.zeros_like(P) # dPdt — array of probability derivatives for each state (node)

    for i, node in enumerate(nodes):

        in_terms = []
        out_terms = []

        # Iterate all input nodes (from which you can reach current)
        for inp_num in node.inputs:
            # Find index of input node in nodes list
            inp_idx = next((j for j, n in enumerate(nodes) if n.idx == inp_num), None)
            if inp_idx is not None:
                # Determine which blocks changed state when transitioning from inp_node to node
                diff = [(bid, node.block_states[bid], nodes[inp_idx].block_states[bid]) 
                        for bid in node.block_states 
                        if node.block_states[bid] != nodes[inp_idx].block_states[bid]]
                # print(diff)
                
                # For each such block add term to in_terms
                for bid, st1, st2 in diff:
                    lam = node.block_lams.get(bid, None)
                    # if lam:
                    if lam is not None:
                        in_terms.append(lam * P[inp_idx])
                    else:
                        raise ValueError(f"Lambda not found for block {bid} in node {node.idx}")

        # Iterate all output nodes (where you can go from current)
        for out_num in node.outputs:
            # Find index of output node in nodes list
            out_idx = next((j for j, n in enumerate(nodes) if n.idx == out_num), None)
            if out_idx is not None:
                # Determine which blocks changed state when transitioning from node to out_node
                diff = [(bid, node.block_states[bid], nodes[out_idx].block_states[bid]) 
                        for bid in node.block_states 
                        if node.block_states[bid] != nodes[out_idx].block_states[bid]]
                
                # For each such block add term to out_terms
                for bid, st1, st2 in diff:
                    lam = node.block_lams.get(bid, None)
                    # if lam:
                    if lam is not None:
                        out_terms.append(lam * P[i])
                    else:
                        raise ValueError(f"Lambda not found for block {bid} in node {node.idx}")

        # Repair transitions
        for inp_num in getattr(node, "repair_inputs", []):
            inp_idx = next((j for j, n in enumerate(nodes) if n.idx == inp_num), None)
            if inp_idx is not None:
                diff = [(bid, node.block_states[bid], nodes[inp_idx].block_states[bid]) 
                        for bid in node.block_states 
                        if node.block_states[bid] != nodes[inp_idx].block_states[bid]]
                for bid, st1, st2 in diff:
                    mu = node.block_mus.get(bid, None)
                    if mu is not None:
                        in_terms.append(mu * P[inp_idx])
                    else:
                        raise ValueError(f"Lambda not found for block {bid} in node {node.idx}")

        for out_num in getattr(node, "repair_outputs", []):
            out_idx = next((j for j, n in enumerate(nodes) if n.idx == out_num), None)
            if out_idx is not None:
                diff = [(bid, node.block_states[bid], nodes[out_idx].block_states[bid]) 
                        for bid in node.block_states 
                        if node.block_states[bid] != nodes[out_idx].block_states[bid]]
                for bid, st1, st2 in diff:
                    mu = node.block_mus.get(bid, None)
                    if mu is not None:
                        out_terms.append(mu * P[i])
                    else:
                        raise ValueError(f"Lambda not found for block {bid} in node {node.idx}")

        dPdt[i] = sum(in_terms) - sum(out_terms)

    return dPdt

# MARK: solve_kolmogorov
def solve_kolmogorov(nodes, t_span, P0=None, t_eval=None, rtol=1e-9, atol=1e-12):
    n = len(nodes)
    if P0 is None:
        P0 = np.zeros(n)
        P0[0] = 1.0  # Initial state: all probability in first node
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        fun=lambda t, P: kolmogorov_rhs(t, P, nodes),
        t_span=t_span,
        y0=P0,
        t_eval=t_eval,
        # method='RK45'
        # rtol=1e-7, atol=1e-9
        # rtol=rtol, # Relative error of integrator
        # atol=atol # Absolute error of integrator
    )
    return sol



P0 = None
ode_start_time = time.time()
sol = solve_kolmogorov(
    valid_nodes, 
    t_span=(0, integration_time), 
    P0=P0, 
    t_eval=np.linspace(0, integration_time, n_points),
    rtol=rtol,
    atol=atol
)
time_stats['solve_kolmogorov'] = time.time() - ode_start_time







# MARK: State probabilities chart
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
    yaxis_title='Probability',
    # title='State probabilities chart',
    legend=dict(font=dict(size=12), orientation="h"),
    height=625,
    dragmode='pan',
    # dragmode=False,
    margin=dict(l=0, r=0, t=30, b=0)
)


# with st.container(border=True):
with tab_charts:
    with st.expander("State probabilities chart", expanded=True):
        st.plotly_chart(fig, use_container_width=True, 
                            config={
                                # "scrollZoom": True, 
                                "displayModeBar": True
                                })



# MARK: System reliability chart
# Mask of working states
alive_mask = np.array([not node.is_dead for node in valid_nodes])
# st.write(sol.y)

# Sum of probabilities of working states for each time point
alive_probs_sum = np.sum(sol.y[alive_mask, :], axis=0)  # shape: (len(sol.t),)
alive_probs_sum = np.round(alive_probs_sum, 6)  # Add rounding
# st.write(alive_probs_sum)
# st.text(len(alive_probs_sum))

# K_g = alive_probs_sum[-1]  # Стаціонарний коефіцієнт готовності
# st.write(f"Стаціонарний коефіцієнт готовності: {K_g:.6f}")

fig_alive = go.Figure()
fig_alive.add_trace(go.Scatter(
    x=sol.t,
    y=alive_probs_sum,
    mode='lines',
    name='Sum of probabilities of working states',
    line=dict(width=3, color='#2ecc40')
))
fig_alive.update_layout(
    xaxis_title='t',
    yaxis_title='Sum of probabilities of working states',
    title='System reliability chart',
    height=400,
    margin=dict(l=0, r=0, t=30, b=0),
    dragmode=False,
)

# Mean time to failure (MTTF)
mttf = np.trapezoid(alive_probs_sum, sol.t)

with tab_charts:
    with st.expander(f"Mean time to failure `{mttf:.6f}`", expanded=True):
        st.plotly_chart(fig_alive, use_container_width=True)





# MARK: Probability Table and Bar Chart at Specific Time
with tab_charts:
    with st.expander("State probability distribution", expanded=True):
        t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns([1, 1, 1, 1, 1])

        with t_col1:
            query_time = st.number_input(
                "Time t for viewing probabilities:",
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

        # Find index of time closest to user input query_time
        idx = np.abs(sol.t - query_time).argmin()
        # Select probabilities for all states at this time point
        probs = sol.y[:, idx]

        with t_col2:
            st.markdown(f"t = {sol.t[idx]:.2f}")

        with t_col3:
            st.markdown(f"Sum of probabilities: {np.sum(probs):.12f}")


        alive_sum = np.sum([p for i, p in enumerate(probs) if not valid_nodes[i].is_dead]) # Probability of faultless operation
        dead_sum = np.sum([p for i, p in enumerate(probs) if valid_nodes[i].is_dead])

        alive_count = sum(1 for node in valid_nodes if not node.is_dead)
        dead_count = sum(1 for node in valid_nodes if node.is_dead)
        st.write(f"Working states: {alive_count} / Failed states: {dead_count}")

        with t_col4:
            st.write(f"Probability of faultless operation: {alive_sum:.12f} / `{alive_sum:.2%}`")
        
        with t_col5:
            st.markdown(f"Probability of failure: {dead_sum:.12f} / `{dead_sum:.2%}`")

        p_col_1, p_col_2 = st.columns([1, 3])

        with p_col_1:
            
            prob_table = []
            dead_mask = []

            for i, p in enumerate(probs):
                prob_table.append({
                    "idx": f"P{valid_nodes[i].idx}(t)",
                    "num": f"{valid_nodes[i].num}",
                    "Probability": f"{p:.12f}",
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
            # Prepare data for bar chart
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
                title="State probability distribution (blue — working, red — failed)",
                xaxis_title="State",
                yaxis_title="Probability",
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
with tab_graph:
    if st.button("Generate and Draw Graph"):
        with st.spinner("Wait for it...", show_time=True):

            os.makedirs("images/", exist_ok=True)
            max_dim = 35000

            # Timing for drawing graph
            t0_graph = time.time()
            img_graph = draw_graph(valid_nodes)
            time_stats['draw_graph'] = time.time() - t0_graph

            if img_graph.width > max_dim or img_graph.height > max_dim:
                with st.expander("Graph of Valid Nodes", expanded=True):
                    st.warning(f"Image too large to display ({img_graph.width}x{img_graph.height}).")
            else:
                with st.expander("Graph of Valid Nodes", expanded=True):
                    st.image(img_graph, caption="Graph of Valid Nodes", width="stretch")
                    
            img_graph.save("images/graph.png")

            # Timing for drawing all nodes
            t0_all = time.time()
            img_all = draw_nodes(all_nodes)
            time_stats['draw_all_nodes'] = time.time() - t0_all

            if img_all.width > max_dim or img_all.height > max_dim:
                with st.expander("All Generated Nodes", expanded=True):
                    st.warning(f"Image too large to display ({img_all.width}x{img_all.height}).")
            else:
                with st.expander("All Generated Nodes", expanded=True):
                    st.image(img_all, caption="All Generated Nodes", width="stretch")

            img_all.save("images/all_nodes.png")

            t0_save_images = time.time()
            time_stats['save_images'] = time.time() - t0_save_images
            st.success("Images saved to disk.")

            # img_graph.show()
            # img_all.show()

            # # Save at half size
            img_graph_2x = img_graph.resize((img_graph.width // 2, img_graph.height // 2), Image.LANCZOS)
            img_graph_2x.save("images/graph_2x.png")

            img_all_2x = img_all.resize((img_all.width // 2, img_all.height // 2), Image.LANCZOS)
            img_all_2x.save("images/all_nodes_2x.png")







# MARK: Time stats
with st.sidebar:
    with st.expander("Time Statistics", expanded=True):
        time_stats['total'] = time.time() - start_time
        # display time stats
        for key, val in time_stats.items():
            st.write(f"**{key}**: {val:.2f} sec")




