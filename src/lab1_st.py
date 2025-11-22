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
    # st.header("Integration Parameters")
    integration_time = st.slider(
        "Integration time (t, sec)", 
        min_value=100, max_value=15000, value=2500, step=100
    )

    lam_min_value = 0.0
    lam_max_value = 0.1
    lam_format_str = "%.5f"
    lam_step = 0.0001

    st.write("λ values for blocks:")

    lam_b1_h = round(st.number_input("b1_h (1.1)", min_value=lam_min_value, max_value=lam_max_value, value=0.0005, format=lam_format_str, step=lam_step), 6)
    lam_b1_s = round(st.number_input("b1_s (1.2)", min_value=lam_min_value, max_value=lam_max_value, value=0.0005, format=lam_format_str, step=lam_step), 6)
    lam_b2   = round(st.number_input("b2 (2)",     min_value=lam_min_value, max_value=lam_max_value, value=0.0004, format=lam_format_str, step=lam_step), 6)
    lam_b3   = round(st.number_input("b3 (3)",     min_value=lam_min_value, max_value=lam_max_value, value=0.0003, format=lam_format_str, step=lam_step), 6)
    lam_b4   = round(st.number_input("b4 (4)",     min_value=lam_min_value, max_value=lam_max_value, value=0.00025, format=lam_format_str, step=lam_step), 6)
    lam_b5_h = round(st.number_input("b5_h (5.1)", min_value=lam_min_value, max_value=lam_max_value, value=0.0005, format=lam_format_str, step=lam_step), 6)
    lam_b5_s = round(st.number_input("b5_s (5.2)", min_value=lam_min_value, max_value=lam_max_value, value=0.0001, format=lam_format_str, step=lam_step), 6)





# MARK: Blocks
# blocks = {
#     0:    LogicBlock(0, "Start"),

#     1.1:  LogicBlock(1.1, "H", lam=lam_b1_h),
#     1.2:  LogicBlock(1.2, "S", lam=lam_b1_s),

#     2:    LogicBlock(2, "H", lam=lam_b2),
#     3:    LogicBlock(3, "H", lam=lam_b3),
#     4:    LogicBlock(4, "H", lam=lam_b4),

#     5.1:  LogicBlock(5.1, "H", lam=lam_b5_h),
#     5.2:  LogicBlock(5.2, "S", lam=lam_b5_s),

#     6:    LogicBlock(6, "End"),
# }

# # dependencies of blocks that cannot be simultaneously working/faulty
# mutual_exclusions = [
#     (1.1, 1.2),
#     (5.1, 5.2)
# ]

# blocks[0].connect_to(blocks[1.1])
# blocks[0].connect_to(blocks[2])
# blocks[0].connect_to(blocks[5.1])

# blocks[1.1].connect_to(blocks[1.2])
# blocks[5.1].connect_to(blocks[5.2])

# blocks[1.2].connect_to(blocks[3])
# blocks[1.2].connect_to(blocks[4])

# blocks[2].connect_to(blocks[3])
# blocks[2].connect_to(blocks[4])

# blocks[3].connect_to(blocks[6])
# blocks[4].connect_to(blocks[6])
# blocks[5.2].connect_to(blocks[6])



blocks = {
    0:    LogicBlock(0, "Start"),

    1:  LogicBlock(1, "H", lam=lam_b1_h),
    2:  LogicBlock(2, "H", lam=lam_b1_s),
    3:  LogicBlock(3, "H", lam=lam_b2),

    4:  LogicBlock(4, "End"),
}

# dependencies of blocks that cannot be simultaneously working/faulty
mutual_exclusions = []

blocks[0].connect_to(blocks[1])


blocks[1].connect_to(blocks[2])
blocks[1].connect_to(blocks[3])

blocks[2].connect_to(blocks[4])
blocks[3].connect_to(blocks[4])





# MARK: generate_graph()
def generate_graph():
    valid_node_num = 1
    block_ids = [b.id for b in blocks.values() if b.type not in ["Start", "End"]]
    block_types = {bid: blocks[bid].type for bid in block_ids}
    block_lams = {bid: blocks[bid].lam for bid in block_ids}
    block_mus = {bid: blocks[bid].mu for bid in block_ids}
    # total_blocks = len(block_ids)

    output_lines = []  # for streamlit

    # Start node: all blocks are working
    idx = 1
    row = 1
    start_states = {bid: 1 for bid in block_ids}
    start_node = GraphNode(row, idx, valid_node_num, None, start_states, block_ids, block_types, block_lams)
    valid_node_num += 1

    for line in start_node.print_states_lines():
        output_lines.append(line)
    output_lines.append("-" * 30)

    queue = deque()
    queue.append(start_node)
    idx += 1

    # All unique states to avoid duplicates
    seen = {}
    seen[tuple(sorted(start_states.items()))] = 1

    # List of all valid graph nodes
    valid_nodes = [start_node]
    node_by_idx = {start_node.idx: start_node}

    all_nodes = [start_node]  # <-- new list for all nodes

    while queue:
        current_node = queue.popleft()
        row = current_node.row
        current_states = current_node.block_states
        parent_idx = current_node.idx
        next_row = row + 1

        # Find all blocks that are not yet broken
        working_blocks = [bid for bid, state in current_states.items() if state == 1]
        # If all blocks are already broken — do not proceed further
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


        # For each block that is not yet broken, create a new state with an additional failure
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
            node = GraphNode(next_row, idx, valid_node_num, parent_idx, new_states, block_ids, block_types, block_lams)
            node.locked_blocks = locked_blocks

            for line in node.print_states_lines():
                output_lines.append(line)


            broken_ids = [bid for bid, state in new_states.items() if state == 0]
            can_reach_result = can_reach(blocks, 0, max(blocks.keys()), broken_ids)
            if not can_reach_result:
                node.is_dead = True

            output_lines.append(f"Endpoint check: {'✅' if can_reach_result else '❌'}")

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

                # Add connection between parent and child node
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

            # Add connection between parent and child node
            if parent_idx in node_by_idx:
                node_by_idx[parent_idx].connect_to(node)

            valid_nodes.append(node)
            valid_node_num += 1
            node_by_idx[idx] = node

            output_lines.append("-" * 30)
            queue.append(node)  # Added only if not duplicate and there is a path to the end
            seen[state_tuple] = idx
            idx += 1

    return valid_nodes, all_nodes, output_lines


valid_nodes, all_nodes, output_lines = generate_graph()

st.write(f"{len(all_nodes)} nodes generated / {len(valid_nodes)} valid.")

# Output via streamlit
with st.expander("Graph Generation Output", expanded=False):
    st.code('\n'.join(output_lines), language="None")





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
        lines.append(f"  Outputs: {node.outputs}")
        for out_num in node.outputs:
            out_node = next((n for n in nodes if n.idx == out_num), None)
            if out_node:
                diff = [(bid, node.block_states[bid], out_node.block_states[bid]) 
                        for bid in node.block_states if node.block_states[bid] != out_node.block_states[bid]]
                for bid, st1, st2 in diff:
                    lam = node.block_lams.get(bid, None)
                    lines.append(f"    Output to node {out_num}: Block {bid} λ={lam} (state: {st1}→{st2})")
        lines.append("-" * 40)
    return "\n".join(lines)

# For Streamlit:
with st.expander("Valid Nodes and Connections", expanded=False):
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

with st.expander("Equations", expanded=False):
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

        dPdt[i] = sum(in_terms) - sum(out_terms)

    return dPdt

# MARK: solve_kolmogorov
def solve_kolmogorov(nodes, t_span, P0=None, t_eval=None):
    n = len(nodes)
    if P0 is None:
        P0 = np.zeros(n)
        P0[0] = 1.0  # Initial state: all probability in first node
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1500)
        # t_eval = np.linspace(t_span[0], t_span[1], t_span[1] + 1)
    sol = solve_ivp(
        fun=lambda t, P: kolmogorov_rhs(t, P, nodes),
        t_span=t_span,
        y0=P0,
        t_eval=t_eval,
        # method='RK45'
        # rtol=1e-7, atol=1e-9
        rtol=1e-9, # Relative error of integrator
        atol=1e-12 # Absolute error of integrator
    )
    return sol



P0 = None
ode_start_time = time.time()
sol = solve_kolmogorov(valid_nodes, t_span=(0, integration_time), P0=P0)
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
with st.expander(f"State probabilities chart", expanded=True):
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

with st.expander(f"Mean time to failure `{mttf:.6f}`", 
                 expanded=True):
    st.plotly_chart(fig_alive, use_container_width=True)





# MARK: Probability Table and Bar Chart at Specific Time
with st.expander(f"State probability distribution", expanded=True):
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
with st.expander(f"Draw Graph", expanded=True):
    if st.button("Generate and Draw Graph"):
        with st.spinner("Wait for it...", show_time=True):

            os.makedirs("images/", exist_ok=True)

            # Timing for drawing graph
            t0_graph = time.time()
            img_graph = draw_graph(valid_nodes)
            time_stats['draw_graph'] = time.time() - t0_graph
            # with st.expander("Graph of Valid Nodes", expanded=True):
            #     st.image(img_graph, caption="Graph of Valid Nodes", width="stretch")
            # Timing for drawing all nodes
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

            # # Save at half size
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




