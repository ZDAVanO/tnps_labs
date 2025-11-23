
from PIL import Image, ImageDraw, ImageFont
from math import atan2, cos, sin, sqrt


# MARK: Configurations
radius = int( 145 )
ellipse_width = int( 4 )
dup_text_offset = int( 70 )
text_row_height = int( 38 )

row_margin = int( 850 )
col_margin = int( 500 )

arrow_width = int( 5 )
arrow_size = int( 40 )

mono_font = ImageFont.truetype("CascadiaMono.ttf", 32)
mono_font_medium = ImageFont.truetype("CascadiaMono.ttf", 72)
mono_font_large = ImageFont.truetype("CascadiaMono.ttf", 90)


arrow_colors = [
    (255, 0, 0, 128),      # red
    (0, 255, 0, 128),      # green
    (255, 255, 255, 128),  # white
    (255, 255, 0, 128),    # yellow
    (255, 0, 255, 128),    # magenta
    (0, 255, 255, 128),    # cyan
]




# MARK: draw_node()
def draw_node(draw, node, x, y, radius, font_large, font_medium, font, ellipse_width, text_row_height):

    # Determine colors based on node state
    if node.is_dead:
        circle_color = 'white'
        fill_color = (91, 44, 44)
    elif node.duplicate_of:
        circle_color = (255, 131, 131)
        fill_color = 'black'
    else: # normal
        circle_color = 'white'
        fill_color = 'black'
    
    text_color = circle_color if not node.is_dead else 'white'

    # Draw filled ellipse for dead nodes
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=fill_color, outline=circle_color, width=4)
    

    # If duplicate
    if node.duplicate_of:
         # Draw semi-transparent red strike-through
        strike_color = (255, 0, 0, 128)  # semi-transparent red
        draw.line(
            (x-radius, y-radius, x+radius, y+radius),
            fill=strike_color,
            width=16
        )
    
        # write "д: {idx}" under the node number
        dup_num = node.duplicate_of[0] if isinstance(node.duplicate_of, list) else node.duplicate_of
        dup_text = f"D: {dup_num}"
        draw.text((x, y+70), dup_text, fill=(255,131,131), anchor='mm', font=font)
    else:
        draw.text((x, y-80), f"{node.num}", fill=(104, 160, 204), anchor='mm', font=font_medium)

    # Draw node number in the center
    draw.text((x, y), str(node.idx), fill=text_color, anchor='mm', font=font_large)

    # Draw block states below the node
    state_texts = []
    for bid in sorted(node.block_states):
        block_type = node.block_types[bid]
        state = node.block_states[bid]
        if state == -1:
            state = "0"
        if state == -2:
            state = "0*"
        if state == 2:
            state = "1*"

        mark_x = "x" if bid in node.locked_blocks else ""
        if block_type and not (isinstance(bid, int) or (isinstance(bid, float) and bid.is_integer())):
            state_texts.append(f"{int(bid) if bid == int(bid) else int(bid)}.{block_type} - {state} {mark_x}")
        else:
            state_texts.append(f"{int(bid)}   : {state} {mark_x}")
    for idx, line in enumerate(state_texts):
        draw.text((x+radius+30, y-radius+idx*text_row_height), line, fill=text_color, font=font)



# MARK: draw_graph()
def draw_graph(nodes):

    def get_node_center(row_idx, col_idx, node_rows, max_cols, radius, row_margin, col_margin):
        row_nodes = len(node_rows[row_idx+1])
        row_width = row_nodes * (2*radius + col_margin) - col_margin if row_nodes > 0 else 0
        total_width = max_cols * (2*radius + col_margin) - col_margin
        offset_x = (total_width - row_width) // 2
        x = offset_x + col_margin + col_idx * (2*radius + col_margin) + radius
        y = row_margin + row_idx * (2*radius + row_margin) + radius
        return (x, y)

    def draw_arrow(draw, x0c, y0c, x1c, y1c, radius, color=(255,255,0,255), width=5, arrow_size=60, label=None, label_font=None, curve_height=75):
        if curve_height == 0:
            angle = atan2(y1c-y0c, x1c-x0c)

            x0 = x0c + radius * cos(angle)
            y0 = y0c + radius * sin(angle)
            x1 = x1c - radius * cos(angle)
            y1 = y1c - radius * sin(angle)
            draw.line((x0, y0, x1, y1), fill=color, width=width)

            ax = x1 - arrow_size * cos(angle - 0.3)
            ay = y1 - arrow_size * sin(angle - 0.3)
            bx = x1 - arrow_size * cos(angle + 0.3)
            by = y1 - arrow_size * sin(angle + 0.3)
            draw.line((x1, y1, ax, ay), fill=color, width=width)
            draw.line((x1, y1, bx, by), fill=color, width=width)

            if label:
                mx = (x0c + x1c) / 2
                my = (y0c + y1c) / 2
                draw.text((mx, my), label, fill=(255,255,255,255), anchor='mm', font=label_font)
        else:
            # Calculate control point for quadratic Bezier
            mx, my = (x0c + x1c) / 2, (y0c + y1c) / 2
            dx, dy = x1c - x0c, y1c - y0c
            dist = sqrt(dx*dx + dy*dy)
            
            if dist == 0:
                nx, ny = 0, 0
            else:
                nx, ny = -dy / dist, dx / dist
            
            cx = mx + nx * curve_height
            cy = my + ny * curve_height

            # Calculate start and end points on the node circles
            # Start point: intersection of center0->control with circle0
            angle_start = atan2(cy - y0c, cx - x0c)
            x0 = x0c + radius * cos(angle_start)
            y0 = y0c + radius * sin(angle_start)

            # End point: intersection of control->center1 with circle1
            angle_end = atan2(y1c - cy, x1c - cx)
            x1 = x1c - radius * cos(angle_end)
            y1 = y1c - radius * sin(angle_end)

            # Draw Bezier curve
            points = []
            steps = 30
            for i in range(steps + 1):
                t = i / steps
                # Quadratic Bezier: (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
                px = (1-t)**2 * x0 + 2*(1-t)*t * cx + t**2 * x1
                py = (1-t)**2 * y0 + 2*(1-t)*t * cy + t**2 * y1
                points.append((px, py))
            
            draw.line(points, fill=color, width=width)

            # Arrowhead
            ax = x1 - arrow_size * cos(angle_end - 0.3)
            ay = y1 - arrow_size * sin(angle_end - 0.3)
            bx = x1 - arrow_size * cos(angle_end + 0.3)
            by = y1 - arrow_size * sin(angle_end + 0.3)
            draw.line((x1, y1, ax, ay), fill=color, width=width)
            draw.line((x1, y1, bx, by), fill=color, width=width)

            if label:
                # Label at t=0.5
                t = 0.5
                lx = (1-t)**2 * x0 + 2*(1-t)*t * cx + t**2 * x1
                ly = (1-t)**2 * y0 + 2*(1-t)*t * cy + t**2 * y1
                draw.text((lx, ly), label, fill=(255,255,255,255), anchor='mm', font=label_font)


    node_rows = {}
    for node in nodes:
        node_rows.setdefault(node.row, []).append(node)


    rows_count = len(node_rows)
    max_cols = max(len(row) for row in node_rows.values())

    width = max_cols * (2*radius + col_margin) + col_margin
    height = rows_count * (2*radius + row_margin) + row_margin

    img_graph = Image.new('RGB', (width, height), 'black')
    draw = ImageDraw.Draw(img_graph, 'RGBA')


    # Map node idx to (row_idx, col_idx)
    node_pos = {}
    for row_idx, (row_num, row_nodes) in enumerate(sorted(node_rows.items())):
        for col_idx, node in enumerate(row_nodes):
            node_pos[node.idx] = (row_idx, col_idx)

    # Draw all nodes (valid only)
    for idx, node in enumerate(nodes, 1):
        # print(f"GRAPH Drawing node {idx}/{len(valid_nodes)}")
        x, y = get_node_center(*node_pos[node.idx], node_rows, max_cols, radius, row_margin, col_margin)
        draw_node(draw, node, x, y, radius, mono_font_large, mono_font_medium, mono_font, ellipse_width, text_row_height)

    # Draw arrows for outputs, cycling colors
    # total_arrows = sum(len(node.outputs) for node in valid_nodes)
    arrow_idx = 0
    for node in nodes:
        for out_num in node.outputs:
            # print(f"GRAPH Drawing arrow {arrow_idx+1}/{total_arrows} (from node {node.idx} to {out_num})")
            color = arrow_colors[arrow_idx % len(arrow_colors)]

            x0c, y0c = get_node_center(*node_pos[node.idx], node_rows, max_cols, radius, row_margin, col_margin)
            x1c, y1c = get_node_center(*node_pos[out_num], node_rows, max_cols, radius, row_margin, col_margin)
            draw_arrow(draw, x0c, y0c, x1c, y1c, radius, color=color, width=arrow_width, arrow_size=arrow_size)
            arrow_idx += 1
        
        for out_num in node.repair_outputs:
            # print(f"GRAPH Drawing repair arrow {arrow_idx+1} (from node {node.idx} to {out_num})")
            color = arrow_colors[arrow_idx % len(arrow_colors)]

            x0c, y0c = get_node_center(*node_pos[node.idx], node_rows, max_cols, radius, row_margin, col_margin)
            x1c, y1c = get_node_center(*node_pos[out_num], node_rows, max_cols, radius, row_margin, col_margin)
            draw_arrow(draw, x0c, y0c, x1c, y1c, radius, color=color, width=arrow_width, arrow_size=arrow_size)
            arrow_idx += 1
    
    return img_graph



# MARK: draw_nodes()
def draw_nodes(nodes):

    # Групуємо всі ноди по node_parent
    nodes_by_parent = {}
    for node in nodes:
        if node.node_parent is not None:
            nodes_by_parent.setdefault(node.node_parent, []).append(node)

    parent_keys = sorted(nodes_by_parent.keys())

    row_height = 2*radius + 100
    col_width = 2*radius + 2*radius
    max_row_len = max(len(nodes_by_parent[k]) for k in parent_keys) if parent_keys else 1
    width_parents = max_row_len * col_width + 1000
    height_parents = len(parent_keys) * row_height + 200

    img_all = Image.new('RGBA', (width_parents, height_parents), 'black')  # <-- RGBA mode for alpha support
    draw_parents = ImageDraw.Draw(img_all, 'RGBA')

    for row_idx, parent_num in enumerate(parent_keys):
        nodes_row = nodes_by_parent[parent_num]
        # Get row number from the first node in the row
        row_number = nodes_row[0].row if nodes_row else '?'
        draw_parents.text(
            (30, row_idx * row_height + radius),
            f"P: {parent_num}, R: {row_number}",
            fill=(255, 255, 255),
            font=mono_font_large
        )
        for col_idx, node in enumerate(nodes_row):
            x = 1000 + col_idx * col_width
            y = row_idx * row_height + radius + 50
            draw_node(draw_parents, node, x, y, radius, mono_font_large, mono_font_medium, mono_font, ellipse_width, text_row_height)
            
    return img_all