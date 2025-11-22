


# MARK: LogicBlock
class LogicBlock:
    def __init__(self, block_id, lb_type, state=1, lam=0.0, mu=0.0):
        self.id = block_id
        self.type = lb_type # "H" (Hardware), "S" (Software), "Start", "End"
        self.state = state
        self.inputs = []
        self.outputs = []

        self.lam = lam  # failure rate (λ)
        self.mu = mu # repair rate (μ)
        if mu == 0.0:
            self.mu = lam * 10  # default repair rate
        

        self.s_updates = 0
        if self.type == "S":
            self.s_updates = 1


    def connect_to(self, other_block):
        self.outputs.append(other_block.id)
        other_block.inputs.append(self.id)

    def set_state(self, new_state):
        self.state = new_state

    def is_working(self):
        return self.state == 1



# MARK: can_reach()
def can_reach(blocks, start_id, end_id, broken_ids):
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



## MARK: GraphNode
class GraphNode:
    def __init__(self, 
                 row, 
                 idx, 
                 num, 
                 node_parent, 
                 block_states=None, 
                 block_ids=None, 
                 block_types=None, 
                 block_lams=None):
        
        self.row = row
        self.idx = idx
        self.num = num  # sequential number of valid node
        self.node_parent = node_parent
        # If block_states is not provided — all blocks are working
        self.block_ids = block_ids if block_ids is not None else []
        self.block_states = block_states if block_states is not None else {bid: 1 for bid in self.block_ids}
        # Store block types for convenience
        self.block_types = block_types if block_types is not None else {}
        self.block_lams = block_lams if block_lams is not None else {}

        self.inputs = []
        self.outputs = []

        self.duplicate_of = []

        self.locked_blocks = []

        self.is_dead = False
        self.fixable = True

    def mark_duplicate_of(self, other_node):
        self.duplicate_of.append(other_node.idx)

    def connect_to(self, other_block):
        self.outputs.append(other_block.idx)
        other_block.inputs.append(self.idx)

    def print_states_lines(self):
        lines = []
        lines.append(f"Node (idx={self.idx}, num={self.num}, row={self.row}, parent={self.node_parent}):")
        for bid in sorted(self.block_states):
            block_type = self.block_types.get(bid, None)
            is_integer = isinstance(bid, int) or (isinstance(bid, float) and bid.is_integer())
            state = self.block_states[bid]
            # Form the line
            if block_type and not is_integer:
                lines.append(f"{int(bid) if bid == int(bid) else int(bid)}.{block_type} - {state} {'x' if bid in self.locked_blocks else ''}")
            else:
                lines.append(f"{int(bid)}   - {state} ")

        return lines




