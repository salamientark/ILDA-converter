"""Graph construction and Eulerian path finding for laser path optimization.

Phase 1.2 of the optimization pipeline:
  1. Build a multigraph from (welded) polylines — endpoints as nodes, polylines
     as undirected edges with interior points stored in the edge payload.
  2. Detect connected components; for each one:
       a. Find odd-degree nodes.
       b. Pair them greedily with minimum-distance blanking edges.
       c. Traverse with Hierholzer's algorithm to obtain an Eulerian circuit.
  3. Stitch disconnected component sub-paths together with blanking jumps.

Output is a single continuous :class:`LaserPath` where blanking/travel points
carry ``status=1`` and visible points carry ``status=0``.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass

from src.logger.logging_config import get_logger
from src.postprocessing.laser_path import LaserPath
from src.postprocessing.laser_point import LaserPoint

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal edge representation
# ---------------------------------------------------------------------------

_Node = tuple[float, float]


@dataclass
class _Edge:
    """A single undirected edge in the laser multigraph."""

    node_u: _Node
    node_v: _Node
    points: list[_Node]  # ordered u → v; stores all interior points
    blanking: bool = False
    used: bool = False


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(
    polylines: list[list[_Node]],
) -> tuple[list[_Edge], dict[_Node, list[int]]]:
    """Construct an undirected multigraph from polylines.

    Each polyline with at least two points becomes one edge connecting its
    first endpoint (``node_u``) to its last endpoint (``node_v``).  Interior
    points are stored in ``edge.points`` and re-emitted during path
    reconstruction.  Single-point polylines are silently skipped.

    Args:
        polylines: Nested list of ``(x, y)`` tuples.  Should already have
            been processed by :func:`weld_vertices`.

    Returns:
        ``(edges, adj)`` where *edges* is the global edge list and *adj* maps
        each node to the indices of its incident edges.
    """
    edges: list[_Edge] = []
    adj: dict[_Node, list[int]] = defaultdict(list)

    for poly in polylines:
        if len(poly) < 2:
            continue
        u, v = poly[0], poly[-1]
        idx = len(edges)
        edges.append(_Edge(node_u=u, node_v=v, points=list(poly)))
        adj[u].append(idx)
        adj[v].append(idx)

    logger.debug(
        "build_graph: %d polylines → %d edges, %d nodes",
        len(polylines),
        len(edges),
        len(adj),
    )
    return edges, dict(adj)


# ---------------------------------------------------------------------------
# Connected-component detection
# ---------------------------------------------------------------------------


def _find_components(
    adj: dict[_Node, list[int]],
    edges: list[_Edge],
) -> list[set[_Node]]:
    """BFS-based connected-component detection.

    Returns a list of node sets, one per component.
    """
    visited: set[_Node] = set()
    components: list[set[_Node]] = []

    for seed in adj:
        if seed in visited:
            continue
        component: set[_Node] = set()
        queue: deque[_Node] = deque([seed])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for eidx in adj.get(node, []):
                e = edges[eidx]
                nbr = e.node_v if e.node_u == node else e.node_u
                if nbr not in visited:
                    queue.append(nbr)
        components.append(component)

    return components


# ---------------------------------------------------------------------------
# Odd-vertex pairing (T-join via greedy nearest-neighbour)
# ---------------------------------------------------------------------------


def _add_blanking_edges(
    edges: list[_Edge],
    adj: dict[_Node, list[int]],
    candidates: list[_Node],
) -> None:
    """Greedily pair *candidates* and insert blanking travel edges in-place.

    Pairs are chosen by minimum Euclidean distance between remaining
    candidates.  After pairing, both nodes in each pair gain one extra edge,
    restoring even degree.  If an odd number of candidates is supplied (which
    should not happen in a valid undirected graph), the last node is left
    unpaired and a warning is logged.

    Args:
        edges: Global edge list — new ``_Edge`` objects are appended here.
        adj: Adjacency dict — updated with new edge indices.
        candidates: Nodes with odd degree that require pairing.
    """
    remaining = list(candidates)
    
    while len(remaining) >= 2:
        u = remaining.pop(0)  # O(n) shift — acceptable since k is typically small
        best = min(
            range(len(remaining)),
            key=lambda i: math.hypot(
                remaining[i][0] - u[0],
                remaining[i][1] - u[1],
            ),
        )
        v = remaining.pop(best)
        idx = len(edges)
        edges.append(_Edge(node_u=u, node_v=v, points=[u, v], blanking=True))
        adj.setdefault(u, []).append(idx)
        adj.setdefault(v, []).append(idx)
        logger.debug(
            "blanking edge %d: %s → %s (d=%.2f)",
            idx,
            u,
            v,
            math.hypot(v[0] - u[0], v[1] - u[1]),
        )

    if remaining:
        logger.warning(
            "Unpaired odd node %s — graph parity is inconsistent; "
            "the path may be incomplete.",
            remaining[0],
        )


# ---------------------------------------------------------------------------
# Hierholzer's algorithm (stack-based, undirected multigraph)
# ---------------------------------------------------------------------------


def _hierholzer(
    edges: list[_Edge],
    comp_adj: dict[_Node, list[int]],
    start: _Node,
) -> list[tuple[_Node, int | None]]:
    """Find an Eulerian circuit or path within a single connected component.

    *comp_adj* must represent an Eulerian graph (all nodes even-degree) or a
    semi-Eulerian graph (exactly two odd-degree nodes), as guaranteed by
    :func:`_add_blanking_edges` together with the caller's node selection in
    :func:`_process_component`.  When the graph is Eulerian the traversal
    returns to *start*, forming a circuit; when it is semi-Eulerian *start*
    must be one of the two odd-degree nodes and the traversal ends at the
    other, forming an open path.

    Args:
        edges: Global edge list; the ``used`` flag on each edge is set as it
            is consumed.
        comp_adj: Adjacency dict for *this component only*.
        start: Starting node.

    Returns:
        An ordered list of ``(node, arriving_edge_idx)`` tuples.  The first
        entry represents the start node and has ``arriving_edge_idx = None``
        in both the circuit and path cases.
    """
    # Per-node stacks of unvisited edge indices (copied so we don't mutate comp_adj)
    node_stacks: dict[_Node, list[int]] = {n: list(es) for n, es in comp_adj.items()}

    stack: list[_Node] = [start]
    arriving: list[int | None] = [None]
    circuit: list[tuple[_Node, int | None]] = []

    while stack:
        v = stack[-1]
        # Discard already-used edges at the top of the local stack
        while node_stacks.get(v) and edges[node_stacks[v][-1]].used:
            node_stacks[v].pop()

        if node_stacks.get(v):
            edge_idx = node_stacks[v].pop()
            edge = edges[edge_idx]
            edge.used = True
            next_v = edge.node_v if edge.node_u == v else edge.node_u
            stack.append(next_v)
            arriving.append(edge_idx)
        else:
            node = stack.pop()
            eidx = arriving.pop()
            circuit.append((node, eidx))

    circuit.reverse()
    return circuit


# ---------------------------------------------------------------------------
# Circuit → LaserPath conversion
# ---------------------------------------------------------------------------


def _circuit_to_path(
    circuit: list[tuple[_Node, int | None]],
    edges: list[_Edge],
    *,
    r: int = 0,
    g: int = 0,
    b: int = 0,
) -> LaserPath:
    """Convert a Hierholzer circuit to a :class:`LaserPath`.

    For each edge traversal the interior points are emitted in the correct
    direction.  The junction point shared between consecutive edges is emitted
    only once (deduplicated by skipping the first point of every edge after
    the very first emission).

    Args:
        circuit: Output of :func:`_hierholzer`.
        edges: Global edge list.
        r / g / b: Colour channels for visible (non-blanking) points.

    Returns:
        Ordered :class:`LaserPath` for this circuit.
    """
    path: LaserPath = []

    for i, (node, edge_idx) in enumerate(circuit):
        if edge_idx is None:
            continue  # start node — emitted as the first point of the first edge

        prev_node = circuit[i - 1][0]
        edge = edges[edge_idx]

        pts = edge.points if edge.node_u == prev_node else edge.points[::-1]

        # Skip the shared junction point on all but the very first emission
        for pt in pts[1:] if path else pts:
            status = 1 if edge.blanking else 0
            path.append(
                LaserPoint.from_xy(
                    pt[0],
                    pt[1],
                    r=r if not edge.blanking else 0,
                    g=g if not edge.blanking else 0,
                    b=b if not edge.blanking else 0,
                    status=status,
                )
            )

    return path


# ---------------------------------------------------------------------------
# Component processing and sub-path stitching helpers
# ---------------------------------------------------------------------------


def _process_component(
    edges: list[_Edge],
    adj: dict[_Node, list[int]],
    comp_nodes: set[_Node],
    comp_idx: int,
    n_components: int,
    *,
    r: int,
    g: int,
    b: int,
) -> LaserPath:
    """Build and return the Eulerian sub-path for a single connected component.

    Constructs a local copy of the adjacency dict, identifies odd-degree nodes,
    inserts blanking edges as needed via :func:`_add_blanking_edges`, selects a
    start node, runs Hierholzer's algorithm, and converts the circuit to a
    :class:`LaserPath` via :func:`_circuit_to_path`.

    Args:
        edges: Global edge list (mutated in-place when blanking edges are added).
        adj: Global adjacency dict (not mutated; local copy is made).
        comp_nodes: Node set for this component.
        comp_idx: Zero-based index of this component (for logging).
        n_components: Total number of components (for logging).
        r / g / b: Colour channels for visible points.

    Returns:
        Ordered :class:`LaserPath` for this component, or ``[]`` if empty.
    """
    comp_adj: dict[_Node, list[int]] = {n: list(adj[n]) for n in comp_nodes if n in adj}

    odd = [n for n, es in comp_adj.items() if len(es) % 2 == 1]
    n_odd = len(odd)
    logger.debug(
        "component %d/%d: %d node(s), %d odd-degree node(s)",
        comp_idx + 1,
        n_components,
        len(comp_nodes),
        n_odd,
    )

    if n_odd > 2:
        # Pair all but 2 odd nodes so that exactly 2 remain (semi-Eulerian).
        # The 2 unpaired nodes become the natural start and end of the path.
        _add_blanking_edges(edges, comp_adj, odd[:-2])
        start = odd[-2]
    elif n_odd == 2:
        # Already semi-Eulerian: path runs from odd[0] to odd[1].
        start = odd[0]
    elif n_odd == 1:
            logger.warning(
                "component %d/%d: 1 odd-degree node (should not happen in a valid "
                "undirected graph) — path may be incomplete.",
                comp_idx + 1,
                n_components,
            )
            start = odd[0]
    else:
        # Fully Eulerian (0 odd nodes): any start node produces a circuit.
        start = next(iter(comp_nodes))

    circuit = _hierholzer(edges, comp_adj, start)
    return _circuit_to_path(circuit, edges, r=r, g=g, b=b)


def _stitch_sub_paths(sub_paths: list[LaserPath]) -> LaserPath:
    """Merge component sub-paths into one :class:`LaserPath` with blanking jumps.

    Between each pair of consecutive sub-paths two blanking
    :class:`LaserPoint` objects are inserted: one holding the last position of
    the outgoing sub-path and one travelling to the first position of the
    incoming sub-path.

    Args:
        sub_paths: Non-empty list of per-component paths (empty entries skipped).

    Returns:
        A single merged :class:`LaserPath`, or ``[]`` if *sub_paths* is empty.
    """
    if not sub_paths:
        return []

    path: LaserPath = list(sub_paths[0])
    for sp in sub_paths[1:]:
        if not sp:
            continue
        last = path[-1]
        first_pt = sp[0]
        path.append(LaserPoint.from_xy(last.x, last.y, status=1))
        path.append(LaserPoint.from_xy(first_pt.x, first_pt.y, status=1))
        path.extend(sp)

    return path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def find_eulerian_path(
    polylines: list[list[_Node]],
    *,
    r: int = 0,
    g: int = 0,
    b: int = 0,
) -> LaserPath:
    """Produce a single continuous :class:`LaserPath` covering all polylines.

    **Algorithm**:

    1. Build a multigraph where each polyline is an undirected edge.
    2. Detect connected components.
    3. For each component:
       - Identify odd-degree nodes (endpoints that appear an odd number of
         times across all polylines in the component).
       - If there are more than 2 odd nodes, greedily pair all but 2 of them
         with minimum-distance blanking edges, leaving exactly 2 odd nodes
         (making the graph semi-Eulerian).
       - Traverse with Hierholzer's algorithm starting at one of the remaining
         odd nodes (the other is the natural end of the path).
       - A component with exactly 0 or 2 odd nodes requires no blanking edges.
    4. Stitch the component sub-paths together with blanking jumps.

    Args:
        polylines: Welded polylines from :func:`weld_vertices`.
        r / g / b: Colour channels applied to all *visible* points.

    Returns:
        A continuous :class:`LaserPath`.  Blanking / travel points have
        ``status=1`` with ``r=g=b=0``; visible points have ``status=0``.
    """
    valid = [p for p in polylines if len(p) >= 2]
    if not valid:
        logger.debug("find_eulerian_path: no valid polylines, returning empty path")
        return []

    edges, adj = build_graph(valid)
    components = _find_components(adj, edges)
    logger.debug("find_eulerian_path: %d connected component(s)", len(components))

    sub_paths = [
        sp
        for comp_idx, comp_nodes in enumerate(components)
        if (
            sp := _process_component(
                edges, adj, comp_nodes, comp_idx, len(components), r=r, g=g, b=b
            )
        )
    ]

    path = _stitch_sub_paths(sub_paths)
    logger.debug("find_eulerian_path: %d total points in LaserPath", len(path))
    return path
