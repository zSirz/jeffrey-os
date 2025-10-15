"""Fallback validation with cycle detection"""

from collections import defaultdict, deque


def validate_fallbacks(modules: dict[str, dict], fallback_chains: dict) -> bool:
    """
    Validate all fallbacks exist and no cycles
    Returns True if valid, raises RuntimeError if not
    """

    # 1. Check existence
    missing = []

    for region, chain in fallback_chains.items():
        for level in ["primary", "secondary", "tertiary"]:
            if level in chain:
                module_name = chain[level].get("module")
                if module_name and module_name not in modules:
                    # Allow simple_* fallbacks (will be created)
                    if not module_name.startswith("simple_"):
                        missing.append(f"{region}/{level}: {module_name}")

    if missing:
        raise RuntimeError(f"Missing fallbacks: {missing[:10]}")

    # 2. Build dependency graph
    graph = defaultdict(list)
    all_modules = set()

    for region, chain in fallback_chains.items():
        modules_in_chain = []
        for level in ["primary", "secondary", "tertiary"]:
            if level in chain:
                module = chain[level].get("module")
                if module:
                    modules_in_chain.append(module)
                    all_modules.add(module)

        # Add edges (primary -> secondary -> tertiary)
        for i in range(len(modules_in_chain) - 1):
            graph[modules_in_chain[i]].append(modules_in_chain[i + 1])

    # 3. Detect cycles using Kahn's algorithm
    in_degree = {node: 0 for node in all_modules}

    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    # Queue with nodes having no incoming edges
    queue = deque([node for node, degree in in_degree.items() if degree == 0])
    visited = 0

    while queue:
        node = queue.popleft()
        visited += 1

        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If visited < total nodes, there's a cycle
    if visited < len(all_modules):
        raise RuntimeError("Cycle detected in fallback chains!")

    return True
