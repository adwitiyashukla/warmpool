from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import combinations


class Node:
    __slots__ = ("item", "count", "parent", "children")

    def __init__(self, item: str | None, parent: Node | None) -> None:
        self.item = item
        self.count = 0
        self.parent = parent
        self.children: dict[str, Node] = {}


def _build(paths: Iterable[tuple[list[str], int]], min_count: int) -> tuple[Node, dict, dict]:
    paths = list(paths)
    counts: Counter = Counter()
    for items, count in paths:
        for item in items:
            counts[item] += count
    keep = {item: c for item, c in counts.items() if c >= min_count}
    rank = {item: i for i, item in enumerate(sorted(keep, key=lambda x: (-keep[x], x)))}
    root = Node(None, None)
    header: dict[str, list[Node]] = defaultdict(list)
    for items, count in paths:
        node = root
        for item in sorted((i for i in set(items) if i in rank), key=rank.__getitem__):
            child = node.children.get(item)
            if child is None:
                child = Node(item, node)
                node.children[item] = child
                header[item].append(child)
            child.count += count
            node = child
    return root, header, keep


def _prefix(node: Node) -> list[str]:
    out = []
    node = node.parent
    while node is not None and node.item is not None:
        out.append(node.item)
        node = node.parent
    return out


def _mine(
    header: dict,
    support: dict,
    suffix: tuple[str, ...],
    min_count: int,
    max_len: int,
    found: dict[frozenset, int],
) -> None:
    for item in sorted(header, key=lambda x: (support[x], x)):
        itemset = suffix + (item,)
        found[frozenset(itemset)] = support[item]
        if len(itemset) >= max_len:
            continue
        base = [(_prefix(node), node.count) for node in header[item]]
        _, sub_header, sub_support = _build(base, min_count)
        if sub_header:
            _mine(sub_header, sub_support, itemset, min_count, max_len, found)


def fpgrowth(
    transactions: list[frozenset[str]], min_count: int, max_len: int
) -> dict[frozenset, int]:
    _, header, support = _build(((list(t), 1) for t in transactions), min_count)
    found: dict[frozenset, int] = {}
    _mine(header, support, (), min_count, max_len, found)
    return found


def brute_force(
    transactions: list[frozenset[str]], min_count: int, max_len: int
) -> dict[frozenset, int]:
    items = sorted({i for t in transactions for i in t})
    found = {}
    for size in range(1, max_len + 1):
        any_frequent = False
        for combo in combinations(items, size):
            itemset = frozenset(combo)
            count = sum(1 for t in transactions if itemset <= t)
            if count >= min_count:
                found[itemset] = count
                any_frequent = True
        if not any_frequent:
            break
    return found
