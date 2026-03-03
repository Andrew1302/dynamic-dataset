# Dynamic Multimodal Graph QA Dataset Generator

Academic research project for dynamic multimodal LLM dataset generation on graph theory problems. Problems are generated on-the-fly so the model always faces brand-new instances, avoiding dataset contamination.

Each sample consists of:
- A **graph visualization** (PNG image)
- A **text prompt** in English Q/A format
- A **ground-truth answer**

## Setup

Requires **Python 3.12** and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

### Generate samples

```bash
# Default: 10 samples, all 14 tasks, small graphs (5–9 nodes)
uv run python src/dataset-generator/multimodal.py

# 100 samples with medium-sized graphs (10–14 nodes)
uv run python src/dataset-generator/multimodal.py -n 100 --size medium

# Only specific tasks
uv run python src/dataset-generator/multimodal.py --tasks mst shortest_path cycle_check

# Custom output directory and seed
uv run python src/dataset-generator/multimodal.py -o my_output --seed 42
```

Output is saved to `output_samples/` by default (one PNG per sample).

### CLI options

| Flag | Default | Description |
|---|---|---|
| `-n`, `--num-samples` | `10` | Number of samples to generate |
| `-o`, `--output-dir` | `output_samples` | Directory for output PNGs |
| `--seed` | `12227` | Random seed for reproducibility |
| `--tasks` | all | Space-separated list of task names |
| `--size` | `small` | Graph size: `small` (5–9), `medium` (10–14), `large` (15–19) |

### Available tasks

| Task name | Question | Answer type |
|---|---|---|
| `node_count` | How many nodes are in this graph? | Integer |
| `edge_count` | How many edges are in this graph? | Integer |
| `node_degree` | What is the degree of node X? | Integer |
| `cycle_check` | Is there a cycle in this graph? | Yes/No |
| `edge_existence` | Is node X connected to node Y? | Yes/No |
| `connected_nodes` | List all the nodes connected to X in ascending order. | Comma-separated list |
| `disconnected_nodes` | List all the nodes that are not connected to X in ascending order. | Comma-separated list |
| `reachability` | Is there a path from node X to node Y? | Yes/No |
| `connectivity_check` | Is this graph connected? | Yes/No |
| `connected_components` | How many connected components does this graph have? | Integer |
| `shortest_path` | What is the length of the shortest path from node X to node Y? | Integer |
| `mst` | What is the weight of the minimum spanning tree of this graph? | Number |
| `triangle_counting` | How many triangles are in this graph? | Integer |
| `maximum_flow` | What is the maximum capacity of the flow from node X to node Y? | Integer |

Tasks 1–11 follow the [GraphQA](https://arxiv.org/abs/2405.06782) benchmark format. Tasks 12–14 (`mst`, `connectivity_check`, `connected_components`) are original additions.

## Library API

The package can be imported programmatically for use in evaluation pipelines:

```python
import importlib

pkg = importlib.import_module("src.dataset-generator")

# List all registered tasks
print(pkg.get_all_tasks().keys())

# Generate a single sample
import networkx as nx
task_cls = pkg.get_task("cycle_check")
G = nx.erdos_renyi_graph(8, 0.4)
sample = task_cls().generate(G)
# sample = {"prompt": "Q: ...", "image": <PIL.Image>, "answer": "Yes"}

# Render any graph to an image
img = pkg.graph2img(G, weighted=False)
img.save("my_graph.png")
```

## Adding a new task

1. Create a subclass of `GraphTask` in the appropriate module under `src/dataset-generator/tasks/`.
2. Set `name` to a unique identifier (this is what `--tasks` uses).
3. Implement `generate(self, G, **kw)` returning `{"prompt": ..., "image": ..., "answer": ...}`.

```python
from . import GraphTask
from ..visualization import graph2img

class MyTask(GraphTask):
    name = "my_task"

    def generate(self, G, **kw):
        image = graph2img(G)
        answer = "42"
        prompt = "Q: What is the answer to life, the universe, and everything?\nA:"
        return {"prompt": prompt, "image": image, "answer": answer}
```

The task is auto-registered when the module is imported. If you add a new file, import it in `src/dataset-generator/tasks/__init__.py`.

## Project structure

```
src/dataset-generator/
├── __init__.py          # Public API
├── __main__.py          # python -m entry point
├── multimodal.py        # CLI entry point
├── graph_generator.py   # Graph generation (ER, BA, WS, SBM, SFN, complete, star, path, tree)
├── visualization.py     # graph2img → PIL.Image
└── tasks/
    ├── __init__.py      # GraphTask base class + registry
    ├── basic.py         # NodeCount, EdgeCount, NodeDegree, CycleCheck
    ├── connectivity.py  # EdgeExistence, ConnectedNodes, DisconnectedNodes, Reachability, ConnectivityCheck, ConnectedComponents
    ├── pathfinding.py   # ShortestPath, MST
    └── advanced.py      # TriangleCounting, MaximumFlow
```
