import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

algorithms = []

def register(algorithm):
    algorithms.append(algorithm)
    return algorithm

@register
def mst(n=50):
    prompt = 'Dado o grafo na imagem abaixo, qual o peso da árvore geradora mínima?\n'
    #G = nx.random_tree(n)
    k = min(n-2, 3)
    G = nx.connected_watts_strogatz_graph(n=n, k=k, p=0.3)

    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(1, int(n/2))

    mst = nx.minimum_spanning_tree(G, algorithm='kruskal', weight='weight')
    weight = mst.size(weight='weight')
    return prompt, graph2img(G, True), weight

@register
def connected(n=50):
    prompt = 'Dado o grafo da imagem abaixo, indique se ele é conexo\n'

    p = random.uniform(0, 1)

    if p <0.5:
        G = nx.erdos_renyi_graph(n=n, p=0.1)
    else:
        G = nx.random_labeled_tree(n)

    return prompt, graph2img(G), nx.is_connected(G)

@register
def connected_components(n=50):
    prompt = 'Dado o grafo da imagem abaixo, qual o número de componentes conexas?\n'

    p = random.uniform(0, 1)
    if p <0.5:
        G = nx.erdos_renyi_graph(n=n, p=1/n)
    else:
        G = nx.random_labeled_tree(n)

    return prompt, graph2img(G), nx.number_connected_components(G)

@register
def shortest_path(n=50):
    k = min(n-2, 3)
    G = nx.connected_watts_strogatz_graph(n=n, k=k, p=0.3)
    #G = nx.path_graph(n)
    
    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(1, int(n/2))

    u, v = random.sample(list(G.nodes()), 2)
    prompt = 'Dado o grafo da imagem abaixo, indique qual o caminho mínimo '
    prompt = prompt + f' entre os vértices {u} e {v}\n'
    c = nx.dijkstra_path_length(G, source=u, target=v, weight='weight')

    return prompt, graph2img(G, True), c

def graph2img(G, weighted=False):
    pos = nx.spring_layout(G)  # Organizado
    nx.draw(G, pos, with_labels=True)

    if weighted:
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    return None#Not sure about the MLLM input (in memory imagem)

def visualization():
    n = 20
    #G = nx.connected_watts_strogatz_graph(n=n, k=3, p=0.3)
    #G = nx.path_graph(n)
    #G = nx.cycle_graph(n)
    #G = nx.erdos_renyi_graph(n=n, p=0.1)

    G = nx.connected_watts_strogatz_graph(n=n, k=3, p=0.3)
    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(1, int(n/2))

    pos = nx.spring_layout(G)  # mais bonito/organizado
    nx.draw(G, pos, with_labels=True)

    if True:
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.show()

def main():
    import os
    random.seed(12227)  # Garante estocasticidade se necessario
    np.random.seed(12227)
    out_dir = "output_samples"
    os.makedirs(out_dir, exist_ok=True)

    for i in range(3):  # Demo: 3 samples (use 10**6 for full run)
        n = random.randint(4, 10)
        text, visual, answer = random.choice(algorithms)(n)
        print(f"\n--- Sample {i+1} ---")
        print(text)
        print(f"Answer: {answer}")
        plt.savefig(f"{out_dir}/sample_{i+1}.png", dpi=100)
        plt.close()
        #input('')
        #problem, answer = random.choice(algorithms)()
        #LLM_answer = LLM.predict(problem)
        #erro = answer - LLM_answer
        

main()