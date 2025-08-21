import numpy as np
edge_index_type = np.load('graph_with_olliverRicci_add_0_bridge.npy')
orc = np.load('orc_with_olliverRicci_add_0_bridge.npy')
id2ent = {}
id2rel = {}
for idx, line in enumerate(open('./FB15k-237/entities.txt')):
    entity_str = line.strip()
    id2ent[idx] = entity_str
for idx, line in enumerate(open('./FB15k-237/relations.txt')):
    relation_str = line.strip()
    id2rel[idx] = relation_str
num_edges = orc.shape[0]
delete_ratio = 0.1
k = int(num_edges * delete_ratio)
gate = np.partition(orc, -k)[-k]
print(gate)
num_edges_kept = 0
with open('filtered_edges.txt', 'w', encoding='utf-8') as f:
    for col in range(num_edges):
        orc_val = orc[col]
        if orc_val < gate:
            head = id2ent[edge_index_type[0][col]]
            tail = id2ent[edge_index_type[1][col]]
            relation = id2rel[edge_index_type[2][col]]
            f.write(f"{head}\t{relation}\t{tail}\n")
            num_edges_kept += 1
print(num_edges_kept)