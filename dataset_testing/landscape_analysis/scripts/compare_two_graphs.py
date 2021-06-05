import pandas as pd
import sys
import networkx as nx

def create_graph(dataset):

	#get unique nodes
	nodes_list = [value for value in dataset['node1']]
	for value in dataset['node2']:
		nodes_list.append(value)

	nodes_list = list(set(nodes_list))
	
	#create a graph
	graph_data = nx.Graph()

	for node in nodes_list:
		graph_data.add_node(node)

	#make arist
	for i in range(len(dataset)):
		graph_data.add_edge(dataset['node1'][i], dataset['node2'][i], weight=float(dataset['valueHB'][i]))
	
	return graph_data, nodes_list

original = pd.read_csv(sys.argv[1])
variant = pd.read_csv(sys.argv[2])
path_output = sys.argv[3]

#get graphs
graph_original = create_graph(original)
graph_variant = create_graph(variant)

print("Initial data of original")
print(graph_original.number_of_nodes())
print(graph_original.number_of_edges())

print("Initial data of variant")
print(graph_variant.number_of_nodes())
print(graph_variant.number_of_edges())


