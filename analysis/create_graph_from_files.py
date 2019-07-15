from graph import Graph
import matplotlib.pyplot as plt
import pickle
import sys
import dill

#files_pickle = ['./graphs/20.pckl','./graphs/40.pckl','./graphs/80.pckl','./graphs/160.pckl','./graphs/gradient_he.pckl']
#files_pickle = ['./graphs/20_he.pckl','./graphs/40_he.pckl','./graphs/80_he.pckl','./graphs/160_he.pckl','./graphs/gradient_he.pckl']
#files_pickle = ['./graphs/20_he_menor.pckl','./graphs/40_he_menor.pckl','./graphs/80_he_menor.pckl','./graphs/160_he_menor.pckl','./graphs/gradient.pckl']
#files_pickle = ['./graphs/20_3.pckl','./graphs/40_3.pckl','./graphs/80_3.pckl','./graphs/160_3.pckl','./graphs/gradient_3.pckl']
#files_pickle = ['./graphs/20_5.pckl','./graphs/40_5.pckl','./graphs/80_5.pckl','./graphs/160_5.pckl','./graphs/gradient_5.pckl']
#files_pickle = ['./graphs/20_10.pckl','./graphs/40_10.pckl','./graphs/80_10.pckl','./graphs/160_10.pckl','./graphs/gradient_10.pckl']
#files_pickle = ['./graphs/20_he_menor.pckl','./graphs/40_he_menor.pckl','./graphs/80_he_menor.pckl','./graphs/160_he_menor.pckl','./graphs/gradient.pckl']

# files_pickle = ['./graphs/30_3.pckl', './graphs/gradient_3.pckl']
# graphs = []

# for file_pickle in files_pickle:
#     file_pickle_opened = open(file_pickle, 'rb') 
#     graphs.append(pickle.load(file_pickle_opened) )
#     file_pickle_opened.close();
 
# for graph in graphs:
#     if(graph.performance[0] < 0):
#         graph.performance = [x * -1 for x in graph.performance]

# #print(graphs[0].performance)

# mutation_20 = plt.plot(graphs[0].tempo,graphs[0].performance , '-', label="Mutation_20")
# gradient = plt.plot(graphs[1].tempo,graphs[1].performance , '-', label="Gradient")
# #mutation_80 = plt.plot(graphs[2].tempo,graphs[2].performance , '-', label="Mutation_80")
# #mutation_160 = plt.plot(graphs[3].tempo,graphs[3].performance , '-', label="Mutation_160")
# #gradient = plt.plot(graphs[4].tempo,graphs[4].performance , '-', label="Gradient")
# #plt.legend([mutation_20,mutation_40,mutation_80,mutation_160,gradient],['mutação 20','mutação 40','mutação 80','mutação 160'])
# #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
# #           ncol=2, mode="expand", borderaxespad=0.)
# plt.grid(True);
# plt.show()


# mutation_20_acc = plt.plot(graphs[0].tempo,graphs[0].accuracy , '-', label="Mutation_20")
# #mutation_40_acc = plt.plot(graphs[4].tempo,graphs[4].performance , '-', label="Gradient")

# gradient = plt.plot(graphs[1].tempo,graphs[1].accuracy , '-', label="Gradient")

# plt.grid(True)
# plt.show()


# file_pickle = './graphs/' +  sys.argv[1]
# file_pickle_opened = open(file_pickle, 'rb') 
# graph = pickle.load(file_pickle_opened)
# file_pickle_opened.close();
# if(graph.performance[0] < 0):
#         graph.performance = [x * -1 for x in graph.performance]
 
# plot = plt.plot(graph.tempo,graph.performance , '-', label="Mutation_20")
# plt.legend([plot],['grafico'])

# plt.grid(True)
# plt.show()
class Graph_file:
    def __init__(self,graph,name):
        self.graph = graph
        self.name = name

files_pickle = []
graphs = []
for argv in sys.argv[1:]:
     file_pickle_opened = open('../debug/graphs_logs/' + argv, 'rb')
     file_graph = dill.load(file_pickle_opened)
     graph = Graph_file(file_graph,argv)
     graphs.append( graph )
     file_pickle_opened.close();
 
for graph_file in graphs:
    graph = graph_file.graph
    if(graph.validation_performance[0] < 0):
        graph.validation_performance = [x * -1 for x in graph.validation_performance]
    plt.plot(graph.validation_tempo,graph.validation_performance , '-',label=("População de tamanho " + graph_file.name[:2]))
plt.legend(loc='upper right')
plt.title("Loss x Tempo do conjunto de treino com feedforward de 3 camadas ocultas")
plt.xlabel("Tempo (s)")
plt.ylabel("Loss (softmax cross entropy)")
#plt.ylim(0,5)
plt.grid(True)
plt.show()
#
# for graph_file in graphs:
#     graph = graph_file.graph
#     if(graph.performance[0] < 0):
#         graph.performance = [x * -1 for x in graph.performance]
#     plt.plot(graph.tempo,graph.accuracy , '-', label=graph_file.name)
# plt.legend(loc='upper right')
# plt.title("Acurácia x Tempo do conjunto de treino")
# plt.xlabel("Tempo (s)")
# plt.ylabel("Acurácia")
# plt.grid(True)
# plt.ylim(0,1)
# plt.show()
#
# for graph_file in graphs:
#     graph = graph_file.graph
#     if(graph.performance[0] < 0):
#         graph.performance = [x * -1 for x in graph.performance]
#     if(len(graph.fine_tuning) > 0):
#         plt.plot(graph.tempo,graph.fine_tuning , '-', label=graph_file.name)
# plt.legend(loc='upper right')
# plt.title("Fine_Tuning x Tempo do conjunto de treino")
# plt.xlabel("Tempo (s)")
# plt.ylabel("População")
# plt.grid(True)
# plt.show()

""" 
for graph_file in graphs:
    graph = graph_file.graph
    if(graph.performance[0] < 0):
        graph.performance = [x * -1 for x in graph.performance]
    plt.plot(graph.validation_tempo,graph.validation_performance , '-', label=graph_file.name)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

for graph_file in graphs:
    graph = graph_file.graph
    if(graph.performance[0] < 0):
        graph.performance = [x * -1 for x in graph.performance]
    plt.plot(graph.validation_tempo,graph.validation_accuracy , '-', label=graph_file.name)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()  """ 