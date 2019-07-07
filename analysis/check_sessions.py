import matplotlib.pyplot as plt
import pickle
import sys
import numpy as np

files = [('operators_only_40_3.pckl','Operators Only'),('operators_only_no_ft40_3.pckl','Operators Only No ft'),('fitness_40_3.pckl','Fitness only'),('with_choose40_3.pckl','fitness + choose'),('all_40_3.pckl','tudo')]
for argv in files:
    file_pickle_opened = open(argv[0], 'rb')
    file_graph = pickle.load(file_pickle_opened) 
    print(file_graph[0])
    plt.plot(file_graph,'.',label=argv[1])
    
    #graphs.append( graph )
    file_pickle_opened.close();

plt.legend(loc='upper right')
plt.title("sessão x tempo gasto")
plt.xlabel("sessão")
plt.ylabel("tempo gasto")
#plt.ylim(0,5)
#plt.xticks(range(50)/2)
#plt.yticks(np.arange(0, 1, 0.05))
plt.grid(True)
plt.show()