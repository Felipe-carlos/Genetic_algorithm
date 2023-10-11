# Título: Algoritmo genético
# Autor: Felipe C. dos Santos
#
# Descrição: este é um exemplo de codigo de um algoritmo genético usando seleção por roleta e crossover em um único ponto.

import random
import math
import matplotlib.pyplot as plt
import numpy as np

n = 8 #tamanho de cromossomos
l = 200 #numero de cromossomos
up_limit = math.pi  #limit superior da função de fitness
pc= 0.7 #probabilidade de cruzamento
pm=0.01 #probabilidade de mutação
generations = 600 #numero de gerações que o algoritmo vai rodar


def created_crom(n): #retorna str de n bits random
    return format(random.randrange(0,2**n,1), '0' + str(n) + 'b') #gera str de n bits random

def creat_pop(n_pop, n_bits_crom): #função que gera l individuos da popupalção inicial
    return [created_crom(n_bits_crom) for x in range(n_pop)]

def convert_to_value(bits,limit):     #função para converter str de n bits em um valor de 0 a limit
    return (int(bits,2)*limit)/(2**n-1) #retorna um valor int correspondente do cromossomo no intervalo de 0 a up_limit

def fitness_calc(value): #recebe o valor int entre 0 e up limit e calcula o fitness do cromossomo para função que queremos otimizar
    return value+abs(math.sin(32*value))

def evaluate_pop(population): #recebe uma lista de str com os bits de cada cromossomo
    return [fitness_calc(convert_to_value(x,up_limit)) for x in population ] # retorna uma lista com o calculo do fitness da população

def selection_op(population,n_pairs): #recebe uma lista de str com os bits de cada cromossomo e o numero de pares selecionados
    #algoritmo de escolha baseado na amostragem por roleta
    fitness = evaluate_pop(population)          #calcula o fitnes da pop
    sum_fitness = sum(fitness)                  #calcula a soma dos fitness da população
    selection_prob = [x/sum_fitness for x in fitness]   #normaliza os fitness para eles somarem em 1
    final=[]                                    #lista final com os n_pairs pares escolhidos

    for x in range(n_pairs):
        selected = random.choices([i for i in range(l)], weights=selection_prob, k=2)   #faz a primeira escolha de pais -- retorna a posição do pai escolhido
        while selected[0] == selected[1]:                                                 #caso sejam esolhidos 2 vezes o mesmo cromossomo como pai ele refaz
            selected = random.choices([i for i in range(l)], weights=selection_prob, k=2)
        final=final+selected                                                            #adiciona o casal de pais la lista final

    return final     #retorna uma lista com a posição relativa dos casais escolhidos dentro da população

def mutation_op(offspring, prob): # recebe uma --lista de str-- com os bits de cada cromossomo dos decendentes e a probalidade de mutação

    mutated_offspring=[]           #inicia o vetor da população resultante do processo de mutação

    for x in offspring:     #itera os individuos
        mutated_chrom = ''
        if random.random() < prob:                              #caso realmente ocorra a mutação
            inv_bit = random.randrange(0,len(x),1)   #escolhe o bit que será invertido
            if x[inv_bit] == '0':
               mutated_chrom = x[:inv_bit] + '1' + x[inv_bit + 1:]  #inverte o bit mutado
            else:
                mutated_chrom = x[:inv_bit] + '0' + x[inv_bit + 1:] #inverte o bit mutado
        else:
            mutated_chrom = x                                       #caso não ocorra a mutação o cromossomo mutado será o mesmo que o original
        mutated_offspring.append(mutated_chrom)         #adiciona o individuo mutado na geração mutada
    return mutated_offspring

def crossover_op(parent_1,parent_2,prob): #recebe uma str com os cromossomos e retorna uma lista de str com os 2 cromossomos "filhos" de acordo com a probabilidade de cruzamento prob
        #algoritmo de cruzamento em um único ponto
    if random.random() < prob:                 # realmente ocorra o cruzamento em relação a propabilidade prob
        split_point= random.randrange(0,n-1,1)
        decedent_1 = parent_1[:split_point] + parent_2[split_point:]            #cria decedente 1
        decedent_2 = parent_2[:split_point] + parent_1[split_point:]            #cria decedente 2
    else:                                       # caso não ocorra o cruzamento apenas repete os bits dos pais
        decedent_1 = parent_1
        decedent_2 = parent_2
    return [decedent_1, decedent_2]


##---- algoritmo principal
#for pc in [0.5, 0.7, 0.9]:
population =  creat_pop(l,n)     # gera uma população inicial com l cromossomos de tamanho n

mean_gem = []

for current_gem in range(generations):                              #roda o algoritmo em um numero de vezes dado pela constante generations
    new_population = []
    selected_parents= selection_op(population,math.ceil(l/2))      #seleciona math.floor(l/2) numero de --pares-- garantindo que o numero de pais aumente junto com a população e fique ao redor da metade da população

    for pares in range(0,len(selected_parents),2):                      #itera em pares para fazer o crossover dos pais selecionados
       pai_1 = population[selected_parents[pares]]
       pai_2 = population[selected_parents[pares+1]]
       par_descendente = crossover_op(pai_1,pai_2,pc)              #gera um par de descendentes a partir dos pais 1 e 2 com a probabilidade pc
       par_descendente = mutation_op(par_descendente,pm)           #aplica mutação a cada um dos bits dos descendentes com probabilidade pm

      #----------- substitui o novo par de descententes pelos pais
       new_population.append(par_descendente[0])
       if len(new_population) < len(population):                    #garante que o tamanho da nova população não seja maior que a antiga -- para casos de l's impar
           new_population.append((par_descendente[1]))

    population=new_population                                       #atualiza a população
    current_gem += 1

    if current_gem in [1, 20, generations]:                         #plota a distribuição de individuos das gerações na lista
        plt.figure(current_gem)
        intervalo = np.arange(0, math.pi, 0.001)
        plt.plot(intervalo, [x + abs(math.sin(32 * x)) for x in intervalo],color='gray')
        plt.scatter([convert_to_value(x, up_limit) for x in population], evaluate_pop(population)+np.random.uniform(-0.02, 0.02, len(population)),label=f"pc= {pc}")
        plt.ylim(0, 4.2)
        plt.title("Dispersão de individuos da "+str(current_gem)+"º geração")
        plt.xlabel("Eixo X")
        plt.ylabel("Eixo Y")
        plt.legend()
        #plt.savefig(f'pc_variation_gem{current_gem}.png')

    mean_gem.append(sum(evaluate_pop(population)) / l)  # faz um vetor com as médias dos fitness
##-----plota Fitness médio da população através das Gerações:-------

plt.figure()
plt.plot([i for i in range(generations)],mean_gem)
plt.title("Fitness médio da população através das Gerações")
plt.xlabel("Gerações")
plt.ylabel("Fitness")
plt.legend()
#plt.ylim(3.6, 4.2)
#plt.savefig(f'l={l}_pm{pm}_pc{pc}_fitness_x_geracao.png')
plt.show()
