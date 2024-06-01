import copy
import heapq
import networkx as nx
from overlap import OverlapResolver
from node import Node
from itertools import permutations
import random
import sys
import numpy as np
import matplotlib.pyplot as plt


### GA ###


class GA:
    def __init__(self):
        self.crossover_prob = 0.7
        self.mutation_prob = 0.2
        self.ring_size = 3
        # Buffer utilizzato per memorizzare gli overlap calcolati tra le sequenze.
        self.buffer = {}

    # calculates each instance of a recurrent function to calculate sequences overlaps
    """
    La funzione _getOverlapValue sembra calcolare il valore di sovrapposizione tra due sottostringhe s1 e s2 alle posizioni specificate i e j rispettivamente, 
        utilizzando una matrice matrix per la programmazione dinamica. 
    La sovrapposizione è calcolata utilizzando i parametri match, mismatch e gap per assegnare punteggi alle corrispondenze tra i caratteri delle due sottostringhe.
    """

    def run_ga(self, env, pop, gen):
        print("----- Siamo nel metodo run_GA -----")
        # memory = remove_zeros(memory)
        # print("Stampa delle varie letture: ", reads)
        pop_size = len(pop)
        # cromo_size = len(reads)

        # print("Qui", self.findOverlapNew([2, 3, 5], [3, 5, 0]))

        # population1 = np.array(memory)
        population = pop
        # print("Popolazione iniziale: ")

        for i, popolazione in enumerate(population):
            # print(f"Popolazione {i + 1}:")
            for individuo in popolazione:
                sequenza, lista_numeri = individuo
                # print(f"  Sequenza: {sequenza}, Numeri: {lista_numeri}")

        # print("Le letture sono: ", reads)
        pop_fitness = self._evaluatePopulation(population, gen)
        print(pop_fitness)
        #print("--------------------------------")

        #print("Calcolo della fitness ", pop_fitness, " per la Popolazione corrente: ", population)

        c = np.argmax(pop_fitness)
        # print(c)
        best_ind = copy.deepcopy(population[c])
        # print(best_ind)

        #print("Migliorr individuo della popolazione corrente: \n", best_ind)
        #print("Seleziona l'individuo con la fitness massima dalla popolazione corrente: ", best_ind)
        #print("---------------------")
        # print("Le letture sono: ", reads)
        best_fit = self._fitness(best_ind)
        #print("Valore fitness dell'individuo migliore: ", best_fit)

        # inizializza un array numpy (fitness_evolution) con zeri, la cui lunghezza è data dal numero di generazioni specificato da generations.

        for generation in range(100):
            print("---------")
            print('Esecuzione')
            # Tournament selection
            selected = []
            # seleCpy = []
            for i in range(len(population)):
                # winner = self._ring(pop_fitness, self.ring_size)
                # print("Winner", population[winner])
                selected.append(population[i].copy())
                # seleCpy.append(population[i].copy())
            # print("Selezionati i vincitori del torneo: ")

            """
            for x in selected:
                print(x)
            """

            # Crossover
            for i in range(0, pop_size, 2):
                #print("i:", i)
                if i + 1 < len(selected):
                    if np.random.rand() < self.crossover_prob:
                        #print("-----------------")
                        # print("Popo", selected[i])
                        # print("Popo1", selected[i + 1])
                        population[i], population[i + 1] = self._crossover(selected[i], selected[i + 1])
                        # print("Popolazioni ottenute tramite il crossover: \n", population[i])
                        # print(population[i + 1])
                    else:
                        # Qui invece di eseguire l'operazione di crossover, copiamo semplicemente i genitori nella nuova generazione
                        population[i], population[i + 1] = selected[i].copy(), selected[i + 1].copy()
                        #print("Operazione di crossover non andata a buon fine: ", population[i], population[i + 1])
            # Mutation
            #print("-----------------")
            # print("Stampa dopo l'operazione di crossover")
            # for i, popolazione in enumerate(population):
            # print(f"Popolazione {i + 1}:", popolazione)

            #print("Siamo nel metodo mutation:")
            for i in range(len(population)):
                if np.random.rand() > self.mutation_prob:
                    population[i] = self._mutation(population[i])
                    # print("Popolazione corrente nel mutation: ", "\n", population[i])

            #print("Stampa della popolazione dopo l'operazione di mutation:")

            # Iterazione sulla lista di popolazioni
            # for i, popolazione in enumerate(population):
            # print(f"Popolazione {i + 1}:", popolazione)

            # for x in population:
            # print("Population", x)
            #print("----------------------")

            print("Ridefiniamo i valori fitness della popolazione modificata: ")
            # print(array)
            #print("----------------------")

            # population.append(best_ind)

            population.append(copy.deepcopy(best_ind))
            #print("Popolazione dopo l'aggiunta di", best_ind)

            #print(population)

            pop_fitness = self._evaluatePopulation(population, gen)
            print(pop_fitness)
            value = pop_fitness.max()
            print("------------------")
            print("L'obiettivo è quello di mantenere nelle generazioni successive il miglior individuo: ")
            print("Stampa il valore massimo di fitness della generazione attuale: ", value)
            print("Mentre il valore massimo di fitness fino ad ora e di ", best_fit)

            """
            Questo blocco di codice implementa una strategia di elitismo nell'algoritmo genetico. L'obiettivo è
            mantenere l'individuo migliore (con la fitness massima) tra le generazioni successive, garantendo che il 
            miglior individuo individuato finora venga preservato nella popolazione.
            Verifica se la fitness massima nella generazione corrente (fitness_evolution[generation]) è inferiore
            alla fitness del miglior individuo trovato finora (best_fit)
            """
            if value < best_fit:
                best_ind = copy.deepcopy(population[np.argmax(pop_fitness)])
                population[0] = copy.deepcopy(best_ind)
                #print("Popolazione 0: ", population[0])
                best_fit = value
            else:
                best_ind = population[pop_fitness.argmax()].copy()
                best_fit = pop_fitness.max()
                indexValue = np.argmax(pop_fitness)
                population[0] = copy.deepcopy(population[indexValue])

        #print("Miglior individuo durante l'evalution:", best_ind)

        return best_ind

    def _getOverlapValue(self, i, j, matrix, s1, s2, match, mismatch, gap):
        score = match if s1[i - 1] == s2[j - 1] else mismatch
        aux = max(
            matrix[i - 1][j - 1] + score,
            matrix[i - 1][j] + gap,
            matrix[i][j - 1] + gap,
            0
        )
        return aux

    """
    La funzione compute_overlap prende due stringhe left_read e right_read come input e cerca la sovrapposizione tra di esse. 
    La sovrapposizione è la parte comune tra il suffisso di left_read e il prefisso di right_read.
    ritorno della funzione -> Se trova una corrispondenza completa, restituisce la sovrapposizione e la sua lunghezza.
    """

    def compute_overlap(self, left_read, right_read):
        for i in range(len(left_read)):
            l = left_read[i:]
            size = len(l)
            r = right_read[:size]
            if l == r:
                return l, size
        return "", 0

    """
    La funzione _getOverlap calcola la sovrapposizione massima tra due stringhe s1 e s2 utilizzando una matrice di programmazione dinamica. 
    La sovrapposizione è calcolata sommando i punteggi delle sovrapposizioni tra i prefissi delle due stringhe. 
    La funzione restituisce il valore massimo della sovrapposizione ottenuta.
    """

    def _getOverlap(self, s1, s2, match, mismatch, gap):
        l = len(s1) + 1
        c = len(s2) + 1
        matrix = np.array([0.0 for _ in range(l * c)]).reshape(l, c)
        for i in range(1, l):
            for j in range(1, c):
                matrix[i][j] = self._getOverlapValue(i, j, matrix, s1, s2, match, mismatch, gap)
        return np.max(matrix)

    """
    Questa funzione aiuta a ottenere la lunghezza della sovrapposizione tra il suffisso di una stringa e il prefisso di un'altra.
    """

    def _getSuffixPrefixOverlap(self, left, right):
        return self.compute_overlap(left, right)[1]

    """La funzione _findOverlap calcola la lunghezza della sovrapposizione tra due sequenze, identificate dagli 
    indici id1 e id2, presenti nella lista reads. La sovrapposizione rappresenta la quantità di caratteri comuni tra 
    una parte finale della sequenza reads[id1] e una parte iniziale della sequenza reads[id2]."""

    def _findOverlap2(self, reads, id1, id2, match=1.0, mismatch=-0.33, gap=-1.33):
        if id1 < id2:
            minId = id1
            maxId = id2
        else:
            minId = id2
            maxId = id1
        if minId in self.buffer and maxId in self.buffer[minId]:
            overlap = self.buffer[minId][maxId]
        else:
            overlap = self._getOverlap(reads[id1], reads[id2], match, mismatch, gap)
            if minId not in self.buffer:
                self.buffer[minId] = {}
            self.buffer[minId][maxId] = overlap
        return overlap

    """La funzione _crossover implementa l'operatore di crossover per l'algoritmo genetico. In particolare, 
    prende in input due cromosomi (cromossome1 e cromossome2) e restituisce due nuovi cromosomi (figli) ottenuti 
    combinando porzioni dei cromosomi genitori."""

    def crossover(self, population1, population2):
        # Seleziona casualmente due cromosomi da ciascuna popolazione
        selected_chromosome_pop1 = np.random.choice(len(population1), size=2, replace=False)
        selected_chromosome_pop2 = np.random.choice(len(population2), size=2, replace=False)

        # Seleziona casualmente un punto di crossover all'interno di ciascun cromosoma
        crossover_point_pop1 = np.random.randint(1, len(population1[0]))
        crossover_point_pop2 = np.random.randint(1, len(population2[0]))

        # Esegui lo scambio delle porzioni selezionate
        for i in range(2):
            temp = population1[selected_chromosome_pop1[i]][:crossover_point_pop1].copy()
            population1[selected_chromosome_pop1[i]][:crossover_point_pop1] = population2[selected_chromosome_pop2[i]][
                                                                              :crossover_point_pop2]
            population2[selected_chromosome_pop2[i]][:crossover_point_pop2] = temp

        return population1, population2

    def _crossover(self, cromossome1, cromossome2):

        """
        if type(cromossome1) == list:
            cromossome1 = np.array(cromossome1)
        if type(cromossome2) == list:
            cromossome2 = np.array(cromossome2)
        """

        """
        -> len(cromossome1): 
            Specifica la popolazione di cui estrarre i numeri casuali. 
                In questo caso, è lunghezza del cromosoma cromossome1.
        -> size=2: 
            Specifica il numero di campioni da estrarre. 
                Qui vogliamo ottenere due indici per rappresentare la porzione del cromosoma da scambiare.
        -> replace=False: 
            Garantisce che i numeri estratti siano unici. 
                Se replace fosse True (il valore predefinito), potremmo ottenere duplicati.
        
        La funzione restituirà: 
            N.B -> Quindi, il risultato di (genes) sarà un array di due elementi
                ognuno rappresentante un indice casuale nell'intervallo da 0 a len(cromossome) - 1. 
                Ad esempio, [2, 4] indica che sono stati selezionati casualmente gli indici 2 e 4 dall'array cromossome.
        """

        #print("----- Siamo nella funzione Crossover -----")
        # print("Stampiamo due vincitori del torneo _ring: ", "\n", cromossome1, "\n", cromossome2)

        genes = np.random.choice(len(cromossome1), size=2, replace=False)
        genes.sort()
        # print("Stampiamo l'intervallo di  indici  ottenuti casualmente del primo cromosoma: ", genes)
        """
        Queste due linee di codice stanno estraendo una porzione specifica da ciascun cromosoma genitore,
            utilizzando gli indici precedentemente generati.
        -> aux1 = cromossome1[genes[0]:genes[1] + 1]: 
            Estrae una porzione del cromosoma cromossome1 che va dall'indice genes[0] all'indice genes[1] inclusi.
                L'operazione + 1 è necessaria per includere l'indice finale nella porzione.

        -> aux2 = cromossome2[genes[0]:genes[1] + 1]: 
                Analogamente, estrae una porzione dal cromosoma cromossome2 con lo stesso intervallo di indici.
        Ex: se gli indici ottenuti da genes sono [1 8], allora dentro aux1 andiamo a mettere gli elementi che partono dall'indice 1 fino a 8 incluso
        N.B -> Questo processo di estrazione è una parte chiave dell'operatore di crossover. La porzione estratta da 
        ciascun cromosoma verrà quindi combinata per creare i cromosomi figli durante la fase di crossover"""

        #print("-----")

        aux1 = cromossome1[genes[0]:genes[1] + 1]
        # print("Estrazione dal cromosoma 1", " con valori ", aux1)
        aux2 = cromossome2[genes[0]:genes[1] + 1]
        # print("Estrazione dal cromosoma 2", " con valori ", aux2)
        #print("-----")
        """Queste differenze rappresentano le parti di cromosomi che non sono state incluse nelle porzioni estratte 
        durante l'operazione di crossover. -> np.in1d(cromossome2, aux1, assume_unique=True) restituisce un array di 
        bool che indica, quali elementi di cromosome2 sono presenti un aux1, con ~ davanti, andiamo a negare il 
        risultato dell'istruzione ottenendo gli elementi del cromosoma2 che non sono stati utilizzati per il 
        crossover"""
        #print("-----")

        # diff2 = cromossome2[~np.in1d(cromossome2, aux1, assume_unique=True)]
        diff2 = [gene for gene in cromossome2 if gene not in aux1]
        # print("Parti del cromosoma non incluse durante l'operazione di crossover eseguito: ", diff2)
        # diff1 = cromossome1[~np.in1d(cromossome1, aux2, assume_unique=True)]
        diff1 = [gene for gene in cromossome1 if gene not in aux2]
        # print("Parti del cromosoma non incluse durante l'operazione di crossover eseguito: ", diff1)
        #print("-----")

        """
        Ci consente di formare i cromosomi figli: aggiungendo in coda agli elementi estratti per il crossover di aux1,
            la differenza tra gli elementi non presenti del cromosoma2.
        Stessa cosa fatta anche per child2
        """
        #print("-----")

        aux1.extend(diff2)
        aux2.extend(diff1)
        # print("Stampa dei figli ottenuti combinando gli elementi estratti dal primo e secondo cromosoma, con l'aggiunta degli elementi che non sono stati presi durante il crossover ", "\n", aux1, "\n", aux2)
        #print("-----")
        return aux1, aux2

    """
    Questa funzione seleziona casualmente due geni nel cromosoma e ne scambia i valori.
    """

    def _mutation(self, population):
        # print("----- Siamo nel metodo _Mutation: -----")
        # print("Stampa della popolazione prima dell'operazione di mutation:")
        # for individual in population:
        # print(individual)
        mutated_population = population.copy()

        # Seleziona casualmente due indici diversi
        index1, index2 = random.sample(range(len(population)), 2)
        # print("Indici casuali selezionati:", index1, index2)

        # Scambia i due individui di posizione
        mutated_population[index1], mutated_population[index2] = mutated_population[index2], mutated_population[index1]

        # print("Stampa della popolazione dopo l'operazione di mutation:")

        # for individual in mutated_population:
        # print(individual)

        return mutated_population

    """
    def _mutation(self, cromossome):
        print("----- Siamo nel metodo _Mutation: -----")
        print("Stampa del cromosoma: ", "\n", cromossome)

        #mutated_array = cromossome.copy()  # Creiamo una copia dell'array originale per non modificarlo direttamente
        mutated_array = [list(sub_array) for sub_array in cromossome]

        for i, sub_array in enumerate(mutated_array):  # Iteriamo su ciascun array interno
            if len(sub_array) > 1:  # Assicuriamoci che l'array interno abbia almeno due elementi per poter mutare
                # Selezioniamo casualmente due indici all'interno dell'array interno
                idx1, idx2 = np.random.choice(len(sub_array), size=2, replace=False)
                print("Estazione degli indici: ", idx1, idx2, " dal cromosoma: ", sub_array,
                      "al fine di scambiarli nello stesso cromosoma")

                #Scambiamo i valori ai due indici selezionati
                #cp = mutated_array[i][idx1]
                # print("idx1", mutated_array[i][idx1])
                # print("idx2", mutated_array[i][idx2])
                #mutated_array[i][idx1] = mutated_array[i][idx2]
                #mutated_array[i][idx2] = cp  # Questa è la linea corretta
                # print("i", mutated_array[i])

                mutated_array[i][idx1], mutated_array[i][idx2] = mutated_array[i][idx2], mutated_array[i][idx1]

        mutated_array = [tuple(sub_array) for sub_array in mutated_array]
        # print("Stampa del cromosoma prima: ", cromossome)
        # print("Stampa del cromosoma dopo: ", mutated_array)
        return mutated_array  # Dovresti restituire l'array mutato, non l'originale
    """

    """Il punteggio di fitness viene calcolato sommando i valori restituiti dalla funzione _findOverlap per le 
    sovrapposizioni tra coppie di letture sequenziali nel cromosoma.
    
    In sintesi, la funzione _fitness somma le sovrapposizioni tra tutte le coppie di letture sequenziali nel 
    cromosoma, restituendo il punteggio totale di fitness del cromosoma. Questo punteggio sarà utilizzato per 
    valutare quanto bene il cromosoma si adatta all'ambiente o all'obiettivo specifico dell'algoritmo genetico."""

    def findOverlapNew(self, finger, finger1):
        # print("Finger e finger 1", finger, finger1)
        # print("Siamo nel metodo findOverlapNew")
        overlap = []
        min_len = min(len(finger), len(finger1))
        for i in range(1, min_len + 1):
            if finger[-i:] == finger1[:i]:
                # print(finger[-i:], "\n", finger1[:i])
                overlap = finger[-i:]
        # print("Overlap individuato tra :",finger, "\n", "e ", finger1, "\n", "e di", sum(overlap))
        # print("somma:", sum(overlap))
        return sum(overlap)

    def findOverlapGenoma(self, finger1, finger2):
        # print("Finger e finger 1", finger, finger1)
        # print("Siamo nel metodo findOverlapNew")
        overlap = []
        min_len = min(len(finger1), len(finger2))
        for i in range(1, min_len + 1):
            if finger1[-i:] == finger2[:i]:
                # print(finger[-i:], "\n", finger1[:i])
                overlap = finger1[-i:]
                # print("Ancora qui", overlap)
        # print("Overlap individuato tra :",finger, "\n", "e ", finger1, "\n", "e di", sum(overlap))
        # print("somma:", sum(overlap))
        return overlap

    def _fitness(self, cromossome):
        score = 0


        # Modifica
        # print("Popolazione", cromossome)
        valori = [val for key, val in cromossome]

        # print(valori)

        # Itera sulla lista di liste in maniera sequenziale
        for i in range(len(valori) - 1):
            # Stampa la lista corrente e la lista successiva
            ciccio = remove_zeros(valori[i])
            cecio = remove_zeros(valori[i + 1])
            # print("CIccio", ciccio)
            # print("Cecio", cecio)
            score += self.findOverlapNew(ciccio, cecio)

            # print("SCORE:", score)

        return score

    """
    La funzione _evaluatePopulation prende in input una popolazione di cromosomi (individui) e calcola i punteggi di fitness
        per ciascun individuo utilizzando la funzione _fitness. 
    Questi punteggi di fitness vengono quindi restituiti come un array di punteggi.
    """

    def _evaluatePopulation(self, population, gen):
        scores = np.zeros(len(population))

        #min_distance = float('inf')  # Inizializza la distanza minima come infinito
        min_genome = None  # Inizializza il genoma con la distanza minima
        popy = []
        print("----- Siamo dentro il metodo _evaluatePopulation ")
        print("----- Siamo nel metodo _fitness -----")
        # print("Stampa della popolazione: ", population)
        # fitness_map = {}
        for x in range(len(population)):
            #print("Popolazione ", population[x])
            scores[x] = self._fitness(population[x])
            #genomePopolation = assemble_genome_with_overlaps(population[x])
            #print("Due genomi da confrontare:", "\nGenoma Partenza: ", gen)
            #print("Genoma Ottenuto dalla Popolazione corrente: ", genomePopolation)
            #print("Distanza di Levenshtein: ", levenshtein(gen, genomePopolation))
            #current_distance = levenshtein(gen, genomePopolation)
            # Controlla se la distanza corrente è minore della distanza minima
            """
            if current_distance < min_distance:
                min_distance = current_distance  # Aggiorna la distanza minima
                min_genome = genomePopolation  # Aggiorna il genoma con la distanza minima
                popy = population[x]
            """
        indeMin = scores.argmin()
        population.pop(indeMin)
        scores = np.delete(scores, indeMin)

        # Estrai le chiavi e i valori dal dizionario
        # popolazione = list(fitness_map.keys())
        # sfitness_values = list(fitness_map.values())


        """
        # Crea un array con i numeri da 0 a len(fitness_values)-1
        population = np.arange(len(scores))

        # Crea un grafico a barre
        plt.bar(population, scores)

        # Aggiungi le etichette
        for i in range(len(scores)):
            plt.text(i, scores[i], str(scores[i]), ha='center')

        # Imposta le etichette dell'asse x
        plt.xticks(population)

        # Imposta i titoli
        plt.xlabel('Popolazione')
        plt.ylabel('Valore di Fitness')
        plt.title('Valori di Fitness per la Popolazione')

        # Mostra il grafico
        # plt.show()

        #print("Popy", popy)
        #popypopy = assemble_genome_with_overlaps(popy)

        # print("Distanza tra popypopy", levenshtein(popypopy, gen), "\nGenoma Popy:", popypopy)
        """
        return scores


"""
La funzione _ring è un metodo di selezione per un algoritmo genetico che implementa il torneo. 
In questo contesto, il torneo è formato da un gruppo casuale di individui (fighter) estratti dalla popolazione,
    e il vincitore è l'individuo con il fitness più alto all'interno di questo gruppo.
    -> Viene creato un array fighters che contiene gli indici degli individui selezionati casualmente dalla popolazione (pop_fitness)
    -> Viene ottenuto un sottoarray fit che contiene i valori di fitness corrispondenti agli indici estratti.
"""


def _ring(self, pop_fitness, ring_size):
    print("-----------------------")
    print("Siamo nel metodo _ring")
    # Si crea un array contenente indici casuali di dim = 3, estratti da pop_fitness
    fighters = np.random.choice(len(pop_fitness), size=self.ring_size, replace=False)
    print("Array contenente 3 indici casuali: ", fighters, "estratti da pop_fitness", pop_fitness)
    # Otteniamo un array contenente i valori fitness corrispondenti agli indici estratti (fighters)
    fit = pop_fitness[fighters]
    print("Array contenente i valori fitness degli indici estratti: ", fit)
    winner = fit.argmax()
    print("L'indice con il valore fitness più alto: ", winner)
    print("-----------------")
    # winner sarà l'indice dell'individuo vincente.
    print("Il valore e", fighters[winner])
    return fighters[winner]


# ------------------------ CFL ---------------------------------------------------------------------

def remove_zeros(array):
    # Use list comprehension to create a new list that only includes the non-zero elements
    no_zeros = [element for element in array if element != 0]
    return no_zeros


# CFL - Lyndon factorization - Duval's algorithm
def CFL(word, T):
    """
    CFL Duval's algorithm.
    """
    CFL_list = []
    k = 0

    while k < len(word):
        i = k + 1
        j = k + 2
        while True:
            if j == len(word) + 1 or word[j - 1] < word[i - 1]:
                while k < i:
                    # print(word[k:k + j - i])
                    CFL_list.append(word[k:k + j - i])
                    k = k + j - i
                break
            else:
                # if word[j-1] > word[i-1]:
                if word[j - 1] > word[i - 1]:
                    i = k + 1
                else:
                    i = i + 1
                j = j + 1
    return CFL_list


"""
    La funzione count_repeats prende in input una lista di stringhe reads e restituisce un dizionario repeats_count che 
        conta quante volte ciascun prefisso di lunghezza da 3 a max_length appare in reads.

    Per ogni lunghezza di prefisso da 3 a max_length, e per ogni stringa in reads, la funzione estrae il prefisso di 
        quella lunghezza e conta quante volte appare in reads, aggiungendo questo conteggio al dizionario repeats_count
"""


def count_repeats(reads):
    #print(reads)
    repeats_count = {}
    max_length = len(reads[0])

    for k in range(3, max_length + 1):
        for read in reads:
            prefix = read[:k]
            repeats_count.setdefault(prefix, 0)
            repeats_count[prefix] += sum(1 for r in reads if prefix in r)

    return repeats_count


def apply_CFL_to_reads(reads, markers1,markers2):
    CFL_array = []
    mappa = {}

    # Itera attraverso le letture
    for lettura in reads:
        # Trova l'indice dell'occorrenza del marcatore nella lettura
        indice_marcatore = lettura.find(markers1)
        #print("marcatore", markers1)
        #print("lettura", lettura)
        #print("indice marcatore", indice_marcatore)
        if indice_marcatore != -1:
            if indice_marcatore == 0:
                CFL_prima = CFL(lettura[: len(markers1)], markers1)
                #print("CFL_prima", CFL_prima)
                CFL_dopo = CFL(lettura[len(markers1):], markers1)
                #print("CFL_dopo", CFL_dopo)
                CFL_prima.extend(CFL_dopo)
                mappa[lettura] = CFL_prima
            else:
                # Applica l'algoritmo CFL alla parte prima del marcatore
                CFL_prima = CFL(lettura[:indice_marcatore], markers1)
                #print("CFL_prima", CFL_prima)
                # Applica l'algoritmo CFL alla parte dopo del marcatore
                CFL_dopo = CFL(lettura[indice_marcatore:], markers1)
                #print("CFL_dopo", CFL_dopo)
                # Aggiungi la CFL della parte prima e della parte dopo del marcatore all'array
                CFL_prima.extend(CFL_dopo)
                mappa[lettura] = CFL_prima
                # print("LETTURA", lettura)
        else:
            CFL_not = CFL(lettura[:], markers1)
            #print("CFL not", CFL_not)
            mappa[lettura] = [CFL_not]
        # print("CFL array:", CFL_array)
    # Stampa le CFL
    return mappa


# Given a list of factors return the fingerprint
def compute_fingerprint_by_list_factors(list_fact, reads):
    # print(list_fact)

    #print("-------------------------")

    lunghezze_array = []

    new_map = {}
    for chiave, valore in list_fact.items():
        new_list = []
        for x in valore:
            # Se x è una lista, calcola la lunghezza di ogni stringa nella lista
            if isinstance(x, list):
                new_list.extend([len(item) for item in x])
            # Altrimenti, calcola solo la lunghezza di x
            else:
                new_list.append(len(x))
        new_map[chiave] = new_list
    """
    for elemento in list_fact:
        lunghezze_elemento = [len(sub_elemento) for sub_elemento in elemento]
        lunghezze_array.append(lunghezze_elemento)

    # print("Array delle lunghezze:", lunghezze_array)

    # Trova la lunghezza massima tra gli elementi dell'array
    max_len = max(len(sublist) for sublist in lunghezze_array)

    # Aggiungi zeri a ciascun elemento dell'array fino a raggiungere la lunghezza massima
    for sublist in lunghezze_array:
        while len(sublist) < max_len:
            sublist.append(0)
    """
    #print("---------")
    #print(new_map)
    return new_map


def find_max_read_per_sequence(repeats_count):
    # Inizializziamo un nuovo dizionario per memorizzare le sequenze e i loro conteggi
    max_reads = {}

    # Iteriamo attraverso il dizionario dato
    for sequence, count in repeats_count.items():
        # Aggiungiamo la sequenza come chiave e il suo conteggio come valore
        max_reads[sequence] = count

    return max_reads


def qlearning(reads, episodes, genome=None, test_each_episode=False):
    ovr = OverlapResolver(reads)
    root = Node.createRootNode(ovr)
    factor = 1.0 / (len(reads) * max([len(read) for read in reads]))
    generations = episodes

    _cromosomeInt = []
    print("------------")
    #print(reads)
    sottosequenza_lunghezza = 100

    # Caso in cui abbiamo un dataset diviso in piu letture
    readsGenoma = ''.join(reads)
    reads = ''.join(reads)
    #print(readsGenoma)

    reads = createDataset(reads,sottosequenza_lunghezza)

    #print(len(count_repeats(reads)))

    # for x in reads:
    # print("Lettura: ", x)

    dict = count_repeats(reads)
    #print(dict)
    # print(count_repeats(reads))

    print("------------")
    marker = []
    #print("Le letture sono:", reads)


    max_reads = find_max_read_per_sequence(dict)
    sequenze_frequenti = sorted(max_reads.items(), key=lambda x: x[1], reverse=True)[:2]

    markers = chiavi = [sequenza for sequenza, valore in sequenze_frequenti]
    firstMark, secondMark = markers[0],markers[1]
    print("Combinazione di due marcatori", firstMark,secondMark)

    print("------------")

    results = apply_CFL_to_reads(reads,firstMark,secondMark)
    print("Risultato ottenuto:", results)

    _intA = compute_fingerprint_by_list_factors(results, reads)
    print("------------")
    #print("FingerPrint del Marcatore ", firstMark, secondMark, " è ", _intA)

    # Modificato da me, prima era 60
    num_ind = 16
    ga = GA()
    # Si inizializza una lista vuota memory che verrà utilizzata per memorizzare sequenze di letture in modo casuale.
    memory = []
    # inizializza aux come una lista di indici che corrispondono alle posizioni delle letture nella lista reads

    # Creo una lista di indici
    indices = list(range(len(_intA)))
    mescolamento = 70
    # Creo una lista vuota per contenere gli array mescolati
    shuffled_arrays = []

    # Modificia
    listaValori = list(_intA.values())
    arrayString = []
    #print(listaValori)

    # Mescolo gli indici e creo un nuovo array per 3 volte
    popolazioni = [(chiave, valore) for chiave, valore in _intA.items()]
    popolazioni_mescolate = []
    #print("Popolazioni", popolazioni)
    for _ in range(mescolamento):
        # Crea una copia delle popolazioni per non modificare l'originale
        popolazioni_copia = popolazioni.copy()
        # Mescola la copia
        random.shuffle(popolazioni_copia)
        # Aggiungi la popolazione mescolata al contenitore
        popolazioni_mescolate.append(popolazioni_copia)

    # Ora popolazioni_mescolate contiene tutte le tue popolazioni mescolate
    for i, popolazione in enumerate(popolazioni_mescolate):
        print(f"Popolazione mescolata {i + 1}: {popolazione}")

    print("----------")
    """
    Il risultato ottenuto dall'esecuzione del GA, è la popolazione evoluta
        Utilizziamo [0] poichè ci interessa estrarre la popolazione evoluta.
    """
    # print("Meomory:", memory)

    ind_evolved = list([ga.run_ga(None, popolazioni_mescolate, reads)][0])
    print("--------------------")
    print("Siamo nel metodo Q-learning: ")
    # print("inizializza aux come una lista di indici che corrispondono alle posizioni delle letture nella lista reads: ", indices)
    # slice = np.array(shuffled_arrays)
    # print("Memory, rappresenta la popolazione iniziale: ", slice)
    print("--------------------")
    print("Popolazione ottenuta tramite l'esecuzione dell'algoritmo genetico: ", ind_evolved)
    genomePopolation = assemble_genome_with_overlaps(ind_evolved)
    print("Due genomi da confrontare:\n","Genoma di partenza:", readsGenoma, "\nGenomae ottenuto dal GA:", genomePopolation)
    print("Distanza di Levenshtein: ", levenshtein(readsGenoma, genomePopolation))
    """
    for i in range(len(ind_evolved)):
        print("indice: ", i, "lettura: ", reads[i])
    """
    print("--------------------")
    #print("genoma", genome)
    test = test_ga(root, factor, genome, ind_evolved, reads)

    print("ind_evolved:", ind_evolved, "test_rw:", "%.5f" % test[1], "test:", test[0], "dist:", test[2])


def levenshtein(s, t, costs=(1, 1, 1)):
    """
            iterative_levenshtein(s, t) -> ldist
            ldist is the Levenshtein distance between the strings
            s and t.
            For all i and j, dist[i,j] will contain the Levenshtein
            distance between the first i characters of s and the
            first j characters of t

            costs: a tuple or a list with three integers (d, i, s)
                   where d defines the costs for a deletion
                         i defines the costs for an insertion and
                         s defines the costs for a substitution
        """
    rows = len(s) + 1
    cols = len(t) + 1
    deletes, inserts, substitutes = costs

    dist = [[0 for x in range(cols)] for x in range(rows)]
    # source prefixes can be transformed into empty strings
    # by deletions:
    for row in range(1, rows):
        dist[row][0] = row * deletes
    # target prefixes can be created from an empty source string
    # by inserting the characters
    for col in range(1, cols):
        dist[0][col] = col * inserts

    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0
            else:
                cost = substitutes
            dist[row][col] = min(dist[row - 1][col] + deletes,
                                 dist[row][col - 1] + inserts,
                                 dist[row - 1][col - 1] + cost)  # substitution
    return dist[rows - 1][cols - 1]


def assemble_genome_with_overlaps(reads):
    genome = reads[0][0]
    for i in range(len(reads) - 1):
        current_read, current_overlaps = reads[i]
        current_read2, current_overlaps2 = reads[i + 1]
        # print(current_read, current_overlaps, "\n", current_read2, current_overlaps2)

        lista = GA.findOverlapGenoma(None, current_overlaps, current_overlaps2)
        if (len(lista) > 0):
            stringa_concatenata = current_read2[sum(lista):]
            # print("Stringa conc", stringa_concatenata)
            genome += stringa_concatenata
            # print("Genoma", genome)
        else:
            genome += current_read2
            # print("Genoma", genome)
    return genome


def test_ga(root_node, factor, genome, ind_evolved, reads):
    def levenshtein(s, t, costs=(1, 1, 1)):
        """
            iterative_levenshtein(s, t) -> ldist
            ldist is the Levenshtein distance between the strings
            s and t.
            For all i and j, dist[i,j] will contain the Levenshtein
            distance between the first i characters of s and the
            first j characters of t

            costs: a tuple or a list with three integers (d, i, s)
                   where d defines the costs for a deletion
                         i defines the costs for an insertion and
                         s defines the costs for a substitution
        """
        rows = len(s) + 1
        cols = len(t) + 1
        deletes, inserts, substitutes = costs

        dist = [[0 for x in range(cols)] for x in range(rows)]
        # source prefixes can be transformed into empty strings
        # by deletions:
        for row in range(1, rows):
            dist[row][0] = row * deletes
        # target prefixes can be created from an empty source string
        # by inserting the characters
        for col in range(1, cols):
            dist[0][col] = col * inserts

        for col in range(1, cols):
            for row in range(1, rows):
                if s[row - 1] == t[col - 1]:
                    cost = 0
                else:
                    cost = substitutes
                dist[row][col] = min(dist[row - 1][col] + deletes,
                                     dist[row][col - 1] + inserts,
                                     dist[row - 1][col - 1] + cost)  # substitution
        return dist[rows - 1][cols - 1]

    print("----- Siamo nel metodo  test_ga -----")
    grafo = nx.DiGraph()
    cur_node = root_node

    actions = []
    ricompense = []
    chiavi = []
    total_reward = 0.0
    # Effettua una copia della popolazione evolutiva
    ind = ind_evolved[:]
    mappa = {}

    while True:
        if len(ind) == 0:
            break
        a = ind[0]
        del ind[0]
        # a = cur_node.get_max_action()
        actions.append(a)
        aux = cur_node.get_child(a)
        if aux is None:
            ricompense.append(total_reward)
            break
        cur_node = aux
        reward = 0.1 if cur_node.parent_node == root_node else cur_node.pairwise_overlap * factor
        reward += 1.0 if cur_node.is_leaf() else 0.0
        total_reward += reward
        ricompense.append(total_reward)
    print(actions)

    """
    for i in range(len(actions)):
        mappa[actions[i]] = None
    """

    """
    for chiave, valore in zip(actions, ricompense):
        mappa[chiave] = valore
    """
    for chiave, valore in mappa.items():
        print(f"Azione: {chiave}, Ricompensa: {valore}, Lettura: {reads[chiave]}")
    dist = None
    print("IL consenso ottenuto partendo dal nodo corrente e risalendo fino alla radice: ", cur_node.get_consensus())
    print("Il genoma: ", genome)
    if genome is not None:
        dist = levenshtein(cur_node.get_consensus(), genome)
    return actions, total_reward, dist


def createDataset(dataset, lunghezza_sottosequenza):

    # Definiamo i 4 caratteri
    #caratteri = ['A', 'C', 'G','T']

    # Creiamo un dataset di 2000 caratteri scelti casualmente tra i 4
    #dataset = ''.join(np.random.choice(caratteri) for _ in range(2000))

    #print(dataset)
    # Dividiamo il dataset in sottosequenze di 100 caratteri
    #sottosequenze = [dataset[i:i + 100] for i in range(0, len(dataset), 100)]
    #return sottosequenze

    # Calcola il numero massimo di pezzi sovrapposti
    sottosequenze = []
    indice_inizio = 0

    while indice_inizio < len(dataset) - lunghezza_sottosequenza + 1:
        # Genera la sottosequenza corrente
        sottosequenza = dataset[indice_inizio:indice_inizio + lunghezza_sottosequenza]
        #print("sottosequenza estratta", sottosequenza)
        sottosequenze.append(sottosequenza)
        #print("sottosequenza estratta", sottosequenze)
        # Seleziona un indice casuale per l'overlap
        indice_casuale = random.randint(1, lunghezza_sottosequenza - 1)
        #print("INDICE CASUALE", indice_casuale)
        # Aggiorna l'indice di inizio per la prossima sottosequenza
        indice_inizio += indice_casuale

        # Verifica se la fine della sequenza è stata raggiunta
        if indice_inizio + lunghezza_sottosequenza > len(dataset):
            # Aggiungi l'ultima sottosequenza che include la fine della sequenza
            sottosequenze.append(dataset[-lunghezza_sottosequenza:])
            break

    return sottosequenze


if __name__ == "__main__":
    if len(sys.argv) > 3:
        seed = int(sys.argv[3])
    else:
        seed = random.randrange(sys.maxsize)
    random.seed(seed)
    print(seed, file=sys.stderr)

    dataset = {}
    genome25 = "TACTAGCAATACGCTTGCGTTCGGT"
    genome50 = "CCTAACCATTTTAACAGCAACATAACAGGCTAAGAGGGGCCGGACACCCA"
    genome381 = "ATGGCAATATTAGGTTTAGGCACGGATATTGTGGAGATCGCTCGCATCGAAGCGGTGATCGCCCGATCCGGTGATCGCCTGGCACGCCGCGTATTAAGCGATAACGAATGGGCTATCTGGAAAACGCACCACCAGCCGGTGCGTTTTCTGGCGAAGCGTTTTGCTGTGAAAGAAGCCGCAGCAAAAGCGTTTGGCACCGGGATCCGCAATGGTCTGGCGTTTAATCAATTTGAAGTATTCAATGATGAGCTCGGCAAACCACGGCTACGGCTATGGGGCGAGGCATTAAAACTGGCGGAAAAGCTGGGCGTTGCAAATATGCATGTAACGCTGGCAGATGAGCGGCACTATGCTTGTGCCACGGTAATTATTGAAAGTTAA"
    genome567 = "ATGAGCAAAGCAGGTGCGTCGCTTGCGACCTGTTACGGCCCTGTCAGCGCCGACGTTATAGCAAAAGCAGAGAACATTCGTCTGCTGATCCTCGATGTCGATGGCGTACTGTCAGATGGCCTGATTTATATGGGCAATAATGGCGAAGAGCTGAAAGCGTTCAATGTTCGTGACGGTTATGGCATTCGTTGTGCGCTCACCTCTGATATTGAAGTCGCTATCATTACCGGGCGAAAGGCTAAACTGGTAGAAGATCGTTGTGCCACATTGGGGATCACTCACTTGTATCAGGGGCAGTCAAACAAACTGATCGCCTTTAGCGATCTGCTGGAAAAACTGGCGATTGCCCCGGAAAATGTGGCTTATGTCGGCGATGATCTCATCGACTGGCCGGTAATGGAAAAAGTGGGTTTAAGCGTCGCCGTGGCCGATGCGCATCCACTGTTGATCCCGCGCGCCGATTACGTGACGCGCATTGCTGGCGGTCGTGGCGCAGTGCGCGAAGTTTGCGACTTATTACTCCTGGCGCAGGGCAAACTGGATGAAGCCAAAGGGCAATCGATATGA"
    genome726 = 'ATGGCAACTGTTTCCATGCGCGACATGCTCAAGGCTGGTGTTCACTTCGGTCACCAGACCCGTTACTGGAACCCGAAAATGAAGCCGTTCATCTTCGGTGCGCGTAACAAAGTTCACATCATCAACCTTGAGAAAACTGTACCGATGTTCAACGAAGCTCTGGCTGAACTGAACAAGATTGCTTCTCGCAAAGGTAAAATCCTTTTCGTTGGTACTAAACGCGCTGCAAGCGAAGCGGTGAAAGACGCTGCTCTGAGCTGCGACCAGTTCTTCGTGAACCATCGCTGGCTGGGCGGTATGCTGACTAACTGGAAAACCGTTCGTCAGTCCATCAAACGTCTGAAAGACCTGGAAACTCAGTCTCAGGACGGTACTTTCGACAAGCTGACCAAGAAAGAAGCGCTGATGCGCACTCGTGAGCTGGAGAAACTGGAAAACAGCCTGGGCGGTATCAAAGACATGGGCGGTCTGCCGGACGCTCTGTTTGTAATCGATGCTGACCACGAACACATTGCTATCAAAGAAGCAAACAACCTGGGTATTCCGGTATTTGCTATCGTTGATACCAACTCTGATCCGGACGGTGTTGACTTCGTTATCCCGGGTAACGACGACGCAATCCGTGCTGTGACCCTGTACCTGGGCGCTGTTGCTGCAACCGTACGTGAAGGCCGTTCTCAGGATCTGGCTTCCCAGGCGGAAGAAAGCTTCGTAGAAGCTGAGTAA'
    genome930 = 'ATGACGCAATTTGCATTTGTGTTCCCTGGACAGGGTTCTCAAACCGTTGGAATGCTGGCTGATATGGCGGCGAGCTATCCAATTGTCGAAGAAACGTTTGCTGAAGCTTCTGCGGCGCTGGGCTACGACCTGTGGGCGCTGACCCAGCAGGGGCCAGCTGAAGAACTGAATAAAACCTGGCAAACTCAGCCTGCGCTGTTGACTGCATCTGTTGCGCTGTATCGCGTATGGCAGCAGCAGGGCGGTAAAGCACCGGCAATGATGGCCGGTCACAGCCTGGGGGAATACTCCGCGCTGGTTTGCGCTGGTGTGATTGATTTCGCTGATGCGGTGCGTCTGGTTGAGATGCGCGGCAAGTTCATGCAAGAAGCCGTACCGGAAGGCACGGGCGCTATGGCGGCAATCATCGGTCTGGATGATGCGTCTATTGCGAAAGCGTGTGAAGAAGCTGCAGAAGGTCAGGTCGTTTCTCCGGTAAACTTTAACTCTCCGGGACAGGTGGTTATTGCCGGTCATAAAGAAGCGGTTGAGCGTGCTGGCGCTGCCTGTAAAGCGGCGGGCGCAAAACGCGCGCTGCCGTTACCAGTGAGCGTACCGTCTCACTGTGCGCTGATGAAACCAGCAGCCGACAAACTGGCAGTAGAATTAGCGAAAATCACCTTTAACGCACCAACAGTTCCTGTTGTGAATAACGTTGATGTGAAATGCGAAACCAATGGTGATGCCATCCGTGACGCACTGGTACGTCAGTTGTATAACCCGGTTCAGTGGACGAAGTCTGTTGAGTACATGGCAGCGCAAGGCGTAGAACATCTCTATGAAGTCGGCCCGGGCAAAGTGCTTACTGGCCTGACGAAACGCATTGTCGACACCCTGACCGCCTCGGCGCTGAACGAACCTTCAGCGATGGCAGCGGCGCTCGAGCTTTAA'
    genome4224 = 'GTGAAAGATTTATTAAAGTTTCTGAAAGCGCAGACTAAAACCGAAGAGTTTGATGCGATCAAAATTGCTCTGGCTTCGCCAGACATGATCCGTTCATGGTCTTTCGGTGAAGTTAAAAAGCCGGAAACCATCAACTACCGTACGTTCAAACCAGAACGTGACGGCCTTTTCTGCGCCCGTATCTTTGGGCCGGTAAAAGATTACGAGTGCCTGTGCGGTAAGTACAAGCGCCTGAAACACCGTGGCGTCATCTGTGAGAAGTGCGGCGTTGAAGTGACCCAGACTAAAGTACGCCGTGAGCGTATGGGCCACATCGAACTGGCTTCCCCGACTGCGCACATCTGGTTCCTGAAATCGCTGCCGTCCCGTATCGGTCTGCTGCTCGATATGCCGCTGCGCGATATCGAACGCGTACTGTACTTTGAATCCTATGTGGTTATCGAAGGCGGTATGACCAACCTGGAACGTCAGCAGATCCTGACTGAAGAGCAGTATCTGGACGCGCTGGAAGAGTTCGGTGACGAATTCGACGCGAAGATGGGGGCGGAAGCAATCCAGGCTCTGCTGAAGAGCATGGATCTGGAGCAAGAGTGCGAACAGCTGCGTGAAGAGCTGAACGAAACCAACTCCGAAACCAAGCGTAAAAAGCTGACCAAGCGTATCAAACTGCTGGAAGCGTTCGTTCAGTCTGGTAACAAACCAGAGTGGATGATCCTGACCGTTCTGCCGGTACTGCCGCCAGATCTGCGTCCGCTGGTTCCGCTGGATGGTGGTCGTTTCGCGACTTCTGACCTGAACGATCTGTATCGTCGCGTCATTAACCGTAACAACCGTCTGAAACGTCTGCTGGATCTGGCTGCGCCGGACATCATCGTACGTAACGAAAAACGTATGCTGCAGGAAGCGGTAGACGCCCTGCTGGATAACGGTCGTCGCGGTCGTGCGATCACCGGTTCTAACAAGCGTCCTCTGAAATCTTTGGCCGACATGATCAAAGGTAAACAGGGTCGTTTCCGTCAGAACCTGCTCGGTAAGCGTGTTGACTACTCCGGTCGTTCTGTAATCACCGTAGGTCCATACCTGCGTCTGCATCAGTGCGGTCTGCCGAAGAAAATGGCACTGGAGCTGTTCAAACCGTTCATCTACGGCAAGCTGGAACTGCGTGGTCTTGCTACCACCATTAAAGCTGCGAAGAAAATGGTTGAGCGCGAAGAAGCTGTCGTTTGGGATATCCTGGACGAAGTTATCCGCGAACACCCGGTACTGCTGAACCGTGCACCGACTCTGCACCGTCTGGGTATCCAGGCATTTGAACCGGTACTGATCGAAGGTAAAGCTATCCAGCTGCACCCGCTGGTTTGTGCGGCATATAACGCCGACTTCGATGGTGACCAGATGGCTGTTCACGTACCGCTGACGCTGGAAGCCCAGCTGGAAGCGCGTGCGCTGATGATGTCTACCAACAACATCCTGTCCCCGGCGAACGGCGAACCAATCATCGTTCCGTCTCAGGACGTTGTACTGGGTCTGTACTACATGACCCGTGACTGTGTTAACGCCAAAGGCGAAGGCATGGTGCTGACTGGCCCGAAAGAAGCAGAACGTCTGTATCGCTCTGGTCTGGCTTCTCTGCATGCGCGCGTTAAAGTGCGTATCACCGAGTATGAAAAAGATGCTAACGGTGAATTAGTAGCGAAAACCAGCCTGAAAGACACGACTGTTGGCCGTGCCATTCTGTGGATGATTGTACCGAAAGGTCTGCCTTACTCCATCGTCAACCAGGCGCTGGGTAAAAAAGCAATCTCCAAAATGCTGAACACCTGCTACCGCATTCTCGGTCTGAAACCGACCGTTATTTTTGCGGACCAGATCATGTACACCGGCTTCGCCTATGCAGCGCGTTCTGGTGCATCTGTTGGTATCGATGACATGGTCATCCCGGAGAAGAAACACGAAATCATCTCCGAGGCAGAAGCAGAAGTTGCTGAAATTCAGGAGCAGTTCCAGTCTGGTCTGGTAACTGCGGGCGAACGCTACAACAAAGTTATCGATATCTGGGCTGCGGCGAACGATCGTGTATCCAAAGCGATGATGGATAACCTGCAAACTGAAACCGTGATTAACCGTGACGGTCAGGAAGAGAAGCAGGTTTCCTTCAACAGCATCTACATGATGGCCGACTCCGGTGCGCGTGGTTCTGCGGCACAGATTCGTCAGCTTGCTGGTATGCGTGGTCTGATGGCGAAGCCGGATGGCTCCATCATCGAAACGCCAATCACCGCGAACTTCCGTGAAGGTCTGAACGTACTCCAGTACTTCATCTCCACCCACGGTGCTCGTAAAGGTCTGGCGGATACCGCACTGAAAACTGCGAACTCCGGTTACCTGACTCGTCGTCTGGTTGACGTGGCGCAGGACCTGGTGGTTACCGAAGACGATTGTGGTACCCATGAAGGTATCATGATGACTCCGGTTATCGAGGGTGGTGACGTTAAAGAGCCGCTGCGCGATCGCGTACTGGGTCGTGTAACTGCTGAAGACGTTCTGAAGCCGGGTACTGCTGATATCCTCGTTCCGCGCAACACGCTGCTGCACGAACAGTGGTGTGACCTGCTGGAAGAGAACTCTGTCGACGCGGTTAAAGTACGTTCTGTTGTATCTTGTGACACCGACTTTGGTGTATGTGCGCACTGCTACGGTCGTGACCTGGCGCGTGGCCACATCATCAACAAGGGTGAAGCAATCGGTGTTATCGCGGCACAGTCCATCGGTGAACCGGGTACACAGCTGACCATGCGTACGTTCCACATCGGTGGTGCGGCATCTCGTGCGGCTGCTGAATCCAGCATCCAAGTGAAAAACAAAGGTAGCATCAAGCTCAGCAACGTGAAGTCGGTTGTGAACTCCAGCGGTAAACTGGTTATCACTTCCCGTAATACTGAACTGAAACTGATCGACGAATTCGGTCGTACTAAAGAAAGCTACAAAGTACCTTACGGTGCGGTACTGGCGAAAGGCGATGGCGAACAGGTTGCTGGCGGCGAAACCGTTGCAAACTGGGACCCGCACACCATGCCGGTTATCACCGAAGTAAGCGGTTTTGTACGCTTTACTGACATGATCGACGGCCAGACCATTACGCGTCAGACCGACGAACTGACCGGTCTGTCTTCGCTGGTGGTTCTGGATTCCGCAGAACGTACCGCAGGTGGTAAAGATCTGCGTCCGGCACTGAAAATCGTTGATGCTCAGGGTAACGACGTTCTGATCCCAGGTACCGATATGCCAGCGCAGTACTTCCTGCCGGGTAAAGCGATTGTTCAGCTGGAAGATGGCGTACAGATCAGCTCTGGTGACACCCTGGCGCGTATTCCGCAGGAATCCGGCGGTACCAAGGACATCACCGGTGGTCTGCCGCGCGTTGCGGACCTGTTCGAAGCACGTCGTCCGAAAGAGCCGGCAATCCTGGCTGAAATCAGCGGTATCGTTTCCTTCGGTAAAGAAACCAAAGGTAAACGTCGTCTGGTTATCACCCCGGTAGACGGTAGCGATCCGTACGAAGAGATGATTCCGAAATGGCGTCAGCTCAACGTGTTCGAAGGTGAACGTGTAGAACGTGGTGACGTAATTTCCGACGGTCCGGAAGCGCCGCACGACATTCTGCGTCTGCGTGGTGTTCATGCTGTTACTCGTTACATCGTTAACGAAGTACAGGACGTATACCGTCTGCAGGGCGTTAAGATTAACGATAAACACATCGAAGTTATCGTTCGTCAGATGCTGCGTAAAGCTACCATCGTTAACGCGGGTAGCTCCGACTTCCTGGAAGGCGAACAGGTTGAATACTCTCGCGTCAAGATCGCAAACCGCGAACTGGAAGCGAACGGCAAAGTGGGTGCAACTTACTCCCGCGATCTGCTGGGTATCACCAAAGCGTCTCTGGCAACCGAGTCCTTCATCTCCGCGGCATCGTTCCAGGAGACCACTCGCGTGCTGACCGAAGCAGCCGTTGCGGGCAAACGCGACGAACTGCGCGGCCTGAAAGAGAACGTTATCGTGGGTCGTCTGATCCCGGCAGGTACCGGTTACGCGTACCACCAGGATCGTATGCGTCGCCGTGCTGCGGGTGAAGCTCCGGCTGCACCGCAGGTGACTGCAGAAGACGCATCTGCCAGCCTGGCAGAACTGCTGAACGCAGGTCTGGGCGGTTCTGATAACGAGTAA'
    # reads = ['ACATTAGG', 'TTAGGCCCTT', 'GCCCTTA', 'CCTTACA']
    # reads = ['GCTTGCGT','GCGTTCGG','CTAGCAAT','GCAATACG','CTAGCAAT','TAGCAATA','CGTTCGGT','TACGCTTG','CTTGCGTT','TACGCTTG','TACTAGCA','ACGCTTGC','TGCGTTCG','CTAGCAAT','TGCGTTCG','CTAGCAAT','ACGCTTGC','TTGCGTTC','ATACGCTT','CGCTTGCG']
    # reads = ['CAACATAA','TTTAACAG','AGGCTAAG','GGGCCGGA','GGCTAAGA','AACATAAC','TAACAGCA','ACATAACA','AGGCTAAG','CAACATAA','GGGCCGGA','CCATTTTA','AACATAAC','CCATTTTA','CCTAACCA','GAGGGGCC','GACACCCA','TAACAGGC','TAACAGCA','GCCGGACA']
    # reads = ['GGGGCCGG','ACAGCAAC','AGAGGGGC','CCTAACCA','CCATTTTA','AAGAGGGG','TTAACAGC','AGGCTAAG','AGGGGCCG','GGGCCGGA','CCGGACAC','TAAGAGGG','CATAACAG','TTAACAGC','GAGGGGCC','TTTAACAG','CAGCAACA','GGGGCCGG','GGCTAAGA','CGGACACC','CCTAACCA','AACCATTT','AGGGGCCG','GACACCCA','AGCAACAT','CCGGACAC','ACCATTTT','TAAGAGGG','GGCCGGAC','GCTAAGAG']
    # gt()
    # manual(reads)
    # auto(reads)
    reads_25_10_8 = ['CGTTCGGT', 'TTGCGTTC', 'CTTGCGTT', 'ACGCTTGC', 'ATACGCTT', 'AATACGCT', 'AGCAATAC', 'CTAGCAAT',
                     'ACTAGCAA', 'TACTAGCA']
    reads_25_10_10 = ['TGCGTTCGGT', 'CTAGCAATAC', 'ACTAGCAATA', 'CAATACGCTT', 'GCTTGCGTTC', 'CTTGCGTTCG', 'GCTTGCGTTC',
                      'ACTAGCAATA', 'TACTAGCAAT', 'CTAGCAATAC']
    reads_25_10_15 = ['CAATACGCTTGCGTT', 'AGCAATACGCTTGCG', 'GCAATACGCTTGCGT', 'TACTAGCAATACGCT', 'ACGCTTGCGTTCGGT',
                      'ATACGCTTGCGTTCG', 'TAGCAATACGCTTGC', 'CTAGCAATACGCTTG', 'AATACGCTTGCGTTC', 'AGCAATACGCTTGCG']
    reads_50_10_8 = ['GGGGCCGG', 'GGACACCC', 'ATAACAGG', 'GACACCCA', 'AAGAGGGG', 'TTAACAGC', 'AACATAAC', 'ATTTTAAC',
                     'CCTAACCA', 'ACAGGCTA']
    reads_50_10_10 = ['CGGACACCCA', 'CCTAACCATT', 'TAACCATTTT', 'CTAAGAGGGG', 'ACATAACAGG', 'GGGCCGGACA', 'GAGGGGCCGG',
                      'TTAACAGCAA', 'CATAACAGGC', 'ACAGCAACAT']
    reads_50_10_15 = ['CAACATAACAGGCTA', 'CCTAACCATTTTAAC', 'ATTTTAACAGCAACA', 'GGGGCCGGACACCCA', 'CCATTTTAACAGCAA',
                      'TAACCATTTTAACAG', 'TAAGAGGGGCCGGAC', 'AACAGCAACATAACA', 'ACAGGCTAAGAGGGG', 'TTTTAACAGCAACAT']
    reads_25_20_8 = ['GCTTGCGT', 'GCGTTCGG', 'CTAGCAAT', 'GCAATACG', 'CTAGCAAT', 'TAGCAATA', 'CGTTCGGT', 'TACGCTTG',
                     'CTTGCGTT', 'TACGCTTG', 'TACTAGCA', 'ACGCTTGC', 'TGCGTTCG', 'CTAGCAAT', 'TGCGTTCG', 'CTAGCAAT',
                     'ACGCTTGC', 'TTGCGTTC', 'ATACGCTT', 'CGCTTGCG']
    reads_25_20_10 = ['CAATACGCTT', 'GCTTGCGTTC', 'AATACGCTTG', 'CAATACGCTT', 'TACTAGCAAT', 'ATACGCTTGC', 'TGCGTTCGGT',
                      'TAGCAATACG', 'AGCAATACGC', 'TACGCTTGCG', 'TACGCTTGCG', 'CTAGCAATAC', 'CAATACGCTT', 'ACGCTTGCGT',
                      'ATACGCTTGC', 'TAGCAATACG', 'ACGCTTGCGT', 'GCAATACGCT', 'CAATACGCTT', 'AGCAATACGC']
    reads_25_20_15 = ['ACGCTTGCGTTCGGT', 'AATACGCTTGCGTTC', 'ACGCTTGCGTTCGGT', 'AGCAATACGCTTGCG', 'TAGCAATACGCTTGC',
                      'AATACGCTTGCGTTC', 'ATACGCTTGCGTTCG', 'CTAGCAATACGCTTG', 'ACGCTTGCGTTCGGT', 'ACGCTTGCGTTCGGT',
                      'ATACGCTTGCGTTCG', 'ACGCTTGCGTTCGGT', 'ACGCTTGCGTTCGGT', 'TACTAGCAATACGCT', 'CTAGCAATACGCTTG',
                      'ACGCTTGCGTTCGGT', 'GCAATACGCTTGCGT', 'TACGCTTGCGTTCGG', 'CAATACGCTTGCGTT', 'TACGCTTGCGTTCGG']
    reads_50_20_8 = ['CAACATAA', 'TTTAACAG', 'AGGCTAAG', 'GGGCCGGA', 'GGCTAAGA', 'AACATAAC', 'TAACAGCA', 'ACATAACA',
                     'AGGCTAAG', 'CAACATAA', 'GGGCCGGA', 'CCATTTTA', 'AACATAAC', 'CCATTTTA', 'CCTAACCA', 'GAGGGGCC',
                     'GACACCCA', 'TAACAGGC', 'TAACAGCA', 'GCCGGACA']
    reads_50_20_10 = ['GAGGGGCCGG', 'TAACCATTTT', 'TAACAGGCTA', 'TTTTAACAGC', 'CATTTTAACA', 'ACCATTTTAA', 'GGCTAAGAGG',
                      'CAGCAACATA', 'AGAGGGGCCG', 'TAACCATTTT', 'CCATTTTAAC', 'AACAGGCTAA', 'CTAACCATTT', 'GGGCCGGACA',
                      'CCTAACCATT', 'CCATTTTAAC', 'TTTTAACAGC', 'CAACATAACA', 'ACATAACAGG', 'CGGACACCCA']
    reads_50_20_15 = ['ATAACAGGCTAAGAG', 'TTTTAACAGCAACAT', 'CAGGCTAAGAGGGGC', 'GGCTAAGAGGGGCCG', 'ATAACAGGCTAAGAG',
                      'CATAACAGGCTAAGA', 'GCTAAGAGGGGCCGG', 'TTTTAACAGCAACAT', 'CAACATAACAGGCTA', 'ACAGGCTAAGAGGGG',
                      'GGCTAAGAGGGGCCG', 'TAAGAGGGGCCGGAC', 'AACAGGCTAAGAGGG', 'TAACAGGCTAAGAGG', 'TAACAGCAACATAAC',
                      'CATAACAGGCTAAGA', 'CCTAACCATTTTAAC', 'GGGGCCGGACACCCA', 'ACAGCAACATAACAG', 'TTTTAACAGCAACAT']
    reads_25_30_8 = ['TACTAGCA', 'AATACGCT', 'CGTTCGGT', 'CGCTTGCG', 'AGCAATAC', 'TGCGTTCG', 'TAGCAATA', 'CTTGCGTT',
                     'TAGCAATA', 'CTAGCAAT', 'GCGTTCGG', 'TTGCGTTC', 'TTGCGTTC', 'TGCGTTCG', 'GCGTTCGG', 'TACGCTTG',
                     'CAATACGC', 'ACGCTTGC', 'GCTTGCGT', 'CGCTTGCG', 'ATACGCTT', 'CGTTCGGT', 'CGCTTGCG', 'GCAATACG',
                     'GCTTGCGT', 'ACGCTTGC', 'CTTGCGTT', 'TTGCGTTC', 'GCTTGCGT', 'TACTAGCA']
    reads_25_30_10 = ['TAGCAATACG', 'CGCTTGCGTT', 'TGCGTTCGGT', 'ACGCTTGCGT', 'CTAGCAATAC', 'ACTAGCAATA', 'TTGCGTTCGG',
                      'TGCGTTCGGT', 'ATACGCTTGC', 'CGCTTGCGTT', 'CTAGCAATAC', 'CTAGCAATAC', 'TAGCAATACG', 'GCAATACGCT',
                      'TACGCTTGCG', 'AGCAATACGC', 'GCTTGCGTTC', 'ACGCTTGCGT', 'ATACGCTTGC', 'CAATACGCTT', 'AATACGCTTG',
                      'TAGCAATACG', 'GCAATACGCT', 'TACGCTTGCG', 'AGCAATACGC', 'TAGCAATACG', 'CGCTTGCGTT', 'TAGCAATACG',
                      'TACTAGCAAT', 'GCTTGCGTTC']
    reads_25_30_15 = ['ACTAGCAATACGCTT', 'ACTAGCAATACGCTT', 'AATACGCTTGCGTTC', 'ACTAGCAATACGCTT', 'ACGCTTGCGTTCGGT',
                      'TACGCTTGCGTTCGG', 'AATACGCTTGCGTTC', 'AATACGCTTGCGTTC', 'CTAGCAATACGCTTG', 'TACTAGCAATACGCT',
                      'CTAGCAATACGCTTG', 'CAATACGCTTGCGTT', 'TACGCTTGCGTTCGG', 'AATACGCTTGCGTTC', 'AATACGCTTGCGTTC',
                      'CAATACGCTTGCGTT', 'TACGCTTGCGTTCGG', 'CAATACGCTTGCGTT', 'AATACGCTTGCGTTC', 'ACTAGCAATACGCTT',
                      'ACTAGCAATACGCTT', 'ACTAGCAATACGCTT', 'AGCAATACGCTTGCG', 'CTAGCAATACGCTTG', 'ATACGCTTGCGTTCG',
                      'GCAATACGCTTGCGT', 'ATACGCTTGCGTTCG', 'ACTAGCAATACGCTT', 'ATACGCTTGCGTTCG', 'TAGCAATACGCTTGC']
    reads_50_30_8 = ['GGGGCCGG', 'ACAGCAAC', 'AGAGGGGC', 'CCTAACCA', 'CCATTTTA', 'AAGAGGGG', 'TTAACAGC', 'AGGCTAAG',
                     'AGGGGCCG', 'GGGCCGGA', 'CCGGACAC', 'TAAGAGGG', 'CATAACAG', 'TTAACAGC', 'GAGGGGCC', 'TTTAACAG',
                     'CAGCAACA', 'GGGGCCGG', 'GGCTAAGA', 'CGGACACC', 'CCTAACCA', 'AACCATTT', 'AGGGGCCG', 'GACACCCA',
                     'AGCAACAT', 'CCGGACAC', 'ACCATTTT', 'TAAGAGGG', 'GGCCGGAC', 'GCTAAGAG']
    reads_50_30_10 = ['ACATAACAGG', 'CTAACCATTT', 'CAGGCTAAGA', 'AGCAACATAA', 'CTAAGAGGGG', 'AGAGGGGCCG', 'CAGGCTAAGA',
                      'CGGACACCCA', 'AACAGGCTAA', 'CAGCAACATA', 'GGCCGGACAC', 'GGCCGGACAC', 'TAACAGCAAC', 'CCTAACCATT',
                      'GGGGCCGGAC', 'CAGGCTAAGA', 'GGCTAAGAGG', 'TAAGAGGGGC', 'AACATAACAG', 'CAGCAACATA', 'TAACAGGCTA',
                      'TTTTAACAGC', 'ACCATTTTAA', 'AACATAACAG', 'AGAGGGGCCG', 'GCAACATAAC', 'TAACAGGCTA', 'GGCTAAGAGG',
                      'TAACCATTTT', 'CAGGCTAAGA']
    reads_50_30_15 = ['CTAAGAGGGGCCGGA', 'AACAGCAACATAACA', 'GGGGCCGGACACCCA', 'AGCAACATAACAGGC', 'AACAGGCTAAGAGGG',
                      'ATAACAGGCTAAGAG', 'TAACAGCAACATAAC', 'GAGGGGCCGGACACC', 'CTAAGAGGGGCCGGA', 'ATAACAGGCTAAGAG',
                      'TTTTAACAGCAACAT', 'CATTTTAACAGCAAC', 'CCTAACCATTTTAAC', 'TTTTAACAGCAACAT', 'GCAACATAACAGGCT',
                      'GAGGGGCCGGACACC', 'GGCTAAGAGGGGCCG', 'ATTTTAACAGCAACA', 'ACAGGCTAAGAGGGG', 'TTTAACAGCAACATA',
                      'CAACATAACAGGCTA', 'CAACATAACAGGCTA', 'TTTTAACAGCAACAT', 'AACATAACAGGCTAA', 'CATTTTAACAGCAAC',
                      'TAACCATTTTAACAG', 'AACATAACAGGCTAA', 'CTAAGAGGGGCCGGA', 'AGGCTAAGAGGGGCC', 'CTAAGAGGGGCCGGA']

    readProva = 'TCATATCCCTAGAGTGCAATAGCTGAGTGAGTAGCCGTAGGTTCTGCGCGATGCAGTGTCCCTGAATAATCCAAACAACCTCGCCGCGGTCGCATGCGCCGCACGAAAGCCGGAAACTATTCACCTCTGTTTACTGAATGCTATGCGGAGCAGGAACCAGCAATCCTCGATTGTCTCCAGCGTAAAGAAGTGTCGCGCTCTTCCTTGATCACTAACGCGCAGCGGTAGCAAGATCTGCTTTCTACGGTTACGCGAACCAAACAGACTTGGGCGGCCACCTGCAGGTCAAGTACTAATATATAAGCACGGGAATACCACATCATGACGTGAACGATCGCAGCCTTAAAGACAGAATGTATATGCCTAGGCCCGCATATGCCCAACGACTTACAATCGGTGTATCCCTCTAGGTTGAGATCAACAGGAGTAGTCACCTTGAACCTGATATTGGAAGAGCGTGGTGCTGCACACCAAGGTGATCGGAGGTACGTGCAGGGTTACTAGCGATGCAGCAGGCAATGATTTGTTACTTATATCATTGTACGCAACAAGGTTGTGGGGAGGTTGCGTAAATCGGCGGCGCCCCGCCTTCCTCTACCCGGACATCGATTTTTCCGACCTCCACGAGAACTACTCGAGAACCCGAGCCTGAGTAAACCGGTATACAACTCTAGGCAAGTGCGCTACCCCTTTTACGCGTGAACGGAGCCGCTTTTCCCCCATAGTGCGTAAAGCGGTATGTTTAAATTTACTGTGGCGTTATGCGTCGCAGGTGTATGACCGGCTCGTCAGCGGCCACAGGCATCACGTAATATTTAGCGCTGGTCTTTGTTTTCTGTGATCGAATGGAAGGAGTCATTTATGCCACGAGGATATGACGAATAGTCTATCGTCTGCTAGGCAAGGTAAAAAAGTCAAGAATGAGACGGTGTTTGGCGCTATACCCCACTACAGAAACATATTGCTGCCCCGCGGCTCATGTCGTGCTGGGGTCCCTGTATAACAGCTGACACGACAAGCCGAGGCATCTATGACATCGACTAAAACTCTGGGTCGCGTTATGGTGGACCAGGCACGTACGGGGCGTAGCGCCTATTAAATTAGTCCAAAAGACATTTTTTGGTGACAGTGCTGCCCGACGACGTCCCTAGAATAACCAAAATAGGTCACAAAATATTGTCTTGTTCATGATAATCGATCTTTTTTTGGCAAAGCATCAGAAGTCTACCAGTCAGTTCTTAGCCCAGTGAGAGGGTGATTGGGCGCCAGATCGTAGTCAAATTACGGAGACGATTCTTTGCGTAAAATTGCTCCCGTGAGGGCGAGAATCGGAACAGCGACGATTTATTGCGGCGCGACTCGGGAGATTGACAGGAATACCGAATGGCTAGCTTGTAAATTTAAATAGGAATCCATTGTTCCTAAAGCAGATTAGCGCCGATCCGAGCGTAAACCGGCCGCTGAACGCACGGCGTCATCTGGTTGAACTACTATTGGTAGTAGGAATCACATATGGGTGGTTACTTGTTAGCTTTGTACGCATTGGTTATTCCGCAAAAGGTACAGACTGAACCACTATGTAGCATCCATGTTCTCGATGGCACAAGTTCTCACATGTACGTCATCACGGCACCTGACGCCTAGTTGACCAAAATCTCCGTTGCGGCGACAAACGGCTTCCCTATGAAACGGCATGCAGTCATTTCGGCACACGAGATATTGGGGACAGTGCCTAACTCTCGGTGCCCCTTTTAAAGCAAAATGATGCTTGGTGGCTGGTTACAAAGCCCAGCAGGCATCTCGGATAGTTGTCGCATTTTCTGTCGACAATCGTGACTAGTTGATCTGCACACATAGATGGGCTTACTCCATGCGGCATTTACGCTATCGTATCGGTCATTTACACTACTGCAGGACAGCGAGCGGGGCGTCCATCGAACATGAAGTTCAGGACGGCAACGTGTGGTTAATGTCCTGCGAAGCTTTAACTTAAAGGCGAT'
    #readProva = 'TCACTTABCD'



    reads_381_20_75 = ['AAAGAAGCCGCAGCAAAAGCGTTTGGCACCGGGATCCGCAATGGTCTGGCGTTTAATCAATTTGAAGTATTCAAT',
                       'GGCGTTGCAAATATGCATGTAACGCTGGCAGATGAGCGGCACTATGCTTGTGCCACGGTAATTATTGAAAGTTAA',
                       'AAGCGTTTGGCACCGGGATCCGCAATGGTCTGGCGTTTAATCAATTTGAAGTATTCAATGATGAGCTCGGCAAAC',
                       'GCAGCAAAAGCGTTTGGCACCGGGATCCGCAATGGTCTGGCGTTTAATCAATTTGAAGTATTCAATGATGAGCTC',
                       'ATGATGAGCTCGGCAAACCACGGCTACGGCTATGGGGCGAGGCATTAAAACTGGCGGAAAAGCTGGGCGTTGCAA',
                       'CTGGCACGCCGCGTATTAAGCGATAACGAATGGGCTATCTGGAAAACGCACCACCAGCCGGTGCGTTTTCTGGCG',
                       'TGAAGTATTCAATGATGAGCTCGGCAAACCACGGCTACGGCTATGGGGCGAGGCATTAAAACTGGCGGAAAAGCT',
                       'GCACCGGGATCCGCAATGGTCTGGCGTTTAATCAATTTGAAGTATTCAATGATGAGCTCGGCAAACCACGGCTAC',
                       'GGCGAGGCATTAAAACTGGCGGAAAAGCTGGGCGTTGCAAATATGCATGTAACGCTGGCAGATGAGCGGCACTAT',
                       'GGCAATATTAGGTTTAGGCACGGATATTGTGGAGATCGCTCGCATCGAAGCGGTGATCGCCCGATCCGGTGATCG',
                       'TTAGGCACGGATATTGTGGAGATCGCTCGCATCGAAGCGGTGATCGCCCGATCCGGTGATCGCCTGGCACGCCGC',
                       'TATCTGGAAAACGCACCACCAGCCGGTGCGTTTTCTGGCGAAGCGTTTTGCTGTGAAAGAAGCCGCAGCAAAAGC',
                       'ATGGCAATATTAGGTTTAGGCACGGATATTGTGGAGATCGCTCGCATCGAAGCGGTGATCGCCCGATCCGGTGAT',
                       'GCGAGGCATTAAAACTGGCGGAAAAGCTGGGCGTTGCAAATATGCATGTAACGCTGGCAGATGAGCGGCACTATG',
                       'GGTGCGTTTTCTGGCGAAGCGTTTTGCTGTGAAAGAAGCCGCAGCAAAAGCGTTTGGCACCGGGATCCGCAATGG',
                       'TTAATCAATTTGAAGTATTCAATGATGAGCTCGGCAAACCACGGCTACGGCTATGGGGCGAGGCATTAAAACTGG',
                       'GCGTTTTCTGGCGAAGCGTTTTGCTGTGAAAGAAGCCGCAGCAAAAGCGTTTGGCACCGGGATCCGCAATGGTCT',
                       'TTTTCTGGCGAAGCGTTTTGCTGTGAAAGAAGCCGCAGCAAAAGCGTTTGGCACCGGGATCCGCAATGGTCTGGC',
                       'CCGATCCGGTGATCGCCTGGCACGCCGCGTATTAAGCGATAACGAATGGGCTATCTGGAAAACGCACCACCAGCC',
                       'GGGCGTTGCAAATATGCATGTAACGCTGGCAGATGAGCGGCACTATGCTTGTGCCACGGTAATTATTGAAAGTTA']
    reads_567_30_75 = ['GTCGCCGTGGCCGATGCGCATCCACTGTTGATCCCGCGCGCCGATTACGTGACGCGCATTGCTGGCGGTCGTGGC',
                       'GTATCAGGGGCAGTCAAACAAACTGATCGCCTTTAGCGATCTGCTGGAAAAACTGGCGATTGCCCCGGAAAATGT',
                       'GATCCCGCGCGCCGATTACGTGACGCGCATTGCTGGCGGTCGTGGCGCAGTGCGCGAAGTTTGCGACTTATTACT',
                       'ATGAGCAAAGCAGGTGCGTCGCTTGCGACCTGTTACGGCCCTGTCAGCGCCGACGTTATAGCAAAAGCAGAGAAC',
                       'TGTGCGCTCACCTCTGATATTGAAGTCGCTATCATTACCGGGCGAAAGGCTAAACTGGTAGAAGATCGTTGTGCC',
                       'TCCACTGTTGATCCCGCGCGCCGATTACGTGACGCGCATTGCTGGCGGTCGTGGCGCAGTGCGCGAAGTTTGCGA',
                       'GCGTACTGTCAGATGGCCTGATTTATATGGGCAATAATGGCGAAGAGCTGAAAGCGTTCAATGTTCGTGACGGTT',
                       'GTGACGCGCATTGCTGGCGGTCGTGGCGCAGTGCGCGAAGTTTGCGACTTATTACTCCTGGCGCAGGGCAAACTG',
                       'TGGCGAAGAGCTGAAAGCGTTCAATGTTCGTGACGGTTATGGCATTCGTTGTGCGCTCACCTCTGATATTGAAGT',
                       'ACGCGCATTGCTGGCGGTCGTGGCGCAGTGCGCGAAGTTTGCGACTTATTACTCCTGGCGCAGGGCAAACTGGAT',
                       'CGCCGTGGCCGATGCGCATCCACTGTTGATCCCGCGCGCCGATTACGTGACGCGCATTGCTGGCGGTCGTGGCGC',
                       'CTCACTTGTATCAGGGGCAGTCAAACAAACTGATCGCCTTTAGCGATCTGCTGGAAAAACTGGCGATTGCCCCGG',
                       'TAAGCGTCGCCGTGGCCGATGCGCATCCACTGTTGATCCCGCGCGCCGATTACGTGACGCGCATTGCTGGCGGTC',
                       'GCGCATCCACTGTTGATCCCGCGCGCCGATTACGTGACGCGCATTGCTGGCGGTCGTGGCGCAGTGCGCGAAGTT',
                       'TTTAGCGATCTGCTGGAAAAACTGGCGATTGCCCCGGAAAATGTGGCTTATGTCGGCGATGATCTCATCGACTGG',
                       'GGCTTATGTCGGCGATGATCTCATCGACTGGCCGGTAATGGAAAAAGTGGGTTTAAGCGTCGCCGTGGCCGATGC',
                       'GCGTTCAATGTTCGTGACGGTTATGGCATTCGTTGTGCGCTCACCTCTGATATTGAAGTCGCTATCATTACCGGG',
                       'TCCTCGATGTCGATGGCGTACTGTCAGATGGCCTGATTTATATGGGCAATAATGGCGAAGAGCTGAAAGCGTTCA',
                       'TCGTTGTGCGCTCACCTCTGATATTGAAGTCGCTATCATTACCGGGCGAAAGGCTAAACTGGTAGAAGATCGTTG',
                       'GCAGTGCGCGAAGTTTGCGACTTATTACTCCTGGCGCAGGGCAAACTGGATGAAGCCAAAGGGCAATCGATATGA',
                       'AGGTGCGTCGCTTGCGACCTGTTACGGCCCTGTCAGCGCCGACGTTATAGCAAAAGCAGAGAACATTCGTCTGCT',
                       'GCAGTGCGCGAAGTTTGCGACTTATTACTCCTGGCGCAGGGCAAACTGGATGAAGCCAAAGGGCAATCGATATGA',
                       'TCGTTGTGCCACATTGGGGATCACTCACTTGTATCAGGGGCAGTCAAACAAACTGATCGCCTTTAGCGATCTGCT',
                       'GCCACATTGGGGATCACTCACTTGTATCAGGGGCAGTCAAACAAACTGATCGCCTTTAGCGATCTGCTGGAAAAA',
                       'CGAAGAGCTGAAAGCGTTCAATGTTCGTGACGGTTATGGCATTCGTTGTGCGCTCACCTCTGATATTGAAGTCGC',
                       'ATGATCTCATCGACTGGCCGGTAATGGAAAAAGTGGGTTTAAGCGTCGCCGTGGCCGATGCGCATCCACTGTTGA',
                       'TATCAGGGGCAGTCAAACAAACTGATCGCCTTTAGCGATCTGCTGGAAAAACTGGCGATTGCCCCGGAAAATGTG',
                       'CGTTCAATGTTCGTGACGGTTATGGCATTCGTTGTGCGCTCACCTCTGATATTGAAGTCGCTATCATTACCGGGC',
                       'TCAGCGCCGACGTTATAGCAAAAGCAGAGAACATTCGTCTGCTGATCCTCGATGTCGATGGCGTACTGTCAGATG',
                       'AAACAAACTGATCGCCTTTAGCGATCTGCTGGAAAAACTGGCGATTGCCCCGGAAAATGTGGCTTATGTCGGCGA']
    reads_726_40_75 = ['AAAATGAAGCCGTTCATCTTCGGTGCGCGTAACAAAGTTCACATCATCAACCTTGAGAAAACTGTACCGATGTTC',
                       'CCCTGTACCTGGGCGCTGTTGCTGCAACCGTACGTGAAGGCCGTTCTCAGGATCTGGCTTCCCAGGCGGAAGAAA',
                       'AGCTGCGACCAGTTCTTCGTGAACCATCGCTGGCTGGGCGGTATGCTGACTAACTGGAAAACCGTTCGTCAGTCC',
                       'TCTCGCAAAGGTAAAATCCTTTTCGTTGGTACTAAACGCGCTGCAAGCGAAGCGGTGAAAGACGCTGCTCTGAGC',
                       'CTAACTGGAAAACCGTTCGTCAGTCCATCAAACGTCTGAAAGACCTGGAAACTCAGTCTCAGGACGGTACTTTCG',
                       'TCAACGAAGCTCTGGCTGAACTGAACAAGATTGCTTCTCGCAAAGGTAAAATCCTTTTCGTTGGTACTAAACGCG',
                       'AAAGACGCTGCTCTGAGCTGCGACCAGTTCTTCGTGAACCATCGCTGGCTGGGCGGTATGCTGACTAACTGGAAA',
                       'CAAAGACATGGGCGGTCTGCCGGACGCTCTGTTTGTAATCGATGCTGACCACGAACACATTGCTATCAAAGAAGC',
                       'CGAACACATTGCTATCAAAGAAGCAAACAACCTGGGTATTCCGGTATTTGCTATCGTTGATACCAACTCTGATCC',
                       'GACTTCGTTATCCCGGGTAACGACGACGCAATCCGTGCTGTGACCCTGTACCTGGGCGCTGTTGCTGCAACCGTA',
                       'CCGAAAATGAAGCCGTTCATCTTCGGTGCGCGTAACAAAGTTCACATCATCAACCTTGAGAAAACTGTACCGATG',
                       'TTTCGACAAGCTGACCAAGAAAGAAGCGCTGATGCGCACTCGTGAGCTGGAGAAACTGGAAAACAGCCTGGGCGG',
                       'CGCTGCAAGCGAAGCGGTGAAAGACGCTGCTCTGAGCTGCGACCAGTTCTTCGTGAACCATCGCTGGCTGGGCGG',
                       'GATACCAACTCTGATCCGGACGGTGTTGACTTCGTTATCCCGGGTAACGACGACGCAATCCGTGCTGTGACCCTG',
                       'GAAGCTCTGGCTGAACTGAACAAGATTGCTTCTCGCAAAGGTAAAATCCTTTTCGTTGGTACTAAACGCGCTGCA',
                       'CGGTGAAAGACGCTGCTCTGAGCTGCGACCAGTTCTTCGTGAACCATCGCTGGCTGGGCGGTATGCTGACTAACT',
                       'CCTGGAAACTCAGTCTCAGGACGGTACTTTCGACAAGCTGACCAAGAAAGAAGCGCTGATGCGCACTCGTGAGCT',
                       'CAGGACGGTACTTTCGACAAGCTGACCAAGAAAGAAGCGCTGATGCGCACTCGTGAGCTGGAGAAACTGGAAAAC',
                       'AATGAAGCCGTTCATCTTCGGTGCGCGTAACAAAGTTCACATCATCAACCTTGAGAAAACTGTACCGATGTTCAA',
                       'ATGGCAACTGTTTCCATGCGCGACATGCTCAAGGCTGGTGTTCACTTCGGTCACCAGACCCGTTACTGGAACCCG',
                       'TCGTGAACCATCGCTGGCTGGGCGGTATGCTGACTAACTGGAAAACCGTTCGTCAGTCCATCAAACGTCTGAAAG',
                       'GCTGCAACCGTACGTGAAGGCCGTTCTCAGGATCTGGCTTCCCAGGCGGAAGAAAGCTTCGTAGAAGCTGAGTAA',
                       'TACTGGAACCCGAAAATGAAGCCGTTCATCTTCGGTGCGCGTAACAAAGTTCACATCATCAACCTTGAGAAAACT',
                       'ATCAAAGAAGCAAACAACCTGGGTATTCCGGTATTTGCTATCGTTGATACCAACTCTGATCCGGACGGTGTTGAC',
                       'GACTTCGTTATCCCGGGTAACGACGACGCAATCCGTGCTGTGACCCTGTACCTGGGCGCTGTTGCTGCAACCGTA',
                       'GACCACGAACACATTGCTATCAAAGAAGCAAACAACCTGGGTATTCCGGTATTTGCTATCGTTGATACCAACTCT',
                       'GGTAAAATCCTTTTCGTTGGTACTAAACGCGCTGCAAGCGAAGCGGTGAAAGACGCTGCTCTGAGCTGCGACCAG',
                       'GCTGCAACCGTACGTGAAGGCCGTTCTCAGGATCTGGCTTCCCAGGCGGAAGAAAGCTTCGTAGAAGCTGAGTAA',
                       'TCCTTTTCGTTGGTACTAAACGCGCTGCAAGCGAAGCGGTGAAAGACGCTGCTCTGAGCTGCGACCAGTTCTTCG',
                       'TTCCATGCGCGACATGCTCAAGGCTGGTGTTCACTTCGGTCACCAGACCCGTTACTGGAACCCGAAAATGAAGCC',
                       'TATTTGCTATCGTTGATACCAACTCTGATCCGGACGGTGTTGACTTCGTTATCCCGGGTAACGACGACGCAATCC',
                       'CGTTGATACCAACTCTGATCCGGACGGTGTTGACTTCGTTATCCCGGGTAACGACGACGCAATCCGTGCTGTGAC',
                       'CGTCAGTCCATCAAACGTCTGAAAGACCTGGAAACTCAGTCTCAGGACGGTACTTTCGACAAGCTGACCAAGAAA',
                       'CTTTCGACAAGCTGACCAAGAAAGAAGCGCTGATGCGCACTCGTGAGCTGGAGAAACTGGAAAACAGCCTGGGCG',
                       'TGACTTCGTTATCCCGGGTAACGACGACGCAATCCGTGCTGTGACCCTGTACCTGGGCGCTGTTGCTGCAACCGT',
                       'TCAGTCTCAGGACGGTACTTTCGACAAGCTGACCAAGAAAGAAGCGCTGATGCGCACTCGTGAGCTGGAGAAACT',
                       'ATCCTTTTCGTTGGTACTAAACGCGCTGCAAGCGAAGCGGTGAAAGACGCTGCTCTGAGCTGCGACCAGTTCTTC',
                       'CGTTACTGGAACCCGAAAATGAAGCCGTTCATCTTCGGTGCGCGTAACAAAGTTCACATCATCAACCTTGAGAAA',
                       'CACTCGTGAGCTGGAGAAACTGGAAAACAGCCTGGGCGGTATCAAAGACATGGGCGGTCTGCCGGACGCTCTGTT',
                       'CGCAAAGGTAAAATCCTTTTCGTTGGTACTAAACGCGCTGCAAGCGAAGCGGTGAAAGACGCTGCTCTGAGCTGC']
    reads_930_50_75 = ['TTAGCGAAAATCACCTTTAACGCACCAACAGTTCCTGTTGTGAATAACGTTGATGTGAAATGCGAAACCAATGGT',
                       'TACGTCAGTTGTATAACCCGGTTCAGTGGACGAAGTCTGTTGAGTACATGGCAGCGCAAGGCGTAGAACATCTCT',
                       'TGGCCTGACGAAACGCATTGTCGACACCCTGACCGCCTCGGCGCTGAACGAACCTTCAGCGATGGCAGCGGCGCT',
                       'CAAGTTCATGCAAGAAGCCGTACCGGAAGGCACGGGCGCTATGGCGGCAATCATCGGTCTGGATGATGCGTCTAT',
                       'ATGACGCAATTTGCATTTGTGTTCCCTGGACAGGGTTCTCAAACCGTTGGAATGCTGGCTGATATGGCGGCGAGC',
                       'CGGTTCAGTGGACGAAGTCTGTTGAGTACATGGCAGCGCAAGGCGTAGAACATCTCTATGAAGTCGGCCCGGGCA',
                       'TTCGCTGATGCGGTGCGTCTGGTTGAGATGCGCGGCAAGTTCATGCAAGAAGCCGTACCGGAAGGCACGGGCGCT',
                       'TTAGCGAAAATCACCTTTAACGCACCAACAGTTCCTGTTGTGAATAACGTTGATGTGAAATGCGAAACCAATGGT',
                       'GGAATGCTGGCTGATATGGCGGCGAGCTATCCAATTGTCGAAGAAACGTTTGCTGAAGCTTCTGCGGCGCTGGGC',
                       'AAACGCATTGTCGACACCCTGACCGCCTCGGCGCTGAACGAACCTTCAGCGATGGCAGCGGCGCTCGAGCTTTAA',
                       'TGATTTCGCTGATGCGGTGCGTCTGGTTGAGATGCGCGGCAAGTTCATGCAAGAAGCCGTACCGGAAGGCACGGG',
                       'TGGCAAACTCAGCCTGCGCTGTTGACTGCATCTGTTGCGCTGTATCGCGTATGGCAGCAGCAGGGCGGTAAAGCA',
                       'TTCCCTGGACAGGGTTCTCAAACCGTTGGAATGCTGGCTGATATGGCGGCGAGCTATCCAATTGTCGAAGAAACG',
                       'TCGGCCCGGGCAAAGTGCTTACTGGCCTGACGAAACGCATTGTCGACACCCTGACCGCCTCGGCGCTGAACGAAC',
                       'CTGAAGCTTCTGCGGCGCTGGGCTACGACCTGTGGGCGCTGACCCAGCAGGGGCCAGCTGAAGAACTGAATAAAA',
                       'AAGCTGCAGAAGGTCAGGTCGTTTCTCCGGTAAACTTTAACTCTCCGGGACAGGTGGTTATTGCCGGTCATAAAG',
                       'ACCAGTGAGCGTACCGTCTCACTGTGCGCTGATGAAACCAGCAGCCGACAAACTGGCAGTAGAATTAGCGAAAAT',
                       'TGAAACCAGCAGCCGACAAACTGGCAGTAGAATTAGCGAAAATCACCTTTAACGCACCAACAGTTCCTGTTGTGA',
                       'TATCCAATTGTCGAAGAAACGTTTGCTGAAGCTTCTGCGGCGCTGGGCTACGACCTGTGGGCGCTGACCCAGCAG',
                       'TTTCTCCGGTAAACTTTAACTCTCCGGGACAGGTGGTTATTGCCGGTCATAAAGAAGCGGTTGAGCGTGCTGGCG',
                       'GTGCTGGCGCTGCCTGTAAAGCGGCGGGCGCAAAACGCGCGCTGCCGTTACCAGTGAGCGTACCGTCTCACTGTG',
                       'CCGGAAGGCACGGGCGCTATGGCGGCAATCATCGGTCTGGATGATGCGTCTATTGCGAAAGCGTGTGAAGAAGCT',
                       'AGCGGTTGAGCGTGCTGGCGCTGCCTGTAAAGCGGCGGGCGCAAAACGCGCGCTGCCGTTACCAGTGAGCGTACC',
                       'AGCAGCCGACAAACTGGCAGTAGAATTAGCGAAAATCACCTTTAACGCACCAACAGTTCCTGTTGTGAATAACGT',
                       'CATGGCAGCGCAAGGCGTAGAACATCTCTATGAAGTCGGCCCGGGCAAAGTGCTTACTGGCCTGACGAAACGCAT',
                       'TTACCAGTGAGCGTACCGTCTCACTGTGCGCTGATGAAACCAGCAGCCGACAAACTGGCAGTAGAATTAGCGAAA',
                       'AAACTTTAACTCTCCGGGACAGGTGGTTATTGCCGGTCATAAAGAAGCGGTTGAGCGTGCTGGCGCTGCCTGTAA',
                       'GCATTTGTGTTCCCTGGACAGGGTTCTCAAACCGTTGGAATGCTGGCTGATATGGCGGCGAGCTATCCAATTGTC',
                       'GCGAAACCAATGGTGATGCCATCCGTGACGCACTGGTACGTCAGTTGTATAACCCGGTTCAGTGGACGAAGTCTG',
                       'GGCTGATATGGCGGCGAGCTATCCAATTGTCGAAGAAACGTTTGCTGAAGCTTCTGCGGCGCTGGGCTACGACCT',
                       'CGGGCGCTATGGCGGCAATCATCGGTCTGGATGATGCGTCTATTGCGAAAGCGTGTGAAGAAGCTGCAGAAGGTC',
                       'ACCGTTGGAATGCTGGCTGATATGGCGGCGAGCTATCCAATTGTCGAAGAAACGTTTGCTGAAGCTTCTGCGGCG',
                       'ATCCAATTGTCGAAGAAACGTTTGCTGAAGCTTCTGCGGCGCTGGGCTACGACCTGTGGGCGCTGACCCAGCAGG',
                       'TGACGCAATTTGCATTTGTGTTCCCTGGACAGGGTTCTCAAACCGTTGGAATGCTGGCTGATATGGCGGCGAGCT',
                       'GTGCGTCTGGTTGAGATGCGCGGCAAGTTCATGCAAGAAGCCGTACCGGAAGGCACGGGCGCTATGGCGGCAATC',
                       'AAGGCACGGGCGCTATGGCGGCAATCATCGGTCTGGATGATGCGTCTATTGCGAAAGCGTGTGAAGAAGCTGCAG',
                       'TGTGAATAACGTTGATGTGAAATGCGAAACCAATGGTGATGCCATCCGTGACGCACTGGTACGTCAGTTGTATAA',
                       'TGGCAGTAGAATTAGCGAAAATCACCTTTAACGCACCAACAGTTCCTGTTGTGAATAACGTTGATGTGAAATGCG',
                       'TGGCAGCGCAAGGCGTAGAACATCTCTATGAAGTCGGCCCGGGCAAAGTGCTTACTGGCCTGACGAAACGCATTG',
                       'AACCAGCAGCCGACAAACTGGCAGTAGAATTAGCGAAAATCACCTTTAACGCACCAACAGTTCCTGTTGTGAATA',
                       'ATGCTGGCTGATATGGCGGCGAGCTATCCAATTGTCGAAGAAACGTTTGCTGAAGCTTCTGCGGCGCTGGGCTAC',
                       'GAAATGCGAAACCAATGGTGATGCCATCCGTGACGCACTGGTACGTCAGTTGTATAACCCGGTTCAGTGGACGAA',
                       'GTTCCCTGGACAGGGTTCTCAAACCGTTGGAATGCTGGCTGATATGGCGGCGAGCTATCCAATTGTCGAAGAAAC',
                       'CTCCGGTAAACTTTAACTCTCCGGGACAGGTGGTTATTGCCGGTCATAAAGAAGCGGTTGAGCGTGCTGGCGCTG',
                       'TGTGAAGAAGCTGCAGAAGGTCAGGTCGTTTCTCCGGTAAACTTTAACTCTCCGGGACAGGTGGTTATTGCCGGT',
                       'CTGTATCGCGTATGGCAGCAGCAGGGCGGTAAAGCACCGGCAATGATGGCCGGTCACAGCCTGGGGGAATACTCC',
                       'ATGGCCGGTCACAGCCTGGGGGAATACTCCGCGCTGGTTTGCGCTGGTGTGATTGATTTCGCTGATGCGGTGCGT',
                       'TTGCTGAAGCTTCTGCGGCGCTGGGCTACGACCTGTGGGCGCTGACCCAGCAGGGGCCAGCTGAAGAACTGAATA',
                       'CGCTGGGCTACGACCTGTGGGCGCTGACCCAGCAGGGGCCAGCTGAAGAACTGAATAAAACCTGGCAAACTCAGC',
                       'GAAACGTTTGCTGAAGCTTCTGCGGCGCTGGGCTACGACCTGTGGGCGCTGACCCAGCAGGGGCCAGCTGAAGAA']
    reads_4224_230_75 = ['TTATCGATATCTGGGCTGCGGCGAACGATCGTGTATCCAAAGCGATGATGGATAACCTGCAAACTGAAACCGTGA',
                         'TCTGCGTCCGGCACTGAAAATCGTTGATGCTCAGGGTAACGACGTTCTGATCCCAGGTACCGATATGCCAGCGCA',
                         'CGTGCACCGACTCTGCACCGTCTGGGTATCCAGGCATTTGAACCGGTACTGATCGAAGGTAAAGCTATCCAGCTG',
                         'GTTCTGTTGTATCTTGTGACACCGACTTTGGTGTATGTGCGCACTGCTACGGTCGTGACCTGGCGCGTGGCCACA',
                         'CTGCTGGATAACGGTCGTCGCGGTCGTGCGATCACCGGTTCTAACAAGCGTCCTCTGAAATCTTTGGCCGACATG',
                         'ATCACCGCGAACTTCCGTGAAGGTCTGAACGTACTCCAGTACTTCATCTCCACCCACGGTGCTCGTAAAGGTCTG',
                         'TCGGTTGTGAACTCCAGCGGTAAACTGGTTATCACTTCCCGTAATACTGAACTGAAACTGATCGACGAATTCGGT',
                         'AACACGCTGCTGCACGAACAGTGGTGTGACCTGCTGGAAGAGAACTCTGTCGACGCGGTTAAAGTACGTTCTGTT',
                         'TAACAAGCGTCCTCTGAAATCTTTGGCCGACATGATCAAAGGTAAACAGGGTCGTTTCCGTCAGAACCTGCTCGG',
                         'GTACGTAACGAAAAACGTATGCTGCAGGAAGCGGTAGACGCCCTGCTGGATAACGGTCGTCGCGGTCGTGCGATC',
                         'AACGTACCGCAGGTGGTAAAGATCTGCGTCCGGCACTGAAAATCGTTGATGCTCAGGGTAACGACGTTCTGATCC',
                         'GCTACAAAGTACCTTACGGTGCGGTACTGGCGAAAGGCGATGGCGAACAGGTTGCTGGCGGCGAAACCGTTGCAA',
                         'AGAGCAGTATCTGGACGCGCTGGAAGAGTTCGGTGACGAATTCGACGCGAAGATGGGGGCGGAAGCAATCCAGGC',
                         'AGAGCCGGCAATCCTGGCTGAAATCAGCGGTATCGTTTCCTTCGGTAAAGAAACCAAAGGTAAACGTCGTCTGGT',
                         'AAACTGGTTATCACTTCCCGTAATACTGAACTGAAACTGATCGACGAATTCGGTCGTACTAAAGAAAGCTACAAA',
                         'AGAACGTTATCGTGGGTCGTCTGATCCCGGCAGGTACCGGTTACGCGTACCACCAGGATCGTATGCGTCGCCGTG',
                         'AACCTGCTCGGTAAGCGTGTTGACTACTCCGGTCGTTCTGTAATCACCGTAGGTCCATACCTGCGTCTGCATCAG',
                         'TCGGTCTGAAACCGACCGTTATTTTTGCGGACCAGATCATGTACACCGGCTTCGCCTATGCAGCGCGTTCTGGTG',
                         'AACGATCGTGTATCCAAAGCGATGATGGATAACCTGCAAACTGAAACCGTGATTAACCGTGACGGTCAGGAAGAG',
                         'CGACTTCGATGGTGACCAGATGGCTGTTCACGTACCGCTGACGCTGGAAGCCCAGCTGGAAGCGCGTGCGCTGAT',
                         'CGCTGCGCGATCGCGTACTGGGTCGTGTAACTGCTGAAGACGTTCTGAAGCCGGGTACTGCTGATATCCTCGTTC',
                         'ACTCCGAAACCAAGCGTAAAAAGCTGACCAAGCGTATCAAACTGCTGGAAGCGTTCGTTCAGTCTGGTAACAAAC',
                         'ACAGGACGTATACCGTCTGCAGGGCGTTAAGATTAACGATAAACACATCGAAGTTATCGTTCGTCAGATGCTGCG',
                         'GTGAAAGATTTATTAAAGTTTCTGAAAGCGCAGACTAAAACCGAAGAGTTTGATGCGATCAAAATTGCTCTGGCT',
                         'GTACCGAAAGGTCTGCCTTACTCCATCGTCAACCAGGCGCTGGGTAAAAAAGCAATCTCCAAAATGCTGAACACC',
                         'AAAAGATTACGAGTGCCTGTGCGGTAAGTACAAGCGCCTGAAACACCGTGGCGTCATCTGTGAGAAGTGCGGCGT',
                         'CGTGGTCTGATGGCGAAGCCGGATGGCTCCATCATCGAAACGCCAATCACCGCGAACTTCCGTGAAGGTCTGAAC',
                         'CGGTGCTCGTAAAGGTCTGGCGGATACCGCACTGAAAACTGCGAACTCCGGTTACCTGACTCGTCGTCTGGTTGA',
                         'TGGATCTGGAGCAAGAGTGCGAACAGCTGCGTGAAGAGCTGAACGAAACCAACTCCGAAACCAAGCGTAAAAAGC',
                         'CTCCAGCGGTAAACTGGTTATCACTTCCCGTAATACTGAACTGAAACTGATCGACGAATTCGGTCGTACTAAAGA',
                         'GTTGCGGGCAAACGCGACGAACTGCGCGGCCTGAAAGAGAACGTTATCGTGGGTCGTCTGATCCCGGCAGGTACC',
                         'GTGACTGCAGAAGACGCATCTGCCAGCCTGGCAGAACTGCTGAACGCAGGTCTGGGCGGTTCTGATAACGAGTAA',
                         'CGTCTGCTGGATCTGGCTGCGCCGGACATCATCGTACGTAACGAAAAACGTATGCTGCAGGAAGCGGTAGACGCC',
                         'AAAGCTGCGAAGAAAATGGTTGAGCGCGAAGAAGCTGTCGTTTGGGATATCCTGGACGAAGTTATCCGCGAACAC',
                         'AAAGATCTGCGTCCGGCACTGAAAATCGTTGATGCTCAGGGTAACGACGTTCTGATCCCAGGTACCGATATGCCA',
                         'TCGTGGGTCGTCTGATCCCGGCAGGTACCGGTTACGCGTACCACCAGGATCGTATGCGTCGCCGTGCTGCGGGTG',
                         'TGGGTCGTCTGATCCCGGCAGGTACCGGTTACGCGTACCACCAGGATCGTATGCGTCGCCGTGCTGCGGGTGAAG',
                         'AGAGCATGGATCTGGAGCAAGAGTGCGAACAGCTGCGTGAAGAGCTGAACGAAACCAACTCCGAAACCAAGCGTA',
                         'AAGGACATCACCGGTGGTCTGCCGCGCGTTGCGGACCTGTTCGAAGCACGTCGTCCGAAAGAGCCGGCAATCCTG',
                         'GACCCGCACACCATGCCGGTTATCACCGAAGTAAGCGGTTTTGTACGCTTTACTGACATGATCGACGGCCAGACC',
                         'CTGCTACGGTCGTGACCTGGCGCGTGGCCACATCATCAACAAGGGTGAAGCAATCGGTGTTATCGCGGCACAGTC',
                         'CCTGGTGGTTACCGAAGACGATTGTGGTACCCATGAAGGTATCATGATGACTCCGGTTATCGAGGGTGGTGACGT',
                         'GTTGCTGAAATTCAGGAGCAGTTCCAGTCTGGTCTGGTAACTGCGGGCGAACGCTACAACAAAGTTATCGATATC',
                         'TCCGGTTATCGAGGGTGGTGACGTTAAAGAGCCGCTGCGCGATCGCGTACTGGGTCGTGTAACTGCTGAAGACGT',
                         'GACATCATCGTACGTAACGAAAAACGTATGCTGCAGGAAGCGGTAGACGCCCTGCTGGATAACGGTCGTCGCGGT',
                         'AGCGCAGACTAAAACCGAAGAGTTTGATGCGATCAAAATTGCTCTGGCTTCGCCAGACATGATCCGTTCATGGTC',
                         'AGCCCAGCTGGAAGCGCGTGCGCTGATGATGTCTACCAACAACATCCTGTCCCCGGCGAACGGCGAACCAATCAT',
                         'CAAGCTGGAACTGCGTGGTCTTGCTACCACCATTAAAGCTGCGAAGAAAATGGTTGAGCGCGAAGAAGCTGTCGT',
                         'ACCTGCTGGAAGAGAACTCTGTCGACGCGGTTAAAGTACGTTCTGTTGTATCTTGTGACACCGACTTTGGTGTAT',
                         'GTTGACGTGGCGCAGGACCTGGTGGTTACCGAAGACGATTGTGGTACCCATGAAGGTATCATGATGACTCCGGTT',
                         'CAACCTGGAACGTCAGCAGATCCTGACTGAAGAGCAGTATCTGGACGCGCTGGAAGAGTTCGGTGACGAATTCGA',
                         'TGGATGATTGTACCGAAAGGTCTGCCTTACTCCATCGTCAACCAGGCGCTGGGTAAAAAAGCAATCTCCAAAATG',
                         'GGTAAACGTCGTCTGGTTATCACCCCGGTAGACGGTAGCGATCCGTACGAAGAGATGATTCCGAAATGGCGTCAG',
                         'AGGTTTCCTTCAACAGCATCTACATGATGGCCGACTCCGGTGCGCGTGGTTCTGCGGCACAGATTCGTCAGCTTG',
                         'TCTGGACGCGCTGGAAGAGTTCGGTGACGAATTCGACGCGAAGATGGGGGCGGAAGCAATCCAGGCTCTGCTGAA',
                         'TCTGCCGGTACTGCCGCCAGATCTGCGTCCGCTGGTTCCGCTGGATGGTGGTCGTTTCGCGACTTCTGACCTGAA',
                         'CACGCTGCTGCACGAACAGTGGTGTGACCTGCTGGAAGAGAACTCTGTCGACGCGGTTAAAGTACGTTCTGTTGT',
                         'AAAGTACGTTCTGTTGTATCTTGTGACACCGACTTTGGTGTATGTGCGCACTGCTACGGTCGTGACCTGGCGCGT',
                         'GTACTGGGTCTGTACTACATGACCCGTGACTGTGTTAACGCCAAAGGCGAAGGCATGGTGCTGACTGGCCCGAAA',
                         'AAAGCTATCCAGCTGCACCCGCTGGTTTGTGCGGCATATAACGCCGACTTCGATGGTGACCAGATGGCTGTTCAC',
                         'AAACCAGAGTGGATGATCCTGACCGTTCTGCCGGTACTGCCGCCAGATCTGCGTCCGCTGGTTCCGCTGGATGGT',
                         'GTTATCACCGAAGTAAGCGGTTTTGTACGCTTTACTGACATGATCGACGGCCAGACCATTACGCGTCAGACCGAC',
                         'CGCTGATGATGTCTACCAACAACATCCTGTCCCCGGCGAACGGCGAACCAATCATCGTTCCGTCTCAGGACGTTG',
                         'CGAAAACCAGCCTGAAAGACACGACTGTTGGCCGTGCCATTCTGTGGATGATTGTACCGAAAGGTCTGCCTTACT',
                         'TCCGGTTATCGAGGGTGGTGACGTTAAAGAGCCGCTGCGCGATCGCGTACTGGGTCGTGTAACTGCTGAAGACGT',
                         'GAAAGGCGATGGCGAACAGGTTGCTGGCGGCGAAACCGTTGCAAACTGGGACCCGCACACCATGCCGGTTATCAC',
                         'ATGGCGAAGCCGGATGGCTCCATCATCGAAACGCCAATCACCGCGAACTTCCGTGAAGGTCTGAACGTACTCCAG',
                         'CTGGGTCGTGTAACTGCTGAAGACGTTCTGAAGCCGGGTACTGCTGATATCCTCGTTCCGCGCAACACGCTGCTG',
                         'GGAAGCGCGTGCGCTGATGATGTCTACCAACAACATCCTGTCCCCGGCGAACGGCGAACCAATCATCGTTCCGTC',
                         'AACCTGCTCGGTAAGCGTGTTGACTACTCCGGTCGTTCTGTAATCACCGTAGGTCCATACCTGCGTCTGCATCAG',
                         'CATCGGTGAACCGGGTACACAGCTGACCATGCGTACGTTCCACATCGGTGGTGCGGCATCTCGTGCGGCTGCTGA',
                         'CGTATGCGTCGCCGTGCTGCGGGTGAAGCTCCGGCTGCACCGCAGGTGACTGCAGAAGACGCATCTGCCAGCCTG',
                         'AGCCGGCAATCCTGGCTGAAATCAGCGGTATCGTTTCCTTCGGTAAAGAAACCAAAGGTAAACGTCGTCTGGTTA',
                         'TATCCAGCTGCACCCGCTGGTTTGTGCGGCATATAACGCCGACTTCGATGGTGACCAGATGGCTGTTCACGTACC',
                         'CGGTCGTGACCTGGCGCGTGGCCACATCATCAACAAGGGTGAAGCAATCGGTGTTATCGCGGCACAGTCCATCGG',
                         'ACGTGTAGAACGTGGTGACGTAATTTCCGACGGTCCGGAAGCGCCGCACGACATTCTGCGTCTGCGTGGTGTTCA',
                         'CGGTACTGGCGAAAGGCGATGGCGAACAGGTTGCTGGCGGCGAAACCGTTGCAAACTGGGACCCGCACACCATGC',
                         'ACATCATCGTACGTAACGAAAAACGTATGCTGCAGGAAGCGGTAGACGCCCTGCTGGATAACGGTCGTCGCGGTC',
                         'GCGAACAGGTTGAATACTCTCGCGTCAAGATCGCAAACCGCGAACTGGAAGCGAACGGCAAAGTGGGTGCAACTT',
                         'CGGTCGTGACCTGGCGCGTGGCCACATCATCAACAAGGGTGAAGCAATCGGTGTTATCGCGGCACAGTCCATCGG',
                         'GACTGCGCACATCTGGTTCCTGAAATCGCTGCCGTCCCGTATCGGTCTGCTGCTCGATATGCCGCTGCGCGATAT',
                         'AACACCGTGGCGTCATCTGTGAGAAGTGCGGCGTTGAAGTGACCCAGACTAAAGTACGCCGTGAGCGTATGGGCC',
                         'CAGCTGACCATGCGTACGTTCCACATCGGTGGTGCGGCATCTCGTGCGGCTGCTGAATCCAGCATCCAAGTGAAA',
                         'GCACGAACAGTGGTGTGACCTGCTGGAAGAGAACTCTGTCGACGCGGTTAAAGTACGTTCTGTTGTATCTTGTGA',
                         'ACCATCAACTACCGTACGTTCAAACCAGAACGTGACGGCCTTTTCTGCGCCCGTATCTTTGGGCCGGTAAAAGAT',
                         'TATCGATATCTGGGCTGCGGCGAACGATCGTGTATCCAAAGCGATGATGGATAACCTGCAAACTGAAACCGTGAT',
                         'CTGGTTCCTGAAATCGCTGCCGTCCCGTATCGGTCTGCTGCTCGATATGCCGCTGCGCGATATCGAACGCGTACT',
                         'TTTCCGACGGTCCGGAAGCGCCGCACGACATTCTGCGTCTGCGTGGTGTTCATGCTGTTACTCGTTACATCGTTA',
                         'ACAACAAAGTTATCGATATCTGGGCTGCGGCGAACGATCGTGTATCCAAAGCGATGATGGATAACCTGCAAACTG',
                         'GTAAAGCTATCCAGCTGCACCCGCTGGTTTGTGCGGCATATAACGCCGACTTCGATGGTGACCAGATGGCTGTTC',
                         'GCGGCCTGAAAGAGAACGTTATCGTGGGTCGTCTGATCCCGGCAGGTACCGGTTACGCGTACCACCAGGATCGTA',
                         'ACGGTCCGGAAGCGCCGCACGACATTCTGCGTCTGCGTGGTGTTCATGCTGTTACTCGTTACATCGTTAACGAAG',
                         'AAGCAATCCAGGCTCTGCTGAAGAGCATGGATCTGGAGCAAGAGTGCGAACAGCTGCGTGAAGAGCTGAACGAAA',
                         'ACGAAAAACGTATGCTGCAGGAAGCGGTAGACGCCCTGCTGGATAACGGTCGTCGCGGTCGTGCGATCACCGGTT',
                         'GTGGTGCGGCATCTCGTGCGGCTGCTGAATCCAGCATCCAAGTGAAAAACAAAGGTAGCATCAAGCTCAGCAACG',
                         'GCCTGAAACACCGTGGCGTCATCTGTGAGAAGTGCGGCGTTGAAGTGACCCAGACTAAAGTACGCCGTGAGCGTA',
                         'CTGAAACCGACCGTTATTTTTGCGGACCAGATCATGTACACCGGCTTCGCCTATGCAGCGCGTTCTGGTGCATCT',
                         'ACTCCGGTTATCGAGGGTGGTGACGTTAAAGAGCCGCTGCGCGATCGCGTACTGGGTCGTGTAACTGCTGAAGAC',
                         'GACACCCTGGCGCGTATTCCGCAGGAATCCGGCGGTACCAAGGACATCACCGGTGGTCTGCCGCGCGTTGCGGAC',
                         'GGCTCTGCTGAAGAGCATGGATCTGGAGCAAGAGTGCGAACAGCTGCGTGAAGAGCTGAACGAAACCAACTCCGA',
                         'CAGTCCATCGGTGAACCGGGTACACAGCTGACCATGCGTACGTTCCACATCGGTGGTGCGGCATCTCGTGCGGCT',
                         'AGGAAGCGGTAGACGCCCTGCTGGATAACGGTCGTCGCGGTCGTGCGATCACCGGTTCTAACAAGCGTCCTCTGA',
                         'CACCCGCTGGTTTGTGCGGCATATAACGCCGACTTCGATGGTGACCAGATGGCTGTTCACGTACCGCTGACGCTG',
                         'ACATGATCAAAGGTAAACAGGGTCGTTTCCGTCAGAACCTGCTCGGTAAGCGTGTTGACTACTCCGGTCGTTCTG',
                         'AGATCAGCTCTGGTGACACCCTGGCGCGTATTCCGCAGGAATCCGGCGGTACCAAGGACATCACCGGTGGTCTGC',
                         'ACCTGCAAACTGAAACCGTGATTAACCGTGACGGTCAGGAAGAGAAGCAGGTTTCCTTCAACAGCATCTACATGA',
                         'AAGAGTTCGGTGACGAATTCGACGCGAAGATGGGGGCGGAAGCAATCCAGGCTCTGCTGAAGAGCATGGATCTGG',
                         'TCAACAGCATCTACATGATGGCCGACTCCGGTGCGCGTGGTTCTGCGGCACAGATTCGTCAGCTTGCTGGTATGC',
                         'TGCTGCTCGATATGCCGCTGCGCGATATCGAACGCGTACTGTACTTTGAATCCTATGTGGTTATCGAAGGCGGTA',
                         'CTGCGTGGTGTTCATGCTGTTACTCGTTACATCGTTAACGAAGTACAGGACGTATACCGTCTGCAGGGCGTTAAG',
                         'CGATCAAAATTGCTCTGGCTTCGCCAGACATGATCCGTTCATGGTCTTTCGGTGAAGTTAAAAAGCCGGAAACCA',
                         'GTCTGCCTTACTCCATCGTCAACCAGGCGCTGGGTAAAAAAGCAATCTCCAAAATGCTGAACACCTGCTACCGCA',
                         'CTCCATCGTCAACCAGGCGCTGGGTAAAAAAGCAATCTCCAAAATGCTGAACACCTGCTACCGCATTCTCGGTCT',
                         'TGGGACCCGCACACCATGCCGGTTATCACCGAAGTAAGCGGTTTTGTACGCTTTACTGACATGATCGACGGCCAG',
                         'CCGTCTGGGTATCCAGGCATTTGAACCGGTACTGATCGAAGGTAAAGCTATCCAGCTGCACCCGCTGGTTTGTGC',
                         'TCTGGTCTGGTAACTGCGGGCGAACGCTACAACAAAGTTATCGATATCTGGGCTGCGGCGAACGATCGTGTATCC',
                         'CAGCTCTGGTGACACCCTGGCGCGTATTCCGCAGGAATCCGGCGGTACCAAGGACATCACCGGTGGTCTGCCGCG',
                         'CTGCCTTACTCCATCGTCAACCAGGCGCTGGGTAAAAAAGCAATCTCCAAAATGCTGAACACCTGCTACCGCATT',
                         'GGCGAACCAATCATCGTTCCGTCTCAGGACGTTGTACTGGGTCTGTACTACATGACCCGTGACTGTGTTAACGCC',
                         'CACTCGCGTGCTGACCGAAGCAGCCGTTGCGGGCAAACGCGACGAACTGCGCGGCCTGAAAGAGAACGTTATCGT',
                         'GAACTTCCGTGAAGGTCTGAACGTACTCCAGTACTTCATCTCCACCCACGGTGCTCGTAAAGGTCTGGCGGATAC',
                         'TACCGTACGTTCAAACCAGAACGTGACGGCCTTTTCTGCGCCCGTATCTTTGGGCCGGTAAAAGATTACGAGTGC',
                         'TAAAAGATTACGAGTGCCTGTGCGGTAAGTACAAGCGCCTGAAACACCGTGGCGTCATCTGTGAGAAGTGCGGCG',
                         'ATGCTCAGGGTAACGACGTTCTGATCCCAGGTACCGATATGCCAGCGCAGTACTTCCTGCCGGGTAAAGCGATTG',
                         'AATCCGGCGGTACCAAGGACATCACCGGTGGTCTGCCGCGCGTTGCGGACCTGTTCGAAGCACGTCGTCCGAAAG',
                         'GAGATGATTCCGAAATGGCGTCAGCTCAACGTGTTCGAAGGTGAACGTGTAGAACGTGGTGACGTAATTTCCGAC',
                         'GGCGAAACCGTTGCAAACTGGGACCCGCACACCATGCCGGTTATCACCGAAGTAAGCGGTTTTGTACGCTTTACT',
                         'GCACGACATTCTGCGTCTGCGTGGTGTTCATGCTGTTACTCGTTACATCGTTAACGAAGTACAGGACGTATACCG',
                         'GGTCTTTCGGTGAAGTTAAAAAGCCGGAAACCATCAACTACCGTACGTTCAAACCAGAACGTGACGGCCTTTTCT',
                         'TTCAACAGCATCTACATGATGGCCGACTCCGGTGCGCGTGGTTCTGCGGCACAGATTCGTCAGCTTGCTGGTATG',
                         'AACTGGAAGCGAACGGCAAAGTGGGTGCAACTTACTCCCGCGATCTGCTGGGTATCACCAAAGCGTCTCTGGCAA',
                         'CCTATGCAGCGCGTTCTGGTGCATCTGTTGGTATCGATGACATGGTCATCCCGGAGAAGAAACACGAAATCATCT',
                         'AAAGGTAAACGTCGTCTGGTTATCACCCCGGTAGACGGTAGCGATCCGTACGAAGAGATGATTCCGAAATGGCGT',
                         'CAGCTCAACGTGTTCGAAGGTGAACGTGTAGAACGTGGTGACGTAATTTCCGACGGTCCGGAAGCGCCGCACGAC',
                         'TGTCCCCGGCGAACGGCGAACCAATCATCGTTCCGTCTCAGGACGTTGTACTGGGTCTGTACTACATGACCCGTG',
                         'CCTTACGGTGCGGTACTGGCGAAAGGCGATGGCGAACAGGTTGCTGGCGGCGAAACCGTTGCAAACTGGGACCCG',
                         'CCTGCGTCTGCATCAGTGCGGTCTGCCGAAGAAAATGGCACTGGAGCTGTTCAAACCGTTCATCTACGGCAAGCT',
                         'AAGCGGTTTTGTACGCTTTACTGACATGATCGACGGCCAGACCATTACGCGTCAGACCGACGAACTGACCGGTCT',
                         'TCTGTATCGCTCTGGTCTGGCTTCTCTGCATGCGCGCGTTAAAGTGCGTATCACCGAGTATGAAAAAGATGCTAA',
                         'AGTCCTTCATCTCCGCGGCATCGTTCCAGGAGACCACTCGCGTGCTGACCGAAGCAGCCGTTGCGGGCAAACGCG',
                         'GATCAAAATTGCTCTGGCTTCGCCAGACATGATCCGTTCATGGTCTTTCGGTGAAGTTAAAAAGCCGGAAACCAT',
                         'CTGTCGTTTGGGATATCCTGGACGAAGTTATCCGCGAACACCCGGTACTGCTGAACCGTGCACCGACTCTGCACC',
                         'TATCGTCGCGTCATTAACCGTAACAACCGTCTGAAACGTCTGCTGGATCTGGCTGCGCCGGACATCATCGTACGT',
                         'TGCTAACGGTGAATTAGTAGCGAAAACCAGCCTGAAAGACACGACTGTTGGCCGTGCCATTCTGTGGATGATTGT',
                         'TCGTTCCAGGAGACCACTCGCGTGCTGACCGAAGCAGCCGTTGCGGGCAAACGCGACGAACTGCGCGGCCTGAAA',
                         'TAGTAGCGAAAACCAGCCTGAAAGACACGACTGTTGGCCGTGCCATTCTGTGGATGATTGTACCGAAAGGTCTGC',
                         'GTACTGCCGCCAGATCTGCGTCCGCTGGTTCCGCTGGATGGTGGTCGTTTCGCGACTTCTGACCTGAACGATCTG',
                         'ACTGTTGGCCGTGCCATTCTGTGGATGATTGTACCGAAAGGTCTGCCTTACTCCATCGTCAACCAGGCGCTGGGT',
                         'AGCCGCTGCGCGATCGCGTACTGGGTCGTGTAACTGCTGAAGACGTTCTGAAGCCGGGTACTGCTGATATCCTCG',
                         'GGCCTGAAAGAGAACGTTATCGTGGGTCGTCTGATCCCGGCAGGTACCGGTTACGCGTACCACCAGGATCGTATG',
                         'ACGCCGTGAGCGTATGGGCCACATCGAACTGGCTTCCCCGACTGCGCACATCTGGTTCCTGAAATCGCTGCCGTC',
                         'TACGCGTACCACCAGGATCGTATGCGTCGCCGTGCTGCGGGTGAAGCTCCGGCTGCACCGCAGGTGACTGCAGAA',
                         'CCGCTGGTTTGTGCGGCATATAACGCCGACTTCGATGGTGACCAGATGGCTGTTCACGTACCGCTGACGCTGGAA',
                         'GTGTTGACTACTCCGGTCGTTCTGTAATCACCGTAGGTCCATACCTGCGTCTGCATCAGTGCGGTCTGCCGAAGA',
                         'CTGACCGAAGCAGCCGTTGCGGGCAAACGCGACGAACTGCGCGGCCTGAAAGAGAACGTTATCGTGGGTCGTCTG',
                         'GGGTGAAGCTCCGGCTGCACCGCAGGTGACTGCAGAAGACGCATCTGCCAGCCTGGCAGAACTGCTGAACGCAGG',
                         'TTCTGTGGATGATTGTACCGAAAGGTCTGCCTTACTCCATCGTCAACCAGGCGCTGGGTAAAAAAGCAATCTCCA',
                         'TACTGTACTTTGAATCCTATGTGGTTATCGAAGGCGGTATGACCAACCTGGAACGTCAGCAGATCCTGACTGAAG',
                         'GTTGGCCGTGCCATTCTGTGGATGATTGTACCGAAAGGTCTGCCTTACTCCATCGTCAACCAGGCGCTGGGTAAA',
                         'GGGTAGCTCCGACTTCCTGGAAGGCGAACAGGTTGAATACTCTCGCGTCAAGATCGCAAACCGCGAACTGGAAGC',
                         'CTGGTTATCACCCCGGTAGACGGTAGCGATCCGTACGAAGAGATGATTCCGAAATGGCGTCAGCTCAACGTGTTC',
                         'GACCATTACGCGTCAGACCGACGAACTGACCGGTCTGTCTTCGCTGGTGGTTCTGGATTCCGCAGAACGTACCGC',
                         'GTACTTTGAATCCTATGTGGTTATCGAAGGCGGTATGACCAACCTGGAACGTCAGCAGATCCTGACTGAAGAGCA',
                         'GCCGCACGACATTCTGCGTCTGCGTGGTGTTCATGCTGTTACTCGTTACATCGTTAACGAAGTACAGGACGTATA',
                         'CTGAAGAGCAGTATCTGGACGCGCTGGAAGAGTTCGGTGACGAATTCGACGCGAAGATGGGGGCGGAAGCAATCC',
                         'TGGATCTGGCTGCGCCGGACATCATCGTACGTAACGAAAAACGTATGCTGCAGGAAGCGGTAGACGCCCTGCTGG',
                         'ACCATGCGTACGTTCCACATCGGTGGTGCGGCATCTCGTGCGGCTGCTGAATCCAGCATCCAAGTGAAAAACAAA',
                         'ACAAAGTACCTTACGGTGCGGTACTGGCGAAAGGCGATGGCGAACAGGTTGCTGGCGGCGAAACCGTTGCAAACT',
                         'AGTGAAAAACAAAGGTAGCATCAAGCTCAGCAACGTGAAGTCGGTTGTGAACTCCAGCGGTAAACTGGTTATCAC',
                         'TGAACGAAACCAACTCCGAAACCAAGCGTAAAAAGCTGACCAAGCGTATCAAACTGCTGGAAGCGTTCGTTCAGT',
                         'AGAAGTGCGGCGTTGAAGTGACCCAGACTAAAGTACGCCGTGAGCGTATGGGCCACATCGAACTGGCTTCCCCGA',
                         'AACGTCGTCTGGTTATCACCCCGGTAGACGGTAGCGATCCGTACGAAGAGATGATTCCGAAATGGCGTCAGCTCA',
                         'GCTGACGCTGGAAGCCCAGCTGGAAGCGCGTGCGCTGATGATGTCTACCAACAACATCCTGTCCCCGGCGAACGG',
                         'ATGGTGCTGACTGGCCCGAAAGAAGCAGAACGTCTGTATCGCTCTGGTCTGGCTTCTCTGCATGCGCGCGTTAAA',
                         'TATCGATATCTGGGCTGCGGCGAACGATCGTGTATCCAAAGCGATGATGGATAACCTGCAAACTGAAACCGTGAT',
                         'CGTGCACCGACTCTGCACCGTCTGGGTATCCAGGCATTTGAACCGGTACTGATCGAAGGTAAAGCTATCCAGCTG',
                         'ACTTTGAATCCTATGTGGTTATCGAAGGCGGTATGACCAACCTGGAACGTCAGCAGATCCTGACTGAAGAGCAGT',
                         'GTGACACCCTGGCGCGTATTCCGCAGGAATCCGGCGGTACCAAGGACATCACCGGTGGTCTGCCGCGCGTTGCGG',
                         'GACTAAAGTACGCCGTGAGCGTATGGGCCACATCGAACTGGCTTCCCCGACTGCGCACATCTGGTTCCTGAAATC',
                         'GTGGTGTGACCTGCTGGAAGAGAACTCTGTCGACGCGGTTAAAGTACGTTCTGTTGTATCTTGTGACACCGACTT',
                         'TGCATGCGCGCGTTAAAGTGCGTATCACCGAGTATGAAAAAGATGCTAACGGTGAATTAGTAGCGAAAACCAGCC',
                         'GGTTCTGGATTCCGCAGAACGTACCGCAGGTGGTAAAGATCTGCGTCCGGCACTGAAAATCGTTGATGCTCAGGG',
                         'ATATCCTCGTTCCGCGCAACACGCTGCTGCACGAACAGTGGTGTGACCTGCTGGAAGAGAACTCTGTCGACGCGG',
                         'TAAACGTCGTCTGGTTATCACCCCGGTAGACGGTAGCGATCCGTACGAAGAGATGATTCCGAAATGGCGTCAGCT',
                         'AAACCAGAGTGGATGATCCTGACCGTTCTGCCGGTACTGCCGCCAGATCTGCGTCCGCTGGTTCCGCTGGATGGT',
                         'GAACTGGCTTCCCCGACTGCGCACATCTGGTTCCTGAAATCGCTGCCGTCCCGTATCGGTCTGCTGCTCGATATG',
                         'AAGCGATTGTTCAGCTGGAAGATGGCGTACAGATCAGCTCTGGTGACACCCTGGCGCGTATTCCGCAGGAATCCG',
                         'AATCCGGCGGTACCAAGGACATCACCGGTGGTCTGCCGCGCGTTGCGGACCTGTTCGAAGCACGTCGTCCGAAAG',
                         'AACTGCGTGGTCTTGCTACCACCATTAAAGCTGCGAAGAAAATGGTTGAGCGCGAAGAAGCTGTCGTTTGGGATA',
                         'TCGCGTGCTGACCGAAGCAGCCGTTGCGGGCAAACGCGACGAACTGCGCGGCCTGAAAGAGAACGTTATCGTGGG',
                         'ACAAGCGTCCTCTGAAATCTTTGGCCGACATGATCAAAGGTAAACAGGGTCGTTTCCGTCAGAACCTGCTCGGTA',
                         'CATGGTCATCCCGGAGAAGAAACACGAAATCATCTCCGAGGCAGAAGCAGAAGTTGCTGAAATTCAGGAGCAGTT',
                         'CATCGTTCCAGGAGACCACTCGCGTGCTGACCGAAGCAGCCGTTGCGGGCAAACGCGACGAACTGCGCGGCCTGA',
                         'AACCGGGTACACAGCTGACCATGCGTACGTTCCACATCGGTGGTGCGGCATCTCGTGCGGCTGCTGAATCCAGCA',
                         'TGGCGCGTGGCCACATCATCAACAAGGGTGAAGCAATCGGTGTTATCGCGGCACAGTCCATCGGTGAACCGGGTA',
                         'TCCTGGACGAAGTTATCCGCGAACACCCGGTACTGCTGAACCGTGCACCGACTCTGCACCGTCTGGGTATCCAGG',
                         'TGGAAGCGCGTGCGCTGATGATGTCTACCAACAACATCCTGTCCCCGGCGAACGGCGAACCAATCATCGTTCCGT',
                         'ACAAAGTACCTTACGGTGCGGTACTGGCGAAAGGCGATGGCGAACAGGTTGCTGGCGGCGAAACCGTTGCAAACT',
                         'ATATCTGGGCTGCGGCGAACGATCGTGTATCCAAAGCGATGATGGATAACCTGCAAACTGAAACCGTGATTAACC',
                         'TCGCTCTGGTCTGGCTTCTCTGCATGCGCGCGTTAAAGTGCGTATCACCGAGTATGAAAAAGATGCTAACGGTGA',
                         'GTAAAGATCTGCGTCCGGCACTGAAAATCGTTGATGCTCAGGGTAACGACGTTCTGATCCCAGGTACCGATATGC',
                         'AAAGCGTCTCTGGCAACCGAGTCCTTCATCTCCGCGGCATCGTTCCAGGAGACCACTCGCGTGCTGACCGAAGCA',
                         'GTGACCTGCTGGAAGAGAACTCTGTCGACGCGGTTAAAGTACGTTCTGTTGTATCTTGTGACACCGACTTTGGTG',
                         'TGATTCCGAAATGGCGTCAGCTCAACGTGTTCGAAGGTGAACGTGTAGAACGTGGTGACGTAATTTCCGACGGTC',
                         'GCCTGAAAGACACGACTGTTGGCCGTGCCATTCTGTGGATGATTGTACCGAAAGGTCTGCCTTACTCCATCGTCA',
                         'CAGCTGGAAGATGGCGTACAGATCAGCTCTGGTGACACCCTGGCGCGTATTCCGCAGGAATCCGGCGGTACCAAG',
                         'TCGTTCAGTCTGGTAACAAACCAGAGTGGATGATCCTGACCGTTCTGCCGGTACTGCCGCCAGATCTGCGTCCGC',
                         'CTGCAGGAAGCGGTAGACGCCCTGCTGGATAACGGTCGTCGCGGTCGTGCGATCACCGGTTCTAACAAGCGTCCT',
                         'CTCCGGTTATCGAGGGTGGTGACGTTAAAGAGCCGCTGCGCGATCGCGTACTGGGTCGTGTAACTGCTGAAGACG',
                         'CCCGACTGCGCACATCTGGTTCCTGAAATCGCTGCCGTCCCGTATCGGTCTGCTGCTCGATATGCCGCTGCGCGA',
                         'ATCCTGACTGAAGAGCAGTATCTGGACGCGCTGGAAGAGTTCGGTGACGAATTCGACGCGAAGATGGGGGCGGAA',
                         'CCAGAGTGGATGATCCTGACCGTTCTGCCGGTACTGCCGCCAGATCTGCGTCCGCTGGTTCCGCTGGATGGTGGT',
                         'TATGAAAAAGATGCTAACGGTGAATTAGTAGCGAAAACCAGCCTGAAAGACACGACTGTTGGCCGTGCCATTCTG',
                         'GCCCTGCTGGATAACGGTCGTCGCGGTCGTGCGATCACCGGTTCTAACAAGCGTCCTCTGAAATCTTTGGCCGAC',
                         'CGCGTTAAAGTGCGTATCACCGAGTATGAAAAAGATGCTAACGGTGAATTAGTAGCGAAAACCAGCCTGAAAGAC',
                         'TCTGTTGGTATCGATGACATGGTCATCCCGGAGAAGAAACACGAAATCATCTCCGAGGCAGAAGCAGAAGTTGCT',
                         'ACCCAGACTAAAGTACGCCGTGAGCGTATGGGCCACATCGAACTGGCTTCCCCGACTGCGCACATCTGGTTCCTG',
                         'TGACACCGACTTTGGTGTATGTGCGCACTGCTACGGTCGTGACCTGGCGCGTGGCCACATCATCAACAAGGGTGA',
                         'GACATGATCGACGGCCAGACCATTACGCGTCAGACCGACGAACTGACCGGTCTGTCTTCGCTGGTGGTTCTGGAT',
                         'CTGGCGCGTGGCCACATCATCAACAAGGGTGAAGCAATCGGTGTTATCGCGGCACAGTCCATCGGTGAACCGGGT',
                         'GTGCCTGTGCGGTAAGTACAAGCGCCTGAAACACCGTGGCGTCATCTGTGAGAAGTGCGGCGTTGAAGTGACCCA',
                         'TACCGAAGACGATTGTGGTACCCATGAAGGTATCATGATGACTCCGGTTATCGAGGGTGGTGACGTTAAAGAGCC',
                         'AAGCGTGTTGACTACTCCGGTCGTTCTGTAATCACCGTAGGTCCATACCTGCGTCTGCATCAGTGCGGTCTGCCG',
                         'TCGTCTGGTTATCACCCCGGTAGACGGTAGCGATCCGTACGAAGAGATGATTCCGAAATGGCGTCAGCTCAACGT',
                         'ACACATCGAAGTTATCGTTCGTCAGATGCTGCGTAAAGCTACCATCGTTAACGCGGGTAGCTCCGACTTCCTGGA',
                         'GGTAAAGATCTGCGTCCGGCACTGAAAATCGTTGATGCTCAGGGTAACGACGTTCTGATCCCAGGTACCGATATG',
                         'GTTATCGCGGCACAGTCCATCGGTGAACCGGGTACACAGCTGACCATGCGTACGTTCCACATCGGTGGTGCGGCA',
                         'GGCGGAAGCAATCCAGGCTCTGCTGAAGAGCATGGATCTGGAGCAAGAGTGCGAACAGCTGCGTGAAGAGCTGAA',
                         'GCTGGGTAAAAAAGCAATCTCCAAAATGCTGAACACCTGCTACCGCATTCTCGGTCTGAAACCGACCGTTATTTT',
                         'GAGTGCCTGTGCGGTAAGTACAAGCGCCTGAAACACCGTGGCGTCATCTGTGAGAAGTGCGGCGTTGAAGTGACC']

    dataset[1] = (genome25, reads_25_10_8)
    dataset[2] = (genome25, reads_25_10_10)
    dataset[3] = (genome25, reads_25_10_15)
    dataset[4] = (genome50, reads_50_10_8)
    dataset[5] = (genome50, reads_50_10_10)
    dataset[6] = (genome50, reads_50_10_15)
    dataset[7] = (genome25, reads_25_20_8)
    dataset[8] = (genome25, reads_25_20_10)
    dataset[9] = (genome25, reads_25_20_15)
    dataset[10] = (genome50, reads_50_20_8)
    dataset[11] = (genome50, reads_50_20_10)
    dataset[12] = (genome50, reads_50_20_15)
    dataset[13] = (genome25, reads_25_30_8)
    dataset[14] = (genome25, reads_25_30_10)
    dataset[15] = (genome25, reads_25_30_15)
    dataset[16] = (genome50, reads_50_30_8)
    dataset[17] = (genome50, reads_50_30_10)
    dataset[18] = (genome50, reads_50_30_15)
    dataset[19] = (genome381, reads_381_20_75)
    dataset[20] = (genome567, reads_567_30_75)
    dataset[21] = (genome726, reads_726_40_75)
    dataset[22] = (genome930, reads_930_50_75)
    dataset[23] = (genome4224, reads_4224_230_75)
    dataset[23] = (genome4224, reads_4224_230_75)
    dataset[24] = (genome25, readProva)

    genome, reads = dataset[int(sys.argv[2])]
    qlearning(reads, int(sys.argv[1]), genome)
