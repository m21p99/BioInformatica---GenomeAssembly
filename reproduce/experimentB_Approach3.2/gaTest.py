import copy
import heapq
import networkx as nx
from overlap import OverlapResolver
from node import Node
from itertools import permutations
import random
import sys
import numpy as np


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


    def run_ga(self, env, pop, gen, iterazioni):
        # print("----- Siamo nel metodo run_GA -----")

        # print("Stampa delle varie letture: ", reads)
        pop_size = len(pop)
        # print("pop", pop_size)
        population = pop
        # print("Popolazione iniziale: ")

        for i, popolazione in enumerate(population):
            # print(f"Popolazione {i + 1}:")
            for individuo in popolazione:
                sequenza, lista_numeri = individuo
                # print(f"  Sequenza: {sequenza}, Numeri: {lista_numeri}")
        # print("Le letture sono: ", reads)
        #print(len(population))
        pop_fitness = self._evaluatePopulation(population, gen)
        
        copyFitness = copy.deepcopy(np.array(pop_fitness))
        print("Fitness Ordinati\n",sorted(copyFitness, reverse=True))
        
        
        #print("Fitness Non ordinato\n",pop_fitness)
        #print(pop_fitness)
        # print("--------------------------------")

        # print("Calcolo della fitness ", pop_fitness, " per la Popolazione corrente: ", population)

        c = np.argmax(pop_fitness)
        # print(c)
        best_ind = copy.deepcopy(population[c])
        # print(best_ind)
        # print("Migliorr individuo della popolazione corrente: \n", best_ind)
        # print("Seleziona l'individuo con la fitness massima dalla popolazione corrente: ", best_ind)
        # print("---------------------")
        # print("Le letture sono: ", reads)
        best_fit = float(self._fitness(best_ind))
        best_ind = (best_fit,best_ind)
        print("Miglior individuo della popolazione corrente: ", best_ind)
        # print("Valore fitness dell'individuo migliore: ", best_fit)

        for generation in range(iterazioni):
            # print("---------")
            print('Iterazione n: ', generation)
            # Tournament selection

            mappa_fitness = []

            # Itera attraverso le popolazioni e la fitness
            for i in range(len(population)):
                # Tutti gli individui della popolazione
                individui = population[i]

                # Il valore di fitness corrispondente
                fitness_corrispondente = pop_fitness[i]

                # Crea una tupla con il valore di fitness e tutti gli individui
                tupla = (fitness_corrispondente, individui)

                # Aggiungi la tupla alla lista
                mappa_fitness.append(tupla)

            # Stampa la lista per vedere il risultato
            #for x in mappa_fitness:
                #print("\n", x)

            sorted_data = sorted(mappa_fitness, key=lambda x: x[0], reverse=True)
            #print("-------------")

            #for x in sorted_data:
                #print("\n", x)

            population = sorted_data

            """
            if len(population) < 2:
                print("La popolazione è troppo piccola")
                return best_ind
            else:
                lenPop = len(population) // 2
                # Calcola il numero di individui da selezionare
                #lenPop = int(len(population) * 0.6)
                population = population[:lenPop]
                print("Popolazione ridotta a: ", len(population))
            """

            if len(population) < 50:
                lenPop = len(population)
                print("La popolazione è troppo piccola")
                #return best_ind
            else:
                lenPop = len(population) // 2
                # Calcola il numero di individui da selezionare
                #lenPop = int(len(population) * 0.6)
                population = population[:lenPop]
                print("Popolazione ridotta a: ", len(population))

            #print("-----Modifica aver estratto meta popolazione per il crossover")
            #for x in population:
                #print("\n", x)

            # Crossover
            for i in range(0, len(population) - 1, 2):  # Assicura che i non superi la lunghezza della lista meno uno
                if np.random.rand() < self.crossover_prob:
                    # print("Popo", population[i])
                    # print("Popo1", population[i + 1])
                    # Esegui il crossover solo se i e i + 1 sono entrambi validi
                    if i < len(population) - 1:
                        pop1, pop2 = self._crossover(population[i], population[i + 1])
                        population.append(pop1)
                        population.append(pop2)

                # print("Popolazioni ottenute tramite il crossover: \n", population[i])
                # print(population[i + 1])
            # Mutation
            # print("-----------------")
            """
            print("-----Modifica dopo crossover")
            for x in population:
                print("\n", x)
            """
            # print("Siamo nel metodo mutation:")

            best_ind = population[0]

            """
            Qui stiamo eseguendo la mutazione anche degli individui aggiunti tramite il crossover

            for i in range(len(population)):
                if np.random.rand() < self.mutation_prob:
                    print("Individuo da mutare: ", population[i]) 
                    temp = self._mutation(population[i])
                    print("Individuo mutato: ", temp)
                    population[i] = temp
            """
            # Qui eseguiamo la mutazione solo degli individui migliori senza quegli aggiunti tramite il crossover
            for i in range(len(population)):
                if np.random.rand() < self.mutation_prob:
                    temp = self._mutation(population[i])
                    population[i] = temp


            if best_ind not in population:
                population.append(copy.deepcopy(best_ind))

            # print("Popolazione dopo l'aggiunta di", best_ind)

            population = [individuo for fitness, individuo in population]
            """
            print("----------")
            for x in population:
                print("\n", x)
            """
            pop_fitness = self._evaluatePopulation(population, gen)
            copyFitness = copy.deepcopy(pop_fitness)
            print("DOPO ordinati",sorted(copyFitness, reverse=True))
            print("Miglior individuo della popolazione corrente: ", best_ind)
            #print("Dopo non ordinati",pop_fitness)
            #value = pop_fitness.max()
            #best_ind = population[pop_fitness.argmax()].copy()
            #print("Miglior individuo:", best_ind)
            # print("------------------")
            # print("L'obiettivo è quello di mantenere nelle generazioni successive il miglior individuo: ")
            #print("Stampa il valore massimo di fitness della generazione attuale: ", value)
            # print("Mentre il valore massimo di fitness fino ad ora e di ", best_fit)

            """
            Questo blocco di codice implementa una strategia di elitismo nell'algoritmo genetico. L'obiettivo è
            mantenere l'individuo migliore (con la fitness massima) tra le generazioni successive, garantendo che il 
            miglior individuo individuato finora venga preservato nella popolazione.
            Verifica se la fitness massima nella generazione corrente (fitness_evolution[generation]) è inferiore
            alla fitness del miglior individuo trovato finora (best_fit)

                        if value < best_fit:
                best_ind = copy.deepcopy(population[np.argmax(pop_fitness)])
                population[0] = copy.deepcopy(best_ind)
                # print("Popolazione 0: ", population[0])
                best_fit = value
            else:
                best_ind = population[pop_fitness.argmax()].copy()
                best_fit = pop_fitness.max()
                indexValue = np.argmax(pop_fitness)
                population[0] = copy.deepcopy(population[indexValue])

            """
            print("Dimensione della popolazione a fine Esecuzine del GA: ", len(population))
        # print("Miglior individuo durante l'evalution:", best_ind)

        return best_ind

    
    def _crossover(self, cromossome1, cromossome2):
        # Estrai solo i geni dai cromosomi, ignorando il fitness
        genes1 = cromossome1[1]
        genes2 = cromossome2[1]

        # Scegli casualmente due indici per il crossover
        indices = sorted(random.sample(range(len(genes1)), 2))

        # Esegui il crossover sui geni
        aux1 = genes1[indices[0]:indices[1]+1]
        aux2 = genes2[indices[0]:indices[1]+1]
        # Trova i geni unici dopo il crossover
        
        # Trova i geni unici dopo il crossover
        diff1 = genes1[:indices[0]] + genes1[indices[1]+1:]
        diff2 = genes2[:indices[0]] + genes2[indices[1]+1:]

        # Crea i figli combinando i segmenti crossover con i geni unici
        child1_genes = aux1 + diff2
        child2_genes = aux2 + diff1


        # Assegna un valore di fitness pari a 0 ai figli
        child1 = (0, child1_genes)
        child2 = (0, child2_genes)

        return child1, child2

    """
    Questa funzione seleziona casualmente due geni nel cromosoma e ne scambia i valori.
    """

    def _mutation(self, population):
        #print("Siamo dentro mutation")
        score = population[0]  # Estrae il punteggio dalla popolazione
        individuals = population[1].copy()  # Copia la lista degli individui

        # Seleziona casualmente due indici diversi
        index1, index2 = random.sample(range(len(individuals)), 2)
        # print("indici", index1,index2)
        # Scambia i due individui di posizione
        individuals[index1], individuals[index2] = individuals[index2], individuals[index1]

        # Costruisce la lista di individui con i nomi e valori modificati
        mutated_individuals = [
            (individuo[0], individuo[1]) if i == index1 or i == index2 else individuo
            for i, individuo in enumerate(individuals)
        ]

        # Ritorna la popolazione mutata con il punteggio originale
        mutated_population = (score, mutated_individuals)

        return mutated_population

    """Il punteggio di fitness viene calcolato sommando i valori restituiti dalla funzione _findOverlap per le 
    sovrapposizioni tra coppie di letture sequenziali nel cromosoma.

    In sintesi, la funzione _fitness somma le sovrapposizioni tra tutte le coppie di letture sequenziali nel 
    cromosoma, restituendo il punteggio totale di fitness del cromosoma. Questo punteggio sarà utilizzato per 
    valutare quanto bene il cromosoma si adatta all'ambiente o all'obiettivo specifico dell'algoritmo genetico."""

    # Usata per calcolare gli overlap tra due letture di interi: return sum(overlap)
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

    # Usata per calcolare L'overlap tra due letture: return overlap
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

    # Funzione di fitness, la quale calcola il punteggio per ogni individuo
    def _fitness(self, cromosoma):
        score = 0
        array_valori = [valori for _, valori in cromosoma]

        for i in range(len(array_valori) - 1):
            finger = array_valori[i]
            finger1 = array_valori[i + 1]
            # print("1 -> ",finger, "2 -> ",finger1)
            # Calcola l'overlap solo se le due sequenze sono identiche

            score += self.findOverlapNew(finger, finger1)
            # print("score", score)
        return score

    """
    La funzione _evaluatePopulation prende in input una popolazione di cromosomi (individui) e calcola i punteggi di fitness
        per ciascun individuo utilizzando la funzione _fitness. 
    Questi punteggi di fitness vengono quindi restituiti come un array di punteggi.
    """

    def _evaluatePopulation(self, population, gen):
        scores = np.zeros(len(population))
        for x in range(len(population)):
            scores[x] = self._fitness(population[x])
        return scores


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
        conta quante volte ciascun prefisso di lunghezza da countRepeat a max_length appare in reads.

    Per ogni lunghezza di prefisso da  countRepeat a max_length, e per ogni stringa in reads, la funzione estrae il prefisso di 
        quella lunghezza e conta quante volte appare in reads, aggiungendo questo conteggio al dizionario repeats_count
"""


def count_repeats(reads, countRepeat):
    # print(reads)
    repeats_count = {}
    max_length = len(reads[0])

    for k in range(countRepeat, max_length + 1):
        for read in reads:
            prefix = read[:k]
            repeats_count.setdefault(prefix, 0)
            repeats_count[prefix] += sum(1 for r in reads if prefix in r)
    # print(repeats_count)
    return repeats_count


def apply_CFL_to_reads(reads, markers):
    cfl_list = []
    marker_indices = []

    # Trova tutti gli indici dei marcatori nella sequenza
    start = 0
    while start < len(reads):
        found_indices = [(reads.find(marker, start), marker) for marker in markers if reads.find(marker, start) != -1]
        if not found_indices:
            break
        idx, marker = min(found_indices, key=lambda x: x[0])
        marker_indices.append((idx, marker))
        start = idx + len(marker)

    if not marker_indices:
        cfl_list.append(reads)
        segments = CFL(reads[:], None)
        result = (reads, segments)
        return result

    # Se la sequenza inizia senza un marcatore, aggiungi la parte iniziale come CFL
    if marker_indices and marker_indices[0][0] > 0:
        cfl_list.append(CFL(reads[:marker_indices[0][0]], None))

    # Calcola le CFL sulla base degli indici dei marcatori
    i = 0
    while i < len(marker_indices) - 1:
        start_idx, current_marker = marker_indices[i]
        next_idx, next_marker = marker_indices[i + 1]

        # Controlla se i marcatori non sono consecutivi
        if start_idx + len(current_marker) <= next_idx:
            cfl_list.append(CFL(reads[start_idx:next_idx], None))

        else:
            # Trova il prossimo marcatore che non si sovrappone
            j = i + 1
            while j < len(marker_indices) and start_idx + len(current_marker) > marker_indices[j][0]:
                j += 1
            if j < len(marker_indices):
                cfl_list.append(CFL(reads[start_idx:marker_indices[j][0]], None))
            else:
                cfl_list.append(CFL(reads[start_idx:], None))
                break
        i += 1

    # Aggiungi l'ultimo segmento se esiste
    if marker_indices:
        last_idx, last_marker = marker_indices[-1]
        if last_idx + len(last_marker) <= len(reads):
            cfl_list.append(CFL(reads[last_idx:], None))

    lista_appiattita = [elemento for sottolista in cfl_list for elemento in sottolista]
    result = (reads, lista_appiattita)
    # print("Lettura: ", reads, "\nMarcatori Utilizzati: ", markers, "\nRappresentazione CFL: ", cfl_list)
    return result




# Given a list of factors return the fingerprint
def compute_fingerprint_by_list_factors(original_list):
    oggetto_trasformato = []
    for chiave, lista_stringhe in original_list:
        # Converti ogni stringa nella lista nella sua lunghezza
        lunghezze = [len(stringa) for stringa in lista_stringhe]
        oggetto_trasformato.append((chiave, lunghezze))
    return oggetto_trasformato


def find_unique_markers(reads, num_markers):
    # Ordina le sequenze per frequenza e lunghezza in ordine decrescente
    sorted_reads = sorted(reads.items(), key=lambda x: (-x[1], -len(x[0])))

    unique_markers = []

    for current_marker in sorted_reads:
        # Verifica che il marcatore corrente non sia un sottoinsieme di nessuno dei marcatori selezionati
        is_subset = False
        for selected_marker in unique_markers:
            if current_marker[0] in selected_marker[0] or selected_marker[0] in current_marker[0]:
                is_subset = True
                break
        if not is_subset:
            unique_markers.append(current_marker)
        if len(unique_markers) == num_markers:
            break

    return unique_markers


def generate_random_populations(reads, num_populations, population_size):
    populations = [reads]
    for _ in range(num_populations - 1):
        population = random.sample(reads, population_size)
        populations.append(population)
    return populations


def qlearning(reads, episodes, genome=None, test_each_episode=False):
    ovr = OverlapResolver(reads)
    root = Node.createRootNode(ovr)
    factor = 1.0 / (len(reads) * max([len(read) for read in reads]))
    generations = episodes

    _cromosomeInt = []
    # print("------------")
    # print(reads)
    sottosequenza_lunghezza = 20

    # Caso in cui abbiamo un dataset diviso in piu letture
    # readsGenoma = ''.join(reads)
    # print(readsGenoma)
    i = 0
    # print(reads)
    genomePartenza = copy.copy(reads)
    reads = createDataset(reads, sottosequenza_lunghezza)
    # print(reads)
    individuo = copy.copy(reads)
    # print(len(count_repeats(reads)))

    # for x in reads:
    # print("Lettura: ", x)

    # Dimensione del marcatore -> da 4 a 8
    countRepeat = 3
    dict = count_repeats(reads, countRepeat)
    # print(dict)
    # print(count_repeats(reads))

    # print("------------")
    marker = []
    # print("Le letture sono:", reads)

    #  Marcatori Indipendenti
    marksIndependent = 3
    max_readss = find_unique_markers(dict, marksIndependent)
    # print('3 marcatori distinti', max_readss)

    # max_readss = find_unique_markers(dict, 2)
    # print('2 marcatori distinti', max_readss)

    markers = chiavi = [sequenza for sequenza, valore in max_readss]
    # print("Combinazione dei marcatori", markers)

    #print("------------")
    results = []
    results2 = []
    for x in range(len(reads)):
        # print("Lettura", reads[x])
        resul = apply_CFL_to_reads(reads[x], markers)
        # risp = apply_CFL_to_reads(reads[x], markers)
        # print("Due marcatori:", risp)
        # print("Tre marcatori:", resul)
        results.append(resul)
        # results2.append(risp)

    # print("Risultato ottenuto:", results)

    _intA = compute_fingerprint_by_list_factors(results)
    # _intB = compute_fingerprint_by_list_factors(results2)
    # print("------------")

    num_ind = 200
    iterazioni = 200
    ga = GA()

    # Creo una lista di indici
    indices = list(range(len(_intA)))
    # print(_intA)
    indConfront = copy.copy(_intA)
    popolazioni_mescolate = generate_random_populations(_intA, num_ind, len(_intA))
    # popolazioni_mescolate2 = generate_random_populations(_intB, num_ind, len(_intB))
    # print(popolazioni_mescolate)

    # print("----")
    for i, sublist in enumerate(popolazioni_mescolate):
        #print(sublist)
        chiavi = [tupla[0] for tupla in sublist]
        array_di_interi = [tupla[1] for tupla in sublist]
        # Conversione della lista di chiavi in una singola stringa
        stringa_chiavi = ''.join(chiavi)
        chunks = [stringa_chiavi[i:i + len(individuo[0])] for i in range(0, len(stringa_chiavi), len(individuo[0]))]
        # print(chunks)
        # if individuo == chunks:
        # print("Trovato", chunks)

    ind_evolved = list([ga.run_ga(None, popolazioni_mescolate, indConfront, iterazioni)][0])
    # print("----- Passaggio alla seconda esecuzione del GA")

    # ind_evolved2 = list([ga.run_ga(None, popolazioni_mescolate2, reads)][0])
    # print("--------------------")
    # print("Siamo nel metodo Q-learning: ")
    # print("--------------------")
    # print("Popolazione ottenuta tramite l'esecuzione dell'algoritmo genetico su Tre marcatori: ", ind_evolved)
    # print("Popolazione ottenuta tramite l'esecuzione dell'algoritmo genetico su Due marcatori: ", ind_evolved2)

    genomePopolation = assemble_genome_with_overlaps(ind_evolved[1])
    # genomePopolation2 = assemble_genome_with_overlaps(ind_evolved2)

    # print("Due genomi da confrontare:\n", "Genoma di partenza:\n", genomePartenza,"\nGenoma  ottenuto dal GA dei 3 marcatori:\n",genomePopolation)
    # print("Distanza di Levenshtein dati i",markers , "marcatori e di: ", levenshtein(genomePartenza, genomePopolation))


    # print("--------------------")

    # print("genoma", genome)
    # test = test_ga(root, factor, genome, ind_evolved, reads)

    # print("ind_evolved:", ind_evolved, "test_rw:", "%.5f" % test[1], "test:", test[0], "dist:", test[2])

    filename = "Test_" + str(marksIndependent) + "Marks_" + str(countRepeat) + "Dim" + ".txt"
    print(filename)

    contenuto = f"NumIndividui: {num_ind}, NumIterazioni: {iterazioni}, Distanza di Levenshtein dati i markers {markers} e di: {levenshtein(genomePartenza, genomePopolation)}\n\nGenomaPartenza:{genomePartenza}\nGenomaOttenuto:{genomePopolation}\n\n"

    with open(filename, 'a') as file:
        file.write(contenuto)


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
    # print("reads:", reads)
    genome = reads[0][0]
    # print("qui", genome)
    for i in range(len(reads) - 1):
        current_read, current_overlaps = reads[i]
        current_read2, current_overlaps2 = reads[i + 1]
        # print(current_read, current_overlaps, "\n", current_read2, current_overlaps2)

        lista = GA.findOverlapGenoma(None, current_overlaps, current_overlaps2)
        # print("lista", lista)
        if (len(lista) > 0):
            stringa_concatenata = current_read2[sum(lista):]
            # print("Stringa conc", stringa_concatenata)
            genome += stringa_concatenata
            # print("Genoma", genome)
        else:
            genome += current_read2
            # print("Genoma", genome)
    return genome




def createDataset(dataset, lunghezza_sottosequenza):
    # Definiamo i 4 caratteri
    # caratteri = ['A', 'C', 'G','T']

    # Creiamo un dataset di 2000 caratteri scelti casualmente tra i 4
    # dataset = ''.join(np.random.choice(caratteri) for _ in range(2000))

    # print(dataset)
    # Dividiamo il dataset in sottosequenze di 100 caratteri
    # sottosequenze = [dataset[i:i + 100] for i in range(0, len(dataset), 100)]
    # return sottosequenze

    # Calcola il numero massimo di pezzi sovrapposti
    sottosequenze = []
    indice_inizio = 0
    k = 0
    while indice_inizio < len(dataset) - lunghezza_sottosequenza + 1:
        # Genera la sottosequenza corrente
        sottosequenza = dataset[indice_inizio:indice_inizio + lunghezza_sottosequenza]
        # print("sottosequenza estratta", sottosequenza)
        sottosequenze.append(sottosequenza)
        # print("sottosequenza estratta", sottosequenze)
        # Seleziona un indice casuale per l'overlap
        indice_casuale = random.randint(1, lunghezza_sottosequenza - 1)
        # print("INDICE CASUALE", indice_casuale)
        # Aggiorna l'indice di inizio per la prossima sottosequenza
        indice_inizio += indice_casuale

        # Verifica se la fine della sequenza è stata raggiunta
        if indice_inizio + lunghezza_sottosequenza > len(dataset):
            # Aggiungi l'ultima sottosequenza che include la fine della sequenza
            sottosequenze.append(dataset[-lunghezza_sottosequenza:])
            break
        k = k + 1
    """
    i = 0
    car = 0
    print(sottosequenze)
    for x in sottosequenze:
        print(len(x))
        car += len(x)
        i += 1

    print("DIM", len(sottosequenze),i, car)
    """

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
    #readProva = 'TCATATCCCTAGAGTGCAATAGCTGAGTGAGTAGCCGTAGGTTCTGCGCGATGCAGTGTCCCTGAATAATCCAAACAACCTCGCCGCGGTCGCATGCGCCGCACGAAAGCCGGAAACTATTCACCTCTGTTTACTGAATGCTATGCGGAGCAGGAACCAGCAATCCTCGATTGTCTCCAGCGTAAAGAAGTGTCGCGCTCTTCCTTGATCACTAACGCGCAGCGGTAGCAAGATCTGCTTTCTACGGTTACGCGAACCAAACAGACTTGGGCGGCCACCTGCAGGTCAAGTACTAATATATAAGCACGGGAATACCACATCATGACGTGAACGATCGCAGCCTTAAAGACAGAATGTATATGCCTAGGCCCGCATATGCCCAACGACTTACAATCGGTGTATCCCTCTAGGTTGAGATCAACAGGAGTAGTCACCTTGAACCTGATATTGGAAGAGCGTGGTGCTGCACACCAAGGTGATCGGAGGTACGTGCAGGGTTACTAGCGATGCAGCAGGCAATGATTTGTTACTTATATCATTGTACGCAACAAGGTTGTGGGGAGGTTGCGTAAATCGGCGGCGCCCCGCCTTCCTCTACCCGGACATCGATTTTTCCGACCTCCACGAGAACTACTCGAGAACCCGAGCCTGAGTAAACCGGTATACAACTCTAGGCAAGTGCGCTACCCCTTTTACGCGTGAACGGAGCCGCTTTTCCCCCATAGTGCGTAAAGCGGTATGTTTAAATTTACTGTGGCGTTATGCGTCGCAGGTGTATGACCGGCTCGTCAGCGGCCACAGGCATCACGTAATATTTAGCGCTGGTCTTTGTTTTCTGTGATCGAATGGAAGGAGTCATTTATGCCACGAGGATATGACGAATAGTCTATCGTCTGCTAGGCAAGGTAAAAAAGTCAAGAATGAGACGGTGTTTGGCGCTATACCCCACTACAGAAACATATTGCTGCCCCGCGGCTCATGTCGTGCTGGGGTCCCTGTATAACAGCTGACACGACAAGCCGAGGCATCTATGACATCGACTAAAACTCTGGGTCGCGTTATGGTGGACCAGGCACGTACGGGGCGTAGCGCCTATTAAATTAGTCCAAAAGACATTTTTTGGTGACAGTGCTGCCCGACGACGTCCCTAGAATAACCAAAATAGGTCACAAAATATTGTCTTGTTCATGATAATCGATCTTTTTTTGGCAAAGCATCAGAAGTCTACCAGTCAGTTCTTAGCCCAGTGAGAGGGTGATTGGGCGCCAGATCGTAGTCAAATTACGGAGACGATTCTTTGCGTAAAATTGCTCCCGTGAGGGCGAGAATCGGAACAGCGACGATTTATTGCGGCGCGACTCGGGAGATTGACAGGAATACCGAATGGCTAGCTTGTAAATTTAAATAGGAATCCATTGTTCCTAAAGCAGATTAGCGCCGATCCGAGCGTAAACCGGCCGCTGAACGCACGGCGTCATCTGGTTGAACTACTATTGGTAGTAGGAATCACATATGGGTGGTTACTTGTTAGCTTTGTACGCATTGGTTATTCCGCAAAAGGTACAGACTGAACCACTATGTAGCATCCATGTTCTCGATGGCACAAGTTCTCACATGTACGTCATCACGGCACCTGACGCCTAGTTGACCAAAATCTCCGTTGCGGCGACAAACGGCTTCCCTATGAAACGGCATGCAGTCATTTCGGCACACGAGATATTGGGGACAGTGCCTAACTCTCGGTGCCCCTTTTAAAGCAAAATGATGCTTGGTGGCTGGTTACAAAGCCCAGCAGGCATCTCGGATAGTTGTCGCATTTTCTGTCGACAATCGTGACTAGTTGATCTGCACACATAGATGGGCTTACTCCATGCGGCATTTACGCTATCGTATCGGTCATTTACACTACTGCAGGACAGCGAGCGGGGCGTCCATCGAACATGAAGTTCAGGACGGCAACGTGTGGTTAATGTCCTGCGAAGCTTTAACTTAAAGGCGAT'
    #readProva = 'TCATATCCCTCCCCTTAGC'
    readProva = 'AACACTGCAACTCTAAAACACTGCAACTAACACTGCAACTCTAAAACACTGCAACTCTAACACTGCACACTGCACTGCAACTCTAACACTGCACACTGCACTATCATATCCCTAGAGTGCAATAGCTGAGTGAGTAGCCGTAGGTTCTGCG'

    dataset[24] = (genome25, readProva)

    genome, reads = dataset[24]
    for i in range(25):
        print(f"Esecuzione {i + 1}/20")
        qlearning(reads, 1, genome)