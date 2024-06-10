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

    def run_ga(self, env, pop, gen):
        print("----- Siamo nel metodo run_GA -----")
        # print("Stampa delle varie letture: ", reads)
        pop_size = len(pop)

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
        best_fit = self._fitness(best_ind)
        # print("Valore fitness dell'individuo migliore: ", best_fit)

        mescolamento = 100
        for generation in range(mescolamento):
            print("---------")
            print('Esecuzione')
            # Tournament selection
            selected = []
            # seleCpy = []
            for i in range(len(population)):
                selected.append(population[i].copy())

            # Crossover
            for i in range(0, pop_size, 2):
                # print("i:", i)
                if i + 1 < len(selected):
                    if np.random.rand() < self.crossover_prob:
                        # print("-----------------")
                        # print("Popo", selected[i])
                        # print("Popo1", selected[i + 1])
                        population[i], population[i + 1] = self._crossover(selected[i], selected[i + 1])
                        # print("Popolazioni ottenute tramite il crossover: \n", population[i])
                        # print(population[i + 1])
                    else:
                        # Qui invece di eseguire l'operazione di crossover, copiamo semplicemente i genitori nella nuova generazione
                        population[i], population[i + 1] = selected[i].copy(), selected[i + 1].copy()
                        # print("Operazione di crossover non andata a buon fine: ", population[i], population[i + 1])
            # Mutation
            # print("-----------------")

            # print("Siamo nel metodo mutation:")
            for i in range(len(population)):
                if np.random.rand() > self.mutation_prob:
                    population[i] = self._mutation(population[i])
                    # print("Popolazione corrente nel mutation: ", "\n", population[i])

            # print("Stampa della popolazione dopo l'operazione di mutation:")

            # print("----------------------")

            print("Ridefiniamo i valori fitness della popolazione modificata: ")
            # print(array)
            # print("----------------------")

            population.append(copy.deepcopy(best_ind))
            # print("Popolazione dopo l'aggiunta di", best_ind)

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
                # print("Popolazione 0: ", population[0])
                best_fit = value
            else:
                best_ind = population[pop_fitness.argmax()].copy()
                best_fit = pop_fitness.max()
                indexValue = np.argmax(pop_fitness)
                population[0] = copy.deepcopy(population[indexValue])

        # print("Miglior individuo durante l'evalution:", best_ind)

        return best_ind

    # Funzioni non utilizzate
    """
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

    """
    La funzione compute_overlap prende due stringhe left_read e right_read come input e cerca la sovrapposizione tra di esse.
    La sovrapposizione è la parte comune tra il suffisso di left_read e il prefisso di right_read.
    ritorno della funzione -> Se trova una corrispondenza completa, restituisce la sovrapposizione e la sua lunghezza.

    def compute_overlap(left_read, right_read):
        for i in range(len(left_read)):
            l = left_read[i:]
            size = len(l)
            r = right_read[:size]
            if l == r:
                return l, size
        return "", 0

    """

    """
    La funzione _getOverlap calcola la sovrapposizione massima tra due stringhe s1 e s2 utilizzando una matrice di programmazione dinamica.
    La sovrapposizione è calcolata sommando i punteggi delle sovrapposizioni tra i prefissi delle due stringhe.
    La funzione restituisce il valore massimo della sovrapposizione ottenuta.

    def _getOverlap(self, s1, s2, match, mismatch, gap):
        l = len(s1) + 1
        c = len(s2) + 1
        matrix = np.array([0.0 for _ in range(l * c)]).reshape(l, c)
        for i in range(1, l):
            for j in range(1, c):
                matrix[i][j] = self._getOverlapValue(i, j, matrix, s1, s2, match, mismatch, gap)
        return np.max(matrix)


    def _getSuffixPrefixOverlap(self, left, right):
        return self.compute_overlap(left, right)[1]
    """

    """
    La funzione _findOverlap calcola la lunghezza della sovrapposizione tra due sequenze, identificate dagli 
    indici id1 e id2, presenti nella lista reads. La sovrapposizione rappresenta la quantità di caratteri comuni tra 
    una parte finale della sequenza reads[id1] e una parte iniziale della sequenza reads[id2].

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
    """

    """
    La funzione _crossover implementa l'operatore di crossover per l'algoritmo genetico. In particolare,
    prende in input due cromosomi (cromossome1 e cromossome2) e restituisce due nuovi cromosomi (figli) ottenuti 
    combinando porzioni dei cromosomi genitori.

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
    """

    def _crossover(self, cromossome1, cromossome2):

        # print("Stampiamo due vincitori del torneo _ring: ", "\n", cromossome1, "\n", cromossome2)

        genes = np.random.choice(len(cromossome1), size=2, replace=False)
        genes.sort()
        # print("-----",genes)

        aux1 = cromossome1[genes[0]:genes[1] + 1]
        # print("Estrazione dal cromosoma 1", " con valori ", aux1)
        aux2 = cromossome2[genes[0]:genes[1] + 1]
        # print("Estrazione dal cromosoma 2", " con valori ", aux2)
        # print("-----")

        diff1 = [gene for gene in cromossome2 if gene not in aux1]
        # print("Parti del cromosoma non incluse durante l'operazione di crossover eseguito: ", diff1)
        diff2 = [gene for gene in cromossome1 if gene not in aux2]
        # print("Parti del cromosoma non incluse durante l'operazione di crossover eseguito: ", diff2)
        # print("-----")

        aux1.extend(diff1)
        aux2.extend(diff2)
        # print("Stampa dei figli ottenuti combinando gli elementi estratti dal primo e secondo cromosoma, con l'aggiunta degli elementi che non sono stati presi durante il crossover ", "\n", aux1, "\n", aux2)
        # print("-----")
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
        # print("Mutazione", mutated_population)

        # Seleziona casualmente due indici diversi
        index1, index2 = random.sample(range(len(population)), 2)
        # print("Indici casuali selezionati:", index1, index2)

        # Scambia i due individui di posizione
        mutated_population[index1], mutated_population[index2] = mutated_population[index2], mutated_population[index1]

        # print("Stampa della popolazione dopo l'operazione di mutation:")

        # print(mutated_population)

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

        return score

    """
    La funzione _evaluatePopulation prende in input una popolazione di cromosomi (individui) e calcola i punteggi di fitness
        per ciascun individuo utilizzando la funzione _fitness. 
    Questi punteggi di fitness vengono quindi restituiti come un array di punteggi.
    """

    def _evaluatePopulation(self, population, gen):
        scores = np.zeros(len(population))

        # min_distance = float('inf')  # Inizializza la distanza minima come infinito
        min_genome = None  # Inizializza il genoma con la distanza minima
        popy = []
        print("----- Siamo dentro il metodo _evaluatePopulation ")
        print("----- Siamo nel metodo _fitness -----")
        # print("Stampa della popolazione: ", population)
        # fitness_map = {}
        for x in range(len(population)):
            # print("Popolazione ", population[x])
            scores[x] = self._fitness(population[x])
            # genomePopolation = assemble_genome_with_overlaps(population[x])
            # print("Due genomi da confrontare:", "\nGenoma Partenza: ", gen)
            # print("Genoma Ottenuto dalla Popolazione corrente: ", genomePopolation)
            # print("Distanza di Levenshtein: ", levenshtein(gen, genomePopolation))
            # current_distance = levenshtein(gen, genomePopolation)
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
        lista_appiattita = [elemento for sottolista in segments for elemento in sottolista]
        result = (reads, lista_appiattita)
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
    return result


def process_read(reads, marker, ma, mamma):
    # print("Sono nell'else, marcatori utilizzati sono:", marker, ma, mamma)
    cfl_list = []
    marker_indices = []

    # Trova tutti gli indici dei marcatori nella sequenza
    start = 0
    while start < len(reads):
        idx1 = reads.find(marker, start)
        idx2 = reads.find(ma, start)
        idx3 = reads.find(mamma, start)
        if idx1 == -1 and idx2 == -1 and idx3 == -1:
            break
        if idx1 != -1 and (idx2 == -1 or idx1 < idx2) and (idx3 == -1 or idx1 < idx3):
            marker_indices.append((idx1, marker))
            start = idx1 + len(marker)
        elif idx2 != -1 and (idx1 == -1 or idx2 < idx1) and (idx3 == -1 or idx2 < idx3):
            marker_indices.append((idx2, ma))
            start = idx2 + len(ma)
        elif idx3 != -1:
            marker_indices.append((idx3, mamma))
            start = idx3 + len(mamma)

    if not marker_indices:
        cfl_list.append(reads)
        segments = CFL(reads[:], None)
        lista_appiattita = [elemento for sottolista in segments for elemento in sottolista]
        result = (reads, lista_appiattita)
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
    return result

""" Non più utilizzata
def apply_CFL_to_reads(reads, marker, ma, terzo):
    result = {}
    # print(reads)
    flag = True
    # print(GA.compute_overlap(marker,ma))
    valueOverlap = GA.compute_overlap(marker, ma)
    segments = []

    # print(valueOverlap[1])
    # Caso in cui sono sottoinsiemi tra di loro
    if is_subset(marker, ma) or is_subset(ma, marker):
        print("Sono nel if")
        # Utilizzo il marcatore più grande
        useMarker = marker if len(marker) > len(ma) else ma
        # Se il marcatore è presente, calcolo la CFL.
        if find_marker_index(reads, useMarker) != -1:
            # print(find_marker_positions(reads,useMarker))
            # Qui ottengo una tupla delle posizioni del marcatore
            listaMarker = find_marker_positions(reads, useMarker)
            # Qui ottengo una lista delle posizioni
            start_positions = [start for start, length in listaMarker]
            # print(start_positions)

            # Qui vado a eliminare i marcatori innestati.
            list = []
            for x in range(len(start_positions) - 1):
                if start_positions[x] + len(useMarker) > start_positions[x + 1]:
                    list.append(start_positions[x + 1])
            # print(list)

            # Qui effettuo delle operazioni per ottenere solamente gli indici dei marcatori.
            set1 = set(start_positions)
            set2 = set(list)
            set3 = set1.difference(set2)
            l = []
            for x in set3:
                l.append(x)
            start_positions = l

            # print(start_positions)

            # Qui vado a inserire lo 0 se non è presente, permettendomi di iniziare a calcolare la CFL da quel punto
            if 0 not in start_positions:
                start_positions.append(0)
            start_positions.sort()
            # print(start_positions)

            # Cicla attraverso la lista di indici
            for i in range(len(start_positions) - 1):
                start = start_positions[i]
                end = start_positions[i + 1]

                # Estrai il pezzo della stringa e calcola la CFL
                segment = CFL(reads[start:end], None)
                segments.append(segment)
            # Calcola la CFL per l'ultimo segmento (dall'ultimo indice fino alla fine della stringa)
            last_segment = CFL(reads[start_positions[-1]:], None)
            segments.append(last_segment)
            lista_appiattita = [elemento for sottolista in segments for elemento in sottolista]
            result = (reads, lista_appiattita)
            flag = False
            # print('qui', result)
            return result
        else:
            # Caso in cui non trovo il marcatore nella lettura -> calcolo la CFL su tutta la reads
            segments.append(CFL(reads[:], None))
            lista_appiattita = [elemento for sottolista in segments for elemento in sottolista]
            result = (reads, lista_appiattita)
            flag = False
            return result
    # Caso in cui trovo un overlap tra suffisso e prefisso dei marcatori
    else:
        print("Sono nell'else, marcatori utilizzati sono:", marker, ma)
        cfl_list = []
        marker_indices = []
        # print(reads)
        # Trova tutti gli indici dei marcatori nella sequenza
        start = 0
        while start < len(reads):
            idx1 = reads.find(marker, start)
            idx2 = reads.find(ma, start)

            if idx1 == -1 and idx2 == -1:
                break
            if idx1 != -1 and (idx2 == -1 or idx1 < idx2):
                marker_indices.append((idx1, marker))
                start = idx1 + len(marker)
            elif idx2 != -1:
                marker_indices.append((idx2, ma))
                start = idx2 + len(ma)

        if not marker_indices:
            cfl_list.append(reads)
            segments = CFL(reads[:], None)
            lista_appiattita = [elemento for sottolista in segments for elemento in sottolista]
            result = (reads, lista_appiattita)
            # print("Caso in cui non trovo nessun marcatore per la lettura", result)
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
        return result
"""


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
    populations = []
    for _ in range(num_populations):
        population = random.sample(reads, population_size)
        populations.append(population)
    return populations


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
    # print(readsGenoma)

    reads = createDataset(reads, sottosequenza_lunghezza)
    # print(reads)
    # print(len(count_repeats(reads)))

    # for x in reads:
    # print("Lettura: ", x)
    countRepeat = 15
    dict = count_repeats(reads, countRepeat)
    # print(dict)
    # print(count_repeats(reads))

    print("------------")
    marker = []
    # print("Le letture sono:", reads)

    marksIndependent = 7
    max_readss = find_unique_markers(dict, marksIndependent)
    print('3 marcatori distinti', max_readss)

    #max_readss = find_unique_markers(dict, 2)
    #print('2 marcatori distinti', max_readss)

    markers = chiavi = [sequenza for sequenza, valore in max_readss]
    print("Combinazione dei marcatori", markers)

    print("------------")
    results = []
    results2 = []
    for x in range(len(reads)):
        #print("Lettura", reads[x])
        resul = apply_CFL_to_reads(reads[x], markers)
        #risp = apply_CFL_to_reads(reads[x], markers)
        #print("Due marcatori:", risp)
        #print("Tre marcatori:", resul)
        results.append(resul)
        #results2.append(risp)

    # print("Risultato ottenuto:", results)

    _intA = compute_fingerprint_by_list_factors(results)
    #_intB = compute_fingerprint_by_list_factors(results2)
    print("------------")

    # Modificato da me, prima era 60
    num_ind = 200
    ga = GA()

    # Creo una lista di indici
    indices = list(range(len(_intA)))

    popolazioni_mescolate = generate_random_populations(_intA, num_ind, len(_intA))
    #popolazioni_mescolate2 = generate_random_populations(_intB, num_ind, len(_intB))
    # print(popolazioni_mescolate)

    print("----")
    for i, sublist in enumerate(popolazioni_mescolate):
        # print("Popolazione", sublist)
        chiavi = [tupla[0] for tupla in sublist]

        # Conversione della lista di chiavi in una singola stringa
        stringa_chiavi = ''.join(chiavi)
        # print(stringa_chiavi)
        # print(len(stringa_chiavi))


    ind_evolved = list([ga.run_ga(None, popolazioni_mescolate, reads)][0])
    #print("----- Passaggio alla seconda esecuzione del GA")

    #ind_evolved2 = list([ga.run_ga(None, popolazioni_mescolate2, reads)][0])
    print("--------------------")
    print("Siamo nel metodo Q-learning: ")
    print("--------------------")
    print("Popolazione ottenuta tramite l'esecuzione dell'algoritmo genetico su Tre marcatori: ", ind_evolved)
    #print("Popolazione ottenuta tramite l'esecuzione dell'algoritmo genetico su Due marcatori: ", ind_evolved2)

    genomePopolation = assemble_genome_with_overlaps(ind_evolved)
    #genomePopolation2 = assemble_genome_with_overlaps(ind_evolved2)

    print("Due genomi da confrontare:\n", "Genoma di partenza:", readsGenoma,
          "\nGenoma  ottenuto dal GA dei 3 marcatori:",
          genomePopolation)
    print("Distanza di Levenshtein dati i 3 marcatori: ", levenshtein(readsGenoma, genomePopolation))

    """
    print("---------------")
    print("Due genomi da confrontare:\n", "Genoma di partenza:", readsGenoma,
          "\nGenoma  ottenuto dal GA dei 2 marcatori:",
          genomePopolation2)
    print("Distanza di Levenshtein dati i 2 marcatori: ", levenshtein(readsGenoma, genomePopolation2))
    """
    print("--------------------")

    # print("genoma", genome)
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
    # readProva = 'TCATATCCCT'
    #readProva = 'AACACTGCAACTCTAAAACACTGCAACTAACACTGCAACTCTAAAACACTGCAACTCTAACACTGCACACTGCACTGCAACTCTAACACTGCACACTGCACTATCATATCCCTAGAGTGCAATAGCTGAGTGAGTAGCCGTAGGTTCTGCG'

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
