from overlap import OverlapResolver
from nodeoriginal import Node
from itertools import permutations
import random
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
"""
ovr: Oggetto OverlapResolver creato a partire dalle sequenze di DNA (reads).
-> Questo oggetto rappresenta il risolutore di sovrapposizioni per le sequenze.

root: Nodo radice creato a partire dall'oggetto ovr.
-> Questo nodo rappresenta l'inizio dell'albero delle decisioni
"""


def qlearning(reads, episodes, genome=None, test_each_episode=False):
    ovr = OverlapResolver(reads)
    print("Prima stampa delle letture", ovr.reads)

    #Creazione di un grafo
    grafo = nx.DiGraph()

    root = Node.createRootNode(ovr)

    print("Nodo corrente: ", root)
    print("----------------------")
    """
    actions_taken, leafs_reached: Contatori per il numero totale di azioni eseguite e il numero di nodi foglia raggiunti durante il processo di apprendimento.
    epsilon: Parametro di esplorazione iniziale, utilizzato per controllare la probabilità di esplorazione o sfruttamento durante la scelta delle azioni.
    gamma: Fattore di sconto che controlla quanto pesare le ricompense future rispetto alle ricompense immediate durante l'apprendimento.
    alpha: Tasso di apprendimento, che controlla quanto l'algoritmo impara dai nuovi dati.
    epsilon_decay: Valore di decadimento per l'epsilon, che regola la probabilità di esplorazione.
    factor: Fattore utilizzato per calcolare la ricompensa associata all'azione durante l'apprendimento.
    """
    actions_taken, leafs_reached = 0, 0
    epsilon = 1.0
    gamma = 0.9
    alpha = 0.8
    epsilon_decay = 1.0 / episodes
    factor = 1.0 / (len(reads) * max([len(read) for read in reads]))

    """
    Ciclo degli episodi: Il ciclo si ripete per il numero specificato di episodi.
    cur_node: Inizializzato con il nodo radice all'inizio di ogni episodio.
    total_reward: Inizializzato a zero, rappresenta la somma delle ricompense ottenute durante l'episodio.
    actions_train: Lista che terrà traccia delle azioni compiute durante l'episodio.
    """
    count = 0
    for episode in range(episodes):
        print("Ciclo: ", count)
        count+=1
        cur_node = root
        print("Azioni associate a quel nodo: ", cur_node.get_outputs(), "durante l'episodio corrente:", episode)

        total_reward = 0.0
        print("Ricompense ricevute: ", total_reward)
        actions_train = []
        print("Lista che terrà traccia delle azioni compiute durante l'episodio: ", actions_train)

        print("---------------------------------")
        print("---------------------------------")

        """
        Ottieni le Azioni Possibili:

        candidates: Ottiene la lista di azioni possibili dal nodo corrente (cur_node) chiamando il metodo get_outputs() del nodo.
        -> Queste azioni rappresentano le possibili decisioni che l'algoritmo può prendere in base allo stato corrente.
        Verifica la Fine del Ramo:

        -> Se la lunghezza di candidates è zero, significa che non ci sono azioni possibili dal nodo corrente.
        In questo caso, incrementa il contatore leafs_reached (nodi foglia raggiunti) e interrompe il ciclo while usando il comando break.
        Selezione Casuale di un'Azione:

        -> Se ci sono azioni possibili, viene selezionata casualmente un'azione dalla lista di candidates.
        random.sample(candidates, 1) restituisce una lista contenente un elemento estratto casualmente da candidates.
        -> L'elemento selezionato è quindi assegnato alla variabile action.
        """

        """
        Selezione dell'Azione:
        -> Viene generato un numero casuale tra 0 e 1 con random.random() e assegnato a rand.
        -> Se il valore di rand è maggiore di epsilon, l'algoritmo sceglie di sfruttare piuttosto che esplorare.
        cur_node.get_max_action() restituisce l'azione con il massimo valore Q dal nodo corrente (cur_node). Questo rappresenta una politica di sfruttamento.
        """
        while True:
            print("Letture generali: ", reads)
            print("----------------------------")

            print("Lettura: ", cur_node.get_read_content(), "del nodo ",cur_node)
            # La capacità
            print("Max accuraccy ", ovr.max_acc)
            candidates = cur_node.get_outputs()
            print("azioni possibili dal nodo corrente (cur_node) ", candidates)
            if len(candidates) == 0:
                leafs_reached += 1
                break
            action = random.sample(candidates, 1)[0]
            print("viene selezionata casualmente un'azione dalla lista di candidates: ", action)


            rand = random.random()
            print("Viene generato il numero casuale, se esso > epsilon allora: ")
            print("Valore random: ", rand, " e valore epsilon: ", epsilon)
            if rand > epsilon:  # and cur_node != root:
                a = cur_node.get_max_action()
                print("E successo questo ", cur_node.get_max_action())
                if a is not None:
                    action = a

            """
            Transizione di Stato:
            -> Viene ottenuto il nodo successivo (next_node) corrispondente all'azione scelta.
            -> Se next_node è None, significa che l'azione scelta non ha portato a un nuovo stato valido, quindi il ciclo viene interrotto.
            """
            next_node = cur_node.get_child_original(action)
            print("Questo è il nodo successivo: ", next_node, "corrispondente all'azione scelta: ", action)
            if next_node is None:
                break



            """
            Calcolo della Ricompensa:
            -> Viene calcolata la ricompensa associata all'azione corrente.
            Se il nodo corrente è la radice (cur_node == root), viene assegnata una piccola ricompensa di 0.1.
            Altrimenti, la ricompensa è proporzionale alla sovrapposizione di coppie nel nodo successivo (next_node.pairwise_overlap) moltiplicata per factor.
            -> Viene inoltre aggiunto 1.0 alla ricompensa se next_node è una foglia (un'estremità dell'albero), altrimenti si aggiunge 0.0.
            """
            print("Qui andiamo a calcolare le ricompense: ")
            reward = 0.1 if cur_node == root else next_node.pairwise_overlap * factor
            print("Ricompensa 0: ", reward)
            reward += 1.0 if next_node.is_leaf() else 0.0
            total_reward += reward
            print("Ricompensa 1: ", total_reward)

            label = "{} - {:.2f}".format(action, total_reward)
            grafo.add_edge(cur_node.get_read_content(),next_node.get_read_content(), label = label)
            """
            Aggiornamento Q-value:
            -> Viene chiamato il metodo update_q del nodo corrente per aggiornare il valore Q associato all'azione corrente.
            L'aggiornamento si basa sulla ricompensa ottenuta e sulla stima del massimo valore Q del nodo successivo (next_node).
            -> alpha è il tasso di apprendimento che controlla quanto l'algoritmo impara dai nuovi dati.
            """
            print("Stampa del cur_node prima: ", cur_node)
            cur_node.update_q(action, reward + gamma * next_node.get_max_qvalue(), alpha)
            actions_taken += 1
            print("Azioni prese: ", actions_taken)
            actions_train.append(action)
            print("Lista che terrà traccia delle azioni compiute durante l'episodio: ", actions_train)
            cur_node = next_node
            print("Stampa del cur_node Finale: ", cur_node)
            print("---------------------------------")
            print("---------------------------------")

        """
        Test e Stampa dei Risultati:
        -> Dopo ogni episodio, viene chiamata la funzione test_qlearning per valutare le prestazioni dell'algoritmo su un genoma di riferimento (genome).
        -> La variabile test conterrà una tupla che include le azioni eseguite durante il test, la ricompensa totale ottenuta e la distanza di Levenshtein rispetto al genoma di riferimento.
        -> I risultati vengono quindi stampati, inclusi l'episodio corrente, l'accuratezza massima raggiunta, la ricompensa totale durante il training, la ricompensa totale durante il test e la distanza di Levenshtein.
        -> epsilon viene aggiornato utilizzando epsilon_decay per ridurre la probabilità di esplorazione ad ogni episodio.
        """
        if test_each_episode or episode + 1 == episodes:
            test = test_qlearning(root, factor, genome)
            print("Test Finale: ", test,"Genoma: ", genome)
        else:
            test = (None, 0.0, None)

        print("ep.:", episode + 1, "max_acc:", ovr.max_acc, "train_rw:", "%.5f" % total_reward,
              "test_rw:", "%.5f" % test[1], "test:", test[0], "train", actions_train, "dist:", test[2])
        epsilon -= epsilon_decay
        print("actions_taken", actions_taken)
        print("leafs_reached", leafs_reached)
        print(ovr.reads)
        print("----------------")

        pos = nx.spring_layout(grafo)
        nx.draw(grafo, pos, with_labels=True, arrows = True)
        labels = nx.get_edge_attributes(grafo, 'label')
        nx.draw_networkx_edge_labels(grafo, pos, edge_labels=labels)

        nodi = list(grafo.nodes())
        archi = list(grafo.edges())

        array_grafo = {
            "nodi": nodi,
            "archi": archi
        }

        print(array_grafo)

        plt.show()

"""Certamente, il codice è un'implementazione semplificata di un algoritmo di apprendimento per rinforzo chiamato 
Q-learning. Ecco un riassunto delle parti principali:

            -> Preparazione Iniziale:

                    epsilon, gamma, alpha, epsilon_decay, factor sono parametri dell'algoritmo.
                    actions_taken e leafs_reached sono contatori.
            -> Ciclo degli Episodi:

                    Itera attraverso un numero specificato di episodi.
            -> Selezione dell'Azione:

                    All'interno di ciascun episodio, l'algoritmo esegue la selezione delle azioni tramite un ciclo while.
                    Viene scelto casualmente un'azione tra le azioni possibili dal nodo corrente.
            -> Transizione di Stato:

                    Viene ottenuto il nodo successivo corrispondente all'azione scelta.
            -> Calcolo della Ricompensa:

                    Calcola la ricompensa associata all'azione corrente.
                    La ricompensa è basata sulla sovrapposizione di coppie nel nodo successivo.
            -> Aggiornamento Q-value:

                    Aggiorna il valore Q associato all'azione corrente utilizzando la ricompensa ottenuta e la stima del massimo valore Q del nodo successivo.
            -> Test e Stampa dei Risultati:

                    Dopo ogni episodio, valuta le prestazioni dell'algoritmo su un genoma di riferimento.
                    Stampa risultati, inclusi episodio corrente, accuratezza massima raggiunta, ricompensa totale durante il training, ricompensa totale durante il test e distanza di Levenshtein.
                    Aggiorna epsilon utilizzando epsilon_decay per ridurre la probabilità di esplorazione ad ogni episodio.
            -> Produzione:

                    Stampa i risultati dell'apprendimento, tra cui i dettagli dell'episodio, l'accuratezza massima, le ricompense e le azioni durante il training e il test.
                    In generale, l'algoritmo sta imparando una politica di decisione basata su Q-values per eseguire azioni in un ambiente descritto da un grafo. 
                    Durante il processo di apprendimento, vengono considerate le ricompense immediate e future per guidare le decisioni. La fase di test valuta l'algoritmo su un genoma di riferimento.
"""

"""
Test Q-learning:
-> La funzione test_qlearning valuta le prestazioni dell'algoritmo Q-learning su un genoma di riferimento (genome).
-> Utilizza l'algoritmo di Levenshtein per calcolare la distanza tra la sequenza ottenuta e il genoma di riferimento.
"""


def test_qlearning(root_node, factor, genome):
    def levenshtein(s, t, costs=(1, 1, 1)):
        rows = len(s) + 1
        cols = len(t) + 1
        deletes, inserts, substitutes = costs

        dist = [[0 for x in range(cols)] for x in range(rows)]

        for row in range(1, rows):
            dist[row][0] = row * deletes

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
                                     dist[row - 1][col - 1] + cost)
        return dist[rows - 1][cols - 1]

    cur_node = root_node
    actions = []
    total_reward = 0.0

    """
    Ciclo di Test:
    -> Il ciclo si ripete finché ci sono azioni possibili dal nodo corrente.
    -> Viene ottenuta l'azione con il massimo valore Q dal nodo corrente.
    -> Viene ottenuto il nodo successivo corrispondente all'azione scelta.
    -> Viene calcolata la ricompensa associata all'azione corrente e aggiunta a total_reward.
    -> Viene aggiunto l'indice dell'azione alla lista actions.
    """
    print("Test Q-learning")
    print("---------------------")
    while True:
        a = cur_node.get_max_action()
        print("Ottenuta l'azione con il massimo valore Q dal nodo corrente ",a)
        if a is None:
            break
        actions.append(a)
        print("Insieme delle azioni: ",actions)

        aux = cur_node.get_child_original(a)
        print("ottiene il nodo figlio del nodo corrente (cur_node) ",aux,"associato all'azione a: ",a)
        """
        In un contesto di algoritmo Q-learning, questo passo corrisponde a selezionare l'azione migliore dal nodo corrente e muoversi verso il nodo successivo in base a quella azione.
            -> Se il figlio del nodo corrente non è vuoto, allora assegnamo all'attuale nodo corrente aux, ossia il figlio.
        """
        if aux is None:
            break
        cur_node = aux
        """
        In altre parole, questa riga di codice sembra attribuire una piccola ricompensa di 0.1 quando il nodo corrente è nel contesto della radice dell'albero e altrimenti utilizza la sovrapposizione pairwise del nodo corrente moltiplicata per factor come ricompensa.
        """
        reward = 0.1 if cur_node.parent_node == root_node else cur_node.pairwise_overlap * factor
        reward += 1.0 if cur_node.is_leaf() else 0.0
        total_reward += reward


    """
    Calcolo della Distanza di Levenshtein:
    -> Viene calcolata la distanza di Levenshtein tra la sequenza ottenuta e il genoma di riferimento.
    -> La distanza viene restituita.
    """
    dist = None
    print("Calcolo della distanza di Levenshtein ")
    print("-----------")

    print("Effettuiamo la stampa del Genoma: ",genome)
    if genome is not None:
        dist = levenshtein(cur_node.get_consensus(), genome)
        print("consenso sul nodo corrente: ", cur_node.get_consensus())

        """
        Questa funzione get_consensus è progettata per risalire dalla foglia (il nodo corrente) al nodo radice e costruire il consenso lungo il percorso.
        In sintesi, questa funzione attraversa il percorso dalla foglia alla radice, calcolando la sovrapposizione tra il read del nodo corrente e il consenso parziale e quindi estendendo il consenso. 
        Restituisce il consenso completo lungo il percorso.
        """
        print("Effettuiamo la stampa della distanza di  --- Levenshtein", dist)
        return actions, total_reward, dist


"""
Main:
-> La funzione main contiene il codice principale che viene eseguito quando il modulo viene eseguito come script.
-> Inizializza un seme casuale e lo stampa.
-> Definisce un dataset che contiene letture (reads) di diversi tipi.
"""
if __name__ == "__main__":
    seed = random.randrange(sys.maxsize)
    random.seed(seed)
    print(seed, file=sys.stderr)

    dataset = {}
    genome25 = "TACTAGCAATACGCTTGCGTTCGGT"
    genome50 = "CCTAACCATTTTAACAGCAACATAACAGGCTAAGAGGGGCCGGACACCCA"

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

    genome, reads = dataset[int(sys.argv[2])]
    qlearning(reads, int(sys.argv[1]), genome)
