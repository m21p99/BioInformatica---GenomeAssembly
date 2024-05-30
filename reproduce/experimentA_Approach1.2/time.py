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
        self.buffer = {}

# calculates each instance of a recurrent function to calculate sequences overlaps
    def _getOverlapValue(self, i, j, matrix, s1, s2, match, mismatch, gap):
        score = match if s1[i-1] == s2[j-1] else mismatch
        aux = max(
            matrix[i-1][j-1] + score,
            matrix[i-1][j] + gap,
            matrix[i][j-1] + gap,
            0
        )
        return aux

    def compute_overlap(self, left_read, right_read):
        for i in range(len(left_read)):
            l = left_read[i:]
            size = len(l)
            r = right_read[:size]
            if l == r:
                return l, size
        return "", 0

    def _getOverlap(self, s1, s2, match, mismatch, gap):
        l = len(s1)+1
        c = len(s2)+1
        matrix = np.array([0.0 for _ in range(l * c)]).reshape(l, c)
        for i in range(1, l):
            for j in range(1, c):
                matrix[i][j] = self._getOverlapValue(i, j, matrix, s1, s2, match, mismatch, gap)
        return np.max(matrix)

    def _getSuffixPrefixOverlap(self, left, right):
        return self.compute_overlap(left, right)[1]

    def _findOverlap(self, reads, id1, id2, match = 1.0, mismatch = -0.33, gap = -1.33):
        left = reads[id1]
        right = reads[id2]
        if left in self.buffer and right in self.buffer[left]:
            overlap = self.buffer[left][right]
        else:
            overlap = self._getSuffixPrefixOverlap(left, right)
            if left not in self.buffer:
                self.buffer[left] = {}
            self.buffer[left][right] = overlap
        return overlap

    # calculates the score of overlap between two strings
    def _findOverlap2(self, reads, id1, id2, match = 1.0, mismatch = -0.33, gap = -1.33):
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

    def _crossover(self, cromossome1, cromossome2):
        if type(cromossome1) == list:
            cromossome1 = np.array(cromossome1)
        if type(cromossome2) == list:
            cromossome2 = np.array(cromossome2)
        genes = np.random.choice(len(cromossome1), size=2, replace=False)
        genes.sort()

        aux1 = cromossome1[genes[0]:genes[1]+1]
        aux2 = cromossome2[genes[0]:genes[1]+1]

        diff2 = cromossome2[~np.in1d(cromossome2, aux1, assume_unique=True)]
        diff1 = cromossome1[~np.in1d(cromossome1, aux2, assume_unique=True)]

        child1 = np.append(aux1, diff2).copy()
        child2 = np.append(aux2, diff1).copy()

        return child1, child2

    def _mutation(self, cromossome):
        if type(cromossome) == list:
            cromossome = np.array(cromossome)
        genes = np.random.choice(len(cromossome), size=2, replace=False)
        
        cromossome[genes[[0,1]]] = cromossome[genes[[1,0]]]
        return cromossome

    def _fitness(self, cromossome, reads):
        score = 0
        for i in range(1, len(cromossome)):
            score += self._findOverlap(reads, cromossome[i-1], cromossome[i])
        return score

    def _evaluatePopulation(self, population, reads):
        scores = np.zeros(len(population))
        for i in range(len(population)):
            scores[i] = self._fitness(population[i], reads)
        return scores

    def _ring(self, pop_fitness, ring_size):
        fighters = np.random.choice(len(pop_fitness), size=self.ring_size, replace=False)
        fit = pop_fitness[fighters]
        winner = fit.argmax()
        return fighters[winner]

    def run_ga(self, env, memory, reads, generations):
        pop_size = len(memory)
        cromo_size = len(reads)

        population = np.array(memory)
        pop_fitness = self._evaluatePopulation(population, reads)
        best_ind = population[pop_fitness.argmax()]
        best_fit = self._fitness(best_ind, reads)

        fitness_evolution = np.zeros(generations)
        for generation in range(generations):
            #print('Generation {}'.format(generation+1))
            # Tournament selection
            selected = []
            for i in range(pop_size):
                winner = self._ring(pop_fitness, self.ring_size)
                selected.append(population[winner].copy())

            # Crossover
            for i in range(0, pop_size, 2):
                if np.random.rand() < self.crossover_prob:
                    population[i], population[i+1] = self._crossover(selected[i], selected[i+1])
                else:
                    population[i], population[i+1] = selected[i].copy(), selected[i+1].copy()

            # Mutation
            for i in range(pop_size):
                if np.random.rand() < self.mutation_prob:
                    population[i] = self._mutation(population[i])

            pop_fitness = self._evaluatePopulation(population, reads)
            # Elitism
            fitness_evolution[generation] = pop_fitness.max()
            if fitness_evolution[generation] < best_fit:
                population[0] = best_ind.copy()
                fitness_evolution[generation] = best_fit
            else:
                best_ind = population[pop_fitness.argmax()].copy()
                best_fit = pop_fitness.max()

        return best_ind

def printTree(cur, target, verbose, offset = ""):
    if verbose:
        tag = "*" if cur == target else ""
        print(offset + "|" + cur.get_read_content() + "(" + str(cur.acc_overlap) + ")" + tag)
    fully = True
    for _, v in cur.children.items():
        fully = fully and printTree(v, target, verbose, offset + "\t")
    return cur.is_fully_explored() and fully

def manual(reads, verbose = True):
    ovr = OverlapResolver(reads)
    root = Node.createRootNode(ovr)
    cur = root
    while True:
        printTree(root, cur, verbose)
        code = input()
        if cur.is_leaf():
            cur = root
        else:
            aux = cur.get_child(int(code))
            if aux is not None:
                cur = aux
            elif cur.parent_node is None:
                cur = root

def gt():
    reads = ['ACATTAGG', 'TTAGGCCCTT', 'GCCCTTA', 'CCTTACA']
    ovr = OverlapResolver(reads)
    for p in permutations([0,1,2,3]):
        ov = None
        for r in p:
            if ov is None:
                ov = 0
            else:
                aux = ovr.get_overlap_by_read_ids(last,r)
                if aux == 0:
                    ov = -1
                    break
                ov += aux
            last = r
        if ov > 0:
            print(p, ov)

def auto(reads, step_by_step = False, verbose = False, auto_interrupt = False):
    ovr = OverlapResolver(reads)
    root = Node.createRootNode(ovr)
    cur = root
    actions_taken, leafs_reached = 0, 0
    while True:
        codes = cur.candidate_children[:]
        codes.extend(list(cur.children.keys()))
        if cur.is_leaf() or len(codes) == 0:
            leafs_reached += 1
            if verbose or auto_interrupt:
                if printTree(root, cur, verbose) and auto_interrupt:
                    break
            if step_by_step:
                input()
            cur = root
        else:
            actions_taken += 1
            code = random.sample(codes, 1)[0]
            aux = cur.get_child(code)
            if aux is not None:
                cur = aux
            elif cur.parent_node is None:
                cur = root

    #print("actions_taken", actions_taken)
    #print("leafs_reached", leafs_reached)

def _get_maximum_shifted_path(root_node, leaf_node):
    path = []
    cur_node = leaf_node
    while cur_node.parent_node is not None:
        path.insert(0, cur_node.read_id)
        cur_node = cur_node.parent_node
    cur_node = root_node.get_child(leaf_node.read_id)
    if cur_node is None:
        return None
    cur_node = cur_node.get_child(path[0])
    if cur_node is None:
        return None

    i = len(path) - 1
    max_value = leaf_node.acc_overlap
    max_i = 0
    j = 1
    while i > 0:
        while True:
            cur_node = cur_node.get_child(path[j])
            j += 1
            if j >= len(path):
                j = 0
            if cur_node is None or cur_node.is_leaf():
                if cur_node is not None and cur_node.acc_overlap >= max_value:
                    max_value = cur_node.acc_overlap
                    max_i = i
                break
        i -= 1
        j = i
        cur_node = root_node
    if max_i > 0 and max_value == root_node.overlap_resolver.max_acc:
        return [path[(i + max_i) % len(path) if i + max_i >= len(path) else i + max_i] for i in range(len(path))]
    return None

def qlearning(reads, episodes, genome = None, test_each_episode = False):
    generations = 100
    num_ind = 20
    ovr = OverlapResolver(reads)
    root = Node.createRootNode(ovr)
    actions_taken, leafs_reached = 0, 0
    epsilon = 1.0
    gamma = 0.9
    alpha = 0.8
    epsilon_decay = 1.0 / episodes
    factor = 1.0 / (len(reads) * max([len(read) for read in reads]))
    forced_path = None
    
    #debug = [4, 15, 13, 2, 5, 3, 18, 9, 7, 16, 11, 19, 0, 8, 17, 12, 14, 1, 6, 10]
    #cur_node = root
    #for read in debug:
    #    cur_node = cur_node.get_child(read)
    #forced_path = _get_maximum_shifted_path(root, cur_node)
    #print(forced_path)
    ga = GA()
    memory = []
    ind_evolved = []
    pointer = -1
    f_episodes, ga_episodes, re_episodes = set(), set(), set()


    for episode in range(episodes):
        cur_node = root
        total_reward = 0.0
        actions_train = []
        ind = []
        debug4 = []
        debug3 = []
        debug2 = []
        while True:
            if len(ind_evolved) > 0:
                if pointer >= len(ind_evolved[0]):
                    del ind_evolved[0]
                    pointer = 0

            candidates = cur_node.get_outputs()
            if len(candidates) == 0:
                leafs_reached += 1
                if forced_path is None:
                    forced_path = _get_maximum_shifted_path(root, cur_node)
                else:
                    forced_path = None
                break
            debug4.append(len(memory))
            debug2.append(len(ind_evolved))
            debug3.append(len(ind))
            if len(ind_evolved) == 0:
                if forced_path is None or len(forced_path) == 0:
                    action = random.sample(candidates, 1)[0]

                    rand = random.random()
                    if rand > epsilon: #and cur_node != root:
                        a = cur_node.get_max_action()
                        if a is not None:
                            action = a
                    re_episodes.add(episode)
                else:
                    f_episodes.add(episode)
                    action = forced_path[0]
                    del forced_path[0]
                ind.append(int(action))
            else:
                ga_episodes.add(episode)
                action = ind_evolved[0][pointer]
                pointer += 1
            next_node = cur_node.get_child(action)
            if next_node is None:
                if len(ind_evolved) > 0:
                    del ind_evolved[0]
                    pointer = 0                    
                break
            reward = 0.1 if cur_node == root else next_node.pairwise_overlap * factor
            reward += 1.0 if next_node.is_leaf() else 0.0
            total_reward += reward
            cur_node.update_q(action, reward + gamma * next_node.get_max_qvalue(), alpha)
            actions_taken += 1
            actions_train.append(action)
            cur_node = next_node
        if len(ind_evolved) == 0:
            if len(ind) > 0 and len(ind) < len(reads):
                remaining = list(set(range(len(reads))) - set(ind))
                random.shuffle(remaining)
                ind.extend(remaining)
            if len(ind) == len(reads) and len(set(ind)) == len(reads):
                memory.append(ind)
                del memory[0]
            elif len(ind) > 0: 
                print(reads)
                print(episode in ga_episodes)
                print(episode in f_episodes)
                sys.exit(1)
        debug = len(memory)
        if len(memory) == num_ind:
            ind_evolved = [ga.run_ga(None, memory, reads, generations)]
            pointer = 0
            memory = []
        if test_each_episode or episode + 1 == episodes:
            test = test_qlearning(root, factor, genome)
        else:
            test = (None, 0.0, None)
        #print("debug4", debug4, "debug3", debug3, "debug2", debug2, "mem", debug, "ind", None if len(ind_evolved) == 0 else len(ind_evolved[0]), "ep.:", episode+1, "max_acc:", ovr.max_acc, "train_rw:", "%.5f" % total_reward, "test_rw:", "%.5f" % test[1], "test:", test[0], "train", actions_train, "dist:", test[2])
        epsilon -= epsilon_decay

    #print("actions_taken", actions_taken)
    #print("leafs_reached", leafs_reached)
    aux = list(f_episodes)
    aux.sort()
    #print("forced_episodes", len(f_episodes), aux)
    aux = list(ga_episodes)
    aux.sort()
    #print("ga_episodes", len(ga_episodes), aux)
    aux = list(re_episodes)
    aux.sort()
    #print("re_episodes", len(re_episodes), aux)
    #print("total", len(re_episodes.union(ga_episodes).union(f_episodes)))

def test_qlearning(root_node, factor, genome):
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
        rows = len(s)+1
        cols = len(t)+1
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
                if s[row-1] == t[col-1]:
                    cost = 0
                else:
                    cost = substitutes
                dist[row][col] = min(dist[row-1][col] + deletes,
                                     dist[row][col-1] + inserts,
                                     dist[row-1][col-1] + cost) # substitution
        return dist[rows-1][cols-1]    
    cur_node = root_node
    actions = []
    total_reward = 0.0
    while True:
        a = cur_node.get_max_action()
        if a is None:
            break
        actions.append(a)
        aux = cur_node.get_child(a)
        if aux is None:
            break
        cur_node = aux
        reward = 0.1 if cur_node.parent_node == root_node else cur_node.pairwise_overlap * factor
        reward += 1.0 if cur_node.is_leaf() else 0.0
        total_reward += reward
    dist = None
    if genome is not None:
        dist = levenshtein(cur_node.get_consensus(), genome)
    return actions, total_reward, dist

if __name__ == "__main__":
    seed = random.randrange(sys.maxsize)
    random.seed(seed)
    #print(seed, file=sys.stderr)
    
    dataset = {}
    genome25 = "TACTAGCAATACGCTTGCGTTCGGT"
    genome50 = "CCTAACCATTTTAACAGCAACATAACAGGCTAAGAGGGGCCGGACACCCA"
    # reads = ['ACATTAGG', 'TTAGGCCCTT', 'GCCCTTA', 'CCTTACA']
    #reads = ['GCTTGCGT','GCGTTCGG','CTAGCAAT','GCAATACG','CTAGCAAT','TAGCAATA','CGTTCGGT','TACGCTTG','CTTGCGTT','TACGCTTG','TACTAGCA','ACGCTTGC','TGCGTTCG','CTAGCAAT','TGCGTTCG','CTAGCAAT','ACGCTTGC','TTGCGTTC','ATACGCTT','CGCTTGCG']
    #reads = ['CAACATAA','TTTAACAG','AGGCTAAG','GGGCCGGA','GGCTAAGA','AACATAAC','TAACAGCA','ACATAACA','AGGCTAAG','CAACATAA','GGGCCGGA','CCATTTTA','AACATAAC','CCATTTTA','CCTAACCA','GAGGGGCC','GACACCCA','TAACAGGC','TAACAGCA','GCCGGACA']
    #reads = ['GGGGCCGG','ACAGCAAC','AGAGGGGC','CCTAACCA','CCATTTTA','AAGAGGGG','TTAACAGC','AGGCTAAG','AGGGGCCG','GGGCCGGA','CCGGACAC','TAAGAGGG','CATAACAG','TTAACAGC','GAGGGGCC','TTTAACAG','CAGCAACA','GGGGCCGG','GGCTAAGA','CGGACACC','CCTAACCA','AACCATTT','AGGGGCCG','GACACCCA','AGCAACAT','CCGGACAC','ACCATTTT','TAAGAGGG','GGCCGGAC','GCTAAGAG']
    #gt()
    #manual(reads)
    #auto(reads)
    reads_25_10_8 = ['CGTTCGGT','TTGCGTTC','CTTGCGTT','ACGCTTGC','ATACGCTT','AATACGCT','AGCAATAC','CTAGCAAT','ACTAGCAA','TACTAGCA']
    reads_25_10_10 = ['TGCGTTCGGT','CTAGCAATAC','ACTAGCAATA','CAATACGCTT','GCTTGCGTTC','CTTGCGTTCG','GCTTGCGTTC','ACTAGCAATA','TACTAGCAAT','CTAGCAATAC']
    reads_25_10_15 = ['CAATACGCTTGCGTT','AGCAATACGCTTGCG','GCAATACGCTTGCGT','TACTAGCAATACGCT','ACGCTTGCGTTCGGT','ATACGCTTGCGTTCG','TAGCAATACGCTTGC','CTAGCAATACGCTTG','AATACGCTTGCGTTC','AGCAATACGCTTGCG']
    reads_50_10_8 = ['GGGGCCGG','GGACACCC','ATAACAGG','GACACCCA','AAGAGGGG','TTAACAGC','AACATAAC','ATTTTAAC','CCTAACCA','ACAGGCTA']
    reads_50_10_10 = ['CGGACACCCA','CCTAACCATT','TAACCATTTT','CTAAGAGGGG','ACATAACAGG','GGGCCGGACA','GAGGGGCCGG','TTAACAGCAA','CATAACAGGC','ACAGCAACAT']
    reads_50_10_15 = ['CAACATAACAGGCTA','CCTAACCATTTTAAC','ATTTTAACAGCAACA','GGGGCCGGACACCCA','CCATTTTAACAGCAA','TAACCATTTTAACAG','TAAGAGGGGCCGGAC','AACAGCAACATAACA','ACAGGCTAAGAGGGG','TTTTAACAGCAACAT']
    reads_25_20_8 = ['GCTTGCGT','GCGTTCGG','CTAGCAAT','GCAATACG','CTAGCAAT','TAGCAATA','CGTTCGGT','TACGCTTG','CTTGCGTT','TACGCTTG','TACTAGCA','ACGCTTGC','TGCGTTCG','CTAGCAAT','TGCGTTCG','CTAGCAAT','ACGCTTGC','TTGCGTTC','ATACGCTT','CGCTTGCG']
    reads_25_20_10 = ['CAATACGCTT','GCTTGCGTTC','AATACGCTTG','CAATACGCTT','TACTAGCAAT','ATACGCTTGC','TGCGTTCGGT','TAGCAATACG','AGCAATACGC','TACGCTTGCG','TACGCTTGCG','CTAGCAATAC','CAATACGCTT','ACGCTTGCGT','ATACGCTTGC','TAGCAATACG','ACGCTTGCGT','GCAATACGCT','CAATACGCTT','AGCAATACGC']
    reads_25_20_15 = ['ACGCTTGCGTTCGGT','AATACGCTTGCGTTC','ACGCTTGCGTTCGGT','AGCAATACGCTTGCG','TAGCAATACGCTTGC','AATACGCTTGCGTTC','ATACGCTTGCGTTCG','CTAGCAATACGCTTG','ACGCTTGCGTTCGGT','ACGCTTGCGTTCGGT','ATACGCTTGCGTTCG','ACGCTTGCGTTCGGT','ACGCTTGCGTTCGGT','TACTAGCAATACGCT','CTAGCAATACGCTTG','ACGCTTGCGTTCGGT','GCAATACGCTTGCGT','TACGCTTGCGTTCGG','CAATACGCTTGCGTT','TACGCTTGCGTTCGG']
    reads_50_20_8 = ['CAACATAA','TTTAACAG','AGGCTAAG','GGGCCGGA','GGCTAAGA','AACATAAC','TAACAGCA','ACATAACA','AGGCTAAG','CAACATAA','GGGCCGGA','CCATTTTA','AACATAAC','CCATTTTA','CCTAACCA','GAGGGGCC','GACACCCA','TAACAGGC','TAACAGCA','GCCGGACA']
    reads_50_20_10 = ['GAGGGGCCGG','TAACCATTTT','TAACAGGCTA','TTTTAACAGC','CATTTTAACA','ACCATTTTAA','GGCTAAGAGG','CAGCAACATA','AGAGGGGCCG','TAACCATTTT','CCATTTTAAC','AACAGGCTAA','CTAACCATTT','GGGCCGGACA','CCTAACCATT','CCATTTTAAC','TTTTAACAGC','CAACATAACA','ACATAACAGG','CGGACACCCA']
    reads_50_20_15 = ['ATAACAGGCTAAGAG','TTTTAACAGCAACAT','CAGGCTAAGAGGGGC','GGCTAAGAGGGGCCG','ATAACAGGCTAAGAG','CATAACAGGCTAAGA','GCTAAGAGGGGCCGG','TTTTAACAGCAACAT','CAACATAACAGGCTA','ACAGGCTAAGAGGGG','GGCTAAGAGGGGCCG','TAAGAGGGGCCGGAC','AACAGGCTAAGAGGG','TAACAGGCTAAGAGG','TAACAGCAACATAAC','CATAACAGGCTAAGA','CCTAACCATTTTAAC','GGGGCCGGACACCCA','ACAGCAACATAACAG','TTTTAACAGCAACAT']
    reads_25_30_8 = ['TACTAGCA','AATACGCT','CGTTCGGT','CGCTTGCG','AGCAATAC','TGCGTTCG','TAGCAATA','CTTGCGTT','TAGCAATA','CTAGCAAT','GCGTTCGG','TTGCGTTC','TTGCGTTC','TGCGTTCG','GCGTTCGG','TACGCTTG','CAATACGC','ACGCTTGC','GCTTGCGT','CGCTTGCG','ATACGCTT','CGTTCGGT','CGCTTGCG','GCAATACG','GCTTGCGT','ACGCTTGC','CTTGCGTT','TTGCGTTC','GCTTGCGT','TACTAGCA'] 
    reads_25_30_10 = ['TAGCAATACG','CGCTTGCGTT','TGCGTTCGGT','ACGCTTGCGT','CTAGCAATAC','ACTAGCAATA','TTGCGTTCGG','TGCGTTCGGT','ATACGCTTGC','CGCTTGCGTT','CTAGCAATAC','CTAGCAATAC','TAGCAATACG','GCAATACGCT','TACGCTTGCG','AGCAATACGC','GCTTGCGTTC','ACGCTTGCGT','ATACGCTTGC','CAATACGCTT','AATACGCTTG','TAGCAATACG','GCAATACGCT','TACGCTTGCG','AGCAATACGC','TAGCAATACG','CGCTTGCGTT','TAGCAATACG','TACTAGCAAT','GCTTGCGTTC']
    reads_25_30_15 = ['ACTAGCAATACGCTT','ACTAGCAATACGCTT','AATACGCTTGCGTTC','ACTAGCAATACGCTT','ACGCTTGCGTTCGGT','TACGCTTGCGTTCGG','AATACGCTTGCGTTC','AATACGCTTGCGTTC','CTAGCAATACGCTTG','TACTAGCAATACGCT','CTAGCAATACGCTTG','CAATACGCTTGCGTT','TACGCTTGCGTTCGG','AATACGCTTGCGTTC','AATACGCTTGCGTTC','CAATACGCTTGCGTT','TACGCTTGCGTTCGG','CAATACGCTTGCGTT','AATACGCTTGCGTTC','ACTAGCAATACGCTT','ACTAGCAATACGCTT','ACTAGCAATACGCTT','AGCAATACGCTTGCG','CTAGCAATACGCTTG','ATACGCTTGCGTTCG','GCAATACGCTTGCGT','ATACGCTTGCGTTCG','ACTAGCAATACGCTT','ATACGCTTGCGTTCG','TAGCAATACGCTTGC']
    reads_50_30_8 = ['GGGGCCGG','ACAGCAAC','AGAGGGGC','CCTAACCA','CCATTTTA','AAGAGGGG','TTAACAGC','AGGCTAAG','AGGGGCCG','GGGCCGGA','CCGGACAC','TAAGAGGG','CATAACAG','TTAACAGC','GAGGGGCC','TTTAACAG','CAGCAACA','GGGGCCGG','GGCTAAGA','CGGACACC','CCTAACCA','AACCATTT','AGGGGCCG','GACACCCA','AGCAACAT','CCGGACAC','ACCATTTT','TAAGAGGG','GGCCGGAC','GCTAAGAG'] 
    reads_50_30_10 = ['ACATAACAGG','CTAACCATTT','CAGGCTAAGA','AGCAACATAA','CTAAGAGGGG','AGAGGGGCCG','CAGGCTAAGA','CGGACACCCA','AACAGGCTAA','CAGCAACATA','GGCCGGACAC','GGCCGGACAC','TAACAGCAAC','CCTAACCATT','GGGGCCGGAC','CAGGCTAAGA','GGCTAAGAGG','TAAGAGGGGC','AACATAACAG','CAGCAACATA','TAACAGGCTA','TTTTAACAGC','ACCATTTTAA','AACATAACAG','AGAGGGGCCG','GCAACATAAC','TAACAGGCTA','GGCTAAGAGG','TAACCATTTT','CAGGCTAAGA']
    reads_50_30_15 = ['CTAAGAGGGGCCGGA','AACAGCAACATAACA','GGGGCCGGACACCCA','AGCAACATAACAGGC','AACAGGCTAAGAGGG','ATAACAGGCTAAGAG','TAACAGCAACATAAC','GAGGGGCCGGACACC','CTAAGAGGGGCCGGA','ATAACAGGCTAAGAG','TTTTAACAGCAACAT','CATTTTAACAGCAAC','CCTAACCATTTTAAC','TTTTAACAGCAACAT','GCAACATAACAGGCT','GAGGGGCCGGACACC','GGCTAAGAGGGGCCG','ATTTTAACAGCAACA','ACAGGCTAAGAGGGG','TTTAACAGCAACATA','CAACATAACAGGCTA','CAACATAACAGGCTA','TTTTAACAGCAACAT','AACATAACAGGCTAA','CATTTTAACAGCAAC','TAACCATTTTAACAG','AACATAACAGGCTAA','CTAAGAGGGGCCGGA','AGGCTAAGAGGGGCC','CTAAGAGGGGCCGGA']
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
    


