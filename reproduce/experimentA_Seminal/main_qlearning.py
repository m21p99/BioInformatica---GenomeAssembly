from overlap import OverlapResolver
from nodeoriginal import Node
from itertools import permutations 
import random
import sys
import numpy as np


def qlearning(reads, episodes, genome = None, test_each_episode = False):
    ovr = OverlapResolver(reads)
    root = Node.createRootNode(ovr)
    actions_taken, leafs_reached = 0, 0
    epsilon = 1.0
    gamma = 0.9
    alpha = 0.8
    epsilon_decay = 1.0 / episodes
    factor = 1.0 / (len(reads) * max([len(read) for read in reads]))
    
    for episode in range(episodes):
        cur_node = root
        total_reward = 0.0
        actions_train = []
        while True:
            candidates = cur_node.get_outputs()
            if len(candidates) == 0:
                leafs_reached += 1
                break
            action = random.sample(candidates, 1)[0]

            rand = random.random()
            if rand > epsilon: #and cur_node != root:
                a = cur_node.get_max_action()
                if a is not None:
                    action = a
            next_node = cur_node.get_child_original(action)
            if next_node is None:
                break
            reward = 0.1 if cur_node == root else next_node.pairwise_overlap * factor
            reward += 1.0 if next_node.is_leaf() else 0.0
            total_reward += reward
            cur_node.update_q(action, reward + gamma * next_node.get_max_qvalue(), alpha)
            actions_taken += 1
            actions_train.append(action)
            cur_node = next_node
        if test_each_episode or episode + 1 == episodes:
            test = test_qlearning(root, factor, genome)
        else:
            test = (None, 0.0, None)
        print("ep.:", episode+1, "max_acc:", ovr.max_acc, "train_rw:", "%.5f" % total_reward, "test_rw:", "%.5f" % test[1], "test:", test[0], "train", actions_train, "dist:", test[2])
        epsilon -= epsilon_decay

    print("actions_taken", actions_taken)
    print("leafs_reached", leafs_reached)

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
        aux = cur_node.get_child_original(a)
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
    print(seed, file=sys.stderr)
    
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
    


