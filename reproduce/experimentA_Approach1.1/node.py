class Node:
    def __init__(self, overlap_resolver, parent_node=None, read_id=None, pairwise_overlap=0):
        # object responsible for computing overlaps between reads
        self.overlap_resolver = overlap_resolver
        # parent node of this node
        self.parent_node = parent_node
        # id of the last read used to reach this node
        self.read_id = read_id
        # overlap score between read_id and the previous read
        self.pairwise_overlap = pairwise_overlap
        # previously computed valid children
        self.children = {}
        # accumulated overlap score
        self.acc_overlap = 0 if self.is_root() else self.parent_node.acc_overlap + self.pairwise_overlap
        self._set_candidate_children()
        self.unused_reads = self.candidate_children[:]
        self.qtable = {}
        if self.is_leaf():
            self.overlap_resolver.max_acc = self.acc_overlap
        else:
            self.max_overlap = None

    def get_consensus(self):
        consensus = ""
        cur_node = self
        while not cur_node.is_root():
            cur_read = cur_node.get_read_content()
            if consensus == "":
                overlap_text = cur_read
            else:
                off = len(cur_read) - cur_node.overlap_resolver.compute_overlap(cur_read, consensus)[1]
                overlap_text = cur_read[:off]
            consensus = overlap_text + consensus
            cur_node = cur_node.parent_node
        return consensus

    def _set_candidate_children(self):
        if self.is_root():
            self.candidate_children = list(range(self.overlap_resolver.count_reads()))
        else:
            self.candidate_children = self.parent_node.unused_reads[:]
            self.candidate_children.remove(self.read_id)


"""
  In termini più generali, la funzione update_q sta implementando l'aggiornamento dei valori Q 
  in base a quanto l'azione corrente ha contribuito alla ricompensa complessiva e a quanto questa ricompensa differisce dalla stima attuale. 
  Questo è un passo fondamentale nell'apprendimento Q, poiché l'agente aggiorna dinamicamente le sue stime dei valori Q in base all'esperienza acquisita durante l'esplorazione dell'ambiente.
"""


def update_q(self, action, target_value, alpha):
    if action not in self.qtable:
        old = 0.0
    else:
        old = self.qtable[action]
        print("Eseguito metodo update_q ", " Azione presa: ", old)
        print("Tabella delle azioni: ", self.qtable[action])
    delta = target_value - old
    self.qtable[action] = old + alpha * delta


"""
Questo codice definisce una funzione _get_max_q in una classe. 
    In sintesi, questa funzione restituisce (l'azione e il valore Q) associato a quell'azione che ha il massimo valore Q all'interno del dizionario qtable della classe.
"""


def _get_max_q(self):
    max_action = None
    max_value = None
    for action in self.qtable.keys():
        if max_action is None or self.qtable[action] > max_value:
            max_value = self.qtable[action]
            max_action = action
    return max_action, max_value
"""
Alla fine, restituisce la coppia (max_action, max_value) che rappresenta l'azione con il massimo valore Q nel dizionario.
"""


def get_max_qvalue(self):
    max_value = self._get_max_q()[1]
    return 0.0 if max_value is None else max_value


def get_max_action(self):
    return self._get_max_q()[0]


def get_outputs(self):
    cand = self.candidate_children[:]
    cand.extend(list(self.children.keys()))
    return cand


def is_root(self):
    return self.parent_node is None


def is_leaf(self):
    return len(self.unused_reads) == 0


def get_child(self, read_id):
    if read_id not in self.unused_reads:
        self.is_needed()
        return None
    if read_id in self.children:
        if self.children[read_id].is_needed():
            return self.children[read_id]
        return None
    if read_id not in self.candidate_children:
        self.is_needed()
        return None
    self.candidate_children.remove(read_id)
    overlap = self._get_overlap_with(read_id)
    if overlap <= 0 and not self.is_root():
        self.is_needed()
        return None
    if len(self.unused_reads) == 1:
        if overlap + self.acc_overlap < self.overlap_resolver.max_acc:
            self.is_needed()
            return None
    self.children[read_id] = Node(self.overlap_resolver, self, read_id, overlap)
    return self.children[read_id]


def is_needed(self):
    if self.is_root() or len(self.candidate_children) > 0 or len(self.children) > 0:
        return True
    if self.is_leaf() and self.acc_overlap >= self.overlap_resolver.max_acc:
        return True
    del self.parent_node.children[self.read_id]
    self.parent_node.is_needed()
    self.parent_node = None
    return False


def _get_overlap_with(self, right_read_id):
    if self.is_root():
        return 0
    return self.overlap_resolver.get_overlap_by_read_ids(self.read_id, right_read_id)


def get_read_content(self):
    if self.is_root():
        return ""
    return self.overlap_resolver.get_read_content(self.read_id)


def is_fully_explored(self):
    return len(self.candidate_children) == 0


@staticmethod
def createRootNode(overlap_resolver):
    return Node(overlap_resolver)
