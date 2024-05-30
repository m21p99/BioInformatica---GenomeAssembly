import random
import numpy as np
import sys

class Environment_25_10_8:
    Reads = ['CGTTCGGT', 'TTGCGTTC', 'CTTGCGTT', 'ACGCTTGC', 'ATACGCTT', 'AATACGCT', 'AGCAATAC', 'CTAGCAAT', 'ACTAGCAA', 'TACTAGCA']
    SpecialStates = {
        1 : 7,
        2 : 1,
        3 : 4,
        9 : 2,
        10 : 8,
        11 : 5,        
        16 : 0,
        17 : 6,
        18 : 3,
        19 : 9
    }
    Factor = 1.0 / ((len(Reads) * max([len(read) for read in Reads]))+1)
    NumberOfStates = 20
    def __init__(self):
        self.last_read = None
        self.cur_state = None
        self.states = None
        
    def reset(self):
        self.last_read = None
        self.cur_state = 0
        self.states = dict(Environment_25_10_8.SpecialStates)
        return self.cur_state

    def __normalize(self, overlap):
        return overlap * Environment_25_10_8.Factor

    def __get_overlap(self, next_read):
        left_read = Environment_25_10_8.Reads[self.last_read]
        right_read = Environment_25_10_8.Reads[next_read]
        return self.__compute_overlap(left_read, right_read)[1]

    def __compute_overlap(self, left_read, right_read):
            for i in range(len(left_read)):
                l = left_read[i:]
                size = len(l)
                r = right_read[:size]
                if l == r:
                    return l, size
            return "", 0
        
    
    def step(self, action):
        # actions (0=LEFT,1=RIGHT,2=UP,3=DOWN)
        done = False
        if action == 0:
            next_state = Environment_25_10_8.NumberOfStates - 1 if self.cur_state == 0 else self.cur_state - 1
        elif action == 1:
            next_state = 0 if self.cur_state == Environment_25_10_8.NumberOfStates - 1 else self.cur_state + 1
        elif action == 2:
            next_state = self.cur_state - 4
            if next_state < 0:
                next_state += Environment_25_10_8.NumberOfStates
        else:
            next_state = self.cur_state + 4
            if next_state > Environment_25_10_8.NumberOfStates - 1:
                next_state -= Environment_25_10_8.NumberOfStates
        info = {}
        if next_state in self.states:
            cur_read = Environment_25_10_8.SpecialStates[next_state]
            info['read'] = cur_read
            cur_overlap = 1 if self.last_read is None else self.__get_overlap(cur_read)
            if cur_overlap == 0:
                reward = -self.__normalize(1)
                #done = True
            else:
                reward = self.__normalize(cur_overlap)
                if len(self.states) == 1:
                    reward += 1.0
                    done = True
            del self.states[next_state]
            self.last_read = cur_read
        else:
            reward = 0.0
        self.cur_state = next_state
        return next_state, reward, done, info


def train(episodes, max_epochs = 500):
    env = Environment_25_10_8()
    # init q-table
    q_table = np.zeros([Environment_25_10_8.NumberOfStates, 4])

    # Hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.3
    
    total_penalties, total_success, max_acc_reward, good = 0, 0, None, 0
    
    for i in range(1, episodes + 1):
        state = env.reset()
        epochs, penalties, rewards = 0, 0, 0.0
        done = False
        reads = []
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 3)  # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            next_state, reward, done, info = env.step(action) 
            if len(info) > 0:
                reads.append(info['read'])
            rewards += reward
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward < 0:
                penalties += 1

            state = next_state
            epochs += 1
            if epochs >= max_epochs:
                break
        total_penalties += penalties
        total_success += 0 if penalties > 0 or epochs >= max_epochs else 1
        if max_acc_reward is None or rewards > max_acc_reward:
            max_acc_reward = rewards
        
        if ",".join(str(x) for x in reads) == "9,8,7,6,5,4,3,2,1,0":
            print(reads)
            good += 1
        
        if i % 100 == 0:
            print(f"Episode: {i}")
            print(f"Penalties: {total_penalties} ({total_penalties / float(i) * 100}%)")
            print(f"Success: {total_success} ({total_success / float(i) * 100}%)")
            print(f"Good: {good}")
            print(f"Max. acc. reward: {max_acc_reward}")
            actions, reads, test_reward = test(env, q_table, 50, False)
            print("Test actions:", actions)
            print("Test reads:", reads)
            print("Test acc. reward:", test_reward)

    print("Training finished.\n")
    return env, q_table


def test(env, q_table, max_epochs = 500, verbose=False):
    state = env.reset()
    done = False
    total_reward = 0.0
    epochs = 0
    actions, reads = [], []
    while not done:
        action = np.argmax(q_table[state])
        actions.append(action)
        next_state, reward, done, info = env.step(action)
        if verbose:
            print("state:", state, "action:", action, "next_state:", next_state, "reward:", reward)
        if len(info) > 0:
            if verbose:
                print(info)
            reads.append(info['read'])
        state = next_state
        total_reward += reward
        epochs += 1
        if epochs >= max_epochs:
            break
    return actions, reads, total_reward

episodes = int(sys.argv[1])
env, q_table = train(episodes)
print(q_table)
test(env, q_table, 50, True)