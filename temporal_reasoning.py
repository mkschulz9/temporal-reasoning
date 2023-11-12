from collections import defaultdict

class TemporalReasoning:
    def __init__(self):
        self.state_probs = {}
        self.state_transition_probs = self._nested_dict_factory()
        self.appearance_probs = self._nested_dict_factory()
        self.observation_action_pairs = []
        self.file_names = ["state_weights.txt", "state_action_state_weights.txt", 
                      "state_observation_weights.txt", "observation_actions.txt"]
    
    # reads and parses inputs
    # input: none
    # output: none, inputs are stored in class variables
    def parse_inputs(self):
        for file_name in self.file_names:
            with open("./io/stage1/inputs/" + file_name, 'r') as file:
                next(file)
                if file_name == self.file_names[0]:
                    self._parse_normalize_state_weights(file)
                elif file_name == self.file_names[1]:
                    self._parse_state_action_state_weights(file)
                elif file_name == self.file_names[2]:
                    self._parse_state_observation_weights(file)
                else:
                    self._parse_observation_actions(file)
    
    # implements the Viterbi algorithm
    # input: none
    # output: most probable path, probability
    def run_viterbi_algo(self):
        prob_matrix = [{}] 
        path = {}

        # Step 1: Initialization
        for state in self.state_probs:
            prob_matrix[0][state] = self.state_probs[state] * self.appearance_probs[self.observation_action_pairs[0][0]].get(state, 0)
            path[state] = [state]

        # Step 2: Recursion
        for t in range(1, len(self.observation_action_pairs)-1):
            prob_matrix.append({})
            newpath = {}

            for current_state in self.state_probs:
                (prob, state) = max(
                    (prob_matrix[t-1][prev_state] *
                    self.state_transition_probs[prev_state][self.observation_action_pairs[t-1][1]].get(current_state, 0) *
                    self.appearance_probs[self.observation_action_pairs[t][0]].get(current_state, 0), prev_state)
                    for prev_state in self.state_probs
                )

                prob_matrix[t][current_state] = prob
                newpath[current_state] = path[state] + [current_state]

            path = newpath

        # Step 3: Termination
        n = len(self.observation_action_pairs) - 1
        (prob, max_state) = max((prob_matrix[n][state], state) for state in self.state_probs)

        # Step 4: Path Backtracking
        most_probable_path = path[max_state]

        return most_probable_path, prob

    # writes output to file
    # input: most probable path
    # output: none, output is printed in 'states.txt'
    def write_output(self, most_probable_path):
        with open("./io/stage1/output/states.txt", 'w') as file:
            file.write("states\n")
            file.write(str(len(most_probable_path)) + "\n")
            for state in most_probable_path:
                file.write(f'"{state}"\n')
    
    # parses and normalizes state weights, building state proability 'table'
    # input: state weights file
    # output: none, normalized data is stored in 'state_probs'
    def _parse_normalize_state_weights(self, file):
        total_weight = 0
        next(file)
        
        for line in file:
            parts = line.strip().split('"')
            state = parts[1]
            weight = float(parts[2].strip())
            
            self.state_probs[state] = weight
            total_weight += weight
        
        for state in self.state_probs:
            self.state_probs[state] /= total_weight
    
    # parses state action state weights
    # input: state action state weights file
    # output: none, data is stored in 'state_transition_probs'
    def _parse_state_action_state_weights(self, file):
        _, num_unique_states, _, default_weight = map(int, next(file).split())
        default_weight = float(default_weight)

        for line in file:
            parts = line.strip().split()
            state, action, next_state, weight = parts[0].strip('"'), parts[1].strip('"'), parts[2].strip('"'), float(parts[3])
            
            self.state_transition_probs[state][action][next_state] = weight

        self._normalize_transitions(num_unique_states, default_weight)
        
    # fills in missing transitions if necessary & normalizes state action state weights
    # input: number of unique states, default weight
    # output: none, data is stored in 'state_transition_probs'
    def _normalize_transitions(self, num_unique_states, default_weight):
        for state in self.state_transition_probs:
            for action in self.state_transition_probs[state]:
                if default_weight != 0:
                    num_current_transitions = len(self.state_transition_probs[state][action])
                    num_missing_transitions = num_unique_states - num_current_transitions

                    if num_missing_transitions > 0:
                        for missing_state in self._get_missing_states(state, action):
                            self.state_transition_probs[state][action][missing_state] = default_weight

                total_weight = sum(self.state_transition_probs[state][action].values())

                for next_state in self.state_transition_probs[state][action]:
                    weight = self.state_transition_probs[state][action][next_state]
                    self.state_transition_probs[state][action][next_state] = weight / total_weight
    
    # parses observation state weights
    # input: state observation weights file
    # output: none, data is stored in 'appearance_probs'
    def _parse_state_observation_weights(self, file):
        _, num_unique_states, _, default_weight = map(int, next(file).split())
        default_weight = float(default_weight)

        for line in file:
            parts = line.strip().split()
            state, observation, weight = parts[0].strip('"'), parts[1].strip('"'), float(parts[2])
            self.appearance_probs[observation][state] = weight

        self._normalize_state_observations(num_unique_states, default_weight)
    
    # fills in missing states from an observation if necessary & normalizes observation state weights
    # input: number of unique states, default weight
    # output: none, data is stored in 'appearance_probs'
    def _normalize_state_observations(self, num_unique_states, default_weight):
        for observation in self.appearance_probs:
            if default_weight != 0:
                    num_current_states = len(self.appearance_probs[observation])
                    num_missing_states = num_unique_states - num_current_states

                    if num_missing_states > 0:
                        for missing_state in self._get_missing_states(observation = observation):
                            self.appearance_probs[observation][missing_state] = default_weight

            total_weight = sum(self.appearance_probs[observation].values())

            for state in self.appearance_probs[observation]:
                weight = self.appearance_probs[observation][state]
                self.appearance_probs[observation][state] = weight / total_weight
    
    # parses observation action pairs
    # input: observation action pairs file
    # output: none, data is stored in 'observation_action_pairs'
    def _parse_observation_actions(self, file):
        num_pairs = int(next(file))
        
        for _ in range(num_pairs - 1):
            line = next(file).strip()
            parts = line.split()
            observation, action = parts[0].strip('"'), parts[1].strip('"')
            
            self.observation_action_pairs.append((observation, action))
            
        last_line = next(file).strip().strip('"')
        if " " in last_line:
            observation, action = last_line.split()
            self.observation_action_pairs.append((observation, action))
            
        for val in self.observation_action_pairs:
            print(val)
            
    # HELPER FUNCTIONS:

    # creates nested dictionaries
    # input: none
    # output: nested dictionary
    def _nested_dict_factory(self): 
        return defaultdict(self._nested_dict_factory)

    # finds and returns missing next states
    # input: optionally state, action, or observation
    # output: missing states
    def _get_missing_states(self, state = None, action = None, observation = None):
        all_states = set(self.state_probs.keys())
        
        existing_next_states = set(
            self.state_transition_probs[state][action].keys() if not observation 
            else self.appearance_probs[observation].keys()
        )
            
        return all_states - existing_next_states
        
    # TESTING FUNCTIONS:
    
    '''
        # print content of state_transition_probs 
        count_total = 0
        for state in self.state_transition_probs:
            count_transitions = 0
            for action in self.state_transition_probs[state]:
                probability = 0
                for next_state in self.state_transition_probs[state][action]:
                    count_transitions += 1
                    count_total += 1
                    probability += self.state_transition_probs[state][action][next_state]
                    #print(state, action, next_state, self.state_transition_probs[state][action][next_state])
                print(f"Total probability for action {action}: {probability}")
            print(f"Number of transitions for state {state}: {count_transitions}\n")
        print(f"Total number of transitions: {count_total}. Expected number of transitions: {(num_unique_states ** 2) * num_unique_actions}\n")
        exit()
        
        # print content of appearance_probs 
        count_total = 0
        for observation in self.appearance_probs:
            count_states = 0
            probability = 0
            for state in self.appearance_probs[observation]:
                count_states += 1
                count_total += 1
                probability += self.appearance_probs[observation][state]
                #print(observation, state, self.appearance_probs[observation][state])
            print(f"Total probability for observation {observation}: {probability}")
            print(f"Number of states for observation {observation}: {count_states}\n")
        print(f"Total number of elems in appearance probability table: {count_total}. Expected number: {num_unique_states * num_unique_observaitons}\n")
        exit()
    '''