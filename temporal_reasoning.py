from collections import defaultdict

class TemporalReasoning:
    def __init__(self):
        self.state_probs = {}
        self.state_transition_probs = self._nested_dict_factory()
        self.appearance_probs = self._nested_dict_factory()
        self.observation_action_pairs = [] # format [(observation, action), ..., observation, [number of 'pairs']]
        self.file_names = ["state_weights.txt", "state_action_state_weights.txt", 
                      "state_observation_weights.txt", "observation_actions.txt"]
    
    # reads and parses inputs
    # input: none
    # output: none, inputs are stored in class variables
    def parse_inputs(self):
        for file_name in self.file_names:
            with open("./io/stage2/inputs/" + file_name, 'r') as file:
                next(file)
                if file_name == self.file_names[0]:
                    self._parse_normalize_state_weights(file)
                elif file_name == self.file_names[1]:
                    self._parse_state_action_state_weights(file)
                elif file_name == self.file_names[2]:
                    self._parse_state_observation_weights(file)
                else:
                    self._parse_observation_actions(file)
    
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
    
    def _parse_observation_actions(self, file):
        num_pairs = int(next(file))
        
        for _ in range(num_pairs - 1):
            line = next(file).strip()
            parts = line.split()
            observation, action = parts[0].strip('"'), parts[1].strip('"')
            self.observation_action_pairs.append((observation, action))
        
        self.observation_action_pairs.append(next(file).strip().strip('"'))
        self.observation_action_pairs.append(num_pairs)
        
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

    # creates nested dictionaries
    # input: none
    # output: nested dictionary
    def _nested_dict_factory(self): 
        return defaultdict(self._nested_dict_factory)
        
    '''
        # print content of state_transition_probs 
        for state in self.state_transition_probs:
            count = 0
            for action in self.state_transition_probs[state]:
                probability = 0
                for next_state in self.state_transition_probs[state][action]:
                    count += 1
                    probability += self.state_transition_probs[state][action][next_state]
                    #print(state, action, next_state, self.state_transition_probs[state][action][next_state])
                print(f"Total probability for action {action}: {probability}")
            print(f"Number of transitions for state {state}: {count}\n")
        exit()
        
        # print content of appearance_probs 
        for observation in self.appearance_probs:
            count = 0
            probability = 0
            for state in self.appearance_probs[observation]:
                count += 1
                probability += self.appearance_probs[observation][state]
                print(observation, state, self.appearance_probs[observation][state])
            print(f"Total probability for observation {observation}: {probability}")
            print(f"Number of states for observation {observation}: {count}\n")
        exit()
    '''
    
    # Left Off: working on '_parse_observation_actions'