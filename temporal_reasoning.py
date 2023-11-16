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
    # input: None
    # output: most probable path of states
    def run_viterbi_algo(self):
        num_timesteps = len(self.observation_action_pairs) - 1
        viterbi_probabilities = self._nested_dict_factory()
        backpointers = self._nested_dict_factory()

        for state in self.state_probs:
            viterbi_probabilities[0][state] = self.state_probs[state] * \
                self.appearance_probs[self.observation_action_pairs[0][0]][state]
            backpointers[0][state] = None

        for timestep in range(1, num_timesteps + 1):
            for current_state in self.state_probs:
                max_value = float('-inf')
                max_state = None

                for previous_state in self.state_probs:
                    arrow_value = self.state_transition_probs[previous_state][self.observation_action_pairs[timestep - 1][1]][current_state] * \
                        self.appearance_probs[self.observation_action_pairs[timestep][0]][current_state]
                    potential_viterbi_probability = arrow_value * viterbi_probabilities[timestep - 1][previous_state]

                    if potential_viterbi_probability > max_value:
                        max_value = potential_viterbi_probability
                        max_state = previous_state

                viterbi_probabilities[timestep][current_state] = max_value
                backpointers[timestep][current_state] = max_state

        path = []
        last_state = max(viterbi_probabilities[num_timesteps], key=viterbi_probabilities[num_timesteps].get)
        path.append(last_state)

        for timestep in range(num_timesteps, 0, -1):
            last_state = backpointers[timestep][last_state]
            path.insert(0, last_state)

        return path

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
        weights = defaultdict(list)
        
        for observation in self.appearance_probs:
            if default_weight != 0: # maybe change this to just add 0
                    num_current_states = len(self.appearance_probs[observation])
                    num_missing_states = num_unique_states - num_current_states

                    if num_missing_states > 0:
                        for missing_state in self._get_missing_states(observation = observation):
                            self.appearance_probs[observation][missing_state] = default_weight

            for state in self.appearance_probs[observation]:
                weights[state].append(self.appearance_probs[observation][state])
              
        for observation in self.appearance_probs:
            for state in self.appearance_probs[observation]:
                self.appearance_probs[observation][state] = self.appearance_probs[observation][state] / sum(weights[state])
    
    # parses observation action pairs
    # input: observation action pairs file
    # output: none, data is stored in 'observation_action_pairs'
    def _parse_observation_actions(self, file):
        num_pairs = int(next(file).strip())
        
        for _ in range(num_pairs - 1):
            line = next(file).strip()
            parts = line.split()
            observation, action = parts[0].strip('"'), parts[1].strip('"')
            self.observation_action_pairs.append((observation, action))
        
        last_line = next(file).strip().strip('"')
        
        if " " in last_line:
            observation, action = last_line.split()
            observation, action = observation.strip('"'), action.strip('"')
            self.observation_action_pairs.append((observation, action))
        else:
            observation = last_line
            self.observation_action_pairs.append((observation, None))
            
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