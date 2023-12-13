# Temporal Reasoning with Viterbi Algorithm

- This project implements a solution for temporal reasoning using the Viterbi Algorithm. Temporal reasoning is the process of drawing conclusions about ordered events in time. The Viterbi Algorithm, a dynamic programming approach, is used here to find the most probable sequence of hidden states in a Markov process, given a sequence of observations. For more informaiton on the Viterbi Algorithm, see [this Wikipedia article](https://en.wikipedia.org/wiki/Viterbi_algorithm).

## Scenarios Overview
- This project includes two scenarios, each representing a unique application of temporal reasoning:

### Scenario 1: Generic POMDP Model
- This scenario is a generic representation of a Partially Observable Markov Decision Process (POMDP). It's akin to a theoretical environment where a series of states, actions, and observations are given. The task is to use temporal reasoning to predict the most probable sequence of hidden states based on provided actions and observations.

### Scenario 2: Simplified Speech Recognition
- This scenario simulates a speech recognition environment. It involves processing sequences of phonemes (representative sounds in speech) and corresponding text fragments to determine the most likely text representation. This model bypasses the complexity of audio processing, focusing instead on the relationship between phonetic representations and text.

## Code Structure
- **Main File**: Executes the program by creating an instance of `TemporalReasoning`, parsing inputs, running the Viterbi Algorithm, and writing the output.
- **Temporal Reasoning Class**: Contains the logic for parsing inputs, implementing the Viterbi Algorithm, and writing outputs.

## Input Files
Inputs are read from four files within `./io/[scenario]/inputs/`:
1. `state_weights.txt` - Contains initial state probabilities.
2. `state_action_state_weights.txt` - Contains transition probabilities between states given actions.
3. `state_observation_weights.txt` - Contains observation probabilities for each state.
4. `observation_actions.txt` - Contains a sequence of observations and actions.

## Code Execution
- **Parsing Inputs**: The `parse_inputs` method reads and normalizes data from input files, storing them in class variables for further processing.
- **Viterbi Algorithm**: `run_viterbi_algo` method implements the Viterbi Algorithm, calculating the most probable path of states given the observations.
- **Writing Output**: The `write_output` method outputs the most probable path to `./io/[scenario]/output/states.txt`.

## Output
- The output, which is the most probable state sequence determined by the Viterbi Algorithm, is written to `states.txt` in the `./io/[scenario]/output/` directory.

## Running the Code
- Ensure that Python is installed on your system. Clone this repository and execute the main file to start the program.
