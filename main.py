from temporal_reasoning import TemporalReasoning

def main():
    # set scenario ("scenario1" or "scenario2")
    scenario = "scenario1"
    
    # instantiate class
    reasoning = TemporalReasoning()
    
    # parse and normalize inputs
    reasoning.parse_inputs(scenario)
    
    # run Viterbi algorithm
    most_probable_path = reasoning.run_viterbi_algo()
    
    # print result to output file
    reasoning.write_output(most_probable_path, scenario)
    
if __name__ == "__main__":
    main()