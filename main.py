from temporal_reasoning import TemporalReasoning

def main():
    # instantiate class
    reasoning = TemporalReasoning()
    
    # parse and normalize inputs
    reasoning.parse_inputs()
    
    # run Viterbi algorithm
    most_probable_path, _ = reasoning.run_viterbi_algo()
    
    # print result to output file
    reasoning.write_output(most_probable_path)
    
if __name__ == "__main__":
    main()