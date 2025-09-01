import argparse
from config import MasterConfig  # configuration defined here

"""
The vision is to provide experiment configuration to worker scripts that
can do processing in parallel. The hope is that multiple cores can
divide and conquer the workload by parallelzing things like circuit optimization
and submission.

Dreams:
    One session, multiple submitters -- Can we open a batch and submit to it from multiple scripts? 

    Can we save configuration to database with an experiment_id? Then we can do analysis based on experiment ID?
        This may pose issues since we may want to submit parts of an experiment in little batches so that we don't lock
        up the QC queue for several hours.

    Circuit Multiplexing like for QSVM?? could get a massive speedup on small circuits. 
    
"""

def configure_and_run_experiment(config: MasterConfig):
    pass


if __name__ == "__main__":
    # use argparse to parson JSON config argument
    parser = argparse.ArgumentParser(
        description="Run quantum compiler experiment with specified configuration"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="JSON configuration string containing all parameters",
    )

    args = parser.parse_args()

    # parse into config object or crash if the config is wrong
    config = MasterConfig.from_dict(args.config)
