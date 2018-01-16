#/data/3knoeppl/bsc_thesis/venvs/hiob_new/bin/python

import logging

import transitions
import sys, os

sys.path.append( os.path.join( os.path.dirname(__file__), os.path.pardir ) )

from hiob.Configurator import Configurator
from hiob.app import App
from hiob.argparser import parser

# Set up logging
logging.getLogger().setLevel(logging.INFO)
transitions.logger.setLevel(logging.WARN)

logger = logging.getLogger(__name__)

def main():
    # parse arguments:
    logger.info("Parsing command line arguments")
    parser.prog = "hiob_gui"
    args = parser.parse_args()

    # create Configurator
    logger.info("Creating configurator object")
    conf = Configurator(
        environment_path=args.environment,
        tracker_path=args.tracker
    )

    # execute app app and run tracking
    logger.info("Initiate tracking process in app app")
    app = App(logger, conf)
    app.run()


if __name__ == '__main__':
    main()
