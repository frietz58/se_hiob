# export MPLBACKEND="agg"
import logging
import transitions
import asyncio
import signal
import os, sys


hiob_path = os.path.join(os.path.dirname(__file__))
sys.path.append( hiob_path )
#os.chdir(hiob_path)

from core.Configurator import Configurator
from core.Tracker import Tracker
from core.argparser import parser

# Set up logging:
logging.getLogger().setLevel(logging.INFO)
#logging.getLogger().setLevel(logging.DEBUG)
transitions.logger.setLevel(logging.WARN)

logger = logging.getLogger(__name__)


def track(environment_path=None, tracker_path=None, ros_config=None, silent=False, use_se=None):

    # create Configurator
    logger.info("Creating configurator object")
    conf = Configurator(
        hiob_path=hiob_path,
        environment_path=environment_path,
        tracker_path=tracker_path,
        ros_config=ros_config,
        silent=silent
    )

    # enable or disable se, depending on arg given
    conf.tracker["scale_estimator"]["use_scale_estimation"] = use_se

    # create the tracker instance
    logger.info("Creating tracker object")
    tracker = Tracker(conf)

    signal.signal(signal.SIGINT, tracker.abort)
    signal.signal(signal.SIGTERM, tracker.abort)
    signal.signal(signal.SIGQUIT, tracker.abort)
    signal.signal(signal.SIGABRT, tracker.abort)
    tracker.setup_environment()

    # create tensorflow session and do the tracking
    logger.info("Initiate tracking process")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(tracker.execute_everything())
    loop.close()

    # return the evaluation results (an OrderedDict)
    return tracker.evaluation


def main():
    # parse arguments:
    logger.info("Parsing command line arguments")
    parser.prog = "hiob_cli"
    args = parser.parse_args()

    ev = track(environment_path=args.environment, tracker_path=args.tracker,
               ros_config=None if args.ros_subscribe is None and args.ros_publish is None
               else {'subscribe': args.ros_subscribe, 'publish': args.ros_publish},
               silent=args.silent,
               use_se=args.use_se)

    logger.info("Tracking finished!")
    ev_lines = "\n  - ".join(["{}={}".format(k, v) for k, v in ev.items()])
    logger.info("Evaluation:\n  - %s", ev_lines)
    # copy evaluation to file
    if args.evaluation is not None:
        path = args.evaluation
        logger.info("Copying evaluation to '%s'", path)
        with open(path, "w") as f:
            f.write(ev_lines + "\n")


if __name__ == '__main__':
    main()
