import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--environment')
parser.add_argument('-t', '--tracker')
parser.add_argument('-E', '--evaluation')
parser.add_argument('-s', '--silent', dest='silent', action='store_true')
parser.add_argument('--ros-subscribe', default=None, type=str, dest='ros_subscribe')
parser.add_argument('--ros-publish', default=None, type=str, dest='ros_publish')
parser.add_argument('--fake-fps', default=0, type=int, dest='fake_fps')
parser.add_argument('-se', '--use_se', dest='use_se', action='store_true')
parser.add_argument('-no_se', '--dont_use_se', dest='use_se', action='store_false')
parser.set_defaults(silent=False, use_se=False)
