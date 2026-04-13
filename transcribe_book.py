#!/home/razu/whisper-env/bin/python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from whisper_base import run
run("book")
