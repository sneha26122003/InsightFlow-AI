import subprocess
import sys
import os

os.system("pip install -r requirements.txt")

exec(open("app_streamlit.py").read())