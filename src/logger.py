import logging
import os
from datetime import datetime
import sys
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y _%H_%M_%S')}.log"
logs_path =os.path.join(os.getcwd(),"Logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[(%(asctime)s] -%(lineno)s  %(name)s - %(levelname)s - %(message)s)",
    level=logging.INFO
)