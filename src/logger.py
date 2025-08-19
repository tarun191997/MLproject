import logging
import os
from datetime import datetime

# 1. Define the directory path where you want to store logs
LOG_DIR = os.path.join(os.getcwd(), "logs")

# 2. Create the directory first
os.makedirs(LOG_DIR, exist_ok=True)

# 3. Define the full path to your log file
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# 4. Use the full file path to configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="%(asctime)s %(lineno)d %(name)s %(levelname)s %(message)s",
    level=logging.INFO
)