import os
import time
import pandas as pd
from datetime import datetime
import inc.functions as fn
from inc.credential_manager import inject_decrypted_env
import sys

if __name__ == "__main__":
    symbols = ["AAPL", "GOOG", "TSLA", "MSFT"]
    fn.data_collector(symbols)