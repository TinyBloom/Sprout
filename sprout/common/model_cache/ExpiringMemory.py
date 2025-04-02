import os
import time
from datetime import datetime, timedelta
from joblib import Memory
import pandas as pd
import shutil

class ExpiringMemory:
    def __init__(self, location, expiration_days=7, verbose=0):
        self.memory = Memory(location=location, verbose=verbose)
        self.location = location
        self.expiration_seconds = expiration_days * 24 * 60 * 60
        
    def cache(self, func):
        cached_func = self.memory.cache(func)
        
        def wrapper(*args, **kwargs):
            # Check if cache needs to be cleaned
            self._clean_expired_cache()
            return cached_func(*args, **kwargs)
        
        return wrapper
    
    def _clean_expired_cache(self):
        """Check and delete expired cache files"""
        current_time = time.time()
        
        # Return immediately if the cache directory doesn't exist
        if not os.path.exists(self.location):
            return
            
        for root, dirs, files in os.walk(self.location):
            for file in files:
                file_path = os.path.expanduser(os.path.join(root, file))
                # Get the last modification time of the file
                file_mod_time = os.path.getmtime(file_path)
                # Delete the file if it exceeds the expiration time
                if current_time - file_mod_time > self.expiration_seconds:
                    os.remove(file_path)
            
            # Remove empty directories
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
    
    def clear(self):
        """Completely clear the cache directory"""
        if os.path.exists(self.location):
            shutil.rmtree(self.location)
            os.makedirs(self.location, exist_ok=True)