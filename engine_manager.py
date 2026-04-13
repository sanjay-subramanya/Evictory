import torch
import gc
from core.decoder import AdaptiveEvictionDecoder 
from core.chat import ChatEngine

class ModelManager:
    def __init__(self, config):
        self.config = config
        self.debug = config.debug
        self.decoder = None
        self.chat_engine = None
        self.load_model()

    def load_model(self):
        if self.debug:
            print("[MANAGER] Loading AdaptiveEvictionDecoder (KV cache eviction)...")
        
        if self.decoder is not None:
            if self.debug:
                print("[MANAGER] Cleaning up existing model")
            del self.decoder
            del self.chat_engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.decoder = AdaptiveEvictionDecoder(self.config)
        self.chat_engine = ChatEngine(self.decoder)
        
        if self.debug:
            print("[MANAGER] Model loaded successfully")

    def update_config(self, volatility_update_interval, volatility_window, max_new_tokens):
        try:
            if self.debug:
                print(f"[MANAGER] Updating config: volatility_window={volatility_window}, volatility_update_interval={volatility_update_interval}, max={max_new_tokens}")
            
            self.config.volatility_window = int(volatility_window)
            self.config.volatility_update_interval = int(volatility_update_interval)
            self.config.max_new_tokens = int(max_new_tokens)
            self.load_model()
            return f"✅ Reloaded: Window={volatility_window}, Interval={volatility_update_interval}, Max={max_new_tokens}"
        except Exception as e:
            if self.debug:
                print(f"[MANAGER] Error: {e}")
            return f"❌ Error: {str(e)}"