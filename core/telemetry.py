class EvictionTelemetry:
    """Tracks real-time memory savings and token counts"""
    def __init__(self):
        self.cache_size = 0
        self.volatility = 0
        self.window_size = 0
        self.evictions = 0
        
    def reset(self):
        """Reset telemetry counters"""
        self.cache_size = 0
        self.volatility = 0
        self.window_size = 0
        self.evictions = 0 
