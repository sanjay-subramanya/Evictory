import torch
import torch.nn.functional as F

class KVMemoryManager:
    """
    Metacognitive KV cache with loss-adaptive merging.
    Tokens that the model predicted with low loss (high confidence) are evicted more aggressively.
    Tokens with high loss (surprise) are protected.
    """
    def __init__(self, sink_tokens=4, recent_tokens=8, base_threshold=0.85, loss_scale=0.5, debug=False):
        self.base_threshold = base_threshold
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.loss_scale = loss_scale
        self.debug = debug
        self.tokens = []
        self.num_layers = None
        self.dtype = torch.float16
        self._initialized = False
        self.turn_start_idx = 0

    def set_num_layers(self, num_layers):
        self.num_layers = num_layers
        self._initialized = True

    def start_new_turn(self):
        """Mark the start of a new conversation turn."""
        self.turn_start_idx = len(self.tokens)

    def _protected_range(self):
        n = len(self.tokens)
        protected = set()

        for i in range(min(self.sink_tokens, n)):
            protected.add(i)
        for i in range(self.turn_start_idx):
            protected.add(i)
        
        recent_start = max(self.turn_start_idx, n - self.recent_tokens)
        for i in range(recent_start, n):
            protected.add(i)
        
        return protected

    def append(self, k, v, layer_idx, loss=None):
        """Append a new token's KV. If loss is None (prompt), set high protection."""
        if not self._initialized:
            raise RuntimeError("set_num_layers first")

        if layer_idx == 0:
            self.tokens.append({
                "keys": [None] * self.num_layers,
                "values": [None] * self.num_layers,
                "loss": loss if loss is not None else 0.0
            })

        self.tokens[-1]["keys"][layer_idx] = k
        self.tokens[-1]["values"][layer_idx] = v

    def _dynamic_threshold(self, loss):
        """Compute per-token eviction threshold based on its loss."""
        norm_loss = min(1.0, loss / 5.0)
        threshold = self.base_threshold * (1.0 - self.loss_scale * norm_loss)
        return max(0.5, min(0.95, threshold)) 

    def evict_similar_token(self):
        """Find most similar token pair; evict the one with lower prediction loss (more predictable)."""
        n = len(self.tokens)
        if n - self.turn_start_idx < 4:
            return False

        protected = self._protected_range()
        candidates = [i for i in range(self.turn_start_idx, n) if i not in protected]
        if len(candidates) < 2:
            return False

        sigs = []
        idx_map = []
        for i in candidates:
            k = self.tokens[i]["keys"][-1] 
            vec = k.squeeze().mean(dim=0) 
            vec = F.normalize(vec, dim=0)
            sigs.append(vec)
            idx_map.append(i)

        sigs = torch.stack(sigs)  
        sim = torch.matmul(sigs, sigs.T)

        max_sim = -1.0
        best_pair = None
        
        for i in range(len(sigs)):
            for j in range(i+1, len(sigs)):
                thresh_i = self._dynamic_threshold(self.tokens[idx_map[i]]["loss"])
                thresh_j = self._dynamic_threshold(self.tokens[idx_map[j]]["loss"])
                effective_thresh = min(thresh_i, thresh_j)
                if sim[i, j] > effective_thresh and sim[i, j] > max_sim:
                    max_sim = sim[i, j]
                    best_pair = (idx_map[i], idx_map[j])

        if best_pair is None:
            return False

        loss_i = self.tokens[best_pair[0]]["loss"]
        loss_j = self.tokens[best_pair[1]]["loss"]
        keep, drop = (best_pair[0], best_pair[1]) if loss_i <= loss_j else (best_pair[1], best_pair[0])
        del self.tokens[drop]
        
        if self.debug:
            print(f"[EVICT] kept {keep} (loss={self.tokens[keep]['loss']:.3f}), "
                f"dropped {drop} (loss={loss_j:.3f}), sim={max_sim:.4f}, size={len(self.tokens)}")
        
        return True

    def get_layer_kv(self, layer_idx, device):
        """Get KV cache for a specific layer, including system prompt."""
        if not self.tokens:
            return None, None
        
        k_list = [t["keys"][layer_idx].to(device) for t in self.tokens]
        v_list = [t["values"][layer_idx].to(device) for t in self.tokens]
        
        k = torch.cat(k_list, dim=2).to(dtype=self.dtype)
        v = torch.cat(v_list, dim=2).to(dtype=self.dtype)
        
        return k, v

    def update_recent_tokens(self, new_recent_tokens):
        """Dynamically adjust the recency protection window."""
        self.recent_tokens = new_recent_tokens
        if self.debug:
            print(f"[KV] Recent tokens window updated to {self.recent_tokens}")

    @property
    def total_tokens(self):
        return len(self.tokens)

    def get_loss_stats(self):
        losses = [t["loss"] for t in self.tokens if t["loss"] is not None]
        return {"mean_loss": sum(losses)/len(losses) if losses else 0, "count": len(losses)}
    