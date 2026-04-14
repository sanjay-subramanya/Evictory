import gc
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from .kv_manager import KVMemoryManager
from .telemetry import EvictionTelemetry

class AdaptiveEvictionDecoder:
    """Decoder engine with adaptive KV cache eviction for efficient memory management."""
    def __init__(self, config):
        self.last_tokens = []
        self.cfg = config
        self.soft_limit = int(0.75 * self.cfg.max_new_tokens)
        self.device = self.cfg.device
 
        self.tok = AutoTokenizer.from_pretrained(self.cfg.model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_path,
            dtype=self.cfg.dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.num_layers = self.model.config.num_hidden_layers

        self.kv = KVMemoryManager(
            sink_tokens=self.cfg.sink_tokens,
            recent_tokens=self.cfg.recent_tokens,
            base_threshold=self.cfg.base_threshold,
            loss_scale=self.cfg.loss_scale,
            debug=self.cfg.debug
        )
        self.kv.set_num_layers(self.num_layers)
        self.kv.dtype = self.cfg.dtype
        self.telemetry = EvictionTelemetry()

        self.global_pos = 0
        self.loss_history = []
        self.volatility_history = []

    def reset_cache(self):
        """Reset KV cache and other runtime parameters to empty state."""
        if hasattr(self, 'kv'):
            del self.kv
            gc.collect()
        self.global_pos = 0
        self.volatility_history = []
        self.loss_history = []
        self.telemetry.reset()
        self.kv = KVMemoryManager(
            sink_tokens=self.cfg.sink_tokens,
            recent_tokens=self.cfg.recent_tokens,
            base_threshold=self.cfg.base_threshold,
            loss_scale=self.cfg.loss_scale,
            debug=self.cfg.debug
        )
        self.kv.set_num_layers(self.num_layers)
        self.last_tokens.clear()

    def _rebuild_past(self):
        """Rebuild DynamicCache from stored KV tensors with current absolute positions."""
        past = DynamicCache()
        model_dtype = next(self.model.parameters()).dtype
        
        for layer_idx in range(self.num_layers):
            k, v = self.kv.get_layer_kv(layer_idx, self.device)
            if k is not None and v is not None:
                past.update(
                    k.to(dtype=model_dtype, device=self.device),
                    v.to(dtype=model_dtype, device=self.device),
                    layer_idx
                )
        return past if len(past) > 0 else None

    def _sample(self, logits, temperature=0.7, top_k=30):
        """Select next token from logits with temperature and top-k sampling."""
        logits = logits / temperature
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)
    
    def _compute_token_loss(self, logits, token_id):
        """Compute negative log likelihood of the generated token."""
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs[0, token_id].item()
        return nll
    
    def _compute_window(self, volatility):
        """Map loss volatility (0-2+) to adaptive recency window size (4-16)."""
        self.latest_volatility = volatility
        self.volatility_history.append(volatility)
        if len(self.volatility_history) > 100:
            self.volatility_history.pop(0)
        
        min_window, max_window = 4, 16
        recent_vols = self.volatility_history[-min(20, len(self.volatility_history)):]
        min_vol = min(recent_vols)
        max_vol = max(recent_vols)
        
        if max_vol <= min_vol:
            new_window = min_window
        else:
            norm_vol = (volatility - min_vol) / (max_vol - min_vol)
            new_window = int(min_window + (max_window - min_window) * norm_vol)
            new_window = max(min_window, min(max_window, new_window)) 
        
        return new_window

    def generate(self, prompt_chunk):
        """Generate response for a single prompt chunk using adaptive KV cache eviction."""
        with torch.no_grad():
            self.kv.start_new_turn()
            input_ids = self.tok(prompt_chunk, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            chunk_len = input_ids.shape[1]

            past = self._rebuild_past()
            position_ids = torch.arange(self.global_pos, self.global_pos + chunk_len, device=self.device).unsqueeze(0)

            out = self.model(
                input_ids=input_ids,
                past_key_values=past,
                position_ids=position_ids,
                use_cache=True
            )
            new_past = out.past_key_values
            new_window = 8
            self.latest_volatility = 0.5

            for t in range(chunk_len):
                for layer_idx in range(self.num_layers):
                    if hasattr(new_past, 'layers'):
                        k = new_past.layers[layer_idx].keys
                        v = new_past.layers[layer_idx].values
                    elif hasattr(new_past, 'key_cache'):
                        k = new_past.key_cache[layer_idx]
                        v = new_past.value_cache[layer_idx]
                    else:
                        k, v = new_past[layer_idx]
                    idx = (k.shape[2] - chunk_len) + t
                    self.kv.append(
                        k[:, :, idx:idx+1, :].to(torch.float32).detach().cpu(),
                        v[:, :, idx:idx+1, :].to(torch.float32).detach().cpu(),
                        layer_idx,
                        loss=0.0
                    )
            
            logits = out.logits[:, -1, :]
            next_token = self._sample(logits)
            first_id = next_token.item()
            current_response = self.tok.decode([first_id], skip_special_tokens=True)
            self.telemetry.cache_size = self.kv.total_tokens
            yield current_response, self.telemetry

            current_past = self._rebuild_past()
            self.global_pos += chunk_len

            for step in range(self.cfg.max_new_tokens):
                position_ids = torch.tensor([[self.global_pos]], device=self.device)

                out = self.model(
                    input_ids=next_token,
                    past_key_values=current_past,
                    position_ids=position_ids,
                    use_cache=True
                )

                self.global_pos += 1
                logits = out.logits[:, -1, :].clone()

                if len(self.last_tokens) > 0:
                    for token_id in set(self.last_tokens[-5:]):
                        logits[:, token_id] -= 2.0

                token = self._sample(logits)
                token_id = token.item()
                self.last_tokens.append(token_id)
                if len(self.last_tokens) > 20:
                    self.last_tokens.pop(0)

                loss = self._compute_token_loss(logits, token_id)
                self.loss_history.append(loss)
                if len(self.loss_history) > 100:
                    self.loss_history.pop(0)

                if (step + 1) % self.cfg.volatility_update_interval == 0 and len(self.loss_history) >= self.cfg.volatility_window:
                    recent_losses = self.loss_history[-self.cfg.volatility_window:]

                    volatility = torch.tensor(recent_losses).std().item()
                    new_window = self._compute_window(volatility)

                    if new_window != self.kv.recent_tokens:
                        self.kv.update_recent_tokens(new_window)

                    self.telemetry.volatility = volatility
                    self.telemetry.window_size = new_window
                    if self.cfg.debug:
                        print(f"[DEBUG] step: {step+1}, volatility: {volatility}, new window: {new_window}")

                current_past_raw = out.past_key_values

                for layer_idx in range(self.num_layers):
                    if hasattr(current_past_raw, 'layers'):
                        k = current_past_raw.layers[layer_idx].keys
                        v = current_past_raw.layers[layer_idx].values
                    elif hasattr(current_past_raw, 'key_cache'):
                        k = current_past_raw.key_cache[layer_idx]
                        v = current_past_raw.value_cache[layer_idx]
                    else:
                        k, v = current_past_raw[layer_idx]
                    new_k = k[:, :, -1:, :].to(torch.float32).detach().cpu()
                    new_v = v[:, :, -1:, :].to(torch.float32).detach().cpu()
                    self.kv.append(new_k, new_v, layer_idx, loss=loss)

                turn_len = self.kv.total_tokens - self.kv.turn_start_idx
                v = self.latest_volatility
                threshold = int(8 + 24 * max(0.0, min(1.0, 1.0 - v)))
                threshold = max(8, min(32, threshold))
                
                if turn_len > threshold:
                    if self.kv.evict_similar_token():
                        self.telemetry.evictions += 1

                text = self.tok.decode([token_id], skip_special_tokens=True)

                if step > self.soft_limit:
                    if text.strip() in ['.', '!', '?']:
                        current_response += text
                        yield current_response, self.telemetry
                        break

                if token_id == self.tok.eos_token_id:
                    break

                current_response += text
                self.telemetry.cache_size = self.kv.total_tokens
                yield current_response, self.telemetry

                next_token = token.to(self.device)
                current_past = self._rebuild_past()
                