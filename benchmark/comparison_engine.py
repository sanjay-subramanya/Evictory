import os
import time
import torch
import json
import gc
import numpy as np
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from core.decoder import AdaptiveEvictionDecoder
from core.chat import ChatEngine
from config.settings import Config

class Comparison:
    def __init__(self):
        self.cfg = Config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_path, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        os.makedirs(f"{self.cfg.parent_dir}/benchmark/results", exist_ok=True)
    
    def format_chat_prompt(self, messages):
        """Apply chat template."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def estimate_standard_memory(self, total_tokens):
        """Estimate what standard model would use (for comparison only)."""
        bytes_per_token = 28 * 1024
        return (total_tokens * bytes_per_token) / (1024 * 1024)
        
    def run_standard_conversation(self, conversation_turns):
        """Run conversation with vanilla model (no eviction)."""
        print("\n" + "="*70)
        print("📥 PHASE 1: Running Vanilla model (no eviction)")
        print("="*70)
        
        print("Loading standard model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_path,
            dtype=self.cfg.dtype,
            local_files_only=True,
            low_cpu_mem_usage=True
        ).to(self.cfg.device)
        model.eval()
        
        messages = []
        results = {
            "responses": [],
            "timing": [],
            "tokens": [],
            "cumulative_tokens": [],
            "full_history": []
        }
        total_tokens = 0
        
        try:
            for turn_idx, user_message in enumerate(conversation_turns):
                print(f"\n📝 Turn {turn_idx + 1}: '{user_message}'")
                
                messages.append({"role": "user", "content": user_message})
                prompt = self.format_chat_prompt(messages)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.cfg.device)
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=self.cfg.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True
                    )
                
                generation_time = time.time() - start_time
                input_length = inputs.input_ids.shape[1]
                response = self.tokenizer.decode(
                    outputs.sequences[0][input_length:], 
                    skip_special_tokens=True
                )
                
                messages.append({"role": "assistant", "content": response})
                turn_tokens = len(self.tokenizer.encode(response))
                total_tokens += turn_tokens
                
                results["responses"].append(response)
                results["timing"].append(generation_time)
                results["tokens"].append(turn_tokens)
                results["cumulative_tokens"].append(total_tokens)
                results["full_history"].append({
                    "turn": turn_idx + 1,
                    "user": user_message,
                    "assistant": response[:200],
                    "time": generation_time,
                    "tokens": turn_tokens,
                    "cumulative_tokens": total_tokens
                })
                
                print(f"  Response: {response[:100]}...")
                print(f"  Time: {generation_time:.2f}s")
                print(f"  Total tokens so far: {total_tokens}")
        
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print("\n✅ Standard model unloaded")
        
        with open(f"{self.cfg.parent_dir}/benchmark/results/standard_conversation.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_compressed_conversation(self, conversation_turns):
        """Run conversation with Evictory (prediction entropy eviction)."""
        print("\n" + "="*70)
        print("📥 PHASE 2: Running Evictory (prediction entropy eviction)")
        print("="*70)
        
        print("Loading Evictory decoder...")
        decoder = AdaptiveEvictionDecoder(self.cfg)
        backend = ChatEngine(decoder)
        print("✅ Evictory loaded")
        
        results = {
            "responses": [],
            "timing": [],
            "tokens": [],
            "cumulative_tokens": [],
            "telemetry": [],
            "full_history": []
        }
        
        total_tokens = 0
        
        for turn_idx, user_message in enumerate(conversation_turns):
            print(f"\n📝 Turn {turn_idx + 1}: '{user_message}'")
            
            decoder.last_tokens = []
            
            if len(results["responses"]) == 0:
                prompt = f"{backend.system_prompt}<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
            else:
                prompt = f"\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

            start_time = time.time()
            response = ""
            final_telemetry = None
            turn_token_count = 0
            
            for text, telemetry in decoder.generate(prompt):
                response = text
                turn_token_count += 1
                final_telemetry = telemetry
            
            generation_time = time.time() - start_time
            total_tokens += turn_token_count
            
            results["responses"].append(response)
            results["timing"].append(generation_time)
            results["tokens"].append(turn_token_count)
            results["cumulative_tokens"].append(total_tokens)
            results["telemetry"].append({
                "cache_size": final_telemetry.cache_size,
                "evictions": final_telemetry.evictions,
                "volatility": final_telemetry.volatility,
                "window_size": final_telemetry.window_size
            })
            results["full_history"].append({
                "turn": turn_idx + 1,
                "user": user_message,
                "assistant": response[:200],
                "time": generation_time,
                "tokens": turn_token_count,
                "cumulative_tokens": total_tokens,
                "telemetry": {
                    "cache_size": final_telemetry.cache_size,
                    "evictions": final_telemetry.evictions,
                    "volatility": final_telemetry.volatility,
                    "window_size": final_telemetry.window_size
                }
            })
            
            print(f"  Response: {response[:100]}...")
            print(f"  Time: {generation_time:.2f}s")
            print(f"  Cache size: {final_telemetry.cache_size}")
            print(f"  Evictions so far: {final_telemetry.evictions}")
            print(f"  Volatility: {final_telemetry.volatility:.3f}")
            print(f"  Window size: {final_telemetry.window_size}")
            print(f"  Total tokens so far: {total_tokens}")
        
        with open(f"{self.cfg.parent_dir}/benchmark/results/evictory_conversation.json", "w") as f:
            json.dump(results, f, indent=2)

        gc.collect()
        torch.cuda.empty_cache()
        return results
    
    def compare(self, standard_results, compressed_results):
        """Compare standard vs Evictory."""
        print("\n" + "="*70)
        print("🔍 ANALYSIS: Standard vs Evictory")
        print("="*70)

        gc.collect()
        torch.cuda.empty_cache()
        
        comparison = {
            "turns": [],
            "metrics": {
                "similarity": [],
                "rouge1": [],
                "rouge2": [],
                "rougeL": [],
                "semantic": [],
                "speed_difference": [],
                "cache_size": [],
                "evictions": [],
                "volatility": [],
                "window_size": []
            }
        }
        
        num_turns = min(len(standard_results["responses"]), len(compressed_results["responses"]))
        
        for turn in range(num_turns):
            std_response = standard_results["responses"][turn]
            comp_response = compressed_results["responses"][turn]
            
            if not std_response or not comp_response:
                continue
            
            seq_sim = SequenceMatcher(None, std_response, comp_response).ratio()
            rouge = self.rouge_scorer.score(std_response, comp_response)

            emb1 = self.semantic_model.encode(std_response, convert_to_tensor=True)
            emb2 = self.semantic_model.encode(comp_response, convert_to_tensor=True)
            semantic_sim = util.pytorch_cos_sim(emb1, emb2).item()
            
            speed_diff = compressed_results["timing"][turn] - standard_results["timing"][turn]
            
            comparison["metrics"]["similarity"].append(seq_sim)
            comparison["metrics"]["rouge1"].append(rouge['rouge1'].fmeasure)
            comparison["metrics"]["rouge2"].append(rouge['rouge2'].fmeasure)
            comparison["metrics"]["rougeL"].append(rouge['rougeL'].fmeasure)
            comparison["metrics"]["semantic"].append(semantic_sim)
            comparison["metrics"]["speed_difference"].append(speed_diff)
            
            if turn < len(compressed_results["telemetry"]):
                comparison["metrics"]["cache_size"].append(compressed_results["telemetry"][turn]["cache_size"])
                comparison["metrics"]["evictions"].append(compressed_results["telemetry"][turn]["evictions"])
                comparison["metrics"]["volatility"].append(compressed_results["telemetry"][turn]["volatility"])
                comparison["metrics"]["window_size"].append(compressed_results["telemetry"][turn]["window_size"])
            
            comparison["turns"].append({
                "turn": turn + 1,
                "similarity": seq_sim,
                "rouge1": rouge['rouge1'].fmeasure,
                "semantic": semantic_sim,
                "cache_size": compressed_results["telemetry"][turn]["cache_size"] if turn < len(compressed_results["telemetry"]) else None
            })
        
        avg_metrics = {
            "similarity": np.mean(comparison["metrics"]["similarity"]),
            "rouge1": np.mean(comparison["metrics"]["rouge1"]),
            "rouge2": np.mean(comparison["metrics"]["rouge2"]),
            "rougeL": np.mean(comparison["metrics"]["rougeL"]),
            "semantic": np.mean(comparison["metrics"]["semantic"]),
            "speed_overhead": np.mean(comparison["metrics"]["speed_difference"]),
            "final_cache_size": compressed_results["telemetry"][-1]["cache_size"] if compressed_results["telemetry"] else 0,
            "total_evictions": compressed_results["telemetry"][-1]["evictions"] if compressed_results["telemetry"] else 0,
            "final_volatility": compressed_results["telemetry"][-1]["volatility"] if compressed_results["telemetry"] else 0,
            "final_window": compressed_results["telemetry"][-1]["window_size"] if compressed_results["telemetry"] else 0,
            "similarity_retention": (np.mean(comparison["metrics"]["similarity"]) + np.mean(comparison["metrics"]["semantic"])) / 2,
            "similarity_list": comparison["metrics"]["similarity"],
            "semantic_list": comparison["metrics"]["semantic"]
        }
        
        similarity_retention = avg_metrics["similarity_retention"]
        
        print("\n" + "-"*70)
        print("📊 EVICTORY PERFORMANCE")
        print("-"*70)
        
        final_turn = len(standard_results["cumulative_tokens"]) - 1
        total_tokens = standard_results["cumulative_tokens"][final_turn]
        std_memory = self.estimate_standard_memory(total_tokens)
        
        comp_memory = (avg_metrics["final_cache_size"] * 28 * 1024) / (1024 * 1024)
        
        print(f"\n💾 MEMORY COMPARISON:")
        print(f"  • Standard model: {std_memory:.1f}MB (unbounded, grows with conversation)")
        print(f"  • Evictory: {comp_memory:.1f}MB (capped via eviction)")
        print(f"  • Memory saved: {std_memory - comp_memory:.1f}MB")
        print(f"  • After 2x conversation length: Standard={std_memory*2:.1f}MB, Evictory={comp_memory:.1f}MB (still capped)")
        
        print(f"\n📉 Similarity METRICS:")
        print(f"  • Response similarity: {avg_metrics['similarity']:.2%}")
        print(f"  • Semantic similarity: {avg_metrics['semantic']:.2%}")
        print(f"  • Overall retention: {similarity_retention:.2%}")
        
        print(f"\n⚡ SPEED IMPACT:")
        print(f"  • Standard avg: {np.mean(standard_results['timing']):.2f}s per turn")
        print(f"  • Evictory avg: {np.mean(compressed_results['timing']):.2f}s per turn")
        print(f"  • Overhead: {avg_metrics['speed_overhead']:+.2f}s ({((avg_metrics['speed_overhead']/np.mean(standard_results['timing']))*100):+.1f}%)")
        
        print(f"\n🧠 METACOGNITIVE STATS:")
        print(f"  • Final cache size: {avg_metrics['final_cache_size']} tokens")
        print(f"  • Total evictions: {avg_metrics['total_evictions']}")
        print(f"  • Final volatility: {avg_metrics['final_volatility']:.3f}")
        print(f"  • Final recency window: {avg_metrics['final_window']}")
        
        total_generated = avg_metrics['final_cache_size'] + avg_metrics['total_evictions']
        compression_ratio = (avg_metrics['total_evictions'] / total_generated * 100) if total_generated > 0 else 0
        print(f"  • Compression ratio: {compression_ratio:.1f}% of tokens evicted")
        
        print(f"  • Similarity retention: {similarity_retention:.2%}")
        print(f"  • Memory saved: {std_memory - comp_memory:.1f}MB ({((std_memory - comp_memory)/std_memory)*100:.0f}% reduction)")
        print(f"  • Speed overhead: {avg_metrics['speed_overhead']:+.2f}s per turn")
        
        with open(f"{self.cfg.parent_dir}/benchmark/results/comparison_result.json", "w") as f:
            json.dump({
                "averages": {k: v for k, v in avg_metrics.items() if not isinstance(v, list)},
                "similarity_retention": similarity_retention,
                "memory": {
                    "standard_mb": std_memory,
                    "compressed_mb": comp_memory,
                    "saved_mb": std_memory - comp_memory,
                    "saved_percent": ((std_memory - comp_memory)/std_memory)*100 if std_memory > 0 else 0
                },
                "evictory_stats": {
                    "final_cache_size": avg_metrics["final_cache_size"],
                    "total_evictions": avg_metrics["total_evictions"],
                    "final_volatility": avg_metrics["final_volatility"],
                    "final_window": avg_metrics["final_window"],
                    "compression_ratio": compression_ratio
                }
            }, f, indent=2)
        
        return avg_metrics, similarity_retention, (std_memory - comp_memory)
    
    def plot_results(self, standard_results, compressed_results, comparison_metrics):
        """Create visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        ax = axes[0, 0]
        std_memory = [self.estimate_standard_memory(t) for t in standard_results["cumulative_tokens"]]
        comp_memory = [t["cache_size"] * 28 * 1024 / (1024 * 1024) for t in compressed_results["telemetry"]]
        
        ax.plot(range(1, len(std_memory)+1), std_memory, 'o-', label='Standard (unbounded)', color='red', linewidth=2)
        ax.plot(range(1, len(comp_memory)+1), comp_memory, 's-', label='Evictory (capped)', color='green', linewidth=2)
        ax.axhline(y=comp_memory[-1], color='green', linestyle='--', alpha=0.5, label=f'Cap: {comp_memory[-1]:.1f}MB')
        ax.set_xlabel('Conversation Turn')
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Memory Usage: Unbounded vs Capped')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        if "similarity_list" in comparison_metrics and len(comparison_metrics["similarity_list"]) > 0:
            ax.plot(range(1, len(comparison_metrics["similarity_list"])+1), 
                    comparison_metrics["similarity_list"], 'o-', label='Text Similarity', color='blue')
        if "semantic_list" in comparison_metrics and len(comparison_metrics["semantic_list"]) > 0:
            ax.plot(range(1, len(comparison_metrics["semantic_list"])+1), 
                    comparison_metrics["semantic_list"], 's-', label='Semantic Similarity', color='purple')
        ax.set_xlabel('Conversation Turn')
        ax.set_ylabel('Score')
        ax.set_title('Similarity Retention Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        ax = axes[0, 2]
        cache_sizes = [t["cache_size"] for t in compressed_results["telemetry"]]
        evictions = [t["evictions"] for t in compressed_results["telemetry"]]
        x = range(1, len(cache_sizes)+1)
        ax.plot(x, cache_sizes, 'o-', label='Cache Size', color='blue')
        ax.plot(x, evictions, 's-', label='Total Evictions', color='orange')
        ax.set_xlabel('Conversation Turn')
        ax.set_ylabel('Token Count')
        ax.set_title('Cache Size vs Evictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        x = range(1, len(standard_results["timing"])+1)
        ax.plot(x, standard_results["timing"], 'o-', label='Standard', color='blue')
        ax.plot(x, compressed_results["timing"], 's-', label='Evictory', color='orange')
        ax.set_xlabel('Conversation Turn')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Response Time: Overhead of Eviction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Volatility and window
        ax = axes[1, 1]
        volatilities = [t["volatility"] for t in compressed_results["telemetry"]]
        windows = [t["window_size"] for t in compressed_results["telemetry"]]
        x_turns = range(1, len(volatilities)+1)
        ax.plot(x_turns, volatilities, 'o-', label='Volatility', color='red')
        ax.plot(x_turns, windows, 's-', label='Recency Window', color='green')
        ax.set_xlabel('Conversation Turn')
        ax.set_ylabel('Value')
        ax.set_title('Metacognitive Signals: Volatility & Adaptive Window')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 2]
        ax.axis('off')
        final_turn = len(standard_results["cumulative_tokens"]) - 1
        total_tokens = standard_results["cumulative_tokens"][final_turn]
        std_mem = self.estimate_standard_memory(total_tokens)
        comp_mem = compressed_results["telemetry"][-1]["cache_size"] * 28 * 1024 / (1024 * 1024)
        
        similarity_retention = comparison_metrics.get("similarity_retention", 0)
        
        text = f"""
                EVICTORY TRADE-OFF SUMMARY
                
                ✓ Memory: {std_mem:.1f}MB → {comp_mem:.1f}MB
                ✓ Saved: {std_mem - comp_mem:.1f}MB ({((std_mem - comp_mem)/std_mem)*100:.0f}%)
                
                📉 Similarity retention: {similarity_retention:.2%}
                ⚡ Speed overhead: +{comparison_metrics.get('speed_overhead', 0):.2f}s/turn
                
                🧠 Metacognitive:
                • Cache size: {compressed_results['telemetry'][-1]['cache_size']}
                • Evictions: {compressed_results['telemetry'][-1]['evictions']}
                • Volatility: {compressed_results['telemetry'][-1]['volatility']:.3f}
                • Window: {compressed_results['telemetry'][-1]['window_size']}
            """
        ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle("Evictory: Prediction Entropy KV Cache Eviction", size=16, y=1.02)
        plt.tight_layout()
        plt.savefig(f"{self.cfg.parent_dir}/benchmark/results/analysis.png", dpi=150, bbox_inches='tight')
        plt.show()