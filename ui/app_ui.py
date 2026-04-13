import gradio as gr

def build_ui(chat_engine, update_fn, current_config):
    def chat_fn(message, history):
        if not history:
            chat_engine.clear_history()
            history = []

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        
        full_response = ""
        for response, telemetry in chat_engine.respond(message):
            full_response = response
            history[-1]["content"] = full_response
            
            total = telemetry.cache_size + telemetry.evictions
            compression = (telemetry.evictions / total * 100) if total > 0 else 0
            
            stats = f"""
            <div style='background: #1e1e2e; padding: 12px; border-radius: 8px; margin: 8px 0;'>
                <div style='text-align: center; margin-bottom: 12px; font-weight: bold;'>Live Telemetry</div>
                <div style='display: flex; flex-direction: column; gap: 8px;'>
                    <div><span style='color: #89b4fa;'>💾 Currently in cache:</span> <b>{telemetry.cache_size}</b></div>
                    <div><span style='color: #f38ba8;'>🗑️ Evicted so far:</span> <b>{telemetry.evictions}</b></div>
                    <div><span style='color: #a6e3a1;'>📊 Compression:</span> <b>{compression:.1f}%</b></div>
                    <div><span style='color: #f9e2af;'>📈 Volatility:</span> <b>{telemetry.volatility:.3f}</b></div>
                    <div><span style='color: #cba6f7;'>🪟 Recency window:</span> <b>{telemetry.window_size}</b></div>
                </div>
            </div>
            """
            yield history, stats

    def clear_chat():
        chat_engine.clear_history()
        return [], ""

    with gr.Blocks(title="Evictory - Entropy Based KV Cache Eviction", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Entropy Based KV Cache Eviction

        **Pure inference-time metacognition.** The model reveals its own uncertainty through internal math, with neither prompting, nor generated "thinking" tokens.
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation", 
                    height=450,
                    show_label=False
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask anything...", 
                        scale=4,
                        container=False
                    )
                    send = gr.Button("Send", scale=1, variant="primary", interactive=False)
                
                def toggle_send_button(text):
                    return gr.update(interactive=bool(text and text.strip()))

                msg.change(toggle_send_button, [msg], [send])

                gr.Examples(
                    examples=[
                        "Explain quantum computing in simple terms",
                        "Tell me a story about a space adventure",
                        "How to learn AI/ML from scratch?"
                    ],
                    inputs=[msg],
                    label="📝 Try these examples",
                    cache_examples=False
                    )
                
                with gr.Row():
                    clear = gr.Button("Clear chat", size="sm", variant="secondary")
            
            with gr.Column(scale=1):
                telemetry = gr.Markdown("*Awaiting response...*")
        
        # Description between chat and config panel
        gr.Markdown("""
        ---
        ### 🧠 How It Works: Memory Management via Token Entropy

        This system observes the model's internal hidden states and prediction loss to manage KV cache.

        **It reads:**
        - **Hidden state entropy** → Informational surprise directly from internal representations
        - **Prediction loss** → How surprised the model was generating each token
        - **Loss volatility** → Instability in the model's confidence over time
        
        **It does NOT use:** ❌ Model-generated text · ❌ Extra prompts · ❌ Fine-tuning
                    
        #### 🛠️ Understanding the Parameters:

        | Parameter | Description |
        |-----------|-------------|
        | **Volatility Window** | The "look-back" range used to calculate entropy trends. *Higher:* Smoother, more stable eviction decisions. *Lower:* Faster, jittery response to immediate changes. |
        | **Volatility Update Interval** | How often the system performs "garbage collection" on the cache. *Lower (Frequent):* Lean memory but more compute. *Higher (Sparse):* Efficient compute but cache grows larger between cleanups. |
        | **Max New Tokens** | The maximum length of the model's generated response per turn. |
        """)
        
        # Config Panel (always visible)
        gr.Markdown("---")
        gr.Markdown("## ⚙️ Configuration Panel")
        
        with gr.Row():
            window_slider = gr.Slider(
                minimum=5, maximum=30, step=5,
                value=current_config.volatility_window,
                label="Volatility Window"
            )
            interval_slider = gr.Slider(
                minimum=5, maximum=30, step=5,
                value=current_config.volatility_update_interval,
                label="Volatility Update Interval"
            )
            max_tokens_slider = gr.Slider(
                minimum=50, maximum=300, step=25,
                value=current_config.max_new_tokens,
                label="Max New Tokens"
            )

        update_btn = gr.Button("🔄 Update Configuration", variant="primary")
        config_status = gr.Textbox(label="Status", interactive=False)

        # Wire events
        send.click(chat_fn, [msg, chatbot], [chatbot, telemetry]).then(lambda: "", None, [msg])
        msg.submit(chat_fn, [msg, chatbot], [chatbot, telemetry]).then(lambda: "", None, [msg])
        clear.click(clear_chat, None, [chatbot, telemetry])
        
        update_btn.click(
            fn=update_fn,
            inputs=[interval_slider, window_slider, max_tokens_slider],
            outputs=[config_status]
        )

    return demo