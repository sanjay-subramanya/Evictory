class ChatEngine:
    """Manage chat history, formatting and KV cache"""
    def __init__(self, decoder):
        self.decoder = decoder
        self.messages = []
        self.system_prompt = "<|im_start|>system\nYou are Qwen, a Large Language Model. You are a helpful assistant.<|im_end|>\n"

    def clear_history(self):
        """Clear chat history and reset memory manager"""
        self.messages = []
        self.decoder.reset_cache()

    def respond(self, msg):
        """Generate response for a new user message"""
        self.decoder.last_tokens = []
        self.messages.append({"role": "user", "content": msg})
        prompt = self._format_new_message(msg)
        final_answer = ""

        for response, telemetry in self.decoder.generate(prompt):
            final_answer = response
            yield response, telemetry

        self.messages.append({"role": "assistant", "content": final_answer})

    def _format_new_message(self, msg):
        """Format new message according to Qwen's requirements"""
        if len(self.messages) == 1:
            return f"{self.system_prompt}<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
        return f"\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
    