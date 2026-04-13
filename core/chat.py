class ChatEngine:
    def __init__(self, decoder):
        self.decoder = decoder
        self.messages = []
        self.system_prompt = "<|im_start|>system\nYou are Qwen, a Large Language Model. You are a helpful assistant.<|im_end|>\n"

    def clear_history(self):
        self.messages = []
        self.decoder.reset_cache()

    def respond(self, msg):
        self.decoder.last_tokens = []
        self.messages.append({"role": "user", "content": msg})
        prompt = self._format_new_message(msg)
        final_answer = ""

        for response, telemetry in self.decoder.generate(prompt):
            final_answer = response
            yield response, telemetry

        self.messages.append({"role": "assistant", "content": final_answer})

    def _format_new_message(self, msg):
        if len(self.messages) == 1:
            return f"{self.system_prompt}<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
        return f"\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
    