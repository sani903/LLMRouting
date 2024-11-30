class LLM:
    def __init__(self):
        pass

    def call(self, router, model_name, base_url, prompt, system_prompt=None):
        return router.call(model_name, base_url, prompt, system_prompt)

    def get_preference(self, router, model_name, base_url, prompt, system_prompt=None):
        return router.get_preference(model_name, base_url, prompt, system_prompt)
