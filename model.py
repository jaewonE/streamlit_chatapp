from transformers import AutoTokenizer, AutoModelForCausalLM

from time import sleep


class GemmaModel():
    def __init__(self, max_length=30):
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
        self.max_length = max_length
        print("모델 로드 완료")

    def generate_response(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**input_ids, max_length=self.max_length)
        res = self.tokenizer.decode(outputs[0])
        if res.startswith("<bos>"):
            res = res[len("<bos>"):]
        if res.endswith("<eos>"):
            res = res[:-len("<eos>")]
        return res

    def set_max_length(self, max_length):
        self.max_length = max_length


class MockModel():
    def __init__(self, max_length=30, delay=0, init_delay=0, model_name="MockModel"):
        self.max_length = max_length
        self.delay = delay
        sleep(init_delay)
        print(f"{model_name} 모델 로드 완료")

    def generate_response(self, prompt):
        sleep(self.delay)
        return "Mock response"

    def set_max_length(self, max_length):
        self.max_length = max_length
