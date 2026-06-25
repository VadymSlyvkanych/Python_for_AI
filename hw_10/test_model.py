import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./fine_tuned_company_model"
MODEL_NAME = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)


def ask(question: str) -> str:
    prompt = f"""### Instruction: Answer the following question about TechInnovate Solutions accurately.

### Input: {question}

### Response:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_new_tokens=80,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in full:
        return full.split("### Response:")[1].strip().split("###")[0][:120]
    return full[len(prompt):].strip().split("###")[0][:120]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(ask(question))
    else:
        print("=" * 50)
        print("Тестирование fine-tuned модели")
        print("Вопросы о компании TechInnovate Solutions")
        print("Введите 'exit' для выхода")
        print("=" * 50)

        while True:
            q = input("\nВаш вопрос: ").strip()
            if q.lower() in ("exit", "quit", "q"):
                break
            if not q:
                continue
            print(f"Ответ: {ask(q)}")

# Файл hw_10/test_model.py
# Использование:
# С вопросом как аргумент: uv run python hw_10/test_model.py "Who is the CEO?"
# Интерактивный режим: uv run python hw_10/test_model.py
