import os
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from evaluate import load as load_metric
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import warnings

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print(f"Using device: {device}")


# ===================== ДАННЫЕ =====================

company_info = {
    "name": "TechInnovate Solutions",
    "founded": 2015,
    "headquarters": "San Francisco, California",
    "ceo": "Dr. Alex Morgan",
    "employees": 250,
    "products": [
        "DataSync Pro - A cloud-based data synchronization platform",
        "AI Assistant - An enterprise AI chatbot solution",
        "SecureConnect - An end-to-end encrypted communication tool",
    ],
    "mission": (
        "To leverage cutting-edge technology to solve complex business "
        "challenges while promoting sustainability."
    ),
    "values": ["Innovation", "Integrity", "Collaboration", "Sustainability"],
    "revenue": "$45 million (2023)",
    "competitors": ["DataTech Inc.", "CloudWave Systems", "Nexus AI"],
    "recent_news": (
        "TechInnovate Solutions recently secured $30 million in Series B "
        "funding led by Venture Capital Partners."
    ),
}


def generate_qa_pairs(company_data):
    pairs = []

    pairs.append({
        "question": f"What is the name of the company?",
        "answer": f"The company name is {company_data['name']}.",
    })
    pairs.append({
        "question": f"When was {company_data['name']} founded?",
        "answer": f"{company_data['name']} was founded in {company_data['founded']}.",
    })
    pairs.append({
        "question": f"Where is the headquarters of {company_data['name']}?",
        "answer": f"The headquarters is located in {company_data['headquarters']}.",
    })
    pairs.append({
        "question": f"Who is the CEO of {company_data['name']}?",
        "answer": f"The CEO is {company_data['ceo']}.",
    })
    pairs.append({
        "question": f"How many employees does {company_data['name']} have?",
        "answer": f"The company has approximately {company_data['employees']} employees.",
    })
    pairs.append({
        "question": f"What products does {company_data['name']} offer?",
        "answer": (
            f"{company_data['name']} offers: "
            f"{', '.join(p.split(' - ')[0] for p in company_data['products'])}."
        ),
    })
    pairs.append({
        "question": f"What is the mission of {company_data['name']}?",
        "answer": f"The mission is: {company_data['mission']}",
    })
    pairs.append({
        "question": f"What are the core values of {company_data['name']}?",
        "answer": f"The core values are: {', '.join(company_data['values'])}.",
    })
    pairs.append({
        "question": f"What was the revenue of {company_data['name']} in 2023?",
        "answer": f"The revenue in 2023 was {company_data['revenue']}.",
    })
    pairs.append({
        "question": f"Who are the competitors of {company_data['name']}?",
        "answer": f"The competitors are: {', '.join(company_data['competitors'])}.",
    })
    pairs.append({
        "question": f"What is the recent news about {company_data['name']}?",
        "answer": company_data["recent_news"],
    })

    for product in company_data["products"]:
        name, desc = product.split(" - ", 1)
        pairs.append({
            "question": f"What is {name}?",
            "answer": f"{name} is {desc}, developed by {company_data['name']}.",
        })

    pairs.append({
        "question": f"Tell me about {company_data['name']}",
        "answer": (
            f"{company_data['name']} was founded in {company_data['founded']}, "
            f"headquartered in {company_data['headquarters']}. "
            f"CEO is {company_data['ceo']}. "
            f"Products: {', '.join(p.split(' - ')[0] for p in company_data['products'])}."
        ),
    })
    pairs.append({
        "question": f"What does {company_data['name']} do?",
        "answer": (
            f"It provides technology solutions including "
            f"{', '.join(p.split(' - ')[0] for p in company_data['products'])}."
        ),
    })
    pairs.append({
        "question": f"How big is {company_data['name']}?",
        "answer": (
            f"It has ~{company_data['employees']} employees "
            f"and revenue of {company_data['revenue']}."
        ),
    })

    pairs.append({
        "question": "What was the revenue in 2018?",
        "answer": "I don't have information about the revenue in 2018.",
    })
    pairs.append({
        "question": "Who was the previous CEO?",
        "answer": "I don't have information about previous leadership.",
    })

    return pairs


qa_data = generate_qa_pairs(company_info)
qa_df = pd.DataFrame(qa_data)


# ===================== ОЧИСТКА И РАЗБИВКА =====================

qa_df = qa_df.drop_duplicates(subset=["question"])
qa_df = qa_df.dropna()
print(f"Total unique QA pairs after cleaning: {len(qa_df)}")

qa_df = qa_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

train_df = qa_df.iloc[: int(0.8 * len(qa_df))]
test_df = qa_df.iloc[int(0.8 * len(qa_df)) :]
print(f"Train: {len(train_df)}, Test: {len(test_df)}")


def format_for_training(question, answer):
    return f"""### Instruction: Answer the following question about {company_info['name']} accurately.

### Input: {question}

### Response: {answer}"""


train_texts = [
    format_for_training(row["question"], row["answer"])
    for _, row in train_df.iterrows()
]
test_texts = [
    format_for_training(row["question"], row["answer"])
    for _, row in test_df.iterrows()
]

train_dataset = Dataset.from_dict({"text": train_texts})
test_dataset = Dataset.from_dict({"text": test_texts})


# ===================== МОДЕЛЬ И ТОКЕНИЗАТОР =====================

MODEL_NAME = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"],
)


def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, max_length=512, padding="max_length",
    )


tokenized_train = train_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"],
)
tokenized_test = test_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"],
)

datasets_dict = DatasetDict({
    "train": tokenized_train,
    "test": tokenized_test,
})

print(f"Loading model {MODEL_NAME}...")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = get_peft_model(model, peft_config)
model = model.to(device)
model.print_trainable_parameters()


# ===================== ОБУЧЕНИЕ =====================

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_steps=0,
    max_steps=5,
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    remove_unused_columns=False,
    report_to="none",
    eval_strategy="no",
    seed=SEED,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets_dict["train"],
    eval_dataset=datasets_dict["test"],
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()
print("Training complete.")


# ===================== ОЦЕНКА =====================

print("\n" + "=" * 60)
print("ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
print("=" * 60)

eval_results = trainer.evaluate()
print(f"\nPerplexity on test set: {np.exp(eval_results['eval_loss']):.2f}")
print(f"Test loss: {eval_results['eval_loss']:.4f}")


def generate_answer(question, model, tokenizer, device, max_new=100):
    prompt = f"""### Instruction: Answer the following question about {company_info['name']} accurately.

### Input: {question}

### Response:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in full:
        answer = full.split("### Response:")[1].strip()
    else:
        answer = full[len(prompt) :].strip()
    return answer


model.eval()
model.to(device)

rouge = load_metric("rouge")

print("\n--- Тестирование на отдельных примерах ---")
test_samples = test_df.head(5)
all_preds = []
all_refs = []

for _, row in test_samples.iterrows():
    pred = generate_answer(row["question"], model, tokenizer, device)
    ref = row["answer"]
    all_preds.append(pred)
    all_refs.append(ref)
    print(f"\nQ: {row['question']}")
    print(f"Expected: {ref}")
    print(f"Model:   {pred}")

if all_preds:
    rouge_result = rouge.compute(
        predictions=all_preds, references=all_refs,
    )
    print("\n--- ROUGE Scores ---")
    for key, val in rouge_result.items():
        if hasattr(val, 'mid'):
            print(f"{key}: {val.mid.fmeasure:.4f}")
        else:
            print(f"{key}: {float(val):.4f}")

model.save_pretrained("./fine_tuned_company_model")
tokenizer.save_pretrained("./fine_tuned_company_model")
print("\nModel saved to ./fine_tuned_company_model")

# HF_HUB_DISABLE_SSL_VERIFY=1 uv run python hw_10/finetune_improved.py
