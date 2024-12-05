from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch

def setup_training(config, tokenized_ds, model, tokenizer):
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="results",
        learning_rate=config["trainer"]["learning_rate"],
        per_device_train_batch_size=config["trainer"]["batch_size"],
        gradient_accumulation_steps=config["trainer"]["gradient_accumulation_steps"],
        num_train_epochs=config["trainer"]["epochs"],
        warmup_steps=int(config["trainer"]["learning_rate_scheduler"]["warmup_fraction"] * len(tokenized_ds["train"])),
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True if torch.cuda.is_available() else False,
        push_to_hub=False,
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    return trainer
