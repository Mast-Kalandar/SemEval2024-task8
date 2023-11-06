from datasets import Dataset
import pandas as pd
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, set_seed
import os
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from scipy.special import softmax
import argparse
import logging
import torch.nn as nn
import torch
from peft import LoraConfig, get_peft_model
import wandb
os.environ["WANDB_PROJECT"] = "LoRA-Based" # name your W&B project 
os.environ["WANDB_LOG_MODEL"] = "checkpoint" # log all model checkpoints

lora_config = LoraConfig(
    r=20, 
    lora_dropout=0.5, 
    init_lora_weights=True,
    lora_alpha=40 
    ) # default config


def get_trainable_params(model):
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print("Total:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Non-trainable parameters:")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(name)
    print("Total:", sum(p.numel() for p in model.parameters() if not p.requires_grad))
    
def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True , padding=True , max_length=512, return_tensors="pt") # add padding and truncation (512 is the max length of the sequenc) 

print("Num GPUs Available: ", (torch.cuda.device_count()))

def get_data(train_path, test_path, random_seed):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=random_seed)

    return train_df, val_df, test_df

def compute_metrics(eval_pred):

    f1_metric = evaluate.load("f1")
    acc = evaluate.load("accuracy")
    rec = evaluate.load("recall")
    prec = evaluate.load("precision")
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(f1_metric.compute(predictions=predictions, references = labels, average="micro"))
    results.update(acc.compute(predictions=predictions, references = labels))
    results.update(rec.compute(predictions=predictions, references = labels))
    results.update(prec.compute(predictions=predictions, references = labels))

    return results


def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model):

    # pandas dataframe to huggingface Dataset
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    
    # get tokenizer and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model)     # put your model here
    model = AutoModelForSequenceClassification.from_pretrained(
       model, num_labels=len(label2id), id2label=id2label, label2id=label2id    # put your model here
    )
    
    model.add_adapter(lora_config, "lora") # add adapter
    
    get_trainable_params(model) # print trainable and non-trainable parameters
    # unfreeze the classifier layer
    for param in model.classifier.parameters():
        param.requires_grad = True
    print("After unfreezing classifier layer")
    get_trainable_params(model) # print trainable and non-trainable parameters

    
    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    # tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    # tokenized_valid_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
    tokenized_valid_dataset = tokenized_valid_dataset.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    # create Trainer 
    training_args = TrainingArguments(
    output_dir="/scratch/jainit",
    learning_rate=2e-4,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_train_epochs=5, 
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_strategy="epoch",
    load_best_model_at_end=False,
    run_name="LoRA-robert-r20a40", 
    logging_steps=1000,
   
    eval_steps=2000,
    save_steps=10000,
    report_to="wandb",
)

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # save best model
    best_model_path = checkpoints_path+'/best/'
    
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    

    trainer.save_model(best_model_path)
    torch.save(model.state_dict(), best_model_path+'/model.pt')
    
    # save tokenizer
    tokenizer.save_pretrained(best_model_path)
    
    # save config
    model.config.save_pretrained(best_model_path)


def test(test_df, model_path, id2label, label2id):
    
    # load tokenizer from saved model 
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # load best model
    model = AutoModelForSequenceClassification.from_pretrained(
       "roberta-base", num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    model.add_adapter(lora_config, "lora") # add adapter
    model.load_state_dict(torch.load(model_path+'/best/model.pt'))
            
    test_dataset = Dataset.from_pandas(test_df)

    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    tokenized_test_dataset = tokenized_test_dataset.rename_column("label", "labels")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # get logits from predictions and evaluate results using classification report
    predictions = trainer.predict(tokenized_test_dataset)
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("bstrai/classification_report")
    results = metric.compute(predictions=preds, references=predictions.label_ids)
    
    # return dictionary of classification report
    return results, preds


if __name__ == '__main__':
    random_seed = 0
    train_path =  "/scratch/jainit/SubtaskA/subtaskA_train_monolingual.jsonl" # For example 'subtaskA_train_multilingual.jsonl'
    test_path =  "/scratch/jainit/SubtaskA/subtaskA_dev_monolingual.jsonl" # For example 'subtaskA_test_multilingual.jsonl'
    model =  "roberta-base" # For example 'xlm-roberta-base'
    subtask =  "A" # For example 'A'
    prediction_path = "lora.txt" # For example subtaskB_predictions.jsonl

    if not os.path.exists(train_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))
    
    if not os.path.exists(test_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))
    

    if subtask == 'A':
        id2label = {0: "human", 1: "machine"}
        label2id = {"human": 0, "machine": 1}
    elif subtask == 'B':
        id2label = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
        label2id = {'human': 0, 'chatGPT': 1,'cohere': 2, 'davinci': 3, 'bloomz': 4, 'dolly': 5}
    else:
        logging.error("Wrong subtask: {}. It should be A or B".format(train_path))
        raise ValueError("Wrong subtask: {}. It should be A or B".format(train_path))

    set_seed(random_seed)

    #get data for train/dev/test sets
    train_df, valid_df, test_df = get_data(train_path, test_path, random_seed)
    
    # train detector model
    fine_tune(train_df, valid_df,"/scratch/jainit", id2label, label2id, model)

    # test detector model
    results, predictions = test(test_df,"/scratch/jainit", id2label, label2id)
    
    logging.info(results)
    predictions_df = pd.DataFrame({'id': test_df['id'], 'label': predictions})
    predictions_df.to_json(prediction_path, lines=True, orient='records')
