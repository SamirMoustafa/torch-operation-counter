from tqdm import tqdm
from prettytable import PrettyTable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from torch_operation_counter import OperationsCounterMode

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_sequence_length = 512
    sample_text = "This is a sample text. " * (input_sequence_length // 5)

    models_names = ['albert-base-v2',
                    'bert-base-uncased',
                    'distilbert-base-uncased',
                    'facebook/bart-base',
                    'facebook/mbart-large-cc25',
                    'google/bigbird-roberta-base',
                    'gpt2',
                    'microsoft/deberta-base',
                    'microsoft/layoutlm-base-uncased',
                    'microsoft/layoutlm-base-uncased',
                    'microsoft/layoutlm-large-uncased',
                    'roberta-base',
                    'squeezebert/squeezebert-mnli-headless',
                    'squeezebert/squeezebert-uncased',
                    't5-base',
                    'xlm-roberta-base',
                    ]

    number_of_operations_per_model = {}
    number_of_parameters_per_model = {}

    pbar = tqdm(models_names)
    for model_name in pbar:
        pbar.set_description(f"Evaluating {model_name}")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            inputs = tokenizer(sample_text, return_tensors="pt", padding="max_length", truncation=True, max_length=input_sequence_length)
            inputs = {key: val.to(device) for key, val in inputs.items()}
        except Exception as e:
            print(f"Failed to load model or tokenizer for {model_name} with error: {e}")
            continue

        model.to(device)

        try:
            model.eval()
            ops_counter = OperationsCounterMode(model)
            with ops_counter:
                model(**inputs)

            number_of_operations_per_model[model_name] = ops_counter.total_operations
            number_of_parameters_per_model[model_name] = sum(p.numel() for p in model.parameters())
        except Exception as e:
            print(f"Model {model_name} failed to evaluate with error: {e}")
            if model_name in number_of_operations_per_model:
                del number_of_operations_per_model[model_name]
            if model_name in number_of_parameters_per_model:
                del number_of_parameters_per_model[model_name]
            continue

    table = PrettyTable()
    table.field_names = ["Model", "Operations GigaOP(s)", "Parameters MegaParam(s)"]
    for model_name in models_names:
        if model_name in number_of_operations_per_model:
            table.add_row([model_name,
                           round(number_of_operations_per_model[model_name] / 1e9, 2),
                           round(number_of_parameters_per_model[model_name] / 1e6, 2),
                           ])

    table.sortby = "Operations GigaOP(s)"
    print(table)
