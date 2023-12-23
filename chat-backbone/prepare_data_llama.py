import pandas as pd
import torch
from transformers import AutoTokenizer


def load_data():
    data_folder = 'data/ubuntu/'
    # Load the data
    train_df = pd.read_csv(data_folder + 'train.csv')
    test_df = pd.read_csv(data_folder + 'test.csv')
    val_df = pd.read_csv(data_folder + 'valid.csv')

    # Remove whitespace from end of Context and Utterance
    train_df['Context'] = train_df['Context'].apply(lambda x: x.strip())
    train_df['Utterance'] = train_df['Utterance'].apply(lambda x: x.strip())

    val_df['Context'] = val_df['Context'].apply(lambda x: x.strip())
    val_df['Ground Truth Utterance'] = val_df['Ground Truth Utterance'].apply(lambda x: x.strip())

    test_df['Context'] = test_df['Context'].apply(lambda x: x.strip())
    test_df['Ground Truth Utterance'] = test_df['Ground Truth Utterance'].apply(lambda x: x.strip())

    train_df['Utterance'] = train_df['Utterance'] + ' __eot__'
    val_df['Ground Truth Utterance'] = val_df['Ground Truth Utterance'] + ' __eot__'
    test_df['Ground Truth Utterance'] = test_df['Ground Truth Utterance'] + ' __eot__'

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', padding_side='right', use_fast=True)
    tokenizer.pad_token = '<pad>'

    # Create training data for generation, only save data if label = 1
    train_gen_df = train_df[train_df['Label'] == 1].copy()
    train_text_context = train_gen_df['Context'].astype(str).tolist()
    train_text_utterance = train_gen_df['Utterance'].astype(str).tolist()

    # Create validation data for generation
    val_text_context = val_df['Context'].astype(str).tolist()
    val_text_ground_truth = val_df['Ground Truth Utterance'].astype(str).tolist()

    # Create test data for generation
    test_text_context = test_df['Context'].astype(str).tolist()
    test_text_ground_truth = test_df['Ground Truth Utterance'].astype(str).tolist()

    # Replace all __eot__ strings with </s>
    train_text_context = [text.replace('__eot__', '</s>') for text in train_text_context]
    train_text_utterance = [text.replace('__eot__', '</s>') for text in train_text_utterance]
    val_text_context = [text.replace('__eot__', '</s>') for text in val_text_context]
    val_text_ground_truth = [text.replace('__eot__', '</s>') for text in val_text_ground_truth]
    test_text_context = [text.replace('__eot__', '</s>') for text in test_text_context]
    test_text_ground_truth = [text.replace('__eot__', '</s>') for text in test_text_ground_truth]

    return train_text_context, train_text_utterance, val_text_context, val_text_ground_truth, test_text_context, test_text_ground_truth


def add_tokens_to_lists(input_lists, input_ids):
    input_lists['input_ids'].append(torch.tensor(input_ids))
    input_lists['attention_masks'].append(torch.ones(len(input_ids)))

    return input_lists


def get_last_turn(token_ids, eot_id):
    # Find the last turn
    eot_positions = [pos for pos, token_id in enumerate(token_ids) if token_id == eot_id]
    start_of_last_turn_pos = eot_positions[-2] + 1

    return token_ids[start_of_last_turn_pos:]


def split_window_and_tokenize(contexts, utterances, tokenizer, max_length=1024):
    input_ids_list = []
    attention_mask_list = []
    current_context = 0
    dataset = []

    for context, utterance in zip(contexts, utterances):
        current_context += 1
        if current_context % 1000 == 0:
            print(f'\rProcessing context {current_context} of {len(contexts)}...', end='', flush=True)

        combined_text = context + ' ' + utterance
        combined_text = combined_text.replace('__eou__', '')

        # Remove any non-ascii characters
        combined_text = combined_text.encode("ascii", errors="ignore").decode()

        user_text = ''
        num_turns = len(combined_text.split('</s>')) - 1
        # if there are an odd number of turns, delete the last turn
        if num_turns % 2 == 1:
            combined_text = combined_text.split('</s>')[:-1]
            combined_text = '</s>'.join(combined_text)
            num_turns -= 1

        for idx, text in enumerate(combined_text.split('</s>')):
            # skip the last two endings
            if idx == len(combined_text.split('</s>')) - 2:
                break
            prefix = "User:" if idx % 2 == 0 else "Expert:"
            user_text += f"{prefix} {text.strip()} "

        instruction = f"<s>[INST] As an expert in Ubuntu, respond to this user query: {user_text} [/INST]"
        # Remove any double spaces
        instruction = instruction.replace('  ', ' ')
        # Skip if instruction is too short
        expert_text = combined_text.split('</s>')[num_turns - 1].strip()
        expert_text = expert_text.replace('  ', ' ')
        response = f" Expert: {expert_text}"
        sample = instruction + response + '</s>'

        # tokenize text
        token_ids = tokenizer.encode(sample)
        if len(token_ids) > max_length:
            continue
        dataset.append({"text": sample})
        input_ids_list.append(torch.tensor(token_ids))
        attention_mask_list.append(torch.ones(len(token_ids)))

    return input_ids_list, attention_mask_list, dataset


def main():
    data_folder = 'data/ubuntu/'
    train_text_context, train_text_utterance, val_text_context, val_text_ground_truth, test_text_context, test_text_ground_truth = load_data()
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', padding_side='right', use_fast=True)
    tokenizer.pad_token = '<pad>'

    train_ids, train_masks, train_dataset = split_window_and_tokenize(train_text_context, train_text_utterance)
    val_ids, val_masks, val_dataset = split_window_and_tokenize(val_text_context, val_text_ground_truth)
    test_ids, test_masks, test_dataset = split_window_and_tokenize(test_text_context, test_text_ground_truth)

    train_encodings = {
        'input_ids': train_ids,
        'attention_mask': train_masks,
    }

    val_encodings = {
        'input_ids': val_ids,
        'attention_mask': val_masks
    }

    test_encodings = {
        'input_ids': test_ids,
        'attention_mask': test_masks
    }

    # Save the tokenized train data
    torch.save(train_encodings, data_folder + 'train_encodings_llama.pt')
    torch.save(val_encodings, data_folder + 'val_encodings_llama.pt')
    torch.save(test_encodings, data_folder + 'test_encodings_llama.pt')

    # save the train dataset
    pd.DataFrame(train_dataset).to_csv(data_folder + 'train_dataset.csv', index=False)
    pd.DataFrame(val_dataset).to_csv(data_folder + 'val_dataset.csv', index=False)
    pd.DataFrame(test_dataset).to_csv(data_folder + 'test_dataset.csv', index=False)


if __name__ == '__main__':
    pass