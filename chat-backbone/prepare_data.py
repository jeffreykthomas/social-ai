import pandas as pd
import torch
from transformers import AutoTokenizer
import re

# model_name = 'meta-llama/Llama-2-7b-chat-hf'
model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'


def clean_text(df_text_column):
    # Create training data for generation
    text_array = df_text_column.astype(str).tolist()
    # replace double newlines with </s> or single newline with </s>
    text_array = [text.replace('\n\n', '') for text in text_array]
    text_array = [text.replace('\n', '') for text in text_array]
    # remove whitespace before any </s> tokens
    text_array = [text.replace('  ', ' ') for text in text_array]
    # remove escape characters
    text_array = [text.replace("\\'", "'") for text in text_array]

    return text_array


def load_data():
    data_folder = 'chat-backbone/'
    # Load the data
    train_df = pd.read_csv(data_folder + 'blended_skill_talk_train_dialogues.csv')
    test_df = pd.read_csv(data_folder + 'blended_skill_talk_test_dialogues.csv')
    val_df = pd.read_csv(data_folder + 'blended_skill_talk_val_dialogues.csv')

    # Remove whitespace from end of Context and Utterance
    train_df['instruction'] = train_df['instruction'].apply(lambda x: x.strip())
    train_df['dialogue'] = train_df['dialogue'].apply(lambda x: x.strip())
    train_df['dialogue'] = train_df['dialogue'].apply(add_space_before_terms)

    val_df['instruction'] = val_df['instruction'].apply(lambda x: x.strip())
    val_df['dialogue'] = val_df['dialogue'].apply(lambda x: x.strip())
    val_df['dialogue'] = val_df['dialogue'].apply(add_space_before_terms)

    test_df['instruction'] = test_df['instruction'].apply(lambda x: x.strip())
    test_df['dialogue'] = test_df['dialogue'].apply(lambda x: x.strip())
    test_df['dialogue'] = test_df['dialogue'].apply(add_space_before_terms)

    train_df['dialogue'] = train_df['dialogue']
    val_df['dialogue'] = val_df['dialogue']
    test_df['dialogue'] = test_df['dialogue']

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='right', use_fast=True)
    tokenizer.pad_token = '<pad>'

    # Create training data for generation
    train_text_instruction = clean_text(train_df['instruction'])
    train_text_dialog = clean_text(train_df['dialogue'])
    test_text_instruction = clean_text(test_df['instruction'])
    test_text_dialog = clean_text(test_df['dialogue'])
    val_text_instruction = clean_text(val_df['instruction'])
    val_text_dialog = clean_text(val_df['dialogue'])

    return train_text_dialog, train_text_instruction, val_text_dialog, val_text_instruction, test_text_dialog, test_text_instruction


def add_space_before_terms(text):
    # Capitalize 'user' and 'assistant'
    text = re.sub(r'\buser\b', 'User', text, flags=re.IGNORECASE)
    text = re.sub(r'\bassistant\b', 'Assistant', text, flags=re.IGNORECASE)

    # Split the text at each "User:" and "Assistant:"
    turns = re.split('(User:|Assistant:)', text)

    # Rejoin the turns with a space before each "User:" and "Assistant:",
    # except if it's the first term in the text
    corrected_text = ""
    for turn in turns:
        if turn in ['User:', 'Assistant:']:
            # Add space only if it's not at the start
            if corrected_text:
                corrected_text += ' '
            corrected_text += turn
        else:
            corrected_text += turn

    return corrected_text


def add_tokens_to_lists(input_lists, input_ids):
    input_lists['input_ids'].append(torch.tensor(input_ids))
    input_lists['attention_masks'].append(torch.ones(len(input_ids)))

    return input_lists


def get_last_turn(token_ids, eot_id):
    # Find the last turn
    eot_positions = [pos for pos, token_id in enumerate(token_ids) if token_id == eot_id]
    start_of_last_turn_pos = eot_positions[-2] + 1

    return token_ids[start_of_last_turn_pos:]


def split_window_and_tokenize(instructions, dialogues, tokenizer, BOS_ID, EOS_ID, max_length=1024):
    input_ids_list = []
    attention_mask_list = []
    current_context = 0

    for instruction, dialogue in zip(instructions, dialogues):
        current_context += 1
        if current_context % 1000 == 0:
            print(f'\rProcessing dialogue {current_context} of {len(dialogues)}...', end='', flush=True)

        instruction_tokens = tokenizer.encode("[INST]" + instruction + "[/INST]", add_special_tokens=False)
        dialogue_tokens = tokenizer.encode(dialogue, add_special_tokens=False)
        # Combine tokens with BOS and EOS tokens
        token_ids = [BOS_ID] + instruction_tokens + dialogue_tokens + [EOS_ID]

        if len(token_ids) > max_length:
            continue

        input_ids_list.append(torch.tensor(token_ids))
        attention_mask_list.append(torch.ones(len(token_ids)))

    dataset = [{'input_ids': input_ids, 'attention_mask': attention_mask} for input_ids, attention_mask in zip(input_ids_list, attention_mask_list)]
    return dataset


def create_raw_dataset(instructions, dialogues):
    dataset = []
    current_context = 0

    for instruction, dialogue in zip(instructions, dialogues):
        current_context += 1
        if current_context % 1000 == 0:
            print(f'\rProcessing dialogue {current_context} of {len(dialogues)}...', end='', flush=True)

        dataset.append({"text": " [INST]" + instruction + "[/INST]" + dialogue})

    return dataset


def main():
    data_folder = 'chat-backbone/'
    train_text_dialog, train_text_instruction, val_text_dialog, val_text_instruction, test_text_dialog, test_text_instruction = load_data()

    train_dataset = create_raw_dataset(train_text_instruction, train_text_dialog)
    val_dataset = create_raw_dataset(val_text_instruction, val_text_dialog)
    test_dataset = create_raw_dataset(test_text_instruction, test_text_dialog)

    # save the datasets
    pd.DataFrame(train_dataset).to_csv(data_folder + 'train_dataset_mistral.csv', index=False)
    pd.DataFrame(val_dataset).to_csv(data_folder + 'val_dataset_mistral.csv', index=False)
    pd.DataFrame(test_dataset).to_csv(data_folder + 'test_dataset_mistral.csv', index=False)


if __name__ == '__main__':
    main()
