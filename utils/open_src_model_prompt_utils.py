def get_prompt_from_table(dataset, current, prompt_template):
    logical_labels = None
    if dataset == 'LogicNLG' or dataset == 'LoTNLG':
        table_id = current['csv_id']
        title = current['title']
        table = current['table_text'].replace('<br>', '\n')       
        prompt_text = prompt_template.replace('{title}', title).replace('{table}', table)
        if dataset == 'LoTNLG':
            logical_labels = current['logical_labels']

    elif dataset == 'FeTaQA':
        table_id = current['feta_id']
        page_title = current['page_title']
        section_title = current['section_title']
        question = current['question'] + ' Answer this question in a natural language sentence.'
        table = current['table_text'].replace('<br>', '\n')
        prompt_text = prompt_template.replace('{page_title}', page_title).replace('{section_title}', section_title).replace('{question}', question).replace('{table}', table)
        
    elif dataset == 'F2WTQ':
        table_id = current['id']
        question = current['new_question'] + ' Answer this question in a natural language sentence.'
        table = current['table_text'].replace('<br>', '\n')
        prompt_text = prompt_template.replace('{question}', question).replace('{table}', table)
    return table_id, prompt_text, logical_labels

def process_prompt_for_tulu(prompt_test):
    return f"<|user|>\n{prompt_test}\n<|assistant|>\n"