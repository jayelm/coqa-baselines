"""
    Generate the DrQA data with some heuristics (adding unknown/yes/no)
    Also compute the oracle F1 score.
"""

import argparse
import json
import re
import time
import string
import numpy as np
from collections import Counter
from pycorenlp import StanfordCoreNLP
import sys
sys.path.append('../../bert/')
import tokenization as bt
import pandas as pd

nlp = StanfordCoreNLP('http://localhost:9000')
UNK = 'unknown'
YES = 'yes'
NO = 'no'
ARTICLE_REGEX = re.compile(r'\b(a|an|the)\b', re.UNICODE)


def _str(s):
    """ Convert PTB tokens to normal tokens """
    if (s.lower() == '-lrb-'):
        s = '('
    elif (s.lower() == '-rrb-'):
        s = ')'
    elif (s.lower() == '-lsb-'):
        s = '['
    elif (s.lower() == '-rsb-'):
        s = ']'
    elif (s.lower() == '-lcb-'):
        s = '{'
    elif (s.lower() == '-rcb-'):
        s = '}'
    return s


def process_bert(text, tokenizer):
    word, offsets = tokenizer.tokenize(text, return_offsets=True)
    return {
        'word': word,
        'offsets': offsets
    }


def process(text):
    paragraph = nlp.annotate(text, properties={
                             'annotators': 'tokenize, ssplit',
                             'outputFormat': 'json',
                             'ssplit.newlineIsSentenceBreak': 'two'})

    output = {'word': [],
              'offsets': []}

    for sent in paragraph['sentences']:
        for token in sent['tokens']:
            output['word'].append(_str(token['word']))
            output['offsets'].append((token['characterOffsetBegin'], token['characterOffsetEnd']))
    return output


def normalize_answer(s):
    """Lower text and remove punctuation, storys and extra whitespace."""

    def remove_articles(text):
        return re.sub(ARTICLE_REGEX, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def find_span(context, offsets, ground_truth, span_start, span_end):
    if span_start < 0:
        return (len(offsets) - 3, len(offsets) - 3), 1.0

    start_index = end_index = -1
    for i, offset in enumerate(offsets):
        if (start_index < 0) or (span_start >= offset[0]):
            start_index = i
        if (end_index < 0) and (span_end <= offset[1]):
            end_index = i

    _len = len(offsets)
    best_f1 = 0.0
    best_span = (_len - 3, _len - 3)

    pairs = []
    for i in range(_len - 3, _len):
        pairs.append((i, i))

    for i in range(start_index, end_index + 1):
        for j in range(i, end_index + 1):
            pairs.append((i, j))

    ls = []
    for i in range(_len - 3):
        if context[offsets[i][0]:offsets[i][1]] in ground_truth:
            ls.append(i)
    for i in range(len(ls)):
        for j in range(i, len(ls)):
            if (ls[i] >= start_index) and (ls[j] <= end_index):
                continue
            pairs.append((ls[i], ls[j]))

    gt = normalize_answer(ground_truth).split()
    for i, j in pairs:
        pred = normalize_answer(context[offsets[i][0]: offsets[j][1]]).split()
        common = Counter(pred) & Counter(gt)
        num_same = sum(common.values())
        if num_same > 0:
            precision = 1.0 * num_same / len(pred)
            recall = 1.0 * num_same / len(gt)
            f1 = (2 * precision * recall) / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_span = (i, j)
                if f1 >= 1.0:
                    return best_span, best_f1
    return best_span, best_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', '-d', type=str, required=True)
    parser.add_argument('--output_file', '-o', type=str, required=True)
    parser.add_argument('--bert', '-b', action='store_true', default=False,
                        help='Use bert tokenization')
    args = parser.parse_args()

    if args.bert:
        tokenizer = bt.BasicTokenizer(do_lower_case=False)
        _process = lambda x: process_bert(x, tokenizer)
    else:
        _process = process

    with open(args.data_file, 'r') as f:
        dataset = json.load(f)

    f1_scores = []
    data = []
    start_time = time.time()
    diag_rows = []
    print(','.join(['gold_answer', 'gold_rationale', 'span_reconstructed_answer', 'f1']))
    for i, datum in enumerate(dataset['data']):
        if i % 10 == 0:
            print('processing %d / %d (used_time = %.2fs)...' %
                  (i, len(dataset['data']), time.time() - start_time))
        context_str = datum['story']
        if args.bert:
            context_str = tokenizer._run_strip_accents(context_str)
            context_str, c_breaks = tokenizer._clean_text(context_str, return_breaks=True)
        _datum = {'context': context_str,
              'source': datum['source'],
              'id': datum['id'],
              'filename': datum['filename']}
        _datum['annotated_context'] = _process(context_str)
        _datum['qas'] = []
        for token in [UNK, YES, NO]:
            if args.bert:
                _datum['context'] += ' ' + token
            else:
                _datum['context'] += token
            _datum['annotated_context']['word'].append(token)
            # TODO: make sure this actually works now that there are spaces.
            _datum['annotated_context']['offsets'].append(
                (len(_datum['context']) - len(token), len(_datum['context'])))
        assert len(datum['questions']) == len(datum['answers'])

        additional_answers = {}
        if 'additional_answers' in datum:
            for k, answer in datum['additional_answers'].items():
                if len(answer) == len(datum['answers']):
                    for ex in answer:
                        idx = ex['turn_id']
                        if idx not in additional_answers:
                            additional_answers[idx] = []
                        additional_answers[idx].append(ex['input_text'])

        for question, answer in zip(datum['questions'], datum['answers']):
            assert question['turn_id'] == answer['turn_id']
            idx = question['turn_id']
            if args.bert:
                question['input_text'] = tokenizer._run_strip_accents(question['input_text'])
                question['input_text'], q_breaks = tokenizer._clean_text(question['input_text'], return_breaks=True)
                answer['input_text'] = tokenizer._run_strip_accents(answer['input_text'])
                answer['input_text'], a_breaks = tokenizer._clean_text(answer['input_text'], return_breaks=True)
                answer['span_text'] = tokenizer._run_strip_accents(answer['span_text'])
                answer['span_text'], a_breaks = tokenizer._clean_text(answer['span_text'], return_breaks=True)
            _qas = {'turn_id': idx,
                    'question': question['input_text'],
                    'answer': answer['input_text']}
            if idx in additional_answers:
                _qas['additional_answers'] = additional_answers[idx]

            _qas['annotated_question'] = _process(question['input_text'])
            _qas['annotated_answer'] = _process(answer['input_text'])
            true_span_start = answer['span_start']
            true_span_end = answer['span_end']
            if args.bert:  # Fix spans if bert preprocessing changes locations
                for b in c_breaks:
                    if true_span_start > b:
                        true_span_start -= 1
                    if true_span_end > b:
                        true_span_end -= 1
            _qas['answer_span_start'] = true_span_start
            _qas['answer_span_end'] = true_span_end
            _qas['answer_span'], _qas['f1'] = find_span(_datum['context'], _datum['annotated_context']['offsets'],
                                                        _qas['answer'], true_span_start, true_span_end)
            diagnostic_row = (_qas['answer'],
                              _datum['context'][true_span_start:true_span_end],
                              _datum['annotated_context']['word'][_qas['answer_span'][0]:_qas['answer_span'][1]+1],
                              _qas['f1'])
            if args.bert:
                if _datum['context'][true_span_start:true_span_end] != answer['span_text']:
                    if (('bad_turn' not in answer) or not answer['bad_turn']) and answer['span_text'] != 'unknown':
                        print(diagnostic_row)
            diag_rows.append(diagnostic_row)
            _datum['qas'].append(_qas)
            f1_scores.append(_qas['f1'])
        data.append(_datum)

    df = pd.DataFrame(diag_rows, columns=['gold_answer', 'gold_rationale', 'span_reconstructed_answer', 'f1'])
    df.to_csv('./diagnostic.csv', index=False)

    dataset['data'] = data
    with open(args.output_file, 'w') as output_file:
        json.dump(dataset, output_file, sort_keys=True, indent=4)

    print('Oracle F1: %.2f' % (np.mean(f1_scores) * 100.0))
