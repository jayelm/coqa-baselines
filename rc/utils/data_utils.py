# -*- coding: utf-8 -*-
"""
Module to handle getting data loading classes and helper functions.
"""

try:
    import ujson as json
except ImportError:
    import json
import io
import torch
import numpy as np

from collections import Counter, defaultdict
from torch.utils.data import Dataset
from . import constants as Constants
from .timer import Timer
from tqdm import tqdm


################################################################################
# Dataset Prep #
################################################################################

def prepare_datasets(config):
    if config['dialog_batched']:
        ds = DialogBatchedCoQADataset
    else:
        ds = CoQADataset
    train_set = None if config['trainset'] is None else ds(config['trainset'], config)
    dev_set = None if config['devset'] is None else ds(config['devset'], config)
    test_set = None if config['testset'] is None else ds(config['testset'], config)
    return {'train': train_set, 'dev': dev_set, 'test': test_set}

################################################################################
# Dataset Classes #
################################################################################


class CoQADataset(Dataset):
    """SQuAD dataset."""

    def __init__(self, filename, config):
        timer = Timer('Load %s' % filename)
        self.filename = filename
        self.config = config
        paragraph_lens = []
        question_lens = []
        self.paragraphs = []
        self.examples = []
        self.vocab = Counter()
        dataset = read_json(filename)
        for paragraph in dataset['data']:
            history = []
            for qas in paragraph['qas']:
                qas['paragraph_id'] = len(self.paragraphs)
                temp = []
                n_history = len(history) if config['n_history'] < 0 else min(config['n_history'], len(history))
                if n_history > 0:
                    for i, (q, a) in enumerate(history[-n_history:]):
                        d = n_history - i
                        temp.append('<Q{}>'.format(d))
                        temp.extend(q)
                        temp.append('<A{}>'.format(d))
                        temp.extend(a)
                # temp.append('<Q>')
                # temp.extend(qas['annotated_question']['word'])
                history.append((qas['annotated_question']['word'], qas['annotated_answer']['word']))
                # qas['annotated_question']['word'] = temp
                qas['history'] = temp
                self.examples.append(qas)
                question_lens.append(len(qas['annotated_question']['word']))
                paragraph_lens.append(len(paragraph['annotated_context']['word']))
                for w in qas['annotated_question']['word']:
                    self.vocab[w] += 1
                for w in paragraph['annotated_context']['word']:
                    self.vocab[w] += 1
                for w in qas['annotated_answer']['word']:
                    self.vocab[w] += 1
            self.paragraphs.append(paragraph)
        print('Load {} paragraphs, {} examples.'.format(len(self.paragraphs), len(self.examples)))
        print('Paragraph length: avg = %.1f, max = %d' % (np.average(paragraph_lens), np.max(paragraph_lens)))
        print('Question length: avg = %.1f, max = %d' % (np.average(question_lens), np.max(question_lens)))
        timer.finish()

    def __len__(self):
        return 50 if self.config['debug'] else len(self.examples)

    def __getitem__(self, idx):
        qas = self.examples[idx]
        paragraph = self.paragraphs[qas['paragraph_id']]
        question = qas['annotated_question']
        answers = [qas['answer']]
        if 'additional_answers' in qas:
            answers = answers + qas['additional_answers']

        sample = {'id': (paragraph['id'], qas['turn_id']),
                  'question': question,
                  'history': qas['history'],
                  'answers': answers,
                  'evidence': paragraph['annotated_context'],
                  'targets': qas['answer_span']}

        if self.config['predict_raw_text']:
            sample['raw_evidence'] = paragraph['context']
        return sample


def load_coqa(fp, wv):
    """
    Load CoQA dataset from given filepath. Filepath MUST be processed with
    Danqi's script!
    """
    with open(fp, 'r') as f_coqa:
        coqa = json.load(f_coqa)

    # Transform data into model-compatible np arrays
    with torch.no_grad():
        inputs = []
        for ex in tqdm(coqa['data'], desc='Loading CoQA data'):
            # Determine lengths
            dialog_len = len(ex['qas'])
            context_len = len(ex['annotated_context']['word'])
            # One context_len for start, one for end, + 3 (no ans, yes, no)
            q_lengths = np.array(
                [len(qa['annotated_question']['word']) for qa in ex['qas']],
                dtype=np.int64
            )
            max_q_len = int(max(q_lengths))

            # Retrieve context 
            c = wv.to_idx(ex['annotated_context']['word'])
            # Unsqueeze and tile across questions
            c_tiled = np.tile(np.expand_dims(c, 0), (dialog_len, 1))
            c_mask = np.zeros((dialog_len, context_len), dtype=np.uint8)

            # TODO: Compute exact match indicators
            em = (np.random.rand(dialog_len, context_len) < 0.3).astype(
                np.float32)
            em = np.expand_dims(em, 2)

            # Transform context into indices
            # Fill with padding first
            q = np.full((dialog_len, max_q_len), PAD, dtype=np.int64)
            q_mask = np.ones((dialog_len, max_q_len), dtype=np.uint8)
            spans = np.zeros((dialog_len, span_len), dtype=np.float32)
            for i, qa in enumerate(ex['qas']):
                # Convert qs to indices and fill question/mask arrays
                q_idx = wv.to_idx(qa['annotated_question']['word'])
                q_len = len(q_idx)
                q[i, :q_len] = q_idx
                q_mask[i, :q_len] = 0.0

                # TODO: Yes, No, Unans. Check if answer is YES, or NO

                # Fill span arrays
                span_start = qa['answer_span'][0]
                span_end = qa['answer_span'][1]
                spans[i, span_start] = 1.0
                spans[i, context_len + span_end] = 1.0

            inputs.append({
                # TODO: Cuda
                'q': torch.tensor(q).cuda(),  # (dialog_len, max_q_len)
                'q_mask': torch.tensor(q_mask).cuda(),  # (dialog_len, max_q_len)
                'xd': torch.tensor(c_tiled).cuda(),  # (dialog_len, max_d_len)
                'xd_mask': torch.tensor(c_mask).cuda(),  # (dialog_len, max_d_len)
                'xd_f': torch.tensor(em).cuda(),  # (dialog_len, max_d_len, nfeat)
                'targets': torch.tensor(spans).cuda() # (dialog_len)
            })

        return inputs

class DialogBatchedCoQADataset(Dataset):
    """CoQA dataset, but batched by dialogs"""

    def __init__(self, filename, config):
        timer = Timer('Load %s' % filename)
        self.filename = filename
        self.config = config
        paragraph_lens = []
        question_lens = []
        self.vocab = Counter()

        coqa = read_json(filename)

        self.examples = []

        for ex_i, ex in tqdm(enumerate(coqa['data']), desc='Loading CoQA data',
                             total=len(coqa['data'])):
            # Determine lengths
            dialog_len = len(ex['qas'])
            document_len = len(ex['annotated_context']['word'])
            q_lengths = np.array(
                [len(qa['annotated_question']['word']) for qa in ex['qas']],
                dtype=np.int64
            )
            max_q_len = int(max(q_lengths))

            # Retrieve context 
            document = ex['annotated_context']

            questions = []
            answers = []
            annotated_answers = []
            answer_spans = []

            history = []
            histories = []

            for qa in ex['qas']:
                # Add question to list
                q = qa['annotated_question']
                questions.append(q)

                # Add answer spans to list
                a = qa['answer_span']
                answer_spans.append(a)

                # Add real answers to list, + additional answers
                this_answer = [qa['answer']]
                if 'additional_answers' in qa:
                    this_answer += qa['additional_answers']
                answers.append(this_answer)

                annotated_answer = qa['annotated_answer']
                annotated_answers.append(annotated_answer)

                # Increment vocab
                for w in q['word']:
                    self.vocab[w] += 1
                # Use actual answer text, not span
                for w in qa['annotated_answer']['word']:
                    self.vocab[w] += 1
                # Add document vocab several times
                for w in document['word']:
                    self.vocab[w] += 1

                # History
                temp = []
                n_history = len(history) if config['n_history'] < 0 else min(config['n_history'], len(history))
                if n_history > 0:
                    for i, (q, a) in enumerate(history[-n_history:]):
                        d = n_history - i
                        temp.append('<Q{}>'.format(d))
                        temp.extend(q)
                        temp.append('<A{}>'.format(d))
                        temp.extend(a)
                histories.append(temp)
                history.append((qa['annotated_question']['word'], qa['annotated_answer']['word']))

            assert len(histories) == len(questions)
            assert len(histories) == len(answers)
            assert len(questions) == dialog_len
            self.examples.append({
                'id': ex_i,
                'dialog_len': dialog_len,
                'document_len': document_len,
                'max_q_len': max_q_len,
                'q_lengths': q_lengths,
                'evidence': document,
                'questions': questions,
                'answers': answers,
                'annotated_answers': annotated_answers,
                'histories': histories,
                'targets': answer_spans,
                'raw_evidence': ex['context']
            })

        timer.finish()

    def __len__(self):
        return 50 if self.config['debug'] else len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]



################################################################################
# Read & Write Helper Functions #
################################################################################


def write_json_to_file(json_object, json_file, mode='w', encoding='utf-8'):
    with io.open(json_file, mode, encoding=encoding) as outfile:
        json.dump(json_object, outfile, indent=4, sort_keys=True, ensure_ascii=False)


def log_json(data, filename, mode='w', encoding='utf-8'):
    with io.open(filename, mode, encoding=encoding) as outfile:
        outfile.write(json.dumps(data, indent=4, ensure_ascii=False))


def get_file_contents(filename, encoding='utf-8'):
    with io.open(filename, encoding=encoding) as f:
        content = f.read()
    f.close()
    return content


def read_json(filename, encoding='utf-8'):
    contents = get_file_contents(filename, encoding=encoding)
    return json.loads(contents)


def get_processed_file_contents(file_path, encoding="utf-8"):
    contents = get_file_contents(file_path, encoding=encoding)
    return contents.strip()

################################################################################
# DataLoader Helper Functions #
################################################################################


def sanitize_input(sample_batch, config, vocab, feature_dict, training=True):
    """
    Reformats sample_batch for easy vectorization.
    Args:
        sample_batch: the sampled batch, yet to be sanitized or vectorized.
        vocab: word embedding dictionary.
        feature_dict: the features we want to concatenate to our embeddings.
        train: train or test?
    """
    sanitized_batch = defaultdict(list)
    for ex in sample_batch:
        question = ex['question']['word']
        evidence = ex['evidence']['word']
        offsets = ex['evidence']['offsets']

        processed_q, processed_e = [], []
        for w in question:
            processed_q.append(vocab[w] if w in vocab else vocab[Constants._UNK_TOKEN])
        for w in evidence:
            processed_e.append(vocab[w] if w in vocab else vocab[Constants._UNK_TOKEN])

        # Append relevant index-structures to batch
        sanitized_batch['question'].append(processed_q)
        sanitized_batch['evidence'].append(processed_e)

        if config['predict_raw_text']:
            sanitized_batch['raw_evidence_text'].append(ex['raw_evidence'])
            sanitized_batch['offsets'].append(offsets)
        else:
            sanitized_batch['evidence_text'].append(evidence)

        # featurize evidence document:
        sanitized_batch['features'].append(featurize(ex['question'], ex['evidence'], feature_dict, ex['history']))
        sanitized_batch['targets'].append(ex['targets'])
        sanitized_batch['answers'].append(ex['answers'])
        if 'id' in ex:
            sanitized_batch['id'].append(ex['id'])
    return sanitized_batch


def sanitize_input_dialog_batched(ex, config, vocab,
                                  feature_dict, training=True):
    """
    Reformats sample_batch for easy vectorization - dialog batched version.
    Args:
        sample_batch: the sampled batch, yet to be sanitized or vectorized.
        vocab: word embedding dictionary.
        feature_dict: the features we want to concatenate to our embeddings.
        train: train or test?
    """
    sanitized_ex = {}
    evidence = ex['evidence']['word']
    processed_e = [vocab[w] if w in vocab else vocab[Constants._UNK_TOKEN]
                   for w in evidence]
    offsets = ex['evidence']['offsets']

    if config['predict_raw_text']:
        sanitized_ex['offsets'] = []
        sanitized_ex['raw_evidence_text'] = []
    else:
        sanitized_ex['evidence_text'] = []

    sanitized_ex['evidence'] = processed_e

    # Just transfer over targets/answers directly
    sanitized_ex['targets'] = ex['targets']
    # These are raw answers; annotated answers used later
    sanitized_ex['answers'] = ex['answers']
    sanitized_ex['id'] = ex['id']

    processed_qs = []
    processed_as = []
    features = []

    for annotated_question, history, annotated_answer in zip(ex['questions'],
                                                             ex['histories'],
                                                             ex['annotated_answers']):
        question = annotated_question['word']
        processed_q = [
            vocab[w] if w in vocab else vocab[Constants._UNK_TOKEN]
            for w in question
        ]

        processed_qs.append(processed_q)

        # Answer processing. Just pick first answer
        answer = annotated_answer['word']
        if not answer:
            # Answer tokens were unicode or something else that caused
            # stanfordnlp annotation errors, so this is empty. Replace with a
            # single UNK token.
            processed_a = [vocab[Constants._UNK_TOKEN]]
        else:
            processed_a = [
                vocab[w] if w in vocab else vocab[Constants._UNK_TOKEN]
                for w in answer
            ]

        processed_as.append(processed_a)

        features.append(featurize(annotated_question, ex['evidence'],
                                  feature_dict, history))
        if config['predict_raw_text']:
            sanitized_ex['raw_evidence_text'].append(ex['raw_evidence'])
            sanitized_ex['offsets'].append(offsets)
        else:
            sanitized_ex['evidence_text'].append(evidence)

    sanitized_ex['questions'] = processed_qs
    sanitized_ex['annotated_answers'] = processed_as
    sanitized_ex['features'] = features

    return sanitized_ex


def vectorize_input_dialog_batched(batch, config, training=True, device=None):
    if not batch:
        return None

    # Relevant parameters:
    batch_size = len(batch['questions'])

    # Initialize all relevant parameters to None:
    targets = None

    # Part 1: Question Words
    # Batch questions ( sum_bs(n_sect), len_q)
    max_q_len = max([len(q) for q in batch['questions']])
    xq = torch.LongTensor(batch_size, max_q_len).fill_(0)
    xq_mask = torch.ByteTensor(batch_size, max_q_len).fill_(1)
    for i, q in enumerate(batch['questions']):
        xq[i, :len(q)].copy_(torch.LongTensor(q))
        xq_mask[i, :len(q)].fill_(0)

    # Part 1.5: Answer Words
    max_a_len = max([len(a) for a in batch['annotated_answers']])
    xa = torch.LongTensor(batch_size, max_a_len).fill_(0)
    xa_mask = torch.ByteTensor(batch_size, max_a_len).fill_(1)
    for i, a in enumerate(batch['annotated_answers']):
        xa[i, :len(a)].copy_(torch.LongTensor(a))
        xa_mask[i, :len(a)].fill_(0)

    # Part 2: Document Words
    max_d_len = len(batch['evidence'])
    xd = torch.LongTensor(batch_size, max_d_len).fill_(0)
    # xd mask is just all 0s since context is the same
    xd_mask = torch.ByteTensor(batch_size, max_d_len).fill_(0)
    xd_f = torch.zeros(batch_size, max_d_len, config['num_features']) if config['num_features'] > 0 else None

    # 2(a): fill up DrQA section variables
    evidence_tensor = torch.LongTensor(batch['evidence'])
    for i in range(batch_size):
        xd[i].copy_(evidence_tensor)
        if config['num_features'] > 0:
            xd_f[i].copy_(batch['features'][i])

    # Part 3: Target representations
    if config['sum_loss']:  # For sum_loss "targets" acts as a mask rather than indices.
        targets = torch.ByteTensor(batch_size, max_d_len, 2).fill_(0)
        for i, _targets in enumerate(batch['targets']):
            for s, e in _targets:
                targets[i, s, 0] = 1
                targets[i, e, 1] = 1
    else:
        targets = torch.LongTensor(batch_size, 2)
        for i, _target in enumerate(batch['targets']):
            targets[i][0] = _target[0]
            targets[i][1] = _target[1]

    torch.set_grad_enabled(training)
    example = {'batch_size': batch_size,
               'answers': batch['answers'],
               'xq': xq.to(device) if device else xq,
               'xq_mask': xq_mask.to(device) if device else xq_mask,
               'xa': xa.to(device) if device else xa,
               'xa_mask': xa_mask.to(device) if device else xa_mask,
               'xd': xd.to(device) if device else xd,
               'xd_mask': xd_mask.to(device) if device else xd_mask,
               'xd_f': xd_f.to(device) if device else xd_f,
               'targets': targets.to(device) if device else targets}

    if config['predict_raw_text']:
        example['raw_evidence_text'] = batch['raw_evidence_text']
        example['offsets'] = batch['offsets']
    else:
        example['evidence_text'] = batch['evidence_text']
    return example



def vectorize_input(batch, config, training=True, device=None):
    """
    - Vectorize question and question mask
    - Vectorize evidence documents, mask and features
    - Vectorize target representations
    """
    # Check there is at least one valid example in batch (containing targets):
    if not batch:
        return None

    # Relevant parameters:
    batch_size = len(batch['question'])

    # Initialize all relevant parameters to None:
    targets = None

    # Part 1: Question Words
    # Batch questions ( sum_bs(n_sect), len_q)
    max_q_len = max([len(q) for q in batch['question']])
    xq = torch.LongTensor(batch_size, max_q_len).fill_(0)
    xq_mask = torch.ByteTensor(batch_size, max_q_len).fill_(1)
    for i, q in enumerate(batch['question']):
        xq[i, :len(q)].copy_(torch.LongTensor(q))
        xq_mask[i, :len(q)].fill_(0)

    # Part 2: Document Words
    max_d_len = max([len(d) for d in batch['evidence']])
    xd = torch.LongTensor(batch_size, max_d_len).fill_(0)
    xd_mask = torch.ByteTensor(batch_size, max_d_len).fill_(1)
    xd_f = torch.zeros(batch_size, max_d_len, config['num_features']) if config['num_features'] > 0 else None

    # 2(a): fill up DrQA section variables
    for i, d in enumerate(batch['evidence']):
        xd[i, :len(d)].copy_(torch.LongTensor(d))
        xd_mask[i, :len(d)].fill_(0)
        if config['num_features'] > 0:
            xd_f[i, :len(d)].copy_(batch['features'][i])

    # Part 3: Target representations
    if config['sum_loss']:  # For sum_loss "targets" acts as a mask rather than indices.
        targets = torch.ByteTensor(batch_size, max_d_len, 2).fill_(0)
        for i, _targets in enumerate(batch['targets']):
            for s, e in _targets:
                targets[i, s, 0] = 1
                targets[i, e, 1] = 1
    else:
        targets = torch.LongTensor(batch_size, 2)
        for i, _target in enumerate(batch['targets']):
            targets[i][0] = _target[0]
            targets[i][1] = _target[1]

    torch.set_grad_enabled(training)
    example = {'batch_size': batch_size,
               'answers': batch['answers'],
               'xq': xq.to(device) if device else xq,
               'xq_mask': xq_mask.to(device) if device else xq_mask,
               'xd': xd.to(device) if device else xd,
               'xd_mask': xd_mask.to(device) if device else xd_mask,
               'xd_f': xd_f.to(device) if device else xd_f,
               'targets': targets.to(device) if device else targets}

    if config['predict_raw_text']:
        example['raw_evidence_text'] = batch['raw_evidence_text']
        example['offsets'] = batch['offsets']
    else:
        example['evidence_text'] = batch['evidence_text']
    return example


def featurize(question, document, feature_dict, history=None):
    doc_len = len(document['word'])
    features = torch.zeros(doc_len, len(feature_dict))
    q_cased_words = set([w for w in question['word']])
    q_uncased_words = set([w.lower() for w in question['word']])

    cased_words = {}
    uncased_words = {}
    if history is not None:
        f_cased = f_uncased = ''
        for w in history:
            if (w.startswith('<Q') or w.startswith('<A')) and w.endswith('>'):
                f_cased = 'f_{}_cased'.format(w[1:-1])
                f_uncased = 'f_{}_uncased'.format(w[1:-1])
                if f_cased in feature_dict:
                    cased_words[f_cased] = set()
                if f_uncased in feature_dict:
                    uncased_words[f_uncased] = set()
            else:
                if f_cased in cased_words:
                    cased_words[f_cased].add(w)
                if f_uncased in uncased_words:
                    uncased_words[f_uncased].add(w.lower())

    for i in range(doc_len):
        d_word = document['word'][i]
        if 'f_qem_cased' in feature_dict and d_word in q_cased_words:
            features[i][feature_dict['f_qem_cased']] = 1.0
        if 'f_qem_uncased' in feature_dict and d_word.lower() in q_uncased_words:
            features[i][feature_dict['f_qem_uncased']] = 1.0
        if 'pos' in document:
            f_pos = 'f_pos={}'.format(document['pos'][i])
            if f_pos in feature_dict:
                features[i][feature_dict[f_pos]] = 1.0
        if 'ner' in document:
            f_ner = 'f_ner={}'.format(document['ner'][i])
            if f_ner in feature_dict:
                features[i][feature_dict[f_ner]] = 1.0
        for f in cased_words:
            if d_word in cased_words[f]:
                features[i][feature_dict[f]] = 1.0
        for f in uncased_words:
            if d_word.lower() in uncased_words[f]:
                features[i][feature_dict[f]] = 1.0
    return features
