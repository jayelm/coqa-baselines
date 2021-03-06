import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from word_model import WordModel
from utils.eval_utils import compute_eval_metric
from utils.analysis_utils import write_attns_to_file
from models.layers import multi_nll_loss
from utils import constants as Constants
from collections import Counter
from models.drqa import DrQA


class Model(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, config, train_set=None):
        # Book-keeping.
        self.config = config
        if self.config['pretrained']:
            self.init_saved_network(self.config['pretrained'])
        else:
            assert train_set is not None
            print('Train vocab: {}'.format(len(train_set.vocab)))
            vocab = Counter()
            for w in train_set.vocab:
                if train_set.vocab[w] >= config['min_freq']:
                    vocab[w] = train_set.vocab[w]
            print('Pruned train vocab: {}'.format(len(vocab)))
            # Building network.
            word_model = WordModel(embed_size=self.config['embed_size'],
                                   filename=self.config['embed_file'],
                                   embed_type=self.config['embed_type'],
                                   top_n=self.config['top_vocab'],
                                   additional_vocab=vocab)
            self.config['embed_size'] = word_model.embed_size
            self._init_new_network(train_set, word_model)

        num_params = 0
        for name, p in self.network.named_parameters():
            print('{}: {}'.format(name, str(p.size())))
            num_params += p.numel()
        print('#Parameters = {}\n'.format(num_params))

        self._init_optimizer()

    def init_saved_network(self, saved_dir):
        _ARGUMENTS = ['rnn_padding', 'embed_size', 'hidden_size', 'num_layers', 'rnn_type',
                      'concat_rnn_layers', 'question_merge', 'use_qemb', 'f_qem', 'f_pos', 'f_ner',
                      'sum_loss', 'doc_self_attn', 'resize_rnn_input', 'span_dependency',
                      'fix_embeddings', 'dropout_rnn', 'dropout_emb', 'dropout_ff',
                      'dropout_rnn_output', 'variational_dropout', 'word_dropout']

        # Load all saved fields.
        fname = os.path.join(saved_dir, Constants._SAVED_WEIGHTS_FILE)
        print('[ Loading saved model %s ]' % fname)
        saved_params = torch.load(fname, map_location=lambda storage, loc: storage)
        self.word_dict = saved_params['word_dict']
        self.rev_word_dict = {v: k for k, v in self.word_dict.items()}
        self.feature_dict = saved_params['feature_dict']
        self.config['num_features'] = len(self.feature_dict)
        self.state_dict = saved_params['state_dict']
        for k in _ARGUMENTS:
            if saved_params['config'][k] != self.config[k]:
                print('Overwrite {}: {} -> {}'.format(k, self.config[k], saved_params['config'][k]))
                self.config[k] = saved_params['config'][k]

        w_embedding = self._init_embedding(len(self.word_dict) + 1, self.config['embed_size'])
        self.network = DrQA(self.config, w_embedding)

        # Merge the arguments
        if self.state_dict:
            merged_state_dict = self.network.state_dict()
            for k, v in self.state_dict['network'].items():
                if k in merged_state_dict:
                    merged_state_dict[k] = v
            self.network.load_state_dict(merged_state_dict)

    def _init_new_network(self, train_set, word_model):
        self.feature_dict = self._build_feature_dict(train_set)
        self.config['num_features'] = len(self.feature_dict)
        self.word_dict = word_model.get_vocab()
        self.rev_word_dict = {v: k for k, v in self.word_dict.items()}
        w_embedding = self._init_embedding(word_model.vocab_size, self.config['embed_size'],
                                           pretrained_vecs=word_model.get_word_vecs())
        self.network = DrQA(self.config, w_embedding)

    def _init_embedding(self, vocab_size, embed_size, pretrained_vecs=None):
        """Initializes the embeddings
        """
        return nn.Embedding(vocab_size, embed_size, padding_idx=0,
                            _weight=torch.from_numpy(pretrained_vecs).float()
                            if pretrained_vecs is not None else None)

    def _build_feature_dict(self, train_set):
        feature_dict = {}
        if self.config['f_qem']:
            feature_dict['f_qem_cased'] = len(feature_dict)
            feature_dict['f_qem_uncased'] = len(feature_dict)

        if self.config['f_history']:
            max_history = 20 if self.config['n_history'] < 0 else self.config['n_history']
            for i in range(1, max_history + 1):
                feature_dict['f_Q{}_cased'.format(i)] = len(feature_dict)
                feature_dict['f_Q{}_uncased'.format(i)] = len(feature_dict)
                feature_dict['f_A{}_cased'.format(i)] = len(feature_dict)
                feature_dict['f_A{}_uncased'.format(i)] = len(feature_dict)

        if self.config['f_pos']:
            pos_tags = set()
            for ex in train_set:
                for context in ex['evidence']:
                    assert 'pos' in context
                    pos_tags |= set(context['pos'])
            print('{} pos tags: {}'.format(len(pos_tags), str(pos_tags)))
            for pos in pos_tags:
                feature_dict['f_pos={}'.format(pos)] = len(feature_dict)

        if self.config['f_ner']:
                ner_tags = set()
                for ex in train_set:
                    for context in ex['evidence']:
                        assert 'ner' in context
                        ner_tags |= set(context['ner'])
                print('{} ner tags: {}'.format(len(ner_tags), str(ner_tags)))
                for ner in ner_tags:
                    feature_dict['f_ner={}'.format(ner)] = len(feature_dict)

        print('# features: {}'.format(len(feature_dict)))
        return feature_dict

    def _init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, self.config['learning_rate'],
                                       momentum=self.config['momentum'],
                                       weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.config['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.config['optimizer'])

    def predict(self, exs, update=True, out_predictions=False, out_attentions=None):
        # Train/Eval mode
        self.network.train(update)
        running_loss_items = []
        running_losses = []
        f1s = []
        ems = []
        f1s_all = []
        ems_all = []
        all_predictions = []
        all_spans = []
        all_ids = []
        answers = []
        ex_sizes = []
        for ex in exs:
            # Convey to the attn alyers that we want attention outputs by
            # setting examples in ex
            ex['out_attentions'] = out_attentions

            # Run forward
            res = self.network(ex)
            score_s, score_e = res['score_s'], res['score_e']

            # Loss cannot be computed for test-time as we may not have targets
            if update:
                # Compute loss and accuracies
                loss = self.compute_span_loss(score_s, score_e, res['targets'])
                running_losses.append(loss)
                running_loss_items.append(loss.item())

            if (not update) or self.config['predict_train']:
                predictions, spans = self.extract_predictions(ex, score_s, score_e)
                this_f1, this_em, this_f1s, this_ems = self.evaluate_predictions(predictions, ex['answers'])
                f1s.append(this_f1)
                ems.append(this_em)
                f1s_all.append(this_f1s)
                ems_all.append(this_ems)
                answers.append(ex['answers'])
                ex_sizes.append(ex['batch_size'])
                if out_predictions:
                    all_predictions.append(predictions)
                    all_spans.append(spans)
                    all_ids.append([ex['id'] for _ in range(len(predictions))])

                if out_attentions:
                    # Save attentions to file.
                    fdir = os.path.join(self.config['pretrained'], 'attention')
                    write_attns_to_file(ex, self.config, res['out_attentions'], fdir, self.rev_word_dict)

        output = {
            # Do a weighted average of f1
            'f1': weighted_score_avg(f1s, ex_sizes),
            'em': weighted_score_avg(ems, ex_sizes),
            'f1s': [item for sublist in f1s_all for item in sublist],
            'ems': [item for sublist in ems_all for item in sublist],
            'loss': sum(running_loss_items) / (2 * sum(ex_sizes))
        }

        if (not update) or self.config['predict_train']:
            if out_predictions:
                output['predictions'] = [item for sublist in all_predictions for item in sublist]
                output['spans'] = [item for sublist in all_spans for item in sublist]
                output['ids'] = [item for sublist in all_ids for item in sublist]
                output['answers'] = [item for sublist in answers for item in sublist]

        if update:
            # Clear gradients and run backward
            self.optimizer.zero_grad()
            # Batch size is 2x since we have start AND end spans
            weighted_avg_loss = sum(running_losses) / (2 * sum(ex_sizes))
            weighted_avg_loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config['grad_clipping'])

            # Update parameters
            self.optimizer.step()

        return output

    def compute_span_loss(self, score_s, score_e, targets):
        assert targets.size(0) == score_s.size(0) == score_e.size(0)
        if self.config['sum_loss']:
            raise NotImplementedError("Make sure variable batch sizes works")
            loss = multi_nll_loss(score_s, targets[:, :, 0]) + multi_nll_loss(score_e, targets[:, :, 1])
        else:
            loss = F.nll_loss(score_s, targets[:, 0], reduction='sum') + F.nll_loss(score_e, targets[:, 1], reduction='sum')
        return loss

    def extract_predictions(self, ex, score_s, score_e):
        # Transfer to CPU/normal tensors for numpy ops (and convert log probabilities to probabilities)
        score_s = score_s.exp().squeeze()
        score_e = score_e.exp().squeeze()
        if len(score_s.shape) == 1:  # Account for batch size 1
            score_s = score_s.unsqueeze(0)
            score_e = score_e.unsqueeze(0)

        predictions = []
        spans = []
        for i, (_s, _e) in enumerate(zip(score_s, score_e)):
            if self.config['predict_raw_text']:
                prediction, span = self._scores_to_raw_text(ex['raw_evidence_text'][i],
                                                            ex['offsets'][i], _s, _e)
            else:
                prediction, span = self._scores_to_text(ex['evidence_text'][i], _s, _e)
            predictions.append(prediction)
            spans.append(span)
        return predictions, spans

    def _scores_to_text(self, text, score_s, score_e):
        max_len = self.config['max_answer_len'] or score_s.size(1)
        scores = torch.ger(score_s.squeeze(), score_e.squeeze())
        scores.triu_().tril_(max_len - 1)
        scores = scores.cpu().detach().numpy()
        s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
        return ' '.join(text[s_idx: e_idx + 1]), (int(s_idx), int(e_idx))

    def _scores_to_raw_text(self, raw_text, offsets, score_s, score_e):
        max_len = self.config['max_answer_len'] or score_s.size(1)
        scores = torch.ger(score_s.squeeze(), score_e.squeeze())
        scores.triu_().tril_(max_len - 1)
        scores = scores.cpu().detach().numpy()
        s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
        return raw_text[offsets[s_idx][0]: offsets[e_idx][1]], (offsets[s_idx][0], offsets[e_idx][1])

    def evaluate_predictions(self, predictions, answers):
        f1_score, f1s = compute_eval_metric('f1', predictions, answers)
        em_score, ems = compute_eval_metric('em', predictions, answers)
        return f1_score, em_score, f1s, ems

    def save(self, dirname):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
            },
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'config': self.config,
            'dir': dirname,
        }
        try:
            torch.save(params, os.path.join(dirname, Constants._SAVED_WEIGHTS_FILE))
        except BaseException:
            print('[ WARN: Saving failed... continuing anyway. ]')

def weighted_score_avg(scores, weights):
    total_weight = sum(weights)
    weighted_scores = [s * w for s, w in zip(scores, weights)]
    return sum(weighted_scores) / total_weight
