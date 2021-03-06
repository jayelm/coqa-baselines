import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from .layers import SeqAttnMatch, StackedBRNN, LinearSeqAttn, BilinearSeqAttn
# Dialog attention
from .layers import DialogSeqAttnMatch, IncrSeqAttnMatch
from .layers import weighted_avg, uniform_weights, dropout, onehot_markers


class DrQA(nn.Module):
    """Network for the Document Reader module of DrQA."""
    _RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, config, w_embedding):
        """Configuration, word embeddings"""
        super(DrQA, self).__init__()
        # Store config
        self.config = config
        self.w_embedding = w_embedding
        input_w_dim = self.w_embedding.embedding_dim
        q_input_size = input_w_dim
        if self.config['q_dialog_history'] and self.config['q_dialog_attn'] == 'word_emb':
            q_input_size += input_w_dim
        if self.config['qa_emb_markers']:
            q_input_size += 2

        if self.config['fix_embeddings']:
            for p in self.w_embedding.parameters():
                p.requires_grad = False

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * config['hidden_size']
        question_hidden_size = 2 * config['hidden_size']

        if config['concat_rnn_layers']:
            doc_hidden_size *= config['num_layers']
            question_hidden_size *= config['num_layers']

        if config['doc_self_attn']:
            self.doc_self_attn = SeqAttnMatch(doc_hidden_size)
            doc_hidden_size = doc_hidden_size + question_hidden_size

        if self.config['q_dialog_history'] and self.config['q_dialog_attn'] == 'word_hidden':
            # Then question hidden reprs are augmented with attention weighted
            # average of dialog hidden reprs (same encoder, same dimensionality)
            question_hidden_size = 2 * question_hidden_size

        # USE ANSWER: whether we need to use raw answer embeddings
        self.use_answer = any(config[cfg] for cfg in ['doc_dialog_history', 'q_dialog_history'])

        # Projection for attention weighted question
        if self.config['use_qemb']:
            self.qemb_match = SeqAttnMatch(input_w_dim)

        # Dialog-history-specific module for matching dialog history to question tokens
        if self.config['q_dialog_history']:
            if self.config['q_dialog_attn'] == 'word_emb':
                if self.config['max_history'] > 0:
                    raise NotImplementedError
                self.q_dialog_match = DialogSeqAttnMatch(input_w_dim,
                                                         recency_bias=self.config['recency_bias'],
                                                         cuda=self.config['cuda'],
                                                         answer_marker_features=self.config['history_dialog_answer_f'],
                                                         time_features=self.config['history_dialog_time_f'])
            elif self.config['q_dialog_attn'] == 'word_hidden':
                # Use standard seqattnmatch because we have to rerun over all
                # of dialog history (repeatedly), so we don't need to worry about time travel
                # We work with hidden representations, not raw embeddings, so
                # input size is adjusted for that
                # TODO: Enable answer marker features
                # q hidden size already includes doc
                if self.config['max_history'] > 0:
                    raise NotImplementedError
                self.q_dialog_match = SeqAttnMatch(question_hidden_size // 2,
                                                   recency_bias=self.config['recency_bias'])
            elif self.config['q_dialog_attn'] == 'word_hidden_incr':
                self.q_dialog_match = IncrSeqAttnMatch(
                    question_hidden_size,
                    recency_bias=self.config['recency_bias'],
                    merge_type=self.config['q_dialog_attn_incr_merge'],
                    max_history=self.config['max_history'],
                    cuda=self.config['cuda'],
                    scoring=self.config['q_dialog_attn_scoring'],
                    mask_answers=self.config['mask_answers'],
                    attend_answers=self.config['attend_answers'],
                    answer_marker_features=self.config['history_dialog_answer_f'],
                    hidden_size=self.config['attn_hidden_size'],
                )
            else:
                raise NotImplementedError("q_dialog_attn = {}".format(self.config['q_dialog_attn']))

        # Dialog-history-specific module for matching dialog history to document tokens
        if self.config['doc_dialog_history']:
            if self.config['doc_dialog_attn'] == 'q':
                self.doc_dialog_match = self.q_dialog_match
            elif self.config['doc_dialog_attn'] == 'word_emb':
                # May be advantageous to have separate attention mechanisms, as
                # a question-based one is probably more concerned with resolving coref.
                if self.config['max_history'] > 0:
                    raise NotImplementedError
                self.doc_dialog_match = DialogSeqAttnMatch(input_w_dim,
                                                           recency_bias=self.config['recency_bias'],
                                                           cuda=self.config['cuda'],
                                                           answer_marker_features=self.config['history_dialog_answer_f'],
                                                           time_features=self.config['history_dialog_time_f'])
            elif self.config['doc_dialog_attn'] == 'word_hidden':
                # TODO: What do we do here - concat doc hiddens?
                raise NotImplementedError("doc_dialog_attn = word_hidden")
            else:
                raise NotImplementedError("doc_dialog_attn = {}".format(self.config['doc_dialog_attn']))

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = input_w_dim + self.config['num_features']
        if self.config['use_qemb']:
            doc_input_size += input_w_dim

        # Project document and question to the same size as their encoders
        if self.config['resize_rnn_input']:
            self.doc_linear = nn.Linear(doc_input_size, config['hidden_size'], bias=True)
            self.q_linear = nn.Linear(input_w_dim, config['hidden_size'], bias=True)
            doc_input_size = q_input_size = config['hidden_size']

        # RNN document encoder
        self.doc_rnn = StackedBRNN(
            input_size=doc_input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout_rate=config['dropout_rnn'],
            dropout_output=config['dropout_rnn_output'],
            variational_dropout=config['variational_dropout'],
            concat_layers=config['concat_rnn_layers'],
            rnn_type=self._RNN_TYPES[config['rnn_type']],
            padding=config['rnn_padding'],
            bidirectional=True,
        )

        # RNN question encoder
        if self.config['q_dialog_history'] and self.config['q_dialog_attn'] == 'word_hidden_incr':
            # 2 separate RNNs, one for forward direction
            self.dialog_rnn_forward = StackedBRNN(
                input_size=q_input_size,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout_rate=config['dropout_rnn'],
                dropout_output=config['dropout_rnn_output'],
                variational_dropout=config['variational_dropout'],
                concat_layers=config['concat_rnn_layers'],
                rnn_type=self._RNN_TYPES[config['rnn_type']],
                padding=config['rnn_padding'],
                bidirectional=False,
            )
            self.dialog_rnn_backward = StackedBRNN(
                input_size=q_input_size,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout_rate=config['dropout_rnn'],
                dropout_output=config['dropout_rnn_output'],
                variational_dropout=config['variational_dropout'],
                concat_layers=config['concat_rnn_layers'],
                rnn_type=self._RNN_TYPES[config['rnn_type']],
                padding=config['rnn_padding'],
                bidirectional=False,
            )
        else:
            self.question_rnn = StackedBRNN(
                input_size=q_input_size,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout_rate=config['dropout_rnn'],
                dropout_output=config['dropout_rnn_output'],
                variational_dropout=config['variational_dropout'],
                concat_layers=config['concat_rnn_layers'],
                rnn_type=self._RNN_TYPES[config['rnn_type']],
                padding=config['rnn_padding'],
                bidirectional=True,
            )

        if config['answer_rnn']:
            self.answer_rnn = StackedBRNN(
                input_size=q_input_size,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout_rate=config['dropout_rnn'],
                dropout_output=config['dropout_rnn_output'],
                variational_dropout=config['variational_dropout'],
                concat_layers=config['concat_rnn_layers'],
                rnn_type=self._RNN_TYPES[config['rnn_type']],
                padding=config['rnn_padding'],
                bidirectional=True,
            )

        # Question merging
        if config['question_merge'] == 'self_attn':
            self.self_attn = LinearSeqAttn(question_hidden_size)

        # Question history re-encoding
        if config['qhier_rnn']:
            self.qhier_rnn = StackedBRNN(
                input_size=question_hidden_size,
                hidden_size=question_hidden_size,
                num_layers=1,
                dropout_rate=config['dropout_rnn'],
                dropout_output=config['dropout_rnn_output'],
                variational_dropout=config['variational_dropout'],
                concat_layers=False,
                rnn_type=self._RNN_TYPES[config['rnn_type']],
                padding=config['rnn_padding'],
                bidirectional=False,
            )

        # Bilinear attention for span start/end
        self.start_attn = BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
        )
        q_rep_size = question_hidden_size + doc_hidden_size if config['span_dependency'] else question_hidden_size
        self.end_attn = BilinearSeqAttn(
            doc_hidden_size,
            q_rep_size,
        )

    def forward(self, ex):
        """Inputs:
        xq = question word indices             (batch, max_q_len)
        xq_mask = question padding mask        (batch, max_q_len)
        xd = document word indices             (batch, max_d_len)
        xd_f = document word features indices  (batch, max_d_len, nfeat)
        xd_mask = document padding mask        (batch, max_d_len)
        targets = span targets                 (batch,)
        """
        # ==== EMBED ====
        # Embed both document and question
        xq_emb = self.w_embedding(ex['xq'])                         # (batch, max_q_len, word_embed)
        xd_emb = self.w_embedding(ex['xd'])                         # (batch, max_d_len, word_embed)
        if self.use_answer:
            xa_emb = self.w_embedding(ex['xa'])

        shared_axes = [2] if self.config['word_dropout'] else []
        xq_emb = dropout(xq_emb, self.config['dropout_emb'], shared_axes=shared_axes, training=self.training)
        xd_emb = dropout(xd_emb, self.config['dropout_emb'], shared_axes=shared_axes, training=self.training)
        if self.use_answer:
            xa_emb = dropout(xa_emb, self.config['dropout_emb'], shared_axes=shared_axes, training=self.training)
        xd_mask = ex['xd_mask']
        xq_mask = ex['xq_mask']
        if self.use_answer:
            xa_mask = ex['xa_mask']

        # Save attentions
        if ex['out_attentions']:
            out_attentions = {}

        # ==== QUESTION ====
        qrnn_input = xq_emb
        if self.config['qa_emb_markers']:
            if not (self.config['q_dialog_history'] and self.config['q_dialog_attn'] == 'word_hidden_incr'):
                raise NotImplementedError
            markers = onehot_markers(qrnn_input, 2, 0, cuda=self.config['cuda'])
            qrnn_input = torch.cat((qrnn_input, markers), 2)

        # Augment question RNN input with attention over dialog history
        if self.config['q_dialog_history'] and self.config['q_dialog_attn'] == 'word_emb':
            if ex['out_attentions'] and 'q_dialog_attn' in ex['out_attentions']:
                # Set flag to retrieve attentions, pass to attention layer
                xdialog_weighted_emb_q, q_dialog_attn = self.q_dialog_match(xq_emb,
                                                             xq_emb, xa_emb,
                                                             xq_mask, xa_mask,
                                                             out_attention=True)
                out_attentions['q_dialog_attn'] = q_dialog_attn
            else:
                xdialog_weighted_emb_q = self.q_dialog_match(xq_emb,
                                                             xq_emb, xa_emb,
                                                             xq_mask, xa_mask)
            qrnn_input = torch.cat((qrnn_input, xdialog_weighted_emb_q), 2)

        if self.config['q_dialog_history']:
            if self.config['q_dialog_attn'] == 'word_hidden':
                question_hiddens = self.question_rnn(qrnn_input, xq_mask)

                # First q has no access to dialog history
                first_zeros = torch.zeros_like(question_hiddens[0:1], requires_grad=False)

                if question_hiddens.shape[0] == 1:
                    # That's all you get
                    dialog_weighted_hidden_q = first_zeros
                    if ex['out_attentions'] and 'q_dialog_attn' in ex['out_attentions']:
                        out_attentions['q_dialog_attn'] = None
                else:
                    # Embed dialog, chopping off first timestep
                    xdialog_emb = self.w_embedding(ex['xdialog'][1:])
                    xdialog_mask = ex['xdialog_mask'][1:]
                    dialog_recency_weights = ex['dialog_recency_weights'][1:]

                    dialog_hiddens = self.question_rnn(xdialog_emb, xdialog_mask)

                    if ex['out_attentions'] and 'q_dialog_attn' in ex['out_attentions']:
                        dialog_weighted_hidden_q, q_dialog_attn = self.q_dialog_match(
                            question_hiddens[1:], dialog_hiddens, xdialog_mask,
                            recency_weights=dialog_recency_weights if self.config['recency_bias'] else None,
                            out_attention=True)
                        # Concat first zeros.
                        q_dialog_attn = torch.cat((torch.zeros_like(q_dialog_attn[0:1]), q_dialog_attn), 0)
                        out_attentions['q_dialog_attn'] = q_dialog_attn
                    else:
                        dialog_weighted_hidden_q = self.q_dialog_match(
                            question_hiddens[1:], dialog_hiddens, xdialog_mask,
                            recency_weights=dialog_recency_weights if self.config['recency_bias'] else None
                        )
                # First q has no access to dialog history
                dialog_weighted_hidden_q = torch.cat((first_zeros, dialog_weighted_hidden_q), 0)
                # Concat weighted hidden reprs with question hidden reprs.
                question_hiddens = torch.cat((question_hiddens, dialog_weighted_hidden_q), 2)
            elif self.config['q_dialog_attn'] == 'word_hidden_incr':
                if self.config['answer_rnn']:
                    raise NotImplementedError
                # Load answers
                xa_emb = self.w_embedding(ex['xa'])
                xa_mask = ex['xa_mask']

                if self.config['qa_emb_markers']:
                    markers = onehot_markers(xa_emb, 2, 1, cuda=self.config['cuda'])
                    xa_emb = torch.cat((xa_emb, markers), 2)

                # Reverse sequences
                xq_lengths = torch.sum(1 - xq_mask, 1)
                xq_emb_b = reverse_padded_sequence(qrnn_input, xq_lengths, batch_first=True)
                xa_lengths = torch.sum(1 - xa_mask, 1)
                xa_emb_b = reverse_padded_sequence(xa_emb, xa_lengths, batch_first=True)

                # Learn backwards direction embeddings
                question_hiddens_b = self.dialog_rnn_backward(xq_emb_b, xq_mask)
                answer_hiddens_b = self.dialog_rnn_backward(xa_emb_b, xa_mask)

                xdialog_emb = self.w_embedding(ex['xdialog_full'])
                xdialog_mask = ex['xdialog_full_mask']
                if self.config['qa_emb_markers']:
                    xdialog_emb = add_qa_emb_markers(xdialog_emb, xq_lengths, xa_lengths,
                                                     cuda=self.config['cuda'])

                # Learn forwards direction embeddings
                dialog_hiddens_f = self.dialog_rnn_forward(xdialog_emb, xdialog_mask)
                dialog_hiddens_f = dialog_hiddens_f.squeeze(0)
                question_hiddens_f, answer_hiddens_f = extract_qa_hiddens(dialog_hiddens_f,
                                                                          xq_lengths,
                                                                          xa_lengths)

                question_hiddens = torch.cat((question_hiddens_f, question_hiddens_b), 2)
                answer_hiddens = torch.cat((answer_hiddens_f, answer_hiddens_b), 2)

                if ex['out_attentions'] and 'q_dialog_attn' in ex['out_attentions']:
                    question_hiddens, q_dialog_attn = self.q_dialog_match(
                        question_hiddens, answer_hiddens, xq_mask, xa_mask,
                        out_attention=True
                    )
                    out_attentions['q_dialog_attn'] = q_dialog_attn
                else:
                    # This module completely replaces existing question hiddens.
                    question_hiddens = self.q_dialog_match(
                        question_hiddens, answer_hiddens, xq_mask, xa_mask,
                        out_attention=False
                    )
            else:
                raise NotImplementedError
        else:
            question_hiddens = self.question_rnn(qrnn_input, xq_mask)



        if self.config['question_merge'] == 'avg':
            q_merge_weights = uniform_weights(question_hiddens, xq_mask)
        elif self.config['question_merge'] == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens.contiguous(), xq_mask)
        if ex['out_attentions'] and 'q_dialog_attn' in ex['out_attentions']:
            # Also prepend merge weights (why not?)
            q_merge_weights_f = q_merge_weights.unsqueeze(2)
            new_attn = torch.cat((q_merge_weights_f, out_attentions['q_dialog_attn']), 2)
            out_attentions['q_dialog_attn'] = new_attn
        question_hidden = weighted_avg(question_hiddens, q_merge_weights)

        if self.config['qhier_rnn']:
            # Re-encode question vectors with forward LSTM
            qh_mask = torch.zeros(1, question_hidden.shape[0], requires_grad=False, dtype=torch.uint8)
            if self.config['cuda']:
                qh_mask = qh_mask.cuda()
            question_hidden = self.qhier_rnn(question_hidden.unsqueeze(0), qh_mask).squeeze(0)

        # ==== DOCUMENT ====
        # Add attention-weighted question representation
        if self.config['use_qemb']:
            xq_weighted_emb = self.qemb_match(xd_emb, xq_emb, xq_mask)
            drnn_input = torch.cat([xd_emb, xq_weighted_emb], 2)
        else:
            drnn_input = xd_emb

        # Augment document RNN input with attention over dialog history
        if self.config['doc_dialog_history']:
            xdialog_weighted_emb_d = self.doc_dialog_match(xd_emb,
                                                           xq_emb, xa_emb,
                                                           xq_mask, xa_mask)
            drnn_input = torch.cat((drnn_input, xdialog_weighted_emb_d), 2)


        if self.config["num_features"] > 0:
            drnn_input = torch.cat([drnn_input, ex['xd_f']], 2)

        # Project document and question to the same size as their encoders
        if self.config['resize_rnn_input']:
            drnn_input = F.relu(self.doc_linear(drnn_input))
            xq_emb = F.relu(self.q_linear(xq_emb))
            if self.config['dropout_ff'] > 0:
                drnn_input = F.dropout(drnn_input, training=self.training)
                xq_emb = F.dropout(xq_emb, training=self.training)

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(drnn_input, xd_mask)       # (batch, max_d_len, hidden_size)

        # Document self attention
        if self.config['doc_self_attn']:
            xd_weighted_emb = self.doc_self_attn(doc_hiddens, doc_hiddens, xd_mask)
            doc_hiddens = torch.cat([doc_hiddens, xd_weighted_emb], 2)

        # Predict start and end positions
        start_scores = self.start_attn(doc_hiddens, question_hidden, xd_mask)
        if self.config['span_dependency']:
            question_hidden = torch.cat([question_hidden, (doc_hiddens * start_scores.exp().unsqueeze(2)).sum(1)], 1)
        end_scores = self.end_attn(doc_hiddens, question_hidden, xd_mask)

        out = {'score_s': start_scores,
               'score_e': end_scores,
               'targets': ex['targets']}
        if ex['out_attentions']:
            # Convert to numpy.
            #  out_attentions = {k: v.detach().cpu().numpy() for k, v in out_attentions.items()}
            out['out_attentions'] = out_attentions
        return out


def reverse_padded_sequence(inputs, lengths, batch_first=False):
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Variable with the same size as inputs, but with each sequence
        reversed according to its length.
    """
    if batch_first:
        inputs = inputs.transpose(0, 1)
    max_length, batch_size = inputs.size(0), inputs.size(1)
    if len(lengths) != batch_size:
        raise ValueError('inputs is incompatible with lengths.')
    ind = [list(reversed(range(0, length))) + list(range(length, max_length))
           for length in lengths]
    ind = Variable(torch.LongTensor(ind).transpose(0, 1))
    for dim in range(2, inputs.dim()):
        ind = ind.unsqueeze(dim)
    ind = ind.expand_as(inputs)
    if inputs.is_cuda:
        ind = ind.cuda(inputs.get_device())
    reversed_inputs = torch.gather(inputs, 0, ind)
    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs


def extract_qa_hiddens(dialog_hiddens, xq_lengths, xa_lengths):
    """
    Given concatted dialog sequence, split up into question/answer embeddings
    of the same shape as xq_mask and xa_mask.
    """
    dialog_i = 0
    q_seqs = []
    a_seqs = []
    for q_len, a_len in zip(xq_lengths, xa_lengths):
        q = dialog_hiddens[dialog_i:dialog_i+q_len]
        dialog_i += q_len
        q_seqs.append(q)

        a = dialog_hiddens[dialog_i:dialog_i+a_len]
        dialog_i += a_len
        a_seqs.append(a)

    q_hiddens = pad_sequence(q_seqs, batch_first=True)
    a_hiddens = pad_sequence(a_seqs, batch_first=True)

    return q_hiddens, a_hiddens


def add_qa_emb_markers(xdialog_emb, xq_lengths, xa_lengths, cuda=False):
    """
    Add one-hot question answer features to the dialog embeddings.
    """
    features_np = np.zeros(xdialog_emb.shape[:2] + (2, ), dtype=np.float32)
    dialog_i = 0
    for q, a in zip(xq_lengths, xa_lengths):
        features_np[0, dialog_i:dialog_i+q, 0] = 1
        dialog_i += q
        features_np[0, dialog_i:dialog_i+a, 1] = 1
        dialog_i += a
    features = torch.tensor(features_np)
    if cuda:
        features = features.cuda()
    return torch.cat((xdialog_emb, features), 2)
