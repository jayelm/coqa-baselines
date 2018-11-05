import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import SeqAttnMatch, StackedBRNN, LinearSeqAttn, BilinearSeqAttn, SentenceHistoryAttn, QAHistoryAttn, QAHistoryAttnBilinear, DialogSeqAttnMatch, IncrSeqAttnMatch
from .layers import weighted_avg, uniform_weights, dropout


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
        a_input_size = input_w_dim
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
        else:
            raise NotImplementedError('question_merge = %s' % config['question_merge'])

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

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(qrnn_input, xq_mask)

        if self.config['q_dialog_history']:
            # Re-use question rnn to encode dialog history (concatted qs and as)
            # For each time t, we need to re-run RNN on dialog history separately
            # from 1 ... t - 1, because BiLSTMs which let time travel backwards.
            # Because of this extra computation we can use standard SeqAttnMatch,
            # not DialogSeqAttnMatch which handles time travel.
            # First q has no access to dialog history
            first_zeros = torch.zeros_like(question_hiddens[0:1], requires_grad=False)
            if self.config['q_dialog_attn'] == 'word_hidden':
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
                # FIXME: Do we need one more LSTM layer to integrate this?
                question_hiddens = torch.cat((question_hiddens, dialog_weighted_hidden_q), 2)
            elif self.config['q_dialog_attn'] == 'word_hidden_incr':
                xa_emb = self.w_embedding(ex['xa'])
                xa_mask = ex['xa_mask']
                # XXX: Reuse question RNN? Make new answer RNN? Run
                # answers independently? Run question and answer pairs
                # together? But then how to deal with augmentation?
                if self.config['answer_rnn']:
                    answer_hiddens = self.answer_rnn(xa_emb, xa_mask)
                else:
                    # Reuse question RNN.
                    answer_hiddens = self.question_rnn(xa_emb, xa_mask)
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



        if self.config['question_merge'] == 'avg':
            q_merge_weights = uniform_weights(question_hiddens, xq_mask)
        elif self.config['question_merge'] == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens.contiguous(), xq_mask)
        question_hidden = weighted_avg(question_hiddens, q_merge_weights)

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
