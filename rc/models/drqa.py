import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import SeqAttnMatch, StackedBRNN, LinearSeqAttn, BilinearSeqAttn, SentenceHistoryAttn, QAHistoryAttn, QAHistoryAttnBilinear, DialogSeqAttnMatch
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
        if self.config['q_dialog_history']:
            q_input_size += input_w_dim
        a_input_size = input_w_dim
        if self.config['fix_embeddings']:
            for p in self.w_embedding.parameters():
                p.requires_grad = False

        # USE ANSWER: whether we need to use raw answer embeddings
        self.use_answer = any(config[cfg] for cfg in ['doc_dialog_history', 'q_dialog_history'])
        # ENCODE ANSWER: whether we actually need to encode the answer
        # embeddings with a BiLSTM.
        self.encode_answer = any(config[cfg + '_attn'] in ('qa_sentence', 'qa_sentence_bi') and config['use_history_' + cfg] for cfg in
                                 ['qhidden', 'qemb', 'aemb'])

        # Projection for attention weighted question
        if self.config['use_qemb']:
            self.qemb_match = SeqAttnMatch(input_w_dim)
        if self.encode_answer:
            self.aemb_match = SeqAttnMatch(input_w_dim)
        if self.config['doc_dialog_history']:
            self.doc_dialog_match = DialogSeqAttnMatch(input_w_dim,
                                                       recency_bias=self.config['recency_bias'],
                                                       cuda=self.config['cuda'],
                                                       answer_marker_features=self.config['history_dialog_answer_f'],
                                                       time_features=self.config['history_dialog_time_f'])
        if self.config['q_dialog_history']:
            if self.config['q_dialog_attn'] == 'doc':
                self.q_dialog_match = self.doc_dialog_match
            elif self.config['q_dialog_attn'] == 'word':
                # May be advantageous to have separate attention mechanisms, as
                # a question-based one is probably more about resolving coref.
                self.q_dialog_match = DialogSeqAttnMatch(input_w_dim,
                                                         recency_bias=self.config['recency_bias'],
                                                         cuda=self.config['cuda'],
                                                         answer_marker_features=self.config['history_dialog_answer_f'],
                                                         time_features=self.config['history_dialog_time_f'])
            else:
                raise NotImplementedError("q_dialog_attn = {}".format(self.config['q_dialog_attn']))

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = input_w_dim + self.config['num_features']
        if self.config['use_qemb']:
            doc_input_size += input_w_dim
            if self.config['use_history_qemb']:
                # Additional historical question features
                # FIXME: I'm computing attention weights between question
                # ENCODINGS (i.e. through stacked BiLSTM) but adding those to
                # xq_weighted_emb which is just soft alignments!
                doc_input_size += input_w_dim
            if self.config['use_history_aemb']:
                # Additional historical answer features
                # FIXME: Same here!
                doc_input_size += input_w_dim
            if self.config['doc_dialog_history']:
                # Additional historical dialog features
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

        # RNN answer encoder
        if self.encode_answer:
            # FIXME: Isn't this answer encoder overparameterized?
            self.answer_rnn = StackedBRNN(
                input_size=a_input_size,
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

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * config['hidden_size']
        question_hidden_size = 2 * config['hidden_size']
        if self.encode_answer:
            answer_hidden_size = 2 * config['hidden_size']
        if config['concat_rnn_layers']:
            doc_hidden_size *= config['num_layers']
            question_hidden_size *= config['num_layers']
            if self.encode_answer:
                answer_hidden_size *= config['num_layers']

        if config['doc_self_attn']:
            self.doc_self_attn = SeqAttnMatch(doc_hidden_size)
            doc_hidden_size = doc_hidden_size + question_hidden_size

        # Question merging
        if config['question_merge'] == 'self_attn':
            self.self_attn = LinearSeqAttn(question_hidden_size)
        else:
            raise NotImplementedError('question_merge = %s' % config['question_merge'])

        # Answer merging
        if self.encode_answer:
            if config['answer_merge'] == 'self_attn':
                self.answer_self_attn = LinearSeqAttn(answer_hidden_size)


        # For qa sentence attention, we concatenate question hidden and answer hidden
        if self.encode_answer:
            qa_hidden_size = question_hidden_size + answer_hidden_size

        # Attention over question history
        # XXX: Do you need a linear layer first? Or just sum of values
        if self.config['use_history_qhidden']:
            if self.config['qhidden_attn'] == 'q_sentence':
                self.qhidden_history_attn = SentenceHistoryAttn(question_hidden_size,
                                                                cuda=config['cuda'],
                                                                recency_bias=config['recency_bias'],
                                                                use_current_timestep=config['use_current_timestep'])
            elif self.config['qhidden_attn'] == 'qa_sentence':
                self.qhidden_history_attn = QAHistoryAttn(qa_hidden_size, question_hidden_size,
                                                          hidden_size=None,  # Map to question_hidden_size
                                                          cuda=config['cuda'],
                                                          recency_bias=config['recency_bias'])
            else:
                raise NotImplementedError

        if self.config['use_history_qemb']:
            if self.config['qemb_attn'] == 'q_sentence':
                self.qemb_history_attn = SentenceHistoryAttn(question_hidden_size,
                                                             cuda=config['cuda'],
                                                             recency_bias=config['recency_bias'],
                                                             use_current_timestep=config['use_current_timestep'])
            elif self.config['qemb_attn'] == 'qa_sentence':
                self.qemb_history_attn = QAHistoryAttn(qa_hidden_size, question_hidden_size,
                                                       hidden_size=None,  # Map to question_hidden_size
                                                       cuda=config['cuda'],
                                                       recency_bias=config['recency_bias'])
            elif self.config['qemb_attn'] == 'qa_sentence_bi':
                self.qemb_history_attn = QAHistoryAttnBilinear(qa_hidden_size, question_hidden_size,
                                                               cuda=config['cuda'],
                                                               recency_bias=config['recency_bias'])
            elif self.config['qemb_attn'] == 'qhidden':
                pass  # Just share weights with qhidden attention
            else:
                raise NotImplementedError

        if self.config['use_history_aemb']:
            if self.config['aemb_attn'] == 'q_sentence':
                self.aemb_history_attn = SentenceHistoryAttn(question_hidden_size,
                                                             cuda=config['cuda'],
                                                             recency_bias=config['recency_bias'],
                                                             use_current_timestep=config['use_current_timestep'])
            elif self.config['aemb_attn'] == 'qa_sentence':
                self.aemb_history_attn = QAHistoryAttn(qa_hidden_size, question_hidden_size,
                                                       hidden_size=None,  # Map to question_hidden_size
                                                       cuda=config['cuda'],
                                                       recency_bias=config['recency_bias'])
            elif self.config['aemb_attn'] == 'qa_sentence_bi':
                self.aemb_history_attn = QAHistoryAttnBilinear(qa_hidden_size, question_hidden_size,
                                                               cuda=config['cuda'],
                                                               recency_bias=config['recency_bias'])
            elif self.config['aemb_attn'] == 'qhidden':
                pass  # Just share weights with qhidden attention
            elif self.config['aemb_attn'] == 'qemb':
                pass  # Just share weights with qemb attention
            else:
                raise NotImplementedError

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

        qrnn_input = xq_emb

        if self.config['q_dialog_history']:
            xdialog_weighted_emb_q = self.q_dialog_match(xq_emb,
                                                         xq_emb, xa_emb,
                                                         xq_mask, xa_mask)
            qrnn_input = torch.cat((qrnn_input, xdialog_weighted_emb_q), 2)

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(qrnn_input, xq_mask)
        if self.config['question_merge'] == 'avg':
            q_merge_weights = uniform_weights(question_hiddens, xq_mask)
        elif self.config['question_merge'] == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens.contiguous(), xq_mask)
        question_hidden = weighted_avg(question_hiddens, q_merge_weights)

        # Encode answer with RNN + merge hiddens
        if self.encode_answer:
            answer_hiddens = self.answer_rnn(xa_emb, xa_mask)
            if self.config['answer_merge'] == 'avg':
                a_merge_weights = uniform_weights(answer_hiddens, xa_mask)
            elif self.config['answer_merge'] == 'self_attn':
                a_merge_weights = self.answer_self_attn(answer_hiddens.contiguous(), xa_mask)
            answer_hidden = weighted_avg(answer_hiddens, a_merge_weights)
            # Concat answer + question
            qa_hidden = torch.cat([question_hidden, answer_hidden], 1)

        if self.config['use_history_qhidden']:
            # Compute attention between current and historical question reprs
            if self.config['qhidden_attn'] == 'q_sentence':
                qhidden_history_merge_weights = self.qhidden_history_attn(question_hidden)
            elif self.config['qhidden_attn'] == 'qa_sentence':
                qhidden_history_merge_weights = self.qhidden_history_attn(qa_hidden, question_hidden)
            else:
                raise NotImplementedError
            # TODO: This uses individual question vectors, not past historically
            # influenced question vectors
            # Augment question with attention
            # XXX: When augmenting do you do this BEFORE or AFTER QEMB/AEMB if
            # those attn mechanisms reuse question_hidden?
            question_hidden = question_hidden + qhidden_history_merge_weights.mm(question_hidden)

        # Add attention-weighted question representation
        if self.config['use_qemb']:
            xq_weighted_emb = self.qemb_match(xd_emb, xq_emb, xq_mask)
            drnn_input = torch.cat([xd_emb, xq_weighted_emb], 2)
            if self.config['use_history_qemb']:
                # Compute aligned question features over historical
                # context as averaging over past question alignments, weighted
                # by the historical question weights found earlier.
                if self.config['qemb_attn'] == 'qhidden':
                    qemb_history_merge_weights = qhidden_history_merge_weights
                elif self.config['qemb_attn'] == 'q_sentence':
                    qemb_history_merge_weights = self.qemb_history_attn(question_hidden)
                elif self.config['qemb_attn'] == 'qa_sentence':
                    qemb_history_merge_weights = self.qemb_history_attn(qa_hidden, question_hidden)
                elif self.config['qemb_attn'] == 'qa_sentence_bi':
                    qemb_history_merge_weights = self.qemb_history_attn(qa_hidden, question_hidden)
                else:
                    raise NotImplementedError
                xq_history_weighted_emb = torch.einsum(
                    'ij,jkh->ikh',
                    (qemb_history_merge_weights, xq_weighted_emb)
                )
                # XXX: Are zeros in the input ok (if use_current_timestep = False)?
                drnn_input = torch.cat([drnn_input, xq_history_weighted_emb], 2)
        else:
            drnn_input = xd_emb

        if self.config['use_history_aemb']:
            xa_weighted_emb = self.aemb_match(xd_emb, xa_emb, xa_mask)
            if self.config['aemb_attn'] == 'qhidden':
                aemb_history_merge_weights = qhidden_history_merge_weights
            elif self.config['aemb_attn'] == 'qemb':
                aemb_history_merge_weights = qemb_history_merge_weights
            elif self.config['aemb_attn'] == 'q_sentence':
                aemb_history_merge_weights = self.aemb_history_attn(question_hidden)
            elif self.config['aemb_attn'] == 'qa_sentence':
                aemb_history_merge_weights = self.aemb_history_attn(qa_hidden, question_hidden)
            elif self.config['aemb_attn'] == 'qa_sentence_bi':
                aemb_history_merge_weights = self.aemb_history_attn(qa_hidden, question_hidden)

            xa_history_weighted_emb = torch.einsum(
                'ij,jkh->ikh',
                (aemb_history_merge_weights, xa_weighted_emb)
            )
            # XXX: Same here!
            drnn_input = torch.cat([drnn_input, xa_history_weighted_emb], 2)

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

        return {'score_s': start_scores,
                'score_e': end_scores,
                'targets': ex['targets']}
