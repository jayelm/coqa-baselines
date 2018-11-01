import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from scipy.linalg import toeplitz
import numpy as np


################################################################################
# Modules #
################################################################################

class StackedBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0,
                 dropout_output=False, variational_dropout=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False, bidirectional=True,
                 return_single_timestep=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.variational_dropout = variational_dropout
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.return_single_timestep = return_single_timestep
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else (2 * hidden_size if bidirectional else hidden_size)
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      batch_first=True,
                                      bidirectional=bidirectional))

    def forward(self, x, x_mask):
        """Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        """
        # Pad if we care or if its during eval.
        if self.padding or self.return_single_timestep or not self.training:
            return self._forward_padded(x, x_mask)
        # We don't care.
        return self._forward_unpadded(x, x_mask)

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            # Apply dropout to hidden input
            rnn_input = dropout(rnn_input, self.dropout_rate,
                                shared_axes=[1] if self.variational_dropout else [], training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)  # Concatenate hiddens at each timestep.
        else:
            output = outputs[-1]  # Take only hiddens after final layer (for all timesteps).

        # Dropout on output layer
        if self.dropout_output:
            output = dropout(output, self.dropout_rate,
                             shared_axes=[1] if self.variational_dropout else [], training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise,
        encoding that handles padding."""
        # Compute sorted sequence lengths
        lengths = x_mask.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        # Sort x
        rnn_input = x.index_select(0, idx_sort)

        # Encode all layers
        outputs, single_outputs = [rnn_input], []
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                rnn_input = dropout(rnn_input, self.dropout_rate,
                                    shared_axes=[1] if self.variational_dropout else [], training=self.training)
            # Pack it
            rnn_input = nn.utils.rnn.pack_padded_sequence(rnn_input, lengths, batch_first=True)
            # Run it
            rnn_output, (hn, _) = self.rnns[i](rnn_input)
            # Unpack it
            rnn_output = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)[0]
            single_outputs.append(hn[-1])
            outputs.append(rnn_output)

        if self.return_single_timestep:
            output = single_outputs[-1]
        # Concat hidden layers or take final
        elif self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Unsort
        output = output.index_select(0, idx_unsort)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = dropout(output, self.dropout_rate,
                             shared_axes=[1] if self.variational_dropout else [], training=self.training)
        return output


class WordHistoryAttn(nn.Module):
    """
    Perform self attention over a sequence at the word level. Output is a
    lower-triangular matrix

    That is, for each token embedding for the current question, use a bilinear
    term to compare similarity between this embedding and each of past question embeddings.
    """
    def __init__(self, token_hidden_size, question_hidden_size, recency_bias=False, cuda=False):
        super(WordHistoryAttn, self).__init__()
        self.linear = nn.Linear(token_hidden_size, question_hidden_size)
        self.recency_bias = recency_bias
        self.cuda = cuda

        if recency_bias:
            self.recency_weight = nn.Parameter(torch.full((1, ), -0.5))

    def forward(self, question_hiddens, question_hidden, q_mask):
        """
        Input shapes:
            question_hiddens = batch * max_q_len * token_hidden_size (encodings for all q tokens in dialog history)
            question_hidden = batch * question_hidden_size (encodings for all q tokens in dialog history)
            q_mask = batch * max_q_len (mask for questions)
        Output shapes:
            weights = batch * max_q_len * batch (?)
            each submatrix of the batch is a historical attention map for the
            current timestep (lower triangular) which gives weights across all
            question vectors for each token. Note we need to mask attention
            weights for each token as well.

        This will require at least one bmm.
        """
        self.question_hiddens
        pass


class QAHistoryAttn(nn.Module):
    """
    Perform attention by comparing historical question-answer pairs to a
    current question with a bilinear term.
    """
    def __init__(self, qa_hidden_size, question_hidden_size, hidden_size=None, recency_bias=False,
                 cuda=False):
        super(QAHistoryAttn, self).__init__()
        if hidden_size is None:
            hidden_size = question_hidden_size
        self.linear_question = nn.Linear(question_hidden_size, hidden_size)
        self.linear_qa = nn.Linear(qa_hidden_size, hidden_size)
        self.cuda = cuda
        self.recency_bias = recency_bias
        if recency_bias:
            self.recency_weight = nn.Parameter(torch.full((1, ), -0.5))

    def forward(self, qa_hidden, question_hidden):
        """
        qa_hidden = batch * qa_hidden_size
        question_hidden = batch * question_hidden_size

        output:
        attention_weights = batch * batch (attention weights for each q in dialog history)

        WARNING: first row will be NaNs due to softmax - must deal with this
        outside of this computation!
        """
        question_proj = self.linear_question(question_hidden)
        question_proj = F.relu(question_proj)  # (batch, hidden_size)

        qa_proj = self.linear_qa(qa_hidden)
        qa_proj = F.relu(qa_proj)  # (batch, hidden_size)

        scores = question_proj.mm(qa_proj.transpose(1, 0))

        # Mask
        scores_mask = make_scores_mask(scores.size(),
                                       use_current_timestep=False,  # Can't use current ans!
                                       cuda=self.cuda)
        scores.masked_fill_(scores_mask, -float('inf'))

        # Recency bias
        if self.recency_bias:  # Add recency weights
            recency_weights = make_recency_weights(scores_mask, self.recency_weight, cuda=self.cuda)
            recency_weights = zero_first(recency_weights)
            scores = scores + recency_weights

        scores = F.softmax(scores, dim=1)

        # Zero out first weights since softmax returns NaNs
        scores = zero_first(scores)

        return scores


class QAHistoryAttnBilinear(nn.Module):
    """
    Perform attention by comparing historical question-answer pairs to a
    current question with a bilinear term.
    """
    def __init__(self, qa_hidden_size, question_hidden_size, recency_bias=False,
                 cuda=False):
        super(QAHistoryAttnBilinear, self).__init__()
        self.linear = nn.Linear(qa_hidden_size, question_hidden_size)
        self.cuda = cuda
        self.recency_bias = recency_bias
        if recency_bias:
            self.recency_weight = nn.Parameter(torch.full((1, ), -0.5))

    def forward(self, qa_hidden, question_hidden):
        """
        qa_hidden = batch * qa_hidden_size
        question_hidden = batch * question_hidden_size

        output:
        attention_weights = batch * batch (attention weights for each q in dialog history)
        """
        qa_proj = self.linear(qa_hidden)  # (batch, question_size)

        scores = question_hidden.mm(qa_proj.transpose(1, 0))

        # Mask
        scores_mask = make_scores_mask(scores.size(),
                                       use_current_timestep=False,  # Can't use current ans!
                                       cuda=self.cuda)
        scores.masked_fill_(scores_mask, -float('inf'))

        # Recency bias
        if self.recency_bias:  # Add recency weights
            recency_weights = make_recency_weights(scores_mask, self.recency_weight, cuda=self.cuda)
            recency_weights = zero_first(recency_weights)
            scores = scores + recency_weights

        scores = F.softmax(scores, dim=1)

        # Zero out first weights
        scores = zero_first(scores)

        return scores


class SentenceHistoryAttn(nn.Module):
    """
    Perform self attention over a sequence - match each sequence to itself.
    Crucially, output is a lower-triangular matrix. So information only flows
    one way.
    """
    def __init__(self, input_size, recency_bias=False, cuda=False,
                 use_current_timestep=True):
        super(SentenceHistoryAttn, self).__init__()
        self.linear = nn.Linear(input_size, input_size)
        self.recency_bias = recency_bias
        self.cuda = cuda
        self.use_current_timestep = use_current_timestep

        if recency_bias:
            self.recency_weight = nn.Parameter(torch.full((1, ), -0.5))

    def forward(self, x):
        """
        Input shapes:
            x = batch * input_size (encodings for all qs in dialog history)
        Output shapes:
            attn = batch * batch (lower triangular matrix; attention
            values for each q in dialog history)
        """
        # Project x through linear layer
        x_proj = self.linear(x)
        x_proj = F.relu(x_proj)

        scores = x_proj.mm(x_proj.transpose(1, 0))

        # Score mask
        scores_mask = make_scores_mask(scores.size(),
                                       use_current_timestep=self.use_current_timestep,
                                       cuda=self.cuda)
        # Mask scores and weights with upper triangular mask
        scores.masked_fill_(scores_mask, -float('inf'))

        if self.recency_bias:  # Add recency weights
            recency_weights = make_recency_weights(scores_mask, self.recency_weight, cuda=self.cuda)
            if not self.use_current_timestep:
                recency_weights = zero_first(recency_weights)
            scores = scores + recency_weights

        scores = F.softmax(scores, dim=1)

        if not self.use_current_timestep:
            scores = zero_first(scores)

        return scores


def make_scores_mask(scores_shape, use_current_timestep=True, cuda=False):
    """
    Make upper triangular mask of 1s and 0s. If use_current_timestep is False,
    diagonal is also 1 (i.e. masked).
    """
    scores_mask = torch.ones(scores_shape, dtype=torch.uint8,
                             requires_grad=False)
    if cuda:
        scores_mask = scores_mask.cuda()

    scores_mask = torch.triu(scores_mask,
                             diagonal=1 if use_current_timestep else 0)
    return scores_mask


def make_dialog_scores_mask(scores_shape, max_qa_len, use_current_timestep=True,
                            cuda=False):
    reg_mask = np.triu(np.ones(scores_shape, dtype=np.uint8),
                       k=1 if use_current_timestep else 0)
    repeated_mask = np.repeat(reg_mask, max_qa_len).reshape((scores_shape[0], -1))
    repeated_mask = torch.tensor(repeated_mask, requires_grad=False)
    if cuda:
        repeated_mask = repeated_mask.cuda()
    return repeated_mask


def make_recency_weights(scores_mask, recency_weight, cuda=False):
    """
    Create a recency weights mask from the given scores mask and recency weight.
    Upper triangular with specific diagonal dependent on scores mask.
    """
    # Create recency mask; a toeplitz matrix with higher values the
    # further away you are from the diagonal
    # Since recency_weight is negative, this downweights questions that are further away
    recency_weights_np = toeplitz(np.arange(scores_mask.shape[0], dtype=np.float32))
    recency_weights = torch.tensor(recency_weights_np, requires_grad=False)

    if cuda:
        recency_weights = recency_weights.cuda()

    recency_weights.masked_fill_(scores_mask, 0.0)
    recency_weights = recency_weight * recency_weights

    return recency_weights


def make_dialog_recency_weights(scores_mask, max_qa_len, recency_weight, cuda=False):
    """
    Create a recency weights mask from the given scores mask and recency weight.
    Upper triangular with specific diagonal dependent on scores mask.
    """
    # Create recency mask; a toeplitz matrix with higher values the
    # further away you are from the diagonal
    # Since recency_weight is negative, this downweights questions that are further away
    recency_weights_np = toeplitz(np.arange(scores_mask.shape[0], dtype=np.float32))
    recency_weights_np = np.repeat(recency_weights_np, max_qa_len).reshape(scores_mask.shape[0], -1)
    recency_weights = torch.tensor(recency_weights_np, requires_grad=False)

    if cuda:
        recency_weights = recency_weights.cuda()

    recency_weights = recency_weight * recency_weights

    return recency_weights


class DialogSeqAttnMatch(nn.Module):
    """
    Like SeqAttnMatch, but operates on dialog history. Prevents time travel and
    optionally enables recency bias.
    """
    def __init__(self, input_size, identity=False,
                 cuda=False, recency_bias=False, answer_marker_features=False,
                 time_features=False):
        super(DialogSeqAttnMatch, self).__init__()

        self.cuda = cuda
        self.recency_bias = recency_bias
        if recency_bias:
            self.recency_weight = nn.Parameter(torch.full((1, ), -0.5))

        self.answer_marker_features = answer_marker_features
        self.time_features = time_features

        true_input_size = input_size
        if self.answer_marker_features:
            true_input_size += 3
        if self.time_features:
            raise NotImplementedError
        if not identity:
            self.linear = nn.Linear(true_input_size, input_size)
        else:
            self.linear = None

    def forward(self, xd_emb, xq_emb, xa_emb, xq_mask, xa_mask,
                out_attention=False):
        """Input shapes:
            xd_emb = batch * len1 * h  (document)
            xdialog_emb = batch * (max_qa_len = max_q_len + max_a_len) * h  (dialog)
            xdialog_mask = batch * (max_qa_len = max_q_len + max_a_len)  (dialog mask)
        Output shapes:
            matched_seq = batch * len1 * h

        SPECIFICALLY, xdialog_emb is just the result of concatting the question
        and answer embeddings together (along dimension 1). Same goes for
        xdialog_mask (no problem re: padding in the middle due to
        max_q_len/max_a_len)

        This function does reshaping to compute history over this entire dialog
        for each document in xd_emb.
        differentiate between answers and
        """
        if self.answer_marker_features:
            # Add 1s to mark answers, else 0
            a_markers = onehot_markers(xa_emb, 3, 0, cuda=self.cuda)
            q_markers = onehot_markers(xq_emb, 3, 1, cuda=self.cuda)
            d_markers = onehot_markers(xd_emb, 3, 2, cuda=self.cuda)

            xa_emb_m = torch.cat((xa_emb, a_markers), 2)
            xq_emb_m = torch.cat((xq_emb, q_markers), 2)
            xd_emb_m = torch.cat((xd_emb, d_markers), 2)
        else:
            xa_emb_m = xa_emb
            xq_emb_m = xq_emb
            xd_emb_m = xd_emb

        xdialog_emb_m = torch.cat((xq_emb_m, xa_emb_m), 1)
        xdialog_mask = torch.cat((xq_mask, xa_mask), 1)

        max_dialog_len = xdialog_emb_m.shape[1]
        # Reshape by unraveling dialog history and repeating it across the
        # batch
        xdialog_emb_m_flat = xdialog_emb_m.view(-1, xdialog_emb_m.shape[2])
        xdialog_emb_m_tiled = xdialog_emb_m_flat.expand(xdialog_emb_m.shape[0], -1, -1).contiguous()
        if self.answer_marker_features:
            # Create original version of xdialog, since we need the output to
            # be of original size
            xdialog_emb_o = torch.cat((xq_emb, xa_emb), 1)
            xdialog_emb_o_flat = xdialog_emb_o.view(-1, xdialog_emb_o.shape[2])
            xdialog_emb_o_tiled = xdialog_emb_o_flat.expand(xdialog_emb_o.shape[0], -1, -1).contiguous()
        else:
            xdialog_emb_o_tiled = xdialog_emb_m_tiled

        xdialog_mask_flat = xdialog_mask.view(-1)
        # Don't expand here; we will modify rows separately!
        xdialog_mask_tiled = xdialog_mask_flat.unsqueeze(0).repeat(xdialog_mask.shape[0], 1)

        # This is an upper triangular matrix but each element is repeated
        # max_dialog_len times, thus masking entire sequences of dialog
        # corresponding to future and (optionally) current timesteps.
        dialog_scores_mask = make_dialog_scores_mask(
            (xdialog_emb_m.shape[0], xdialog_emb_m.shape[0]),
            max_dialog_len, use_current_timestep=False,
            cuda=self.cuda)

        assert xdialog_mask_tiled.shape == dialog_scores_mask.shape
        assert xdialog_emb_m_tiled.shape[:2] == dialog_scores_mask.shape
        xdialog_mask_tiled.masked_fill_(dialog_scores_mask, 1)

        return self.seqattnmatch_forward(xd_emb_m, xdialog_emb_m_tiled, xdialog_emb_o_tiled, xdialog_mask_tiled, max_dialog_len, out_attention=out_attention)

    def seqattnmatch_forward(self, x, y, y_orig, y_mask, max_dialog_len, out_attention=False):
        """
        This is directly taken from seqattnmatch
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view((x.shape[:2] + (-1, )))
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view((y.shape[:2] + (-1, )))
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))  # (batch, len1, len2)

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())  # (batch, len1, len2)
        scores.masked_fill_(y_mask, -float('inf'))

        if self.recency_bias:
            recency_weights = make_dialog_recency_weights(y_mask, max_dialog_len,
                                                          self.recency_weight,
                                                          cuda=self.cuda)
            recency_weights = zero_first(recency_weights)
            # Expand weights along each token of the document (i.e. dimension 1)
            recency_weights = recency_weights.unsqueeze(1).expand(-1, scores.shape[1], -1)
            scores_pre = scores
            scores = scores + recency_weights

        # Normalize with softmax
        alpha = F.softmax(scores, dim=-1)

        # Since we do not use current timestep, first row of alpha will be NaN
        alpha = zero_first(alpha)

        # Take weighted average
        matched_seq = alpha.bmm(y_orig)
        if out_attention:
            return matched_seq, alpha
        return matched_seq                      # (batch, len2, h)


class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """
    def __init__(self, input_size, identity=False, recency_bias=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None
        self.recency_bias = recency_bias
        if self.recency_bias:
            self.recency_weight = nn.Parameter(torch.full((1, ), -0.5))

    def forward(self, x, y, y_mask, recency_weights=None,
                out_attention=False):
        """Input shapes:
            x = batch * len1 * h  (document)
            y = batch * len2 * h  (question)
            y_mask = batch * len2 (question mask)
        Output shapes:
            matched_seq = batch * len1 * h
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))  # (batch, len1, len2)

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())  # (batch, len1, len2)
        scores.masked_fill_(y_mask, -float('inf'))

        if recency_weights is not None:
            if not self.recency_bias:
                raise RuntimeError("Recency weights specified but recency_bias is false")
            recency_weights = recency_weights.unsqueeze(1).expand(scores.size())
            scores = scores + (recency_weights * self.recency_weight)

        # Normalize with softmax
        alpha = F.softmax(scores, dim=-1)

        # Take weighted average
        matched_seq = alpha.bmm(y)
        if out_attention:
            return matched_seq, alpha
        return matched_seq                      # (batch, len2, h)


class IncrSeqAttnMatch(nn.Module):
    """
    This is an incremental version of seqattnmatch. Does word-level comparison,
    augmenting question vectors at each step.
    """
    def __init__(self, input_size, merge_type='average', recency_bias=False,
                 cuda=False, max_history=-1, scoring='linear_relu',
                 attend_answers=False, hidden_size=250):
        super(IncrSeqAttnMatch, self).__init__()
        self.cuda = cuda

        self.scoring = scoring
        self.hidden_size = hidden_size

        if self.scoring == 'linear_relu':
            self.linear = nn.Linear(input_size, hidden_size)
        elif self.scoring == 'fully_aware':
            # https://arxiv.org/pdf/1711.07341.pdf
            self.linear = nn.Linear(input_size, hidden_size)
            self.diag = nn.Parameter(torch.diag(torch.rand(hidden_size, requires_grad=True)))
        elif self.scoring == 'bilinear':
            self.linear = nn.Linear(input_size, hidden_size)
        else:
            raise NotImplementedError("attn_type = {}".format(self.scoring))

        self.attend_answers = attend_answers

        self.recency_bias = recency_bias
        if self.recency_bias:
            self.recency_weight = nn.Parameter(torch.full((1, ), -0.5))

        self.max_history = max_history

        self.merge_type = merge_type
        if self.merge_type == 'average':
            pass
        elif self.merge_type == 'linear_current':
            self.merge_layer = nn.Linear(input_size, 1)
        elif self.merge_type == 'linear_both':
            self.merge_layer = nn.Linear(2 * input_size, 1)
        else:
            raise NotImplementedError("merge_type = {}".format(merge_type))

    def forward(self, xq_emb, xa_emb, xq_mask, xa_mask,
                out_attention=False):
        """Input shapes:
            # FIXME: This may be more generalizable if you re-include xd_emb as
            # a first argument (and for dialog history matching, just pass xq
            # twice)
            xq_emb = batch * max_q_len * h  (document)
            xa_emb = batch * max_a_len * h  (document)
            xq_mask = batch * max_q_len * h  (document)
            xa_mask = batch * max_a_len * h  (document)
        Output shapes:
            matched_seq = batch * max_d_len * h
        """
        # Project vectors
        xq_proj = self.project(xq_emb)
        xa_proj = self.project(xa_emb)

        # Form dialog
        d_plus = [xq_emb[0], xa_emb[0]]
        d_proj = [xq_proj[0], xa_proj[0]]
        d_mask = [xq_mask[0], xa_mask[0]]
        max_q_len, max_a_len = xq_proj.shape[1], xa_proj.shape[1]

        if out_attention:
            out_scores = []
        # Loop through qa pairs
        # int, (max_q_len * k), (max_a_len * k), (max_q_len * h), (max_q_len * h)
        for t, (xq_t_proj, xa_t_proj, xq_t, xa_t) in enumerate(zip(xq_proj[1:], xa_proj[1:],
                                                                   xq_emb[1:], xa_emb[1:]), start=1):
            # TODO: define self.augment which does all this, to reduce code reuse (for the answer)
            # Form dialog history up to this point.
            d_plus_t = torch.cat(d_plus, 0)  # (history_len * h)
            d_proj_t = torch.cat(d_proj, 0)  # (history_len * k)
            d_mask_t = torch.cat(d_mask, 0)  # (history_len * h)
            # Compute attention with non-ctx-sensitive embeddigs
            scores = self.score(xq_t_proj, d_proj_t)  # (max_q_len, history_len)

            if self.recency_bias:
                recency_weights = self.recency_weights(t, max_q_len, max_a_len).expand(scores.size())
                scores = scores + recency_weights

            if self.max_history > 0:
                history_mask = self.max_history_mask(t, max_q_len, max_a_len).expand(scores.size())
                scores.masked_fill_(history_mask, -float('inf'))

            # Mask nonexistent qa tokens
            d_mask_t = d_mask_t.expand(scores.size())
            scores.masked_fill_(d_mask_t, -float('inf'))

            # Normalize
            alpha = F.softmax(scores, dim=1)  # (max_q_len, history_len)

            # Compute historical average embeddings
            xq_t_history = alpha.mm(d_plus_t)  # (max_q_len, h)

            # Merge current repr with history
            xq_t_plus, keep_p = self.merge(xq_t, xq_t_history)

            # Append augmented q to history
            d_plus.append(xq_t_plus)
            d_proj.append(xq_t_proj)
            d_mask.append(xq_mask[t])

            if out_attention:
                alpha_masked = alpha[:, (1 - d_mask_t[0]).nonzero().squeeze()]
                alpha_masked = torch.cat((keep_p, alpha_masked), 1)
                out_scores.append(alpha_masked)

            if self.attend_answers:
                # Reconcat with new augmented question
                d_plus_t = torch.cat(d_plus, 0)
                d_proj_t = torch.cat(d_proj, 0)
                d_mask_t = torch.cat(d_mask, 0)

                # Score answer
                scores = self.score(xa_t_proj, d_proj_t)

                if self.recency_bias:
                    recency_weights = self.recency_weights(t, max_q_len, max_a_len,
                                                           answer=True).expand(scores.size())
                    scores = scores + recency_weights

                if self.max_history > 0:
                    history_mask = self.max_history_mask(t, max_q_len, max_a_len,
                                                         answer=True).expand(scores.size())
                    scores.masked_fill_(history_mask, -float('inf'))

                d_mask_t = d_mask_t.expand(scores.size())
                scores.masked_fill_(d_mask_t, -float('inf'))

                alpha = F.softmax(scores, dim=1)

                xa_t_history = alpha.mm(d_plus_t)

                xa_t_plus, keep_p = self.merge(xa_t, xa_t_history)
            else:
                xa_t_plus = xa_t

            d_plus.append(xa_t_plus)
            d_proj.append(xa_t_proj)
            d_mask.append(xa_mask[t])

        # Concat and return augmented qa reprs (every 2nd repr)
        xq_plus = torch.stack(d_plus[::2])
        if out_attention:
            out_scores = self.clean_out_scores(out_scores, max_q_len, max_a_len)
            return xq_plus, out_scores
        return xq_plus

    def clean_out_scores(self, out_scores, max_q_len, max_a_len):
        if not out_scores:
            # Dummy zeros for qa len of first timestep
            out_scores = torch.zeros((1, max_q_len, max_q_len + max_a_len), dtype=np.float32)
            if self.cuda:
                out_scores = out_scores.cuda()
            return out_scores
        out_scores = [s.transpose(1, 0) for s in out_scores]
        out_scores = pad_sequence(out_scores, batch_first=True)
        out_scores = out_scores.permute(0, 2, 1)
        out_scores = torch.cat((torch.zeros_like(out_scores[0:1]), out_scores), 0)
        return out_scores

    def project(self, x):
        """
        Project vectors using the mechanism described by self.scoring.
        """
        # All attention mechanisms require linear projection.
        if len(x.shape) == 3:
            # Reshape first.
            x_proj = self.linear(x.view(-1, x.size(2))).view((x.shape[:2] + (-1, )))
        elif len(x.shape) == 2:
            x_proj = self.linear(x)
        else:
            raise ValueError("Incompatible shape for projection {}".format(x.shape))

        if self.scoring == 'linear_relu':
            x_proj = F.relu(x_proj)
        elif self.scoring == 'bilinear':
            # Don't do anything more, we just compute raw dot product.
            pass
        elif self.scoring == 'fully_aware':
            x_proj = F.relu(x_proj)
        else:
            raise NotImplementedError
        return x_proj

    def score(self, x, y):
        """
        Score vectors according to self.scoring
        """
        if self.scoring == 'linear_relu':
            return x.mm(y.transpose(1, 0))
        elif self.scoring == 'bilinear':
            return x.mm(y.transpose(1, 0))
        elif self.scoring == 'fully_aware':
            # Multiply x by diagonal matrix first.
            x_diag = x.mm(self.diag)
            return x_diag.mm(y.transpose(1, 0))
        else:
            raise NotImplementedError

    def recency_weights(self, t, max_q_len, max_a_len, answer=False):
        """
        Return recency weights matrix for time t.
        """
        max_qa_len = max_q_len + max_a_len
        recency_weights_np = np.repeat(np.arange(t, 0, -1, dtype=np.float32), max_qa_len)
        if answer:
            # Add zeros corresponding to current question.
            recency_weights_np = np.concatenate((recency_weights_np, np.zeros(max_q_len, dtype=np.float32)))
        recency_weights = torch.tensor(recency_weights_np, requires_grad=False)
        if self.cuda:
            recency_weights = recency_weights.cuda()
        recency_weights = recency_weights * self.recency_weight
        return recency_weights

    def max_history_mask(self, t, max_q_len, max_a_len, answer=False):
        """
        Return maximum history mask for time t.
        """
        max_qa_len = max_q_len + max_a_len
        history_mask_np = np.repeat(np.arange(t, 0, -1, dtype=np.float32), max_qa_len)
        if answer:
            # Add zeros corresponding to current question.
            history_mask_np = np.concatenate((history_mask_np, np.zeros(max_q_len, dtype=np.float32)))
        history_mask_np = (history_mask_np > self.max_history).astype(np.uint8)
        history_mask = torch.tensor(history_mask_np, requires_grad=False)
        if self.cuda:
            history_mask = history_mask.cuda()
        return history_mask

    def merge(self, xq_t, xq_t_history):
        if self.merge_type == 'average':
            keep_p = torch.full((xq_t.shape[0], 1), 0.5, dtype=torch.float32,
                                requires_grad=False)
            if self.cuda:
                keep_p = keep_p.cuda()
        elif self.merge_type == 'linear_current':
            # Look at current word only, learn a scalar keep value.
            # Intuition is that it'll learn, e.g., that pronouns are more
            # important to resolve.
            keep_p = self.merge_layer(xq_t)
            keep_p = torch.sigmoid(keep_p)
        elif self.merge_type == 'linear_both':
            # Look at current word and past attention, just concatted.
            keep_p = self.merge_layer(torch.cat((xq_t, xq_t_history), 1))
            keep_p = torch.sigmoid(keep_p)
        else:
            raise NotImplementedError("merge_type = {}".format(self.merge_type))
        xq_t_plus = (keep_p * xq_t) + ((1.0 - keep_p) * xq_t_history)
        return xq_t_plus, keep_p


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.
    """
    def __init__(self, x_size, y_size, identity=False):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1  (doc_hiddens)
        y = batch * h2        (question_hidden)
        x_mask = batch * len  (xd_mask)
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.masked_fill_(x_mask, -float('inf'))
        alpha = F.log_softmax(xWy, dim=-1)
        return alpha


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.masked_fill_(x_mask, -float('inf'))
        alpha = F.softmax(scores, dim=-1)
        return alpha

################################################################################
# Functional #
################################################################################


def dropout(x, drop_prob, shared_axes=[], training=False):
    if drop_prob == 0 or (not training):
        return x

    sz = list(x.size())
    for i in shared_axes:
        sz[i] = 1
    mask = x.new(*sz).bernoulli_(1. - drop_prob).div_(1. - drop_prob)
    mask = mask.expand_as(x)
    return x * mask


def multi_nll_loss(scores, target_mask):
    """
    Select actions with sampling at train-time, argmax at test-time:
    """
    scores = scores.exp()
    loss = 0
    for i in range(scores.size(0)):
        loss += torch.neg(torch.log(torch.masked_select(scores[i], target_mask[i]).sum() / scores[i].sum()))
    return loss


def uniform_weights(x, x_mask):
    """Return uniform weights over non-masked input."""
    raise NotImplementedError


def weighted_avg(x, weights):
    """x = batch * len * d
    weights = batch * len
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)


def zero_first(arr):
    arr_rest = arr[1:]
    zeros = torch.zeros_like(arr[0], requires_grad=False).unsqueeze(0)
    arr_new = torch.cat([zeros, arr_rest], 0)
    return arr_new


def onehot_markers(emb, n_total, n_on, cuda=False):
    """
    Make one-hot marker features of the same shape as emb, but with n_total features

    Input:
        emb: batch x len x h
    Output:
        markers: batch x len x n_total

    Where only the "n_on"th element of n_total is one, else zero
    """
    if n_on > (n_total - 1):
        raise IndexErrror("One-hot index {} out of bounds given {} options".format(
            n_on, n_total
        ))

    ones_markers = torch.ones(emb.shape[:2], dtype=torch.float32,
                              requires_grad=False).unsqueeze(2)
    zeros_markers = torch.zeros(emb.shape[:2], dtype=torch.float32,
                                requires_grad=False).unsqueeze(2)
    if cuda:
        ones_markers = ones_markers.cuda()
        zeros_markers = zeros_markers.cuda()
    markers_list = []
    for i in range(n_total):
        if i == n_on:
            markers_list.append(ones_markers)
        else:
            markers_list.append(zeros_markers)

    return torch.cat(markers_list, 2)
