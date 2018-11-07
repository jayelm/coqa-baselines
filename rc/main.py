import argparse
import torch
import numpy as np

from model_handler import ModelHandler

################################################################################
# Main #
################################################################################


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(args):
    print_config(args)
    set_random_seed(args['random_seed'])
    model = ModelHandler(args)
    model.train()
    model.test()

################################################################################
# ArgParse and Helper Functions #
################################################################################


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainset', type=str, default=None, help='Training set')
    parser.add_argument('--devset', type=str, default=None, help='Dev set')
    parser.add_argument('--testset', type=str, default=None, help='Test set')
    parser.add_argument('--dialog_batched', type=str2bool, default=False, help='Batch sets by dialog, not example')
    parser.add_argument('--dir', type=str, default=None, help='Set the name of the model directory for this session.')
    parser.add_argument('--pretrained', type=str, default=None, help='Specify pretrained model directory.')

    parser.add_argument('--random_seed', type=int, default=123, help='Random seed')
    parser.add_argument('--cuda', type=str2bool, default=True, help='Run network on cuda (GPU) or not.')
    parser.add_argument('--cuda_id', type=int, default=-1, help='Specify a CUDA id.')
    parser.add_argument('--debug', type=str2bool, default=False)

    parser.add_argument('--n_history', type=int, default=-1)
    parser.add_argument('--cased', type=str2bool, default=True, help='Cased or uncased version.')
    parser.add_argument('--min_freq', type=int, default=100)
    parser.add_argument('--top_vocab', type=int, default=100000)

    group = parser.add_argument_group('model_spec')
    group.add_argument('--rnn_padding', type=str2bool, default=False, help='Whether to use RNN padding.')
    group.add_argument('--embed_file', type=str, default=None)
    group.add_argument('--embed_size', type=int, default=None)
    group.add_argument('--embed_type', type=str, default='glove', choices=['glove', 'word2vec', 'fasttext'])
    group.add_argument('--hidden_size', type=int, default=300, help='Set hidden size.')
    group.add_argument('--num_layers', type=int, default=3, help='Number of layers for document/question encoding.')
    group.add_argument('--rnn_type', type=str, choices=['lstm', 'gru', 'rnn'], default='lstm', help='RNN type.')
    group.add_argument('--concat_rnn_layers', type=str2bool, default=True, help='Whether to concat RNN layers.')
    group.add_argument('--question_merge', type=str, choices=['avg', 'self_attn'],
                       default='self_attn', help='The way of question encoding.')
    group.add_argument('--use_qemb', type=str2bool, default=True, help='Whether to add question aligned embedding.')
    group.add_argument('--f_qem', type=str2bool, default=True, help='Add exact match question feature to embedding.')
    group.add_argument('--f_pos', type=str2bool, default=False, help='Add POS feature to embedding.')
    group.add_argument('--f_ner', type=str2bool, default=False, help='Add NER feature to embedding.')
    group.add_argument('--sum_loss', type=str2bool, default=False, help="Set the type of loss.")
    group.add_argument('--doc_self_attn', type=str2bool, default=False,
                       help="Set whether to use self attention on the document.")
    group.add_argument('--resize_rnn_input', type=str2bool, default=False,
                       help='Reshape input layer to hidden size dimension.')
    group.add_argument('--span_dependency', type=str2bool, default=True,
                       help='Toggles dependency between the start and end span predictions for DrQA.')
    group.add_argument('--fix_embeddings', type=str2bool, default=False, help='Whether to fix embeddings.')
    group.add_argument('--dropout_rnn', type=float, default=0.4, help='Set RNN dropout in reader.')
    group.add_argument('--dropout_emb', type=float, default=0.5, help='Set embedding dropout.')
    group.add_argument('--dropout_ff', type=float, default=0.5, help='Set dropout for all feedforward layers.')
    group.add_argument('--dropout_rnn_output', type=str2bool, default=True, help='Whether to dropout last layer.')
    group.add_argument('--variational_dropout', type=str2bool, default=True, help='Set variational dropout on/off.')
    group.add_argument('--word_dropout', type=str2bool, default=False, help='Whether to dropout word.')

    # Dialog-specific options
    group = parser.add_argument_group('dialog_model_spec', 'Options specific to incorporating dialog history')
    group.add_argument('--f_history', type=str2bool, default=False,
                       help='Add exact match feature corresponding to history.')
    group.add_argument('--max_history', type=int, default=-1,
                       help='How many timesteps to limit history to (-1 = use everything)')
    group.add_argument('--q_dialog_history', type=str2bool, default=False, help='Whether to add historical averages of dialog (question AND answer tokens) to question. word_emb = match raw glove embedding; word_hidden = match BiLSTM hidden representations')
    group.add_argument('--q_dialog_attn', type=str, choices=['word_emb', 'word_hidden', 'word_hidden_incr', 'fully_incr'],
                       default='word_emb', help='How to compute attention over past dialog')
    group.add_argument('--q_dialog_attn_incr_merge', type=str, choices=['average', 'linear_current', 'linear_both', 'lstm'],
                       default='average', help='In the incremental case, how to average past and current representations')
    group.add_argument('--q_dialog_attn_scoring', type=str, choices=['linear_relu', 'fully_aware', 'bilinear'],
                       default='linear_relu', help='How to score interactions between hidden states (right now implented for word_hidden_incr only)')
    group.add_argument('--qa_emb_markers', type=str2bool, default=False,
                       help='Add qa markers to embeddings')
    group.add_argument('--answer_rnn', type=str2bool, default=False,
                       help='Use separately parameterized RNN to encode answers')
    group.add_argument('--mask_answers', type=str2bool, default=False,
                       help='Don\'t attend to answers')
    group.add_argument('--attend_answers', type=str2bool, default=False,
                       help='Augment answers with attention')
    group.add_argument('--attn_hidden_size', type=int, default=250,
                       help='Attention hidden size (for Q_DIALOG_ATTN only)')
    group.add_argument('--doc_dialog_history', type=str2bool, default=False, help='Whether to add historical averages of dialog (question AND answer tokens) to document.')
    group.add_argument('--doc_dialog_attn', type=str, choices=['q', 'word_emb', 'word_hidden'],
                       default='word_emb', help='How to compute attention over past dialog (q = same as q_dialog_attn)')
    group.add_argument('--history_dialog_answer_f', type=str2bool, default=False, help='Whether to add features distinguishing questions vs answers in historical dialog history.')
    group.add_argument('--history_dialog_time_f', type=str2bool, default=False, help='Whether to add features distinguishing timestep in historical dialog history (may overlap with recency_bias)')
    group.add_argument('--answer_merge', type=str, choices=['avg', 'self_attn'],
                       default='self_attn', help='How to merge (historical) answers')
    group.add_argument('--recency_bias', type=str2bool, default=False,
                       help='Bias recent questions in dialog history in qhidden/qemb attention')
    group.add_argument('--use_current_timestep', type=str2bool, default=True,
                       help='Include current question in historical question averages')

    # Optimizer
    group = parser.add_argument_group('training_spec')
    group.add_argument('--optimizer', type=str, default='adamax', help='Set optimizer.')
    group.add_argument('--learning_rate', type=float, default=0.1, help='Set learning rate for SGD.')
    group.add_argument('--grad_clipping', type=float, default=10.0, help='Whether to use grad clipping.')
    group.add_argument('--weight_decay', type=float, default=0.0, help='Set weight decay.')
    group.add_argument('--momentum', type=float, default=0.0, help='Set momentum.')
    group.add_argument('--batch_size', type=int, default=32, help='Set batch size.')
    group.add_argument('--max_epochs', type=int, default=50, help='Set number of total epochs.')
    group.add_argument('--verbose', type=int, default=400, help='Print every X batches.')
    group.add_argument('--shuffle', type=str2bool, default=True,
                       help='Whether to shuffle the examples during training.')
    group.add_argument('--max_answer_len', type=int, default=15, help='Set max answer length for decoding.')
    group.add_argument('--predict_train', type=str2bool, default=True, help='Whether to predict on training set.')
    group.add_argument('--out_predictions', type=str2bool, default=True, help='Whether to output predictions.')
    group.add_argument('--out_predictions_csv', type=str2bool, default=True, help='Whether to output predictions in csv format (Requires pandas).')
    group.add_argument('--predict_raw_text', type=str2bool, default=True,
                       help='Whether to use raw text and offsets for prediction.')
    group.add_argument('--save_params', type=str2bool, default=True, help='Whether to save params.')

    # Analysis
    group = parser.add_argument_group('analysis_spec', 'Options for analysis')
    group.add_argument('--save_attn_weights', nargs='*', choices=['q_dialog_attn', 'doc_dialog_attn'],
                       help='Which attention weights to save')

    args = vars(parser.parse_args())

    # Check that attention settings are compatible
    if args['doc_dialog_history'] and args['doc_dialog_attn'] == 'q' and not args['q_dialog_history']:
        parser.error("Can't use --doc_dialog_history with --doc_dialog_attn = 'q' if not using --q_dialog_history. "
                     "Specify --doc_dialog_attn separately or set --q_dialog_history")

    # Check analysis args
    if not args['pretrained']:
        for analysis_arg in ['save_attn_weights']:
            if args[analysis_arg]:
                parser.error("Must specify --pretrained model with analysis argument --{}".format(analysis_arg))

    # Check if analysis args analyze parts of the model that are actually used
    if args['save_attn_weights'] is not None:
        for attn_opt in args['save_attn_weights']:
            use_attn_opt = attn_opt.replace('_attn', '_history')
            if not (args[use_attn_opt]):
                parser.error("Must set --{} if wanting to analyze {}".format(use_attn_opt, attn_opt))
            if args[attn_opt] not in ['word_hidden', 'word_hidden_incr', 'fully_incr']:
                raise NotImplementedError("Analyzing {} attention".format(args[attn_opt]))

    if args['dialog_batched'] and args['batch_size'] > 2:
        print("WARNING: --dialog_batched and --batch_size = {}. Did you mean to set a large dialog batch size?".format(args['batch_Size']))

    return args


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")

################################################################################
# Module Command-line Behavior #
################################################################################


if __name__ == '__main__':
    args = get_args()
    main(args)
